from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForMaskedLM
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.nn.functional as F
from util import MAX_TURN, PREVENT_FACTOR, PROMOTE_FACTOR, PREVENT_LIST, REDUCE_LIST, STOP_LIST, boolean_string, \
    top_k_top_p_filtering, bits2int, int2bits, num_same_from_beg, kl

NOI_ID = 1

def arithmetic_embed(bitstrs, model, input_ids, segment_ids, input_mask, no_ins = None, device='cuda', temperature=1.0, args=None, tokenizer=None, sample_num=1, top_k=500, top_p=0.95, prevent=None, promote=None, reduce=None, verbose = None):
    sec = ""
    precision = 16

    if not verbose:
        verbose = args.verbose
    zero_list = ["[", "]", "(", ")"]
    zero_ids = [tokenizer.vocab.get(x) for x in zero_list]
    no_ins_cur = no_ins[0][:(no_ins[0] == -1).nonzero()[0]]

    total_num = 0
    total_kl = 0

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    with torch.no_grad():
        # print("Turn:", ip)
        for ip in range(MAX_TURN):
            result = model(input_ids, segment_ids, input_mask)
            mask_prediction_scores = result[0]
            input_len = torch.sum(input_mask, 1)

            base_log_prob = F.log_softmax(mask_prediction_scores,dim=-1)
            base_log_prob, _ = torch.topk(base_log_prob, dim=2, k=top_k)

            noi_temp = min(float(ip) / args.noi_decay, 1.0)
            mask_prediction_scores[:, :, 1] = mask_prediction_scores[:, :, 1] * noi_temp
            logits = mask_prediction_scores / temperature

            if prevent:
                for p in prevent:
                    logits[:, :, p] = logits[:, :, p] * PREVENT_FACTOR
            if reduce:
                reduce_factor = min(float(ip) / args.reduce_decay, 1.0)
                for p in reduce:
                    logits[:, :, p] = logits[:, :, p] * reduce_factor
            if promote:
                for p in promote:
                    logits[:, :, p] = logits[:, :, p] * PROMOTE_FACTOR
            if args.lessrepeat:
                for p in input_ids.cpu().numpy()[0]:
                    logits[:, :, p] = logits[:, :, p] * 0.8

            logits[:, :, zero_ids] = -1e10
            for i in range(args.max_seq_length):
                logits[:, i] = top_k_top_p_filtering(logits[:, i].squeeze(), top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)

            len_input = len([x for x in input_ids[0].detach().cpu().numpy() if x != 0])
            args.num_token += len_input

            _, top_id = torch.topk(probs, dim=2, k=1)
            if torch.sum(top_id[0, :len_input, 0] == 1) == len_input:
                avg_kl = total_kl / total_num if total_num != 0 else 0
                return input_ids, avg_kl, sec

            # k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), top_k)

            top_probs, top_ids = torch.topk(probs, dim=2, k=1000)

            input_ids_new = torch.zeros_like(input_ids)
            top_predicts = torch.zeros([input_ids.shape[0], input_ids.shape[1], 3], dtype=torch.long)
            mask_predicts = torch.zeros_like(input_ids, dtype=torch.long)

            for t in range(args.max_seq_length):
                first = top_ids[:, t, 0]
                if ip < 3:

                    # print("probs_temp_int:", probs_temp_int, probs_temp_int.shape)
                    if len_input <= t or t == len_input - 1 and 102 in input_ids[0] or 1 == first or 102==first or top_probs[:, t, 0] == 1:
                        mask_predicts[:, t] = first
                    else:

                        probs_temp = top_probs[0, t, :]
                        ids_temp = top_ids[0, t, :]


                        # arithmetic encode
                        # Cutoff low probabilities that would be rounded to 0

                        cur_int_range = cur_interval[1] - cur_interval[0]
                        cur_threshold = 1 / cur_int_range

                        # print(probs_temp)
                        kk = min(max(2,(probs_temp < cur_threshold).nonzero()[0].item()),top_k)
                        # print("top k",kk)

                        probs_temp_int = probs_temp[ : kk]
                        ids_temp = ids_temp[ : kk]

                        #remove [NOI] [PAD] [SEP] [CLS]
                        probs_list = probs_temp_int.tolist()
                        ids_list = ids_temp.tolist()

                        del_idx = []
                        for i in range(len(ids_list)):
                            if ids_list[i] in [0,1,102,101]:
                                del_idx.append(i)
                        if (len(del_idx)) != 0:
                            for i in range(len(del_idx)):
                                idx = del_idx[i] - i
                                probs_list = probs_list[:idx] + probs_list[idx + 1:]
                                ids_list = ids_list[:idx] + ids_list[idx + 1:]

                        probs_temp_int = torch.tensor(probs_list, device=device)
                        ids_temp = torch.tensor(ids_list, device=device)

                        # Rescale to correct range
                        probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                        # Round probabilities to integers given precision
                        probs_temp_int = probs_temp_int.round().long()
                        cum_probs = probs_temp_int.cumsum(0)

                        # Remove any elements from the bottom if rounding caused the total prob to be too large
                        overfill_index = (cum_probs > cur_int_range).nonzero()
                        if len(overfill_index) > 0:
                            cum_probs = cum_probs[:overfill_index[0]]

                        # Add any mass to the top if removing/rounding causes the total prob to be too small
                        cum_probs += cur_int_range - cum_probs[-1]  # add

                        # Get out resulting probabilities
                        probs_final = cum_probs.clone()
                        probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                        # Convert to position in range
                        cum_probs += cur_interval[0]

                        # Get selected index based on binary fraction from message bits
                        message_bits = bitstrs[args.global_pos : args.global_pos + precision]

                        message_idx = bits2int(message_bits)
                        selection = (cum_probs > message_idx).nonzero()[0].item()
                        # print("selection:",selection)

                        # Calculate new range as ints
                        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                        new_int_top = cum_probs[selection]

                        # Convert range to bits
                        new_int_bottom_bits_inc = int2bits(new_int_bottom, precision)
                        new_int_top_bits_inc = int2bits(new_int_top - 1, precision)  # -1 here because upper bound is exclusive

                        # Consume most significant bits which are now fixed and update interval
                        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

                        sec += bitstrs[args.global_pos : args.global_pos+num_bits_encoded]
                        # print("sec info",bitstrs[args.global_pos : args.global_pos+num_bits_encoded])
                        args.global_pos += num_bits_encoded


                        new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + '0' * num_bits_encoded
                        new_int_top_bits =  new_int_top_bits_inc[num_bits_encoded:] + '1' * num_bits_encoded

                        cur_interval[0] = bits2int(new_int_bottom_bits)
                        cur_interval[1] = bits2int(new_int_top_bits) + 1  # +1 here because upper bound is exclusive

                        # index = random.randint(1, 10) % 2
                        # print("index:", index)
                        # sec += str(index)
                        q = probs_final.float() / probs_final.sum()
                        # print("q",q)
                        # print("logp",torch.exp(base_log_prob[0,t,:len(q)]))
                        logq = q.log()
                        total_kl += kl(q, logq, base_log_prob[0,t,:len(q)])
                        total_num += 1

                        mask_predicts[:, t] = ids_temp[selection]
                else:
                    if first != 102 and first != 0 and first != 101:
                        mask_predicts[:, t] = first
                    else:
                        mask_predicts[:, t] = torch.tensor(1).to(device)

                    # mask_predicts[:, t] = first

                # mask_predicts[:, t] = torch.multinomial(probs[:, t, :], num_samples=1)
                top_predicts[:, t] = torch.topk(probs[:, t, :], k=3)[1]

            logit_new = torch.zeros_like(input_ids, dtype=torch.float)
            input_ids_ori = input_ids
            top_predicts_new = torch.zeros_like(top_predicts)
            i = 0
            j = 0
            k = 0
            sep_tok = tokenizer.vocab['[SEP]']
            # update no_ins
            mask_predicts[0][no_ins_cur] = NOI_ID  #
            new_no_ins_cur = no_ins_cur.clone().detach()
            while np.max([i, j, k]) < args.max_seq_length - 1:
                # print(i,j,k)
                input_ids_new[0, k] = input_ids[0, i]
                if input_ids[0, i] == 0:  # padding, ignore prediction
                    break
                if input_ids[0, i] == sep_tok:
                    break

                i += 1
                k += 1

                if mask_predicts[0, j].cpu().numpy() != 1:
                    input_ids_new[0, k] = mask_predicts[0, j]
                    logit_new[0, k] = probs[0, j, mask_predicts[0, j]]
                    top_predicts_new[0, k, :] = top_predicts[0, j, :]
                    if len(no_ins_cur) > 0 and no_ins_cur[-1] > j:
                        new_no_ins_cur[torch.where(no_ins_cur > j)[0][0]:] += 1
                    k += 1
                    j += 1
                else:
                    j += 1

            no_ins_cur = new_no_ins_cur
            mask_pos = input_ids_new > 1
            input_ids = input_ids_new
            input_mask = mask_pos


    # print("sec_info:", sec)

    if total_num == 0:
        avg_kl = 0
    else:
        avg_kl = total_kl / total_num
    return input_ids, avg_kl, sec
