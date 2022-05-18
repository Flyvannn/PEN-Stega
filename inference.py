import time
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
from util import MAX_TURN, PREVENT_FACTOR, PROMOTE_FACTOR, PREVENT_LIST, REDUCE_LIST, STOP_LIST, boolean_string, top_k_top_p_filtering, kl
from huffman_encode import *
from arithmetic_encode import *
from full_binary_encode import full_binary_tree_embed
from multi_base_encode import *
InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids no_ins")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)
NOI_ID = 1
class Node(object):
    def __init__(self, input_ids, segment_ids, input_mask, score, shift, length, pos_start, input_len_start):
        super(Node, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids  # parent Node, None for root
        self.input_mask = input_mask
        self.score = score
        self.shift = shift
        self.length=length
        self.pos_start=pos_start
        self.input_len_start=input_len_start

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def convert_example_to_features(example, tokenizer, max_seq_length, id = 0, no_ins_at_first = False, tokenizing = False):
    tokens = ["[CLS]"] + example  
    if len([x for t in tokens for x in tokenizer.encode(t)]) > max_seq_length:
        logging.info(f"Warning: input id-{id} exceeds max sequence length limit!")
        tokens = ["[CLS]"] + ["Error : Input exceeds length limit;"]

    no_ins = [0] if no_ins_at_first else []
    if tokenizing:
        #input_ids = tokenizer.encode(" ".join(tokens))
        input_ids = [x for t in tokens for x in tokenizer.encode(t)]
        input_ids_lens = [len(tokenizer.encode(t)) for t in tokens]
        cur = 0
        for l in input_ids_lens:
            if l >=2 :
                no_ins.extend([cur + x for x in range(0,l-1)])
            cur += l
    else:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)

    no_ins_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)

    no_ins_array[:len(no_ins)] = no_ins

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             no_ins=no_ins_array,
                             )
    return features

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, max_seq_len = 256, sep=" ", no_ins_at_first = False, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path
        num_samples = sum(1 for line in open(data_file))
        self.num_samples = num_samples
        seq_len = max_seq_len
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            no_ins = np.memmap(filename=self.working_dir/'no_ins.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            no_ins =  np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                if i >= num_samples:
                    break
                line = line.strip()
                example = [s.lstrip().strip() for s in line.split(sep)]
                features = convert_example_to_features(example, tokenizer, seq_len, no_ins_at_first = no_ins_at_first, id = i, tokenizing=True)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                no_ins[i] = features.no_ins
        if i != num_samples - 1:
            logging.info("i={} not equal to num_samples={}".format(i, num_samples))
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.no_ins = no_ins


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.no_ins[item].astype(np.int64)),
                )


def main():
    parser = ArgumentParser()
    parser.add_argument('--keyfile', type=Path, default="./input/news.5key.txt")
    parser.add_argument('--output_dir', type=Path, required=False, default="./result/all_the_news_result")
    parser.add_argument("--bert_model", type=str, default=r"./ckpt/all_the_news_model", help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", 
                        type=boolean_string, 
                        default=False, 
                        )
    parser.add_argument("--reduce_memory",                        
                        type=boolean_string, 
                        default=False, 
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", 
                        type=boolean_string, 
                        default=False, 
                        help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16', 
                        type=boolean_string, 
                        default=False, 
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed',
                        type=int,
                        default=1111,
                        help="random seed for initialization")
    parser.add_argument("--type",
                        default="sampling",
                        type=str,
                        choices=['greedy','sampling','greedy_embed','greedy_extract','multi_base_embed','arithmetic_embed','huffman_embed'],
                        help="greedy: greedy generation. sampling: top-k sampling")
    parser.add_argument('--noi_decay',
                        type=int,
                        default=1,
                        help="round number to decay NOI prob") 
    parser.add_argument('--reduce_decay',
                        type=int,
                        default=1,
                        help="round number to decay reduce prob") 
    parser.add_argument('--verbose', type=int,
                        default=0,
                        help="verbose level") 
    parser.add_argument('--n_test',
                        type=int,
                        default=5000,
                        help="number of test examples")
    parser.add_argument('--prevent', 
                        type=boolean_string, 
                        default=True,
                        help="avoid generating several words")
    parser.add_argument('--reduce_stop',
                        type=boolean_string, 
                        default=True, 
                        help="reduce stopwords")    
    parser.add_argument('--lessrepeat',
                        type=boolean_string, 
                        default=True, 
                        help="reduce repetition (only for tokenwise)")
    parser.add_argument('--sep',
                         type=str, default=" ", help="token to seperate keywords")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=50,
                        help="max sequence length") 
    parser.add_argument("--no_ins_at_first",
                        type=boolean_string,
                        default=False,
                        help="Do not insert at the begining of the text")
    parser.add_argument("--global_pos",
                        type=int,
                        default=0,
                        help="the start of secret info")
    parser.add_argument('--top_k',
                        type=int,
                        default=8,
                        help="top k ")
    parser.add_argument('--num_token',
                        type=int,
                        default=0,
                        help="tokens")

    args = parser.parse_args()



    if not args.output_dir:
        args.output_dir = args.bert_model

    epoch_file = args.keyfile
    # args.max_seq_length = 256
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    args.output_mode = "classification"


    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)


    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)

    sep_tok = tokenizer.vocab['[SEP]']
    cls_tok = tokenizer.vocab['[CLS]']
    pad_tok = tokenizer.vocab['[PAD]']

    model.to(device)
    model.eval()

    print(args)    

    epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.keyfile, tokenizer=tokenizer, max_seq_len = args.max_seq_length, sep=args.sep, no_ins_at_first = args.no_ins_at_first, num_data_epochs=1)
    epoch_sampler = SequentialSampler(epoch_dataset)
    generate_dataloader = DataLoader(epoch_dataset, sampler=epoch_sampler,batch_size=args.batch_size)
    if args.type == 'arithmetic_embed':
        file_name = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}_{args.top_k}.txt")
    else:
        file_name = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}.txt")

    #extract
    ext_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}.txt")
    if "extract" in args.type:
        gen_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type[:-7]}embed.txt")
        word_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type[:-7]}embed_words.txt")
    else:
        word_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}_words.txt")

    num_key = 0
    with open(args.keyfile) as r:
        keys = r.readline()
    for kk in keys:
        num_key += len(kk.split())

    if "embed" in args.type:
        sec_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}_info.txt")
        f_sec = open(sec_file, "w", 1)

    if "extract" in args.type:
        with open(gen_file) as f1:
            lines = f1.readlines()
            # print(lines)
        f = open(ext_file, "w", 1)
    else:
        f = open(file_name, "w", 1)

    if "greedy_extract" == args.type:
        with open(word_file) as f2:
            words = f2.readlines()
            # print(words)

    with open("./data/bit_stream.txt", "r", encoding='utf-8') as r:
        bitstrs = r.readlines()
    bitstrs = bitstrs[0]

    logging.info("***** Running generation *****")
    logging.info(f"  Num examples = {epoch_dataset.num_samples}")
    logging.info("  Batch size = %d", args.batch_size)
    logging.info(f"  Save to {file_name}")


    prevent = [ tokenizer.vocab.get(x) for x in PREVENT_LIST] if args.prevent else None
    if args.reduce_stop:
        # import pdb; pdb.set_trace()
        reduce_l = REDUCE_LIST |  STOP_LIST
    reduce = None
    if args.prevent:
        reduce = [ tokenizer.vocab.get(x) for x in reduce_l]  
        reduce = [s for s in reduce if s]


    all_kl = 0
    total_num = 0
    start_time = time.time()
    with tqdm(total=len(generate_dataloader), desc=f"Epoch {0}") as pbar:
        for step, batch in enumerate(generate_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, no_ins = batch
            # print("no_ins",type(no_ins),no_ins)
            if args.type == "greedy":
                predict_ids, avg_kl = greedy_search(model, input_ids, segment_ids, input_mask, no_ins = no_ins, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'sampling':
                predict_ids, avg_kl = sample_generate(model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'greedy_embed':
                predict_ids, embed_words, avg_kl, sec = greedy_embed(model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'full_binary_tree_embed':
                predict_ids, avg_kl, sec = full_binary_tree_embed(model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'multi_base_embed':
                predict_ids, avg_kl, sec = multi_base_embed(bitstrs, model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=1.0, args=args, tokenizer=tokenizer, top_k=300, top_p=0.95, prevent=prevent, reduce= reduce)
            elif args.type == 'huffman_embed':
                predict_ids, avg_kl, sec = huffman_embed(bitstrs, model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'arithmetic_embed':
                predict_ids, avg_kl, sec = arithmetic_embed(bitstrs, model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=1.0, args=args, tokenizer=tokenizer, top_k=args.top_k, top_p=0.95, prevent=prevent, reduce= reduce)
            elif args.type == 'greedy_extract':
                gen_str = lines[step]
                emb_wds = words[step]
                sec_info = greedy_extract(gen_str, emb_wds, model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
                print("sec_info:",sec_info)
            else:
                raise NotImplementedError
            all_kl += avg_kl
            if "extract" not in args.type:
                output = " ".join(
                    [str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in
                     predict_ids[0].detach().cpu().numpy() if
                     x != sep_tok and x != pad_tok and x != cls_tok]) + "\n"
                output = output.replace(" ##", "")
                # print("result:", output)
                f.write(output)
                total_num += len(output.split())

            else:
                f.write(sec_info+"\n")

            if "embed" in args.type:
                f_sec.write(sec + '\n')

            if args.type == "greedy_embed":
                with open(word_file, "a+") as ff:
                    for i in range(len(embed_words)):
                        ff.write(str(embed_words[i]) + " ")
                    ff.write("\n")

            pbar.update(1)
    cost_time = time.time() - start_time
    print("words/s:", (total_num-num_key)/cost_time)
    print("+noi words/s:", (args.num_token)/cost_time)
    print("avg len:",total_num/len(generate_dataloader))
    print("mean kl:{}".format(all_kl/len(generate_dataloader)))

    if "embed" in args.type:
        sec_file = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(
            args.bert_model) + f".{args.type}_info.txt")
        bpw = 0
        with open(file_name, "r", encoding='utf-8') as f:
            lines = f.readlines()
        with open(sec_file, "r", encoding='utf-8') as r:
            bitstrs = r.readlines()
        for i in range(len(lines)):
            bpw += len(bitstrs[i].strip()) / len(lines[i].split())
        print("bpw:{}".format(bpw / len(lines)))

if __name__ == '__main__':
    main()
