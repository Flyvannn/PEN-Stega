def sample_generate(model, input_ids, segment_ids, input_mask, no_ins=None, device='cuda', temperature=1.0, args=None,
                    tokenizer=None, sample_num=1, top_k=10, top_p=0.9, prevent=None, promote=None, reduce=None,
                    verbose=None):
    if not verbose:
        verbose = args.verbose
    zero_list = ["[", "]", "(", ")"]
    zero_ids = [tokenizer.vocab.get(x) for x in zero_list]
    # if verbose > 0:
    # print("\nInput %s" % (" ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in input_ids[0].detach().cpu().numpy() if x!=0])))
    no_ins_cur = no_ins[0][:(no_ins[0] == -1).nonzero()[0]]

    total_kl = 0
    total_num = 0
    with torch.no_grad():
        for ip in range(MAX_TURN):
            # print("Turn:", ip)

            result = model(input_ids, segment_ids, input_mask)
            mask_prediction_scores = result[0]
            input_len = torch.sum(input_mask, 1)

            base_log_prob = F.log_softmax(mask_prediction_scores, dim=-1)
            # base_log_prob, _ = torch.topk(base_log_prob, dim=2, k=(mask_prediction_scores.shape[-1]))

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
            # probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            len_input = len([x for x in input_ids[0].detach().cpu().numpy() if x != 0])
            _, top_ids = torch.topk(probs, dim=2, k=1)
            if torch.sum(top_ids[0, :len_input, 0] == 1) == len_input:
                return input_ids, total_kl / total_num

            if ip < 3:
                for i in range(len_input):
                    total_kl += kl(probs[0, i, :], log_probs[0, i, :], base_log_prob[0, i, :])
                    total_num += 1

            input_ids_new = torch.zeros_like(input_ids)
            top_predicts = torch.zeros([input_ids.shape[0], input_ids.shape[1], 3], dtype=torch.long)
            mask_predicts = torch.zeros_like(input_ids, dtype=torch.long)
            for t in range(args.max_seq_length):
                mask_predicts[:, t] = torch.multinomial(probs[:, t, :], num_samples=1)
                # top_predicts[:,t] = torch.topk(probs[:,t,:], k=3)[1]

            logit_new = torch.zeros_like(input_ids, dtype=torch.float)
            input_ids_ori = input_ids
            # top_predicts_new = torch.zeros_like(top_predicts)
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
                    # top_predicts_new[0,k,:] = top_predicts[0,j,:]
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

    # print("total kl:{}".format(total_kl))
    avg_kl = total_kl / total_num
    # print("average kl:{}".format(avg_kl))
    return input_ids, avg_kl