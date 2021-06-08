import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse

from src.utlis import calc_rank, ndcg_binary_at_k_batch, recall_at_k_batch, sp_to_tensor, hit_at_k_batch, mrr_rec_np

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

# this test_data use idx of tagged movie
def evaluation_IR(model, test_data, tag_modification, word_embedding, eval_on_all,
                  sampled_test=True, hits=[50], eval_batch=15, midt_to_mid_interaction=None):
    tag_embedding = model.get_tag_embedding(word_embedding)
    if sampled_test:
        item_a = model.items[model.midt_to_mid_array[test_data[:, 0].to('cpu').numpy()]]
        target_b = model.items[model.midt_to_mid_array[test_data[:, 1].to('cpu').numpy()]]
        num_tag = tag_embedding.shape[0]
        neg_tag_idx = np.random.choice(num_tag, 2 * model.args.negative_num * item_a.shape[0])
        neg_modification_tag_embedding = tag_embedding[neg_tag_idx]
        pos_modification_tag_embedding = torch.mm(tag_modification, tag_embedding)
        modification = torch.cat([pos_modification_tag_embedding, neg_modification_tag_embedding], dim=0)
        modification = modification.view(-1, item_a.shape[0], modification.shape[-1])
        modification = modification.permute(1, 0, 2)  # (100, 21, 1024)
        
        if model.args.norm_CL_vec:
            item_a = F.normalize(item_a, dim=-1, p=2)
            
        reshaped_a = item_a.unsqueeze(1)  # (100, 1, 1024)
        pred_b = modification + reshaped_a
        # pred_b = model.compose_forward(reshaped_a, modification)
        if model.args.norm_CL_score:
            pred_b = F.normalize(pred_b, dim=-1, p=2)
            target_b = F.normalize(target_b, dim=-1, p=2)

        target_b = target_b.unsqueeze(1)
        pred_score = (pred_b * target_b).contiguous().view(-1, item_a.shape[-1]).sum(dim=-1)
        preds = -pred_score.view(-1, 1 + 2 * model.args.negative_num)
        preds = preds.to('cpu').numpy()
        tmp = np.where(preds.argsort(axis=1) == 0)[1]
        mrr = np.mean(1.0 / (1 + tmp))
        return mrr, [0]
    else:
        item_embedding = model.get_all_item_embed()
        ranks = calc_rank(model.args, model, model.midt_to_mid_array, item_embedding, tag_embedding, test_data,
                          tag_modification, eval_batch, eval_on_all, midt_to_mid_interaction)
        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))
        avg_count_list = []
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
            avg_count_list.append(avg_count.item())
        return mrr.item(), avg_count_list


def save_vis(run_str, pid_to_itemId_array, modification_to_str, in_vis_result, de_vis_result):
    # save vis cube to text
    (in_scored_pid_cube, in_modification_pair, in_mode) = in_vis_result # scored_pid_cube: all_item * topN * steps
    (de_scored_pid_cube, de_modification_pair, de_mode) = de_vis_result
    file_name = run_str + '.txt'
    with open(parent_path + '/vis/' + file_name, 'w') as file:
        for i in range(in_modification_pair.shape[0]):
            # increase
            itemId = pid_to_itemId_array[in_modification_pair[i,0]]
            modif_str = modification_to_str[in_modification_pair[i,1]]
            vis_per_item = in_scored_pid_cube[i].T # steps * topN
            for j in range(vis_per_item.shape[0]):
                line = str(int(itemId)) + ' ' + in_mode + ' ' + modif_str + ' ' \
                       + ' '.join([str(int(pid_to_itemId_array[id])) for id in list(vis_per_item[j])]) \
                       + ' step_{}'.format(j+1) + '\n'
                file.write(line)
            # decrease
            for i in range(de_modification_pair.shape[0]):
                itemId = pid_to_itemId_array[de_modification_pair[i, 0]]
                modif_str = modification_to_str[de_modification_pair[i, 1]]
                vis_per_item = de_scored_pid_cube[i].T  # steps * topN
                for j in range(vis_per_item.shape[0]):
                    line = str(int(itemId)) + ' ' + de_mode + ' ' + modif_str + ' ' \
                           + ' '.join([str(int(pid_to_itemId_array[id])) for id in list(vis_per_item[j])]) \
                           + ' step_{}'.format(j + 1) + '\n'
                    file.write(line)


def gradient_retrival(model, word_embedding, pid_to_pidt_dict, pid_to_tid_score, pid_with_score_array, tag_vec, mode,
                            start_pid, gradient_modification, gradient_eval_modification_idx,  modification_pair,
                            gradient_test_batch_size, topN=20, step_num=20):
    if mode == 'increase':
        step_size = 0.05
    elif mode == 'decrease':
        step_size = -0.05
    else:
        step_size = None
        print("wrong mode!")
        exit(1)

    tag_embedding = model.get_tag_embedding(word_embedding)
    test_num = start_pid.shape[0]
    item_embedding = model.get_all_item_embed()
    batch_size = gradient_test_batch_size
    n_batch = (test_num + batch_size - 1) // batch_size
    batch_retrieved_score_steps = []
    scored_pid_list = []
    for idx in range(n_batch):
        alpha = 0.0
        batch_start = idx * batch_size
        batch_end = min(test_num, (idx + 1) * batch_size)
        batch_start_pid = start_pid[batch_start:batch_end]
        batch_gradient_modification = gradient_modification[batch_start:batch_end]
        batch_gradient_eval_modification_idx = gradient_eval_modification_idx[batch_start:batch_end]
        retrieved_score_steps_list = []
        scored_pid_steps = []
        
        for i in range(step_num): # i is for each step
            emb_a = item_embedding[batch_start_pid]
            modification = torch.mm(batch_gradient_modification, tag_embedding)
            assert modification.shape[0] == emb_a.shape[0]

            if model.args.norm_CL_vec:
                emb_a = F.normalize(emb_a, dim=-1, p=2)

            alpha_modification = alpha * modification
            emb_ar = model.compose_forward(emb_a,  alpha_modification)
            scored_item_embedding = item_embedding[pid_with_score_array]

            if model.args.norm_CL_score:
                emb_ar = F.normalize(emb_ar, dim=-1, p=2)
                scored_item_embedding = F.normalize(scored_item_embedding, dim=-1, p=2)

            emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # size: D x E x 1
            emb_c = scored_item_embedding.transpose(0, 1).unsqueeze(1)  # size: D x 1 x V
            out_prod = torch.bmm(emb_ar, emb_c)  # size D x E x V
            score = torch.sum(out_prod, dim=0).to('cpu').numpy()  # size E x V

            sorted_idx = np.argsort(score, axis=1)
            top_n_pid_idx = sorted_idx[:, :topN]
            batch_modif_idx_repeat = np.repeat(batch_gradient_eval_modification_idx, topN)
            batch_retrived_item_score = pid_to_tid_score[(pid_with_score_array[top_n_pid_idx.reshape(-1)], batch_modif_idx_repeat)].reshape(-1,topN) # batch_item * topN score
            assert (batch_retrived_item_score >= 0).all()
            retrieved_score_steps_list.append(np.expand_dims(batch_retrived_item_score.sum(axis=-1), axis=-1)) # list of batch_item * 1
            scored_pid = np.expand_dims(pid_with_score_array[top_n_pid_idx.reshape(-1)].reshape(-1,topN), axis=-1) # batch_item * topN * 1
            scored_pid_steps.append(scored_pid)
            alpha += step_size # update alpha
            
        batch_retrieved_score_steps.append(np.concatenate(retrieved_score_steps_list, axis=-1)) # list of: batch_item * num_steps
        scored_pid_list.append(np.concatenate(scored_pid_steps, axis=-1)) # batch_item * topN * steps
    gradient_score = np.concatenate(batch_retrieved_score_steps, axis=0) # all_item * num steps
    scored_pid_cube = np.concatenate(scored_pid_list, axis=0) # all_item * topN * steps

    assert len(start_pid) == scored_pid_cube.shape[0]
    irr_score_list = []
    for pid_idx, pid in enumerate(start_pid):
        pidt = pid_to_pidt_dict[pid]
        pid_tag_vec = tag_vec[pidt]
        pid_retrieved_items = scored_pid_cube[pid_idx].reshape(-1,) # topN * steps
        if mode == 'increase':
            assert pid_tag_vec[gradient_eval_modification_idx[pid_idx]] == 0
        else:
            assert pid_tag_vec[gradient_eval_modification_idx[pid_idx]] == 1
        orig_tag_idxs = np.nonzero(pid_tag_vec)[0]
        pid_irr_score_list = []
        for tag_idx in orig_tag_idxs:
            if tag_idx != gradient_eval_modification_idx[pid_idx]: # not the modification tag
                irr_tag_score = pid_to_tid_score[(pid_retrieved_items, np.repeat(tag_idx, topN*step_num))].reshape(topN, step_num) # topN * step_num
                pid_irr_score_list.append(np.expand_dims(irr_tag_score, axis=0)) # list of 1*topN*step_num
        if len(pid_irr_score_list) > 0:
            pid_irr_cube = np.concatenate(pid_irr_score_list, axis=0) # num_irr_tag*topN*step_num
            pid_irr_score_cube = pid_irr_cube.sum(axis=1) # num_irr_tag*step_num
            pid_irr_score_diff_cube = pid_irr_score_cube[:, 1:] - pid_irr_score_cube[:,:-1]
            pid_irr_score_diff_cube[pid_irr_score_diff_cube > 0] = 1
            pid_irr_score_diff_cube[pid_irr_score_diff_cube < 0] = -1
            pid_irr_score = pid_irr_score_diff_cube.mean(axis=1).mean()
            irr_score_list.append(pid_irr_score)
        else:
            irr_score_list.append(0)

    diff_with_prev_score = gradient_score[:, 1:] - gradient_score[:, :-1]
    diff_with_prev_score_binary = np.ones_like(diff_with_prev_score) # 1
    if mode=='increase':
        diff_with_prev_score_binary[diff_with_prev_score <= 0] = 0
    elif mode=='decrease':
        diff_with_prev_score_binary[diff_with_prev_score >= 0] = 0
    
    # score_array =
    score_array_per_item = diff_with_prev_score_binary.mean(axis=1)
    irr_score_array = 1-abs(np.array(irr_score_list))
    finally_score_array = irr_score_array*score_array_per_item
    threshold = 0.9
    if (finally_score_array>threshold).sum() > 0:
        vis_result = (scored_pid_cube[finally_score_array>threshold], modification_pair[finally_score_array>threshold], mode)
    else:
        vis_result = None
    return finally_score_array.mean(), vis_result, irr_score_array.mean(), score_array_per_item.mean()


def independence_level(embed):
    # embed: b*d, tensor
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    embed = embed.t()
    embed_m = embed - embed.mean(dim=1, keepdim=True)
    # Sum of squares across rows
    ss_embed = (embed_m ** 2).sum(1) # ?? mean?
    corr_coeff = torch.mm(embed_m, embed_m.t()) / torch.sqrt(torch.mm(ss_embed[:, None], ss_embed[None]))
    d = corr_coeff.shape[0]
    independence_level = 1 - (1/(d*(d-1))) * (corr_coeff.sum() - d)
    return independence_level.item()