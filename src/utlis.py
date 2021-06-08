import sys
import time

import torch
import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs
import torch.nn.functional as F
import pandas as pd
import bottleneck as bn

def negative_sampling(pos_samples, num_entity, negative_num):
    pos_samples = pos_samples.unsqueeze(2)
    neg_samples = pos_samples.repeat(1, 1, 2 * negative_num).float()
    neg_samples[:, 0, :negative_num].uniform_(0, num_entity).long()
    neg_samples[:, 1, negative_num:].uniform_(0, num_entity).long()
    neg_samples = neg_samples.long()
    samples = torch.cat([pos_samples, neg_samples], dim=2)
    labels = torch.zeros((pos_samples.shape[0], 2 * negative_num + 1))
    labels[:, 0] = 1
    return samples.transpose(1, 2).reshape(-1, 2), labels.reshape(-1)


def get_noise_features(n_samples, n_features, noise_amount):
    noise_x, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                            cluster_std=noise_amount, centers=np.array([np.zeros(n_features)]))
    return noise_x


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def calc_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, test_pairs, tag_modification, eval_bz, eval_on_all,
#               item_item_data):
#     with torch.no_grad():
#         a = test_pairs[:, 0]
#         b = test_pairs[:, 1]
#         r = tag_modification
#         test_size = test_pairs.shape[0]
#         # perturb subject
#         ranks_s = perturb_and_get_raw_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, b, r, a, test_size,
#                                            eval_on_all, eval_bz, item_item_data)
#         # perturb object
#         ranks_o = perturb_and_get_raw_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, a, r, b, test_size,
#                                            eval_on_all, eval_bz, item_item_data)
#
#     ranks = torch.cat([ranks_s, ranks_o])
#     ranks += 1  # change to 1-indexed
#     return ranks


def calc_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, test_pairs, tag_modification, eval_bz, eval_on_all,
              midt_to_mid_interaction):
    with torch.no_grad():
        item_a = test_pairs[:, -2]
        item_b = test_pairs[:, -1]
        assert test_pairs.shape[1] == 2
        rel_ab = tag_modification
        test_size = test_pairs.shape[0]
        ranks_o = perturb_and_get_raw_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, item_a,
                                           rel_ab, item_b, test_size, eval_on_all, eval_bz, midt_to_mid_interaction)
        ranks = ranks_o
        ranks += 1  # change to 1-indexed
    return ranks


def perturb_and_get_raw_rank(args, model, midt_to_mid_array, item_embedding, tag_embedding, item_a, rel_ab, item_b,
                             test_size, eval_on_all, batch_size=None, midt_to_mid_interaction=None):
    """
    Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_item_a = item_a[batch_start:batch_end]
        batch_rel = rel_ab[batch_start:batch_end]
        target_item_b = item_b[batch_start:batch_end]
        assert batch_item_a.shape[0] == batch_rel.shape[0]
        assert target_item_b.shape[0] == batch_rel.shape[0]
        batch_item_a_idx = batch_item_a.to('cpu').numpy()
        bool_mask_matrix = sp_to_tensor(midt_to_mid_interaction[batch_item_a_idx]) > 0
        emb_a = item_embedding[midt_to_mid_array[batch_item_a_idx]]

        if eval_on_all:  # eval on all item
            eval_item_embedding = item_embedding
            target = midt_to_mid_array[target_item_b.to('cpu').numpy()]
        else:  # eval on tagged item
            print("eval on tagged item is deprecated.")
            exit(1)
            eval_item_embedding = item_embedding[midt_to_mid_array]
            target = target_item_b.to('cpu').numpy()
        
        if args.norm_CL_vec:
            emb_a = F.normalize(emb_a, dim=-1, p=2)
        
        emb_ar = model.compose_forward(emb_a, torch.mm(batch_rel, tag_embedding))
        
        # out-prod and reduce sumargpartition
        # =========
        if args.norm_CL_score:
            emb_ar = F.normalize(emb_ar, dim=-1, p=2)
            eval_item_embedding = F.normalize(eval_item_embedding, dim=-1, p=2)
        # =========
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # size: D x E x 1
        emb_c = eval_item_embedding.transpose(0, 1).unsqueeze(1)  # size: D x 1 x V
        # =========
        out_prod = torch.bmm(emb_ar, emb_c)  # size D x E x V
        score = torch.sum(out_prod, dim=0).detach()  # size E x V
        
        if torch.__version__ == '1.6.0':
            score = score.masked_fill(bool_mask_matrix.type(torch.BoolTensor).to('cuda'), float('-inf'))
        else:
            score = score.masked_fill(bool_mask_matrix.to('cuda'), float('-inf'))

        torch.cuda.empty_cache()
        ranks.append(sort_and_rank(score, torch.LongTensor(target).to(score.device)))
        
    return torch.cat(ranks)


def sort_and_rank(score, target):
    score[(range(target.shape[0]), target)] += -1e-11
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices.cpu()


def sp_to_tensor(interaction_data):
    if sparse.isspmatrix(interaction_data):
        user_interaction_data = interaction_data.toarray()
    else:
        user_interaction_data = interaction_data
    user_interaction_data = user_interaction_data.astype('float32')
    user_interaction_data = torch.from_numpy(user_interaction_data)
    return user_interaction_data


def split_train_test_proportion(data, test_prop=0.2, support_item_set=None):
    if support_item_set is not None:
        vad_data = data.loc[data['mid'].isin(support_item_set)]
    else:
        vad_data = data
    data_grouped_by_user = vad_data.groupby('uid')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u),
                                 replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def batch_item_similarity_ratio_matrix(tag_vec, batch_size=6000, target_tag_vec=None):
    num_item = tag_vec.shape[0]
    n_batch = (num_item + batch_size - 1) // batch_size
    tmp_list = []
    tag_tensor = torch.from_numpy(tag_vec).to('cuda')
    if target_tag_vec is not None:
        target_tag_vec = torch.from_numpy(target_tag_vec).to('cuda')
    for idx in range(n_batch):
        print('{}/{}.'.format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(num_item, (idx + 1) * batch_size)
        batch_tag_tensor = tag_tensor[batch_start:batch_end]
        if target_tag_vec is not None:
            similarity_matrix = torch.mm(batch_tag_tensor, target_tag_vec.t())
            batch_tag_distance_vec = -(
                        2 * similarity_matrix - batch_tag_tensor.sum(dim=1).reshape(-1, 1) - target_tag_vec.t().sum(
                    dim=0).reshape(1, -1))
            batch_tag_distance_vec = (similarity_matrix+1) / batch_tag_distance_vec
        else:
            similarity_matrix = torch.mm(batch_tag_tensor, tag_tensor.t())
            batch_tag_distance_vec = -(
                        2 * similarity_matrix - batch_tag_tensor.sum(dim=1).reshape(-1, 1) - tag_tensor.t().sum(
                    dim=0).reshape(1, -1))
            batch_tag_distance_vec = (similarity_matrix+1) / batch_tag_distance_vec
        assert (batch_tag_distance_vec >= 0).all()
        del similarity_matrix
        torch.cuda.empty_cache()
        batch_tag_distance_vec = batch_tag_distance_vec.to('cpu').int()
        torch.cuda.empty_cache()
        tmp_list.append(batch_tag_distance_vec)
    t = time.time()
    tag_distance_vec_full = torch.cat(tmp_list, dim=0).numpy()
    print("create tag_distance_vec_full: {}s".format(time.time() - t))
    return tag_distance_vec_full


def batch_item_similarity_matrix(tag_vec, batch_size=6000, target_tag_vec=None):
    num_item = tag_vec.shape[0]
    n_batch = (num_item + batch_size - 1) // batch_size
    tmp_list = []
    tag_tensor = torch.from_numpy(tag_vec).to('cuda')
    if target_tag_vec is not None:
        target_tag_vec = torch.from_numpy(target_tag_vec).to('cuda')
    for idx in range(n_batch):
        print('{}/{}.'.format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(num_item, (idx + 1) * batch_size)
        batch_tag_tensor = tag_tensor[batch_start:batch_end]
        if target_tag_vec is not None:
            similarity_matrix = torch.mm(batch_tag_tensor, target_tag_vec.t())
            batch_tag_distance_vec = -(
                        2 * similarity_matrix - batch_tag_tensor.sum(dim=1).reshape(-1, 1) - target_tag_vec.t().sum(
                    dim=0).reshape(1, -1))
        else:
            similarity_matrix = torch.mm(batch_tag_tensor, tag_tensor.t())
            batch_tag_distance_vec = -(
                        2 * similarity_matrix - batch_tag_tensor.sum(dim=1).reshape(-1, 1) - tag_tensor.t().sum(
                    dim=0).reshape(1, -1))
        assert (batch_tag_distance_vec >= 0).all()
        del similarity_matrix
        torch.cuda.empty_cache()
        batch_tag_distance_vec = batch_tag_distance_vec.to('cpu').int()
        torch.cuda.empty_cache()
        tmp_list.append(batch_tag_distance_vec)
    t = time.time()
    tag_distance_vec_full = torch.cat(tmp_list, dim=0).numpy()
    print("create tag_distance_vec_full: {}s".format(time.time() - t))
    return tag_distance_vec_full


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def old_add_tag_infomation(filtered_positive_array, tag_vec):
    # 4.convert pair to tuple, add tag modification information
    modification_tags = tag_vec[filtered_positive_array[:, -2]] - tag_vec[filtered_positive_array[:, -1]]
    # assert (np.sum(np.abs(modification_tags), axis=1) > 0).all()
    return -modification_tags



def add_tag_infomation(filtered_positive_array, tag_vec, keep_one_per_row):
    modification_info = old_add_tag_infomation(filtered_positive_array, tag_vec)
    if not keep_one_per_row:
        return modification_info
    else:
        print('keep_one_per_row')
        exit(1)
        masked_modification_info = np.zeros_like(modification_info)
        for i in range(modification_info.shape[0]):
            selected_idx = np.random.choice(np.nonzero(modification_info[i])[0])
            masked_modification_info[i, selected_idx] = modification_info[i, selected_idx]
        return masked_modification_info


def create_modification_matrix(pairs, tag_vec, keep_one_per_row):
    modification_info = add_tag_infomation(pairs, tag_vec, keep_one_per_row)
    tag_modification_all = torch.zeros(modification_info.shape[0], modification_info.shape[1])
    if torch.__version__ == '1.6.0':
        tag_modification_all = tag_modification_all.masked_fill(torch.BoolTensor(modification_info > 0.1), 1)
        tag_modification_all = tag_modification_all.masked_fill(torch.BoolTensor(modification_info < -0.1), -1)
    else:
        tag_modification_all = tag_modification_all.masked_fill(torch.ByteTensor(modification_info > 0.1), 1)
        tag_modification_all = tag_modification_all.masked_fill(torch.ByteTensor(modification_info < -0.1), -1)
    # TODO: make sure only one nonzero value in each row of tag_modification_all
    return tag_modification_all


def mrr_rec(x_pred, heldout_batch):
    values, indices = torch.sort(-x_pred, dim=1)
    heldout_value = (-x_pred) * heldout_batch
    result = torch.searchsorted(values, heldout_value)
    rank_m = (result + 1) * heldout_batch
    idx = torch.nonzero(rank_m)
    rank = rank_m[idx[:,0], idx[:,1]]
    return rank.cpu().numpy()

def mrr_rec_np(x_pred, heldout_batch):
    values, indices = torch.sort(-x_pred, dim=1)
    heldout_value = (-x_pred) * heldout_batch
    values = values.to('cpu').numpy()
    heldout_value = heldout_value.to('cpu').numpy()
    heldout_batch = heldout_batch.to('cpu').numpy()
    # result = torch.searchsorted(values, heldout_value)
    result_list = []
    for i in range(values.shape[0]):
        tmp = np.searchsorted(values[i], heldout_value[i])
        tmp_rank = (tmp+1) * heldout_batch[i]
        idx = np.nonzero(tmp_rank)
        rank = tmp_rank[idx]
        result_list.append(rank)
    return np.concatenate(result_list)


def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def recall_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0
    return recall


def hit_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    return tmp.sum()


def filter_pair_by_class(positive_array, tagged_movie_genre_table, eval_batch=10000):
    import torch
    common_genre_num_list = []
    tagged_movie_genre_table = torch.from_numpy(tagged_movie_genre_table).to('cuda')
    data_size = positive_array.shape[0]
    n_batch = (data_size + eval_batch - 1) // eval_batch
    for idx in range(n_batch):
        batch_start = idx * eval_batch
        batch_end = min(data_size, (idx + 1) * eval_batch)
        batch_positive_array = positive_array[batch_start:batch_end]
        left_movie_genre = tagged_movie_genre_table[batch_positive_array[:, -2], :]
        right_movie_genre = tagged_movie_genre_table[batch_positive_array[:, -1], :]
        common_genre_num = torch.sum(left_movie_genre * right_movie_genre, dim=1).to('cpu').numpy()
        common_genre_num_list.append(common_genre_num)
    common_genre_num_all = np.concatenate(common_genre_num_list)
    assert common_genre_num_all.shape[0] == positive_array.shape[0]
    filtered_positive_array = positive_array[common_genre_num_all > 0]
    return filtered_positive_array


def create_positive_pair(tag_distance_vec_full, level=3, type='less_than', include_zero_dist=0):
    # TODO: use a->b and b->a
    tag_distance_vec_triu = tag_distance_vec_full
    # tag_distance_vec_triu = np.triu(tag_distance_vec_full)
    positive_list = []
    if include_zero_dist:
        print("do not include_zero_dist")
        exit(1)
        offset = 0
    else:
        offset = 1
    if type == 'less_than':
        for i in range(level):
            d = i+offset
            dist_tuple = np.where(tag_distance_vec_triu == d)
            positive_list += list(zip(dist_tuple[0], dist_tuple[1]))
    elif type == 'equal_to':
        print("do not equal_to")
        exit(1)
        dist_tuple = np.where(tag_distance_vec_triu == level)
        positive_list += list(zip(dist_tuple[0], dist_tuple[1]))
    else:
        print("error type.")
        exit(1)
    positive_array = np.array(positive_list)
    return positive_array


def check_inf_nan_tensor(x):
    if torch.isnan(x).any():
        print("has nan!")
        print(x)
    if torch.isinf(x).any():
        print("has inf!")
        print(x)