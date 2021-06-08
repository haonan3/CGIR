import csv
import os
import pickle
import time
from collections import defaultdict
import random

import pandas as pd
from scipy import sparse
import numpy as np
import logging

from spacy.cli.init_model import read_vectors

from src.utlis import filter_pair_by_class, batch_item_similarity_matrix

logging.basicConfig(level=logging.DEBUG)

from src.preprocess_ml import create_positive_pair, create_movie_genre_table, create_genre_table, load_data

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def load_modification_score_data(args, tag_vec, pid_to_itemId_array, num_modif=200):
    if args.data == 'ali':
        # For ali data, we use binary value as score, and because all items have tag list, therefore, all items have score
        pid_to_tid_score = tag_vec
        # pid_with_score_array is all mid
        pid_to_pidt_dict = load_pickle(args.data, 'pid_to_pidt_dict')
        pid_with_score_array = np.zeros((len(pid_to_pidt_dict, )))
        for idx, (k, v) in enumerate(pid_to_pidt_dict.items()):
            pid_with_score_array[idx] = v
        assert pid_with_score_array.max()+1 == len(pid_with_score_array)
        tag_to_tid_dict = load_pickle('ali', 'tag_to_tid_dict')
    else:
        pid_to_tid_score = load_pickle(args.data, 'pid_to_tid_score')
        pid_with_score_array = load_pickle(args.data, 'pid_with_score_array')
        tag_to_tid_dict = {}
        tag_table_data = load_data(parent_path + '/data/{}/pro_sg/standard_tag_table.csv'.format(args.data))
        for data in tag_table_data:
            tag_to_tid_dict[data[1]] = int(data[0])
    
    # create add tag item pair and modification array
    add_modification_pair = load_pickle(args.data, 'add_modification_pair')
    random.Random(args.seed).shuffle(add_modification_pair)
    add_modification_pair = np.array(add_modification_pair)[:num_modif] # random test a small part for speedup
    print(add_modification_pair.shape)
    add_gradient_modification = np.zeros((add_modification_pair[:, 0].shape[0], tag_vec.shape[1]))
    add_gradient_modification_idx = list(zip(list(range(add_modification_pair[:, 0].shape[0])),
                                                            list(add_modification_pair[:, 1])))
    add_gradient_modification[tuple(np.array(add_gradient_modification_idx).T)] = 1

    # create remove tag item pair and modification array
    remove_modification_pair = load_pickle(args.data, 'remove_modification_pair')
    random.Random(args.seed).shuffle(remove_modification_pair)
    remove_modification_pair = np.array(remove_modification_pair)[:num_modif] # random test a small part for speedup
    print(remove_modification_pair.shape)
    remove_gradient_modification = np.zeros((remove_modification_pair[:, 0].shape[0], tag_vec.shape[1]))
    add_gradient_modification_idx = list(zip(list(range(remove_modification_pair[:, 0].shape[0])),
                                                            list(remove_modification_pair[:, 1])))
    remove_gradient_modification[tuple(np.array(add_gradient_modification_idx).T)] = 1
    pid_with_score_array = pid_with_score_array.astype(int)
    pid_with_score_to_itemId_array = pid_to_itemId_array[pid_with_score_array].astype(int)
    assert (pid_with_score_to_itemId_array > 0).all()

    modification_to_str = {}
    for tag, tid in tag_to_tid_dict.items():
        modification_to_str[tid] = tag
        
    return pid_to_tid_score, pid_with_score_array, add_modification_pair, add_gradient_modification, \
           remove_modification_pair, remove_gradient_modification, pid_with_score_to_itemId_array, modification_to_str

def new_load_data(args):
    print("Loading dataset: {}.".format(args.data))
    pid_to_pidt_dict = load_pickle(args.data, 'pid_to_pidt_dict')
    tag_vec = load_pickle(args.data, 'tag_vec')
    tag_vec = tag_vec.toarray() if args.data == 'ali' else tag_vec
    itemId_to_pid_dict = load_pickle(args.data, 'itemId_to_pid_dict')
    w2v_slice_array = load_pickle(args.data, 'w2v_slice_array')
    tid_over_used_word_matrix = load_pickle(args.data, 'tid_over_used_word_matrix')
    num_items = len(itemId_to_pid_dict)
    if args.data in ['ml-25m', 'ml-20m']:
        genre_dict = create_genre_table(args.data)
        category_table = create_movie_genre_table(genre_dict, itemId_to_pid_dict, pid_to_pidt_dict, args.data)
    elif args.data == 'ali':
        sp_tagged_item_cate_table = load_pickle('ali', 'sp_tagged_item_cate_table')
        category_table = sp_tagged_item_cate_table.toarray()
    else:
        category_table = None
        print("{} doesn't exist.".format(args.data))
        exit(1)
    user_interaction_data = load_interaction_data(parent_path + '/data/{}/pro_sg/user_item_interaction.csv'.format(args.data), num_items, args.data)
    tag_distance_vec_full = batch_item_similarity_matrix(tag_vec, batch_size=5000)
    test_pairs = load_pickle(args.data, 'test_pair')
    tag_distance_vec_full[(test_pairs[:, 0], test_pairs[:, 1])] = -1
    print("test pairs num: {}".format(len(test_pairs)))
    
    pidt_to_pid_dict = {}
    for k,v in pid_to_pidt_dict.items():
        pidt_to_pid_dict[v] = k

    pidt_to_pid_array = np.zeros(len(pidt_to_pid_dict))
    for pidt,pid in pidt_to_pid_dict.items():
        pidt_to_pid_array[pidt] = pid
        
    pid_to_itemId_array = -np.ones(max(list(itemId_to_pid_dict.values()))+1)
    for itemId, pid in itemId_to_pid_dict.items():
        pid_to_itemId_array[pid] = itemId
    
    positive_array = create_positive_pair(tag_distance_vec_full, level=args.modification_level, type='less_than', include_zero_dist=args.include_zero_dist)
    filtered_positive_array = filter_pair_by_class(positive_array, category_table)
    
    # mask
    num_midt = len(pid_to_pidt_dict)
    num_mid = len(itemId_to_pid_dict)
    tail_mid = pidt_to_pid_array[positive_array[:,1]]
    head_midt = positive_array[:,0]
    train_midt_to_mid_interaction = sparse.csr_matrix((np.ones_like(head_midt), (head_midt, tail_mid)),
                                                      dtype='float64', shape=(num_midt, num_mid))
    
    # word2vec
    print("Loading word mebedding data...")
    word_dict, w2v_data = load_word_embedding(dataset=args.data, debug=False)
    
    print("Finish Data Loading!")
    return user_interaction_data, test_pairs, filtered_positive_array, pid_to_pidt_dict, pidt_to_pid_dict, \
           w2v_data, tag_vec, category_table, w2v_slice_array, tid_over_used_word_matrix,\
           train_midt_to_mid_interaction, pid_to_itemId_array


def load_word_embedding(dataset, debug=False):
    if debug:
        return None, np.zeros((10000, 300))
    else:
        if dataset == 'ali':
            cn_w2v = parent_path + '/data/sgns.weibo.bigram'
            vectors, iw, wi, dim = read_vectors(cn_w2v)
            w2v_data_list = []
            for k, v in vectors.items():
                w2v_data_list.append(v.reshape(1, -1))
            data = np.concatenate(w2v_data_list, axis=0)
            word_dict = None
        else:
            lines = open(parent_path + '/data/new_glove.6B.300d.txt').readlines()
            data = []
            word_dict = {}
            for idx, line in enumerate(lines):
                tokens = line.strip('\n')
                word, vec = tokens.split('\t')
                vec_nums = vec.split(' ')
                word_dict[word] = idx
                temp_ = [float(i) for i in vec_nums]
                assert len(temp_) == 300
                data.append(temp_)
            data = np.array(data)
            assert data.shape[1] == 300
            print("Loaded data. #shape = " + str(data.shape))
            print(" #words = %d " % (len(word_dict)))
        return word_dict, data


def load_pickle(dataset_name, var):
    with open(parent_path + '/data/' + dataset_name + '/pro_sg/' + var + '.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def load_interaction_data(csv_file, n_items, dataset="ml-25m"):
    item_str = 'pid'
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1
    rows, cols = tp['uid'], tp[item_str] # user 160775, item = 38715
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data


def compute_sparsity(X):
    non_zeros = 1. * np.count_nonzero(X)
    total = X.size
    sparsity = 100. * (1 - (non_zeros) / total)
    return sparsity


def dump_vectors(X, outfile, words):
    print("shape", X.shape)
    assert len(X) == len(words)
    fw = open(outfile, 'w')
    for i in range(len(words)):
        fw.write(words[i] + " ")
        for j in X[i]:
            fw.write(str(j) + " ")
        fw.write("\n")
    fw.close()