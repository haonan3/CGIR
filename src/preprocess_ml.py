'''
This script is for ml-25m data processing.
Raw data location: data/ml-25m/ml-25m-raw
Processed data location: data/ml-25m/pro_sg

1. we need user-item interaction table. uid, sid (some filter trick look paper)
2. we need item-tag talbe. sid,tag1,tag2,...
    (1.all tag cannot be low frequency
    2.all movie at least two high-confidence tag)
3. base on 2 create (target, query, modification) dataset
4. all modification text (will used by sparse text auto-encoder)
'''

import os
import pickle
import time
from collections import defaultdict
import random
import re
import pandas as pd
from functools import partial
import torch
from scipy import sparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np


from src.utlis import split_train_test_proportion, batch_item_similarity_matrix, add_tag_infomation, \
    filter_pair_by_class, create_positive_pair

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def load_movies_data(dataset_name):
    movies_file_path = parent_path + '/data/{}/{}-raw/movies.csv'.format(dataset_name, dataset_name)
    data_cache = []
    with open(movies_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0: # read head
                meta_list = line.strip().split(',')
                num_col = len(meta_list)
                print("Meta list of {}: {}.".format(movies_file_path.split('/')[-1], ' '.join(meta_list)))
            else:
                movieid, rest_str = line.strip().split(',', 1)
                title_str, genres = rest_str.rsplit(",", 1)
                data_cache.append([movieid, genres.strip().lower()])
    return data_cache


def load_tags_data(dataset_name):
    data_cache = []
    tags_file_path = parent_path + '/data/{}/{}-raw/tags.csv'.format(dataset_name, dataset_name)
    with open(tags_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0: # read head
                meta_list = line.strip().split(',')
                num_col = len(meta_list)
                print("Meta list of {}: {}.".format(tags_file_path.split('/')[-1], ' '.join(meta_list)))
            else:
                userid, rest_str = line.strip().split(",", 1)
                movieid, rest_str = rest_str.split(",", 1)
                raw_tag, _ = rest_str.rsplit(",", 1)
                data_cache.append([userid, movieid, raw_tag.strip().lower()])
    return data_cache


def load_data(path):
    t = time.time()
    num_col = -1
    data_cache = []
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0: # read head
                meta_list = line.strip().split(',')
                num_col = len(meta_list)
                print("Meta list of {}: {}.".format(path.split('/')[-1], ' '.join(meta_list)))
            else:
                data_list = line.strip().split(',')
                data_cache.append(data_list)
    print("Load {}, spend {}s.".format(path.split('/')[-1], time.time()-t))
    return data_cache


def data_stat(tag_tags_data_new, tag_tags_data, user_ratings_data, movie_ratings_data,
              user_tags_data, movie_tags_data, movie_movies_data, movie_scores_data,
              tag_scores_data, genome_tag_set, tags_data, movies_data, composed_genre_data):
    # TODO: Data Stat
    #  (1)check num_user in ratings and num_user in tags
    #  (2)use interaction rule from jianxin's paper, check how many movies and users are filtered
    #  (3)check distribution of No.tag over movies and No.movies over tag --> find cutoff value
    #  (4) how many movie with only one tag
    print(len(tag_tags_data_new)) # 65409
    print(len(tag_tags_data)) # 73021
    print(len(user_ratings_data))  # 162541
    print(len(movie_ratings_data)) # 59047
    print(len(user_tags_data))     # 14592
    print(len(movie_tags_data))    # 45251
    print(len(movie_movies_data))  # 62423
    print(len(movie_ratings_data.union(movie_tags_data))) # 62423
    assert user_tags_data.issubset(user_ratings_data) # True
    print(len(movie_ratings_data.intersection(movie_tags_data))) # 41875
    print(len(movie_scores_data)) # 13816
    print(len(tag_scores_data)) # 1128
    print(len(tag_tags_data)) # 73021
    print(len(genome_tag_set.intersection(tag_tags_data_new)))  # 1128
    print(len(tag_tags_data_new)) # 65409


    print(len(composed_genre_data)) # 1639
    composed_genre_to_movies = defaultdict(list)
    for movie in movies_data:
        composed_genre_to_movies[movie[1]].append(movie[0])
    composed_genre_to_movie_num = defaultdict(int)
    for k,v in composed_genre_to_movies.items():
        composed_genre_to_movie_num[k] = len(v)


    tag_to_appear_num_dict = defaultdict(int)
    for tag in tags_data: # userId,movieId,tag_str
        tag_to_appear_num_dict[tag[2].lower()] += 1

    sorted(tag_to_appear_num_dict.items(), key=lambda k_v: k_v[1][0])


    genre_set = set()
    for movie in movies_data:
        for genre in movie[1].split('|'):
            genre_set.add(genre.lower())

    genre_to_movies = defaultdict(int)
    for movie in movies_data:
        for genre in movie[1].split('|'):
            genre_to_movies[genre] += 1
    # defaultdict(int,
    #             {'Adventure': 4145,
    #              'Animation': 2929,
    #              'Children': 2935,
    #              'Comedy': 16870,
    #              'Fantasy': 2731,
    #              'Romance': 7719,
    #              'Drama': 25606,
    #              'Action': 7348,
    #              'Crime': 5319,
    #              'Thriller': 8654,
    #              'Horror': 5989,
    #              'Mystery': 2925,
    #              'Sci-Fi': 3595,
    #              'IMAX': 195,
    #              'Documentary': 5605,
    #              'War': 1874,
    #              'Musical': 1054,
    #              'Western': 1399,
    #              'Film-Noir': 353,
    #              '(no genres listed)': 5062}) # we can just remove those (no genres listed)


    # verify can we just use genre

    movies_data = load_movies_data(dataset_name)
    valid_movie_num = 0
    total_genre_tag = 0
    movies_num_over_tag_num = defaultdict(int)
    for movie in movies_data:
        if movie[1] != '(no genres listed)':
            valid_movie_num += 1
            total_genre_tag += len(movie[1].split('|'))
            movies_num_over_tag_num[len(movie[1].split('|'))] += 1
    print(valid_movie_num) # 57361
    print(total_genre_tag/valid_movie_num) # 1.86
    print(sorted(movies_num_over_tag_num.items(), key=lambda k_v: k_v[0]))
    # [(1, 25569), (2, 18326), (3, 9852), (4, 2784), (5, 680), (6, 123), (7, 24), (8, 2), (10, 1)]


def interaction_process(all_ratings_data, movie_wo_genre):
    #  1.user must have watched at least five movies, cast >3.5 to 1, ow to 0
    #  2.remove no genre movies, movie must have ratings
    #  4.tag to lower case
    
    # 1.To binarize rating
    movieId_in_pos_ratings_data = set()
    pos_interaction_data = []
    for rating in all_ratings_data:
        user_id = int(rating[0])
        movie_id = int(rating[1])
        if float(rating[2]) > 3.5:
            pos_interaction_data.append([user_id, movie_id])
            movieId_in_pos_ratings_data.add(movie_id)
    print("all interaction number: {}.".format(len(all_ratings_data))) # 25000095
    print("pos interaction number: {}.".format(len(pos_interaction_data))) # 12452811
    
    # 2.movie: movie must have rating and genre
    valid_pos_movieId_set = movieId_in_pos_ratings_data - movie_wo_genre
    print("len movieId_in_pos_ratings_data: {}.".format(len(movieId_in_pos_ratings_data))) # 40858
    print("len movie_wo_genre: {}.".format(len(movie_wo_genre))) # 5062
    print("len valid_pos_movieId_set: {}.".format(len(valid_pos_movieId_set))) # 38715

    # 3.user: filter out users that have less than 5 times pos rating.
    user_pos_rated_movies_dict = defaultdict(set)
    for rating in pos_interaction_data:
        pos_user_id = int(rating[0])
        pos_movie_id = int(rating[1])
        if pos_movie_id in valid_pos_movieId_set:
            user_pos_rated_movies_dict[pos_user_id].add(pos_movie_id)
    filtered_user_id_set = set()
    for user, pos_rated_movies in user_pos_rated_movies_dict.items():
        if len(pos_rated_movies) >= 5:
            filtered_user_id_set.add(user)
    print("len user_pos_rated_movies_dict: {}.".format(len(user_pos_rated_movies_dict))) # 162342
    print("filtered_user_id_set: {}".format(len(filtered_user_id_set))) # 160775
    
    filtered_interaction_data = []
    for rating in pos_interaction_data:
        userId = rating[0]
        movie_id = rating[1]
        if userId in filtered_user_id_set and movie_id in valid_pos_movieId_set:
            filtered_interaction_data.append([userId, movie_id])
    
    print("len valid_pos_movieId_set {}.".format(len(valid_pos_movieId_set))) # 38715
    print("len filtered_user_id_set {}.".format(len(filtered_user_id_set))) # 160775
    print("len filtered_interaction_data {}.".format(len(filtered_interaction_data))) # 12437739
    # after filter by min_uc=5: user 160580, movie 40775
    return valid_pos_movieId_set, filtered_user_id_set, filtered_interaction_data


def process_genome_tag_str(tag_str):
    if "'s " in tag_str or "n't" in tag_str or "s'" in tag_str or "'re" in tag_str:
        tag_str = tag_str.replace("'s ", ' ').replace("n't ", 'not ').replace("s'", '').replace("'re", '')
    tag_str = tag_str.strip()
    pattern = re.compile(r"[!\"#\$%&\'\(\)\*\+,-.\/:;<=>\?@\[\\\]\^_\`\{\~\|]+")
    tag_str = re.sub(pattern, ' ', tag_str)
    tag_str = tag_str.strip()
    tag_str = ' '.join(tag_str.split())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(tag_str)
    tag_str_list = []
    for token in tokens:
        lemma_token = lemmatizer.lemmatize(token)
        if lemma_token not in stop_words:
            tag_str_list.append(lemma_token)
    tag_str = " ".join(tag_str_list)
    return tag_str


def process_genome_tag(valid_movie_set, dataset_name):
    genome_tags_data = load_data(parent_path + '/data/{}/{}-raw/genome-tags.csv'.format(dataset_name, dataset_name))
    standard_tag_list = set()
    for data in genome_tags_data:
        standard_tag_list.add(data[-1])
    standard_tag_list = list(standard_tag_list)
    
    processed_standard_tag_set = set()
    for tag_str in standard_tag_list:
        tag_str = process_genome_tag_str(tag_str)
        if not tag_str.isspace() and tag_str != '':
            processed_standard_tag_set.add(tag_str)
    
    tags_data = load_tags_data(dataset_name)  # userId,movieId,raw_tag
    tags_data_filtered = []
    for movie_tag in tags_data:
        if int(movie_tag[1]) not in valid_movie_set:
            continue
        tags_data_filtered.append([movie_tag[0], movie_tag[1], movie_tag[2]])
    tags_data = tags_data_filtered
    return tags_data, processed_standard_tag_set


def tag_process(valid_movie_set, dataset_name):
    # process high frequency tag
    tags_data, processed_standard_tag_set = process_genome_tag(valid_movie_set, dataset_name)
    auxillary_tag_in_standard_dict = defaultdict(set)
    auxillary_tag_ex_standard_dict = defaultdict(int)
    cleaned_tags = []
    cleaned_tags_standard = []
    for tag_info in tqdm(tags_data):
        if 'http' not in tag_info[2] and not tag_info[2].isspace() and tag_info[2] != '':
            tag_str = tag_info[2]
            if "'s " in tag_str or "n't" in tag_str or "s'" in tag_str or "'re" in tag_str:
                tag_str = tag_str.replace("'s ", ' ').replace("n't ", ' not ').replace("s'", '').replace("'re", '')
            tag_str = tag_str.strip()
            pattern = re.compile(r"[!\"#\$%&\'\(\)\*\+,-.\/:;<=>\?@\[\\\]\^_\`\{\~\|]+")
            tag_str = re.sub(pattern, ' ', tag_str)
            tag_str = tag_str.strip()
            tag_str = ' '.join(tag_str.split())
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(tag_str)
            tag_str_list = []
            for token in tokens:
                lemma_token = lemmatizer.lemmatize(token)
                if lemma_token not in stop_words:
                    tag_str_list.append(lemma_token)
            tag_str = " ".join(tag_str_list)

            if tag_str.isspace() and tag_info[2] == '':
                continue

            cleaned_tags.append([tag_info[0], tag_info[1], tag_str])

            ex_standard = True
            for standard_tag in processed_standard_tag_set:
                can_include = False
                if ' ' in standard_tag or len(standard_tag) > 3:
                    if standard_tag in tag_str:
                        can_include = True
                else:
                    if standard_tag in tag_str.split(' '):
                        can_include = True

                if not can_include:
                    if ' ' in tag_str or len(tag_str) > 4:
                        if tag_str in standard_tag:
                            can_include = True
                    else:
                        if tag_str in standard_tag.split(' '):
                            can_include = True

                if can_include:
                    cleaned_tags_standard.append([tag_info[0], tag_info[1], standard_tag])
                    auxillary_tag_in_standard_dict[tag_str].add(standard_tag)
                    ex_standard = False

            if ex_standard:
                auxillary_tag_ex_standard_dict[tag_str] += 1
                
    return processed_standard_tag_set, cleaned_tags, cleaned_tags_standard, \
        auxillary_tag_ex_standard_dict, auxillary_tag_in_standard_dict


def data_stat_main(dataset_name):
    ratings_data = load_data(parent_path + '/data/{}/{}-raw/ratings.csv'.format(dataset_name, dataset_name))
    user_ratings_data = set()
    movie_ratings_data = set()
    for rating in ratings_data:
        user_ratings_data.add(int(rating[0]))
        movie_ratings_data.add(int(rating[1]))

    movies_data = load_movies_data(dataset_name)
    movie_movies_data = set()
    composed_genre_data = set()
    for movie in movies_data:
        movie_movies_data.add(int(movie[0]))
        composed_genre_data.add(movie[1])

    tags_data = load_tags_data(dataset_name)
    user_tags_data = set()
    movie_tags_data = set()
    tag_tags_data = set()
    for tag in tags_data:
        user_tags_data.add(int(tag[0]))
        movie_tags_data.add(int(tag[1]))
        tag_tags_data.add(tag[2])
    tag_tags_data_new = set()
    for i in tag_tags_data:
        tag_tags_data_new.add(i.lower())

    scores_data = load_data(parent_path + '/data/{}/{}-raw/genome-scores.csv'.format(dataset_name, dataset_name))
    movie_scores_data = set()
    tag_scores_data = set()
    for score in scores_data:
        movie_scores_data.add(int(score[0]))
        tag_scores_data.add(int(score[1]))

    genome_tags_data = load_data(parent_path + '/data/{}/{}-raw/genome-tags.csv'.format(dataset_name, dataset_name))
    genome_tag_set = set()
    for genome_tag in genome_tags_data:
        genome_tag_set.add(genome_tag[1])

    data_stat(tag_tags_data_new, tag_tags_data, user_ratings_data, movie_ratings_data,
              user_tags_data, movie_tags_data, movie_movies_data, movie_scores_data,
              tag_scores_data, genome_tag_set, tags_data, movies_data, composed_genre_data)


def data_process_main(dataset_name):
    # find moive wihtout genres
    movies_data = load_movies_data(dataset_name)
    movie_wo_genre_set = set()
    for movie in movies_data:
        if '(no genres listed)' in movie[1].lower():
            movie_wo_genre_set.add(int(movie[0]))

    # 1.clean interaction
    ratings_data = load_data(parent_path + '/data/{}/{}-raw/ratings.csv'.format(dataset_name, dataset_name))
    valid_movieId_set, valid_userId_set, filtered_interaction_data\
        = interaction_process(ratings_data, movie_wo_genre_set)
    
    # create mapping table
    movieId_to_mid_dict = {}
    userId_to_uid_dict = {}
    for movieId in valid_movieId_set:
        assert movieId not in movieId_to_mid_dict
        movieId_to_mid_dict[movieId] = len(movieId_to_mid_dict)
    for userId in valid_userId_set:
        assert userId not in userId_to_uid_dict
        userId_to_uid_dict[userId] = len(userId_to_uid_dict)
    
    with open(parent_path+'/data/{}/pro_sg/user_item_interaction.csv'.format(dataset_name), 'w') as file:
        file.write('uid,pid\n')
        for [user_id, movie_id] in filtered_interaction_data:
            file.write(str(userId_to_uid_dict[user_id]) + ',' + str(movieId_to_mid_dict[movie_id]) + '\n')

    # 2.clean tag text
    processed_standard_tag_set, cleaned_tags_raw, cleaned_tags_standard, \
        auxillary_tag_ex_standard_dict,auxillary_tag_in_standard_dict = tag_process(valid_movieId_set, dataset_name)
    
    # save data
    with open(parent_path + '/data/{}/pro_sg/standard_tag_table.csv'.format(dataset_name), 'w') as file:
        processed_standard_tag_list = list(processed_standard_tag_set)
        processed_standard_tag_list.sort()
        file.write('tid,tag\n')
        for idx, tag in enumerate(processed_standard_tag_list):
            file.write(str(idx) + ',' + tag + '\n')

    with open(parent_path + '/data/{}/pro_sg/item_raw_tag.csv'.format(dataset_name), 'w') as file:
        file.write('pid,tag\n')
        for data in cleaned_tags_raw:
            mid = movieId_to_mid_dict[int(data[1])]
            file.write(str(mid) + ',' + data[2] + '\n')

    covered_movie_set = set()
    with open(parent_path + '/data/{}/pro_sg/item_standard_tag.csv'.format(dataset_name), 'w') as file:
        file.write('pid,tag\n')
        for data in cleaned_tags_standard:
            mid = movieId_to_mid_dict[int(data[1])]
            file.write(str(mid) + ',' + data[2] + '\n')
            covered_movie_set.add(int(data[1]))
    print("covered_movie_set: {}.".format(len(covered_movie_set)))  # covered_movie_set: 36494.
    
    auxiliary_tag_ex_standard_list = sorted(auxillary_tag_ex_standard_dict.items(), key=lambda k_v: k_v[1])
    with open(parent_path + '/data/{}/{}-raw/aux_tag_ex_standard.txt'.format(dataset_name, dataset_name), 'w') as file:
        for (k, v) in auxiliary_tag_ex_standard_list:
            file.write(str(k) + ',' + str(v) + '\n')

    with open(parent_path + '/data/{}/{}-raw/aux_tag_in_standard.txt'.format(dataset_name, dataset_name), 'w') as file:
        for (k, v_list) in auxillary_tag_in_standard_dict.items():
            file.write(str(k) + ' | ')
            for v in v_list:
                file.write(str(v) + ',')
            file.write('\n')
            
    # current user number:
    print("user number after filter: {}.".format(len(valid_userId_set)))
    # current movie number:
    print("movie number after filter: {}.".format(len(valid_movieId_set)))
    
    with open(parent_path + '/data/{}/pro_sg/userId_to_uid_dict.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(userId_to_uid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(parent_path + '/data/{}/pro_sg/itemId_to_pid_dict.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(movieId_to_mid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_tag_table(dataset_name):
    # For movie with tag, create one hot tag vector
    # 1.load tag dict 2.for each movie, create tag vector
    tag_table_data = load_data(parent_path+'/data/{}/pro_sg/standard_tag_table.csv'.format(dataset_name))
    tag_tid_dict = {}
    tid_tag_dict = {}
    for data in tag_table_data:
        tag_tid_dict[data[1]] = int(data[0]) # Note: tid and tagId are different
        tid_tag_dict[int(data[0])] = data[1]
    return tag_tid_dict, tid_tag_dict


def movie_with_standard_tag_to_vec(tag_tid_dict, dataset_name):
    movie_standard_tag_data = load_data(parent_path+'/data/{}/pro_sg/item_standard_tag.csv'.format(dataset_name))
    mid_with_tag_set = set()
    for data in movie_standard_tag_data:
        mid_with_tag_set.add(int(data[0]))
    mid_to_midt_dict = {}
    for mid in mid_with_tag_set:
        mid_to_midt_dict[mid] = len(mid_to_midt_dict)
    assert len(mid_to_midt_dict) == len(mid_with_tag_set)

    midt_tid_dict = defaultdict(set)
    for data in movie_standard_tag_data: # mid,tag
        midt = mid_to_midt_dict[int(data[0])] # midt means mid-tagged
        assert data[1] in tag_tid_dict
        midt_tid_dict[midt].add(tag_tid_dict[data[1]])
    tagged_movie_num = len(midt_tid_dict)
    assert tagged_movie_num == len(mid_with_tag_set)

    tag_vec = np.zeros((tagged_movie_num, len(tag_tid_dict)))
    for k,v_set in tqdm(midt_tid_dict.items()):
        for v in v_set:
            tag_vec[k,v] = 1
    return tag_vec, midt_tid_dict, mid_to_midt_dict


def ml_create_tag_movie_similarity_matrix(tag_vec):
    # 3.for each movie calculate positive samples
    tag_tensor = torch.from_numpy(tag_vec).to('cuda')
    similarity_matrix = torch.mm(tag_tensor, tag_tensor.t()).to('cpu')
    # similarity_matrix = np.dot(tag_vec, tag_vec.T)
    # assert (similarity_matrix - similarity_matrix.t()).sum() == 0
    tag_distance_vec = -( 2*similarity_matrix
                          - tag_tensor.sum(dim=1).reshape(-1,1).to('cpu')
                          - tag_tensor.t().sum(dim=0).reshape(1,-1).to('cpu') )
    assert (tag_distance_vec >= 0).all()
    tag_distance_vec_full = tag_distance_vec.numpy()
    del tag_tensor
    del similarity_matrix
    torch.cuda.empty_cache()
    return tag_distance_vec_full.astype('int')


def create_genre_table(dataset_name):
    # 1.create genre table
    genre_table = load_data(parent_path + '/data/{}/pro_sg/genre_table.csv'.format(dataset_name))
    genre_dict = {}
    for data in genre_table:
        genre_dict[data[1].lower()] = int(data[0]) # genre txt to idx
    return genre_dict


def create_movie_genre_table(genre_dict, movieId_to_mid_dict, mid_to_midt_dict, dataset_name):
    # 3.create movie-genre table
    tagged_movie_genre_table = np.zeros((len(mid_to_midt_dict), len(genre_dict)))
    movies_data = load_movies_data(dataset_name)
    for data in movies_data:
        movieId = int(data[0])
        if movieId not in movieId_to_mid_dict:
            continue
        mid = movieId_to_mid_dict[movieId]
        if mid in mid_to_midt_dict:
            midt = mid_to_midt_dict[mid]
            for genre_str in data[1].split('|'):
                try:
                    assert genre_str in genre_dict
                except:
                    print(data)
                    continue
                genre_str_idx = genre_dict[genre_str]
                tagged_movie_genre_table[midt, genre_str_idx] = 1
    return tagged_movie_genre_table


def check_tags_with_word2vec(tag_str_set, w2v_path):
    lines = open(parent_path + w2v_path).readlines()
    words = set()
    for line in lines:
        tokens = line.strip().split()
        words.add(tokens[0])
    
    total_tag_num = len(tag_str_set)
    direct_have = 0
    indirect_have = 0
    not_have = set()
    for tag in tag_str_set:
        if tag in words:
            direct_have += 1
        elif ' ' in tag:
            tag_word_set = set(tag.split())
            if tag_word_set.issubset(words):
                indirect_have += 1
        else:
            not_have.add(tag)
    print("total_tag_num {}".format(total_tag_num))
    print("direct_have {}".format(direct_have))
    print("indirect_have {}".format(indirect_have))
    print("not_have {}".format(len(not_have)))
    print(not_have)


def load_pickle(dataset_name, var):
    data = None
    with open(parent_path + '/data/' + dataset_name + '/pro_sg/' + var + '.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def debug_tag(idx, test_pairs, tid_tag_dict, tag_vec):
    pair = test_pairs[idx].reshape((-1,2))
    l = pair[:,0]
    l_tag_list = []
    r = pair[:,1]
    r_tag_list = []
    l_tag_vec = tag_vec[l]
    r_tag_vec = tag_vec[r]
    for i in list(l_tag_vec.nonzero()[1]):
        assert i in tid_tag_dict
        l_tag_list.append(tid_tag_dict[i])
    for i in list(r_tag_vec.nonzero()[1]):
        assert i in tid_tag_dict
        r_tag_list.append(tid_tag_dict[i])
    print(l_tag_list)
    print(r_tag_list)
    
    modification = add_tag_infomation(pair, tag_vec)
    modification_add = modification > 0
    modification_add = modification_add.nonzero()[1]
    modification_add_list = []
    modification_remove = modification < 0
    modification_remove = modification_remove.nonzero()[1]
    modification_remove_list = []
    for i in modification_add:
        assert i in tid_tag_dict
        modification_add_list.append(tid_tag_dict[i])
    for i in modification_remove:
        assert i in tid_tag_dict
        modification_remove_list.append(tid_tag_dict[i])
    print(modification_add_list)
    print(modification_remove_list)
    return l_tag_list, r_tag_list, modification_add_list, modification_remove_list


def load_genome_scores(dataset_name):
    genome_scores_path = parent_path + '/data/{}/{}-raw/genome-scores.csv'.format(dataset_name, dataset_name)
    raw_genome_scores_data = pd.read_csv(genome_scores_path, header=0)
    scores_data = raw_genome_scores_data.values.tolist()
    genome_movieId_set = set()
    genome_tagId_set = set()
    movieId_to_tagId_score_dict = defaultdict(list)
    for data in scores_data:
        movieId = int(data[0])
        tagId = int(data[1])
        score = float(data[2])
        movieId_to_tagId_score_dict[movieId].append([tagId, score])
        genome_movieId_set.add(movieId)
        genome_tagId_set.add(tagId)
    return genome_movieId_set, movieId_to_tagId_score_dict


def gradient_movie_retrival(dataset_name):
    # 1.load genome-scores
    genome_movieId_set, movieId_to_tagId_score_dict = load_genome_scores(dataset_name)
    movieId_to_mid_dict = load_pickle(dataset_name, 'itemId_to_pid_dict')
    used_movieId_set = set(movieId_to_mid_dict.keys())
    used_mid_set = set(movieId_to_mid_dict.values())
    common_movieId_set = genome_movieId_set.intersection(used_movieId_set)
    print("Common movie num:{}".format(len(common_movieId_set)))    # 13771
    print("genome_movieId_set: {}".format(len(genome_movieId_set))) # 13816
    print("used_movieId_set: {}".format(len(used_movieId_set)))     # 38715
    
    mid_with_score_list = []
    for movieId in list(common_movieId_set):
        mid_with_score_list.append(movieId_to_mid_dict[movieId])
    mid_with_score_array = np.array(mid_with_score_list)
    
    with open(parent_path + '/data/{}/pro_sg/pid_with_score_array.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(mid_with_score_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save pid_with_score_array")
        print(max(mid_with_score_array)) # 38715
        
    standard_tag_to_tid_dict = {}
    tag_table_data = load_data(parent_path + '/data/{}/pro_sg/standard_tag_table.csv'.format(dataset_name))
    for data in tag_table_data:
        standard_tag_to_tid_dict[data[1]] = int(data[0])
    
    tagId_to_tid = {}
    genome_tags_data = load_data(parent_path + '/data/{}/{}-raw/genome-tags.csv'.format(dataset_name, dataset_name))
    for data in genome_tags_data:
        tag_str = data[1]
        processed_tag_str = process_genome_tag_str(tag_str)
        tid = standard_tag_to_tid_dict[processed_tag_str]
        tagId_to_tid[int(data[0])] = tid # multiple tagId can be mapped to same one tid
    
    tid_to_tagId_dict = defaultdict(list)
    for k,v in tagId_to_tid.items():
        tid_to_tagId_dict[v].append(k)

    num_tag = len(tid_to_tagId_dict)
    mid_to_tid_score = -np.ones((max(used_mid_set)+1, num_tag))
    for k,v_list in movieId_to_tagId_score_dict.items():
        if k in common_movieId_set:
            assert k in movieId_to_mid_dict
            mid = movieId_to_mid_dict[k]
            tid_score_array = np.zeros((num_tag,))
            score_dict = defaultdict(list)
            for [tagId, score] in v_list:
                tid = tagId_to_tid[tagId]
                score_dict[tid].append(score)
            for tid,score_list in score_dict.items():
                tid_score_array[tid] = sum(score_list)/len(score_list)
            mid_to_tid_score[mid] = tid_score_array.copy()
    
    with open(parent_path + '/data/{}/pro_sg/pid_to_tid_score.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(mid_to_tid_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save pid_to_tid_score")
        
        
def create_gradient_test_data(dataset_name): # create start mid(those mid need in mid-midt) and relevant modification
    # 1. used common movie between tagged movie and scored movie
    # 2. create a midt to genre lookup table
    # 3. create a genre to tag frequency table
    # 4. sample some test movie, for each movie sample a modification with certain prob(had tag id need to be masked out and convert the rest to prob)
    genome_movieId_set, movieId_to_tagId_score_dict = load_genome_scores(dataset_name)
    movieId_genre_data = load_movies_data(dataset_name)
    movieId_to_mid_dict = load_pickle(dataset_name, 'itemId_to_pid_dict')
    mid_to_midt_dict = load_pickle(dataset_name, 'pid_to_pidt_dict')
    used_movieId_set = set(movieId_to_mid_dict.keys())
    used_mid_set = set(movieId_to_mid_dict.values())
    used_and_genome_common_set_movieId = genome_movieId_set.intersection(used_movieId_set)
    tagged_mid_set = set(mid_to_midt_dict.keys())
    assert len(tagged_mid_set) < len(used_mid_set)
    
    tagged_and_genome_common_set_mid = set()
    for movieId in used_and_genome_common_set_movieId:
        mid = movieId_to_mid_dict[movieId]
        if mid in tagged_mid_set:
            tagged_and_genome_common_set_mid.add(mid)
    
    # create movieId_to_midt_dict, mid_to_movieId_dict
    mid_to_movieId_dict = {}
    for k,v in movieId_to_mid_dict.items():
        mid_to_movieId_dict[v] = k
    movieId_to_midt_dict = {}
    for mid,midt in mid_to_midt_dict.items():
        movieId = mid_to_movieId_dict[mid]
        movieId_to_midt_dict[movieId] = midt
    
    # genre over tag count
    tag_vec = load_pickle(dataset_name, 'tag_vec')
    genre_to_tag_freq_dict = defaultdict(partial(np.ndarray, 0))
    genre_to_wo_tag_freq_dict = defaultdict(partial(np.ndarray, 0))
    midt_to_genre_dict = defaultdict(set)
    for [movieId, genre_str] in movieId_genre_data:
        if int(movieId) not in movieId_to_midt_dict:
            continue
        genres = genre_str.split('|')
        midt = movieId_to_midt_dict[int(movieId)]
        midt_tag_array = tag_vec[midt]

        for genre in genres:
            midt_to_genre_dict[midt].add(genre)
            if genre_to_tag_freq_dict[genre].shape[0] == 0:
                genre_to_tag_freq_dict[genre] = midt_tag_array
                genre_to_wo_tag_freq_dict[genre] = 1-midt_tag_array
            else:
                genre_to_tag_freq_dict[genre] += midt_tag_array
                genre_to_wo_tag_freq_dict[genre] += 1 - midt_tag_array

    # sample test data
    add_modification_pair, remove_modification_pair = [], []
    for mid in tagged_and_genome_common_set_mid:
        midt = mid_to_midt_dict[mid]
        sampled_genre = random.sample(midt_to_genre_dict[midt], 1)[0]
        assert sampled_genre in genre_to_tag_freq_dict
        freq_vec = genre_to_tag_freq_dict[sampled_genre].copy()
        freq_vec[tag_vec[midt] != 0] = 0
        freq_wo_vec = genre_to_wo_tag_freq_dict[sampled_genre].copy()
        freq_wo_vec[tag_vec[midt] == 0] = 0
        if freq_vec.sum() != 0:
            modification_idx = np.random.choice(freq_vec.shape[0], 1, p=freq_vec/freq_vec.sum())[0]
            add_modification_pair.append([mid, modification_idx])
        if freq_wo_vec.sum() != 0:
            remove_modification_idx = np.random.choice(freq_wo_vec.shape[0], 1, p=freq_wo_vec/freq_wo_vec.sum())[0]
            remove_modification_pair.append([mid, remove_modification_idx])

    with open(parent_path + '/data/{}/pro_sg/add_modification_pair.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(add_modification_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save add_modification_pair")

    with open(parent_path + '/data/{}/pro_sg/remove_modification_pair.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(remove_modification_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save remove_modification_pair")


def load_interaction_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['mid']  # user 160775, item = 38715
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data


def create_used_word(word_dict, word_set, tag_tid_dict):
    tid_to_full_word_idx = defaultdict(list)
    out_of_dict_list = []
    out_of_dict_w2v = {}
    for tag, tid in tag_tid_dict.items():
        if tag in word_dict:
            tid_to_full_word_idx[tid].append(word_dict[tag])
        elif ' ' in tag:
            tag_word_list = tag.split()
            tag_word_set = set(tag_word_list)
            if len(tag_word_set - word_set) < len(tag_word_set):
                for w in tag_word_list:
                    if w in word_set:
                        tid_to_full_word_idx[tid].append(word_dict[w])
            else:
                tid_to_full_word_idx[tid].append(len(word_dict) + len(out_of_dict_list))
                out_of_dict_list.append(tag)
                out_of_dict_w2v[tag] = np.random.rand(300,)
        else:
            tid_to_full_word_idx[tid].append(len(word_dict) + len(out_of_dict_list))
            out_of_dict_list.append(tag)
            out_of_dict_w2v[tag] = np.random.rand(300,)
            
    word_id_to_used_word_id = {}
    for tid, word_id_list in tid_to_full_word_idx.items():
        for w in word_id_list:
            if w not in word_id_to_used_word_id:
                word_id_to_used_word_id[w] = len(word_id_to_used_word_id)

    w2v_slice_array = np.zeros(len(word_id_to_used_word_id))
    for k,v in word_id_to_used_word_id.items():
        w2v_slice_array[v] = k

    num_tag = len(tid_to_full_word_idx)
    num_used_words = len(word_id_to_used_word_id)
    tid_over_used_word_matrix = np.zeros((num_tag, num_used_words))
    for k,w_list in tid_to_full_word_idx.items():
        for w in w_list:
            assert w in word_id_to_used_word_id
            tid_over_used_word_matrix[k,word_id_to_used_word_id[w]] = 1
    assert (tid_over_used_word_matrix.sum(axis=1) > 0).all()
    return tid_over_used_word_matrix, w2v_slice_array, out_of_dict_list, out_of_dict_w2v


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def numerize(tp, profile2id, show2id):
    uid = tp['uid'].apply(lambda x: profile2id[x])
    sid = tp['mid'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'mid': sid}, columns=['uid', 'mid'])



def parsers_parser():
    parser = argparse.ArgumentParser()
    arser.add_argument('--dataset_name', type=str, default='ml-20m', help='[ml-25m, ml-20m]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parsers_parser()
    dataset_name = args.dataset_name

    data_process_main(dataset_name)

    tag_tid_dict, tid_tag_dict = create_tag_table(dataset_name)
    tag_vec, midt_tid_dict, mid_to_midt_dict = movie_with_standard_tag_to_vec(tag_tid_dict, dataset_name)
    with open(parent_path + '/data/{}/pro_sg/tag_vec.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(tag_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(parent_path + '/data/{}/pro_sg/pidt_tid_dict.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(midt_tid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(parent_path + '/data/{}/pro_sg/pid_to_pidt_dict.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(mid_to_midt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ########################################################################

    # step3 create transductive test setting
    genre_dict = create_genre_table(dataset_name)
    tag_vec = load_pickle(dataset_name, 'tag_vec')
    with open(parent_path + '/data/{}/pro_sg/itemId_to_pid_dict.pkl'.format(dataset_name), 'rb') as handle:
        movieId_to_mid_dict = pickle.load(handle)
    mid_to_midt_dict = load_pickle(dataset_name, 'pid_to_pidt_dict')

    tagged_movie_genre_table = create_movie_genre_table(genre_dict, movieId_to_mid_dict, mid_to_midt_dict, dataset_name)
    tag_distance_vec_full = ml_create_tag_movie_similarity_matrix(tag_vec)
    positive_array = create_positive_pair(np.triu(tag_distance_vec_full), level=1) # only use distance within 3 as positive

    filtered_positive_array = filter_pair_by_class(positive_array, tagged_movie_genre_table)
    positive_num = filtered_positive_array.shape[0]
    print("num of pair for dist=1: {}".format(positive_num))
    test_idx = random.sample(range(0, positive_num), min(int(0.2 * positive_num), 5000))
    filtered_test_positive_array = filtered_positive_array[test_idx]
    print("test pair num: {}".format(filtered_test_positive_array.shape[0]))

    movie_set = set()
    movie_set.update(list(filtered_test_positive_array[:, 0]))
    movie_set.update(list(filtered_test_positive_array[:, 1]))
    print(len(movie_set))
    print(filtered_test_positive_array.shape)

    with open(parent_path + '/data/{}/pro_sg/test_pair.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(filtered_test_positive_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #### step4 prepare w2v
    w2v_path = parent_path + '/data/glove.6B.300d.txt'
    new_w2v_save_path = parent_path + '/data/new_glove.6B.300d.txt'
    lines = open(w2v_path).readlines()
    data = []
    word_dict = {}
    with open(new_w2v_save_path, 'w') as file:
        for idx, line in enumerate(lines):
            tokens = line.strip('\n')
            tokens_list = tokens.split(' ')
            word = tokens_list[0]
            vec_nums = tokens_list[1:]
            temp_ = [float(i) for i in vec_nums]
            assert len(temp_) == 300
            word_vec_info_str = " ".join([str(i) for i in temp_])
            file.write(word + '\t' + word_vec_info_str + '\n')

    tag_tid_dict, tid_tag_dict = create_tag_table(dataset_name)
    lines = open(parent_path + '/data/new_glove.6B.300d.txt').readlines()
    word_dict = {}

    for idx, line in enumerate(lines):
        tokens = line.strip('\n')
        word, _ = tokens.split('\t')
        if idx % 10000 == 0:
            print(word) # just for test
        word_dict[word] = idx
    word_set = set(word_dict.keys())

    tid_over_used_word_matrix, w2v_slice_array, out_of_dict_list, out_of_dict_w2v \
        = create_used_word(word_dict, word_set, tag_tid_dict)

    with open(parent_path + '/data/{}/pro_sg/tid_over_used_word_matrix.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(tid_over_used_word_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(parent_path + '/data/{}/pro_sg/w2v_slice_array.pkl'.format(dataset_name), 'wb') as handle:
        pickle.dump(w2v_slice_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(parent_path + '/data/new_glove.6B.300d.txt', 'a') as file:
        for word in out_of_dict_list:
            word_vec = list(np.round(out_of_dict_w2v[word], 6))
            word_vec_info = word + '\t' + " ".join([str(i) for i in word_vec])
            file.write(word_vec_info+'\n')

    # step5: process data for trainable gradient eval
    gradient_movie_retrival(dataset_name)
    create_gradient_test_data(dataset_name)