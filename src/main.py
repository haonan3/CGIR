import argparse
import os
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch import optim
from tqdm import tqdm

from src.dataloader import new_load_data, load_modification_score_data
import numpy as np
import torch.utils.data as tdata
from src.evaluation import evaluation_IR, independence_level, save_vis, gradient_retrival
from src.model_IR import gradientIR

from src.utlis import get_noise_features, sp_to_tensor, create_modification_matrix
from torch.utils.tensorboard import SummaryWriter

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', default=False, help='Disables CUDA training.')
    # disentangled item embedding
    parser.add_argument('--data', type=str, default='ml-25m', help='[ml-25m, ml-20m]')
    parser.add_argument('--seed', type=int, default=14, help='Random seed. Ignored if < 0.')
    parser.add_argument('--keep', type=float, default=0.5,  help='Keep probability for dropout, in (0,1].')
    parser.add_argument('--beta', type=float, default=0.1, help='Strength of disentanglement, in (0,oo).')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature of sigmoid/softmax, in (0,oo).')
    parser.add_argument('--std', type=float, default=0.075, help='Standard deviation of the Gaussian prior.')
    parser.add_argument('--dfac', type=int, default=1024, help='Dimension of each facet.')
    parser.add_argument('--disentanglement_batch_size', type=int, default=512, help='Training batch size.')
    # sparse word embedding
    parser.add_argument('--w_hdim', type=int, default=1024, help='resultant embedding size')
    parser.add_argument('--denoising', default=True, help='noise amount for denoising auto-encoder')
    parser.add_argument('--noise_level', type=float, default=0.2, help='noise amount for denoising auto-encoder')
    parser.add_argument('--sparsity', type=float, default=0.85, help='sparsity')
    parser.add_argument('--w2v_input_path',  default="/data/new_glove.6B.300d.txt", help='input src')
    parser.add_argument('--w2v_output_path', default="/data/new_glove.6B.300d.txt.spine", help='output')
    parser.add_argument('--word_batch_size', type=int, default=512, help='batch size')
    # contrastive loss
    parser.add_argument('--cl_batch_size', type=int, default=512, help='Training batch size.')
    parser.add_argument('--negative_num', type=int, default=5, help='negative sample num.')
    # general args
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval.')
    parser.add_argument('--cl_margin', type=int, default=1, help='loss margin.')
    parser.add_argument('--sampled_test', type=int, default=0)
    parser.add_argument('--eval_on_all', type=int, default=1)
    parser.add_argument('--eval_batch', type=int, default=15)
    parser.add_argument('--model_name', type=str, default='gradientIR')
    # eval
    parser.add_argument('--gradient_test', type=int, default=0)
    parser.add_argument('--independence_test', type=int, default=0)
    parser.add_argument('--IR_test', type=int, default=1)
    # structure
    parser.add_argument('--modification_level', type=int, default=1)
    parser.add_argument('--norm_CL_vec', type=int, default=0, help='default is 0')
    parser.add_argument('--norm_CL_score', type=int, default=1, help='default is 1')
    parser.add_argument('--include_zero_dist', type=int, default=0)
    parser.add_argument('--sparse_factor', type=float, default=0.2)
    # for visualization
    parser.add_argument('--vis', type=int, default=1)
    parser.add_argument('--topN', type=int, default=20)
    parser.add_argument('--fix_item', type=int, default=0, help='fix embedding table or not for baseline')

    args = parser.parse_args()
    return args


def set_env(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        args.device = 'cuda'
        torch.cuda.manual_seed(args.seed)
    else:
        args.device = 'cpu'


def main(args):
    # load data
    print(args)
    t = time.time()
    user_interaction_data, test_pairs, positive_array, pid_to_pidt_dict, \
    pidt_to_pid_dict, word_embedding, tag_vec, category_table, w2v_slice_array, \
    tid_over_used_word_matrix, midt_to_mid_interaction, pid_to_itemId_array = new_load_data(args)
    if args.gradient_test:
        pid_to_tid_score, pid_with_score_array, add_modification_pair, \
        add_gradient_modification, remove_modification_pair, remove_gradient_modification, \
        pid_with_score_to_itemId_array, modification_to_str = load_modification_score_data(args, tag_vec, pid_to_itemId_array)
    num_tagged_product = len(pidt_to_pid_dict)
    print("In total, load dat use {}s.".format(time.time() - t))
    
    w2v_slice_array = w2v_slice_array.astype(int)
    args.w2v_dim = word_embedding.shape[1]
    num_items = user_interaction_data.shape[1]
    num_users = user_interaction_data.shape[0]
    print("num user: {}".format(num_users))
    print("num item: {}".format(num_items))
    
    # create data loader
    print("Creating DataLoader...")
    t = time.time()
    filtered_positive_array = torch.from_numpy(positive_array)
    filtered_positive_array_idx = np.array(list(range(filtered_positive_array.shape[0])))
    w2c_idx = np.array(list(range(word_embedding.shape[0])))
    interaction_idx = np.array(list(range(user_interaction_data.shape[0])))

    tuple_dataloader = tdata.DataLoader(torch.from_numpy(filtered_positive_array_idx), batch_size=args.cl_batch_size, shuffle=True)
    word_embedding_loader = tdata.DataLoader(torch.from_numpy(w2c_idx), batch_size=args.word_batch_size, shuffle=True)
    interaction_data_loader = tdata.DataLoader(torch.from_numpy(interaction_idx), batch_size=args.disentanglement_batch_size, shuffle=True)
    print("#Positive:{};  #Word:{};  #Interaction:{}.".format(filtered_positive_array_idx.shape[0], w2c_idx.shape[0], interaction_idx.shape[0]))
    print("DataLoader Finish used {}s".format(time.time() - t))
    
    # create model & optim
    print("Create Model & Optim...")
    t = time.time()
    tid_over_used_word_matrix = torch.FloatTensor(tid_over_used_word_matrix)
    word_embedding = torch.FloatTensor(word_embedding)
    if args.cuda:
        tid_over_used_word_matrix = tid_over_used_word_matrix.to(args.device)
        word_embedding = word_embedding.to(args.device)

    model = gradientIR(args, num_items, num_tagged_product, pidt_to_pid_dict, w2v_slice_array,
                    tid_over_used_word_matrix, num_users, user_interaction_data, interaction_data_loader)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.to(args.device)
    print("Model created used {}s".format(time.time() - t))
    
    # process test data
    t = time.time()
    test_tag_modification = create_modification_matrix(test_pairs, tag_vec, keep_one_per_row=False)
    test_pairs = torch.LongTensor(test_pairs)
    if args.cuda:
        test_tag_modification = test_tag_modification.to(args.device)
        test_pairs = test_pairs.to(args.device)
        if args.gradient_test:
            add_gradient_modification = torch.from_numpy(add_gradient_modification).float().to(args.device)
            remove_gradient_modification = torch.from_numpy(remove_gradient_modification).float().to(args.device)
    print("process test data: {}s".format(time.time() - t))
    
    # process training data
    t = time.time()
    tag_modification_all = create_modification_matrix(filtered_positive_array, tag_vec, keep_one_per_row=False)
    print(user_interaction_data.shape)
    print("process train data: {}s".format(time.time() - t))

    # create temporal variable
    user_num = user_interaction_data.shape[0]
    num_batches = int(np.ceil(float(user_num) / args.disentanglement_batch_size))
    total_anneal_steps = 5 * num_batches
    update_count = 0.0
    batch_loss = []
    batch_disentangle_loss = []
    batch_sparse_loss = []
    batch_cl_loss = []
    
    model_name_args = "{}_{}_modif:{}_norm_vec:{}_norm_score:{}_sparse_factor:{}_fixItem:{}_beta:{}".\
        format(args.data, args.model_name, args.modification_level, args.norm_CL_vec,
               args.norm_CL_score, args.sparse_factor, args.fix_item, args.beta)
    writer = SummaryWriter(comment=model_name_args)

    max_gradient_score = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        # train
        for batch_tuple_t_idx, batch_word_embedding_idx, batch_interaction_idx in zip(tuple_dataloader, word_embedding_loader, interaction_data_loader):
            batch_interaction = user_interaction_data[batch_interaction_idx]
            batch_interaction = sp_to_tensor(batch_interaction)
            batch_word_embedding = word_embedding[batch_word_embedding_idx]
            batch_tuple_t = filtered_positive_array[batch_tuple_t_idx]
            batch_tag_modification = tag_modification_all[batch_tuple_t_idx]
            assert batch_tag_modification.shape[0] == batch_tuple_t.shape[0]

            if args.cuda:
                batch_interaction = batch_interaction.to(args.device)
                batch_tag_modification = batch_tag_modification.float().to(args.device)

            anneal = min(args.beta, 1. * update_count / total_anneal_steps) if total_anneal_steps > 0 else args.beta
            batch_word_embedding_y = batch_word_embedding.clone()

            if args.denoising:
                noise = get_noise_features(batch_word_embedding_y.shape[0], batch_word_embedding_y.shape[1], args.noise_level)
                batch_word_embedding_x = batch_word_embedding + torch.FloatTensor(noise).to(args.device)
            else:
                batch_word_embedding_x = batch_word_embedding
            
            optimizer.zero_grad()
            logits, kl, w_hidden, w_out, pred_score, self_score \
                = model.forward(batch_interaction, batch_word_embedding_x, batch_tuple_t,
                                batch_tag_modification, word_embedding, keep_prob=args.keep, is_training=1)
            loss, disentangled_vae_loss, sparse_vae_loss, cl_loss \
                = model.loss_function(batch_interaction, logits, kl, w_out, w_hidden, batch_word_embedding_y, pred_score,
                                      self_score, anneal, vae_factor=args.beta, sparse_factor=args.sparse_factor)
            loss.backward()

            batch_loss.append(loss.item())
            batch_disentangle_loss.append(disentangled_vae_loss.item())
            batch_sparse_loss.append(sparse_vae_loss.item())
            batch_cl_loss.append(cl_loss.item())
            optimizer.step()
            update_count += 1
        
        avg_batch_loss = sum(batch_loss)/len(batch_loss)
        print("Train Loss@{}: {}.".format(epoch, avg_batch_loss))
        writer.add_scalar('train/Loss', avg_batch_loss, epoch)
        avg_batch_disentangle_loss = sum(batch_disentangle_loss) / len(batch_disentangle_loss)
        writer.add_scalar('train/Disentangle_Loss', avg_batch_disentangle_loss, epoch)
        avg_batch_sparse_loss = sum(batch_sparse_loss) / len(batch_sparse_loss)
        writer.add_scalar('train/Sparse_Loss', avg_batch_sparse_loss, epoch)
        avg_batch_CL_loss = sum(batch_cl_loss) / len(batch_cl_loss)
        writer.add_scalar('train/CL_Loss', avg_batch_CL_loss, epoch)

        # test
        if epoch % args.test_interval == 0:
            model.eval()
            with torch.no_grad():

                if args.IR_test > 0:
                    t = time.time()
                    hits_list = [20, 50]
                    test_mrr, test_avg_count_list = evaluation_IR(model, test_pairs, test_tag_modification,
                                word_embedding, eval_on_all=args.eval_on_all, sampled_test=args.sampled_test,
                            hits=hits_list, eval_batch=args.eval_batch, midt_to_mid_interaction=midt_to_mid_interaction)
                    writer.add_scalar('IR_test/MRR', test_mrr, epoch)
                    for idx, hit in enumerate(hits_list):
                        writer.add_scalar('IR_test/Hit{}'.format(hit), test_avg_count_list[idx], epoch)
                    print("IR_test time: {}.".format(time.time() - t))

                if args.independence_test > 0:
                    t = time.time()
                    independence = independence_level(model.get_all_item_embed())
                    print("independence: {}".format(independence))
                    writer.add_scalar('test/Independence_Level', independence, epoch)
                    print("independence_test time: {}.".format(time.time() - t))

                if args.gradient_test > 0:
                    t = time.time()
                    final_increase_score, increase_vis_result, irr_increase_score, rel_increase_score = \
                        gradient_retrival(model, word_embedding, pid_to_pidt_dict, pid_to_tid_score, pid_with_score_array, tag_vec, 'increase',
                    add_modification_pair[:, 0], add_gradient_modification, add_modification_pair[:, 1], add_modification_pair,
                            args.eval_batch, topN=args.topN)
                    final_decrease_score, decrease_vis_result, irr_decrease_score, rel_decrease_score = \
                        gradient_retrival(model, word_embedding, pid_to_pidt_dict, pid_to_tid_score, pid_with_score_array, tag_vec, 'decrease',
                    remove_modification_pair[:, 0], remove_gradient_modification, remove_modification_pair[:, 1], remove_modification_pair,
                            args.eval_batch, topN=args.topN)

                    num_add_test = add_modification_pair.shape[0]
                    num_remove_test = remove_modification_pair.shape[0]
                    num_total_test = num_add_test + num_remove_test
                    MGS_C = (num_add_test/num_total_test)*rel_increase_score + (num_remove_test/num_total_test)*rel_decrease_score
                    MGS_R = (num_add_test/num_total_test)*irr_increase_score + (num_remove_test/num_total_test)*irr_decrease_score
                    gradient_score = (num_add_test/num_total_test)*final_increase_score + (num_remove_test/num_total_test)*final_decrease_score
                    print("overall_consistency_score: {}".format(MGS_C))
                    print("overall_restrictiveness_score: {}".format(MGS_R))
                    print("overall_gradient_score: {}".format(gradient_score))
                    writer.add_scalar('test/consistency_score', MGS_C, epoch)
                    writer.add_scalar('test/restrictiveness_score', MGS_R, epoch)
                    writer.add_scalar('test/gradient_score', gradient_score, epoch)

                    if args.vis and gradient_score > max_gradient_score \
                            and increase_vis_result is not None and decrease_vis_result is not None:
                        save_vis(model_name_args, pid_to_itemId_array, modification_to_str, increase_vis_result, decrease_vis_result)
                        max_gradient_score = gradient_score
                        print("save increase_vis!")
                    print("gradient_test time: {}.".format(time.time() - t))


if __name__ == '__main__':
    args = parsers_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    set_env(args)
    main(args)
