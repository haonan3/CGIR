import math

import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np
from src.utlis import negative_sampling, check_inf_nan_tensor


class gradientIR(nn.Module):
    def __init__(self, args, num_items, num_tagged_movie, midt_to_mid_dict, w2v_slice_array,
                 tid_over_used_word_matrix, num_users=None, sparse_user_interaction_data=None, interaction_data_loader=None):
        super(gradientIR, self).__init__()
        self.args = args

        # part1 disentangle_vae: two method, apply disentangled loss on 1.item embedding or 2.user embedding
        dfac = args.dfac
        self.n_items = num_items
        if self.args.model_name == 'CGIR-Sparse':
            assert self.args.sparse_factor == 0
        if self.args.model_name not in ['CGIR-VAE', 'CGIR-Both']:
            self.q0 = nn.Linear(num_items, dfac)
            self.q_mu = nn.Linear(dfac, dfac)
            self.q_logvar = nn.Linear(dfac, dfac)
            self.items = nn.Parameter(torch.Tensor(num_items, dfac))
        else:
            assert num_users is not None
            self.item_linear1 = nn.Linear(num_users, dfac)
            self.item_linear2 = nn.Linear(dfac, dfac)
            self.item_linear3 = nn.Linear(dfac, dfac)
            self.sparse_user_interaction_data = sparse_user_interaction_data
            self.interaction_data_loader = interaction_data_loader

        # part2 sparse_vae
        self.w2v_dim = args.w2v_dim
        self.w_hdim = args.w_hdim
        self.noise_level = args.noise_level
        self.getReconstructionLoss = nn.MSELoss()
        self.rho_star = 1.0 - args.sparsity
        self.used_word_idx = w2v_slice_array
        self.tag_over_word = tid_over_used_word_matrix
        # autoencoder
        self.linear1 = nn.Linear(self.args.w2v_dim, self.args.w_hdim)
        if self.args.sparse_factor > 0:
            self.linear2 = nn.Linear(self.args.w_hdim, self.args.w2v_dim)
        
        # part3 item geology
        self.num_tagged_movie = num_tagged_movie
        self.midt_to_mid_array = np.zeros(len(midt_to_mid_dict))
        assert max(set(midt_to_mid_dict.keys())) + 1 == len(midt_to_mid_dict)
        for k,v in midt_to_mid_dict.items():
            self.midt_to_mid_array[k] = int(v)
        self.cl_margin = self.args.cl_margin
        
        # reset params
        if self.args.model_name not in ['CGIR-VAE', 'CGIR-Both']:
            self.reset_params()
        
        
    def reset_params(self):
        nn.init.xavier_uniform_(self.items)
    
    
    def forward_interaction(self, interaction, keep_prob=1, is_training=0):
        items = F.normalize(self.items, dim=-1, p=2)
        h = F.normalize(interaction, dim=-1, p=2)
        h = torch.dropout(h, keep_prob, train=self.training)

        # q-network
        h = self.q0(h)
        h = torch.tanh(h)
        mu_q = self.q_mu(h)
        mu_q = F.normalize(mu_q, dim=-1, p=2)
        lnvarq_sub_lnvar0 = -self.q_logvar(h)
        check_inf_nan_tensor(lnvarq_sub_lnvar0)
        std0 = self.args.std
        std_q = torch.exp(0.5*lnvarq_sub_lnvar0) * std0
        check_inf_nan_tensor(std_q)
        # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
        kl = (0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.0)).sum(dim=1).mean()
        check_inf_nan_tensor(kl)
        epsilon = torch.randn(std_q.shape).to(self.args.device)
        z_k = mu_q + is_training * epsilon * std_q

        # p-network
        z_k = F.normalize(z_k, dim=-1, p=2)
        logits = torch.matmul(z_k, items.t()) / self.args.tau
        logits = torch.log_softmax(logits, dim=-1)

        logits = torch.log_softmax(logits, dim=-1)
        return logits, kl
    
    
    def forward_word(self, word_batch_x):
        # forward
        linear1_out = self.linear1(word_batch_x)
        w_hidden = linear1_out.clamp(min=0, max=1)  # capped relu
        if self.args.sparse_factor > 0:
            w_out = self.linear2(w_hidden)
        else:
            w_out = None
        return w_hidden, w_out


    def sp_to_dense_tensor(self, item_sp_array):
        item_array = item_sp_array.toarray()
        item_array = item_array.astype('float32')
        item_tensor = torch.from_numpy(item_array)
        if self.args.cuda:
            item_tensor = item_tensor.to('cuda')
        return item_tensor

    # the following two func is for the variant of CGIR: CGIR-VAE, CGIR-Both
    def get_batch_item_embedding(self, batch_idx):
        batch_item_tensor = self.sp_to_dense_tensor(self.sparse_user_interaction_data[batch_idx])
        batch_item_tensor = F.normalize(batch_item_tensor, dim=-1, p=2)
        hidden = self.item_linear1(batch_item_tensor)
        hidden = torch.relu(hidden)
        hidden = self.item_linear2(hidden)
        hidden = torch.relu(hidden)
        output = self.item_linear3(hidden)
        return output



    def get_all_item_embed(self):
        if self.args.model_name not in ['CGIR-VAE', 'CGIR-Both']:
            return self.items.detach()
        else:
            item_embedding = []
            for batch_idx in self.interaction_data_loader:
                batch_item_embedding = self.get_batch_item_embedding(batch_idx).detach()
                item_embedding.append(batch_item_embedding)
                torch.cuda.empty_cache()
            item_embedding = torch.cat(item_embedding, dim=0)
            return item_embedding


    def compose_forward(self, reshaped_a, modification):
        pred_b = modification + reshaped_a
        return pred_b


    def forward_triple(self, pos_triple_t, tag_modification, tag_embedding):
        if self.args.model_name not in ['CGIR-VAE', 'CGIR-Both']:
            a = self.items[self.midt_to_mid_array[pos_triple_t[:, 0]]]
            target_b = self.items[self.midt_to_mid_array[pos_triple_t[:, 1]]]
        else:
            a = self.get_batch_item_embedding(self.midt_to_mid_array[pos_triple_t[:, 0]])
            target_b = self.get_batch_item_embedding(self.midt_to_mid_array[pos_triple_t[:, 1]])

        num_tag = tag_embedding.shape[0]
        neg_tag_idx = np.random.choice(num_tag, 2*self.args.negative_num*a.shape[0])
        neg_modification_tag_embedding = tag_embedding[neg_tag_idx]
        pos_modification_tag_embedding = torch.mm(tag_modification, tag_embedding)
        modification = torch.cat([pos_modification_tag_embedding, neg_modification_tag_embedding], dim=0)
        modification = modification.view(-1, a.shape[0], modification.shape[-1])
        modification = modification.permute(1, 0, 2) # (100, 21, 1024)
        
        neg_item_idx = np.random.choice(self.n_items, 2*self.args.negative_num*a.shape[0])

        if self.args.model_name not in ['CGIR-VAE', 'CGIR-Both']:
            neg_item_embedding = self.items[neg_item_idx]
            pos_item_embdding = self.items[self.midt_to_mid_array[pos_triple_t[:, 0]]]
        else:
            neg_item_embedding = self.get_batch_item_embedding(neg_item_idx)
            pos_item_embdding = self.get_batch_item_embedding(self.midt_to_mid_array[pos_triple_t[:, 0]])

        items = torch.cat([pos_item_embdding, neg_item_embedding], dim=0)
        items = items.view(-1, a.shape[0], modification.shape[-1])
        items = items.permute(1, 0, 2) # (100, 21, 1024)
        
        if self.args.norm_CL_vec:
            a = F.normalize(a, dim=-1, p=2)
            
        pred_b = self.compose_forward(a.unsqueeze(1), modification)
        
        if self.args.norm_CL_score:
            pred_b = F.normalize(pred_b, dim=-1, p=2)
            target_b = F.normalize(target_b, dim=-1, p=2)

            pos_item_embdding = F.normalize(pos_item_embdding, dim=-1, p=2)
            items = F.normalize(items, dim=-1, p=2) # (100, 21, 1024)

        target_b = target_b.unsqueeze(1)
        pos_item_embdding = pos_item_embdding.unsqueeze(1)
        pred_score = (pred_b * target_b).contiguous().view(-1, a.shape[-1]).sum(dim=-1)
        self_pred_score = (items * pos_item_embdding).contiguous().view(-1, a.shape[-1]).sum(dim=-1)
        return pred_score, self_pred_score
        
    
    def forward(self, batch_interaction, batch_word_embedding_x, batch_triple_t, tag_modification, word_embedding, keep_prob=1, is_training=0):
        logits, kl = torch.tensor([0], dtype=torch.float).to(self.args.device), torch.tensor([0], dtype=torch.float).to(self.args.device)
        w_hidden, w_out = self.forward_word(batch_word_embedding_x)
        tag_embedding = self.get_tag_embedding(word_embedding)
        pred_score, self_score = self.forward_triple(batch_triple_t, tag_modification, tag_embedding)
        return logits, kl, w_hidden, w_out, pred_score, self_score

    def get_tag_embedding(self, word_embedding):
        used_word_embedding = word_embedding[self.used_word_idx]
        sparse_word_embedding = self.forward_word(used_word_embedding)[0]
        tag_embedding = torch.mm(self.tag_over_word, sparse_word_embedding)
        return tag_embedding
        
    
    def loss_function(self, interaction, logits, kl, w_out, w_hidden, batch_word_embedding_y, pred_score, self_score=None, anneal=1, vae_factor=1.0, cl_factor=1.0, sparse_factor=1.0):
        # disentanglement loss
        disentangle_recon_loss = (-logits * interaction).sum(dim=-1).mean()
        disentangled_vae_loss = disentangle_recon_loss + anneal * kl

        # contrastive loss
        scores = pred_score.view(-1, 1 + 2 * self.args.negative_num)
        cl_loss = torch.mean(F.relu(-(scores[:, 0].unsqueeze(1) - scores[:, 1:]) + self.cl_margin))

        if self_score is not None:
            self_score = self_score.view(-1, 1 + 2 * self.args.negative_num)
            cl_loss2 = torch.mean(F.relu(-(self_score[:, 0].unsqueeze(1) - self_score[:, 1:]) + self.cl_margin))
            cl_loss = cl_loss + cl_loss2

        if self.args.sparse_factor == 0:
            sparse_vae_loss = torch.tensor([0], dtype=torch.float).to(self.args.device)
        else:
            psl_loss = self._getPSLLoss(w_hidden, batch_word_embedding_y.shape[0])  # partial sparsity loss
            asl_loss = self._getASLLoss(w_hidden)  # average sparsity loss
            sparse_vae_loss = psl_loss + asl_loss

        loss = vae_factor * disentangled_vae_loss + sparse_factor * sparse_vae_loss + cl_factor * cl_loss

        return loss, disentangled_vae_loss, sparse_vae_loss, cl_loss
    
    
    def _getPSLLoss(self, h, batch_size):
        return torch.sum(h * (1 - h)) / (batch_size * self.w_hdim)


    def _getASLLoss(self, h):
        temp = torch.mean(h, dim=0) - self.rho_star
        temp = temp.clamp(min=0)
        return torch.sum(temp * temp) / self.w_hdim