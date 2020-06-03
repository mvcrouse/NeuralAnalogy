import signal, time, random, itertools, copy, math, sys
import pickle as pkl
# numpy imports
import numpy as np
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
# code imports

####
# Basic NN classes
####

class Emb(nn.Module):
    
    def __init__(self, emb_ct, emb_dim, device=torch.device('cpu'), 
                 sparse_grads=True):
        super().__init__()
        self.embedding = nn.Embedding(emb_ct + 1, emb_dim, 
                                      padding_idx=emb_ct, sparse=sparse_grads)
        self.embedding = self.embedding.to(device)

    def forward(self, x):
        return self.embedding(x)

class MLP(nn.Module):

    def __init__(self, inp_dim, out_dim, hid_dim=None, device=torch.device('cpu'),
                 mlp_act='elu', inner_act='elu', mlp_layers=2,
                 norm_type='batchnorm'):
        super().__init__()
        if hid_dim == None: hid_dim = out_dim

        act_dict = { 'relu' : nn.ReLU(), 'tanh' : nn.Tanh(), 
                     'sigmoid' : nn.Sigmoid(), 'elu' : nn.ELU() }
        if_norm_dicts = []
        for use_dim in [hid_dim, out_dim]:
            if_norm_dicts.append({ 'batchnorm' : nn.BatchNorm1d(use_dim), 
                                   'layer' : nn.LayerNorm(use_dim), 
                                   'ident' : nn.Identity(), None : nn.Identity() })

        f_act = act_dict[mlp_act]
        i_act = act_dict[inner_act]
        i_norm = if_norm_dicts[0][norm_type]
        f_norm = if_norm_dicts[1][norm_type]

        modules = []
        if mlp_layers == 1:
            modules.append(nn.Linear(inp_dim, out_dim))
        else:
            for l in range(mlp_layers - 1):
                i_dim = inp_dim if l == 0 else hid_dim
                modules.extend([nn.Linear(i_dim, hid_dim), i_norm, i_act])
            modules.append(nn.Linear(hid_dim, out_dim))
        if norm_type in ['batchnorm', 'layer']:
            if mlp_act != 'sigmoid': modules.append(f_norm)
        else:
            modules.append(f_norm)
        modules.append(f_act)
        self.ff = nn.Sequential(*modules).to(device)

    def forward(self, x):
        return self.ff(x)

class EmbProj(nn.Module):
    # simple embedding projection
    def __init__(self, concept_ct, concept_emb_dim, concept_state_dim=None,
                 device=torch.device('cpu'), sparse_grads=True, layer=False):
        super().__init__()
        if concept_state_dim == None: concept_state_dim = concept_emb_dim
        self.layer = layer
        if layer:
            self.emb_layer = Emb(concept_ct, concept_emb_dim, device=device,
                                 sparse_grads=sparse_grads)
            self.bn = nn.BatchNorm1d(concept_emb_dim).to(device)
            self.proj_layer = MLP(concept_emb_dim, concept_state_dim,
                                  mlp_layers=1, device=device)
        else:
            self.emb_layer = Emb(concept_ct, concept_state_dim, device=device,
                                 sparse_grads=sparse_grads)
            self.bn = nn.BatchNorm1d(concept_state_dim).to(device)
        self.act = nn.ELU()

    def forward(self, x):
        if self.layer:
            return self.proj_layer(self.act(self.bn(self.emb_layer(x))))
        return self.act(self.bn(self.emb_layer(x)))
        
###
# Skip connection
###

class SkipConnection(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

###
# Attention
###

class MultiHeadAttention(nn.Module):
    
    def __init__(self, query_dim, key_dim, hid_dim, num_heads=1,
                 device=torch.device('cpu')):
        super().__init__()
        assert hid_dim % num_heads == 0, 'Need even split for MHA...'
        self.query_dim = query_dim
        self.hid_dim = hid_dim
        self.inner_dim = int(hid_dim / num_heads)
        self.sqrt_dk = np.sqrt(self.inner_dim)
        self.device = device
        self.num_heads = num_heads

        heads = []
        for i in range(self.num_heads):
            head_lst = []
            W_q = nn.Linear(query_dim, self.inner_dim, bias=False)
            head_lst.append(W_q.to(device))
            W_k = nn.Linear(key_dim, self.inner_dim, bias=False)
            head_lst.append(W_k.to(device))
            W_v = nn.Linear(key_dim, self.inner_dim, bias=False)
            head_lst.append(W_v.to(device))
            heads.append(nn.ModuleList(head_lst))
        self.heads = nn.ModuleList(heads)

        W_o = nn.Linear(hid_dim, query_dim, bias=False)
        self.W_o = W_o.to(device)

    def forward(self, q, k, v, mask=None):
        concat_block = []
        for W_q, W_k, W_v in self.heads:
            q_block = W_q(q)
            k_block = W_k(k).transpose(1, 2)
            v_block = W_v(v)
            qk_scores = torch.bmm(q_block, k_block) / self.sqrt_dk
            if mask is not None:
                assert mask.size() == qk_scores.size()
                qk_scores = qk_scores.masked_fill(mask==True, float('-inf'))
            masked_scores = nn.Softmax(dim=2)(qk_scores)
            att = masked_scores.matmul(v_block)
            concat_block.append(att)
        concat_block = torch.cat(concat_block, 2)
        out_matr = self.W_o(concat_block)
        return out_matr

class SingleHeadAttention(nn.Module):
    
    def __init__(self, model_dim, hid_dim, device=torch.device('cpu')):
        super().__init__()
        self.model_dim = model_dim
        self.hid_dim = hid_dim
        self.sqrt_dk = np.sqrt(hid_dim)
        self.device = device

        W_q = nn.Linear(model_dim, hid_dim, bias=False)
        self.W_q = W_q.to(device)
        W_k = nn.Linear(model_dim, hid_dim, bias=False)
        self.W_k = W_k.to(device)

    def forward(self, q, k, mask=None):
        q_block = self.W_q(q)
        k_block = self.W_k(k).transpose(1, 2)
        qk_scores = q_block.matmul(k_block) / self.sqrt_dk
        if mask is not None:
            # if bias == True in linears, this will not work, this is a hack
            # until I figure out the issue with the actual boolean mask
            qk_scores = qk_scores.masked_fill(mask==True, float('-inf'))
        masked_scores = nn.Softmax(dim=2)(qk_scores)
        return masked_scores

###
# Encoder / Decoder Attention
###

class EncoderAttentionLayer(nn.Module):
    
    def __init__(self, query_dim, hid_dim, num_heads=1, device=torch.device('cpu')):
        super().__init__()
        self.self_mha = MultiHeadAttention(query_dim, query_dim, hid_dim, 
                                           num_heads=num_heads, device=device)
        #self.norm1 = nn.BatchNorm1d(query_dim).to(device)
        self.norm1 = nn.LayerNorm(query_dim).to(device)
        #self.norm1 = nn.Identity(query_dim).to(device)

        self.ff = MLP(query_dim, query_dim, query_dim * 2,
                      device=device, norm_type=None)
        #self.norm2 = nn.BatchNorm1d(query_dim).to(device)
        self.norm2 = nn.LayerNorm(query_dim).to(device)
        #self.norm2 = nn.Identity(query_dim).to(device)

    def forward(self, q, mask=None):
        # runs multi-headed attention then puts everything into one
        # matrix for normalization and whatnot
        ext_q_shp = (len(q) * len(q[0]), len(q[0][0]))
        compact_q_shp = (len(q), len(q[0]), len(q[0][0]))
        ext_q = q.contiguous().view(ext_q_shp)

        self_mha = self.self_mha(q, q, q, mask=mask).contiguous().view(ext_q_shp)

        skip1 = ext_q + self_mha
        n1 = self.norm1(skip1)
        
        # feedforward
        ff = self.ff(n1)
        skip2 = n1 + ff
        n2 = self.norm2(skip2)
        
        # rebatching results
        re_batched = n2.contiguous().view(compact_q_shp)

        return re_batched
    
class DecoderAttentionLayer(nn.Module):
    
    def __init__(self, query_dim, key_dim, hid_dim, num_heads=1, 
                 device=torch.device('cpu')):
        super().__init__()
        self.self_mha = MultiHeadAttention(query_dim, query_dim, hid_dim, 
                                           num_heads=num_heads, device=device)
        #self.norm1 = nn.BatchNorm1d(query_dim).to(device)
        self.norm1 = nn.LayerNorm(query_dim).to(device)
        #self.norm1 = nn.Identity(query_dim).to(device)

        self.enc_mha = MultiHeadAttention(query_dim, key_dim, hid_dim, 
                                          num_heads=num_heads, device=device)
        #self.norm2 = nn.BatchNorm1d(query_dim).to(device)
        self.norm2 = nn.LayerNorm(query_dim).to(device)
        #self.norm2 = nn.Identity(query_dim).to(device)

        self.ff = MLP(query_dim, query_dim, query_dim * 2, 
                      device=device, norm_type=None)
        #self.norm3 = nn.BatchNorm1d(query_dim).to(device)
        self.norm3 = nn.LayerNorm(query_dim).to(device)
        #self.norm3 = nn.Identity(query_dim).to(device)

    def forward(self, q, k, mask=None):
        # runs multi-headed attention then puts everything into one
        # matrix for normalization and whatnot
        ext_q_shp = (len(q) * len(q[0]), len(q[0][0]))
        compact_q_shp = (len(q), len(q[0]), len(q[0][0]))
        ext_q = q.contiguous().view(ext_q_shp)
        
        self_mha = self.self_mha(q, q, q).contiguous().view(ext_q_shp)
        skip1 = ext_q + self_mha
        n1 = self.norm1(skip1)

        # rebatching results
        new_q = n1.contiguous().view(compact_q_shp)
        enc_mha = self.enc_mha(new_q, k, k, mask=mask).contiguous().view(ext_q_shp)

        skip2 = ext_q + enc_mha
        n2 = self.norm2(skip2)
        
        # feedforward
        ff = self.ff(n2)
        skip3 = n2 + ff
        n3 = self.norm3(skip3)
        
        re_batched = n3.contiguous().view(compact_q_shp)
        return re_batched
        
