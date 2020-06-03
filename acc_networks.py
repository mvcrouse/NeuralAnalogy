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
from graph_encoder import *
from nn_classes import *

class DagLSTMACC(nn.Module):
    # graph lstm accumulator neural network
    def __init__(self, dataset_params):
        super().__init__()
        self.device = dataset_params['device']
        self.is_cuda = (self.device != torch.device('cpu'))
        self.state_dim = dataset_params['state_dim']
        self.emb_dim = dataset_params['emb_dim']

        self.node_ct = dataset_params['node_ct']
        self.edge_ct = dataset_params['edge_ct']

        self.node_embedder = EmbProj(self.node_ct, self.emb_dim, 
                                     device=self.device)

        stdv = 1. / math.sqrt(self.state_dim)
        edge_matr_size = (self.edge_ct + 1, self.state_dim, self.state_dim)
        
        self.W_i = nn.Linear(self.emb_dim, self.state_dim, 
                             bias=False).to(self.device)
        U_i = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_i.uniform_(-stdv, stdv)
        self.U_i = nn.Parameter(U_i)
        self.b_i = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_o = nn.Linear(self.emb_dim, self.state_dim, 
                             bias=False).to(self.device)
        U_o = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_o.uniform_(-stdv, stdv)
        self.U_o = nn.Parameter(U_o)
        self.b_o = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_c = nn.Linear(self.emb_dim, self.state_dim, 
                             bias=False).to(self.device)
        U_c = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_c.uniform_(-stdv, stdv)
        self.U_c = nn.Parameter(U_c)
        self.b_c = nn.Parameter(torch.tensor(0., device=self.device))

        self.W_f = nn.Linear(self.emb_dim, self.state_dim, 
                             bias=False).to(self.device)
        U_f = torch.zeros(edge_matr_size, dtype=torch.float, device=self.device)
        U_f.uniform_(-stdv, stdv)
        self.U_f = nn.Parameter(U_f)
        self.b_f = nn.Parameter(torch.tensor(0., device=self.device))


    def compute_node_reprs(self, node_emb_inds, edge_emb_inds, upd_layers,
                           ret_init_embs=False):
        node_tensor = torch.tensor(node_emb_inds, device=self.device)
        node_states = self.node_embedder(node_tensor)
        node_zero_vec = torch.zeros((1, self.emb_dim), device=self.device)
        node_states = torch.cat((node_states, node_zero_vec), 0)

        edges_w_zv = edge_emb_inds + [self.edge_ct]
        edge_tensor = torch.tensor(edge_emb_inds + [self.edge_ct],
                                   device=self.device)

        node_reprs = torch.zeros(len(node_states), self.state_dim,
                                 device=self.device)
        node_mem = torch.zeros(len(node_states), self.state_dim,
                               device=self.device)
        node_to_node_sz = torch.Size([len(node_states), len(node_states)])
        
        W_i = self.W_i(node_states)
        W_o = self.W_o(node_states)
        W_c = self.W_c(node_states)
        W_f = self.W_f(node_states)

        for dir_upd_layer in upd_layers:

            diag = [(x, x) for x in set([y[0] for y in dir_upd_layer])]
            upd_diag = get_adj_matr(diag, node_to_node_sz, is_cuda=self.is_cuda)
            
            w_i = torch.mm(upd_diag, W_i)
            w_o = torch.mm(upd_diag, W_o)
            w_c = torch.mm(upd_diag, W_c)
            
            upd_layer = add_zv_to_no_deps(dir_upd_layer, len(node_states) - 1,
                                          len(edge_tensor) - 1)

            cat_upd_layer, i_matrs, o_matrs, c_matrs, f_matrs = [], [], [], [], []
            # we group together every update using the same edge matrix
            ext_upd_layer = [(ni, nj, eij, edges_w_zv[eij]) 
                             for ni, nj, eij in upd_layer]
            for grp_layer in group_similar_tup_sizes(ext_upd_layer, 3,
                                                     no_split=True):
                cont_layers, ind_map = [], {}
                for upd in grp_layer:
                    if not upd[3] in ind_map: 
                        ind_map[upd[3]] = len(cont_layers)
                        cont_layers.append([])
                    cont_layers[ind_map[upd[3]]].append(upd)
                cl_lens = [len(cl) for cl in cont_layers]
                max_sz = max(cl_lens)
                zv_tup = (len(node_states) - 1, len(node_states) - 1, 
                          len(edge_tensor) - 1, self.edge_ct)

                # we pad different for length matrix operations
                pad_layers = [cl + [zv_tup for _ in range(max_sz - len(cl))] 
                              for cl in cont_layers]
                cs_add_inds = torch.tensor([[add for _, add, _, _ in cl]
                                            for cl in pad_layers],
                                           device=self.device)

                # edge_lst is the minimal list of exactly which edges are to be used
                edge_lst = [cl[0][3] for cl in pad_layers]
                edge_inds = torch.tensor(edge_lst, device=self.device)

                # we get all the hidden states in the order they need to be 
                # multiplied and similarly with the edge matrices
                add_matrs = node_reprs[cs_add_inds]
                i_edge_matrs = self.U_i[edge_inds]
                o_edge_matrs = self.U_o[edge_inds]
                c_edge_matrs = self.U_c[edge_inds]
                f_edge_matrs = self.U_f[edge_inds]
                
                # we multiply the padded hidden states with the edge matrices
                i_res_matr = torch.bmm(add_matrs, i_edge_matrs)
                o_res_matr = torch.bmm(add_matrs, o_edge_matrs)
                c_res_matr = torch.bmm(add_matrs, c_edge_matrs)
                f_res_matr = torch.bmm(add_matrs, f_edge_matrs)
                
                # now we flatten our results into one matrix and discard the padding 
                for c_i, cl in enumerate(cont_layers):
                    cat_upd_layer.extend([(ni, nj, eij) for ni, nj, eij, _ in cl])
                    i_matrs.append(i_res_matr[c_i, :len(cl)])
                    o_matrs.append(o_res_matr[c_i, :len(cl)])
                    c_matrs.append(c_res_matr[c_i, :len(cl)])
                    f_matrs.append(f_res_matr[c_i, :len(cl)])

            i_edge_matr = torch.cat(i_matrs, 0)
            o_edge_matr = torch.cat(o_matrs, 0)
            c_edge_matr = torch.cat(c_matrs, 0)

            # in the order specified by the sub-batching operation we just did
            # we now grab our input and argument results
            src_inds = torch.tensor([src for src, _, _ in cat_upd_layer],
                                    device=self.device)
            add_inds = torch.tensor([add for _, add, _ in cat_upd_layer],
                                    device=self.device)

            edge_to_node_sz = torch.Size([len(node_reprs), len(src_inds)])
            adj_pairs = [(x[0], acc_pos) for acc_pos, x in enumerate(cat_upd_layer)]
            adj_matr = get_adj_matr(adj_pairs, edge_to_node_sz,
                                    is_cuda=self.is_cuda).to(self.device)

            u_i = torch.mm(adj_matr, i_edge_matr)
            i_gate = nn.Sigmoid()(w_i + u_i + self.b_i)

            u_o = torch.mm(adj_matr, o_edge_matr)
            o_gate = nn.Sigmoid()(w_o + u_o + self.b_o)

            u_c = torch.mm(adj_matr, c_edge_matr)
            ch_gate = nn.Tanh()(w_c + u_c + self.b_c)

            u_f = torch.cat(f_matrs, 0)
            w_f = W_f.index_select(0, src_inds)
            f_gate = nn.Sigmoid()(w_f + u_f + self.b_f)

            arg_mem = node_mem.index_select(0, add_inds)
            par_mem = i_gate * ch_gate + torch.mm(adj_matr, f_gate * arg_mem)

            # ensure only memory cells in current layer are updated
            restr_par_mem = torch.mm(upd_diag, par_mem)
            node_mem = torch.add(node_mem, restr_par_mem)
            
            # ensure only node hidden states in current layer are updated
            out_reprs = o_gate * nn.Tanh()(node_mem)
            new_node_reprs = torch.mm(upd_diag, out_reprs)

            node_reprs = torch.add(node_reprs, new_node_reprs)

        # removing 0 vector
        node_reprs = node_reprs[:len(node_reprs)-1]
        node_states = node_states[:len(node_states)-1]
        if ret_init_embs: return node_reprs, node_states 
        return node_reprs


