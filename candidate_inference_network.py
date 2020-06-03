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
from nn_classes import *

####
# Pointer network
####

class CandidateInferenceNetwork(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.device = dataset_params['device']
        self.is_cuda = self.device != torch.device('cpu')
        self.comp_dim = dataset_params['state_dim']
        self.hid_dim = dataset_params['ptr_hid_dim']
        self.beam_size = dataset_params['dec_beam_size']

        self.dec_nn = CandInfDecoder(dataset_params)

        seq_ends = []
        seq_end = torch.empty(self.comp_dim, dtype=torch.float, device=self.device)
        nn.init.normal_(seq_end)
        self.seq_end = nn.Parameter(seq_end)

        dec_start = torch.empty(self.comp_dim, dtype=torch.float, 
                                device=self.device)
        nn.init.normal_(dec_start)
        self.dec_start = nn.Parameter(dec_start)

    def train_selector(self, batch_inp_nodes, batch_cand_nodes, batch_cand_infs):
        output_probs, t_pos, f_pos, f_neg = [], 0, 0, 0
        
        # initial encoding is used as initial decoder states
        enc_reprs, enc_mask, _ = get_padded_repr(batch_inp_nodes, self.device)
        dec_reprs, dec_mask, _ = get_padded_repr(batch_cand_nodes, self.device)

        # now we compute predictions
        out_probs, sel_inds = self.train_on_examples(enc_reprs, enc_mask,
                                                     dec_reprs, dec_mask,
                                                     batch_cand_infs)
        output_probs.extend(out_probs)
        for in_nodes, sel_nodes in zip(batch_cand_infs, sel_inds):
            t_pos += len(set(sel_nodes).intersection(in_nodes))
            f_pos += len(set(sel_nodes).difference(in_nodes))
            f_neg += len(set(in_nodes).difference(sel_nodes))
        return output_probs, (t_pos, f_pos, f_neg)

    def train_on_examples(self, enc_reprs, enc_mask, dec_reprs, dec_mask,
                          batch_in_nodes):
        batch_size, seq_end_ind = len(dec_mask), len(dec_mask[0])
        batch_in_nodes = [in_nodes + [seq_end_ind] for in_nodes in batch_in_nodes]
        # encoder states
        seq_end = self.seq_end.expand(batch_size, 1, len(self.seq_end))
        att_opts = torch.cat((dec_reprs, seq_end), 1)

        # running selector
        out_probs, selected = [], [[] for _ in range(batch_size)]
        
        # dec state
        dec_states = self.dec_start
        dec_states = dec_states.expand(batch_size, 1, self.comp_dim)
        bool_row = torch.zeros(batch_size, 1, 1, device=self.device).bool()
        dec_mask = torch.cat((dec_mask, bool_row), 1).squeeze(2)

        # masks for decoding
        selected_tensors = []

        # actual selection
        for a_ind in range(seq_end_ind + 1):
            # get action probabilities
            distr = self.dec_nn.get_distr(enc_reprs, dec_states,
                                          att_opts, enc_mask, dec_mask)
            distr_max = torch.max(distr, dim=1)[1]
            new_inds = []
            for ex_num, in_nodes in enumerate(batch_in_nodes):
                if a_ind < len(in_nodes):
                    dec_mask[ex_num, in_nodes[a_ind]] = True
                    out_probs.append(distr[ex_num, in_nodes[a_ind]])
                    used_ind = in_nodes[a_ind]
                    if a_ind < len(in_nodes) - 1:
                        selected[ex_num].append(int(distr_max[ex_num]))
                else:
                    used_ind = 0
                new_inds.append(used_ind)
 
            # adding selected actions
            new_add_tensor = torch.tensor(new_inds, device=self.device)
            range_tensor = torch.arange(len(new_inds), device=self.device)
            new_actions = att_opts[range_tensor, new_add_tensor].unsqueeze(1)

            # new dec state
            dec_states = torch.cat((dec_states, new_actions), 1)

        return out_probs, selected

    def predict(self, batch_sel_nodes, batch_cand_nodes):
        # initial encoding is used as initial decoder states
        enc_reprs, enc_mask, _ = get_padded_repr(batch_sel_nodes, self.device)
        dec_reprs, dec_mask, _ = get_padded_repr(batch_cand_nodes, self.device)

        # now we compute predictions
        out_probs, sel_inds = self.predict_on_examples(enc_reprs, enc_mask, 
                                                       dec_reprs, dec_mask)
        return out_probs, sel_inds

    def predict_on_examples(self, enc_reprs, enc_mask, dec_reprs, dec_mask):
        batch_size, seq_end_ind = len(dec_mask), len(dec_mask[0])
        assert batch_size == 1, 'Batched beam search not yet implemented'
        # encoder states
        seq_end = self.seq_end.expand(batch_size, 1, len(self.seq_end))
        att_opts = torch.cat((dec_reprs, seq_end), 1)

        # running selector
        out_probs, selected = [], []

        # dec state
        dec_states = self.dec_start
        dec_states = dec_states.expand(batch_size, 1, self.comp_dim)
        bool_row = torch.zeros(batch_size, 1, 1, device=self.device).bool()
        dec_mask = torch.cat((dec_mask, bool_row), 1).squeeze(2)

        is_done = torch.zeros(1, device=self.device).bool()

        beams = [[1.0, dec_states, enc_mask, dec_mask, selected, out_probs]]

        fin_beams = []
        # actual selection
        for a_ind in range(seq_end_ind + 1):
            if len(fin_beams) >= self.beam_size: continue
            new_beams = []
            while beams:
                ( prob, dec_states, enc_mask, dec_mask, 
                  selected, out_probs ) = beams.pop(0)
                # get action probabilities
                distr = self.dec_nn.get_distr(enc_reprs, dec_states,
                                              att_opts, enc_mask, dec_mask)
                ch_k = self.beam_size
                if len(distr[0]) < self.beam_size: ch_k = len(distr[0])

                distr_top_k = torch.topk(distr, ch_k, dim=1)
                # because batch size is 1
                top_k_prob = distr_top_k[0][0]
                top_k_inds = distr_top_k[1][0]
                for pred_prob, pred_ind in zip(top_k_prob, top_k_inds):
                    if pred_prob == 0.: continue
                    new_prob = float(pred_prob * prob)
                    new_action = att_opts[:, pred_ind].unsqueeze(1)
                    new_dec_states = torch.cat((dec_states, new_action), 1)
                    if int(pred_ind) == seq_end_ind:
                        fin_beams.append([new_prob, selected, out_probs])
                    else:
                        new_enc_mask = enc_mask
                        new_dec_mask = dec_mask.clone()
                        new_dec_mask[:, pred_ind] = True
                        new_selected = selected + [int(pred_ind)]
                        new_out_probs = out_probs + [float(pred_prob)]
                        new_prob = prob * float(pred_prob)
                        new_beams.append([new_prob, new_dec_states, new_enc_mask,
                                          new_dec_mask, new_selected,new_out_probs])
            new_beams = sorted(new_beams, key=lambda x : x[0], reverse=True)
            beams = new_beams[:self.beam_size]
        ranked_beams = sorted(fin_beams, key = lambda x : x[0], reverse=True)

        best_prob, best_selected, best_probs = ranked_beams[0]
        # because we don't have batching working well
        best_probs = [best_probs]
        best_selected = [best_selected]
        return best_probs, best_selected


#########
# Encoder
#########

class CandInfEncoder(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.comp_dim = dataset_params['state_dim']
        self.hid_dim = dataset_params['ptr_hid_dim']
        self.num_heads = dataset_params['num_mha_heads']
        self.num_layers = dataset_params['num_att_layers']
        self.device = dataset_params['device']
        layers = []
        for _ in range(self.num_layers):
            layers.append(EncoderAttentionLayer(self.comp_dim, self.hid_dim,
                                                self.num_heads, self.device))
        self.layers = nn.ModuleList(layers)

    def encode_inp_nodes(self, inp_lst):
        #
        # padding and getting masks for padded positions
        #
        att_comp, bool_mask, exp_bool_mask = get_padded_repr(inp_lst, self.device)
        
        #
        # now we actually get the encoder attention
        #
        att_comp = self.layer_transforms(att_comp, exp_bool_mask)

        #
        # now we generate the encoder context
        #
        return att_comp, bool_mask

    def layer_transforms(self, att_comp, exp_bool_mask):
        for comp_layer in self.layers:
            att_comp = comp_layer(att_comp, mask=exp_bool_mask)
        return att_comp

#########
# Decoder
#########

class CandInfDecoder(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.comp_dim = dataset_params['state_dim']
        self.hid_dim = dataset_params['ptr_hid_dim']
        self.num_heads = dataset_params['num_mha_heads']
        self.num_layers = dataset_params['num_att_layers']
        self.device = dataset_params['device']

        # multi-head layers
        layers = []
        for _ in range(self.num_layers):
            layers.append(DecoderAttentionLayer(self.comp_dim, self.comp_dim, 
                                                self.hid_dim, self.num_heads, 
                                                self.device))
        self.layers = nn.ModuleList(layers)

        self.W_q = nn.Linear(self.comp_dim, self.hid_dim, 
                             bias=False).to(self.device)
        self.W_k = nn.Linear(self.comp_dim, self.hid_dim, 
                             bias=False).to(self.device)
        self.sqrt_dk = np.sqrt(self.hid_dim)

        self.clip_to = 1.

        self.controller = MLP(3, 1, hid_dim=64, norm_type=None, device=self.device,
                              mlp_layers=3)

    def get_distr(self, enc_states, dec_states, comp_outs,
                  enc_mask=None, dec_mask=None):

        # mask expansion
        upd_enc_mask = enc_mask.expand(-1, -1, len(dec_states[0])).transpose(1, 2)

        '''## generate new hidden states
        for comp_layer in self.layers:
            dec_states = comp_layer(dec_states, enc_states, upd_enc_mask)'''

        d_mask = torch.zeros(len(dec_states), len(dec_states[0]), 1,
                             device=self.device).bool()
        comb_mask = torch.cat((enc_mask, d_mask), 1)
        upd_comb_mask = comb_mask.expand(-1, -1, len(comb_mask[0])).transpose(1, 2)
        comb_states = torch.cat((enc_states, dec_states), 1)
        
        # now we compute single head attention to determine next action

        #q_block = self.W_q(dec_states)
        #q_block = self.W_q(enc_states)
        q_block = self.W_q(comb_states)
        k_block = self.W_k(comp_outs).transpose(1, 2)
        qk_sims = q_block.matmul(k_block) / self.sqrt_dk

        qk_sims = nn.Tanh()(qk_sims) * self.clip_to

        max_scores = torch.max(qk_sims, dim=1)[0].unsqueeze(2)
        mean_scores = torch.mean(qk_sims, dim=1).unsqueeze(2)
        min_scores = torch.min(qk_sims, dim=1)[0].unsqueeze(2)

        mmm_scores = torch.cat((max_scores, mean_scores, min_scores), 2)
        fin_scores = self.controller(mmm_scores).squeeze(2)
                
        #fin_scores = torch.max(qk_sims, dim=1)[0]
        #fin_scores = torch.min(qk_sims, dim=1)[0]
        
        if dec_mask is not None:
            fin_scores = fin_scores.masked_fill(dec_mask==True, float('-inf'))

        distr = nn.Softmax(dim=1)(fin_scores)

        return distr

#########
# Utilities
#########

def get_padded_repr(inp_lst, device):
    #
    # padding and getting masks for padded positions
    #
    max_len = max([len(inp) for inp in inp_lst])
    bool_mask = torch.zeros(len(inp_lst), max_len, device=device).bool()
    padded_comp = []
    for row, pc in enumerate(inp_lst):
        inp_size = len(pc)
        if len(pc) < max_len:
            mask_trues = torch.ones(max_len - inp_size, device=device)
            bool_mask[row, torch.arange(inp_size, max_len)] = mask_trues.bool()
            new_inp = []
            rem_zeros = torch.zeros(max_len - inp_size, len(pc[0]),
                                    device=device)
            pc = torch.cat((pc, rem_zeros), 0)
        padded_comp.append(pc)
    stacked = torch.stack(padded_comp)
    bool_mask = bool_mask.unsqueeze(2)
    exp_bool_mask = bool_mask.expand(-1, -1, len(bool_mask[0]))
    exp_bool_mask = exp_bool_mask.transpose(1, 2)
    return stacked, bool_mask, exp_bool_mask

