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

class MatcherNetwork(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.device = dataset_params['device']
        self.is_cuda = self.device != torch.device('cpu')
        self.comp_dim = dataset_params['ptr_comp_dim']
        self.sig_dim = dataset_params['state_dim']
        self.hid_dim = dataset_params['ptr_hid_dim']
        self.beam_size = dataset_params['dec_beam_size']

        self.enc_nn = MatcherEncoder(dataset_params)

        self.dec_nn = MatcherDecoder(dataset_params)

        seq_ends = []
        for dim in [self.comp_dim, self.sig_dim, self.sig_dim]:
            seq_end = torch.empty(dim, dtype=torch.float, device=self.device)
            nn.init.normal_(seq_end)
            seq_ends.append(nn.Parameter(seq_end))
        self.seq_ends = nn.ParameterList(seq_ends)

        dec_start = torch.empty(self.comp_dim, dtype=torch.float, 
                                device=self.device)
        nn.init.normal_(dec_start)
        self.dec_start = nn.Parameter(dec_start)

        left_start = torch.empty(self.sig_dim, dtype=torch.float, 
                                 device=self.device)
        nn.init.normal_(left_start)
        self.left_start = nn.Parameter(left_start)

        right_start = torch.empty(self.sig_dim, dtype=torch.float, 
                                  device=self.device)
        nn.init.normal_(right_start)
        self.right_start = nn.Parameter(right_start)

    def train_matcher(self, forests, batch_in_edges):
        output_probs, t_pos, f_pos, f_neg = [], 0, 0, 0
        
        # initial encoding is used as initial decoder states
        enc_reprs, enc_mask = self.enc_nn.encode_input(forests)

        # now we compute predictions
        out_probs, sel_inds = self.train_on_examples(enc_reprs, enc_mask, 
                                                     batch_in_edges)
        output_probs.extend(out_probs)
        for in_edges, sel_edges in zip(batch_in_edges, sel_inds):
            t_pos += len(set(sel_edges).intersection(in_edges))
            f_pos += len(set(sel_edges).difference(in_edges))
            f_neg += len(set(in_edges).difference(sel_edges))
        return output_probs, (t_pos, f_pos, f_neg)

    def train_on_examples(self, enc_info, enc_mask, batch_in_edges):
        enc_reprs, enc_ls, enc_rs = enc_info

        batch_size, seq_end_ind = len(enc_mask), len(enc_mask[0])
        batch_in_edges = [in_edges + [seq_end_ind] for in_edges in batch_in_edges]
        # encoder states
        seq_ends = []
        for i, seq_end in enumerate(self.seq_ends):
            if i == 0:
                seq_ends.append(seq_end.expand(batch_size, 1, len(seq_end)))
            else:
                normed_se = F.normalize(seq_end, dim=0, p=2)
                seq_ends.append(normed_se.expand(batch_size, 1, len(seq_end)))
        att_opts = [torch.cat((enc_base, seq_end), 1)
                    for enc_base, seq_end in zip(enc_info, seq_ends)]

        # running selector
        out_probs, selected = [], [[] for _ in range(batch_size)]
        
        # dec state
        dec_states = self.dec_start
        dec_states = dec_states.expand(batch_size, 1, self.comp_dim)

        # unit vectors for actions taken
        left_actions = F.normalize(self.left_start, dim=0, p=2)
        left_actions = left_actions.expand(batch_size, 1, self.sig_dim)
        right_actions = F.normalize(self.right_start, dim=0, p=2)
        right_actions = right_actions.expand(batch_size, 1, self.sig_dim)

        # masks for decoding
        bool_row = torch.zeros(batch_size, 1, 1, device=self.device).bool()
        dec_mask = torch.cat((enc_mask, bool_row), 1).squeeze(2)

        selected_tensors = []

        # actual selection
        for a_ind in range(seq_end_ind + 1):
            # get action probabilities
            distr = self.dec_nn.get_distr(enc_reprs, dec_states,
                                          left_actions, right_actions, 
                                          att_opts, enc_mask, dec_mask)

            distr_max = torch.max(distr, dim=1)[1]
            new_inds = []
            for ex_num, in_edges in enumerate(batch_in_edges):
                if a_ind < len(in_edges):
                    dec_mask[ex_num, in_edges[a_ind]] = True
                    out_probs.append(distr[ex_num, in_edges[a_ind]])
                    used_ind = in_edges[a_ind]
                    if a_ind < len(in_edges) - 1:
                        selected[ex_num].append(int(distr_max[ex_num]))
                else:
                    used_ind = 0
                new_inds.append(used_ind)
 
            # adding selected actions
            new_add_tensor = torch.tensor(new_inds, device=self.device)
            range_tensor = torch.arange(len(new_inds), device=self.device)
            new_actions = att_opts[0][range_tensor, new_add_tensor].unsqueeze(1)
            new_lefts = att_opts[1][range_tensor, new_add_tensor].unsqueeze(1)
            new_rights = att_opts[2][range_tensor, new_add_tensor].unsqueeze(1)

            # new dec state
            dec_states = torch.cat((dec_states, new_actions), 1)

            # unit vector concat
            left_actions = torch.cat((left_actions, new_lefts), 1)
            right_actions = torch.cat((right_actions, new_rights), 1)

        return out_probs, selected

    def predict(self, forests):
        # initial encoding is used as initial decoder states
        enc_reprs, enc_mask = self.enc_nn.encode_input(forests)
        
        # now we compute predictions
        out_probs, sel_inds = self.predict_on_examples(enc_reprs, enc_mask)
        return out_probs, sel_inds

    def predict_on_examples(self, enc_info, enc_mask):
        enc_reprs, enc_ls, enc_rs = enc_info
        batch_size, seq_end_ind = len(enc_mask), len(enc_mask[0])
        assert batch_size == 1, 'Batched beam search not yet implemented'
        # encoder states
        seq_ends = []
        for i, seq_end in enumerate(self.seq_ends):
            if i == 0:
                seq_ends.append(seq_end.expand(1, 1, len(seq_end)))
            else:
                normed_se = F.normalize(seq_end, dim=0, p=2)
                seq_ends.append(normed_se.expand(1, 1, len(seq_end)))
        att_opts = [torch.cat((enc_base, seq_end), 1)
                    for enc_base, seq_end in zip(enc_info, seq_ends)]

        # running selector
        out_probs, selected = [], []

        # dec state
        dec_states = self.dec_start
        dec_states = dec_states.expand(batch_size, 1, self.comp_dim)

        # unit vectors for actions taken
        left_actions = F.normalize(self.left_start, dim=0, p=2)
        left_actions = left_actions.expand(batch_size, 1, self.sig_dim)
        right_actions = F.normalize(self.right_start, dim=0, p=2)
        right_actions = right_actions.expand(batch_size, 1, self.sig_dim)

        # masks for decoding
        bool_row = torch.zeros(1, 1, 1, device=self.device).bool()
        dec_mask = torch.cat((enc_mask, bool_row), 1).squeeze(2)

        is_done = torch.zeros(1, device=self.device).bool()

        beams = [[1.0, dec_states, left_actions, right_actions, enc_mask, dec_mask,
                  selected, out_probs]]

        fin_beams = []
        # actual selection
        for a_ind in range(seq_end_ind + 1):
            if len(fin_beams) >= self.beam_size: continue
            new_beams = []
            while beams:
                ( prob, dec_states, left_actions, right_actions,
                  enc_mask, dec_mask, selected, out_probs ) = beams.pop(0)
                # get action probabilities
                distr = self.dec_nn.get_distr(enc_reprs, dec_states,
                                              left_actions, right_actions,
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
                    new_action = att_opts[0][:, pred_ind].unsqueeze(1)
                    new_left = att_opts[1][:, pred_ind].unsqueeze(1)
                    new_right = att_opts[2][:, pred_ind].unsqueeze(1)
                    new_dec_states = torch.cat((dec_states, new_action), 1)
                    new_left_actions = torch.cat((left_actions, new_left), 1)
                    new_right_actions = torch.cat((right_actions, new_right), 1)
                    if int(pred_ind) == seq_end_ind:
                        fin_beams.append([new_prob, selected, out_probs])
                    else:
                        new_enc_mask = enc_mask
                        new_dec_mask = dec_mask.clone()
                        new_dec_mask[:, pred_ind] = True
                        new_selected = selected + [int(pred_ind)]
                        new_out_probs = out_probs + [float(pred_prob)]
                        new_prob = prob * float(pred_prob)
                        new_beams.append([new_prob, new_dec_states,new_left_actions,
                                          new_right_actions, new_enc_mask,
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

class MatcherEncoder(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.comp_dim = dataset_params['ptr_comp_dim']
        self.hid_dim = dataset_params['ptr_hid_dim']
        self.num_heads = dataset_params['num_mha_heads']
        self.num_layers = dataset_params['num_att_layers']
        self.device = dataset_params['device']
        layers = []
        for _ in range(self.num_layers):
            layers.append(EncoderAttentionLayer(self.comp_dim, self.hid_dim,
                                                self.num_heads, self.device))
        self.layers = nn.ModuleList(layers)

    def encode_input(self, inp_lst):
        #
        # padding and getting masks for padded positions
        #
        max_len = max([len(inp[0]) for inp in inp_lst])
        bool_mask = torch.zeros(len(inp_lst), max_len, device=self.device).bool()
        padded_comp, padded_ls, padded_rs = [], [], []
        for row, inp in enumerate(inp_lst):
            pc, pl, pr = inp
            inp_size = len(pc)
            if len(pc) < max_len:
                mask_trues = torch.ones(max_len - inp_size, device=self.device)
                bool_mask[row, torch.arange(inp_size, max_len)] = mask_trues.bool()
                new_inp = []
                for inp_of in [pc, pl, pr]:
                    rem_zeros = torch.zeros(max_len - inp_size, len(inp_of[0]),
                                            device=self.device)
                    new_inp.append(torch.cat((inp_of, rem_zeros), 0))
                pc, pl, pr = new_inp
            padded_comp.append(pc)
            padded_ls.append(pl)
            padded_rs.append(pr)
        bool_mask = bool_mask.unsqueeze(2)
        exp_bool_mask = bool_mask.expand(-1, -1, len(bool_mask[0]))
        exp_bool_mask = exp_bool_mask.transpose(1, 2)
        #
        # now we actually get the encoder attention
        #
        att_comp = torch.stack(padded_comp)
        for comp_layer in self.layers:
            att_comp = comp_layer(att_comp, mask=exp_bool_mask)
        # cosine vectors unmodified
        att_ls = torch.stack(padded_ls)
        att_rs = torch.stack(padded_rs)
        return [att_comp, att_ls, att_rs], bool_mask
    
#########
# Decoder
#########

class MatcherDecoder(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.comp_dim = dataset_params['ptr_comp_dim']
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

        self.combiner = MLP(3, 1, hid_dim=64, norm_type=None, device=self.device,
                            mlp_layers=3)

        self.controller = MLP(3, 1, hid_dim=64, norm_type=None, device=self.device,
                              mlp_layers=3)

    def get_distr(self, enc_outs, dec_states, left_actions, right_actions, out_opts,
                  enc_mask=None, dec_mask=None):

        comp_outs, ls_outs, rs_outs = out_opts

        # mask expansion
        comp_mask = enc_mask.expand(-1, -1, len(dec_states[0])).transpose(1, 2)

        # generate new hidden states
        for comp_layer in self.layers:
            dec_states = comp_layer(dec_states, enc_outs, comp_mask)
        
        # now we compute single head attention to determine next action
        q_block = self.W_q(dec_states)
        k_block = self.W_k(comp_outs).transpose(1, 2)
        qk_sims = nn.Tanh()(q_block.matmul(k_block) / self.sqrt_dk)

        ls_sims = left_actions.bmm(ls_outs.transpose(1, 2))
        rs_sims = right_actions.bmm(rs_outs.transpose(1, 2))
        
        exp_sims = [sim.unsqueeze(3) for sim in [qk_sims, ls_sims, rs_sims]]
        mixed_sims = torch.cat(exp_sims, 3)

        comb_sims = self.combiner(mixed_sims).squeeze(3)
        
        max_scores = torch.max(comb_sims, dim=1)[0].unsqueeze(2)
        mean_scores = torch.mean(comb_sims, dim=1).unsqueeze(2)
        min_scores = torch.min(comb_sims, dim=1)[0].unsqueeze(2)

        mmm_scores = torch.cat((max_scores, mean_scores, min_scores), 2)
        fin_scores = self.controller(mmm_scores).squeeze(2)
        
        if dec_mask is not None:
            fin_scores = fin_scores.masked_fill(dec_mask==True, float('-inf'))

        distr = nn.Softmax(dim=1)(fin_scores)

        return distr

