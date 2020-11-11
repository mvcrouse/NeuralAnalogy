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
from acc_networks import *
from matcher_network import *
from candidate_inference_network import *
import parse_input_forms as pf

class SMN(nn.Module):

    def __init__(self, dataset_params):
        super().__init__()
        self.node_ct = dataset_params['node_ct']
        self.edge_ct = dataset_params['edge_ct']
        self.device = dataset_params['device']
        self.aug_ct = dataset_params['aug_ct']
        self.node_state_dim = dataset_params['state_dim']
        self.lr = dataset_params['lr']
        self.tiered_ident_thresh = dataset_params['ti_thresh']
        self.graph_encoder = dataset_params['graph_encoder']

        self.use_cos_sim = dataset_params['cos_sim']
        
        self.repr_nn = DagLSTMACC(dataset_params)
        self.sig_nn = DagLSTMACC(dataset_params)

        dataset_params['ptr_comp_dim'] = self.node_state_dim * 4
        self.matcher_nn = MatcherNetwork(dataset_params)
        self.cand_inf_nn = CandidateInferenceNetwork(dataset_params)

        sparse_param_ids = []
        for m in self.modules():
            if type(m) == torch.nn.modules.sparse.Embedding:
                sparse_param_ids.extend([id(p) for p in m.parameters()])
        dense_params = [param for param in self.parameters()
                        if not id(param) in sparse_param_ids]
        sparse_params = [param for param in self.parameters()
                         if id(param) in sparse_param_ids]
        self.dense_optimizer = torch.optim.Adam(dense_params, lr=self.lr)
        self.sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=self.lr)

        self.cr_loss_wt = 1.
        self.ci_loss_wt = 0.1

    #########
    # Training stuff
    #########

    def train_batch(self, node_inds, sig_inds, edge_inds,
                    upd_layers, func_sets, gr_ranges, in_edges, cand_infs):
        self.dense_optimizer.zero_grad()
        self.sparse_optimizer.zero_grad()
        loss = torch.tensor(0., device=self.device)

        bm_res = self.batch_matrs(node_inds, sig_inds, edge_inds, upd_layers, 
                                  func_sets, gr_ranges, in_edges=in_edges)

        cr_loss, cr_prec, cr_rec, cr_f1 = self.get_correspondence_loss(bm_res, 
                                                                       in_edges)
        #cr_loss, cr_prec, cr_rec, cr_f1 = 0., 0., 0., 0.
        base_incl = [[b for b, _ in in_e] for in_e in in_edges]
        ci_loss, ci_prec, ci_rec, ci_f1 = self.get_candidate_inf_loss(bm_res, 
                                                                      base_incl,
                                                                      cand_infs)

        loss = self.cr_loss_wt * cr_loss + self.ci_loss_wt * ci_loss
        loss.backward()
        self.dense_optimizer.step()
        self.sparse_optimizer.step()
        
        return float(loss), [(cr_prec, cr_rec, cr_f1),
                             (ci_prec, ci_rec, ci_f1)]

    def get_correspondence_loss(self, bm_res, in_edges):
        repr_matrs, sig_matrs, norm_matrs, idents = bm_res
        # now computing performance loss
        forests, in_pos, pos_dicts = [], [], []
        z_iter = zip(repr_matrs, sig_matrs, norm_matrs, in_edges, idents)
        for repr_pr, sig_pr, norm_pr, in_e, ident in z_iter:
            match_forest, pos_d, rev_p = self.build_match_forest(repr_pr, sig_pr,
                                                                 norm_pr, ident)
            forests.append(match_forest)
            in_p = [pos_d[e] for e in in_e]
            in_pos.append(in_p)

        false_pos, false_neg, true_pos = 0, 0, 0
        out_probs, metrics = self.matcher_nn.train_matcher(forests, in_pos)

        targets = torch.ones(len(out_probs), device=self.device, dtype=torch.float)
        loss = nn.BCELoss()(torch.stack(out_probs), targets)

        true_pos, false_pos, false_neg = metrics
        precision, recall, f1 = calc_metrics(true_pos, false_pos, false_neg)

        return loss, precision, recall, f1

    def get_candidate_inf_loss(self, bm_res, incl_nodes, cand_infs):
        _, sig_matrs, _, _ = bm_res
        # now computing performance loss
        sel_nodes, cand_nodes, in_pos = [], [], []
        z_iter = zip(sig_matrs, incl_nodes, cand_infs)
        for (base_sig_pr, _), i_ns, ci_ns in z_iter:
            b_ns = [i for i in range(len(base_sig_pr)) if not i in i_ns]
            if not i_ns: continue
            if not b_ns: continue
            i_ns_tensor = torch.tensor(i_ns, device=self.device)
            b_ns_tensor = torch.tensor(b_ns, device=self.device)
            sel_nodes.append(base_sig_pr.index_select(0, i_ns_tensor))
            cand_nodes.append(base_sig_pr.index_select(0, b_ns_tensor))
            in_p = [position(i, b_ns) for i in ci_ns]
            in_pos.append(in_p)

        if not sel_nodes: return 0., 0., 0., 0.

        false_pos, false_neg, true_pos = 0, 0, 0
        out_probs, metrics = self.cand_inf_nn.train_selector(sel_nodes, cand_nodes,
                                                             in_pos)

        targets = torch.ones(len(out_probs), device=self.device, dtype=torch.float)
        loss = nn.BCELoss()(torch.stack(out_probs), targets)

        true_pos, false_pos, false_neg = metrics
        precision, recall, f1 = calc_metrics(true_pos, false_pos, false_neg)

        return loss, precision, recall, f1

    #########
    # Testing stuff
    #########

    def batch_structural_match(self, node_inds, sig_inds, edge_inds, upd_layers, 
                               func_sets, gr_ranges, req_corrs=None):
        bm_res = self.batch_matrs(node_inds, sig_inds, edge_inds, upd_layers, 
                                  func_sets, gr_ranges, req_corrs=req_corrs)
        selected_corrs, base_preds = self.get_correspondences(bm_res)
        selected_cis = self.get_candidate_inferences(bm_res, base_preds)

        return selected_corrs, selected_cis

    def get_correspondences(self, bm_res):
        repr_matrs, sig_matrs, norm_matrs, idents = bm_res
        forests, in_pos, rev_dicts = [], [], []
        z_iter = zip(repr_matrs, sig_matrs, norm_matrs, idents)
        for repr_pr, sig_pr, norm_pr, ident in z_iter:
            match_forest, pos_d, rev_p = self.build_match_forest(repr_pr, sig_pr,
                                                                 norm_pr, ident)
            forests.append(match_forest)
            rev_dicts.append(rev_p)
        #
        # get base correspondence predictions
        #
        selected_corrs, base_preds = [], []
        out_probs, selected_inds = self.matcher_nn.predict(forests)
        for probs, sel_inds, rev_p in zip(out_probs, selected_inds, rev_dicts):
            base_preds.append([rev_p[ind][0] for ind in sel_inds])
            sel_lst = [(float(prob), rev_p[ind])
                       for prob, ind in zip(probs, sel_inds)]
            selected_corrs.append(sel_lst)

        return selected_corrs, base_preds

    def get_candidate_inferences(self, bm_res, base_preds):
        _, sig_matrs, _, _ = bm_res
        #
        # get candidate inference predictions
        #
        sel_nodes, cand_nodes, rev_dicts = [], [], []
        z_iter = zip(sig_matrs, base_preds)
        for (base_sig_pr, _), i_ns in z_iter:
            b_ns = [i for i in range(len(base_sig_pr)) if not i in i_ns]
            if not (i_ns and b_ns): continue
            i_ns_tensor = torch.tensor(i_ns, device=self.device)
            b_ns_tensor = torch.tensor(b_ns, device=self.device)
            sel_nodes.append(base_sig_pr.index_select(0, i_ns_tensor))
            cand_nodes.append(base_sig_pr.index_select(0, b_ns_tensor))
            rev_dicts.append(dict(list(enumerate(b_ns))))
        if not sel_nodes: return [[]]
        selected_cis = []
        out_probs, selected_inds = self.cand_inf_nn.predict(sel_nodes, cand_nodes)
        for probs, sel_inds, rev_p in zip(out_probs, selected_inds, rev_dicts):
            sel_lst = [(float(prob), rev_p[ind])
                       for prob, ind in zip(probs, sel_inds)]
            selected_cis.append(sel_lst)
        return selected_cis

    #####
    # Formatting
    #####

    def structural_match(self, b_node_lst, t_node_lst, k_encs=32, req_corrs=None,
                         score_by_prob=False, output_k=False, hashable_syms=False, use_iou=True):
        best_mapping = (float('-inf'), float('-inf'), [], [], (0, 0))
        all_mappings = []
        for _ in range(k_encs):
            lst_res = self.listify_input_pair(b_node_lst, t_node_lst,
                                              hashable_syms=hashable_syms)
            ( node_ranges, node_inds, sig_inds, edge_inds,
              offset_upd, offset_funcs, asgns ) = lst_res
            ind_req_corrs = []
            if req_corrs:
                ind_req_corrs = [(asgns[b], asgns[t]) for b, t in req_corrs]
            mapping, cis = self.batch_structural_match(node_inds, sig_inds, 
                                                       edge_inds, offset_upd, 
                                                       [offset_funcs],[node_ranges],
                                                       req_corrs=ind_req_corrs)
            mapping, cis = mapping[0], cis[0]
            if not mapping: continue
            mapping_prob = np.prod([prob for prob, _ in mapping])
            mapping_edges = [(b_node_lst[b], t_node_lst[t]) 
                             for _, (b, t) in mapping]
            valid_edges, sm_score, violations = get_sm_from_mapping(mapping_edges)
            mapping_val = (mapping_prob, sm_score, mapping, cis, 
                           valid_edges, violations)
            all_mappings.append(mapping_val)
            comp_score = mapping_prob if score_by_prob else sm_score
            best_score = best_mapping[0] if score_by_prob else best_mapping[1]
            v_ct, bv_ct = sum(violations), sum(best_mapping[len(best_mapping)-1])
            # take best score, otherwise break ties with violation count
            if (comp_score == best_score and v_ct < bv_ct) or \
               (comp_score > best_score):
                best_mapping = mapping_val
        iou_rankings = []
        for m1 in all_mappings:
            m1_prob = m1[0]
            edges1 = set([e for (p, e) in m1[2]])
            iou = 0
            for m2 in all_mappings:
                edges2 = [e for (p, e) in m2[2]]
                iou += (len(edges1.intersection(edges2)) / len(edges1.union(edges2)))
            iou_rankings.append((iou, m1_prob, m1))
        iou_rankings = sorted(iou_rankings, key=lambda x : x[0], reverse=True)
        if use_iou: best_mapping = iou_rankings[0][2]
        if output_k: return all_mappings
        return best_mapping

    def listify_input_pair(self, b_node_lst, t_node_lst, hashable_syms=False):
        node_ranges, node_inds, sig_inds = [], [], []
        edge_inds, offset_upd, offset_funcs, asgn_dict = [], [], [], {}
        self.graph_encoder.reset_assignments()
        base_info = self.graph_encoder.get_update_lsts(b_node_lst, 
                                                       hashable_syms=hashable_syms)
        targ_info = self.graph_encoder.get_update_lsts(t_node_lst,
                                                       hashable_syms=hashable_syms)
        for i, n in enumerate(b_node_lst): asgn_dict[n] = i
        for i, n in enumerate(t_node_lst): asgn_dict[n] = i + len(b_node_lst)
        for (n_inds, s_inds, e_inds, updates, funcs) in [base_info, targ_info]:
            n_offset, e_offset = len(node_inds), len(edge_inds)
            node_ranges.append([n_offset, n_offset + len(n_inds)])
            # embedding indices dont get offset
            node_inds.extend(n_inds)
            sig_inds.extend(s_inds)
            edge_inds.extend(e_inds)
            offset_funcs.append([f + n_offset for f in funcs])
            # get offset updates and merge them across batch
            for i, upd_layer in enumerate(updates):
                offset_layer = []
                for (n_i, n_j, e_ij) in upd_layer:
                    if n_j != None:
                        off_ex = (n_i + n_offset, n_j + n_offset, e_ij + e_offset)
                    else:
                        off_ex = (n_i + n_offset, None, None)
                    offset_layer.append(off_ex)
                if len(offset_upd) > i:
                    offset_upd[i].extend(offset_layer)
                else:
                    offset_upd.append(offset_layer)

        return node_ranges, node_inds, sig_inds, edge_inds, offset_upd, offset_funcs, asgn_dict

    def batch_matrs(self, node_inds, sig_inds, edge_inds, upd_layers, func_sets, 
                    gr_ranges, req_corrs=None, in_edges=None):
        if in_edges == None: in_edges = [[] for _ in gr_ranges]
        l_reqs, r_reqs = {}, {}
        if req_corrs:
            l_reqs, r_reqs = dict(req_corrs), dict([(t, b) for b, t in req_corrs])
        sig_matr = self.sig_nn.compute_node_reprs(sig_inds,edge_inds, upd_layers)
        #sig_matr = self.sig_nn.compute_node_reprs(sig_inds, upd_layers)
        if self.use_cos_sim:
            norm_sig_matr = F.normalize(sig_matr, dim=1, p=2)
        else:
            norm_sig_matr = sig_matr

        repr_matr, emb_matr = self.repr_nn.compute_node_reprs(node_inds, edge_inds,
                                                              upd_layers, 
                                                              ret_init_embs=True)
        batch_reprs, batch_sigs, batch_norm_sigs, batch_idents = [], [], [], []
        z_iter = zip(func_sets, gr_ranges, in_edges)
        for ex_num, ((l_funcs, r_funcs), (l, r), in_e) in enumerate(z_iter):
            l_span = torch.tensor(range(l[0], l[1]), dtype=torch.long,
                                  device=self.device)
            r_span = torch.tensor(range(r[0], r[1]), dtype=torch.long,
                                  device=self.device)
            
            l_rm = repr_matr.index_select(0, l_span)
            r_rm = repr_matr.index_select(0, r_span)
            batch_reprs.append((l_rm, r_rm))

            l_sm = norm_sig_matr.index_select(0, l_span)
            r_sm = norm_sig_matr.index_select(0, r_span)
            batch_sigs.append((l_sm, r_sm))

            l_sm = norm_sig_matr.index_select(0, l_span)
            r_sm = norm_sig_matr.index_select(0, r_span)
            batch_norm_sigs.append((l_sm, r_sm))

            l_em = emb_matr.index_select(0, l_span)
            r_em = emb_matr.index_select(0, r_span)

            mf_inds = set()
            for l_orig_ind in range(len(l_rm)):
                l_ind = l[0] + l_orig_ind
                for r_orig_ind in range(len(r_rm)):
                    r_ind = r[0] + r_orig_ind
                    # computes node embedding similarity
                    dist = torch.sum((l_rm[l_orig_ind] - r_rm[r_orig_ind]).pow(2))
                    # last conjunct in clause specifies functors
                    # to match regardless of labels
                    if (l_orig_ind, r_orig_ind) in in_e or \
                       (dist <= self.tiered_ident_thresh and \
                        not (l_ind in l_reqs or r_ind in r_reqs)) or \
                        (l_ind in l_reqs and l_reqs[l_ind] == r_ind):
                        #(l_ind in l_funcs and r_ind in r_funcs):
                        mf_inds.add((l_orig_ind, r_orig_ind))
            # this will restrict it to only training
            if in_edges and self.aug_ct > 0:
                aug_edges = self.augment_edges(l[0], r[0], len(l_rm), len(r_rm), 
                                               mf_inds)
                mf_inds = mf_inds.union(aug_edges)
            batch_idents.append(mf_inds)
        return batch_reprs, batch_sigs, batch_norm_sigs, batch_idents

    def augment_edges(self, l_0, r_0, l_max, r_max, curr_edges):
        new_edges = set()
        for l in set([l for l, _ in curr_edges]):
            if r_max > self.aug_ct:
                aug = np.random.choice(list(range(r_max)), self.aug_ct,
                                       replace=False)
            else: aug = list(range(r_max))
            for a in aug: new_edges.add((l_0 + l, r_0 + a))
        for r in set([r for _, r in curr_edges]):
            if l_max > self.aug_ct:
                aug = np.random.choice(list(range(l_max)), self.aug_ct,
                                       replace=False)
            else: aug = list(range(l_max))
            for a in aug: new_edges.add((l_0 + a, r_0 + r))
        return new_edges

    #################
    # Pointer input
    #################

    def build_pointer_inputs(self, l_len, r_len, allowable_matches):
        pos_d, sd_ct, l_c, r_c = {}, 0, [], []
        for i in range(l_len):
            for j in range(r_len):
                if not (i, j) in allowable_matches: continue
                pos_d[(i, j)] = sd_ct
                sd_ct += 1
                l_c.append(i)
                r_c.append(j)
        l_c = torch.tensor(l_c, device=self.device)
        r_c = torch.tensor(r_c, device=self.device)
        return sd_ct, l_c, r_c, pos_d

    def build_match_forest(self, repr_pair, sig_pair, norm_pair, ident):
        l_repr_matr, r_repr_matr = repr_pair
        l_sig_matr, r_sig_matr = sig_pair
        l_norm_matr, r_norm_matr = norm_pair
        l_len, r_len = len(l_repr_matr), len(r_repr_matr)
        sd_inds, l_c, r_c, pos_d = self.build_pointer_inputs(l_len, r_len, ident)
        rev_p = dict([(v, k) for k, v in pos_d.items()])
        l_rm = l_repr_matr.index_select(0, l_c)
        r_rm = r_repr_matr.index_select(0, r_c)
        l_sm = l_sig_matr.index_select(0, l_c)
        r_sm = r_sig_matr.index_select(0, r_c)
        l_nm = l_norm_matr.index_select(0, l_c)
        r_nm = r_norm_matr.index_select(0, r_c)
        match_forest = [torch.cat((l_rm, r_rm, l_nm, r_nm), 1), l_sm, r_sm]
        return match_forest, pos_d, rev_p

    #################    

