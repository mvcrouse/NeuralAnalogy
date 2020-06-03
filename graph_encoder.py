# python imports
import sys, random
# torch imports
import torch
import torch.nn as nn
# numpy imports
import numpy as np
# code imports
from node_classes import *
from utilities import *

class GraphEncoder:

    def __init__(self, **kw_args):

        self.node_ct = kw_args['node_ct']
        self.edge_ct = kw_args['edge_ct']
        self.max_arity = kw_args['max_arity']
        self.node_prop_asgns = kw_args['nodes_bprop']
        # having these divisibility constraints makes the
        # embedding assignments easier

    def reset_assignments(self):
        self.node_assignments = {}
        self.ind_assignments = {}
        self.node_signatures = {}
        self.sig_assignments = {}

    def listify_graph_example(self, graph_ex, hashable_syms=False):
        b_nodes, t_nodes, matched_edges, cand_inf_nodes = graph_ex

        # node inds assigned consistently but randomly by example
        # edge inds assigned consistently across all examples
        self.reset_assignments()

        ( b_embs, b_sigs, b_edges, b_upds, b_funcs
          ) = self.get_update_lsts(b_nodes, hashable_syms=hashable_syms)
        ( t_embs, t_sigs, t_edges, t_upds, t_funcs
          ) = self.get_update_lsts(t_nodes, hashable_syms=hashable_syms)

        in_edges = []
        for b, t in matched_edges:
            edge = (position(b, b_nodes), position(t, t_nodes))
            in_edges.append(edge)

        cand_infs = []
        for b in cand_inf_nodes:
            cand_infs.append(position(b, b_nodes))
            
        return [(b_embs, b_sigs, b_edges, b_upds, b_funcs),
                (t_embs, t_sigs, t_edges, t_upds, t_funcs),
                in_edges, cand_infs]

    def get_update_lsts(self, node_lst, hashable_syms=False):
        edge_lst = [(node, pos) for node in node_lst 
                    for pos in range(len(node.args))]

        positions = {}
        for i, node in enumerate(node_lst): positions[node] = i

        node_emb_inds = [None for _ in range(len(node_lst))]
        node_sig_inds = [None for _ in range(len(node_lst))]
        edge_emb_inds = [None for _ in range(len(edge_lst))]
        nne_updates, funcs = [], []

        for upd_layer in topological_sort(node_lst):
            nne_layer, key_refs = [], {}
            for node in upd_layer:
                assert not node in key_refs
                emb_inp_label = node.label
                if len(node.args) == 0: emb_inp_label = 'const_plc'

                node_pos = positions[node]

                if node.type_of == Node.func_type: funcs.append(node_pos)

                bpr_k = node.key_form()
                srtd = sorted(self.node_prop_asgns.keys())
                lower_ind = 0
                for k_srtd in srtd:
                    if k_srtd == bpr_k: break
                    lower_ind += self.node_prop_asgns[k_srtd]
                upper_ind = lower_ind + self.node_prop_asgns[bpr_k]

                emb_node_key = (emb_inp_label, node.key_form())
                sig_node_key = id(node)
                key_refs[node] = [emb_node_key, sig_node_key]
                if node.args: sig_node_key = node.key_form()
                gl_tup = (emb_node_key, node_emb_inds,
                          self.ind_assignments, self.node_assignments)
                loc_tup = (sig_node_key, node_sig_inds,
                           self.sig_assignments, self.node_signatures)
                for tup_ind, tup_info in enumerate([gl_tup, loc_tup]):
                    key_in, ret_inds, ind_assigns, node_assigns = tup_info
                    ind_opts = list(range(lower_ind, upper_ind))
                    # if it's a non-leaf node then the signature will
                    # be the zero vector
                    if node.args and key_in == sig_node_key: 
                        # zero vector is at self.node_ct because we give the 
                        # pytorch embedding object a padding dimension
                        ind_opts = [self.node_ct]
                    else:
                        ind_opts = [x for x in ind_opts if not x in ind_assigns]
                    if not key_in in node_assigns:
                        if hashable_syms and not ind_opts:
                            # we can do hashing a bit smarter here, if we don't
                            # assign to the index of a node that is in the same
                            # layer, we're less likely to get a bad collision
                            bad_assigns = [node_assigns[key_refs[n][tup_ind]] 
                                           for n in upd_layer if n in key_refs and \
                                           key_refs[n][tup_ind] in node_assigns]
                            pick_from = [k for k in ind_assigns.keys()
                                         if not k in bad_assigns]
                            if not pick_from: pick_from = list(ind_assigns.keys())
                            chosen_ind = np.random.choice(pick_from)
                        else:
                            chosen_ind = np.random.choice(ind_opts)
                        node_assigns[key_in] = chosen_ind
                        ind_assigns[chosen_ind] = key_in
                    ret_inds[node_pos] = node_assigns[key_in]

                # getting edge inds
                for a_ind, arg in enumerate(node.args):
                    edge_pos = position((node, a_ind), edge_lst)
                    edge_emb_ind = 0 if node.ordered else int(self.edge_ct / 2)
                    edge_emb_ind += sum([i for i in range(len(node.args))])
                    edge_emb_ind += (a_ind if node.ordered else 0)
                    edge_emb_inds[edge_pos] = edge_emb_ind
                    arg_pos = positions[arg]
                    nne = [node_pos, arg_pos, edge_pos]
                    nne_layer.append(nne)
                    
                if not node.args:
                    # adding this specifically for leaf nodes
                    nne_layer.append((node_pos, None, None))
            nne_updates.append(nne_layer)

        assert not any(x == None for x in node_emb_inds)
        assert not any(x == None for x in node_sig_inds)
        assert not any(x == None for x in edge_emb_inds)

        return node_emb_inds, node_sig_inds, edge_emb_inds, nne_updates, funcs
    
