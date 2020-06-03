# python imports
import sys, os
# torch imports
import torch
import torch.nn as nn
# numpy imports
import numpy as np
# graphviz imports
from graphviz import Digraph
# code imports
import settings_file as sf
from node_classes import *
import parse_input_forms as pf

#################
# General utilities
#################

def extract_sm_test_files():
    test_set_dir = sf.dataset_params['parsed_sm_tests_loc']
    f_pairs = {}
    for test_file in os.listdir(test_set_dir):
        if not '.sme' in test_file: continue
        if not test_file in f_pairs: f_pairs[test_file] = [None, None, None]
        with open(os.path.join(test_set_dir, test_file), 'r') as f:
            all_lines = [l for l in list(f.readlines()) if not l[0] in ['\n', ';']]
            all_lines = ' '.join(all_lines).replace('\n', '')
            if 'base' in test_file:
                f_pairs[test_file][0] = all_lines
            elif 'ta' in test_file:
                f_pairs[test_file][1] = all_lines
    return f_pairs

def uniquify(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]

def position(el, lst, key=None):
    if key:
        return next(i for i, x in enumerate(lst) if key(x) == key(el))
    return next(i for i, x in enumerate(lst) if x == el)

def bucketize_sim_matr(matr, thresh=0.0, div=10):
    buckets = {}
    for i in range(len(matr)):
        for j in range(len(matr[0])):
            num_key = round(matr[i, j], 1)
            if num_key < thresh: continue
            key_in = str(round(matr[i, j], 1))
            if not key_in in buckets: buckets[key_in] = []
            buckets[key_in].append((matr[i, j], (i, j)))
    return buckets

def group_similar_tup_sizes(tuples, key_in=0, no_split=False, grp_sp=10, min_bk=10):
    if no_split: return [tuples]
    indiv_buckets = {}
    for tup in tuples:
        src = tup[key_in]
        if not src in indiv_buckets: indiv_buckets[src] = []
        indiv_buckets[src].append(tup)
    buckets = {}
    for src, tups in indiv_buckets.items():
        if len(tups) <= min_bk: bucket_id = -min_bk
        else: bucket_id = round(len(tups) / grp_sp)
        if not bucket_id in buckets: buckets[bucket_id] = []
        buckets[bucket_id].extend(tups)
    return list(buckets.values())

def get_subgraph_size_from_adj_d(adj):
    deps, s_sizes = {}, {}
    for par, args in adj.items():
        if not par in deps: aggr_deps(par, adj, deps)
        s_sizes[par] = len(deps[par])
    return s_sizes

def aggr_deps(start_from, adj, deps):
    if start_from in deps:
        return deps[start_from]
    elif start_from in adj:
        args = adj[start_from]
        for a in args: aggr_deps(a, adj, deps)
        deps[start_from] = set([start_from])
        for a in args:
            deps[start_from] = deps[start_from].union(deps[a])
    else:
        deps[start_from] = set([start_from])
        return deps

def calc_metrics(true_pos, false_pos, false_neg, true_neg=None):
    ex_prec = 1. if false_pos == 0 else (true_pos / (true_pos + false_pos))
    ex_rec = 1. if false_neg == 0 else (true_pos / (true_pos + false_neg))
    if ex_prec + ex_rec == 0: ex_f1 = 0.
    else: ex_f1 = 2 * ex_prec * ex_rec / (ex_prec + ex_rec)
    if true_neg is not None:
        t_sum = true_pos + false_pos + false_neg + true_neg
        ex_acc = 1. if t_sum == 0 else (true_pos + true_neg) / t_sum
        if false_pos + true_neg == 0: ex_tnr = 1.
        else: ex_tnr = true_neg / (false_pos + true_neg)
        return ex_prec, ex_rec, ex_f1, ex_acc, ex_tnr
    return ex_prec, ex_rec, ex_f1

#################
# Matching utilities
#################

def get_sm_from_mapping(mapping, gold_standard=False):
    # gets the score, will be the size of each
    # distinct mapping, where a mapping is determined by the maximal
    # expression within it
    
    violations = []
    # first we get 1-to-1 violations
    fd, bd = {}, {}
    for b, t in mapping:
        if not b in fd: fd[b] = []
        fd[b].append(t)
        if not t in bd: bd[t] = []
        bd[t].append(b)
    forward, backward, one_to_one_missed = {}, {}, 0
    for k, v in fd.items():
        if len(v) > 1: one_to_one_missed += len(v)
        elif len(v) == 1: forward[k] = v[0]
    for k, v in bd.items():
        if len(v) > 1: one_to_one_missed += len(v)
        elif len(v) == 1: backward[k] = v[0]
    violations.append(one_to_one_missed)

    # now pulling out valid subset of mapping that doesn't violate
    # parallel connectivity
    pc_violations, tiered_identicality_violations = 0, 0
    good_edges = set()
    for edge in mapping:
        if not validate_structural_match(edge[0], edge[1], forward, backward,
                                         recurse=False):
            pc_violations += 1
        elif edge[0].type_of == Node.func_type and \
             edge[1].type_of == Node.func_type and \
             edge[0].label != edge[1].label and \
             not any((p1, p2) in mapping for p1 in edge[0].parents 
                     for p2 in edge[1].parents):
            tiered_identicality_violations += 1
        elif edge[0].type_of == Node.pred_type and \
             edge[1].type_of == Node.pred_type and \
             edge[0].label != edge[1].label:
            tiered_identicality_violations += 1
        # this does a recursive validation of this match
        elif validate_structural_match(edge[0], edge[1], forward, backward,
                                       recurse=True):
            good_edges.add(edge)
    violations.append(pc_violations)
    violations.append(tiered_identicality_violations)

    degen_const_violations = 0
    l_c = set([edge[0] for edge in good_edges])
    r_c = set([edge[1] for edge in good_edges])
    m_l_c = set([edge[0] for edge in mapping])
    m_r_c = set([edge[1] for edge in mapping])
    valid_edges = []
    # this filters out degenerate constants
    for edge in good_edges:
        if edge[0].args:
            valid_edges.append(edge)
        elif any(p in l_c for p in edge[0].parents) and \
             any(p in r_c for p in edge[1].parents):
            valid_edges.append(edge)
        elif not (any(p in m_l_c for p in edge[0].parents) and \
                  any(p in m_r_c for p in edge[1].parents)):
            degen_const_violations += 1
    violations.append(degen_const_violations)

    str_score = 0
    if gold_standard: valid_edges = mapping
    for edge in valid_edges:
        assert gold_standard or edge[0] in forward and edge[1] in backward
        if not (any(other_edge[0] in edge[0].parents 
                    for other_edge in valid_edges) or \
                # this avoids the extension symbols from being counted
                (pf.arg_ext_base in edge[0].label) or \
                (pf.arg_ext_base in edge[1].label)):
            assert edge[0].args
            dep_sg = edge[0].dep_subgraph()
            # this avoids the extension symbols from being counted
            dep_sg = [n for n in dep_sg if not pf.arg_ext_base in n.label]
            str_score += len(dep_sg)
    violations.append(len(mapping) - len(valid_edges))

    return valid_edges, str_score, violations

def validate_structural_match(base, targ, bd, td, recurse=True):
    if len(base.args) != len(targ.args): return False
    if base.ordered != targ.ordered: return False
    if not (base in bd and targ in td and bd[base] == targ and td[targ] == base):
        return False
    # now we ensure total subgraph alignment
    if not base.args:
        return True
    elif base.ordered:
        for b_arg, t_arg in zip(base.args, targ.args):
            if not (b_arg in bd and t_arg in td): return False
            if bd[b_arg] != t_arg: return False
            if td[t_arg] != b_arg: return False
            if recurse and not validate_structural_match(b_arg, t_arg, bd, td):
                return False
    else:
        b_arg_mapping, t_arg_mapping = {}, {}
        for b_arg in base.args:
            if not b_arg in bd: return False
            if not b_arg in b_arg_mapping: b_arg_mapping[b_arg] = []
            for t_arg in targ.args:
                if bd[b_arg] == t_arg: b_arg_mapping[b_arg].append(t_arg)
        for t_arg in targ.args:
            if not t_arg in td: return False
            if not t_arg in t_arg_mapping: t_arg_mapping[t_arg] = []
            for b_arg in base.args:
                if td[t_arg] == b_arg: t_arg_mapping[t_arg].append(b_arg)

        for b_arg, t_args in b_arg_mapping.items():
            if len(set(t_args)) != 1: return False
            if not t_args[0] in t_arg_mapping: return False
            if len(t_arg_mapping[t_args[0]]) == 0: return False
            if t_arg_mapping[t_args[0]][0] != b_arg: return False
        for t_arg, b_args in t_arg_mapping.items():
            if len(set(b_args)) != 1: return False
            if not b_args[0] in b_arg_mapping: return False
            if len(b_arg_mapping[b_args[0]]) == 0: return False
            if b_arg_mapping[b_args[0]][0] != t_arg: return False

        # now recursively validate
        for b_arg, t_args in b_arg_mapping.items():
            # only need to validate first 
            if recurse and not validate_structural_match(b_arg, t_args[0], bd, td):
                return False
    # if it passes everything then just return true
    return True

#################
# Candidate inference utilities
#################

def get_cand_infs_from_corrs(correspondences, criterion='liberal'):#'conservative'):
    just_base = [b for b, _ in correspondences]
    anc_cand_infs, base_cand_infs = [], []
    to_check = list(set(just_base))
    # we add candidate inferences in this weird way to get a particular ordering
    while to_check:
        curr = to_check.pop(0)
        for p in curr.parents:
            if p in anc_cand_infs: continue
            anc_cand_infs.append(p)
            if not p in to_check: to_check.append(p)
    if criterion == 'liberal':
        base_cand_infs, to_check = anc_cand_infs, anc_cand_infs + []
        while to_check:
            curr = to_check.pop(0)
            for a in curr.args:
                if (a in base_cand_infs) or (a in just_base): continue
                base_cand_infs.append(a)
                if not a in to_check: to_check.append(a)
    elif criterion == 'conservative':
        ci_support = set(just_base)
        new_node_added = True
        while new_node_added:
            new_node_added = False
            for b in anc_cand_infs:
                if b in base_cand_infs: continue
                if not all(arg in ci_support for arg in b.args): continue
                base_cand_infs.append(b)
                ci_support.add(b)
                new_node_added = True
    else:
        raise ValueError('Unknown candidate inference criterion: '+str(criterion))
    base_cand_infs = [b for b in base_cand_infs if not b in just_base]
    return base_cand_infs

#################
# Graph utilities
#################

def coalesce_graph(obj_lst, obj_dict=None):
    if obj_dict == None: obj_dict = {}
    return list(set([coalesce_subgraph(obj, obj_dict) for obj in obj_lst]))
    
def coalesce_subgraph(obj, obj_dict=None):
    if obj_dict == None: obj_dict = {}
    if str(obj) in obj_dict: return obj_dict[str(obj)]
    if not obj.args:
        obj_dict[str(obj)] = obj
        return obj
    new_obj = Node(obj.label, ordered=obj.ordered, type_of=obj.type_of)
    new_args = [coalesce_subgraph(arg, obj_dict) for arg in obj.args]
    new_obj.args = new_args
    for a in new_args: a.parents.append(new_obj)
    obj_dict[str(new_obj)] = new_obj
    return new_obj

def topological_sort(node_lst):
    par_dict = {}
    for node in node_lst:
        if not node in par_dict: par_dict[node] = set()
        for par in node.parents:
            par_dict[node].add(par)
        # should be redundant, but just in case...
        for arg in node.args:
            if not arg in par_dict: par_dict[arg] = set()
            par_dict[arg].add(node)

    # actual layers
    update_layers = []
    rem_nodes = node_lst + []
    while rem_nodes:
        layer_nodes = [node for node in rem_nodes if not par_dict[node]]
        for node in layer_nodes:
            for arg in node.args:
                if node in par_dict[arg]: par_dict[arg].remove(node)
        rem_nodes = [node for node in rem_nodes if not node in layer_nodes]
        update_layers.append(layer_nodes)

    # ensures leaf nodes are in the very first layer
    # and root nodes in the very last
    leaf_nodes, non_leaf_nodes = [], []
    for layer in reversed(update_layers):
        new_layer = []
        for node in layer:
            if node.args:
                new_layer.append(node)
            else:
                leaf_nodes.append(node)
        if new_layer:
            non_leaf_nodes.append(new_layer)

    return [leaf_nodes] + non_leaf_nodes

#################
# Encoder utilities
#################

def flip_upd_layers(upd_layers):
    new_upd_layers = []
    restr_upd_layers = [[(a, d, e) for a, d, e in upd_layer if d != None]
                        for upd_layer in upd_layers]
    restr_upd_layers = [upd_layer for upd_layer in restr_upd_layers if upd_layer]
    desc = set([y for upd_layer in restr_upd_layers for _, y, _ in upd_layer])
    asc = set([x for upd_layer in restr_upd_layers for x, _, _ in upd_layer])
    roots = [(x, None, None) for x in asc.difference(desc)]
    for upd_layer in reversed(restr_upd_layers):
        new_upd_layers.append([(d, a, e) for a, d, e in upd_layer])
    return [x for x in ([roots] + new_upd_layers) if x]

def add_zv_to_no_deps(dir_upd_layer, node_zv, edge_zv):
    upd_layer = []
    for src, add, edge in dir_upd_layer:
        add_triple = (src, add, edge)
        if add == None: add_triple = (src, node_zv, edge_zv)
        upd_layer.append(add_triple)
    return upd_layer

def zero_out_non_leafs(dir_upd_layer, node_zv):
    upd_layer = []
    for src, add, _ in dir_upd_layer:
        if add == None: 
            add_triple = (src, src, node_zv)
        else:
            add_triple = (src, node_zv, add)
        upd_layer.append(add_triple)
    return upd_layer

#################
# PyTorch utilities
#################

def switch_device(model, new_device):
    true_flags = []
    false_flags = []
    if new_device == torch.device('cpu'): false_flags.append('is_cuda')
    switch_devices, t_flags, f_flags = set(), set(), set()
    for f, val in model.__dict__.items():
        if type(val) == torch.device:
            switch_devices.add(f)
        if f in false_flags:
            f_flags.add(f)
        if f in true_flags:
            t_flags.add(f)
    for f in switch_devices: model.__dict__[f] = new_device
    for f in f_flags: model.__dict__[f] = False
    for f in t_flags: model.__dict__[f] = True
    for m in model._modules:
        switch_device(model._modules[m], new_device)

def get_adj_matr(pairs, size, is_cuda=False, mean=True):
    if is_cuda:
        i = torch.cuda.LongTensor(pairs)
    else:
        i = torch.LongTensor(pairs)
        
    if mean:
        src_ct = {}
        for src, _ in pairs:
            if not src in src_ct: src_ct[src] = 0
            src_ct[src] += 1
        if is_cuda:
            v = torch.cuda.FloatTensor([1 / src_ct[src] for src, _ in pairs])
        else:
            v = torch.FloatTensor([1 / src_ct[src] for src, _ in pairs])
    else:
        if is_cuda:
            v = torch.cuda.FloatTensor([1 for _ in range(len(pairs))])
        else:
            v = torch.FloatTensor([1 for _ in range(len(pairs))])
    if is_cuda:
        return torch.cuda.sparse.FloatTensor(i.t(), v, size)
    return torch.sparse.FloatTensor(i.t(), v, size)

#################
# Visualization
#################

def visualize_alignment(nodes1, nodes2, alignments, cand_infs=None, 
                        file_app='', col='green', constr_disp=False,
                        horiz=True):
    if cand_infs == None: cand_infs = []
    #dot - filter for drawing directed graphs
    #neato - filter for drawing undirected graphs
    #twopi - filter for radial layouts of graphs
    #circo - filter for circular layout of graphs
    #fdp - filter for drawing undirected graphs
    #sfdp - filter for drawing large undirected graphs
    #patchwork - filter for tree maps
    eng = 'dot'
    #eng = 'sfdp'
    #eng = 'neato'
    #eng = 'twopi'
    dag = Digraph(filename=sf.dataset_params['vis_data_loc'] + file_app,
                  engine=eng)
    ci_nodes = dict([(nodes1[ind], prob) for prob, ind in cand_infs])
    good_base, good_targ = set(), set()
    for _, ind in cand_infs: 
        good_base.add(ind)
        for n in nodes1[ind].dep_subgraph(): good_base.add(position(n, nodes1))
    for _, (b_ind, t_ind) in alignments:
        good_base.add(b_ind)
        good_targ.add(t_ind)
        for n in nodes1[b_ind].dep_subgraph(): good_base.add(position(n, nodes1))
        for n in nodes2[t_ind].dep_subgraph(): good_targ.add(position(n, nodes2))

    for i, nodes in enumerate([nodes1, nodes2]):
        gr_name = 'base' if i == 0 else 'target'
        # graph name must begin with 'cluster' for graphviz
        with dag.subgraph(name='cluster_' + gr_name) as g:
            g.attr(color='black')
            g.attr(label=gr_name)
            g.attr(style='invis')
            tsrt = reversed(topological_sort(nodes))
            added_nodes = set()
            for layer in tsrt:
                for node in layer:
                    node_col = '0 0 0'
                    sty = 'filled'
                    fill_col = 'white'
                    if node in ci_nodes:
                        ci_np = max(ci_nodes[node], 0) * 3
                        node_col = '0.9 1 1'# + str(ci_np)
                        sty = 'bold'
                    n_shape = 'ellipse' if node.ordered else 'rectangle'
                    n_pos = position(node, nodes)
                    if (not constr_disp) or (n_pos in good_base and i == 0) or \
                       (n_pos in good_targ and i == 1):
                        added_nodes.add(node)
                        g.node(str(id(node)), label=node.label, shape=n_shape,
                               style=sty,fillcolor=fill_col, color=node_col)
            for node in nodes:
                for arg in node.args:
                    if (node in added_nodes) and (arg in added_nodes):
                        g.edge(str(id(node)), str(id(arg)))

    if col == 'green': col_val = '0.33 '
    elif col == 'blue': col_val = '0.5 '

    for prob, (n1_ind, n2_ind) in alignments:
        prob = max(prob, 0) * 3
        dag.edge(str(id(nodes1[n1_ind])), str(id(nodes2[n2_ind])), 
                 constraint='false', dir='none', color=col_val + str(prob) + ' 1')

    dag.view()

def visualize_alignment_gen_sg(nodes1, nodes2, alignments, 
                               file_app='', col='green'):
    def check_in_al(node):
        return node in [a for al in alignments for a in al]

    def get_node_color(node, src):
        if check_in_al(node): return '0.33 1 0.35'
        elif src == 0: return '0.7 1 0.5'
        else: return '1 1 0.75'

    dag = Digraph(filename=sf.dataset_params['vis_data_loc'] + file_app +'_genproc')

    for i, nodes in enumerate([nodes1, nodes2]):
        gr_name = 'base' if i == 0 else 'target'
        # graph name must begin with 'cluster' for graphviz
        with dag.subgraph(name='cluster_' + gr_name) as g:
            g.attr(color='black')
            g.attr(label=gr_name)
            g.attr(style='invis')
            tsrt = reversed(topological_sort(nodes))
            for layer in tsrt:
                for node in layer:
                    n_shape = 'ellipse' if node.ordered else 'rectangle'
                    col_val = get_node_color(node, i)
                    g.node(str(id(node)), label=node.label, shape=n_shape,
                           color=col_val)
            for node in nodes:
                for arg in node.args:
                    col_val = get_node_color(node, i)
                    g.edge(str(id(node)), str(id(arg)), color=col_val)

    for node1, node2 in alignments:
        dag.edge(str(id(node1)), str(id(node2)), 
                 constraint='false', dir='none', 
                 color='0.33 1 1')

    dag.view()

def visualize_alignment_ps(nodes1, nodes2, alignments, file_app='', col='green'):
    dag = Digraph(filename=sf.dataset_params['vis_data_loc'] + file_app)
    dag.attr(nodesep='0.47')
    dag.attr(ranksep='0.35')
    #dag.attr(rankdir="LR")
    for i, nodes in enumerate([nodes1, nodes2]):
        tsrt = reversed(topological_sort(nodes))
        for layer in tsrt:
            for node in layer:
                n_shape = 'ellipse'# if node.ordered else 'rectangle'
                dag.node(str(id(node)), label=node.label, shape=n_shape)
        for node in nodes:
            for arg in node.args:
                dag.edge(str(id(node)), str(id(arg)))

    col_val = '0.33 '
    col_val = '0.6 '
    
    for prob, (n1_ind, n2_ind) in alignments:
        prob = max(prob, 0)
        dag.edge(str(id(nodes1[n1_ind])), str(id(nodes2[n2_ind])), 
                 constraint='false', dir='none', color=col_val + str(prob) + ' 1')
    dag.view()

def visualize_graphs(node_lsts, file_app=''):
    dag = Digraph(filename=sf.dataset_params['vis_data_loc'] + file_app)
    for i, nodes in enumerate(node_lsts):
        gr_name = 'node_lst_' + str(i)
        # graph name must begin with 'cluster' for graphviz
        with dag.subgraph(name='cluster_' + gr_name) as g:
            g.attr(color='black')
            #g.attr(label=gr_name)
            g.attr(style='invis')
            tsrt = reversed(topological_sort(nodes))
            for layer in tsrt:
                for node in layer:
                    n_shape = 'ellipse' if node.ordered else 'rectangle'
                    g.node(str(id(node)), label=node.label, shape=n_shape)
            for node in nodes:
                for arg in node.args:
                    g.edge(str(id(node)), str(id(arg)))
    dag.view()
