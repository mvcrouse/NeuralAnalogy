# python imports
import signal, time, random, itertools, copy, math, os, string
import pickle as pkl
# numpy imports
import numpy as np
# torch imports
import torch
from torch.utils import data
# code imports
import settings_file as sf
from node_classes import *
from parse_input_forms import *
from graph_encoder import *
from utilities import *
from dataset import *

#####################
# dataset generators
#####################

def gen_dataset(graph_encoder, dataset_params):
    num_val_ex = dataset_params['num_val_ex']
    num_te_ex = dataset_params['num_te_ex']
    data_dir = dataset_params['data_dir']
    tr_exs, val_exs, te_exs = [], [], []
    all_labels = set()
    prev_disp = 0
    num_all_ex = num_val_ex + num_te_ex
    b_expr_ct, t_expr_ct, b_ent_ct, t_ent_ct = 0, 0, 0, 0
    b_rel_ct, t_rel_ct, ms_ct = 0, 0, 0
    for i in range(num_all_ex):
        if round(100 * i / num_all_ex) != prev_disp:
            prev_disp = round(100 * i / num_all_ex)
            print('Dataset ' + str(round(prev_disp)) + '% generated...')
        # get base and target

        gr_enc_tries = 15
        for enc_attempt in range(gr_enc_tries + 1):
            try:
                graph_ex = gen_valid_graph_example(dataset_params, 
                                                   gr_encoder=graph_encoder)
                lst_ex = graph_encoder.listify_graph_example(graph_ex)
                break
            except ValueError as e:
                if enc_attempt == gr_enc_tries and 'no samples' in str(e): raise e
                
        b_expr_ct += len([x for x in graph_ex[0] if x.args])
        t_expr_ct += len([x for x in graph_ex[1] if x.args])
        b_ent_ct += len([x for x in graph_ex[0] if not x.args])
        t_ent_ct += len([x for x in graph_ex[1] if not x.args])
        b_rel_ct += len(set([x.label for x in graph_ex[0] if x.args]))
        t_rel_ct += len(set([x.label for x in graph_ex[1] if x.args]))
        ms_ct += len(graph_ex[2])

        # get labels from generated graphs
        for node in graph_ex[0] + graph_ex[1]: all_labels.add(node.label)

        # graph obj file
        gr_id = os.path.join(data_dir, 'gen_graph_pair_' + str(i))
        lst_id = os.path.join(data_dir, 'gen_list_form_' + str(i))
        pkl.dump(graph_ex, open(gr_id + '.pt', 'wb'))
        torch.save(lst_ex, lst_id + '.pt')

        # decide whether training example or validation
        if i < num_val_ex:
            val_exs.append(lst_id)
        else:
            te_exs.append(gr_id)

    stats = (round(b_expr_ct / num_all_ex, 2), 
             round(t_expr_ct / num_all_ex, 2),
             round(b_ent_ct / num_all_ex, 2), 
             round(t_ent_ct / num_all_ex, 2),
             round(b_rel_ct / num_all_ex, 2), 
             round(t_rel_ct / num_all_ex, 2), 
             round(ms_ct / num_all_ex, 2))

    return Dataset(val_exs), te_exs, stats

def gen_valid_graph_example(dataset_params, try_ct=32, gr_encoder=None,
                            req_cand_infs=None):
    if req_cand_infs == None:
        req_cand_infs = random.randint(0, 100) < dataset_params['cand_inf_chance']
    try:
        graph_ex = gen_random_graph_example(dataset_params, 
                                            req_cand_infs=req_cand_infs)
        while not graph_ex:
            graph_ex = gen_random_graph_example(dataset_params,
                                                req_cand_infs=req_cand_infs)
        if gr_encoder:
            gr_encoder.listify_graph_example(graph_ex)
        return graph_ex
    except Exception as e:
        if 'KeyboardInterrupt' in str(e): raise e
        if try_ct > 0: return gen_valid_graph_example(dataset_params, try_ct - 1,
                                                      req_cand_infs=req_cand_infs)
        else: raise ValueError('Could not generate a valid DAG')

def gen_random_graph_example(dataset_params, req_cand_infs=False):
    min_matched = dataset_params['min_matched']
    max_matched = dataset_params['max_matched']
    min_unmatched = dataset_params['min_unmatched']
    max_unmatched = dataset_params['max_unmatched']
    min_duplicate = dataset_params['min_duplicate']
    max_duplicate = dataset_params['max_duplicate']

    # 
    obj_dict = {}

    # shared subgraphs
    num_shared = random.randint(min_matched, max_matched)
    shared_graphs = [gen_random_dag(dataset_params) for _ in range(num_shared)]
    shared_nodes = [node for graph in shared_graphs for node in graph]
    shared_nodes = coalesce_graph(shared_nodes, obj_dict)

    # unshared subgraphs
    num_base_unshared = random.randint(min_unmatched, max_unmatched)
    unshared_base_graphs = [gen_random_dag(dataset_params) 
                            for _ in range(num_base_unshared)]
    unshared_base_nodes = [node for graph in unshared_base_graphs for node in graph]
    unshared_base_nodes = coalesce_graph(unshared_base_nodes, obj_dict)

    num_targ_unshared = random.randint(min_unmatched, max_unmatched)
    unshared_targ_graphs = [gen_random_dag(dataset_params) 
                            for _ in range(num_targ_unshared)]
    unshared_targ_nodes = [node for graph in unshared_targ_graphs for node in graph]
    unshared_targ_nodes = coalesce_graph(unshared_targ_nodes, obj_dict)

    mixed_base_nodes = list(set(unshared_base_nodes + shared_nodes))
    mixed_base_nodes = coalesce_graph(mixed_base_nodes, obj_dict)
    mixed_targ_nodes = list(set(unshared_targ_nodes + shared_nodes))
    mixed_targ_nodes = coalesce_graph(mixed_targ_nodes, obj_dict)

    if max_duplicate > 0:
        dup_ct = random.randint(min_duplicate, max_duplicate)
        new_shared_nodes = []
        for _ in range(dup_ct):
            new_shared_nodes.extend(copy.deepcopy(shared_nodes))
        for i in range(len(new_shared_nodes)):
            nsn = new_shared_nodes[i]
            if not nsn.args:
                nsn.label = gen_random_node_label(dataset_params, nsn)
        mixed_base_nodes = new_shared_nodes + mixed_base_nodes
        mixed_targ_nodes = new_shared_nodes + mixed_targ_nodes

    # now constructing an overarching graph joining the distinct subgraphs
    bg_nodes = gen_random_dag(dataset_params, init_node_lst=mixed_base_nodes)
    bg_nodes = coalesce_graph(bg_nodes, obj_dict)

    tg_nodes = gen_random_dag(dataset_params, init_node_lst=mixed_targ_nodes)
    tg_nodes = coalesce_graph(tg_nodes, obj_dict)

    # the strategy to train our system is to feed it entire subgraph at a time
    in_edges, added = [], set()
    in_both = set(bg_nodes).intersection(tg_nodes)
    for node in sorted(in_both, key=lambda x : len(x.dependencies()), reverse=True):
        if node in added: continue
        add_q = [node]
        while add_q:
            new_q = []
            for add_el in add_q:
                for arg in add_el.args:
                    if arg in added: continue
                    added.add(arg)
                    new_q.append(arg)
                # getting positions of edges
                in_edges.append((position(add_el, bg_nodes), 
                                 position(add_el, tg_nodes)))
                added.add(node)
            add_q = new_q
    
    bs_in_inds, tg_in_inds = [b for b, _ in in_edges], [t for _, t in in_edges]
    all_base_nodes = copy.deepcopy(bg_nodes)
    all_targ_nodes = copy.deepcopy(tg_nodes)
    for node in all_base_nodes:
        node.parents = [p for p in node.parents if p in all_base_nodes]
        if not node.args: node.label = 'const'
        
    for node in all_targ_nodes:
        node.parents = [p for p in node.parents if p in all_targ_nodes]
        if not node.args: node.label = 'const'

    # replacing positions with actual nodes
    in_edges = [(all_base_nodes[x], all_targ_nodes[y]) for x, y in in_edges]

    str_in_edges = [(str(b), str(t)) for b, t in in_edges]
    for b_node in all_base_nodes:
        for t_node in all_targ_nodes:
            if str(b_node) == str(t_node) and b_node.args and \
               not (str(b_node), str(t_node)) in str_in_edges:
                return None

    # removing degenerate constants
    rem_edges = set()
    b_ie = [b for b, _ in in_edges]
    t_ie = [t for _, t in in_edges]
    for b, t in in_edges:
        if (b.args and t.args): continue
        if not (any(p in b_ie for p in b.parents) and \
                any(p in t_ie for p in t.parents)):
            rem_edges.add((b, t))
    in_edges = [edge for edge in in_edges if not edge in rem_edges]
    if not in_edges: return None

    all_base_nodes = [b for b in all_base_nodes
                      if ((not b.args) and b.parents) or b.args]
    all_targ_nodes = [t for t in all_targ_nodes
                      if ((not t.args) and t.parents) or t.args]

    random.shuffle(all_base_nodes)
    random.shuffle(all_targ_nodes)

    #random.shuffle(in_edges)
    in_edges = sorted(in_edges, key=lambda x : len(x[0].dependencies()),
                      reverse=True)

    base_cand_infs = get_cand_infs_from_corrs(in_edges)

    if req_cand_infs:
        # validate there are nodes that are not candidate inferences
        matched_b_nodes = [b for b, _ in in_edges]
        not_cand_infs = set(all_base_nodes).difference(base_cand_infs)
        not_cand_infs = not_cand_infs.difference(matched_b_nodes)
        assert base_cand_infs and not_cand_infs, 'Bad example for candidate inferences'

    ret_tuple = [all_base_nodes, all_targ_nodes, in_edges, base_cand_infs]

    return ret_tuple

#################
# DAG generators
#################

def gen_random_dag(dataset_params, init_node_lst=None):
    min_ranks = dataset_params['min_ranks']
    max_ranks = dataset_params['max_ranks']
    rank_specs = dataset_params['rank_specs']
    max_arity = dataset_params['max_arity']
    ord_chance = dataset_params['ordered_chance']
    if init_node_lst == None: init_node_lst, node_lst = [], []
    else: node_lst = init_node_lst + []
    graph_ranks = random.randint(min_ranks, max_ranks)
    for i in range(graph_ranks):
        min_per_rank, max_per_rank = rank_specs[i]
        nodes_at_rank = random.randint(min_per_rank, max_per_rank)
        curr_nodes = [Node('') for _ in range(nodes_at_rank)]
        for curr_node in curr_nodes:
            arity = random.randint(0, min(max_arity, len(node_lst)))
            if arity > 0:
                args = np.random.choice(node_lst, size=arity, replace=False)
                for arg in args:
                    curr_node.args.append(arg)
                    arg.parents.append(curr_node)
                curr_node.type_of = Node.pred_type
        node_lst = curr_nodes + node_lst
    # node_lst has the highest ranked nodes first, so we'll reverse here to 
    # make sure we can build compositional functions
    f_nodes = []
    for node in reversed(node_lst):
        if node in init_node_lst:
            f_nodes.append(node)
        else:
            f_nodes.append(change_node_randomly(dataset_params, node))
    f_nodes = list(reversed(f_nodes))
    return f_nodes

def change_node_randomly(dataset_params, node):
    ordered_chance = dataset_params['ordered_chance']
    if node.label != '': return node
    if node.args: is_ord = (random.randint(0,100) < ordered_chance)
    else: is_ord = True
    node.ordered = is_ord
    if node.args and node.parents and \
       all(a.type_of in [Node.func_type, Node.const_type] for a in node.args):
        func_chance = dataset_params['func_chance']
        if random.randint(0,100) < func_chance: node.type_of = Node.func_type
        else: node.type_of = Node.pred_type
    n_label = gen_random_node_label(dataset_params, node)
    node.label = n_label
    return node

def gen_random_node_label(dataset_params, node):
    if node.args: n_label = 'ord_' if node.ordered else 'unord_'
    else: n_label = ''
    up_to = dataset_params['nodes_bprop'][node.key_form()]
    if node.type_of == Node.pred_type: n_label = 'rel_'
    elif node.type_of == Node.func_type: n_label = 'func_'
    elif node.type_of == Node.const_type: n_label = 'const_'
    n_label += random.choice(string.ascii_letters[:up_to])
    #n_label += '_' + str(random.randint(0, 1))
    n_label += '_' + str(len(node.args))
    #n_label = n_label.lower()
    return n_label


#################
# Main
#################

if __name__ == '__main__':
    print('\nGenerating structural match training / validation / test sets...\n')
    gr_encoder = GraphEncoder(**sf.dataset_params)
    pkl.dump(gr_encoder, open(sf.dataset_params['encoder_loc'], 'wb'))
    str_val, str_te, stats = gen_dataset(gr_encoder, sf.dataset_params)
    print(stats)
    pkl.dump(str_val, open(sf.dataset_params['val_data_loc'], 'wb'))
    pkl.dump(str_te, open(sf.dataset_params['te_data_loc'], 'wb'))
