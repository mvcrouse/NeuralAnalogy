# python imports
import signal, time, random, itertools, copy, math, sys, os
import pickle as pkl
import argparse as ap
import warnings
# numpy imports
import numpy as np
# torch imports
import torch
import torch.nn as nn
# code imports
import settings_file as sf
from node_classes import *
from generate_dataset import *
from utilities import *
from sm_network import *
from parse_input_forms import *

# atom analogy strings
solar_system_str = '(CAUSES (AND (ATTRACTS sun planet) (GREATER (MASS sun) (MASS planet))) (REVOLVES-AROUND planet sun))'
solar_system_str += '(YELLOW sun)'
solar_system_str += '(GREATER (TEMPERATURE sun) (TEMPERATURE planet))'

atom_str = '(GREATER (MASS nucleus) (MASS electron))'
atom_str += '(ATTRACTS nucleus electron)'
atom_str += '(REVOLVES-AROUND electron nucleus)'

# criss cross strings
criss_cross_base_str = '(AND a (REL b c)) (AND (REL c d) e)'
criss_cross_targ_str = '(AND a (REL b c)) (AND (REL c d) e)'

incr_base_str = '(AND (REL a b) (REL c d))'
incr_targ_str = '(AND (REL a b) (REL c d))'

def visualize_hol_theory():

    base_str = solar_system_str

    orig_graph = parse_s_expr_to_gr(base_str)
    const_graph = parse_s_expr_to_gr(base_str)

    orig_node_lst = set()
    for top_node in orig_graph:
        orig_node_lst = orig_node_lst.union(top_node.dep_subgraph())
    orig_node_lst = list(orig_node_lst)

    visualize_graphs([orig_node_lst], 'orig_solar_system')
    input('Press \'Enter\' to continue...')

    const_node_lst = set()
    for top_node in const_graph:
        const_node_lst = const_node_lst.union(top_node.dep_subgraph())
    const_node_lst = list(const_node_lst)
    for node in const_node_lst:
        if not node.args: node.label = 'const'
    
    visualize_graphs([const_node_lst], 'const_solar_system')
    input('Press \'Enter\' to continue...')

def visualize_complete_gr():
    base_str = '(! A (& (= (+ A 0) (+ 0 A)) (= (+ A 0) A)))'
    targ_str = '(! A (? B (= (+ A B) 0)))'
    
    b_graph = parse_s_expr_to_gr(base_str)
    t_graph = parse_s_expr_to_gr(targ_str)
       
    b_node_lst, t_node_lst = set(), set()
    for top_node in b_graph:
        b_node_lst = b_node_lst.union(top_node.dep_subgraph())
    for top_node in t_graph:
        t_node_lst = t_node_lst.union(top_node.dep_subgraph())
    b_node_lst, t_node_lst = list(b_node_lst), list(t_node_lst)
    
    pred_edges = [(1.0, (i, j)) for i in range(len(b_node_lst)) 
                  for j in range(len(t_node_lst))
                  if (b_node_lst[i].label == t_node_lst[j].label) or \
                  (b_node_lst[i].label in ['A', 'B'] and \
                   t_node_lst[j].label in ['A', 'B'])]

    visualize_alignment_ps(b_node_lst, t_node_lst, pred_edges)

    input('DONE')
    input('DONE')
    input('DONE')

def visualize_gen_examples(model):
    for j in range(0, 1000):
        gr_id = os.path.join(sf.data_dir, 'gen_graph_pair_' + str(j) + '.pt')
        b_node_lst, t_node_lst, in_edges, cand_infs = pkl.load(open(gr_id, 'rb'))

        true_edges = [(1.0, (position(x, b_node_lst),
                             position(y, t_node_lst)))
                      for x, y in in_edges]
        true_cand_infs = [(1.0, (position(x, b_node_lst)))
                          for x in cand_infs]
        tr_valid_edges, tr_match_score, _ = get_sm_from_mapping(in_edges)

        match_res = model.structural_match(b_node_lst, t_node_lst,
                                           score_by_prob=False)
        match_prob, match_score, pred_edges, pred_cand_infs, _, _ = match_res

        b_edges = set([e for _, e in pred_edges])
        tr_edges = set([e for _, e in true_edges])

        print('Shared predictions: ' + str(len(b_edges.intersection(tr_edges))))
        print('Wrong predictions: ' + str(len(b_edges.difference(tr_edges))))
        print('Missed predictions: ' + str(len(tr_edges.difference(b_edges))))
        print('Predicted match SME score: ' + str(match_score))
        print('Target match SME score: ' + str(tr_match_score))
        print()

        visualize_alignment(b_node_lst, t_node_lst, pred_edges, 
                            cand_infs=pred_cand_infs,
                            file_app='best_h_pred', col='green')
        visualize_alignment(b_node_lst, t_node_lst, true_edges, 
                            cand_infs=true_cand_infs,
                            file_app='true', col='blue')
        #visualize_alignment_gen_sg(all_base_nodes, all_targ_nodes, in_edges,
        #                           file_app='true', col='blue')

        print('Predicted alignment vs. true alignment...')
        input('Press \'Enter\' to continue...')

def visualize_criss_cross_alignment(model):

    base_str = criss_cross_base_str
    targ_str = criss_cross_targ_str

    b_graph = parse_s_expr_to_gr(base_str)
    t_graph = parse_s_expr_to_gr(targ_str)
       
    b_node_lst, t_node_lst = set(), set()
    for top_node in b_graph:
        b_node_lst = b_node_lst.union(top_node.dep_subgraph())
    for top_node in t_graph:
        t_node_lst = t_node_lst.union(top_node.dep_subgraph())
    b_node_lst, t_node_lst = list(b_node_lst), list(t_node_lst)
        
    match_res = model.structural_match(b_node_lst, t_node_lst)
    match_prob, match_score, pred_edges, cand_infs, _, _ = match_res
    
    for prob, (l, r) in pred_edges:
        print(b_node_lst[l])
        print(t_node_lst[r])
        print(prob)
        print()

    #pred_edges = [(1.0, edge) for edge in pred_edges]
    #pred_edges = []

    visualize_alignment(b_node_lst, t_node_lst, pred_edges)
    input('Press \'Enter\' to continue...')

def visualize_custom_example(model):

    max_arity = model.graph_encoder.max_arity

    def lf_enter(src):
        print('\nPlease enter expressions for the ' + src + ' at the prompt...')
        print('Type END to finish...')
        ret_strs = []
        while True:
            user_inp = input('>>> ')
            try:
                if user_inp == 'END':
                    return ret_strs
                else:
                    p_graph = parse_s_expr_to_gr(user_inp, conv_isa=True, 
                                                 max_arity=max_arity)
                    ret_strs.append(user_inp)
            except:
                print('Syntax error detected, please try again...')
        

    b_strs = lf_enter('Base')
    t_strs = lf_enter('Target')
    base_str, targ_str = ' '.join(b_strs), ' '.join(t_strs)
    b_graph = parse_s_expr_to_gr(base_str, conv_isa=True, max_arity=max_arity)
    t_graph = parse_s_expr_to_gr(targ_str, conv_isa=True, max_arity=max_arity)
    
    b_node_lst, t_node_lst = set(), set()
    for top_node in b_graph:
        b_node_lst = b_node_lst.union(top_node.dep_subgraph())
    for top_node in t_graph:
        t_node_lst = t_node_lst.union(top_node.dep_subgraph())
    b_node_lst, t_node_lst = list(b_node_lst), list(t_node_lst)

    dk_encs = 16
    if len(b_node_lst) < 20 and len(t_node_lst) < 20: dk_encs = 32

    match_res = model.structural_match(b_node_lst, t_node_lst,
                                       k_encs=dk_encs)
    match_prob, match_score, pred_edges, cand_infs, _, _ = match_res

    visualize_alignment(b_node_lst, t_node_lst, pred_edges, cand_infs)
    print('Alignment predicted by best model')
    input('Press \'Enter\' to continue...')

def visualize_atom_alignment(model):

    base_str = solar_system_str
    targ_str = atom_str

    b_graph = parse_s_expr_to_gr(base_str, conv_isa=True)
    t_graph = parse_s_expr_to_gr(targ_str, conv_isa=True)

    b_node_lst, t_node_lst = set(), set()
    for top_node in b_graph:
        b_node_lst = b_node_lst.union(top_node.dep_subgraph())
    for top_node in t_graph:
        t_node_lst = t_node_lst.union(top_node.dep_subgraph())
    b_node_lst, t_node_lst = list(b_node_lst), list(t_node_lst)
    
    match_res = model.structural_match(b_node_lst, t_node_lst)
    match_prob, match_score, pred_edges, cand_infs, _, _ = match_res
    
    for prob, (l, r) in pred_edges:
        print(b_node_lst[l])
        print(t_node_lst[r])
        print(prob)
        print()

    #visualize_graphs([b_node_lst])
    #input('--')
    #visualize_graphs([t_node_lst])
    #input('--')

    visualize_alignment(b_node_lst, t_node_lst, pred_edges, cand_infs)
    print('Alignment predicted by best model')
    input('Press \'Enter\' to continue...')


def visualize_sm_corpus_alignment(model, focus_test_dir, up_to=3):
    max_arity = model.graph_encoder.max_arity
    exp_range = 16
    all_files = []
    parsed_dir = sf.dataset_params['parsed_sm_tests_loc']
    for test_dir in os.listdir(parsed_dir):
        if test_dir != focus_test_dir: continue
        for test_file in os.listdir(os.path.join(parsed_dir, test_dir)):
            all_files.append(os.path.join(parsed_dir, test_dir, test_file))

    for test_file in all_files:#[:up_to]:
        
        with open(test_file, 'r') as f:
            all_lines = ''.join(list(f.readlines()))
            base_str, targ_str, matched_str = all_lines.split('+++++++++++')

        b_graph = parse_s_expr_to_gr(base_str, conv_isa=True, max_arity=max_arity)
        t_graph = parse_s_expr_to_gr(targ_str, conv_isa=True, max_arity=max_arity)
        m_graph = parse_s_expr_to_gr(matched_str,conv_isa=True,max_arity=max_arity)

        b_node_lst, t_node_lst, m_node_lst = set(), set(), set()
        for top_node in b_graph:
            b_node_lst = b_node_lst.union(top_node.dep_subgraph())
        for top_node in t_graph:
            t_node_lst = t_node_lst.union(top_node.dep_subgraph())
        for top_node in m_graph:
            m_node_lst = m_node_lst.union(top_node.dep_subgraph())

        b_node_lst = list(b_node_lst)
        t_node_lst = list(t_node_lst)
        m_node_lst = list(m_node_lst)
        
        #if any('_EVAL' in n.label or '_arg' in n.label 
        #       for n in b_node_lst + t_node_lst): continue
        #if len(b_node_lst) + len(t_node_lst) > 100: continue
        print(focus_test_dir)

        in_edges = []
        for m_node in m_node_lst:
            if m_node.label == 'matchBetween':
                m_a1, m_a2 = str(m_node.args[0]), str(m_node.args[1])
                try:
                    b_ind = position(m_a1, b_node_lst, key=lambda x : str(x))
                    t_ind = position(m_a2, t_node_lst, key=lambda x : str(x))
                    b, t = b_node_lst[b_ind], t_node_lst[t_ind]
                    if b and t: in_edges.append((b, t))
                except ValueError as e: pass
                
        for m_node in m_node_lst:
            if m_node.label == 'matchBetween':
                a1, a2 = m_node.args
                check_pairs = [(a1, a2)]
                arg1_label_chk = a1.label + '_arg'
                arg2_label_chk = a1.label + '_arg'
                while check_pairs:
                    l, r = check_pairs.pop()
                    if len(l.args) != max_arity: continue
                    if len(r.args) != max_arity: continue
                    if arg1_label_chk not in l.args[max_arity - 1].label: continue
                    if arg2_label_chk not in r.args[max_arity - 1].label: continue
                    new_pair = (l.args[max_arity - 1], r.args[max_arity - 1])
                    check_pairs.append(new_pair)
                    if not new_pair in in_edges: 
                        try:
                            b_ind = position(new_pair[0], b_node_lst, 
                                             key=lambda x:str(x))
                            t_ind = position(new_pair[1], t_node_lst, 
                                             key=lambda x:str(x))
                            b, t = b_node_lst[b_ind], t_node_lst[t_ind]
                            if b and t: in_edges.append((b, t))
                        except ValueError as e: pass

        true_edges = [(1.0, (position(x, b_node_lst),
                             position(y, t_node_lst)))
                      for x, y in in_edges]

        match_res = model.structural_match(b_node_lst, t_node_lst)
        match_prob, match_score, pred_edges, cand_infs, _, violations = match_res

        b_node_lst_wp = pf.revert_arg_ext(b_node_lst, w_pos=True)
        t_node_lst_wp = pf.revert_arg_ext(t_node_lst, w_pos=True)
        b_node_lst = [b for _, _, b in b_node_lst_wp]
        t_node_lst = [b for _, _, b in t_node_lst_wp]
        b_map = dict([(old_pos, new_pos) for old_pos, new_pos, _ in b_node_lst_wp])
        t_map = dict([(old_pos, new_pos) for old_pos, new_pos, _ in t_node_lst_wp])
        pred_edges = [(p, (b_map[b], t_map[t])) for p, (b, t) in pred_edges 
                      if b in b_map and t in t_map]
        true_edges = [(p, (b_map[b], t_map[t])) for p, (b, t) in true_edges 
                      if b in b_map and t in t_map]
        cand_infs = [(p, b_map[b]) for p, b in cand_infs if b in b_map]
        
        b_edges = set([e for _, e in pred_edges])
        tr_edges = set([e for _, e in true_edges])

        tr_valid_edges, tr_match_score, _ = get_sm_from_mapping(in_edges)
        if match_score < tr_match_score or sum(violations) != 0: continue
        if len(pred_edges) > 20: continue

        print('Shared predictions: ' + str(len(b_edges.intersection(tr_edges))))
        print('Different predictions: ' + str(len(b_edges.difference(tr_edges))))
        print('Missed predictions: ' + str(len(tr_edges.difference(b_edges))))
        print('Predicted match SME score: ' + str(match_score))
        print('Target match SME score: ' + str(tr_match_score))
        print()

        visualize_alignment(b_node_lst, t_node_lst, pred_edges, 
                            cand_infs=cand_infs,file_app='best_h_pred',col='green',
                            constr_disp=True)
        visualize_alignment(b_node_lst, t_node_lst, true_edges,
                            file_app='true', col='blue')

        print('Predicted alignment vs. true alignment...')
        input('Press \'Enter\' to continue...')


def visualize_incremental_alignment(model):
    base_str = incr_base_str
    targ_str = incr_targ_str

    #incr_base_str = '(AND (REL a b) (REL c d))'
    #incr_targ_str = '(AND (REL a b) (REL c d))'

    b_graph = parse_s_expr_to_gr(base_str)
    t_graph = parse_s_expr_to_gr(targ_str)

    b_node_lst, t_node_lst = set(), set()
    for top_node in b_graph:
        b_node_lst = b_node_lst.union(top_node.dep_subgraph())
    for top_node in t_graph:
        t_node_lst = t_node_lst.union(top_node.dep_subgraph())
    b_node_lst, t_node_lst = list(b_node_lst), list(t_node_lst)

    pairings = []
    pairings.append({})
    pairings.append({ 'a' : 'a' })
    pairings.append({ 'a' : 'c' })
    for pair_together in pairings:
        req_corrs = []
        for b in b_node_lst:
            for t in t_node_lst:
                if b.label in pair_together and pair_together[b.label] == t.label:
                    req_corrs.append((b, t))

        with torch.no_grad():
            match_res = model.structural_match(b_node_lst, t_node_lst, 
                                               req_corrs=req_corrs)
            mapping_prob, sm_score, pred_edges, cand_infs, _, violations = match_res
            one_to_one_m, pc_m = violations

        for prob, (l, r) in pred_edges:
            print(b_node_lst[l])
            print(t_node_lst[r])
            print(prob)
            print()

        print(sm_score)
        print(violations)
        print()

        visualize_alignment(b_node_lst, t_node_lst, pred_edges)
        print('Alignment predicted by best model')
        print('Forced alignment: ' + str(pair_together))
        input('Press \'Enter\' to continue...')

def main():
    parser = ap.ArgumentParser(description='Visualize analogy model outputs')
    parser.add_argument('--domain', help='Visualize specified domain')
    args = parser.parse_args()

    domain = 'atom'
    if args.domain:
        supp_domains = ['synthetic', 'moraldm', 'geometry', 
                        'oddity', 'atom', 'custom']
        assert args.domain in supp_domains, 'Unsupported domain, must be one of ' + \
            ', '.join(supp_domains)
        domain = args.domain

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #model_loc = sf.dataset_params['best_model_loc']
        model_loc = sf.dataset_params['latest_model_loc']
        model = torch.load(model_loc, map_location=torch.device('cpu'))
        switch_device(model, torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        #visualize_complete_gr()
        #visualize_hol_theory()
        if domain == 'atom':
            visualize_atom_alignment(model)
        if domain == 'moraldm':
            visualize_sm_corpus_alignment(model, 'moraldm')
        if domain == 'geometry':
            visualize_sm_corpus_alignment(model, 'geometry')
        if domain == 'oddity':
            visualize_sm_corpus_alignment(model, 'oddity')
        if domain == 'synthetic':
            visualize_gen_examples(model)
        if domain == 'custom':
            visualize_custom_example(model)
        #visualize_incremental_alignment(model)
        #visualize_criss_cross_alignment(model)
        #visualize_small_ex_alignment(model)

if __name__ == '__main__':
    main()
