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


gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'

def record_ex_res(b_node_lst, t_node_lst, in_edges,
                  prob_results, struct_results, exp_range, t_len):

    tr_valid_edges, tr_match_score, tr_v = get_sm_from_mapping(in_edges,
                                                               gold_standard=True)

    with torch.no_grad():
        match_res = model.structural_match(b_node_lst, t_node_lst,
                                           output_k=True, k_encs=exp_range,
                                           hashable_syms=True)

    for res_i, results in enumerate([prob_results, struct_results]):
        for rng in results.keys():
            best_res = max(match_res[:rng], key=lambda x : x[res_i])
            ( mapping_prob, sm_score, mapping, 
              cand_infs, val_mapping, violations ) = best_res
            one_to_one_m, pc_m, ml_m, ov_m, tv_m = violations

            # get gold standard candidate inferences
            full_corrs = [(b_node_lst[b], t_node_lst[t]) for _, (b, t) in mapping]
            matched_b_nodes = [b_node_lst[b] for _, (b, _) in mapping]
            gs_cand_infs = set(get_cand_infs_from_corrs(full_corrs))
            base_cand_inf_nodes = set([b_node_lst[b] for _, b in cand_infs])
            not_cand_infs = set(b_node_lst).difference(base_cand_inf_nodes)
            not_cand_infs = not_cand_infs.difference(matched_b_nodes)
            # get cand inf metrics
            f_true_pos = len(gs_cand_infs.intersection(base_cand_inf_nodes))
            f_false_pos = len(base_cand_inf_nodes.difference(gs_cand_infs))
            f_false_neg = len(gs_cand_infs.difference(base_cand_inf_nodes))
            f_true_neg = len(not_cand_infs.difference(base_cand_inf_nodes))
            fci_prec, fci_rec, fci_f1, fci_acc, fci_tnr = calc_metrics(f_true_pos, 
                                                                       f_false_pos, 
                                                                       f_false_neg, 
                                                                       f_true_neg)

            ( match_scores, fci_precs, fci_recs, fci_f1s, fci_accs, fci_tnrs,
              fci_ncts, fci_pcts,
              one_to_one_violations, pc_violations, 
              mismatch_violations, degen_violations, err_dist,
              num_better, num_perfect, num_error_free, num_smt_sat ) = results[rng]
            
            if sum(violations) == 0:
                if sm_score == tr_match_score: num_perfect += 1
                if sm_score > tr_match_score: num_better += 1
                num_error_free += 1
            if sum(violations[:-2]) == 0:
                num_smt_sat += 1

            m_l = len(mapping) if len(mapping) > 0 else 1

            match_scores.append((sm_score, tr_match_score))
            fci_precs.append(fci_prec)
            fci_recs.append(fci_rec)
            fci_f1s.append(fci_f1)
            fci_accs.append(fci_acc)
            if len(not_cand_infs) > 0: fci_tnrs.append(fci_tnr)
            if len(not_cand_infs) > 0: fci_ncts.append(len(not_cand_infs))
            if len(gs_cand_infs) > 0: fci_pcts.append(len(gs_cand_infs))
            # one_to_one is divided by m_l * 2 because that is number of 
            # nodes violating one-to-one divided by total number of nodes
            one_to_one_violations.append((one_to_one_m, m_l * 2))
            pc_violations.append((pc_m, m_l))
            mismatch_violations.append((ml_m, m_l))
            degen_violations.append((ov_m, m_l))

            err_perc = round(tv_m / m_l, 2) * 100
            err_perc = int(err_perc)
            if not err_perc in err_dist: err_dist[err_perc] = 0
            err_dist[err_perc] += 1
            results[rng] = ( match_scores, fci_precs, fci_recs, fci_f1s, fci_accs,
                             fci_tnrs, fci_ncts, fci_pcts,
                             one_to_one_violations, pc_violations,
                             mismatch_violations, degen_violations, err_dist,
                             num_better, num_perfect, num_error_free, num_smt_sat )

    ( match_scores, fci_precs, fci_recs, fci_f1s, 
      fci_accs, fci_tnrs, fci_ncts, fci_pcts,
      one_to_one_violations, pc_violations,
      mismatch_violations, degen_violations, err_dist, num_better, num_perfect, 
      num_error_free, num_smt_sat ) = struct_results[exp_range]
    avg_amn_score = np.mean([x[0] for x in match_scores])
    avg_sme_score = np.mean([x[1] for x in match_scores])
    avg_eff = np.mean([x[0] / x[1] for x in match_scores])
    avg_fci_prec = np.mean(fci_precs)
    avg_fci_rec = np.mean(fci_recs)
    avg_fci_f1 = np.mean(fci_f1s)
    avg_fci_acc = np.mean(fci_accs)
    avg_fci_tnr = np.mean(fci_tnrs)
    avg_fci_nct = np.mean(fci_ncts)
    avg_fci_pct = np.mean(fci_pcts)
    avg_1to1 = np.mean([x[0] / x[1] for x in one_to_one_violations])
    avg_pc = np.mean([x[0] / x[1] for x in pc_violations])
    avg_mm = np.mean([x[0] / x[1] for x in mismatch_violations])
    avg_ov = np.mean([x[0] / x[1] for x in degen_violations])
    print('Test completion: ' + \
          str(round(len(match_scores) / t_len * 100, 2)) + '%')
    print('AMN match score: ' + str(avg_amn_score))
    print('SME match score: ' + str(avg_sme_score))
    print('Full match candidate inference precision: ' + str(avg_fci_prec))
    print('Full match candidate inference recall: ' + str(avg_fci_rec))
    print('Full match candidate inference f1: ' + str(avg_fci_f1))
    print('Full match candidate inference acc: ' + str(avg_fci_acc))
    print('Full match candidate inference true negative rate: ' + str(avg_fci_tnr))
    print('Average non-candidate inference nodes: ' + str(avg_fci_nct))
    print('Average candidate inference nodes: ' + str(avg_fci_pct))
    print('Total non-candidate inference problems: ' + str(len(fci_ncts)))
    print('Match effectiveness: ' + str(avg_eff))
    print('1-to-1 error rate: ' + str(avg_1to1))
    print('PC error rate: ' + str(avg_pc))
    print('Mismatch error rate: ' + str(avg_mm))
    print('Degenerate constant error rate: ' + str(avg_ov))
    print('Number total: ' + str(len(match_scores)))
    print('Number better: ' + str(num_better))
    print('Number perfect: ' + str(num_perfect))
    print('Number error-free: ' + str(num_error_free))
    print('Number SMT satisfying: ' + str(num_smt_sat))
    print('Error breakdown: ' + str(list(sorted(err_dist.items()))))
    print()

def record_final_res(te_results_file, prob_results, struct_results):
    with open(os.path.join(sf.dataset_params['results_dir'], 
                           te_results_file), 'w') as f:
        for res_i, results in enumerate([prob_results, struct_results]):
            if res_i == 0: f.write('MaxProb\n')
            else: f.write('MaxStruct\n')
            f.write('k,avg_sme, avg_amn,avg_eff,')
            f.write('fci_prec,fci_rec,fci_f1,fci_acc,fci_tnr,fci_nct,fci_pct,')
            f.write('fci_nct_tot,')
            f.write('avg_1to1,avg_pc,avg_mm,avg_degen,num_total,')
            f.write('num_better,num_perfect,num_error_free,num_smt_sat,')
            f.write('frac_better,frac_perfect,frac_error_free,frac_smt_sat,')
            f.write(','.join([str(i) for i in range(101)]))
            f.write('\n')
            for k, v in sorted(results.items()):
                ( match_scores, fci_precs, fci_recs, fci_f1s, 
                  fci_accs, fci_tnrs, fci_ncts, fci_pcts,
                  one_to_one_violations, pc_violations, 
                  mismatch_violations, degen_violations, err_dist,
                  num_better, num_perfect, num_error_free, num_smt_sat ) = v
                avg_amn_score = np.mean([x[0] for x in match_scores])
                avg_sme_score = np.mean([x[1] for x in match_scores])
                avg_eff = np.mean([x[0] / x[1] for x in match_scores])
                avg_fci_prec = np.mean(fci_precs)
                avg_fci_rec = np.mean(fci_recs)
                avg_fci_f1 = np.mean(fci_f1s)
                avg_fci_acc = np.mean(fci_accs)
                avg_fci_tnr = np.mean(fci_tnrs)
                avg_fci_nct = np.mean(fci_ncts)
                avg_fci_pct = np.mean(fci_pcts)
                avg_1to1 = np.mean([x[0] / x[1] for x in one_to_one_violations])
                avg_pc = np.mean([x[0] / x[1] for x in pc_violations])
                avg_mm = np.mean([x[0] / x[1] for x in mismatch_violations])
                avg_ov = np.mean([x[0] / x[1] for x in degen_violations])
                frac_better = num_better / len(match_scores)
                frac_perf = num_perfect / len(match_scores)
                frac_ef = num_error_free / len(match_scores)
                frac_sm_s = num_smt_sat / len(match_scores)
                metrs = [k, avg_sme_score, avg_amn_score, avg_eff,
                         avg_fci_prec,avg_fci_rec,avg_fci_f1,
                         avg_fci_acc,avg_fci_tnr,avg_fci_nct,avg_fci_pct,
                         len(fci_ncts),
                         avg_1to1, avg_pc, avg_mm, avg_ov,
                         len(match_scores), 
                         num_better, num_perfect, num_error_free, num_smt_sat,
                         frac_better, frac_perf, frac_ef, frac_sm_s]
                ext_err_dist = [0 for _ in range(101)]
                for k, v in sorted(err_dist.items()): 
                    ext_err_dist[k] = v
                metrs.extend(ext_err_dist)
                write_str = ','.join([str(x) for x in metrs])
                f.write(write_str + '\n')

def init_results_dict(exp_range):
    r_d = dict([(i + 1, [[], [], [], [], [], [], [], [], [], [], [], [],{},0,0,0,0]) 
                for i in range(exp_range)])
    return r_d

def eval_synthetic_exp(model):

    exp_range = 16
    test_set = pkl.load(open(sf.dataset_params['te_data_loc'], 'rb'))
    match_scores, one_to_one_violations, pc_violations = [], [], []
    mismatch_violations, degen_violations = [], []
    num_perfect, num_error_free = 0, 0
    
    prob_results = init_results_dict(exp_range)
    struct_results = init_results_dict(exp_range)

    for test_ex in test_set:
        b_node_lst, t_node_lst, in_edges, _ = pkl.load(open(test_ex + '.pt', 'rb'))
        record_ex_res(b_node_lst, t_node_lst, in_edges,
                      prob_results, struct_results, exp_range, len(test_set))

    te_results_file = sf.model_param_str + '_te_sm_synthetic_results.csv'
    record_final_res(te_results_file, prob_results, struct_results)


def eval_sm_exp(model):
    max_arity = model.graph_encoder.max_arity
    exp_range = 16

    test_dirs = ['moraldm', 'oddity', 'geometry']#,'thermo']
    
    for test_dir in test_dirs:
        eval_sm_exp_on_domain(test_dir, max_arity, exp_range)

def eval_sm_exp_on_domain(focus_test_dir, max_arity, exp_range):

    all_files = []
    parsed_dir = sf.dataset_params['parsed_sm_tests_loc']
    for test_dir in os.listdir(parsed_dir):
        if test_dir != focus_test_dir: continue
        for test_file in os.listdir(os.path.join(parsed_dir, test_dir)):
            all_files.append(os.path.join(parsed_dir, test_dir, test_file))

    prob_results = init_results_dict(exp_range)
    struct_results = init_results_dict(exp_range)

    for test_file in all_files:
        print('Running: ' + test_file)
        with open(test_file, 'r') as f:
            all_lines = ''.join(list(f.readlines()))
            base_str, targ_str, matched_str = all_lines.split('+++++++++++')
            if (not (base_str and targ_str and matched_str)) or \
               not all(any(c.isalnum() for c in str_src) 
                       for str_src in [base_str, targ_str, matched_str]):
                print('\n\n\n\nSKIPPING FILE: ' + test_file + '\n\n\n\n')
                continue
        
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

        record_ex_res(b_node_lst, t_node_lst, in_edges,
                      prob_results, struct_results, exp_range, len(all_files))

    te_results_file = sf.model_param_str + '_te_sm_' + focus_test_dir + \
                      '_results.csv'
    record_final_res(te_results_file, prob_results, struct_results)



if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Test neural analogy module')
    parser.add_argument('--use_prob',help='Rerun for probability maximization')
    args = parser.parse_args()

    use_prob = False
    if args.use_prob:
        assert args.use_prob in ['True', 'False']
        use_prob = args.use_prob == 'True'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #epoch_loc = sf.dataset_params['best_model_loc']
        epoch_loc = sf.dataset_params['latest_model_loc']
        model = torch.load(epoch_loc, map_location=torch.device('cpu'))
        switch_device(model, torch.device('cpu'))
    model.eval()

    eval_res = eval_synthetic_exp(model) 
    eval_res = eval_sm_exp(model)
