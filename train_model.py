# python imports
import signal, time, random, itertools, copy, math, sys, gc
import pickle as pkl
import argparse as ap
# numpy imports
import numpy as np
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
# code imports
import settings_file as sf
from node_classes import *
from generate_dataset import *
from utilities import *
from sm_network import *

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Train neural analogy module')
    parser.add_argument('--use_prior_model', help='Use stored existing model')
    parser.add_argument('--use_cos_sim', help='Use cosine similarity')
    parser.add_argument('--test_val', help='Test validation performance')

    args = parser.parse_args()

    use_prior_model = False
    if args.use_prior_model:
        assert args.use_prior_model in ['True', 'False']
        use_prior_model = args.use_prior_model == 'True'

    use_cos_sim = True
    if args.use_cos_sim:
        assert args.use_cos_sim in ['True', 'False']
        use_cos_sim = args.use_cos_sim == 'True'
    sf.dataset_params['cos_sim'] = use_cos_sim

    test_val = False
    if args.test_val:
        assert args.test_val in ['True', 'False']
        test_val = args.test_val == 'True'

    # get validation
    validation_generator = []
    if test_val:
        validation_set = pkl.load(open(sf.dataset_params['val_data_loc'], 'rb'))
        validation_generator = data.DataLoader(validation_set, 
                                               collate_fn=collate_graph_exs,
                                               **sf.test_params)

    # get encoder to store with model
    graph_encoder = pkl.load(open(sf.dataset_params['encoder_loc'], 'rb'))
    sf.dataset_params['graph_encoder'] = graph_encoder

    if use_prior_model:
        model = torch.load(sf.dataset_params['latest_model_loc'],
                           map_location=sf.dataset_params['device'])
        switch_device(model, sf.dataset_params['device'])
    else:
        model = SMN(sf.dataset_params)

    # actually train model
    best_performance = -1
    time_info, batch_losses = [], []
    batch_prfs = [('matcher', [], [], []), ('candidate inference', [], [], [])]
    track_for_ct = 50
    gap_filler = '\n' + '='.join(['' for _ in range(30)]) + '\n'
    best_f1_yet, best_matcher_ci_f1_yet = 0.0, 0.0
    for curr_round in range(sf.dataset_params['max_rounds']):

        print(gap_filler)
        print('Starting training...')
        print(gap_filler)

        tr_ex_ct, val_ex_ct = 0, 0

        gr_enc_tries = 15
 
        prev_pc = None
        unchanged = True
        
        # training
        model.train()
        num_tr_ex_per_round = sf.dataset_params['num_tr_ex_per_round']
        while tr_ex_ct < num_tr_ex_per_round:


            for enc_attempt in range(gr_enc_tries + 1):
                try:
                    base_graph = gen_valid_graph_example(sf.dataset_params,
                                                         gr_encoder=graph_encoder)
                    # we use sf.repetitions here so that we can have batchnorm in 
                    # addition to the same example but with multiple 
                    # valid relabelings
                    rep_ct = sf.dataset_params['repetitions']
                    listified_exs = [graph_encoder.listify_graph_example(base_graph)
                                     for rep in range(rep_ct)]
                    break
                except ValueError as e:
                    if enc_attempt < gr_enc_tries and 'no samples' in str(e): 
                        print(e)
                    else: raise e
            batch = collate_graph_exs(listified_exs)

            st_time = time.time()

            ( node_inds, sig_inds, edge_inds, upd_layers, func_sets,
              in_edges, cand_infs, ex_ranges ) = batch
        
            try:
                b_loss, batch_stats = model.train_batch(node_inds, sig_inds, 
                                                        edge_inds, upd_layers,
                                                        func_sets, ex_ranges, 
                                                        in_edges, cand_infs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    for param in model.parameters():
                        if param.grad is not None: del param.grad
                    print(e)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            assert not np.isnan(b_loss), 'Loss is nan...'

            tr_ex_ct += len(in_edges)
            time_info.append(time.time() - st_time)
            batch_losses.append(b_loss)
            if len(time_info) > track_for_ct: time_info.pop(0)
            if len(batch_losses) > track_for_ct: batch_losses.pop(0)
            b_zip = zip(batch_prfs, batch_stats)
            for (_, batch_precs, batch_recs, batch_f1s), (prec, rec, f1) in b_zip:
                batch_precs.append(prec)
                batch_recs.append(rec)
                batch_f1s.append(f1)
                if len(batch_precs) > track_for_ct: batch_precs.pop(0)
                if len(batch_recs) > track_for_ct: batch_recs.pop(0)
                if len(batch_f1s) > track_for_ct: batch_f1s.pop(0)
            
            _, matcher_precs, matcher_recs, matcher_f1s = batch_prfs[0]
            _, ci_precs, ci_recs, ci_f1s = batch_prfs[1]

            if np.mean(matcher_f1s) > best_f1_yet and \
               len(matcher_f1s) >= track_for_ct:
                torch.save(model, sf.dataset_params['best_model_loc'])
                best_f1_yet = np.mean(matcher_f1s)
                best_matcher_ci_f1_yet = np.mean(ci_f1s)

            write_to_file = True
            if not write_to_file: continue
            tr_results_file = sf.model_param_str + '_tr_results.txt'
            with open(os.path.join(sf.dataset_params['results_dir'], 
                                   tr_results_file), 'w') as f:
                f.write('At round: ' + str(curr_round) + '\n')
                f.write('Average training batch time: ' + \
                        str(np.mean(time_info)) + '\n')
                f.write('Average training example time: ' + \
                        str(np.mean(time_info) / len(ex_ranges)) + '\n')
                f.write('Average training example loss: ' + \
                        str(np.mean(batch_losses)) + '\n')
                for (loss_type, batch_precs, batch_recs, batch_f1s) in batch_prfs:
                    f.write('Average ' + loss_type + ' precision: ' + \
                            str(np.mean(batch_precs)) + '\n')
                    f.write('Average ' + loss_type + ' recall: ' + \
                            str(np.mean(batch_recs)) + '\n')
                    f.write('Average ' + loss_type + ' f1: ' + \
                            str(np.mean(batch_f1s)) + '\n')
                f.write('Best matcher f1: ' + str(best_f1_yet) + '\n')
                f.write('Best matcher candidate inference f1: ' + \
                        str(best_matcher_ci_f1_yet) + '\n')
                f.write('Training examples processed: ' + \
                        str(tr_ex_ct) + '\n')
                f.write('Training completion: ' + \
                        str(round(tr_ex_ct / num_tr_ex_per_round * 100, 2)) + \
                        '%' + '\n')

        print('\nBeginning validation...\n')

        # validation
        model.eval()
        val_ex_ct = 0
        fft_stats = [('matcher', 0, 0, 0), ('candidate inference', 0, 0, 0)]
        false_pos, false_neg, true_pos = 0, 0, 0
        for batch in validation_generator:

            ( node_inds, sig_inds, edge_inds, upd_layers, func_sets,
              in_edges, cand_infs, ex_ranges ) = batch
        
            with torch.no_grad():
                pred_edges, pred_cis = model.batch_structural_match(node_inds, 
                                                                    sig_inds,
                                                                    edge_inds,
                                                                    upd_layers,
                                                                    func_sets,
                                                                    ex_ranges)
            val_ex_ct += 1
            new_fft_stats = []
            p_zip = zip(fft_stats, [(in_edges, pred_edges), (cand_infs, pred_cis)])
            for (_, false_pos, false_neg, true_pos), (in_info, pred_info) in p_zip:
                for in_i, preds in zip(in_info, pred_info):
                    for prob, i in preds:
                        if i in in_i: true_pos += 1
                    else: false_pos += 1
                for i in in_i:
                    if not i in preds: false_neg += 1
                new_fft_stats.append((false_pos, false_neg, true_pos))
            write_to_file = False
            if not write_to_file: continue
            val_compl_file = sf.model_param_str + '_val_compl.txt'
            with open(os.path.join(sf.dataset_params['results_dir'], 
                                   val_compl_file), 'w') as f:
                f.write('Validation completion: ' + \
                        str(round(val_ex_ct / sf.num_str_val_ex * 100, 2)) + \
                        '%' + '\n')

        print('Validation finished...\n')

        results_file = sf.model_param_str + '_val_results.csv'
        with open(os.path.join(sf.dataset_params['results_dir'], 
                               results_file), 'a') as f:
            write_stats = []
            for perf_type, false_pos, false_neg, true_pos in fft_stats:
                if true_pos == 0: round_prec = 0
                else: round_prec = true_pos / (true_pos + false_pos)
                if true_pos == 0: round_rec = 0 
                else: round_rec = true_pos / (true_pos + false_neg)
                if round_prec + round_rec == 0:
                    round_f1 = 0
                else:
                    round_f1 = 2 * (round_prec * round_rec)
                    round_f1 /= (round_prec + round_rec)
                print('F1 for ' + perf_type + ' at ' + str(round_f1) + '...\n')
                write_stats.extend([str(curr_round), str(round_prec), 
                                    str(round_rec), str(round_f1)])
                if perf_type == 'matcher':
                    if round_f1 > best_performance:
                        best_performance = round_f1
                        torch.save(model, sf.dataset_params['best_val_model_loc'])
                    print(gap_filler)
                    print('Round precision: ' + str(round_prec))
                    print('Round recall: ' + str(round_rec))
                    print('Round f1: ' + str(round_f1))
                    print(gap_filler)

            f.write(','.join(write_stats) + '\n')


        torch.save(model, sf.dataset_params['latest_model_loc'])

              
