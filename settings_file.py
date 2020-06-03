# python imports
import sys, os
# torch imports
import torch
from node_classes import Node

# setting recursion limit to 10000 by default
sys.setrecursionlimit(10000)
    
data_dir = os.path.join('.', 'data')
dataset_dir = os.path.join('.', 'datasets')
models_dir = os.path.join('.', 'models')
vis_dir = os.path.join('.', 'visualizations')
dataset_params = {
    'data_dir' : data_dir,
    'models_dir' : models_dir,
    'results_dir' : os.path.join('.', 'results'),
    'vis_dir' : vis_dir,
    'tr_data_loc' : os.path.join(data_dir, 'tr_data_obj.pkl'),
    'val_data_loc' : os.path.join(data_dir, 'val_data_obj.pkl'),
    'te_data_loc' : os.path.join(data_dir, 'te_data_obj.pkl'),
    'parsed_sm_tests_loc' : os.path.join(dataset_dir, 'parsed-sm-tests'),
    'encoder_loc' : os.path.join(data_dir, 'encoder_obj.pkl'),
    'best_model_loc' : os.path.join(models_dir, 'best_model_obj.pkl'),
    'best_val_model_loc' : os.path.join(models_dir, 'best_val_model_obj.pkl'),
    'latest_model_loc' : os.path.join(models_dir, 'latest_model_obj.pkl'),
    'vis_data_loc' : os.path.join(vis_dir, 'alignment_visualization.gv'),
    # neural network parameters
    'nodes_bprop' : { (Node.const_type, 0, True) : 50, 
                      # functions
                      (Node.func_type, 1, True) : 75, 
                      (Node.func_type, 1, False) : 5, 
                      (Node.func_type, 2, False) : 5,
                      (Node.func_type, 2, True) : 30, 
                      (Node.func_type, 3, True) : 5,
                      (Node.func_type, 3, False) : 5,
                      # attributes
                      (Node.pred_type, 1, True) : 75, 
                      # predicates
                      (Node.pred_type, 1, False) : 5, 
                      (Node.pred_type, 2, False) : 5, 
                      (Node.pred_type, 2, True) : 30, 
                      (Node.pred_type, 3, True) : 5, 
                      (Node.pred_type, 3, False) : 5 },
    'edge_ct' : 40,
    'emb_dim' : 16,
    'state_dim' : 32,
    'lr' : 0.01,
    'ti_thresh' : 1e-5,
    'aug_ct' : 0,
    # pointer lstm params
    'ptr_hid_dim' : 128,
    'num_att_layers' : 2,
    'num_mha_heads' : 4,
    'dec_beam_size' : 1,
    # general parameters
    'max_arity' : 3,
    # dataset parameters
    'num_tr_ex_per_round' : 50000,
    'num_val_ex' : 1000,
    'num_te_ex' : 1000,
    # dataset stats
    # (26.72, 26.86, 14.25, 14.37, 14.18, 14.24, 26.72)
    'min_matched' : 1,
    'max_matched' : 2,
    'min_duplicate' : 1,
    'max_duplicate' : 2,
    'min_unmatched' : 0,
    'max_unmatched' : 2,
    'rank_specs' : { 0 : (2, 5), 1 : (1, 2), 2 : (1, 3), 3 : (0, 3),
                     4 : (0, 3), 5 : (0, 3), 6 : (0, 3) },
    'min_ranks' : 2,
    'max_ranks' : 7,
    'ordered_chance' : 75,
    'cand_inf_chance' : 50,
    'ti_func_chance' : 50,
    'func_chance' : 50,
    # training parameters
    'max_rounds' : 2,
    'repetitions' : 8 }

dataset_params['node_ct'] = sum([v for v in dataset_params['nodes_bprop'].values()])

test_params = { 'batch_size': 1,
                'shuffle': True,
                'num_workers': 8 }
assert test_params['batch_size'] == 1, 'Beam search with batch size > 1 will error'

if torch.cuda.is_available(): dataset_params['device'] = torch.device('cuda')
else: dataset_params['device'] = torch.device('cpu')


model_param_str = '_'.join([str(x) for x in [dataset_params['state_dim'], 
                                             dataset_params['emb_dim'], 
                                             dataset_params['num_att_layers'],
                                             dataset_params['num_mha_heads']]])
