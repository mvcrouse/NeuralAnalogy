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

#######################
# Custom dataset class
#######################

class Dataset(data.Dataset):
    def __init__(self, data_ids):
        self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        X = torch.load(self.data_ids[index] + '.pt')
        return X

def collate_graph_exs(data):
    node_emb_inds, node_sig_inds, edge_emb_inds, offset_funcs = [], [], [], []
    offset_upd, all_in_edges, all_cand_infs, batch_ranges = [], [], [], []
    for ex in data:
        (base_info, target_info, in_edges, cand_infs) = ex
        # ranges for batch
        node_ranges = []
        func_sets = []
        # offset each example component where necessary
        for (n_inds, s_inds, e_inds, updates, funcs) in [base_info, target_info]:
            n_offset, e_offset = len(node_emb_inds), len(edge_emb_inds)
            node_ranges.append([n_offset, n_offset + len(n_inds)])
            # embedding indices dont get offset
            node_emb_inds.extend(n_inds)
            node_sig_inds.extend(s_inds)
            edge_emb_inds.extend(e_inds)
            func_sets.append([f + n_offset for f in funcs])
            # get offset updates and merge them across batch
            internal_offset_upd = []
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
                internal_offset_upd.append(offset_layer)
        offset_funcs.append(func_sets)
        all_in_edges.append(in_edges)
        all_cand_infs.append(cand_infs)
        batch_ranges.append(node_ranges)
    return (node_emb_inds, node_sig_inds, edge_emb_inds, offset_upd, offset_funcs,
            all_in_edges, all_cand_infs, batch_ranges)
