'''Unified interface to all dynamic graph model experiments'''
import argparse
import logging
import math
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange

from graph import SubgraphNeighborFinder
from inductive_util import load_indudctive_data, load_mask_data
from subgnn_np import SubGnnNp
from utils import (EarlyStopMonitor, RandEdgeSampler, get_free_gpu,
                        set_random_seed)

#import numba



# Argument and global variables
if True:
    parser = argparse.ArgumentParser(
        'Interface for subgraph experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='ia-contact')
    parser.add_argument('-t',
                        '--task',
                        default='inductive',
                        choices=['inductive', 'edge', 'node'])
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='Inductive-TIP',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=2,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=2,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=120,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=120,
                        help='Dimentions of the time embedding')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--num_prop', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--no-scale',
                        action='store_false',
                        dest='scale',
                        help='MinMaxScaler for node features')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# set_random_seed()

# Arguments
if True:
    import warnings
    warnings.filterwarnings('always')

    PREFIX = args.prefix
    SCALE = args.scale
    TASK = args.task
    FREEZE = args.freeze
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    
    UNIFORM = args.uniform
    ATTN_MODE = args.attn_mode
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim

    # Specific arguments
    NUM_PROP = args.num_prop
    NUM_MLP_LAYERS = args.num_mlp_layers
    ALPHA = args.alpha

    # Model initialize
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device = torch.device('cuda:{}'.format(GPU))

    import socket

    DEVICE_STR = f'{socket.gethostname()}-{device.index}'
    PARAM_STR = f'{FREEZE}-{NUM_LAYER}-{NUM_HEADS}-{NUM_NEIGHBORS}'
    PARAM_STR += f'-{NUM_PROP}-{NUM_MLP_LAYERS}-{ALPHA}'
    PARAM_STR += f'-{BATCH_SIZE}-{DROP_OUT}-{UNIFORM}'

    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{TASK}-{PARAM_STR}-{DATA}.pth'

    def get_checkpoint_path(epoch):
        return f'./ckpt/{args.prefix}-{TASK}-{DEVICE_STR}-{PARAM_STR}-{DATA}-{epoch}.pth'


# set up logger
if True:
    # set up logger
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)


def eval_one_epoch(tgan, edges, sampler):
    if len(edges) <= 0:
        return -1.0, -1.0, -1.0, -1.0
    assert sampler.seed is not None
    sampler.reset_random_state()
    src_l = edges['from_node_id'].to_numpy()
    dst_l = edges['to_node_id'].to_numpy()
    ts_l = edges['timestamp'].to_numpy()

    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src_l)
        num_test_batch = math.ceil(len(src_l) / TEST_BATCH_SIZE)
        scores = []
        labels = []

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(s_idx + TEST_BATCH_SIZE, num_test_instance)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)
            
            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            scores.extend(list(pred_score))
            labels.extend(list(true_label))

        pred_label = np.array(scores) > 0.5
        pred_prob = np.array(scores)
        label = np.array(labels)

        acc = accuracy_score(label, pred_label)
        ap = average_precision_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_prob)
    return acc, ap, f1, auc


# Load data and train val test split
if True:
    edges, nodes = load_indudctive_data(DATA)
    # We add a dummy node 0 for the empty sequence.
    if SCALE:
        n_feat = MinMaxScaler().fit_transform(nodes.iloc[:, 4:])
    else:
        n_feat = nodes.iloc[:, 4:].to_numpy()
    NODE_DIM = n_feat.shape[1]
    zero_pad = np.zeros((1, NODE_DIM))
    n_feat = np.vstack([zero_pad, n_feat])
    edges['from_node_id'] += 1
    edges['to_node_id'] += 1

    train_edges, ii_val_edges, nn_val_edges, ii_test_edges, nn_test_edges = load_mask_data(edges, nodes, 0.7, 0.85)
    n_nodes = len(nodes)

    g_df = edges[['from_node_id', 'to_node_id', 'timestamp']].copy()
    g_df['idx'] = np.arange(1, len(g_df) + 1)
    g_df.columns = ['u', 'i', 'ts', 'idx']

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    # if FREEZE:
    #     n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    # else:
    # bound = np.sqrt(6 / (2 * NODE_DIM))
    # n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

# set train, validation, test datasets
if True:
    train_src_l = train_edges['from_node_id'].to_numpy()
    train_dst_l = train_edges['to_node_id'].to_numpy()
    train_ts_l = train_edges['timestamp'].to_numpy()
    
    train_sampler = RandEdgeSampler(train_edges, dst_l)
    val_sampler = RandEdgeSampler(edges, dst_l, seed=0)
    nn_val_sampler = RandEdgeSampler(nn_val_edges, dst_l, seed=1)
    test_sampler = RandEdgeSampler(edges, dst_l, seed=2)
    nn_test_sampler = RandEdgeSampler(nn_test_edges, dst_l, seed=3)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
# full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)
ngh_finder = SubgraphNeighborFinder(full_adj_list,
                                    ts_l,
                                    graph_type='numpy',
                                    task=TASK,
                                    dataset=DATA,
                                    uniform=UNIFORM)

# Model initialize
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# device = torch.device('cuda:{}'.format(GPU))
tgan = SubGnnNp(ngh_finder,
                n_feat,
                e_feat,
                n_feat_freeze=FREEZE,
                attn_mode=ATTN_MODE,
                num_layers=NUM_LAYER,
                num_prop=NUM_PROP,
                num_mlp_layers=NUM_MLP_LAYERS,
                alpha=ALPHA,
                n_head=NUM_HEADS,
                drop_out=DROP_OUT)

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor(max_round=20)
epoch_bar = trange(NUM_EPOCH)
for epoch in epoch_bar:
    # Training
    np.random.shuffle(idx_list)
    batch_bar = trange(num_batch)
    for k in batch_bar:
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake,
                                           ts_l_cut, NUM_NEIGHBORS)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(),
                                         (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc = accuracy_score(true_label, pred_label)
            ap = average_precision_score(true_label, pred_label)
            f1 = f1_score(true_label, pred_label)
            auc = roc_auc_score(true_label, pred_score)
            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

    val_acc, val_ap, val_f1, val_auc = eval_one_epoch(tgan, ii_val_edges, val_sampler)
    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(tgan, nn_val_edges, nn_val_sampler)
    epoch_bar.set_postfix(auc=val_auc, nn_auc=nn_val_auc)

    if early_stopper.early_stop_check(nn_val_auc):
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

logger.info('No improvment over {} epochs, stop training'.format(
    early_stopper.max_round))
logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
tgan.load_state_dict(torch.load(best_model_path))
logger.info(
    f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
tgan.eval()

val_acc, val_ap, val_f1, val_auc = eval_one_epoch(tgan, ii_val_edges, val_sampler)
nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(tgan, nn_val_edges, nn_val_sampler)

test_acc, test_ap, test_f1, test_auc = eval_one_epoch(tgan, ii_test_edges, test_sampler)
nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(tgan, nn_test_edges, nn_test_sampler)

logger.info('Test statistics: acc: {:.4f}, ap {:.4f}, f1:{:.4f} auc: {:.4f}'.format(
    test_acc, test_ap, test_f1, test_auc))
logger.info('Unseen test statistics: acc: {:.4f}, ap {:.4f}, f1:{:.4f} auc: {:.4f}'.format(
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc))

logger.info('Saving Subgraph model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('Subgraph models saved')

res_path = 'inductive-results/{}-Subgraph.csv'.format(DATA)
headers = ['method', 'dataset', 'valid_auc', 'accuracy', 'ap', 'f1', 'auc', 'nn_accuracy', 'nn_ap', 'nn_f1', 'nn_auc', 'params']
if not os.path.exists(res_path):
    f = open(res_path, 'w+')
    f.write(','.join(headers) + '\r\n')
    f.close()
    os.chmod(res_path, 0o777)
config = f'scale={SCALE},num_prop={NUM_PROP},num_mlp_layers={NUM_MLP_LAYERS},alpha={ALPHA}'
config += f',n_layer={NUM_LAYER},n_head={NUM_HEADS},freeze={FREEZE}'
config += f',num_neighbors={NUM_NEIGHBORS},dropout={DROP_OUT},batchsize={BATCH_SIZE},uniform={UNIFORM}'
with open(res_path, 'a') as file:
    test_metric = '{:.4f},{:.4f},{:.4f},{:.4f}'.format(test_acc, test_ap, test_f1, test_auc)
    nn_test_metric = '{:.4f},{:.4f},{:.4f},{:.4f}'.format(nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc)
    file.write('{},{},{:.4f},{},{},\"{}\"'.format(
        PREFIX, DATA, val_auc, test_metric, nn_test_metric, config))
    file.write('\n')
