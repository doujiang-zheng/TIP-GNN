"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import os
import sys
import argparse

from tqdm import trange
import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from data_util import load_graph, load_label_data, load_data
from graph import make_label_data
from subgnn_np import SubGnnNp
from graph import SubgraphNeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler, get_free_gpu, set_random_seed

# Argument and global variables
if True:
    parser = argparse.ArgumentParser(
        'Interface for TGAT experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='ia-contact')
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='TIP',
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:{}".format(GPU))

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


def eval_one_epoch(hint, tgan, src, dst, ts, label):
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(len(src) / TEST_BATCH_SIZE)
        scores = []

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(s_idx + TEST_BATCH_SIZE, num_test_instance)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            prob_score = tgan.forward(src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
            scores.extend(list(prob_score.cpu().numpy()))
        pred_label = np.array(scores) > 0.5
        pred_prob = np.array(scores)
    return accuracy_score(label, pred_label), average_precision_score(
        label,
        pred_label), f1_score(label,
                              pred_label), roc_auc_score(label, pred_prob)


# Load data and train val test split
if True:
    if TASK == "edge":
        edges, n_nodes, val_time, test_time = load_graph(DATA)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(g_df) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]
    elif TASK == "node":
        edges, nodes = load_data(DATA, "format")
        n_nodes = len(nodes) + 1
        # padding node is 0, so add 1 here.
        id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(edges) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if FREEZE:
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

# set train, validation, test datasets
if True:
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]

    train_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_sampler = RandEdgeSampler(src_l, dst_l)
    test_sampler = RandEdgeSampler(src_l, dst_l)

# set validation, test datasets
if True:
    if TASK == "edge":
        _, val_data, test_data = load_label_data(dataset=DATA)

        val_src_l = val_data.u.values
        val_dst_l = val_data.i.values
        val_ts_l = val_data.ts.values
        val_label_l = val_data.label.values

        test_src_l = test_data.u.values
        test_dst_l = test_data.i.values
        test_ts_l = test_data.ts.values
        test_label_l = test_data.label.values
    elif TASK == "node":
        # select validation and test dataset
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time

        val_src_l, val_dst_l, val_ts_l, val_label_l = make_label_data(
            src_l, dst_l, ts_l, valid_val_flag, test_sampler)
        test_src_l, test_dst_l, test_ts_l, test_label_l = make_label_data(
            src_l, dst_l, ts_l, valid_test_flag, test_sampler)
    else:
        raise NotImplementedError(TASK)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
# full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)
ngh_finder = SubgraphNeighborFinder(full_adj_list,
                                    ts_l,
                                    graph_type="numpy",
                                    task=TASK,
                                    dataset=DATA,
                                    uniform=UNIFORM)

# Model initialize
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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

    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes',
                                                      tgan, val_src_l,
                                                      val_dst_l, val_ts_l,
                                                      val_label_l)
    # epoch_bar.update()
    epoch_bar.set_postfix(acc=val_acc, f1=val_f1, auc=val_auc)

    if early_stopper.early_stop_check(val_auc):
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

_, _, _, val_auc = eval_one_epoch('val for old nodes', tgan, val_src_l,
                                  val_dst_l, val_ts_l, val_label_l)
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes',
                                                      tgan, test_src_l,
                                                      test_dst_l, test_ts_l,
                                                      test_label_l)

logger.info('Test statistics: acc: {:.4f}, f1:{:.4f} auc: {:.4f}'.format(
    test_acc, test_f1, test_auc))

logger.info('Saving Subgraph model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('Subgraph models saved')

res_path = "results/{}-Subgraph.csv".format(DATA)
headers = ["method", "dataset", "valid_auc", "accuracy", "f1", "auc", "params"]
if not os.path.exists(res_path):
    f = open(res_path, 'w+')
    f.write(",".join(headers) + "\r\n")
    f.close()
    os.chmod(res_path, 0o777)
config = f"num_prop={NUM_PROP},num_mlp_layers={NUM_MLP_LAYERS},alpha={ALPHA}"
config += f",n_layer={NUM_LAYER},n_head={NUM_HEADS},freeze={FREEZE}"
config += f",num_neighbors={NUM_NEIGHBORS},dropout={DROP_OUT},batchsize={BATCH_SIZE},uniform={UNIFORM}"
with open(res_path, "a") as file:
    file.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f},\"{}\"".format(
        PREFIX, DATA, val_auc, test_acc, test_f1, test_auc, config))
    file.write("\n")
