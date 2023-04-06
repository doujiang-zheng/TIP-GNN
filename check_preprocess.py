import argparse
import logging
import math

import numpy as np
from tqdm import trange

from data_util import _iterate_datasets
from graph import SubgraphNeighborFinder
from preprocess import load_data_var, init_adj
from sampling import NeighborFinder


def check(edges, sg_ngh_finder, ngh_finder, BATCHSIZE=200, NUM_NGH=20):
    src_l = edges["from_node_id"].to_numpy()
    dst_l = edges["to_node_id"].to_numpy()
    ts_l = edges["timestamp"].to_numpy()

    num_batch = int(math.ceil(len(src_l) / BATCHSIZE))
    for k in trange(num_batch):
        s_idx = k * BATCHSIZE
        e_idx = min(len(src_l), s_idx + BATCHSIZE)
        src_l_cut, dst_l_cut, ts_l_cut = src_l[s_idx:e_idx], dst_l[
            s_idx:e_idx], ts_l[s_idx:e_idx]

        batch_n2n, batch_nid, batch_e2n, batch_eid, batch_ets = sg_ngh_finder.get_neighbor_np(
            src_l_cut, ts_l_cut, NUM_NGH)
        ngh_node_batch, ngh_eidx_batch, ngh_t_batch = ngh_finder.get_temporal_neighbor(
            src_l_cut, ts_l_cut, NUM_NGH)
        for i in range(batch_nid.shape[0]):
            mask1 = batch_nid[i] > 0
            mask2 = ngh_node_batch[i] > 0
            assert np.all(
                np.unique(batch_nid[i][mask1]) == np.unique(ngh_node_batch[i]
                                                            [mask2]))
            assert np.all(
                np.unique(batch_eid[i]) == np.unique(ngh_eidx_batch[i]))

        batch_n2n, batch_nid, batch_e2n, batch_eid, batch_ets = sg_ngh_finder.get_neighbor_np(
            dst_l_cut, ts_l_cut, NUM_NGH)
        ngh_node_batch, ngh_eidx_batch, ngh_t_batch = ngh_finder.get_temporal_neighbor(
            dst_l_cut, ts_l_cut, NUM_NGH)
        for i in range(batch_nid.shape[0]):
            mask1 = batch_nid[i] > 0
            mask2 = ngh_node_batch[i] > 0
            assert np.all(
                np.unique(batch_nid[i][mask1]) == np.unique(ngh_node_batch[i]
                                                            [mask2]))
            assert np.all(
                np.unique(batch_eid[i]) == np.unique(ngh_eidx_batch[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper for preprocessing interaction subgraphs.")
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    parser.add_argument("-m", "--m", default=20, type=int)
    args = parser.parse_args()

    if args.task == "node":
        datasets = ["JODIE-wikipedia", "JODIE-mooc", "JODIE-reddit"]
    else:
        datasets = _iterate_datasets()[:13]

    for data in datasets:
        print(data)
        edges = load_data_var(data, args.task)
        ts_l = edges["timestamp"].to_numpy()
        adj_list = init_adj(edges)
        sg_ngh_finder = SubgraphNeighborFinder(adj_list,
                                               ts_l,
                                               graph_type="numpy",
                                               task=args.task,
                                               dataset=data)
        ngh_finder = NeighborFinder(adj_list, uniform=False)
        check(edges, sg_ngh_finder, ngh_finder)
