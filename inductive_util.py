import argparse
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from data_util import load_data, _iterate_datasets
from utils import set_logger, set_random_seed

def load_indudctive_data(dataset):
    mode_dir = 'data/format_data'
    edges = pd.read_csv("{}/{}.edges".format(mode_dir, dataset))
    nodes = pd.read_csv("{}/inductive-{}.nodes".format(mode_dir, dataset))
    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
    edges['from_node_id'] = edges['from_node_id'].map(id2idx)
    edges['to_node_id'] = edges['to_node_id'].map(id2idx)
    edges = edges.sort_values(by='timestamp').reset_index(drop=True)
    return edges, nodes

def load_mask_data(edges, nodes, val_ratio=0.70, test_ratio=0.85):
    ts = edges['timestamp'].to_numpy()
    val_time, test_time = np.quantile(ts, [val_ratio, test_ratio])
    train_mask = ts < val_time
    val_mask = np.logical_and(ts >= val_time, ts < test_time)
    test_mask = ts >= test_time

    train_edges = edges[train_mask]
    val_edges = edges[val_mask]
    test_edges = edges[test_mask]

    train_nodes = set(train_edges['from_node_id']).union(set(train_edges['to_node_id']))
    ii_val_mask = np.logical_and(val_edges['from_node_id'].isin(train_nodes), val_edges['to_node_id'].isin(train_nodes))
    nn_val_mask = np.logical_not(ii_val_mask)
    ii_test_mask = np.logical_and(test_edges['from_node_id'].isin(train_nodes), test_edges['to_node_id'].isin(train_nodes))
    nn_test_mask = np.logical_not(ii_test_mask)

    return train_edges, val_edges[ii_val_mask], val_edges[nn_val_mask], test_edges[ii_test_mask], test_edges[nn_test_mask]
        

def generate_node_features(dataset):
    edges, nodes = load_data(dataset, mode='format')
    adj = defaultdict(lambda: dict())
    for name, group in edges.groupby(['from_node_id', 'to_node_id']):
        fid, tid = name[0], name[1]
        adj[fid][tid] = len(group)
    edge_tuples = [(fid, tid, {'weight':adj[fid][tid]}) for fid in adj for tid in adj[fid]]
    graph = nx.Graph(edge_tuples)

    node_list = nodes['node_id'].tolist()
    central_funcs = [nx.degree_centrality, # nx.closeness_centrality, 
        # nx.betweenness_centrality, nx.load_centrality,
        nx.pagerank, nx.clustering, nx.triangles]
    central_cols = [f.__name__ for f in central_funcs]
    central_dict = dict()
    # for func, name in zip(central_funcs, central_cols):
    #     logger.info(name)
    #     vals = func(graph)
    #     node_vals = [vals[nid] for nid in node_list]
    #     central_dict[name] = node_vals
    
    # logger.info('laplacian spectrum')
    nid2idx = {nid:idx for idx, nid in enumerate(graph.nodes())}
    idx_list = [nid2idx[nid] for nid in node_list]
    # vals = nx.laplacian_spectrum(graph)
    # laplacian_vals = vals[idx_list]
    # central_dict['laplacian_spectrum'] = laplacian_vals

    # logger.info('HITS: hubs and authority')
    # zero_vals = {nid: 0.0 for nid in node_list}
    # try:
    #     hubs, auths = nx.hits(graph)
    # except nx.exception.PowerIterationFailedConvergence:
    #     hubs, auths = zero_vals, zero_vals
    # central_dict['hub'] = [hubs[nid] for nid in node_list]
    # central_dict['authority'] = [auths[nid] for nid in node_list]
    
    logger.info('Eigen vector decomposition')
    EIGEN_DIM = 50
    lap = nx.laplacian_matrix(graph).asfptype()
    # w, v = np.linalg.eigh(lap.toarray())
    w, v = scipy.sparse.linalg.eigs(lap, k=EIGEN_DIM)
    print(v.shape)
    # eigen_vectors = np.real(v.T[:, :EIGEN_DIM])
    eigen_vectors = np.real(v)
    
    reorder_eigen_vectors = eigen_vectors[idx_list]
    for i in range(EIGEN_DIM):
        col_name = 'eigen_vec{:03d}'.format(i)
        central_dict[col_name] = reorder_eigen_vectors[:, i]

    for col in central_dict:
        nodes[col] = central_dict[col]
    nodes.to_csv('data/format_data/inductive-{}.nodes'.format(dataset), index=None)

if __name__ == '__main__':
    logger = set_logger()
    logger.info('Begin generate inductive node features.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='fb-forum', type=str)
    args = parser.parse_args()

    # datasets = _iterate_datasets()
    # cpu_count = multiprocessing.cpu_count()
    # Parallel(n_jobs=cpu_count//2, verbose=100)(delayed(generate_node_features)(name) for name in datasets)
    # dataset = 'ia-slashdot-reply-dir'
    generate_node_features(dataset=args.dataset)
    # for name in datasets:
    #     logger.info(name)
    #     generate_node_features(dataset=name)
