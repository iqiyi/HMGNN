# _*_ coding:utf-8 _*_

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    #max_ = adj.max(axis=1)
    y = adj.transpose(copy=True)
    adj_normalized = normalize_adj(adj + y + sp.eye(adj.shape[0]))
    if FLAGS.adj_power == 1:
        return sparse_to_tuple(adj_normalized)
    else:
        cur_adj = list()
        mat = sp.eye(adj.shape[0])
        base = sp.coo_matrix(adj_normalized)
        for i in range(FLAGS.adj_power):
            mat = mat.dot(base)
            cur_adj.append(sparse_to_tuple(mat.tocoo()))
        print(f"len(cur_adj) = {len(cur_adj)}")
        return cur_adj


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    if FLAGS.adj_power == 1:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    else:
        feed_dict.update({placeholders['support'][i][j]: support[i][j] for j in range(FLAGS.adj_power) for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def print_flags(FLAGS):
    print('***************** FLAGS Configurations: *************************')
    for name, value in FLAGS.__flags.items():
        value = value.value
        if type(value) == float:
            print(f"{name:>40s}:\t {value:.7f}")
        elif type(value) == int:
            print(f"{name:>40s}:\t {value:d}")
        elif type(value) == str:
            print(f"{name:>40s}:\t {value}")
        elif type(value) == bool:
            print(f"{name:>40s}:\t {value}")
        else:
            print(f"{name:>40s}:\t {value}")            
    print("******************************************************************")
