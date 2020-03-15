# _*_ coding:utf-8 _*_

import os
import numpy as np
import random
import math


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(data_path):
    labels = np.load(os.path.join(data_path, "labels.npy"))

    node_num = labels.shape[0]
    classify = labels.shape[1]

    ids = [i for i in range(node_num)]
    random.shuffle(ids)

    train_ids = ids[0: math.ceil(0.6*node_num)]
    test_ids = ids[math.ceil(0.6*node_num): math.ceil(0.8*node_num)]
    val_ids = ids[math.ceil(0.8*node_num):]

    train_mask = sample_mask(train_ids, node_num)
    val_mask = sample_mask(val_ids, node_num)
    test_mask = sample_mask(test_ids, node_num)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    edges = np.load(os.path.join(data_path, "edges_mat.npy"))
    features = np.load(os.path.join(data_path, "features.npy"))

    row = list(edges[0,:])
    col = list(edges[1,:])
    weight = [1 for _ in range(len(row))]
    adj = [row, col, weight, node_num]
    adjs = [adj]

    return adjs, features, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask


if __name__ == '__main__':
    load_data(".")