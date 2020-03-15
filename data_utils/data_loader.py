# _*_ coding:utf-8 _*_

import numpy as np
import random
import math


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    labels = np.load("labels.npy")

    node_num = labels.shape[0]
    classify = labels.shape[1]

    ids = [i for i in range(node_num)]
    random.shuffle(ids)
    print(ids)

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

    print(y_train)




if __name__ == '__main__':
    load_data()