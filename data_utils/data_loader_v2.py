# _*_ coding:utf-8 _*_

import os, sys, time

ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(ABSPATH)
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse as sp
import random

from label_utils import LabelLoader
from feature_utils import FeatureLoader
from edge_utils import EdgeLoader

import pickle

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


# tf.app.flags.DEFINE_string('f', '', 'kernel')

def sample_mask(idx, n):
    """Create specific mask."""
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def onehot_encode(label, classes=2):
    """Onehot encoder"""
    n = len(label)
    onehot = np.zeros((n, classes), dtype=np.int32)
    onehot[np.arange(n), label] = 1
    return onehot


class DataLoader():
    """
    8 steps in data loader:
        1) load features, labels and edges
        2) select features, labels and edges by date
            has been selected in each utils
        3) merge features and labels
        4) filter uids by the size of invitation graph
        5) delete extra edges
        6) generate node ids from edges
        7) rename uid in feature_label to id, and modify features
        8) rename uid in edges to id
        9) generate adjancency matrix

    The data include:
        self.edge_name
        self.feature_label, self.uid_id, self.edge, self.cordinate_list, self.adj_list
    """

    def __init__(self):
        print(f"start loading data ...")
        begin_time = time.time()

        self.uid_id_path = FLAGS.preprocessed_data_path + "uid_id.csv"
        self.feature_path = FLAGS.preprocessed_data_path + "feature.pkl"
        self.label_path = FLAGS.preprocessed_data_path + "label.csv"

        self._preprocess()
        end_time = time.time() - begin_time
        print(f"finish loading data, time elapsed: {end_time:.3f}s ...\n")

    def _preprocess(self):
        if not FLAGS.use_preprocessed_data or not os.path.exists(self.feature_path):
            self._load_feature_label_edge()
            self._select_feature_label_edge()
            self._merge_feature_label()
            self._filter_uids_by_invitation_size()
            self._delete_extra_edges()
            self._gen_node_id()
            self._merge_feature_label_edge()
            self._rename_edges()
            self._gen_adj_matrix()

            # save preprocessed data
            if FLAGS.save_preprocessed_data:
                print(f"begin saving data at {FLAGS.preprocessed_data_path}")
                if not os.path.exists(FLAGS.preprocessed_data_path):
                    os.makedirs(FLAGS.preprocessed_data_path)

                self.uid_id.to_csv(self.uid_id_path, index=False)

                with open(self.feature_path, "wb") as f:
                    pickle.dump({'feature': self.feature,
                                 "vanilla_feature": self.vanilla_feature,
                                 "vanilla_adj": self.vanilla_adj}, f)
                self.label.to_csv(self.label_path, index=False)

                for index in range(len(self.edge_name)):
                    cur_path = FLAGS.preprocessed_data_path + self.edge_name[index]
                    self.edge[index].to_csv(cur_path + "_edge.csv", index=False)  # save edges
                    np.save(cur_path + "_cord.npy", self.cordinate_list[index])  # save cordinate
                    sp.save_npz(cur_path + "_adj.npz", self.adj_list[index])  # save adj matrix
        else:
            time_tick = time.time()
            print(f"\t load uid_id map from {self.uid_id_path} ", end="")
            self.uid_id = pd.read_csv(self.uid_id_path)
            print(f"time elapsed={time.time() - time_tick:.3f}s")

            print(f"\t load feature from {self.feature_path} ", end="")
            with open(self.feature_path, 'rb') as f:
                all_data = pickle.load(f)
                self.feature = all_data['feature']
                self.vanilla_feature = all_data['vanilla_feature']
                self.vanilla_adj = all_data['vanilla_adj']
                del all_data

                print(f"time elapsed={time.time() - time_tick:.3f}s")

            print(f"\t load label from {self.label_path} ", end="")
            self.label = pd.read_csv(self.label_path)
            print(f"time elapsed={time.time() - time_tick:.3f}s")

            self.edge_name = ['invite', 'ip', 'did', 'dfp']
            self.edge, self.cordinate_list, self.adj_list = [], [], []
            for index in range(len(self.edge_name)):
                cur_path = FLAGS.preprocessed_data_path + self.edge_name[index]

                print(f"\t load {self.edge_name[index]} edges from {cur_path}_edge.csv/_cord.npy/_adj.npz ", end="")
                self.edge.append(pd.read_csv(cur_path + "_edge.csv"))
                self.cordinate_list.append(np.load(cur_path + "_cord.npy"))
                self.adj_list.append(sparse.load_npz(cur_path + "_adj.npz"))
                print(f"time elapsed={time.time() - time_tick:.3f}s")

    def _load_feature_label_edge(self):
        print(f"1): load features, labels and edges")
        time_tick = time.time()

        # load feature
        self.feature = FeatureLoader().get_data()
        self.feature_num = len(self.feature["feature"][0].split(","))
        print("")

        # load label
        self.label = LabelLoader().get_data()
        print("")

        # load edge
        self.edge, self.edge_name = EdgeLoader().get_data()
        print("")

        print(
            f"\t feature_shape={self.feature.shape} feature_num={self.feature_num} feature_columns={self.feature.columns.values}")
        print(f"\t label_shape={self.label.shape} label_columns={self.label.columns.values}")
        for index, edge in enumerate(self.edge):
            if index == 0: print(f"\t ", end="")
            print(f"{self.edge_name[index]}={self.edge[index].shape}", end=" ")
        print("")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _select_feature_label_edge(self):
        print(f"2): select features, labels and edges by date")
        time_tick = time.time()

        self.label = self.label.drop('dt', axis=1)

        for index in range(len(self.edge)):
            self.edge[index] = self.edge[index].drop('dt', axis=1)

            def weight_sum(df):
                array = df["weight"].values.astype(np.int32)
                return array.sum()

            before_shape = self.edge[index].shape
            self.edge[index] = self.edge[index].groupby(by=["uid1", "uid2"]).apply(
                lambda x: weight_sum(x)).reset_index()

            self.edge[index].rename(columns={0: 'weight'}, inplace=True)
            if self.edge_name[index] != 'invite' and FLAGS.multi_edges_w_1:
                self.edge[index]['weight'] = 1
            elif self.edge_name[index] == 'invite' and FLAGS.invite_edges_w_1:
                self.edge[index]['weight'] = 1
            print(
                f"\t edges from {self.edge_name[index]:>6s}: before={before_shape} after={self.edge[index].shape} time={time.time() - time_tick:.3f}s")

        '''
        # delete features that are not in range [FLAGS.begin_date, FLAGS.end_date]
        before_shape = self.feature.shape
        del_mask = self.feature['dt'].apply(lambda x: x < FLAGS.begin_date or x > FLAGS.end_date)
        del_row_list = self.feature[del_mask].index.tolist()
        self.feature = self.feature.drop(del_row_list)#.drop('dt', axis=1)
        print(f"\t feature: before={before_shape} after={self.feature.shape}")

        # delete labels that are not in range [FLAGS.begin_date, FLAGS.end_date]
        before_shape = self.label.shape
        del_mask = self.label['dt'].apply(lambda x: x < FLAGS.begin_date or x > FLAGS.end_date)
        del_row_list = self.label[del_mask].index.tolist()
        self.label = self.label.drop(del_row_list).drop('dt', axis=1)
        print(f"\t label: before={before_shape} after={self.label.shape}")

        # delete edges that are not in range [FLAGS.begin_date, FLAGS.end_date]
        for index in range(len(self.edge)):
            del_mask = self.edge[index]['dt'].apply(lambda x: x < FLAGS.begin_date or x > FLAGS.end_date)
            del_row_list = self.edge[index][del_mask].index.tolist()
            self.edge[index] = self.edge[index].drop(del_row_list).drop('dt', axis=1)

            def weight_sum(df):
                array = df["weight"].values.astype(np.int32)
                return array.sum()

            before_shape = self.edge[index].shape
            self.edge[index] = self.edge[index].groupby(by=["uid1", "uid2"]).apply(lambda x: weight_sum(x)).reset_index()

            self.edge[index].rename(columns={0: 'weight'}, inplace=True)
            if self.edge_name[index] != 'invite' and FLAGS.multi_edges_w_1:
                self.edge[index]['weight'] = 1
            elif self.edge_name[index] == 'invite' and FLAGS.invite_edges_w_1:
                self.edge[index]['weight'] = 1
            print(f"\t edges from {self.edge_name[index]:>6s}: before={before_shape} after={self.edge[index].shape} time={time.time() - time_tick:.3f}s")
        '''

        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _merge_feature_label(self):
        print(f"3): merge features and labels")
        time_tick = time.time()
        self.feature_label = pd.merge(self.feature, self.label, on='uid', how='left')
        self.feature_label['label'].fillna(-1, inplace=True)

        print(
            f"\t feature_label.shape={self.feature_label.shape} feature_label.columns.values={self.feature_label.columns.values}")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _filter_uids_by_invitation_size(self):
        print(f"4): filter uids by the size of invitation graph")
        time_tick = time.time()

        # get all invite edge uids
        tmp_df = pd.concat([self.edge[0]["uid1"], self.edge[0]["uid2"]]).to_frame('uid')
        invite_uids = set(tmp_df["uid"])
        invite_uid_num = len(invite_uids)
        del tmp_df
        print(f"\t invite_uid_num in invitation graph = {invite_uid_num}")

        # get uid to id dictionary
        uid2id = dict()
        for index, uid in enumerate(invite_uids):
            uid2id[uid] = index

        # get invite edges list
        invite_edge_list = [[] for _ in range(invite_uid_num)]
        invite_uid1_list = self.edge[0]["uid1"].values.tolist()
        invite_uid2_list = self.edge[0]["uid2"].values.tolist()
        for uid1, uid2 in zip(invite_uid1_list, invite_uid2_list):
            id1, id2 = uid2id[uid1], uid2id[uid2]
            invite_edge_list[id1].append(id2)
            invite_edge_list[id2].append(id1)

        # get the size of invitation graph
        def bfs_uid_neighbor(start_id):
            if visited[start_id]: return []
            Q = [start_id]
            visited[start_id] = True
            all_visited = [start_id]

            while len(Q) > 0:
                nQ = []
                for u in Q:
                    for v in invite_edge_list[u]:
                        if visited[v]: continue
                        visited[v] = True
                        all_visited.append(v)
                        nQ.append(v)
                Q = nQ
            return all_visited

        visited = [False for _ in range(invite_uid_num + 1)]
        invitation_graph_size = [0 for _ in range(invite_uid_num + 1)]
        # invitation_graph_size[-1] = 10
        for uid in invite_uids:
            if not visited[uid2id[uid]]:
                all_visited = bfs_uid_neighbor(uid2id[uid])
                for v in all_visited:
                    invitation_graph_size[v] = len(all_visited)

        # filter feature uids by the size of invitation graph
        ftr_uids = self.feature_label["uid"].values.tolist()
        invite_uid_num_ftr = 0
        tmp_neighbor_num = []
        tmp_invitation_graph_size = []
        for uid in ftr_uids:
            if uid not in uid2id.keys():
                tmp_neighbor_num.append(0)
                tmp_invitation_graph_size.append(0)
                uid2id[uid] = invite_uid_num
            else:
                tmp_invitation_graph_size.append(invitation_graph_size[uid2id[uid]])
                tmp_neighbor_num.append(len(set(invite_edge_list[uid2id[uid]])))
                invite_uid_num_ftr += 1
        print(f"\t invite_uid_num in features = {invite_uid_num_ftr}")
        # print(f"\t max(invitation_graph_size) = {max(invitation_graph_size)}")
        print(f"\t set(invitation_graph_size) = {set(invitation_graph_size)}")
        print(f"\t # invitation_graph_size >= 5 = {sum(np.array(invitation_graph_size) >= 5)}")

        self.tmp_info = pd.DataFrame({"uid": self.feature_label["uid"].values.tolist(),
                                      "label": self.feature_label["label"].values.tolist(),
                                      "neighbor_num": tmp_neighbor_num,
                                      "invitation_graph_size": tmp_invitation_graph_size})

        before_shape, before_unlabel = self.feature_label.shape, sum(self.feature_label["label"] == -1)
        before_pos, before_neg = sum(self.feature_label["label"] == 1), sum(self.feature_label["label"] == 0)

        mask = self.feature_label["uid"].apply(lambda x: invitation_graph_size[uid2id[x]] >= FLAGS.filter_graph_size)
        del_rows = self.feature_label[mask].index.tolist()
        self.feature_label = self.feature_label[mask]  # .drop(del_rows)

        after_shape, after_unlabel = self.feature_label.shape, sum(self.feature_label["label"] == -1)
        after_pos, after_neg = sum(self.feature_label["label"] == 1), sum(self.feature_label["label"] == 0)
        print(
            f"before/after: shape={before_shape}/{after_shape} pos={before_pos}/{after_pos} neg={before_neg}/{after_neg} \
        unlabel={before_unlabel}/{after_unlabel} time={time.time() - time_tick:.3f}s")

    def _delete_extra_edges(self):
        print(f"5): delete extra edges")
        time_tick = time.time()
        ftr_uids = set(self.feature_label["uid"])

        for index in range(len(self.edge)):
            before_shape = self.edge[index].shape

            mask1 = self.edge[index]['uid1'].apply(lambda x: x in ftr_uids)
            mask2 = self.edge[index]['uid2'].apply(lambda x: x in ftr_uids)
            self.edge[index] = self.edge[index][mask1 | mask2]
            after_shape = self.edge[index].shape
            print(
                f"\t {self.edge_name[index]}: before={before_shape} after={after_shape}, time={time.time() - time_tick:.3f}s")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _gen_node_id(self):
        print(f"6): generate node ids from edges")
        time_tick = time.time()

        # get all uids from edges
        all_uids = set()
        uid_set = []
        for index in range(len(self.edge_name)):
            tmp_df = pd.concat([self.edge[index]["uid1"], self.edge[index]["uid2"]]).to_frame(
                'uid').drop_duplicates().reset_index(drop=True)
            cur_set = set(tmp_df["uid"])
            all_uids = all_uids | cur_set
            uid_set.append(cur_set)

        # create uid list and id list
        index = 0
        uid_list, id_list = [], []
        for uids in all_uids:
            uid_list.append(uids)
            id_list.append(index)
            index += 1

        # print the number of uids in each graph
        for i in range(1, len(uid_set)):
            len0, leni, len0_i = len(uid_set[0]), len(uid_set[i]), len(uid_set[0] & uid_set[i])
            print(f"\t {self.edge_name[0]}_uid_num={len0}, {self.edge_name[i]}_uid_num={leni}, Intersection={len0_i}")
            # assert leni == len0_i

        self.uid_id = pd.DataFrame({"id": id_list, "uid": uid_list})
        self.node_num = self.uid_id.shape[0]

        print(f"\t self.uid_id.columns.values = {self.uid_id.columns.values}")
        print(f"\t # nodes in graph: {self.node_num}, time elapsed = {time.time() - time_tick:.3f}s")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _merge_feature_label_edge(self):
        print(f"7): rename uid in feature_label to id, and modify features")
        time_tick = time.time()

        edge_uid = set(self.uid_id['uid'])
        feature_uid = set(self.feature_label['uid'])

        print(
            f"\t len(edge_uid)={len(edge_uid)} len(feature_uid)={len(feature_uid)} len(edge&feature)={len(edge_uid & feature_uid)}")

        del edge_uid
        del feature_uid

        self.feature_label = pd.merge(self.uid_id, self.feature_label, on='uid', how='left')
        self.feature_label.sort_values('id', inplace=True)

        self.feature_label['label'].fillna(-1, inplace=True)
        self.feature_label['dt'].fillna(FLAGS.begin_date, inplace=True)
        null_ftr = ','.join(["0" for i in range(self.feature_num)])

        null_rows_num = self.feature_label["feature"].isnull().sum(axis=0)
        print(
            f"\t have merged uid_id_feature_label, null_rows_num in feature is {null_rows_num}, time elapsed:{time.time() - time_tick:.3f}s")

        for i in range(self.feature_label.shape[0]):
            assert self.feature_label['id'].iloc[
                       i] == i, f"i={i} self.feature_label['id'].iloc[i]={self.feature_label['id'].iloc[i]}"
        assert self.feature_label.shape[0] == self.uid_id.shape[0]

        # preprocess labels
        self.label = self.feature_label[["label", "dt"]]
        pos_num, neg_num = sum(self.label["label"] == 1), sum(self.label["label"] == 0)
        unlabel_num = sum(self.label["label"] == -1)
        print(
            f"\t have preprocessed labels, pos/neg/unlabel={pos_num}/{neg_num}/{unlabel_num} time={time.time() - time_tick:.3f}s")

        # 用来数据分析
        self.tmp_feature = pd.DataFrame({"uid": self.feature_label["uid"].values.tolist(),
                                         "label": self.feature_label["label"].values.tolist()})

        # preprocess features
        null_rows_num = 0
        null_row_ftr = [0 for i in range(self.feature_num)]
        null_row_ftr.append(1)
        null_rows_state = self.feature_label["feature"].isnull()

        self.feature = []
        for i in range(self.node_num):
            if null_rows_state[i]:
                self.feature.append(null_row_ftr)
                null_rows_num += 1
            else:
                ftr = [float(p) for p in self.feature_label['feature'].iloc[i].split(',')]
                ftr.append(0)
                self.feature.append(ftr)

        self.vanilla_feature = self.feature
        self.feature = sp.csr_matrix(self.feature).tolil()
        del self.feature_label

        print(f"\t null_rows_num={null_rows_num} feature={self.feature.shape} label={self.label.shape}")
        print(f"time elapsed = {time.time() - time_tick:.3f}s")

    def _rename_edges(self):
        print(f"8): rename uid in edges to id")
        time_tick = time.time()
        for index, edges_df in enumerate(self.edge):
            # print(f"self.uid_id.columns.values = {self.uid_id.columns.values}")
            uid_id_tmp = self.uid_id.rename(columns={'uid': 'uid1'}, inplace=False)
            tmp = edges_df.merge(uid_id_tmp, on="uid1", how="left")
            tmp.rename(columns={'id': 'id1'}, inplace=True)

            uid_id_tmp = self.uid_id.rename(columns={'uid': 'uid2'}, inplace=False)
            tmp = tmp.merge(uid_id_tmp, on="uid2", how="left")
            tmp.rename(columns={'id': 'id2'}, inplace=True)

            edge_id_df = tmp.copy(deep=True)
            del edge_id_df["uid1"]  # to save memory
            del edge_id_df["uid2"]  # to save memory

            before_shape = edge_id_df.shape

            edge_id_df = edge_id_df.dropna(axis=0, how="any")  # remove id is NaN
            edge_id_df["id1"] = edge_id_df["id1"].astype(int)
            edge_id_df["id2"] = edge_id_df["id2"].astype(int)

            after_shape = edge_id_df.shape

            self.edge[index] = edge_id_df
            print(
                f"\t edge from {self.edge_name[index]} shape while removing NaN: before={before_shape} after={after_shape}, time={time.time() - time_tick:.3f}s")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def _gen_adj_matrix(self):
        print(f"9): generate adjacency matrix ")
        time_tick = time.time()

        self.adj_list, self.cordinate_list = [], []

        self.vanilla_adj = []
        for index, edge_df in enumerate(self.edge):
            row = edge_df["id1"].values.tolist()
            col = edge_df["id2"].values.tolist()
            if self.edge_name[index] != 'invite' and FLAGS.multi_edges_w_1 and FLAGS.use_multi_edges:
                weight = list(np.ones((len(row),), dtype=np.uint8))
            elif self.edge_name[index] == 'invite' and FLAGS.invite_edges_w_1:
                weight = list(np.ones((len(row),), dtype=np.uint8))
            else:
                weight = edge_df["weight"].values.tolist()

            print(f"len(row) = {len(row)} len(col)={len(col)} type(col[0]) = {type(col[0])}")
            self.vanilla_adj.append([row, col, weight, self.node_num])

            cordinate = np.array([row, col])
            adj = sp.csr_matrix((weight, (row, col)), shape=(self.node_num, self.node_num))
            self.cordinate_list.append(cordinate)
            self.adj_list.append(adj)
            # print(f"\t edge from {self.edge_name[index]}, time={time.time()-time_tick:.3f}s")
        print(f"time elapsed: {time.time() - time_tick:.3f}s")

    def load_data(self):
        print(f"load data for training, testing, validation")
        labels = np.array(self.label['label'], dtype=np.int32).reshape(-1, 1)
        total_num = labels.shape[0]

        if FLAGS.predict_date != "-1":
            if "/" in FLAGS.predict_date:
                predict_begin_date, predict_end_date = FLAGS.predict_date.split('/')
            else:
                predict_begin_date, predict_end_date = FLAGS.predict_date, FLAGS.predict_date

            test_mask = self.label['dt'].apply(lambda x: x >= predict_begin_date and x <= predict_end_date)
            test_list = self.label[test_mask].index.tolist()
            test_label = labels[test_list, :]

            print(f"*********** predict_date={predict_begin_date}/{predict_end_date} len(test_list)={len(test_list)}")

            pos_mask = self.label['label'].apply(lambda x: x == 1)
            neg_mask = self.label['label'].apply(lambda x: x == 0)
            no_predict_mask = self.label['dt'].apply(lambda x: x < predict_begin_date or x > predict_end_date)

            pos_index = self.label[pos_mask & no_predict_mask].index.tolist()
            neg_index = self.label[neg_mask & no_predict_mask].index.tolist()

            random.shuffle(pos_index)
            random.shuffle(neg_index)

            pos_len, neg_len = len(pos_index), len(neg_index)

            if FLAGS.balance:
                min_num = min(len(pos_index), len(neg_index))
                pos_index = pos_index[:min_num]
                neg_index = neg_index[:min_num]
            else:
                min_num = min(pos_len, neg_len) * 3
                pos_index = pos_index[:min(pos_len, min_num)]
                neg_index = neg_index[:min(neg_len, min_num)]

            pos_num, neg_num = len(pos_index), len(neg_index)

            split1 = 0.8
            train_list = pos_index[:int(pos_num * split1)] + neg_index[:int(neg_num * split1)]
            val_list = pos_index[int(pos_num * split1):] + neg_index[int(neg_num * split1):]

            train_mask = sample_mask(train_list, total_num)
            val_mask = sample_mask(val_list, total_num)
            test_mask = sample_mask(test_list, total_num)

            y_train = np.zeros((total_num, 2))
            y_val = np.zeros((total_num, 2))
            y_train[train_mask, :] = onehot_encode(labels[train_mask, :].flatten())
            y_val[val_mask, :] = onehot_encode(labels[val_mask, :].flatten())

            print(f"\t total_num = {total_num} features_shape={self.feature.shape}")
            print(f"\t train_num={len(train_list)} val_num={len(val_list)} test_num={len(test_list)}")
            print(f"\t In training and validation data, pos_num:neg_num = {pos_num}:{neg_num}")
            return self.vanilla_adj, self.vanilla_feature, self.adj_list, self.feature, labels, self.uid_id, y_train, y_val, test_label, train_mask, val_mask, test_mask

        data_path = FLAGS.preprocessed_data_path + "data.pkl"
        if True or not os.path.exists(data_path):
            pos_mask = self.label['label'].apply(lambda x: x == 1)
            neg_mask = self.label['label'].apply(lambda x: x == 0)
            pos_index = self.label[pos_mask].index.tolist()
            neg_index = self.label[neg_mask].index.tolist()

            #             pos_index = list(np.where(self.label == 1))
            #             neg_index = list(np.where(self.label == 0))
            #             print(pos_index[:10], neg_index[:10])
            #             pos_index, neg_index = [], []
            #             for i in range(self.label.shape[0]):
            #                 if self.label[i] == 1:
            #                     pos_index.append(i)
            #                 elif self.label[i] == 0:
            #                     neg_index.append(i)

            random.shuffle(pos_index)
            random.shuffle(neg_index)

            extra_pos, extra_neg = [], []
            pos_len, neg_len = len(pos_index), len(neg_index)

            if FLAGS.balance:
                min_num = min(len(pos_index), len(neg_index))
                pos_index = pos_index[:min_num]
                neg_index = neg_index[:min_num]
            else:
                min_num = min(pos_len, neg_len) * 3
                pos_index = pos_index[:min(pos_len, min_num)]
                neg_index = neg_index[:min(neg_len, min_num)]
                if min_num < pos_len:
                    extra_pos = pos_index[min_num:]
                if min_num < neg_len:
                    extra_neg = neg_index[min_num:]

            pos_num, neg_num = len(pos_index), len(neg_index)

            split1 = FLAGS.train_ratio
            split2 = FLAGS.train_ratio + FLAGS.test_ratio

            train_pos_list = pos_index[:int(pos_num * split1)]
            test_pos_list = pos_index[int(pos_num * split1): int(pos_num * split2)]
            val_pos_list = pos_index[int(pos_num * split2):]

            train_neg_list = neg_index[:int(neg_num * split1)]
            test_neg_list = neg_index[int(neg_num * split1): int(neg_num * split2)]
            val_neg_list = neg_index[int(neg_num * split2):]

            train_list = train_pos_list + train_neg_list
            test_list = test_pos_list + test_neg_list
            val_list = val_pos_list + val_neg_list

            uid_id_map = self.uid_id

            train_mask = sample_mask(train_list, total_num)
            val_mask = sample_mask(val_list, total_num)
            test_mask = sample_mask(test_list, total_num)

            y_train = np.zeros((total_num, 2))
            y_val = np.zeros((total_num, 2))
            y_train[train_mask, :] = onehot_encode(labels[train_mask, :].flatten())
            y_val[val_mask, :] = onehot_encode(labels[val_mask, :].flatten())

            #             train_mask =  sample_mask(train_list, self.label)
            #             val_mask =  sample_mask(val_list, self.label)
            #             test_mask =  sample_mask(test_list, self.label)

            #             y_train = np.zeros((self.label.shape[0],2))
            #             y_val = np.zeros((self.label.shape[0],2))
            #             y_test = np.zeros((self.label.shape[0],2))
            #             y_train[train_mask,:] = onehot_encode(labels[train_mask,:].flatten())
            #             y_val[val_mask,:] = onehot_encode(self.label[val_mask,:].flatten())
            #             # y_test[test_mask,:] = onehot_encode(self.label[test_mask,:].flatten())
            test_label = labels[test_mask, :]

            # save the resplited train and test data
            if FLAGS.save_resplit_data or (not os.path.exists(data_path)):
                resplit_path = FLAGS.preprocessed_data_path + "resplit_data.pkl"
                if (not os.path.exists(data_path)):
                    resplit_path = data_path

                print(f"begin saving resplited training and testing data in {resplit_path} ...")
                if not os.path.exists(FLAGS.preprocessed_data_path):
                    os.mkdir(FLAGS.preprocessed_data_path)

                with open(resplit_path, "wb") as f:
                    pickle.dump({"y_train": y_train, "y_val": y_val, "test_label": test_label,
                                 "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
                                 "pos_num": pos_num, "neg_num": neg_num}, f)

                print(f"finish saving resplited training and testing data")
            else:
                print(f"NOT save resplited training and testing data!")
        else:
            print(f"\t load data from saved file: {data_path}")
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                y_train, y_val, test_label = data['y_train'], data['y_val'], data['test_label']
                train_mask, val_mask, test_mask, = data['train_mask'], data['val_mask'], data['test_mask']
                pos_num, neg_num = data['pos_num'], data['neg_num']
                del data

        print(f"\t total_num = {total_num} features_shape={self.feature.shape}")
        print(f"\t train_num={sum(train_mask)} val_num={sum(val_mask)} test_num={sum(test_mask)}")
        print(f"\t In training, validation and testing data, pos_num:neg_num = {pos_num}:{neg_num}")
        return self.vanilla_adj, self.vanilla_feature, self.adj_list, self.feature, labels, self.uid_id, y_train, y_val, test_label, train_mask, val_mask, test_mask


if __name__ == "__main__":
    feature_table = 'columbus_common_invite_jisu_material_features_v1'
    invite_edge_table = 'columbus_common_invite_jisu_material_edges_invite'
    device_edge_table = 'columbus_common_invite_jisu_material_edges_device'
    label_table = 'columbus_common_invite_jisu_material_labels'

    ## table name, data stored in MySql
    flags.DEFINE_string('feature_table', feature_table, 'raw features saved in MySql')
    flags.DEFINE_string('label_table', label_table, 'label table name in MySql')
    flags.DEFINE_string('invite_edge_table', invite_edge_table, 'invite edge table name in MySql')
    flags.DEFINE_string('device_edge_table', device_edge_table, 'device edge table name in MySql')

    flags.DEFINE_string('begin_date', '2019-09-15', "begin date")
    flags.DEFINE_string('end_date', '2019-10-17', "end date")
    flags.DEFINE_string('predict_date', '2019-10-11/2019-10-17', 'predict date.')

    flags.DEFINE_integer("filter_graph_size", 5, 'the minimum size of invitation graph')
    flags.DEFINE_boolean('filter_label', False, "whether or not filter labels")

    flags.DEFINE_boolean('use_vanilla_data', True, 'whether or not use vanilla data')
    flags.DEFINE_boolean('save_vanilla_data', False, 'whether or not save vanilla data')
    flags.DEFINE_boolean('use_preprocessed_data', False, 'whether or not use preprocessed data')
    flags.DEFINE_boolean('save_preprocessed_data', False, 'whether or not save preprocessed data')
    flags.DEFINE_boolean('balance', True, 'whether or not balance postive and negative samples')

    flags.DEFINE_string('preprocessed_data_path', '../preprocessed_data/', 'preprocessed data path')

    flags.DEFINE_boolean('resplit_data', True, 'whether or not resplit data')
    flags.DEFINE_boolean('save_resplit_data', False, 'whether or not save resplited data')

    flags.DEFINE_boolean('use_multi_edges', True, 'whether or not train or predict.')
    flags.DEFINE_boolean('multi_edges_w_1', True, 'whether or not train or predict.')
    flags.DEFINE_boolean('invite_edges_w_1', False, 'whether or not train or predict.')

    ## directory
    flags.DEFINE_string('data_dir', './data/', 'input data path.')
    flags.DEFINE_string('output_dir', './output/', 'model save path.')
    flags.DEFINE_string('model_dir', './model/', 'model save path.')
    flags.DEFINE_string('data_file_name', './data/data.pkl', 'save preprocessed data')

    flags.DEFINE_float('train_ratio', 0.6, 'the ratio of training data.')
    flags.DEFINE_float('test_ratio', 0.2, 'the ratio of testing data.')
    flags.DEFINE_float('val_ratio', 0.2, 'the ration of validation data.')

    data_loader = DataLoader()
    _ = data_loader.load_data()
