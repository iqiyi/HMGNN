from __future__ import division
from __future__ import print_function

import os,time
import argparse
import tensorflow as tf
import matplotlib as plt
import pickle

from utils import *
import hparams
import pandas as pd
from scipy import sparse
import scipy.sparse as sp

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
try:
    tf.app.flags.DEFINE_string('f', '', 'kernel')
except:
    pass

def evaluate(sess, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['dropout']: 0.})
    outs_val = sess.run([model.loss, model.accuracy, model.evaluation], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2][:-1], (time.time() - t_test)

def train():
    print(f"\nstart training process ...........")
    train_begin_time = time.time()
    with tf.Session() as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_loss_list = []
        saver = tf.train.Saver(model.vars, max_to_keep=5)

        # Train model
        best_f1 = 0
        for epoch in range(FLAGS.epochs):
            epoch_begin = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            train_outs = sess.run([model.opt_op, model.loss, model.accuracy, model.evaluation], feed_dict=feed_dict)
            train_loss, train_acc, train_preds, train_eval = train_outs[1], train_outs[2], train_outs[3][-1], train_outs[3][:-1]

            train_time = time.time() - epoch_begin

            # Validation
            val_loss, val_acc, val_eval, val_time = evaluate(sess, features, support, y_val, val_mask, placeholders)
            val_loss_list.append(val_loss)

            # test
            # test_loss, test_acc, test_eval, test_time = evaluate(sess, features, support, y_test, test_mask, placeholders)


            epoch_end = time.time() - epoch_begin
            # Print results
            print(f"Epoch:{epoch+1:3d},   loss    acc  precision recall    f1      tpr     tnr  time, time elapsed={epoch_end:.3f}s --------")
            train_outs = ' '.join([("%.5f" % _) for _ in train_eval])
            val_outs = ' '.join([("%.5f" % _) for _ in val_eval])
            # test_outs = ' '.join([("%.5f" % _) for _ in test_eval])
            print(f"Train:     {train_loss:.5f} {train_acc:.5f} {train_outs} {train_time:.3f}s")
            print(f"Valid:     {val_loss:.5f} {val_acc:.5f} {val_outs} {val_time:.3f}s")
            # print(f"Test :     {test_loss:.5f} {test_acc:.5f} {test_outs} {test_time:.3f}s")

            if FLAGS.attention and epoch > 0 and epoch % 20 == 0:
                print(f"subgraph attention: {[_[0] for _ in sess.run(model.att)]}")
            
            # if train_eval[2] > best_f1:
            if val_eval[2] > best_f1:
                best_f1 = max(val_eval[2], best_f1)
                # best_f1 = max(train_eval[2], best_f1)
                if FLAGS.model_version >= 0:
                    save_name = FLAGS.model_name + "-Version" + str(FLAGS.model_version)
                else:
                    save_name = FLAGS.model_name # "GCN"
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_date, save_name)
                model.save(checkpoint_path, sess, epoch, saver)
            print("")
#             if epoch > FLAGS.early_stopping and val_loss_list[-1] > np.mean(val_loss_list[-(FLAGS.early_stopping//2+1):-1]) + 0.05:
#                 print(f"Early Stopping: epoch={epoch} val_loss_list[-1]={val_loss_list[-1]:.5f}", end=' ')
#                 print(f"before_mean={np.mean(val_loss_list[-(FLAGS.early_stopping//2+1):-1]):.5f}")
#                 break
    train_end_time = time.time() - train_begin_time
    print(f"finish training process, time elapsed: {train_end_time:.3f}s ...................")


def test_eval():
    print("\nstart testing on test data ...............")
    begin_test_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_path = os.path.join(FLAGS.model_dir, FLAGS.model_date)
        print(f"load_path = {load_path}")
        model.load(load_path, sess)

        test_loss, test_acc, test_eval, test_time = evaluate(sess, features, support, y_test, test_mask, placeholders)
        test_outs = ' '.join([("%.5f" % _) for _ in test_eval])
        print(f"Perf:   loss    acc  precision recall    f1      tpr     tnr  time")
        print(f"Test: {test_loss:.5f} {test_acc:.5f} {test_outs} {test_time:.3f}s")

    end_test_time = time.time() - begin_test_time
    print(f"finish evaluating on test data, time elapsed = {end_test_time:.3f}s ...................")

def get_all_preds():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_path = os.path.join(FLAGS.model_dir, FLAGS.model_date)
        print(f"get all preds, model restored from {load_path}")
        model.load(load_path, sess)
        feed_dict_test = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict_test.update({placeholders['dropout']: 0.})
        all_preds, all_probs = sess.run([model.preds, model.probs], feed_dict=feed_dict_test)
    return all_preds, all_probs

def evaluate_test(preds, truth, threshold=None):
    if threshold is not None:
        preds[preds >= threshold] = 1
        preds[preds < threshold] = -1
    preds[preds == 0] = -1
    truth[truth == -1] = -2
    truth[truth == 0] = -1

    P, N, TP, FP, TN, FN = 0, 0, 0, 0, 0, 0
    unlabel_num, unlabel_pos, unlabel_neg = 0, 0, 0
    label_num, label_pos, label_neg = 0, 0, 0
    extra_pos, vanilla_pos = 0, 0
    for i in range(preds.shape[0]):
        if truth[i] == -2:
            unlabel_num += 1
            if preds[i] == 1:
                unlabel_pos += 1
            else:
                unlabel_neg += 1
        elif truth[i] == 1:
            if preds[i] == 1: TP += 1
        else:
            if preds[i] == -1: TN += 1

        if truth[i] == 1:
            vanilla_pos += 1
        elif preds[i] == 1:
            extra_pos += 1

        if preds[i] == 1 and truth[i] == -1: FP += 1
        if preds[i] == -1 and truth[i] == 1: FN += 1
    P, N = sum(preds == 1), sum(preds == -1)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2*precision*recall) /(precision + recall)
    P, N, unlabel_num = max(P, 1), max(N, 1), max(unlabel_num, 1)
    print(f"precison={precision:.4f} recall={recall:.4f} F1={F1:.4f} TPR={TPR:.4f} TNR={TNR:.4f}")
    print(f"unlabel_black = {unlabel_pos}/{unlabel_num} = {unlabel_pos/unlabel_num:.4f} pos_num={P} neg_num={N}")
    print(f"extra_pos/vanilla_pos={extra_pos}/{vanilla_pos}={extra_pos/vanilla_pos:.3f}")
    
def write_test_preds(test_uids, test_preds, test_probs, threshold=0.95):
    uid_list, probs_list = [], []
    for i in range(len(test_uids)):
        if test_probs[i] >= threshold:
            uid_list.append(test_uids[i])
            probs_list.append(test_probs[i])
    results = pd.DataFrame({"uid": uid_list, "probs": probs_list})
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    result_path = FLAGS.output_dir + "black.csv"
    results.to_csv(result_path, index=False)
    
def prepare_super_nodes(vani_adj):
    super_nodes = []
    for index, per_graph in enumerate(vani_adj):
        ### 给定一张大图的adj返回这个大图包含多少个小图，每个小图的大小，每个小图包含的id
        def bfs_graph(Graph):
            node_num = Graph[3]
            visited = [False for i in range(node_num)]
            neighbor = [[] for i in range(node_num)]

            for u, v in zip(list(Graph[0]), list(Graph[1])):
                neighbor[u].append(v)
                neighbor[v].append(u)

            # get the size of graph
            def bfs_neighbor(start_id):
                if visited[start_id]: return []
                Q = [start_id]
                visited[start_id] = True
                all_visited = [start_id]

                while len(Q) > 0:
                    nQ = []
                    for u in Q:
                        for v in neighbor[u]:
                            if visited[v]: continue
                            visited[v] = True
                            all_visited.append(v)
                            nQ.append(v)
                    Q = nQ
                return all_visited

            subgraph_num = 0
            subgraph_size = []
            subgraph_ids = []

            for i in range(node_num):
                if visited[i]: continue
                tmp_visited = bfs_neighbor(i)
                if len(tmp_visited) < FLAGS.minimum_subgraph_size: continue
                subgraph_num += 1
                subgraph_size.append(len(tmp_visited))
                subgraph_ids.append(tmp_visited)
            return subgraph_num, subgraph_size, subgraph_ids
        graph_num, graph_size, graph_ids = bfs_graph(per_graph)
        super_nodes.append([graph_num, graph_size, graph_ids])
    return super_nodes

if __name__ == "__main__":
    train_begin = time.time()

    print(f"---------------------------------- Begin initializing FLAGS ----------------------------------")
    begin_time = time.time()
    FLAGS = hparams.create()
    FLAGS.model_date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    end_time = time.time() - begin_time
    print(f"---------------------------------- Finish initializing FLAGS: time elapsed: {end_time:.3f}s -----------\n")


    print(f"---------------------------------- Begin loading data ----------------------------------")
    begin_load_data_time = time.time()
    ####### Please specify your own data loader
    loader = DataLoader()
    vani_adjs, vani_ftr, adjs, features, labels, uid_id_map, y_train, y_val, y_test, train_mask, val_mask, test_mask = loader.load_data()
    
    # graph_info = [normal_node_num, super_node_num, super_links vani_adjs, preprocessed_adjs, preprocessed_feature]
    super_nodes = prepare_super_nodes(vani_adjs) # [graph_num, graph_size, graph_ids]
    normal_node_num = len(vani_ftr)
    super_node_num = sum([per[0] for per in super_nodes])
    total_num = normal_node_num + super_node_num
    vani_ftr_np = np.array(vani_ftr)
    
    edge_name = ['e1', 'e2', 'e3', 'e4']
    num_supports = len(vani_adjs)
    vanillla_normal_feature = features.copy()
    
    print(f"\n normal_nodes_num={normal_node_num}", end=" ")
    for i in range(num_supports):
        print(f"{edge_name[i]}_num={normal_node_num+super_nodes[i][0]}({super_nodes[i][0]})", end=" ")
    print(f"total_num={total_num} time={time.time()-begin_load_data_time:.3f}s")
    
    # 每张图的节点数量一致，包括三种类型的节点：原始节点、超点、空点
    null_row_ftr = [0 for i in range(FLAGS.feature_dim)] # 空点特征
    
    # 先计算每张图前后图的节点数量
    pre_graph_sum, suf_graph_sum = [0 for i in range(num_supports)], [0 for i in range(num_supports)]
    for i in range(1, num_supports):
        pre_graph_sum[i] = pre_graph_sum[i-1] + super_nodes[i-1][0]
    for i in range(num_supports - 2, -1, -1):
        suf_graph_sum[i] = suf_graph_sum[i+1] + super_nodes[i+1][0]
        
    # 计算每张关系图的4个subsupport的坐标范围
    # [[top_i, bottom_i, left_i, right_i]] 1 <= i <= 4  [都是闭区间]
    subsupport_range = []
    for i in range(num_supports):
        cur_support = []
        # 子节点 和 子节点之间的图
        top, bottom, left, right = 0, normal_node_num - 1, 0, normal_node_num - 1        
        cur_support.append([top, bottom, left, right])
        
        # 子节点和超点之间的图
        top, bottom = 0, normal_node_num - 1
        left = normal_node_num + pre_graph_sum[i]
        right = left + super_nodes[i][0] - 1
        cur_support.append([top, bottom, left, right])
        
        # 超点和子节点之间的图
        top = normal_node_num + pre_graph_sum[i]
        bottom = top + super_nodes[i][0] - 1
        left, right = 0, normal_node_num - 1
        cur_support.append([top, bottom, left, right])
        
        # 超点和超点之间的图
        top = normal_node_num + pre_graph_sum[i]
        bottom = top + super_nodes[i][0] - 1
        left, right = top, bottom
        cur_support.append([top, bottom, left, right])
        subsupport_range.append(cur_support)
    
    features = vani_ftr.copy()
    whole_support = [] # 保存最终每张图预处理后的 邻接矩阵
    for i in range(num_supports):
        # vanilla_adjs --> [[row, col, weight, node_num]]
        adjs = vani_adjs[i].copy()
        for j in range(super_nodes[i][0]): # 第 j 个超点
            features.append(list(np.mean(vani_ftr_np[super_nodes[i][2][j]], axis=0)))
        
    # 对于每个超点找 K近邻
    features_np = np.array(features)
    K = 10
    from sklearn.neighbors import NearestNeighbors
    for i in range(num_supports):
        st = normal_node_num + pre_graph_sum[i]
        ed = st + super_nodes[i][0]
        super_features = features_np[st:ed]
        clf = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(super_features)
        distances, indices = clf.kneighbors(super_features)
        
        adjs = vani_adjs[i].copy()
        for index, per in enumerate(indices):
            u = normal_node_num + pre_graph_sum[i] + index
            for vv in per:
                v = normal_node_num + pre_graph_sum[i] + vv
                if u != v:
                    adjs[0].append(u)
                    adjs[1].append(v)
                    adjs[2].append(1)
        p_adj = sp.csr_matrix((adjs[2], (adjs[0], adjs[1])), shape=(total_num, total_num))    
        p_adj = preprocess_adj(p_adj)
        print(f"{edge_name[i]}, shape = {p_adj[2]} edges_num={len(adjs[0])}")
        whole_support.append(p_adj)
    
    print(f"get all whole supports ok, time={time.time()-begin_load_data_time:.3f}s")
    
    support = whole_support
    features = sp.csr_matrix(features).tolil()
    features = preprocess_features(features)
    
    # 扩展 labels, y_train, y_val, y_test, train_mask, val_mask, test_mask
    # labels = y_test = np.array(normal_node_num, ), y_train = y_val = np.array(normal_node_num, 2)
    # train_mask = val_mask = test_mask = np.array(noraml_node_num, )
    super_node_labels = [-1 for i in range(super_node_num)]
    labels = list(labels)
    labels.extend(super_node_labels)
    labels = np.array(labels)

    super_node_mask = [False for i in range(super_node_num)]
    train_mask = list(train_mask)
    train_mask.extend(super_node_mask)
    train_mask = np.array(train_mask, dtype=np.bool)

    val_mask = list(val_mask)
    val_mask.extend(super_node_mask)
    val_mask = np.array(val_mask, dtype=np.bool)

    test_mask = list(test_mask)
    test_mask.extend(super_node_mask)
    test_mask = np.array(test_mask, dtype=np.bool)

    super_node_one_hot = np.zeros((super_node_num, 2), dtype = np.int32)
    y_train = np.vstack((y_train, super_node_one_hot))
    y_val = np.vstack((y_val, super_node_one_hot))

    end_load_data_time = time.time() - begin_load_data_time
    print(f"label_kinds={FLAGS.label_kinds} num_supports={num_supports} input_dim={features[2][1]}")
    print(f"total_num = normal_node_num + super_node_num = {normal_node_num} + {super_node_num} = {total_num}")
    print(f"---------------------------------- Finish loading data: time elapsed: {end_load_data_time:.3f}s -----------\n")


    print(f"\n---------------------------------- Begin initializing model ----- {FLAGS.model_name} --------------")
    begin_initialize = time.time()
    from model_graph.models import HMMG

    model_func = HMMG
    sparse_adj_shape = [[support[i][0].shape[0], support[i][0], support[i][-1]] for i in range(num_supports)]
    
    # define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) if FLAGS.adj_power == 1 else [tf.sparse_placeholder(tf.float32)
                        for i in range(FLAGS.adj_power)] for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # build model
    model = model_func(placeholders, 
                       input_dim=FLAGS.feature_dim,
                       hidden_dim=hidden_dim,
                       output_dim=FLAGS.label_kinds,
                       input_num=total_num,
                       normal_node_num=normal_node_num,
                       support_num=num_supports,
                       reweight_adj=FLAGS.reweight_adj,
                       residual=FLAGS.residual,
                       attention=FLAGS.attention,
                       sparse_adj_shape = sparse_adj_shape,
                       logging=True)
    end_initializing = time.time() - begin_initialize
    print(f"---------------------------------- Finish initialzing model, time elapsed: {end_initializing:.3f}s -------------\n")

    total_num, val_num, test_num = labels.shape[0], sum(val_mask), sum(test_mask)
    val_pos, val_neg = sum(labels[val_mask] == 1), sum(labels[val_mask] == 0)
    train_num, train_pos, train_neg = total_num-sum(train_mask)-val_num, sum(labels[train_mask] == 1), sum(labels[train_mask] == 0)
    test_pos, test_neg, test_unl = sum(labels[test_mask] == 1), sum(labels[test_mask] == 0), sum(labels[test_mask] == -1)

    print(f"begin_date={FLAGS.begin_date} end_date={FLAGS.end_date} predict_date={FLAGS.predict_date} total={total_num}")
    print(f"train: total={train_num} pos={train_pos} neg={train_neg} unlabel={train_num-train_pos-train_neg}")
    print(f"val  : total={val_num} pos={val_pos}  neg={val_neg}")
    print(f"test : total={test_num} pos={test_pos} neg={test_neg} unlabel={test_unl}")

    # train model
    train()
    train_end = time.time() - train_begin
    print(f"----------------------- Total Training Time = {train_end:.3f}s----------------------------")

    all_preds, all_probs = get_all_preds()
    
    # get uid2id and id2uid dictionary
    id_list, uid_list = uid_id_map["id"].values.tolist(), uid_id_map["uid"].values.tolist()
    id_uid_dict = dict()
    for index in range(len(id_list)):
        id_uid_dict[id_list[index]] = uid_list[index]
        
    print(f"begin_date={FLAGS.begin_date} end_date={FLAGS.end_date} predict_date={FLAGS.predict_date} total={total_num}")
    print(f"train: total={train_num} pos={train_pos} neg={train_neg} unlabel={train_num-train_pos-train_neg}")
    print(f"val  : total={val_num} pos={val_pos}  neg={val_neg}")
    print(f"test : total={test_num} pos={test_pos} neg={test_neg} unlabel={test_unl}")

    for i in range(all_probs.shape[0]):
        if all_preds[i] == 0 and all_probs[i] > 0.5:
            all_probs[i] = 1 - all_probs[i]

    print("\ntest on validation data ...")
    val_preds, val_probs, val_labels = all_preds[val_mask], all_probs[val_mask], labels[val_mask]
    evaluate_preds(val_preds, val_probs, val_labels)

    print("\ntest on test data ...")
    test_preds, test_probs, test_labels = all_preds[test_mask], all_probs[test_mask], labels[test_mask]
    evaluate_preds(test_preds, test_probs, test_labels)
        
#     test_uids = [id_uid_dict[id_list[index]] for index in range(labels.shape[0]) if test_mask[index] == 1]
#     test_preds, test_probs = all_preds[test_mask], all_probs[test_mask]
#     for i in range(test_preds.shape[0]):
#         if test_preds[i] == 0:
#             test_probs[i] = 1 - test_probs[i]
#     write_test_preds(test_uids, test_preds, test_probs, 0.95)
