# _*_ coding:utf-8 _*_

import numpy as np
import scipy.sparse as sp
import pickle as pkl
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
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

def normalize_adj_asymmetric(adj):
    """Asymmetric normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum,-1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

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
    
def preprocess_adj_asymmetric(adj):
    adj_normalized = normalize_adj_asymmetric(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    # feed_dict.update({placeholders['loss_weight']: loss_weight})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    # feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    if FLAGS.adj_power == 1:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    else:
        feed_dict.update({placeholders['support'][i][j]: support[i][j] for j in range(FLAGS.adj_power) for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def print_flags(FLAGS):
    print('***************** FLAGS Configurations: *************************')
    for name, value in FLAGS.__flags.items():
        #print(f"name = {name} value = {value}")
        value=value.value
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
    
def evaluate_preds(preds, probs, truth, draw_curve=False):
    from sklearn.metrics import precision_score, recall_score, f1_score , roc_auc_score, roc_curve, auc
    assert preds.shape == truth.shape, f"Error in function evaluate preds: preds.shape={preds.shape} != truth.shape={truth.shape}"
    
    pos_label, neg_label = 1, 0
    
    positive = sum(truth == 1)
    tp = sum(truth*preds == 1)
    negative = sum(truth == 0)
    tn = sum((truth-1)*(preds-1) == 1) 
    total = preds.shape[0]
    
    for i in range(total):
        if preds[i] == 0:
            probs[i] = 1 - probs[i]
    
    tpr = tp / positive
    tnr = tn / negative
    precision = tp / sum(preds == pos_label)
    recall = tp / sum(truth == pos_label)
    f1 = 2.0 * precision * recall / (precision + recall)
    roc_auc =  roc_auc_score(truth, probs)
    
    if draw_curve:
        import matplotlib.pyplot as plt
        sklearn_fpr, sklearn_tpr, sklearn_thresholds = roc_curve(truth, probs, pos_label=pos_label)
        plt.figure()
        plt.plot(sklearn_fpr, sklearn_tpr, color='darkorange', linewidth=4)
        # print(f"auc(sklearn_fpr, sklearn_tpr)={auc(sklearn_fpr, sklearn_tpr):.5f}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        #plt.legend(loc="lower right")
        plt.show()
        print(f"positive = {positive} negative = {negative} total = {total}")
        
    return precision, recall, f1, tpr, tnr, roc_auc

# evaluate predictions, positive = black = 1, negative = white = 0
def evaluate_preds(preds, probs, truth, threshold=None):
    from sklearn.metrics import precision_score, recall_score, f1_score , roc_auc_score, roc_curve, auc
    labeled_probs, labeled_truth = [], []
    for i in range(probs.shape[0]):
        if truth[i] != -1:
            labeled_probs.append(probs[i])
            labeled_truth.append(truth[i])
    auc =  roc_auc_score(labeled_truth, labeled_probs)
    
    if threshold is not None:
        preds[preds >= threshold] = 1
        preds[preds < threshold] = -1
    preds[preds == 0] = -1
    truth[truth == -1] = -2
    truth[truth == 0] = -1
    
    P, N, TP, FP, TN, FN, unlabel_num = 0, 0, 0, 0, 0, 0, 0
    predict_pos, vanilla_pos = set(), set()
    for i in range(preds.shape[0]):
        if truth[i] == -2:
            unlabel_num += 1
        elif truth[i] == 1:
            P += 1
            if preds[i] == 1: TP += 1
        else:
            N += 1
            if preds[i] == -1: TN += 1
        
        if preds[i] == 1 and truth[i] == -1: FP += 1
        if preds[i] == -1 and truth[i] == 1: FN += 1
        
        if truth[i] == 1: vanilla_pos.add(i)
        if preds[i] == 1: predict_pos.add(i)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2*precision*recall) /(precision + recall)
    
    P, N = max(P, 1), max(N, 1)
    TPR, TNR = TP/P, TN/N
    
    pred_pos_num, vani_pos_num, inter_pos_num = len(predict_pos), len(vanilla_pos), len(predict_pos & vanilla_pos)
    improvement = (pred_pos_num - inter_pos_num) / vani_pos_num
    total_num = unlabel_num + P + N
    black_ratio = pred_pos_num / total_num
    
    print(f"info of input: unlable_num={unlabel_num} label_pos={P} label_neg={N}")
    print(f"perf on labeled data: precison={precision:.4f} recall={recall:.4f} F1={F1:.4f} TPR={TPR:.4f} TNR={TNR:.4f} AUC={auc:.4f}")
    print(f"improvement: (predict-intersection)/vanilla=({pred_pos_num}-{inter_pos_num})/{vani_pos_num}={improvement:.4f}")
    print(f"predict black ratio: predict_black/total_num={pred_pos_num}/{total_num}={black_ratio:.4f}")

def get_ensemble_result(all_preds, all_probs, all_eval, y_truth):
    y_preds = all_preds[0].copy()
    y_probs = all_probs[0].copy()
    for i in range(y_preds.shape[0]):
        zeros, ones = 0, 0
        for j in range(len(all_preds)):
            if all_preds[j][i] == 0:
                zeros += 1
            else:
                ones += 1
        if ones >= zeros:
            y_preds[i] = 1
        else:
            y_preds[i] = 0

        y_probs[i] = 0
        for j in range(len(all_preds)):
            if all_preds[j][i] == y_preds[i]:
                y_probs[i] += all_probs[j][i]
        y_probs[i] /= max(ones, zeros)

    ensemble_eval = evaluate_preds(y_preds, y_probs, y_truth, draw_curve=True)
    average_eval = np.mean(np.array(all_eval), axis=0)
    average_outs = ' '.join([("%.5f" % _) for _ in average_eval])
    ensemble_outs = ' '.join([("%.5f" % _) for _ in ensemble_eval])

    print(f"\t precision recall    f1      tpr     tnr    auc")
    print(f"Average : {average_outs}")
    print(f"Ensemble: {ensemble_outs}")
    return y_preds, y_probs

if __name__ == "__main__":
    load_data()
