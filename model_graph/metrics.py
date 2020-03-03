# _*_ coding:utf-8 _*_

import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask, w_psedu=0):
    """Softmax cross-entropy loss with masking."""
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def weighted_masked_softmax_cross_entropy(preds, labels, mask, loss_weight, w_psedu=0):
    """Weighted Softmax cross-entropy loss with masking."""
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, 1), logits=preds, weights=loss_weight)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def multi_class_hinge_loss(preds, labels, mask):
    """Softmax hinge-loss with masking."""
    label_preds = preds[mask]
    true_classes = tf.argmax(label_preds, 0)
    idx_flattened = tf.range(0, 7) * tf.shape(label_preds) + \
                        tf.cast(true_classes, dtype=tf.int32)
    true_scores = tf.gather(tf.reshape(tf.transpose(preds), [-1]),
                            idx_flattened)
    loss = tf.nn.relu((1 - true_scores + preds) * (1 - labels))

    return tf.reduce_mean(loss)

def pricision_recall_f1(preds,labels,mask):
    """precison, recall, f1 score with msking"""
    preds,labels = tf.argmax(preds,1),tf.argmax(labels,1)
    preds,labels = tf.cast(preds,tf.float32),tf.cast(labels,tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    TP = tf.count_nonzero(preds * labels*mask)
    TN = tf.count_nonzero((preds-1) * (labels-1)*mask)
    FP = tf.count_nonzero(preds * (labels-1)*mask)
    FN = tf.count_nonzero((preds-1) * labels*mask)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2*precision*recall) /(precision + recall)
    
    return precision,recall,F1,preds

def precision_recall_f1_tpr_tnr_preds(preds, labels, mask):
    """precision, recall, f1, score, tpr, tnr, preds with masking"""
    preds_bak = preds
    preds, labels = tf.argmax(preds, 1), tf.argmax(labels, 1)
    preds, labels = tf.cast(preds, tf.float32), tf.cast(labels, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    TP = tf.count_nonzero(preds * labels*mask)
    TN = tf.count_nonzero((preds-1) * (labels-1)*mask)
    FP = tf.count_nonzero(preds * (labels-1)*mask)
    FN = tf.count_nonzero((preds-1) * labels*mask)
    
    P = tf.count_nonzero(labels * mask)
    N = tf.count_nonzero((labels-1) * mask)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2*precision*recall) /(precision + recall)
    
    return precision, recall, F1, TP/P, TN/N, preds_bak
       