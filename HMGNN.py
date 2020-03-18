# _*_ coding:utf-8 _*_

from __future__ import division
from __future__ import print_function

import os
import warnings
import time
import hparams
from data_utils.data_loader import load_data
from establish_super_nodes import establish
from model_graph.models import HMGNN
from utils import *

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


def evaluate(sess, model, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['dropout']: 0.})
    outs_val = sess.run([model.loss, model.accuracy, model.evaluation], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2][:-1], (time.time() - t_test)


def main():
    train_begin = time.time()

    print(f"---------------------------------- Begin initializing FLAGS ----------------------------------")
    begin_time = time.time()
    FLAGS = hparams.create()
    FLAGS.model_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    end_time = time.time() - begin_time
    print(f"---------------------------------- Finish initializing FLAGS: time elapsed: {end_time:.3f}s -----------\n")

    print(f"---------------------------------- Begin loading data ----------------------------------")
    begin_load_data_time = time.time()
    # your own data loader can specify here
    vani_adjs, vani_ftr, vani_labels, y_train, y_test, y_val, train_mask, test_mask, val_mask = load_data(FLAGS)

    # establish super nodes
    support, features, y_train, y_val, train_mask, val_mask, super_node_num = \
        establish(FLAGS, vani_adjs, vani_ftr, vani_labels, y_train, y_test, y_val, train_mask, test_mask, val_mask)

    # nodes count
    num_supports = len(vani_adjs)                  # different kinds of graph
    normal_node_num = len(vani_ftr)                # nodes count without super nodes
    total_num = normal_node_num + super_node_num   # nodes count with super nodes

    end_load_data_time = time.time() - begin_load_data_time
    print(f"---------------------------- Finish loading data: time elapsed: {end_load_data_time:.3f}s -----------\n")

    print(f"\n---------------------------------- Begin initializing model ----- {FLAGS.model_name} --------------")
    begin_initialize = time.time()

    model_func = HMGNN
    sparse_adj_shape = [[support[i][0].shape[0], support[i][0], support[i][-1]] for i in range(num_supports)]

    # define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) if FLAGS.adj_power == 1
                    else [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.adj_power)]
                    for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # build model
    hidden_dim = [FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3, FLAGS.hidden4, FLAGS.hidden5]
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
                       sparse_adj_shape=sparse_adj_shape,
                       logging=True)
    end_initializing = time.time() - begin_initialize
    print(f"------------------- Finish initialzing model, time elapsed: {end_initializing:.3f}s -------------\n")

    # train model
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
            train_loss, train_acc, train_preds, train_eval = \
                train_outs[1], train_outs[2], train_outs[3][-1], train_outs[3][:-1]

            train_time = time.time() - epoch_begin

            # Validation
            val_loss, val_acc, val_eval, val_time = evaluate(sess, model, features, support, y_val, val_mask, placeholders)
            val_loss_list.append(val_loss)

            epoch_end = time.time() - epoch_begin
            # Print results
            print(
                f"Epoch:{epoch + 1:3d},   loss    acc    time, time elapsed={epoch_end:.3f}s --------")
            print(f"Train:     {train_loss:.5f} {train_acc:.5f} {train_time:.3f}s")
            print(f"Valid:     {val_loss:.5f} {val_acc:.5f} {val_time:.3f}s")

            if FLAGS.attention and epoch > 0 and epoch % 20 == 0:
                print(f"subgraph attention: {[_[0] for _ in sess.run(model.att)]}")

            if val_eval[2] > best_f1:
                best_f1 = max(val_eval[2], best_f1)
                if FLAGS.model_version >= 0:
                    save_name = FLAGS.model_name + "-Version" + str(FLAGS.model_version)
                else:
                    save_name = FLAGS.model_name  # "GCN"
                checkpoint_path = os.path.join(FLAGS.model_dir, save_name)
                model.save(checkpoint_path, sess, epoch, saver)
            print("")
    train_end_time = time.time() - train_begin_time
    print(f"finish training process, time elapsed: {train_end_time:.3f}s ...................")

    train_end = time.time() - train_begin
    print(f"----------------------- Total Training Time = {train_end:.3f}s----------------------------")


if __name__ == "__main__":
    main()
