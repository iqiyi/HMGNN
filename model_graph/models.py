# _*_ coding:utf-8 _*_

import tensorflow as tf
import os,sys
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(ABSPATH)
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    #with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        #_alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
        #                         dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.residual = False

        self.inputs = None
        self.outputs = None
        self.preds = None
        self.probs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.evaluation = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
        # with tf.compat.v1.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        hidden = None
        self.activations.append(self.inputs)
        for layer in self.layers:
            if self.residual:
                hidden = layer(self.activations[-1], self.inputs)
            else:
                hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        # -------------------------------
        self.outputs = self.activations[-1]
        preds = tf.argmax(self.predict(), 1)
        self.preds = tf.cast(preds,tf.float32)
        self.probs = tf.reduce_max(self.predict(), reduction_indices=[1])
        # -------------------------------

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        #variables = tf.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        # variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._evaluate()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def save(self, checkpoint_path, sess=None, step=0, saver=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not saver:
            saver = tf.train.Saver(self.vars, max_to_keep=5)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = saver.save(sess, checkpoint_path, global_step=step)
        print("Model saved in file: %s" % save_path)

    def load(self, checkpoint_path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars, max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print(f"checkpoint_path = {checkpoint_path} ckpt = {ckpt}")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from file: %s" % ckpt.model_checkpoint_path)
        
class HMMG(Model):
    def __init__(self, placeholders, input_dim, hidden_dim, output_dim, input_num, normal_node_num, 
                 support_num, reweight_adj, residual, attention, sparse_adj_shape, pure_support=None, **kwargs):
        super(HMMG, self).__init__(**kwargs)
        
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        
        logging = kwargs.get('logging', False)
        self.logging = logging
        
        self.vars = {}
        self.layers = []
        self.activations = []
        
        self.outputs = None
        self.preds = None
        self.probs = None

        self.loss = 0
        self.accuracy = 0
        self.opt_op = None
        self.evaluation = None
        
        self.inputs = placeholders['features']
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.support = placeholders["support"]
        self.labels = placeholders["labels"]
        self.labels_mask = placeholders["labels_mask"]
        self.dropout_ratio = placeholders["dropout"]
        # self.loss_weight = placeholders["loss_weight"]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_num = input_num
        self.normal_node_num = normal_node_num
        self.support_num = support_num
        
        self.sparse_adj_shape = sparse_adj_shape
        self.beta_constant = []
        for i in range(self.support_num):
            beta_values = tf.constant(np.array([FLAGS.beta for i in range(self.sparse_adj_shape[i][2][0])]), dtype=tf.float32)
            self.beta_constant.append(beta_values)
        
        
        self.reweight_adj = reweight_adj
        self.residual = residual
        self.use_attention = attention

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        for var in self.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)
        # self.loss += weighted_masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask, self.loss_weight)
        tf.summary.scalar("loss", self.loss)
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.labels_mask)
    def _evaluate(self):
        '''evaluation = [precision, recall, f1, tpr, tnr, preds]'''
        self.evaluation = precision_recall_f1_tpr_tnr_preds(self.outputs,self.labels, self.labels_mask)
        eval_names = ['precision', 'recall', 'F1', 'TPR', 'TNR']
        for index in range(len(eval_names)):
            tf.summary.scalar(eval_names[index], self.evaluation[index])

    def _build(self):
        # the input and output dimension of middle layers, including first and last layer
        # for vanilla GCN, mid_dim = [self.input_dim, FLAGS.hidden1, self.output_dim] = [feature_dim, 16, 1]
        
        mid_dim = [self.input_dim] 
        mid_dim.extend(self.hidden_dim)
        mid_dim.append(self.output_dim)

        # create attention variables
        if self.use_attention:
            with tf.variable_scope("attention"):
                # print(f"tf.get_variable_scope().original_name_scope = {tf.compat.v1.get_variable_scope().original_name_scope}")
                if FLAGS.adj_power > 10:
                    shape = [FLAGS.adj_power, 1]
                    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
                    self.att = []
                    for i in range(self.support_num):
                        initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
                        self.att.append(tf.nn.softmax(tf.get_variable("att_"+str(i), initializer=initial), dim=0))
                else:
                    shape = [self.support_num, 1]
                    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
                    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
                    self.att = tf.nn.softmax(tf.get_variable("att", initializer=initial), dim=0)
        

        for i in range(len(mid_dim) - 1):
            input_dim, output_dim = mid_dim[i], mid_dim[i+1]
            residual = self.residual
            if i < len(mid_dim) - 2:
                activation = tf.nn.elu   # middle layer
            else:
                activation = lambda x: x            # last layer
                residual = False

            if i > 0: sparse_input = False  # middle layer
            else: sparse_input = True      # first layer

            self.layers.append(HMMGConvolution(input_dim=input_dim,
                                    output_dim=output_dim,
                                    input_num=self.input_num,
                                    adj_support = self.support,
                                    num_features_nonzero = self.num_features_nonzero,
                                    dropout_ratio=self.dropout_ratio,
                                    act=activation,
                                    dropout=True,
                                    sparse_inputs = sparse_input,
                                    logging=self.logging,
                                    reweight_adj=self.reweight_adj,
                                    use_attention=self.use_attention,
                                    beta_constant=self.beta_constant,
                                    residual=residual))

    def predict(self):
        return tf.nn.softmax(self.outputs)
    
    def save(self, checkpoint_path, sess=None, step=0, saver=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not saver:
            saver = tf.train.Saver(self.vars, max_to_keep=5)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = saver.save(sess, checkpoint_path, global_step=step)
        print("Model saved in file: %s" % save_path)

    def load(self, checkpoint_path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars, max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print(f"checkpoint_path = {checkpoint_path} ckpt = {ckpt}")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from file: %s" % ckpt.model_checkpoint_path)
