# _*_ coding:utf-8 _*_

import tensorflow as tf
import time
import numpy as np
import os,sys
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(ABSPATH)
from inits import *


flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs, vanilla_feature=None):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            if vanilla_feature is None:
                outputs = self._call(inputs)
            else:
                outputs = self._call(inputs, vanilla_feature)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class HMGConvolution(Layer):
    """Heterogeneous Multiple Mini-Graphs Convolution Layer."""
    def __init__(self, input_dim, output_dim, input_num, adj_support,
                 num_features_nonzero, dropout_ratio, reweight_adj, beta_constant, dropout=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, use_attention=False, residual=False, **kwargs):
        super(HMGConvolution, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = dropout_ratio
        else:
            self.dropout = 0.
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_num = input_num
        self.support = adj_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        
        self.bias = bias
        self.use_attention = use_attention
        self.residual = residual
        self.reweight_adj = reweight_adj
        self.act = act
        
        self.beta_constant = beta_constant

        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero
        
        if self.use_attention:
            with tf.variable_scope('attention', reuse=True):
                self.att = tf.nn.softmax(tf.get_variable(name="att"), dim=0)

        with tf.variable_scope(self.name + '_vars'):
            if FLAGS.adj_power > 1:
                for i in range(len(self.support)):
                    for j in range(FLAGS.adj_power):
                        self.vars['weights_adj_'+str(i)+'_power_'+str(j)] = glorot([input_dim, output_dim], 
                                                           name='weights_adj_'+str(i)+'_power_'+str(j))
            elif FLAGS.multi_weight:
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            else:
                self.vars['weight'] = glorot([input_dim, output_dim], name='weight')
                
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
            
            if self.residual:
                self.vars['residual'] = glorot([FLAGS.feature_dim, output_dim], name='residual')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, vanilla_feature=None):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if FLAGS.adj_power > 1:
                cur_support = list()
                for j in range(FLAGS.adj_power):
                    w = self.vars['weights_adj_'+str(i)+'_power_'+str(j)]
                    
                    if not self.featureless:
                        pre_sup = dot(x, w, sparse=self.sparse_inputs)
                    else:
                        pre_sup = w

                    support = dot(self.support[i][j], pre_sup, sparse=True)
                    cur_support.append(tf.multiply(support, 1.0/FLAGS.adj_power))
                supports.append(tf.add_n(cur_support))
            else:
                if FLAGS.multi_weight:
                    w = self.vars['weights_' + str(i)]
                else:
                    w = self.vars['weight']

                if not self.featureless:
                    pre_sup = dot(x, w, sparse=self.sparse_inputs)
                else:
                    pre_sup = w

                if self.reweight_adj and self.sparse_inputs == False:   # dense features
                    # self-attention based on feature x
                    
                    seq = tf.expand_dims(x, axis=0)
                    out_sz = self.output_dim
                    nb_nodes = self.input_num
                    adj_mat = self.support[i]
                    coef_drop = 0.0
                    attn_drop = 0.0
                    in_drop = 0.0
                    
                    seq_fts = tf.expand_dims(pre_sup, axis=0)

                    # simplest self-attention possible
                    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
                    f_2 = tf.layers.conv1d(seq_fts, 1, 1)

                    f_1 = tf.reshape(f_1, (nb_nodes, 1))
                    f_2 = tf.reshape(f_2, (nb_nodes, 1))
                    
                    f_1 = adj_mat * f_1
                    f_2 = adj_mat * tf.transpose(f_2, [1, 0])

                    logits = tf.sparse_add(f_1, f_2)
                    lrelu = tf.SparseTensor(indices=logits.indices, 
                            values=tf.nn.leaky_relu(logits.values), 
                            dense_shape=logits.dense_shape)
                    coefs = tf.sparse_softmax(lrelu)

                    if coef_drop != 0.0:
                        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                dense_shape=coefs.dense_shape)
                    if in_drop != 0.0:
                        seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

                    # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
                    # here we make an assumption that our input is of batch size 1, and reshape appropriately.
                    # The method will fail in all other cases!
                    coefs = tf.sparse.reshape(coefs, [nb_nodes, nb_nodes])
                    coefs = tf.sparse.add(coefs.__mul__(self.beta_constant[i]), adj_mat)

                    support = tf.sparse_tensor_dense_matmul(coefs, pre_sup)
                else:
                    support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
        
        if self.use_attention:
            output = tf.add_n([tf.multiply(supports[i], tf.multiply(self.att[i], len(supports))) for i in range(len(supports))])
        else:
            output = tf.add_n(supports)
        
        # bias
        if self.bias:
            output += self.vars['bias']
        
        if self.residual and vanilla_feature is not None:
            output += dot(vanilla_feature, self.vars['residual'], sparse=True)
        return self.act(output)


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, adj_support, num_features_nonzero, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = adj_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
            if FLAGS.adj_power > 1:
                for i in range(FLAGS.adj_power - 1):
                    support = dot(self.support[i], support, sparse = True)
                    supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
