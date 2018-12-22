import sys
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import numpy as np
import tensorflow as tf
from models import SpGAT
from utils import process
model = SpGAT
from utils import layers

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data('cora')
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.preprocess_adj_bias(adj)


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print(f_1.shape)
        print('adj_mat.dense_shape: ')
        print(sess.run(adj_mat.dense_shape))
        print('adj_mat.values: ')
        print(adj_mat.values.shape)
        f_1 = adj_mat * f_1
        print('f_1.dense_shape: ')
        print(sess.run(f_1.dense_shape))
        print('f_1.values: ')
        print(f_1.values.shape)
        ff = tf.transpose(f_2, [1, 0])
        print(ff.shape)
        f_2 = adj_mat * ff
        print('f_2.dense_shape: ')
        print(sess.run(f_2.dense_shape))
        print('f_2.values: ')
        print(f_2.values.shape)

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
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

sp_attn_head(tf.convert_to_tensor(features),out_sz=8,adj_mat=tf.SparseTensor(biases[0],biases[1],biases[2]),activation=tf.nn.elu,nb_nodes=nb_nodes)