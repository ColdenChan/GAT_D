import sys
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import numpy as np
import tensorflow as tf


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):   #python版本为3.0以后
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

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
    return features.todense(), sparse_to_tuple(features)
features, spars = preprocess_features(features)

def adj_to_bias(adj, t, sizes, nhood=1):
    nb_graphs = adj.shape[0]  # 第一维是新增加的，维数为1 np.newaixs
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            # mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))      #单位矩阵*（邻接矩阵+单位矩阵）,所以该步有点多余,改为下句
            mt[g] = adj[g] + np.eye(adj.shape[1])  # 邻接矩阵+单位阵 （加了个自循环）
        # 经过前面验证，传进来的邻接矩阵是只有0和1的，再加上一个单位阵
        # for i in range(sizes[g]):
        #     for j in range(sizes[g]):
        #         if mt[g][i][j] > 0.0:
        #             mt[g][i][j] = 1.0
        mt[g][mt[g] > 1.0] = 1.0

    adj2 = np.array(np.matmul(adj[g], adj[g]),dtype=np.float)
    adj2 = adj2 - np.diag(adj2.diagonal())
    adj2[adj2 > 0] = t
    adj2 = adj2[np.newaxis]
    adj12 = mt + adj2
    adj12[adj12 == 0] = -1e9
    adj12[adj12 == 1] = 0
    adj12[adj12 == 1 + t] = 0
    return adj12

nb_nodes = features.shape[0]    #node个数
ft_size = features.shape[1]     #feature个数
nb_classes = y_train.shape[1]   #class个数

adj = adj.todense()
                                #增加一维
features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = adj_to_bias(adj, -0.1, [nb_nodes], nhood=1)

conv1d = tf.layers.conv1d
#layers.attn_head(inputs, bias_mat=bias_mat, out_sz=hid_units[0], activation=activation, in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation
tr_step = 0
batch_size = 1
ftr_in = features[0]
bias_in = biases[0]
attn_head(tf.convert_to_tensor(features), bias_mat=tf.convert_to_tensor(biases,dtype=tf.float32), out_sz=8, activation=tf.nn.elu, in_drop=0, coef_drop=0, residual=False)