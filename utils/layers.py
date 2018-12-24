import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d
#layers.attn_head(inputs, bias_mat=bias_mat, out_sz=hid_units[0], activation=activation, in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop) #shape: nb_nodes*ft_size

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  #shape: batch_size, nb_nodes, out_sz

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)   #shape: batch_size, nb_nodes, 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)   #shape: batch_size, nb_nodes, 1
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) #shape: batch_size, nb_nodes, nb_nodes
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  #attention coefficients
                                                                    # shape: batch_size, nb_nodes, nb_nodes
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)        #shape: batch_size, nb_nodes, out_sz
        ret = tf.contrib.layers.bias_add(vals)  #Adds a bias to the vals.
                                                #shape: batch_size, nb_nodes, out_sz

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq #shape: batch_size, nb_nodes, out_sz

        return activation(ret)  # activation #shape: batch_size, nb_nodes, out_sz

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes,W1, W2, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  #shape: batch_size, nb_nodes, out_sz

        f_1 = tf.matmul(seq_fts, W1)          #shape: batch_size, nb_nodes, n_dimension
        f_2 = tf.matmul(seq_fts, W2)          #shape: batch_size, nb_nodes, n_dimension
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)   #shape: batch_size, nb_nodes, 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)   #shape: batch_size, nb_nodes, 1

        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))    #shape: batch_size*nb_nodes, n_dimension
        f_2 = tf.reshape(f_2, (nb_nodes, 1))    #shape: batch_size*nb_nodes, n_dimension

        # f_1 = tf.contrib.layers.bias_add(f_1)
        # f_2 = tf.contrib.layers.bias_add(f_2)
        #
        # f_1 = tf.expand_dims(f_1, 1)            #shape: batch_size*nb_nodes, 1, n_dimension
        # f_2 = tf.expand_dims(f_2, 0)            #shape: 1, batch_size*nb_nodes, n_dimension

        W = tf.Variable(tf.random_normal([nb_nodes,nb_nodes], stddev=0.35))
        adj_mat = adj_mat*W
                                                #adj_mat:SparseTensor(indices=Tensor(shape(非零元素个数, 2)), values=Tensor(shape(非零元素个数,)), dense_shape=Tensor(shape(2,)))
        #f_1的每一个元素分别乘以adj_mat的每一列     #adj_mat: shape= nb_nodes, nb_nodes
        f_1 = adj_mat*f_1                       #SparseTensor(indices=Tensor(shape(非零元素个数, 2)), values=Tensor(shape(非零元素个数,)), dense_shape=Tensor(shape(2,)))
        #f_2转置后变成行向量Tensor，稀疏Tensor和行向量Tensor相乘的结果是f_2的每一个元素分别乘以adj_mat的每一行
                                                #f_1: shape = nb_nodes, nb_nodes
        f_2 = adj_mat * tf.transpose(f_2, [1,0])#SparseTensor(indices=Tensor(shape(非零元素个数, 2)), values=Tensor(shape(非零元素个数,)), dense_shape=Tensor(shape(2,)))

        logits = tf.sparse_add(f_1, f_2)        #SparseTensor(indices=Tensor(shape(?, 2)), values=Tensor(shape(?,)), dense_shape=Tensor(shape(2,)))
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), #对value进行激活
                dense_shape=logits.dense_shape) #SparseTensor(indices=Tensor(shape(?, 2)), values=Tensor(shape(?,)), dense_shape=Tensor(shape(2,)))
        coefs = tf.sparse_softmax(lrelu)        #SparseTensor(indices=Tensor(shape(?, 2)), values=Tensor(shape(?,)), dense_shape=Tensor(shape(2,)))

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])  #SparseTensor(indices=Tensor(shape(?, 2)), values=Tensor(shape(?,)), dense_shape=Tensor(shape(2,)))
        seq_fts = tf.squeeze(seq_fts)   #该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果 shape：nb_nodes, out_sz
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)    #shape：nb_nodes, out_sz
        vals = tf.expand_dims(vals, axis=0)                     #shape：1, nb_nodes, out_sz
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)                  #shape：1, nb_nodes, out_sz

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

