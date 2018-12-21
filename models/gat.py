import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
#inference(ftr_in, nb_classes, nb_nodes, is_train, attn_drop, ffd_drop,
        # bias_mat=bias_in, hid_units=hid_units, n_heads=n_heads, residual=residual, activation=nonlinearity)
        #======================================
        #hid_units = [8] # numbers of hidden units per each attention head in each layer
        #n_heads = [8, 1] # additional entry for the output layer
        #residual = False
        #nonlinearity = tf.nn.elu
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []  #存放n_heads[0]=8个attention加权过后的特征， 每一个的shape: batch_size, nb_nodes, out_sz
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1) #隐藏层1   shape: batch_size, nb_nodes, out_sz*n_heads[0]
        for i in range(1, len(hid_units)):  #类似处理接下来的隐藏层
            #h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []    #n_heads[-1]个 shape: batch_size, nb_nodes, out_sz
        for i in range(n_heads[-1]):    #处理最后一层，nb_classes：类别数
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits   #shape: batch_size, nb_nodes, nb_classes
