import time
import numpy as np
import tensorflow as tf
import os
from models import GAT
from utils import process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', type=str, help='choose a dataset: cora, citeseer, pubmed')
parser.add_argument('--gpuid', default='0' ,type=str, help='choose a gpu to excute program:0, 1, 2, 3')
opts = parser.parse_args()

checkpt_file = 'pre_trained/'+ opts.dataset +'/mod_'+ opts.dataset +'.ckpt'

dataset = opts.dataset    #cora, citeseer, pubmed

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
###
#'cora':
#   adj:2708*2708
#   features:2708*1433
#   y_train, y_val, y_test:2708*7; y_train:前140行有类标， y_val:140-639行有类标（500行）， y_test:最后1000行有类标
#   train_mask, val_mask, test_mask:2708*1; train_mask:前140列为true, val_mask:140-639列为true（500列）, test_mask:最后1000列为true
#
#'citeseer':
#   adj:3327*3327
#   features:3327*3703
#   y_train, y_val, y_test:2708*6; y_train:前120行有类标， y_val:120-619行有类标（500行）， y_test:最后1015行中的1000行有类标（there are some isolated nodes in the graph）
#   train_mask, val_mask, test_mask:2708*1; train_mask:前120列为true, val_mask:120-619列为true（500列）, test_mask:最后1015列中的1000列为true
#
#'pubmed':
#   adj:19717*19717
#   features:19717*500
#   y_train, y_val, y_test:19717*3; y_train:前60行有类标， y_val:60-559行有类标（500行）， y_test:最后1000行有类标
#   train_mask, val_mask, test_mask:2708*1; train_mask:前60列为true, val_mask:60-559列为true（500列）, test_mask:最后1000列为true
###
features, spars = process.preprocess_features(features)     # featrues是一个规范化（每个元素除以该行和）的dense矩阵；
                                                            # spars则是features的稀疏矩阵元组表示（非零元素的坐标，非零元素值，shape）

nb_nodes = features.shape[0]    #node个数
ft_size = features.shape[1]     #feature个数
nb_classes = y_train.shape[1]   #class个数

adj = adj.todense() #dense邻接矩阵
##tt = np.sum(adj>1) 结果都等于0  说明三个数据集的邻接矩阵都是0,1矩阵
#tt = np.sum(adj.diagonal()) citeseer数据集的结果等于124 说明存在自循环
#tt = np.sum(adj.diagonal()) pubmed数据集的结果等于3 说明存在自循环
#tt = np.where(adj.diagonal()==1)

#增加一维，计算conv1d
features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)  #mask: 对角线和每个节点对应的一阶领域置为0，其余的置为-1e9


with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))    #feature matrix
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))  #mask matrix
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))   #class matrix
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))   #mask vector
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())  #attention dropout
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())   #feedforword dropout
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity) #shape=(batch_size, nb_nodes, nb_classes)

    log_resh = tf.reshape(logits, [-1, nb_classes]) #shape=(batch_size*nb_nodes, nb_classes)
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes]) #shape=(batch_size*nb_nodes, nb_classes)
    msk_resh = tf.reshape(msk_in, [-1])             #shape=(batch_size*nb_nodes,)
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh) #交叉熵损失
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)  #分类准确率

    train_op = model.training(loss, lr, l2_coef)    #tf.train.AdamOptimizer(learning_rate=lr).minimize(loss+lossL2)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpuid
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        #nb_epochs = 100000
        for epoch in range(nb_epochs):
            #trainning
            tr_step = 0
            tr_size = features.shape[0] #=1; features.shape: (1,nb_nodes,ft_size)

            while tr_step * batch_size < tr_size:   #batch_size = 1
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            #validating: without train_op
            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:  #验证集准确率提升并且验证集损失降低，则存储模型相关参数
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:   #patience = 100
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        #testing
        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
