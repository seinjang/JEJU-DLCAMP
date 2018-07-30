from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

def relationnet(image_object, text_object):
    return generate_object(image_object, text_object)


def rn_g(i, j, q, scope='g_', reuse=True):
    rn_g_1 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=256, activation=tf.nn.relu, name='g_1', reuse=reuse)
    rn_g_2 = tf.layers.dense(inputs=rn_g_1, units=256, activation=tf.nn.relu, name='g_2', reuse=reuse)
    rn_g_3 = tf.layers.dense(inputs=rn_g_2, units=256, activation=tf.nn.relu, name='g_3', reuse=reuse)
    rn_g_4 = tf.layers.dense(inputs=rn_g_3, units=256, activation=tf.nn.relu, name='g_4', reuse=reuse)

    return rn_g_4

def rn_f(g):
    logging.info('Relation_f')
    rn_f_1 = tf.layers.dense(inputs=g, units=256, activation=tf.nn.relu)
    rn_f_2 = tf.layers.dense(inputs=rn_f_1, units=256, activation=tf.nn.relu)
    rn_f_2 = tf.layers.dropout(inputs=rn_f_2, training=True, rate=0.5)
    rn_f_3 = tf.layers.dense(inputs=rn_f_2, units=10, activation=None)

    return rn_f_3

def concat_coor(o, i, d):
    coor = tf.tile(tf.expand_dims(
        [float(int(i / d)) / d, (i % d) / d], axis=0), [32, 1])
    o = tf.concat([o, tf.to_float(coor)], axis=1)
    return o

def generate_object(image_object, text_object):
    d = image_object.get_shape().as_list()[1]
    G = []
    for i in range(d*d):
        object_i = image_object[:, int(i / d), int(i % d), :]
        object_i = concat_coor(object_i, i, d)
        for j in range(d*d):
            object_j = image_object[:, int(j / d), int(j % d), :]
            object_j = concat_coor(object_j, j, d)
            if i == 0 and j == 0:
                g = rn_g(object_i, object_j, text_object, reuse=False)
            else:
                g = rn_g(object_i, object_j, text_object, reuse=True)
        G.append(g)
    G = tf.stack(G, axis=0)
    G = tf.reduce_mean(G, axis=0, name='G')
    logits = rn_f(G)

    return logits
