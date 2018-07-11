from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def relationnet(image_object, text_object):
    return generate_object(image_object, text_object)


def rn(i, j, q):
    rn_1 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=192, activation=tf.nn.relu)
    rn_2 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=192, activation=tf.nn.relu)
    rn_3 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=20, activation=tf.nn.relu)
    #rn_4 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=96, activation=tf.nn.relu)
    #rn_5 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=48, activation=tf.nn.relu)
    #rn_6 = tf.layers.dense(inputs=tf.concat([i, j, q], axis=1), units=20, activation=tf.nn.relu)
    return rn_3


def generate_object(image_object, text_object):
    # image_object has shape [Batch, d, d, k] which is [128, 7, 7, 512]
    # text_object has shape [Batch, k] which is [128, 512]
    d = image_object.get_shape().as_list()[1]
    G = []
    for i in range(d*d):
        object_i = image_object[:, int(i / d), int(i % d), :]
        for j in range(d*d):
            object_j = image_object[:, int(j / d), int(j % d), :]
            g = rn(object_i, object_j, text_object)
        G.append(g)
    G = tf.stack(G, axis=0)
    G = tf.reduce_mean(G, axis=0, name='G')
    return G
