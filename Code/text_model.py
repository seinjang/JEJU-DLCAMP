from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def fully_connected(inputs):
    x = tf.layers.dense(inputs=inputs, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)

    return x
