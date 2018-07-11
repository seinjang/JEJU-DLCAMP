from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def fully_connected(inputs):
    #inputs = tf.cast(inputs, dtype=tf.int32)
    output = tf.layers.dense(inputs=inputs, units=256, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=output, units=512, activation=tf.nn.relu)

    return output
