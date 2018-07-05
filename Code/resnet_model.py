
# coding: utf-8

# In[5]:


"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    data_format='channels_first'):
    """Performs a batch normalization followed by a ReLU."""
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()
        
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = 3
        
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)
    
    if relu:
        inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
    """Pads the input along the spatial dimentions independently of input size."""
    pad_total = kernel_size -1
    pad_beg = pad_total // 2
    pad_end = pad_total = pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
        
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format='channels_first'):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format=data_format)
        
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def residual_block(inputs, filters, is_training, strides, use_projection=False, data_format='channels_first'):
    """Standard building block for residual networks with BN after convolutions."""
    shortcut = inputs
    if use_projection:
        # Projection shortcut in first layer to match filters and strides
        shortcut = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=strides,
            data_format=data_format)
        shortcut = batch_norm_relu(shortcut, is_training, relu=False, data_format=data_format)
        
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
    
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True, data_format=data_format)
    
    return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs, filters, is_trining, strides, use_projection=False, data_format='channels_first'):
    shortcut = inputs
    if use_projection:
        # Projection shortcut only in first block within a group. Bottlenect blocks
        # end with 4 times the number of filters.
        filters_out = 4 * filters
        shortcut = conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, data_format=data_format)
        shortcut = batch_norm_relu(shortcut, is_training, relu=False, data_format=data_format)
        
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1, data_foramt=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
    
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
    
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True, data_format=data_format)
    
    return tf.nn.relu(inputs + shortcut)


def block_group(inputs, filters, block_fn, blocks, strides, is_training, name, data_format='channels_first'):
    """Creates one group of blocks for the ResNet model."""
    # Only the first block per block_group uses projection shortcut and strides.
    inputs = block_fn(inputs, filters, is_training, strides, use_projection=True, data_format=data_format)
    
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, 1, data_format=data_format)
        
    return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, num_classes, data_format='channels_first'):
    """Generator for ResNet v1 models."""
    def model(inputs, is_training):
        """Creation of the model graph."""
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
        
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        
        inputs = block_group(
            inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1,  is_training=is_training, name='block_group1',
            data_format=data_format)
        inputs = block_group(
            inputs=inputs, filters=128, block_fn=block_fn, blocks=laters[1],
            strides=2, is_training=is_training, name='block_group2',
            data_format=data_format)
        inputs = block_group(
            inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2,  is_training=is_training, name='block_group3',
            data_format=data_format)
        inputs = block_group(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=laters[3],
            strides=2, is_training=is_training, name='block_group4',
            data_format=data_format)
        
        # The activation is 7x7 so this is a global average pool.
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=7, strides=1, padding='VALID', data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 2048 if block_fn is bottleneck_block else 512])
        return inputs
    
    model.default_image_size = 224
    return model


def resnet_v1(resnet_depth, num_classes, data_format='channels_first'):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]},
    }
    
    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)
        
    params = model_params[resnet_depth]
    return resnet_v1_generator(
        params['block'], params['layers'], num_classes, data_format)

