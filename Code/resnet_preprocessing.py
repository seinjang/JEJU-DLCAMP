"""Preprocessing for ResNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32


def _decode_and_center(image_bytes):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize_bicubic([image], [IMAGE_SIZE, IMAGE_SIZE])[0]
    
    return image

def preprocess_for_train(image_bytes, use_bfloat16):
    image = _decode_and_center(image_bytes)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(
        image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image

def preprocess_for_eval(image_bytes, use_bfloat16):
    image = _decode_and_center(image_bytes)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(
        image, dtype=tf.bloat16 if use_bfloat16 else tf.float32)
    return image

def preprocess_image(image_bytes, is_training=False, use_bfloat16=False):
    if is_training:
        return preprocess_for_train(image_bytes, use_bfloat16)
    else:
        return preprocess_for_eval(image_bytes, use_bfloat16)
    
