
# coding: utf-8

# In[2]:


"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools

import resnet_preprocessing


def image_serving_input_fn():
    """Serving input fn for raw images."""
    
    def _preprocess_image(image_bytes):
        """Preprocess a single raw image."""
        image = resnet_preprocessing.preprocess_image(
            image_bytes=image_bytes, is_training=False)
        return image
    
    image_bytes_list = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    images = tf.map_fn(
        _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(
        images, {'image_bytes': image_bytes_list})

class ImageInput(object):
    """Generates Image input_fn for training or evaluation."""
    
    def __init__(self, is_training, data_dir, use_bfloat16, transpose_input=True):
        self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
        self.is_training = is_training
        self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input
        
    def set_shapes(self, batch_size, images, labels):
        """Statically set the batch_size dimension."""
        if self.transpose_input:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
        else:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
            
        return images, labels
    
    def dataset_parser(self, value):
        """Parse an Imagedata record from a serialized string Tensor."""
        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, ''),
            'image/format':
                tf.FixedLenFeature((), tf.string, 'jpeg'),
            'image/class.label':
                tf.FixedLenFeature([], tf.int64, -1),
            'image/object/bbox/xmin':
                tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label':
                tf.VarLenFeature(dtype=tf.int64),
        }
        
        parsed = tf.parse_single_example(value, keys_to_features)
        image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
        
        image = self.image_preprocessing_fn(
            image_bytes=image_bytes,
            is_training=self.is_training,
            use_bfloat16=self.use_bfloat16)
        
        label = tf.cast(
            tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1
        
        return image, label
    
    def input_fn(self,params):
        """Input function which provides a single batch for train or eval."""
        if self.data_dir == None:
            tf.logging.info('Using fake input.')
            return self.input_fn_null(params)
        
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contfib.tpu.RunConfig for details.
        batch_size = params['batch_size']
        
        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)
        
        if self.is_training:
            dataset = dataset.repeat()
            
        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset
        
        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=64, sloppy=True))
        dataset = dataset.shuffle(1024)
        
        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=8, 
                drop_remainder=True))
        
        # Transpose for performance on TPU
        if self.transpose_input:
            dataset = dataset.map(
                lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
                num_parallel_calls=8)
            
        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))
        
        # Perfetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
    
    def input_fn_null(self, params):
        """Input function which provides null(black) images."""
        batch_size = params['batch_size']
        dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
        dataset = dataset.prefetch(batch_size)
        
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        if self.transpose_input:
            dataset = dataset.map(
                lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
                num_parallel_calls=8)
            
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))
        
        dataset = dataset.prefetch(32)
        tf.logging.info('Input dataset: %s', str(datset))
        return dataset
    
    def _get_null_input(self, _):
        null_image = tf.zeros([224, 224, 3], tf.bfloat16
                              if self.use_bfloat16 else tf.float32)
        return (null_image, tf.constant(0, tf.int32))
        
        
    

