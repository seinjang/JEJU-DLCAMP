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
        # self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

    def set_shapes(self, batch_size, images, questions, labels,
                keywords, centers, places, num_places, boundaries):
        """Statically set the batch_size dimension."""
        if self.transpose_input:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size, 20])))
            questions.set_shape(questions.get_shape().merge_with(
                tf.TensorShape([batch_size, 3])))
            keywords.set_shape(keywords.get_shape().merge_with(
                tf.TensorShape([batch_size, 4])))
            centers.set_shape(centers.get_shape().merge_with(
                tf.TensorShape([batch_size, None])))
            places.set_shape(places.get_shape().merge_with(
                tf.TensorShape([batch_size, None, 20])))
            num_places.set_shape(num_places.get_shape().merge_with(
                tf.TensorShape([batch_size])))
            boundaries.set_shape(boundaries.get_shape().merge_with(
                tf.TensorShape([batch_size, None])))

        else:
            """
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size, 20])))
            questions.set_shape(questions.get_shape().merge_with(
                tf.TensorShape([batch_size, 3])))
            keywords.set_shape(keywords.get_shape().merge_with(
                tf.TensorShape([batch_size, 4])))
            centers.set_shape(centers.get_shape().merge_with(
                tf.TensorShape([batch_size, None])))
            places.set_shape(places.get_shape().merge_with(
                tf.TensorShape([batch_size, None, 20])))
            num_places.set_shape(num_places.get_shape().merge_with(
                tf.TensorShape([batch_size])))
            boundaries.set_shape(boundaries.get_shape().merge_with(
                tf.TensorShape([batch_size, None])))
            """
        return images, questions, labels, keywords, centers, places, num_places, boundaries

    def dataset_parser(self, value):
        """Parse an Imagedata record from a serialized string Tensor."""
        keys_to_features = {
            'text/question':tf.VarLenFeature(tf.int64),
            'text/keywords/key':tf.VarLenFeature(tf.int64),
            'image/center/x':tf.FixedLenFeature((), tf.float32),
            'image/center/y':tf.FixedLenFeature((), tf.float32),
            'image/places/x':tf.VarLenFeature(tf.float32),
            'image/places/y':tf.VarLenFeature(tf.float32),
            'image/places/number':tf.FixedLenFeature([], tf.int64),
            'image/boundary/xmin':tf.FixedLenFeature((), tf.float32),
            'image/boundary/xmax':tf.FixedLenFeature((), tf.float32),
            'image/boundary/ymin':tf.FixedLenFeature((), tf.float32),
            'image/boundary/ymax':tf.FixedLenFeature((), tf.float32),
            'image/object/label':tf.VarLenFeature(tf.int64),
            'image/encoded':tf.FixedLenFeature((), tf.string, '')
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        question = tf.sparse_tensor_to_dense(parsed['text/question'])
        keyword = tf.sparse_tensor_to_dense(parsed['text/keywords/key'])
        center = tf.stack([parsed['image/center/x'],parsed['image/center/y']])
        place = tf.stack([tf.sparse_tensor_to_dense(parsed['image/places/x']),
                           tf.sparse_tensor_to_dense(parsed['image/places/y'])])
        num_place = parsed['image/places/number']
        boundary = tf.stack([parsed['image/boundary/xmin'],
                             parsed['image/boundary/xmax'],
                             parsed['image/boundary/ymin'],
                             parsed['image/boundary/ymax']])
        label = tf.sparse_tensor_to_dense(parsed['image/object/label'])
        image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
        image = self.image_preprocessing_fn(
            image_bytes=image_bytes,
            is_training=self.is_training)

        return image, question, label, keyword, center, place, num_place, boundary


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
            self.data_dir, 'train/*' if self.is_training else 'validation/*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)


        if self.is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 100 * 100
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=64, sloppy=True))
        dataset = dataset.shuffle(100)

        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=8,
                drop_remainder=True))

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Perfetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        (image, question, label, keyword,
        center, place, num_place, boundary) = dataset.make_one_shot_iterator().get_next()
        features = {}
        features['image'] = image
        features['question'] = tf.cast(question, dtype=tf.float32)
        features['keyword'] = tf.cast(keyword, dtype=tf.float32)
        features['center'] = center
        features['place'] = place
        features['num_place'] = tf.cast(tf.shape(tf.expand_dims(num_place, 1)), dtype=tf.float32)
        features['boundary'] = boundary
        features['question'] = tf.concat([features['question'], features['keyword'],
                                    features['center'], features['num_place'],
                                    features['place'][:,0,:], features['place'][:,1,:],
                                    features['boundary']],1)
        return features, label

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
