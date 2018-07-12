"""Train a ResNet-34 model on TPU"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

from absl import flags
import absl.logging as _logging
import tensorflow as tf

import Dataset_input
import resnet_model # resnet for process map image
import text_model # densenet for process question
import relationnet_model # for relation reasoning

from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=None,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'resnet_depth', default=34,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=10000,
    help=('The number of steps to use for training. Default is 112603 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=128, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=128, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1206, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=304, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=20, help='Number of classes, at least 2')

flags.DEFINE_integer(
    'steps_per_eval', default=1000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help=(
        'Maximum seconds between checkpoints before evaluation terminates.'))

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=10,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

# TODO(chrisying): remove this flag once --transpose_tpu_infeed flag is enabled
# by default for TPU
flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_string(
    'precision', default='bfloat16',
    help=('Precision to use; one of: {bfloat16, float32}'))

flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))

# Learning rate schedule
LR_SCHEDULE = [ # (vultiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def learning_rate_schedule(current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay."""
    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

    decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                  current_epoch / LR_SCHEDULE[0][1])
    for mult, start_epoch in LR_SCHEDULE:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)
    return decay_rate


def resnet_model_fn(features, labels, mode, params):
    """The model_fn for ResNet to be used with TPUEstimator."""

    if isinstance(features, dict):
        feature_image = features['image']
        feature_question = features['question']

    """
    if FLAGS.data_format == 'channels_first':
        assert not FLAGS.transpose_input
        feature_image = tf.transpose(feature_image, [0, 3, 1, 2])

    if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        feature_image = tf.transpose(feature_image, [3, 0, 1, 2])
    """

    # Normalize the image to zero mean and unit variance.
    
    # on the TPU
    if FLAGS.use_tpu:
        feature_image -= tf.constant(MEAN_RGB, shape=[16, 1, 1, 3], dtype=feature_image.dtype)
        feature_image /= tf.constant(STDDEV_RGB, shape=[16, 1, 1, 3], dtype=feature_image.dtype)
    else:
        # on the CPU
        feature_image -= tf.constant(MEAN_RGB, shape=[128, 1, 1, 3], dtype=feature_image.dtype)
        feature_image /= tf.constant(STDDEV_RGB, shape=[128, 1, 1, 3], dtype=feature_image.dtype)

    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def resnet_network():
        network = resnet_model.resnet_v1(
            resnet_depth=FLAGS.resnet_depth,
            data_format=FLAGS.data_format)
        return network(
            inputs=feature_image, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    def text_network():
        return text_model.fully_connected(
            inputs=feature_question)

    def relation_network(feature_maps, text_inputs):
        return relationnet_model.relationnet(feature_maps, text_inputs)


    image_feature_maps = resnet_network() # feature maps from resnet
    text_feature = text_network()
    logits = relation_network(image_feature_maps, text_feature)
    # raise ValueError(image_feature_maps, text_feature, rn_feature)

    batch_size = params['batch_size']

    # Calulate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = labels
    one_hot_labels = tf.cast(one_hot_labels, dtype=tf.float32)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name])


    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step = tf.train.get_global_step()
        batches_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
        current_epoch = (tf.cast(global_step, tf.float32) / batches_per_epoch)
        learning_rate = learning_rate_schedule(current_epoch)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)


def main(unused_argv):

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)
    else:
        tpu_cluster_resolver = None

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=max(600, FLAGS.iterations_per_loop),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_cores))
            # per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long


    RN_classifier = tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
        'Invalid value for --precision flag; must be bfloat16 or float32.')
    tf.logging.info('Precision: %s', FLAGS.precision)
    use_bfloat16 = FLAGS.precision == 'bfloat16'

    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    imagenet_train, imagenet_eval = [Dataset_input.ImageInput(
        is_training=is_training,
        data_dir=FLAGS.data_dir,
        transpose_input=FLAGS.transpose_input,
        use_bfloat16=use_bfloat16) for is_training in [True, False]]

    if FLAGS.mode == 'eval':
        eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

        # Run evaluation when there's a new checkpoint
        for ckpt in evaluation.checkpoints_iterator(
            FLAGS.model_dir, timeout=FLAGS.eval_timeout):
            tf.logging.info('Starting to evaluate.')
            try:
                start_timestamp = time.time()  # This time will include compilation time
                eval_results = RN_classifier.evaluate(
                    input_fn=imagenet_eval.input_fn,
                    steps=eval_steps,
                    checkpoint_path=ckpt)
                elapsed_time = int(time.time() - start_timestamp)
                tf.logging.info('Eval results: %s. Elapsed seconds: %d' %
                                (eval_results, elapsed_time))

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                if current_step >= FLAGS.train_steps:
                    tf.logging.info(
                        'Evaluation finished after training step %d' % current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

    else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
        current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
        batches_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
        tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                        ' step %d.' % (FLAGS.train_steps,
                                       FLAGS.train_steps / batches_per_epoch,
                                       current_step))

        start_timestamp = time.time()  # This time will include compilation time
        if FLAGS.mode == 'train':
            RN_classifier.train(
                input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)

        else:
            assert FLAGS.mode == 'train_and_eval'
            while current_step < FLAGS.train_steps:
                # Train for up to steps_per_eval number of steps.
                # At the end of training, a checkpoint will be written to --model_dir.
                next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                                      FLAGS.train_steps)
                RN_classifier.train(
                    input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
                current_step = next_checkpoint

                # Evaluate the model on the most recent model in --model_dir.
                # Since evaluation happens in batches of --eval_batch_size, some images
                # may be consistently excluded modulo the batch size.
                tf.logging.info('Starting to evaluate.')
                eval_results = RN_classifier.evaluate(
                    input_fn=imagenet_eval.input_fn,
                    steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
                tf.logging.info('Eval results: %s' % eval_results)


            elapsed_time = int(time.time() - start_timestamp)
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                            (FLAGS.train_steps, elapsed_time))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
