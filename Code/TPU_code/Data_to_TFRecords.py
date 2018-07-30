import math
import os
import random
import tarfile
import urllib
import json

from absl import flags
import tensorflow as tf
import numpy as np
#from google.cloud import storage

# 두 지점 사이의 거리 계산
from math import sin, cos, sqrt, atan2, radians

def dis(lat, lng, cen_lat, cen_lng):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat)
    lon1 = radians(lng)
    lat2 = radians(cen_lat)
    lon2 = radians(cen_lng)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# convert the dataset into TF-Records
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width,
                        keyword, question, center, boundary, places, num_place):
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'
    text_length = 10
    places_x = np.array(places)[:,0]
    places_y = np.array(places)[:,1]

    example = tf.train.Example(features=tf.train.Features(feature={
        'text/question': _int64_feature(question),
        'text/keywords/key': _int64_feature(keyword),
        'image/center/x': _float_feature([center[0]]),
        'image/center/y': _float_feature([center[1]]),
        'image/places/x': _float_feature(places_x),
        'image/places/y': _float_feature(places_y),
        'image/places/number': _int64_feature(num_place),
        'image/boundary/xmin': _float_feature([boundary[0]]),
        'image/boundary/xmax': _float_feature([boundary[1]]),
        'image/boundary/ymin': _float_feature([boundary[2]]),
        'image/boundary/ymax': _float_feature([boundary[3]]),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/object/label': _int64_feature(label),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(str.encode(filename))),
        'image/encoded': _bytes_feature(image_buffer)}))

    return example


class ImageCoder(object):

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.PNG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb',) as f:
        image_data = f.read()

    # PNG to JPEG
    tf.logging.info('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, data):
    """Processes and saves list of images as TFRecords.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        output_file: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        labels: map of string to integer; id for all synset labels
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    questions = []
    keywords = []
    centers = []
    boundaries = []
    places = []
    labels = []
    num_places = []
    idx = 0

    keyword = ['beachfront', 'beach', 'cafe', 'coffee', 'nearest', 'near',
               'restaurant', 'food']
    key = {k: v for v, k in enumerate(keyword)}



    for i in range(len(data)):
        if data[i]['question'][0] == 'Where is the beachfront cafe':
            questions.append([1,0,0])
        elif data[i]['question'][0] == 'Where is the beachfront restaurant':
            questions.append([0,1,0])
        else:
            questions.append([0,0,1])
        keywords.append([key[k] for k in data[i]['keyword']])
        centers.append(data[i]['center'])
        boundaries.append(data[i]['Bounds'])

        poi_inside = []
        for poi in data[i]['allpoi']:
            for boundary in [data[i]['Bounds']]:
                if poi[0] < boundary[0] and poi[0] > boundary[1] and poi[1] < boundary[2] and poi[1] > boundary[3]:
                    poi_inside.append(poi)
        num_places.append(len(poi_inside))

        if len(poi_inside) < 20:
            for i in range(20-len(poi_inside)):
                poi_inside.append([0, 0])

        else:
            poi_inside = random.sample(poi_inside, 20)
        places.append(poi_inside)

        c = []
        for point in poi_inside:
            if point != 0:
                c.append(dis(point[0], point[1], data[0]['center'][0], data[0]['center'][1]))
        poi_label = poi_inside[c.index(min(c))]

        label = []

        for i in range(20):
            if poi_inside[i] == poi_label:
                label.append(1)
            else:
                label.append(0)
        labels.append(label)

    for filename in (filenames):
        image_buffer, height, width = _process_image(filename, coder)
        label = labels[idx]
        question = questions[idx]
        keyword = keywords[idx]
        center = centers[idx]
        boundary = boundaries[idx]
        place = places[idx]
        num_place = num_places[idx]
        example = _convert_to_example(filename, image_buffer, label, height, width,
                        keyword, question, center, boundary, place, num_place)
        writer.write(example.SerializeToString())
        break
        idx += 1
    writer.close()


def _process_dataset(filenames, output_directory, prefix, num_shards, data):
    """Processes and saves list of images as TFRecords.

    Args:
        filenames: list of strings; each string is a path to an image file
        labels: map of string to integer; id for all synset labels
        output_directory: path where output files should be created
        prefix: string; prefix for each file
        num_shards: number of chucks to split the filenames into

    Returns:
        files: list of tf-record filepaths created from processing the dataset.
    """
    chunksize = int(math.ceil(len(filenames) / num_shards))
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files, data)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the Image dataset into TF-Record dumps."""
    # Get JSON DATA
    with open('../Dataset/train.json') as f:
        train_data = json.load(f)

    with open('../Dataset/validation.json') as f:
        validation_data = json.load(f)

    # Shuffle training records to ensure we are distributing classes
    random.seed(0)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)
        return order

    # Glob all the training files
    training_files = tf.gfile.Glob(
        os.path.join(raw_data_dir, 'train_image', '*.PNG'))
    #print(training_files)
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]

    # Glob all the validation files
    validation_files = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, 'validation_image', '*.PNG')))

    # Create training data
    tf.logging.info('Processing the training data.')
    training_records = _process_dataset(
        training_files, os.path.join('../Dataset/', 'train'), 'train', 1216, train_data)

    # Create validation data
    tf.logging.info('Processing the validation data.')
    validation_records = _process_dataset(
        validation_files, os.path.join('../Dataset/', 'validation'), 'validation', 304, validation_data)

    return training_records, validation_records



def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    training_records, validation_records = convert_to_tf_records('../Dataset/')

if __name__ == '__main__':
    tf.app.run()
