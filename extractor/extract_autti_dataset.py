import io
import os
import logging
import logging.config

import pandas as pd
import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.protos import string_int_label_map_pb2
from utils import get_all_labels, setup_logging
import cv2

flags = tf.app.flags
flags.DEFINE_string('label_map_path', 'data/autti/label_map_path.pbtxt', 'Path to label map proto')
flags.DEFINE_string('input_path', 'data/autti/labels.csv', 'Path to label map proto')
flags.DEFINE_string('train_path', 'data/autti/train.record', 'train tf_records path')
flags.DEFINE_string('val_path', 'data/autti/val.record', 'test tf_records path')
flags.DEFINE_string('debug_dir', '/tmp/autti/', 'debugging directory')
flags.DEFINE_string('data_dir', 'data/autti/', 'Data directory')
flags.DEFINE_integer('image_width', 1920, 'Image Width')
flags.DEFINE_integer('image_height', 1200, 'Image Height')
flags.DEFINE_integer('num_class', 5, 'number of classes')
flags.DEFINE_string('image_encoding', "b'jpg'", 'Image encoding')
FLAGS = flags.FLAGS


setup_logging()

log = logging.getLogger('application_log')

def extract_label(data, label_map_path):

    labels = set(data['label'].values)
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()

    for i, label in enumerate(labels):
        item = label_map_proto.item.add()
        item.id = i+1
        item.name = label
        item.display_name = label

    with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(str(label_map_proto))


def extract_tf_record(examples, output, label2IdMap):
    writer = tf.python_io.TFRecordWriter(output)


    label_map = label_map_util.load_labelmap(FLAGS.label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=FLAGS.num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    log.debug('CATEGORY INDEX %s', category_index)

    for frame, example in examples.groupby(['frame']):
        log.debug('Creating tf_record for %s', frame)
        tf_example = create_tf_example(example, label2IdMap, category_index)
        writer.write(tf_example.SerializeToString())
    writer.close()


def create_tf_example(example, label2IdMap, category_index):
    filename = os.path.join(FLAGS.data_dir, example['frame'].values[0])  # Filename of the image. Empty if image is not from file
    ofilename = os.path.join(FLAGS.debug_dir, 'images', example['frame'].values[0])

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()                 # Encoded image bytes

    image_format = FLAGS.image_encoding.encode()   # b'jpeg' or b'png'

    width = FLAGS.image_width
    height = FLAGS.image_height

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    #  (1 per box)
    classes_text = []   # List of string class name of bounding box (1 per box)
    classes = []    # List of integer class id of bounding box (1 per box)

    for idx, bbox in example.iterrows():
        if bool(bbox.get('occluded')):
            continue
        xmins.append(float(bbox['x_min'] / width))
        xmaxs.append(float(bbox['x_max'] / width))
        ymins.append(float(bbox['y_min'] / height))
        ymaxs.append(float(bbox['y_max'] / height))
        classes_text.append(bbox['label'].encode())
        classes.append(int(label2IdMap.get(bbox['label'], -1)))

    img = cv2.imread(filename)
    boxes = np.array(list((zip(ymins, xmins, ymaxs, xmaxs))))
    scores = np.array([1 for _ in classes])
    log.debug(category_index)
    vis_util.visualize_boxes_and_labels_on_image_array(img, boxes, classes, scores, category_index, use_normalized_coordinates=True)
    cv2.imwrite(ofilename, img)

    log.debug('Saving %s', ofilename)
    log.debug('xmins %s', xmins)
    log.debug('xmaxs %s', xmaxs)
    log.debug('ymins %s' ,ymins)
    log.debug('ymaxs %s', ymaxs)
    log.debug('classes_test %s', classes_text)
    log.debug('classes %s', classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def main(_):
    train_path = FLAGS.train_path
    val_path = FLAGS.val_path
    dataset = pd.read_csv(FLAGS.input_path, sep=' ')

    if not os.path.exists(FLAGS.label_map_path):
        extract_label(dataset, FLAGS.label_map_path)

    label2IdMap = get_all_labels(FLAGS.label_map_path)

    log.debug('label2IdMap %s', label2IdMap)

    dataset.reindex(np.random.permutation(dataset.index))
    N = int(len(dataset) * 0.8)
    train = dataset[:N]
    test = dataset[N:]
    log.debug('Dataset: %s' % len(dataset))
    log.debug('TRAIN: %s' % len(train))
    log.debug('VAL: %s' % len(test))
    extract_tf_record(train, train_path, label2IdMap)
    extract_tf_record(test, val_path, label2IdMap)


if __name__ == '__main__':
    log.info('Started extracting tf_record for autti-dataset')
    tf.app.run()