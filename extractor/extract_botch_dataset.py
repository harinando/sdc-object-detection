import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from utils import get_all_labels
import yaml
import os
import numpy as np

"""
    Usage:
        cat dataset_train.zip.* > dataset_train.zip

        python extractor/extract_botch_dataset.py --input_path=data/sim_data_large_bosch_format.yaml

        # Bosch dataset
        python extractor/extract_botch_dataset.py        \
            --input_path=data/bosch_train.yaml           \
            --label_map_path=data/bosch_label_map.pbtxt  \
            --data_dir=data                              \
            --train_path=tf_records/bosch_train.record   \
            --val_path=tf_records/bosch_val.record       \
            --image_width=1280                           \
            --image_heith=720                            \
            --image_encoding='png'


        # Bosch dataset training
        nohup python /workspace/tensorflow/models/research/object_detection/train.py \
                    --logtostderr \
                    --train_dir=output/train/bosch \
                    --pipeline_config_path=config/bosch_faster_rcnn_resnet101_coco.config &

        # Tensorboard
        /usr/local/bin/tensorboard --logdir output/train/bosch --port 4567

        # Bosch evaluate
        python /workspace/tensorflow/models/research/object_detection/eval.py \
                    --logtostderr \
                    --checkpoint_dir=output/train/bosch \
                    --eval_dir=output/eval \
                    --pipeline_config_path=config/bosch_faster_rcnn_resnet101_coco.config

        # create inference graph
        python /workspace/tensorflow/models/research/object_detection/export_inference_graph.py \
                    --input_type image_tensor \
                    --pipeline_config_path config/bosch_faster_rcnn_resnet101_coco.config \
                    --trained_checkpoint_prefix output/train/bosch/model.ckpt-5517 \
                    --output_directory output/frozen-model/bosch-5517


        python /workspace/tensorflow/models/research/object_detection/export_inference_graph.py \
                    --input_type image_tensor \
                    --pipeline_config_path config/sim_faster_rcnn_resnet101_coco.config \
                    --trained_checkpoint_prefix output/train/model.ckpt-3871 \
                    --output_directory output/output_inference_graph-3871.pb

    RESOURCES:
        TF: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        BOSH: https://github.com/bosch-ros-pkg/bstld
"""

flags = tf.app.flags
flags.DEFINE_string('input_path', 'data/bosch_train.yaml', 'Path to yaml')
flags.DEFINE_string('label_map_path', 'data/label_map_path.pbtxt', 'Path to label map proto')
flags.DEFINE_string('data_dir', 'data', 'Data directory')
flags.DEFINE_string('train_path', 'tf_records/train.record', 'train tf_records path')
flags.DEFINE_string('val_path', 'tf_records/val.record', 'test tf_records path')
flags.DEFINE_integer('image_width', 800, 'Image Width')
flags.DEFINE_integer('image_height', 600, 'Image Height')
flags.DEFINE_string('image_encoding', 'jpg', 'Image encoding')
FLAGS = flags.FLAGS

def create_tf_example(example, label2IdMap):
    height = FLAGS.image_height   # Image height
    width = FLAGS.image_width     # Image width
    filename = os.path.join(FLAGS.data_dir, example.get('path', '').encode())  # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()                 # Encoded image bytes

    image_format = FLAGS.image_encoding.encode()   # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    #  (1 per box)
    classes_text = []   # List of string class name of bounding box (1 per box)
    classes = []    # List of integer class id of bounding box (1 per box)

    for bbox in example.get('boxes', []):
        xmins.append(float(bbox['x_min'] / width))
        xmaxs.append(float(bbox['x_max'] / width))
        ymins.append(float(bbox['y_min'] / height))
        ymaxs.append(float(bbox['y_max'] / height))

        classes_text.append(bbox['label'].encode())
        classes.append(int(label2IdMap.get(bbox['label'])))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
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

def extract_tf_record(examples, output, label2IdMap):
    writer = tf.python_io.TFRecordWriter(output)
    for example in examples:
        tf_example = create_tf_example(example, label2IdMap)
        writer.write(tf_example.SerializeToString())
    writer.close()

def extract_label(dataset, label_map_path):

    labels = set()
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()

    for data in dataset:
        for bbox in data.get('boxes', []):
            label = bbox.get('label', None)

            if label is not None:
                labels.add(label)

    for i, label in enumerate(labels):
        item = label_map_proto.item.add()
        item.id = i+1
        item.name = label
        item.display_name = label

    with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(str(label_map_proto))


def main(_):
    train_path = FLAGS.train_path
    val_path = FLAGS.val_path
    dataset = yaml.load(open(FLAGS.input_path, 'rb').read())

    if not os.path.exists(FLAGS.label_map_path):
        extract_label(dataset, FLAGS.label_map_path)

    label2IdMap = get_all_labels(FLAGS.label_map_path)

    np.random.shuffle(dataset)
    N = int(len(dataset) * 0.8)
    train = dataset[:N]
    test = dataset[N:]

    print('Dataset: %s' % len(dataset))
    print('TRAIN: %s' % len(train))
    print('TEST: %s' % len(test))

    extract_tf_record(train, train_path, label2IdMap)
    extract_tf_record(test, val_path, label2IdMap)

if __name__ == '__main__':
    tf.app.run()