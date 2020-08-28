import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession, Session
import time

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './yolov4/checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


class Detector:
    def __init__(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = Session(config=config)
        self.STRIDES, self.ANCHORS, self.NUM_CLASS, self.XYSCALE = utils.load_config(FLAGS)
        self.input_size = FLAGS.size
        self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']
        self.names = [s.strip() for s in open("./data/classes/coco.names").readlines()]

    def __del__(self):
        self.session.close()

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        res = pred_bbox
        rows, cols, d = frame.shape
        objects = [ {'x': int(z[0][0]*cols),
                     'y': int(z[0][1]*rows),
                     'w': int((z[0][2] - z[0][0])*cols),
                     'h': int((z[0][3] - z[0][1])*rows),
                     'type': self.names[int(z[2])],
                     'val': z[1]}
                    for z in (zip(res[0][0],res[1][0], res[2].astype(int)[0])) if z[0][0] > 0.00001]
        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)
        return objects

FLAGS(['foobar'])
