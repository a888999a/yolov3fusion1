#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_rgb:0","input/input_lwir:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path_rgb  = r"C:\Users\gary\Desktop\b09\test\JPEGImages\rgb\set06_V000_I00019.jpg"
image_path_lwir = r"C:\Users\gary\Desktop\b09\test\JPEGImages\lwir\set06_V000_I00019.jpg"
num_classes     = 1
input_size      = 416
graph           = tf.Graph()

original_rgb = cv2.imread(image_path_rgb)
original_lwir = cv2.imread(image_path_lwir)

original_image_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
original_image_lwir = cv2.cvtColor(original_lwir, cv2.COLOR_BGR2RGB)

original_image_size = original_image_rgb.shape[:2]

image_rgb,image_lwir = utils.image_preporcess(np.copy(original_image_rgb),np.copy(original_image_lwir), [input_size, input_size])

image_rgb = image_rgb[np.newaxis, ...]
image_lwir = image_lwir[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[2], return_tensors[3], return_tensors[4]],
                feed_dict={ return_tensors[0]: image_rgb,return_tensors[1]: image_lwir})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image_rgb, bboxes)
image = Image.fromarray(image)
image.show()




