#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from core.yolov3 import YOLOV3

pb_file = "./yolov3_0910.pb"
ckpt_file = "./checkpoint/yolov3_RGB_0910_loss=11.9811.ckpt-122"
output_node_names = ["input/input_rgb","input/input_lwir", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_rgb = tf.placeholder(dtype=tf.float32, name='input_rgb')
    input_lwir = tf.placeholder(dtype=tf.float32, name='input_lwir')
model = YOLOV3(input_rgb,input_lwir, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())



