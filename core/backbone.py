#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf


def darknet53(input_rgb,input_lwir, trainable):

    with tf.variable_scope('darknet'):
##############################RGB###############################################
        input_rgb = common.convolutional(input_rgb, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_rgb = common.convolutional(input_rgb, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_rgb = common.residual_block(input_rgb,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_rgb = common.convolutional(input_rgb, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_rgb = common.residual_block(input_rgb, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_rgb = common.convolutional(input_rgb, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)
##############################RGB###############################################
        
##############################LWIR###############################################
        input_lwir = common.convolutional(input_lwir, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0_lwir')
        input_lwir = common.convolutional(input_lwir, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1_lwir', downsample=True)

        for i in range(1):
            input_lwir = common.residual_block(input_lwir,  64,  32, 64, trainable=trainable, name='residual%d_lwir' %(i+0))

        input_lwir = common.convolutional(input_lwir, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4_lwir', downsample=True)

        for i in range(2):
            input_lwir = common.residual_block(input_lwir, 128,  64, 128, trainable=trainable, name='residual%d_lwir' %(i+1))

        input_lwir = common.convolutional(input_lwir, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9_lwir', downsample=True)
##############################LWIR###############################################


##############################cont###############################################

        input_data = tf.concat(axis=3, values=[input_rgb, input_lwir])
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 256),trainable=trainable, name='conv_b09')        
        
##############################cont###############################################        

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data




