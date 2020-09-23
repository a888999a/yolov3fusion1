#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image


return_elements = ["input/input_rgb:0","input/input_lwir:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_0910.pb"
rgb_path      = r"D:\0907_data\RGB_toZoo2.avi"
lwir_path      = r"D:\0907_data\Thermal_toZoo2.avi"
fusion_path      = r"D:\0907_data\fusion_toZoo2.avi"
# video_path      = 0
num_classes     = 8
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

output_path = './output/RGB_toZoo2.avi'


with tf.Session(graph=graph) as sess:
    writeVideo_flag = True
    if writeVideo_flag:
        rgb = cv2.VideoCapture(rgb_path)
        lwir = cv2.VideoCapture(lwir_path)
        fusion = cv2.VideoCapture(fusion_path)    

#        if not rgb.isOpened() or lwir.isOpened():
#            raise IOError("Couldn't open webcam or video")
        video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
        video_fps       = rgb.get(cv2.CAP_PROP_FPS)
        video_size      = (int(rgb.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        isOutput = True if output_path != "" else False
        if isOutput:
            #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
            list_file = open('detection.txt', 'w')
            frame_index = -1
        
    while True:
        return_value_rgb, frame_rgb = rgb.read()
        return_value_lwir, frame_lwir = lwir.read()
        return_value_fusion, frame_fusion = fusion.read()
        
        if return_value_rgb and return_value_lwir:
#            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
#            frame_lwir = cv2.cvtColor(frame_lwir, cv2.COLOR_BGR2RGB)           
            image_rgb = Image.fromarray(frame_rgb)
            image_lwir  = Image.fromarray(frame_lwir)            
        else:
            cv2.destroyAllWindows()
            out.release()
            raise ValueError("No image!")
        frame_size = frame_lwir.shape[:2]
        image_rgb,image_lwir = utils.image_preporcess(np.copy(frame_rgb),np.copy(frame_lwir), [input_size, input_size])
        image_rgb = image_rgb[np.newaxis, ...]
        image_lwir = image_lwir[np.newaxis, ...]
        
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[2], return_tensors[3], return_tensors[4]],
                    feed_dict={ return_tensors[0]: image_rgb,return_tensors[1]: image_lwir})
        pred_time = time.time()
        
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.5)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame_rgb, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        
        result = np.asarray(image)
#        info = "time: %.2f ms" %(1000*exec_time)
        info = "time:" + str(round(1000 * exec_time, 2)) + " ms, FPS: " + str(round((1000 / (1000 * exec_time)), 1))
        # cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1, color=(255, 0, 0), thickness=2)        
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if writeVideo_flag:
            # save a frame
            out.write(result)        
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break




