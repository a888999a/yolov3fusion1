# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 02:43:34 2020

@author: gary
"""


import cv2
import glob
 
fps = 5   
 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('rgb.avi',fourcc,fps,(640,512))#最后一个是保存图片的尺寸
imgs=glob.glob(r"C:\Users\gary\Desktop\b09\test\JPEGImages\rgb\*.jpg")
for imgname in imgs:
    frame = cv2.imread(imgname)
    videoWriter.write(frame)
    print(imgname)

