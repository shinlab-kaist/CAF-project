# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:01:40 2021

@author: Minwoo Kang, Thi Shin Lab, KAIST
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from tqdm import tqdm

path = "images/NHDF_test/images/"

train_images = []
images = os.listdir(path)
for i, image_name in tqdm(enumerate(images), total=len(images)):
    if (image_name.split(".")[1] =="tif"):
        img = cv2.imread(path+image_name)
        h_zeros = cv2.cvtColor(np.expand_dims(np.zeros((1216,128), np.uint8), 2), cv2.COLOR_GRAY2BGR)
        img = np.hstack((img, h_zeros))
        v_zeros = cv2.cvtColor(np.expand_dims(np.zeros((64,2048), np.uint8), 2), cv2.COLOR_GRAY2BGR)
        img = np.vstack((img, v_zeros))
        cv2.imwrite("images/NHDF_test/images_pad/" + "train_pad"+str(i) + ".tif", img)
        train_images.append(img)
   

path_mask = "images/NHDF_test/masks/"
   
train_masks = []
images = os.listdir(path_mask)
for i, image_name in tqdm(enumerate(images), total=len(images)):
    if (image_name.split(".")[1] =="tif"):
        img = cv2.imread(path_mask+image_name)
        h_zeros = cv2.cvtColor(np.expand_dims(np.zeros((1216,128), np.uint8), 2), cv2.COLOR_GRAY2BGR)
        img = np.hstack((img, h_zeros))
        v_zeros = cv2.cvtColor(np.expand_dims(np.zeros((64,2048), np.uint8), 2), cv2.COLOR_GRAY2BGR)
        img = np.vstack((img, v_zeros))
        cv2.imwrite("images/NHDF_test/masks_pad/" + "mask_pad"+str(i) + ".tif", img)
        train_masks.append(img)    
   
