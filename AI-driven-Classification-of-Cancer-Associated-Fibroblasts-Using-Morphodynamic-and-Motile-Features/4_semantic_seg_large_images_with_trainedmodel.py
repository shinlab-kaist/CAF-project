# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:32:48 2021

@author: Minwoo Kang, The Shin Lab, KAIST
"""



import numpy as np
from patchify import patchify, unpatchify
import os
import cv2
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.utils import normalize
import natsort

model = keras.models.load_model("C:/Users/user/Desktop/invitroBreast_CAF/NHDF_5x_2021-12-07.h5", compile=False)


#creating recon image directory
recon_image_directory = "C:/Users/user/Desktop/20220929_invitroBreastCAF/with7_30_recon"
if not os.path.exists(recon_image_directory):
    os.makedirs(recon_image_directory)


large_image_path = "C:/Users/user/Desktop/20220929_invitroBreastCAF/with7_30/" 
check_images = natsort.natsorted(os.listdir(large_image_path)) #natsort module for sorting complex microscopy image sequence names.

for num, large_image_name in tqdm(enumerate(check_images), total=len(check_images)):
    if (large_image_name.split('.')[1] == "tif"):
        img = cv2.imread(large_image_path + large_image_name, 0)
        patches = patchify(img, (256, 256), step=256)
        
        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                
                single_patch = patches[i,j,:,:] #(256, 256)
                single_patch_norm = normalize(np.array(single_patch), axis=1) 
                single_patch_input = np.stack((single_patch_norm,)*3, axis=-1) # (256, 256, 3) 
                single_patch_input = np.expand_dims(single_patch_input, 0) #(1,256,256,3)
                
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0]>0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction)
                
        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
        reconstructed_image = unpatchify(predicted_patches_reshaped, img.shape)
        
        cv2.imwrite(recon_image_directory + "/with7_30_recon"+'_' + str(num) +  ".tif", reconstructed_image)

