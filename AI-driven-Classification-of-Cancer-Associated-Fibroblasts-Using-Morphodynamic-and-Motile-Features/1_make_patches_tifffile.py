# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:38:28 2021

@author: Minwoo Kang, The Shin Lab, KAIST
"""
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify

import tifffile as tiff

image_stack = tiff.imread("C:/Users/user/Desktop/Dinesh/trainig_set.tif")
mask_stack = tiff.imread("C:/Users/user/Desktop/Dinesh/trainig_anno.tiff")


for img in range(image_stack.shape[0]):

    image = image_stack[img]
    
    patches_img = patchify(image, (256, 256), step=256)  #Step=256 for 256 patches means no overlap 
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('C:/Users/user/Desktop/Dinesh/train_patches/' + 'image_' + str(img) + '_' + str(i)+str(j)+ ".tif", single_patch_img)



for msk in range(mask_stack.shape[0]):
    mask = mask_stack[msk]
    patches_msk = patchify(mask, (256,256), step=256)
    
    for i in range(patches_msk.shape[0]):
        for j in range(patches_msk.shape[1]):
            
            single_patch_mask = patches_msk[i,j,:,:]
            tiff.imwrite("C:/Users/user/Desktop/Dinesh/masks_patches/" + 'mask_' +str(msk) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            # single_patch_mask = single_patch_mask / 255.
    