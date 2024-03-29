# -*- coding: utf-8 -*-
"""
Created on Fri May 28 18:36:06 2021

@author: Minwoo Kang, KAIST
"""

from skimage import io
import random
import os
import albumentations as A

images_to_generate = 50000


images_path="C:/Users/user/Desktop/Dinesh/train_patches/" #path to original images
masks_path = "C:/Users/user/Desktop/Dinesh/masks_patches/"
img_augmented_path="C:/Users/user/Desktop/Dinesh/aug_images_patches/" # path to store aumented images
msk_augmented_path="C:/Users/user/Desktop/Dinesh/aug_masks_patches/" # path to store aumented images
images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))


#interpolation = cv2.INTER_LINEAR As default
#border_mode =  cv2.BORDER_REFLECT_101  As default
aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.8),
    
    A.RandomBrightnessContrast(p=0.8)
    ])

#random.seed(42) #살리면 아래 난수인 number가 고정됨.

i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    number = random.randint(0, len(images)-1)  #Pick a number to select an image & mask
    image = images[number]
    mask = masks[number]
    print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1
