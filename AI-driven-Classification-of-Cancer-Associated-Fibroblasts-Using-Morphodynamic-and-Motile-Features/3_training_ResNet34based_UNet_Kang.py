# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:51:51 2021

@author: Minwoo Kang, The Shin Lab, KAIST
"""
#%%
import tensorflow as tf
from tensorflow.keras.utils import normalize #L2 norm
import numpy as np
import segmentation_models as sm
import glob
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

train_images = []
path = "C:/Users/user/Desktop/Dinesh/aug_images_patches/*.*"

for file in tqdm(sorted(glob.glob(path))):
    img = cv2.imread(file)  #Check the file has 3 channels. ResNet34 requires 3 channels for input.
    train_images.append(img)
   
train_images = np.array(train_images) 


train_masks = []
path_masks = "C:/Users/user/Desktop/Dinesh/aug_masks_patches/*.*"
       
for file in tqdm(sorted(glob.glob(path_masks))):
    mask = cv2.imread(file, 0)
    train_masks.append(mask)
    
train_masks = np.array(train_masks)

###################################################
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

X = np.float32(normalize(train_images, axis=1))
Y = np.float32(train_masks)



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
#model = sm.Unet(BACKBONE, input_shape=(None, None, 1),encoder_weights=None)
model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[sm.metrics.iou_score, 'mse'])

print(model.summary())

#%%
#############################################################################
#Sanity check, view few images
import random
import numpy as np
image_number = random.randint(0, len(x_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(x_train[image_number], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number], cmap='gray')
plt.show()
#############################################################################
#%%
#Start training

history = model.fit(x_train, 
          y_train,
          batch_size=32, 
          epochs=100,
          verbose=1,
          validation_data=(x_val, y_val))

#%%


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
IOU = history.history['iou_score']
val_IOU = history.history['val_iou_score']
epochs = range(1, len(loss) + 1)

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=300)


plt.subplot(1, 2, 1)
plt.tight_layout()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, IOU, label = "Training IOU")
plt.plot(epochs, val_IOU, label = "Validation IOU")
plt.title('Training and validation IOU score')
plt.xlabel('Epochs')
plt.ylabel('IOU score')
plt.legend()
plt.show()


y_pred=model.predict(x_val)
y_pred = np.squeeze(y_pred, axis=-1)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_val, y_pred_thresholded)
union = np.logical_or(y_val, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#%%


model.save('C:/Users/user/Desktop/Dinesh/NHDF_2021-12-07.h5')


#%%

#If you want to train your model again, you have to mount the file with compile=True

# from tensorflow import keras

#모델을 다시 트레이닝 시키려면 로드하면서 compile = True로 해야함.
#keras.models 에 정의 되어 있지 않는 loss fuction이나 metric들은 custom_objects를 이용해서 딕셔너리로 다음과 같이 정의해줘야 Unkown metric function 에러가 안남.
model = keras.models.load_model("C:/Users/user/Desktop/kangmw/practice_NHDF_2021-09-16.h5",  custom_objects={'binary_crossentropy_plus_jaccard_loss' : sm.losses.bce_jaccard_loss, 
                                                                                                      'iou_score' : sm.metrics.iou_score}, compile=True)
                                                    

#%%

#test small patches
test_img_number = random.randint(0, len(x_val)-1)
test_img = x_val[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
ground_truth=y_val[test_img_number]
prediction = model.predict(test_img_input)
prediction = prediction[0,:,:,0]

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()




#%%
# Test original full size images
from patchify import patchify, unpatchify
#To use your trained model, make sure that compile=False
#모델을 로드해서 저장된 가중치로 prediction을 할때는 compile = False로 로드하면 됨.
model = tf.keras.models.load_model("C:/Users/user/Desktop/Dinesh/NHDF_2021-12-07.h5", compile=False)

large_image = cv2.imread('C:/Users/user/Desktop/Dinesh/test_fullsize.tif', 0) #patchify가 (m, n ,p) shape을 input으로 요구해서 흑백으로 받음

patches = patchify(large_image, (256, 256), step=256)

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        #print(i,j)
        
        single_patch = patches[i,j,:,:] #(256,256)
        single_patch_norm = normalize(np.array(single_patch), axis=1) #Because we normalized the input with the function
        single_patch_input = np.stack((single_patch_norm,)*3, axis=-1) # make the shape as (256, 256, 3). Add channels
        single_patch_input = np.expand_dims(single_patch_input, 0) #(1,256,256,3) Add numbers of pictures arg. 
        #single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
       
        #Predict and threshold for values above 0.5 probability
        #어차피 한장이고, 흑백인데 shape 맞춰주누라 (1,256,256,3) 된거라서 픽셀값만 비교.
        single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) 
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
#plt.imshow(reconstructed_image, cmap='gray')
#plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

#plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

#final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
#plt.imshow(final_prediction)

original_mask = np.float32(cv2.imread("C:/Users/user/Desktop/Dinesh/mask_fullsize.tif"))

plt.figure(figsize=(16, 8), dpi=300)
plt.subplot(231)
plt.title('Original Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(232)
plt.title('Original mask')
plt.imshow(original_mask, cmap='gray')
plt.subplot(233)
plt.title('Prediction')
plt.imshow(reconstructed_image, cmap='gray')
plt.show()
#############################################################################

