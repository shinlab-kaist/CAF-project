# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:59:21 2021

@author: Minwoo Kang, The Shin Lab, KAIST

The part of frame matching and re-labeling part(cost matrix and linear assignment problem (LAP)) 
is adapted from the code in DynaMorph https://github.com/czbiohub/dynamorph/blob/master/SingleCellPatch/generate_trajectories.py
"""
#%%
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure, color
from skimage.segmentation import clear_border
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import itertools
from operator import itemgetter 
from itertools import groupby
from natsort import natsort, natsorted, index_natsorted, order_by_index, natsort_keygen
import mahotas

#'path' will be used as a value for 'image_stack' column of the dataframe, so the string should be unique
# recon_image : binary images(from semantic segmentation) for morphological features 
# inten_image : phase-contrast microscope images for Zernike momonets and Haralick featrues
sample_condition ='with231'

path = "C:/Users/user/Desktop/invitroBreast_CAF/with231/with231_43_recon_ed_12h/"
recon_image = natsort.natsorted(os.listdir(path))
path_intensity = "C:/Users/user/Desktop/invitroBreast_CAF/with231/with231_43_12h/"
inten_image = natsort.natsorted(os.listdir(path_intensity))

scale = 1.86 # um/pixel

#area_thresh indicates that cells havnig smaller area than the value will be ignored.
area_thresh = 200*scale*scale
region_props_list = []
file_names = []
box_stats_withoutsmallarea = []
box_centroids_withoutsmallarea = []
cell_label = []
zm_list = []
hf_list = []
lst_contour_bbx = []
for num, (image, image_intensity) in tqdm(enumerate(zip(recon_image, inten_image)), total=len(recon_image)):
# for num, image in tqdm(enumerate(recon_image), total=len(recon_image)):
    if (image.split(".")[1] == "tif"):
        img = cv2.imread(path + image)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img_grey = img[:,:,0]
        img_grey = cv2.resize(img_grey, dim)
        img_grey = clear_border(img_grey)
        img_intensity = cv2.imread(path_intensity + image_intensity)
        img_intensity = cv2.cvtColor(img_intensity, cv2.COLOR_BGR2GRAY)
        img_intensity = cv2.resize(img_intensity, dim)
        # img_intensity = img_intensity * scale        

        # plt.imshow(img_intensity, cmap='gray') #sanity check
        '''
        ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
        opening = clear_border(opening) #delete cells touching the border
        sure_bg = cv2.dilate(opening, kernel, iterations = 5)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        ret2, sure_fg = cv2.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)
        ret3, markers, box_stats, box_centroids = cv2.connectedComponentsWithStats(sure_fg)
        '''      
        #box stats : 5values, top left coordinates of x and y, width, height, area.
        ret3, markers, box_stats, box_centroids = cv2.connectedComponentsWithStats(img_grey)
        img_without_smallarea = np.zeros((markers.shape), np.uint8)

        for k in range(1, ret3): # 0 indicates background, so the number starts from 1. ret3: num of rows of box_stats
            (x, y, w, h, area) = box_stats[k]
            if area > area_thresh:   # cells havning area smaller than 500 will be ignored(=only cells larger than 500 pixels will be marked with value 1(white)).
                img_without_smallarea[markers == k] = 1
        
              
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(img_without_smallarea)
        img2 = color.label2rgb(labels, bg_label=0)
   
        
        props = measure.regionprops_table(labels, intensity_image=img_grey,
                                              properties=['area','extent','centroid','perimeter','minor_axis_length','major_axis_length', 'equivalent_diameter','solidity','bbox'])
        
        haralick_features_name = ['angular_second_moment', 'contrast','correlation','variance','inverse_difference_moment',
                              'sum_average','sum_variance','sum_entropy','entropy','difference_variance','difference_entropy',
                              'information_measures_of_correlation_1','information_measures_of_correlation_2']    

        contours, hierarchy = cv2.findContours(img_without_smallarea.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
        for c in contours:
            empty_plane = np.zeros((img_without_smallarea.shape[0],img_without_smallarea.shape[1]), dtype="uint8")
            cv2.drawContours(empty_plane, [c], -1, 255, -1)  
            # cell_mask와 동일한 모양이지만 cell_mask는 array이고 mask는 contour임
            (xc, yc, wc, hc) = cv2.boundingRect(c)
            contour_bbx_nparray = np.array([xc, yc, xc+wc, yc+hc]).reshape(1,4)
            
            cropped_cell_binary = empty_plane[yc:yc + hc, xc:xc + wc] #crop an image with bounding rectangle
            center, radius = cv2.minEnclosingCircle(c) 
            zernike_moments = mahotas.features.zernike_moments(cropped_cell_binary, radius, degree=9)
            # default로 cropped image의 center of mass를 씀
            # radius는 세포의 가장 작은 bounding circle의 반지름 사용
            # degree를 높이면 zernike_moments 개수도 증가
            # 0th zernike moment(A00) = 1/pi    
            # degree 9 -> 30 zernike_moments 
            # degree 26 -> 195 zernike_moments
            #degree 50 -> 676 zernike_moments
            cropped_cell_intensity = (img_intensity*img_without_smallarea)[yc:yc + hc, xc:xc + wc]
            haralick_features = np.mean(mahotas.features.haralick(cropped_cell_intensity, ignore_zeros=True),axis=0)
            # Co-occurence matrix를 만들 때 ↔, ↕, ↗, ↘ 4 방향이 가능하여  4 x 13(haralick feature)을 1 x 13으로 평균내어 축소
            # 평균을 냈다는 것은 rotation에 robust하게 만들었다는 의미
            df_zernike_moments = pd.DataFrame(zernike_moments).T
            for i in range(0, zernike_moments.shape[0]):
                df_zernike_moments = df_zernike_moments.rename(columns={i: 'zernike_moments_%s' % str(i)})

        
            df_haralick_features = pd.DataFrame(haralick_features).T
            for j in range(0, haralick_features.shape[0]):
                df_haralick_features = df_haralick_features.rename(columns={j: haralick_features_name[j]})

                
            zm_nparray = df_zernike_moments.to_numpy()
            hf_nparray = df_haralick_features.to_numpy()
            zm_list.append(zm_nparray) 
            hf_list.append(hf_nparray)
            lst_contour_bbx.append(contour_bbx_nparray)
            df_zm = pd.DataFrame.from_records(itertools.chain.from_iterable(zm_list))
            df_hf = pd.DataFrame.from_records(itertools.chain.from_iterable(hf_list))
            df_cont_bbx = pd.DataFrame.from_records(itertools.chain.from_iterable(lst_contour_bbx))
         #itertools to vstack the data
        
        
        df_props0 = pd.DataFrame(props)
        props_nparray = df_props0.to_numpy()
        region_props_list.append(props_nparray)
        df_props = pd.DataFrame.from_records(itertools.chain.from_iterable(region_props_list))
        df_props.columns = ['area','extent','y_c','x_c','perimeter','minor_axis_length','major_axis_length', 'equivalent_diameter','solidity','bbox_y1','bbox_x1','bbox_y2','bbox_x2']
                                        #bbox_x1/y1 indicate top left coordinates, other ones bottom right corrdinates
        for i in range(1, nlabel):
            (x, y, w, h, area) = stats[i]
            (x_c, y_c) = centroids[i]
            
            #Sanity check for bbox and bbox centroid
            bbox_image = cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 255)) #bbox 처리
            bbox_centroid = cv2.circle(img2, (int(x_c), int(y_c)), 3, (255,255,255), 5) #cv2.circle needs int values for center position.
            
           
            #plt.imshow(bbox_image)
            #plt.imshow(bbox_centroid)
          
            
            file_names.append(image)
            df0 = pd.DataFrame(file_names, columns = ['file_name'])
            
            df = pd.concat([df0, df_props], axis = 1)
                        
        # cv2.imwrite("C:/Users/user/Desktop/Dinesh/bbox/bbox_" + str(num) + ".png", img2)         
       
  

df = df[['file_name','area','x_c','y_c','extent','perimeter','minor_axis_length','major_axis_length', 'equivalent_diameter','solidity','bbox_x1','bbox_y1','bbox_x2','bbox_y2']]        
df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
df['circularity'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2)
# df['roundness'] = 4 * df['area'] / (np.pi * df['major_axis_length']**2) #inverse of the aspect ratio, so deleted.
df['compactness'] = df['perimeter'] ** 2 / df['area']
df_cont_bbx.columns = ['bbox_x1','bbox_y1','bbox_x2','bbox_y2'] # same column name to merge two dataframes.

df_hf = pd.DataFrame(data = df_hf.values, columns = df_haralick_features.columns)
df_zm = pd.DataFrame(data = df_zm.values, columns = df_zernike_moments.columns)
df_temp = pd.concat([df_cont_bbx, df_hf, df_zm], axis = 1)




df = pd.merge(df, df_temp)

      
# to sort by file name correctly
# df = df.reindex(df['file_name'].str.extract('(\d+)', expand=False).astype(int).sort_values().index).reset_index(drop=True)
df.sort_values(by="file_name",key=natsort_keygen(), inplace=True, ignore_index=True) #ignore_index = True로 안하면, 소팅은 되는데 index가 꼬임.


#####################################################0################################################

fname = list(df.loc[:,'file_name'].unique())
fname=natsorted(fname)

######################################################


'''새로추가된 부분
morpological featres 중복되는 row 삭제, area랑 x_c가 같은 행 찾아서 삭제
'''
df_ed_list = []

for tm in range(0, len(fname)):
   df_ed0 = df[df.loc[:,'file_name'] == fname[tm]].drop_duplicates(subset=['area','x_c'], keep='first', ignore_index=True)
   df_ed_nparray = df_ed0.to_numpy()
   df_ed_list.append(df_ed_nparray)
   df_ed = pd.DataFrame.from_records(itertools.chain.from_iterable(df_ed_list))
   df_ed.columns = df.columns
   
df=df_ed  

######################################################
######################################################
# label cells for each frame

for tmstp in range(0, len(fname)):
    lbl = len(df[df.loc[:,'file_name'] == fname[tmstp]].values)
    
    for j in range(0, lbl):
        cell_label.append(j)
        df1 = pd.DataFrame(cell_label, columns=['cell_label'])


df = pd.concat([df, df1], axis = 1)



# df.to_csv("C:/Users/user/Desktop/Dinesh/Controldata.csv", index = True)
# 
#%%
#frame matching
int_eff = 1.4
dist_cutoff = 100*scale #number is the pixel value
size_cutoff = 2

for timestp in range(0, len(fname)-1):
    
    ct00 = df[df.loc[:,'file_name'] == fname[timestp]] #to use the same indice
    ct01 = df[df.loc[:,'file_name'] == fname[timestp +1]] #to use the same indice
    idx0 = list(ct00.index)
    idx1 = list(ct01.index)
    
    ct0 = df[df.loc[:,'file_name'] == fname[timestp]].values
    ct1 = df[df.loc[:,'file_name'] == fname[timestp + 1]].values
        
    cent0 = ct0[:, 2:4]
    cent1 = ct1[:, 2:4]
    area0 = ct0[:, 1] #(number,) no column number
    area1 = ct1[:, 1]
        
    # cent_0 = np.array(cent0).reshape((-1,2))
    # cent_1 = np.array(cent1).reshape((-1,2))
    area0 = np.array(area0).reshape((-1, 1)) #(number,number) column number defined.
    area1 = np.array(area1).reshape((-1, 1))
    
    int_dist_mat = area1.reshape((1, -1)) / area0.reshape((-1, 1)) #r_ij
    int_dist_mat = int_dist_mat + 1/int_dist_mat # r_ij + (r_ij)^-1
    int_dist_mat[np.where(int_dist_mat >= (size_cutoff + 1/size_cutoff))] = 20. # maximum size difference 2 folds so 2+1/2 = 2.5. Then, put bigger values for cost matrix
    int_dist_mat = int_dist_mat ** int_eff
    int_dist_baseline = np.percentile(int_dist_mat, 10)
        
    #cost matrix shoud be a square matrix.
    cost_mat = np.ones((len(cent0)+len(cent1), len(cent0)+len(cent1))) * (dist_cutoff ** 2 * 10) * int_dist_baseline
    dist_mat = cdist(cent0, cent1) ** 2 #d_ij ^2
    dist_mat[np.where(dist_mat>=(dist_cutoff**2))] = (dist_cutoff ** 2 * 10)
    
    cost_mat[:len(cent0), :len(cent1)] = dist_mat * int_dist_mat # (r_ij + (r_ij)^-1) * d_ij ^2 . Fill linking particle index, ref) Fig.1 of Jaqaman 2008 Nat Method
      
         
    # Cost of no match placeholder
    for i in range(len(cent0)):
        cost_mat[i, i+len(cent1)] = 1.05 * (dist_cutoff ** 2) * int_dist_baseline #1.05 ->1.005
    for j in range(len(cent1)):
        cost_mat[len(cent0)+j, j] = 1.05 * (dist_cutoff ** 2) * int_dist_baseline
  
    cost_mat[len(cent0):, len(cent1):] = np.transpose(dist_mat)
    links = linear_sum_assignment(cost_mat)
    pairs = []
    costs = []
    for pair in zip(*links): # * operator to unzip the list
        if pair[0] < len(cent0) and pair[1] < len(cent1): #only take linking particle indice from the cost matrix.
            pairs.append(pair)
            costs.append(cost_mat[pair[0], pair[1]])
           
#Re-labeling of matching pairs of frame 2.    
    pairs.sort(key = lambda x: x[1])
    
    lbl0 = ct0[:, -1].tolist()  
    lbl1 = ct1[:, -1].tolist()
    
    num_cell_0 = len(ct0)
    num_cell_1 = len(ct1)

    second_frame =(df.index > max(idx0)) & (df.index <= max(idx1))
   #change labels(in second frame) which have matching pairs
    matching_idx = []
    for m in range(0, len(pairs)):
        #save matching indices into list of int.
        matching_idx.append(ct01[ct01.loc[:,'cell_label'] == ct1[(pairs[m][1]),-1]].index.astype(int)[0])
        # if the (condition 1 & condition 2) meets the criterion, change the values of df
        # condition 1 : indexing from 2nd image.
        # condition 2 : cells in 2nd frame having matching pairs.
        # Then, change the values of matching cells of 2nd frame, to labeling of 1st frame(= pairs[i][0]).
        df.loc[ct01[ct01.loc[:,'cell_label'] == ct1[(pairs[m][1]),-1]].index.astype(int),'cell_label'] = ct0[(pairs[m][0]),-1] 

        
    no_matching_idx = list(set(idx1) - set(matching_idx))
    no_matching_idx.sort()

    #The number cells without matching
    num_no_lbl_cell_1 = len(idx1) - len(pairs) 
    #New labels starting from the (maximum value of the total 'label' of dataframe +1). This results in non-consecutive numbers in cell labels
    lst_new_lbl = []
    lst_new_lbl.append(max(df.loc[:,'cell_label'])+1)
 
    
    if not bool(num_no_lbl_cell_1):# If all the cells in the second frame have matching pairs, work done.
        continue 
    else: #If some cells don't have matching pairs, create new labelings to lst_new_lbl.
       for i in range(1, num_no_lbl_cell_1): # num_no_lbl_cell_1 >= 1
                lst_new_lbl.append(lst_new_lbl[0] + i)
              
       #Impose new lables(from lst_new_label) to no_matching indices of 2nd frame.    d
    for i in range(0, len(no_matching_idx)): #if len(no_matching_idx)=0 ->  this block won't run.
        j = no_matching_idx[i]
        df.loc[j,'cell_label'] = lst_new_lbl[i]
                 
# change non-consecutive numbers in labels to make consecutive labels
all_lbl_list = list(df.loc[:,'cell_label'].unique())

group_consec_lbl = []

for k,g in groupby(enumerate(all_lbl_list), lambda x:x[0]-x[1]):
    group = (map(itemgetter(1),g))
    group = list(map(int,group))
    group_consec_lbl.append((group[0],group[-1]))

if len(group_consec_lbl) == 1: # 끊어지는 숫자가 없는 경우
    common_diff = 0
else:
    common_diff = group_consec_lbl[1][0] - (group_consec_lbl[0][1] + 1)

for nl in range(0, len(df)):
    if df.loc[nl,'cell_label'] > group_consec_lbl[0][1]:
        df.loc[nl,'cell_label'] = df.loc[nl,'cell_label'] - common_diff
        
df['image_stack'] = path
df['condition'] = sample_condition

#%%        

#creating bbox directory and images
bbox_directory = "C:/Users/user/Desktop/invitroBreast_CAF/with231/bbox_with231_43"
if not os.path.exists(bbox_directory):
    os.makedirs(bbox_directory)

# recon_image = os.listdir(path)
recon_image = natsort.natsorted(os.listdir(path))


for num, image in tqdm(enumerate(recon_image), total=len(recon_image)):
    if (image.split(".")[1] == "tif"):
        img = cv2.imread(path + image)
        img_b = img.copy()
        img_b = cv2.resize(img_b, dim)
        x_top_left = df.loc[:,'bbox_x1'][df.loc[:,'file_name']==fname[num]].tolist()
        y_top_left = df.loc[:,'bbox_y1'][df.loc[:,'file_name']==fname[num]].tolist()
        x_bottom_right = df.loc[:,'bbox_x2'][df.loc[:,'file_name']==fname[num]].tolist()
        y_bottom_right = df.loc[:,'bbox_y2'][df.loc[:,'file_name']==fname[num]].tolist()
        bbox_lbl = df.loc[:,'cell_label'][df.loc[:,'file_name']==fname[num]].tolist()
            
        for nnlbl in range(0, len(bbox_lbl)):
            x_start = x_top_left[nnlbl]
            y_start = y_top_left[nnlbl]
            x_end = x_bottom_right[nnlbl]
            y_end = y_bottom_right[nnlbl]
            bb_lbl = bbox_lbl[nnlbl]
            cv2.rectangle(img_b, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0,255,255),2)
            cv2.putText(img_b, '%d'%bb_lbl, (int(x_start), int(y_start)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (36,255,12), 2)
            
            cv2.imwrite(bbox_directory + "/bbox_with231_43"+"_" + str(num) + ".png", img_b) 

            
#%%        
#save the dataframe as csv format 

df.sort_values(by=['cell_label', 'file_name'], inplace=True, ignore_index=True)



#make 'time_point' colmun
df['time_point'] = 0
time_interval = 10 #time interval of your microscopy, [min]
time_point = []
for fn in range(0, len(fname)):

    timept = df[df.loc[:,'file_name'] == fname[fn]].index
    
    for timept_value in timept:
        df['time_point'][timept_value] = fn * time_interval

df.sort_values(by=['cell_label', 'time_point'], key=natsort_keygen(), inplace=True, ignore_index=True)


df.to_csv("C:/Users/user/Desktop/invitroBreast_CAF/with231/features_20220921_with231_43_scaled.csv", index = True)
            