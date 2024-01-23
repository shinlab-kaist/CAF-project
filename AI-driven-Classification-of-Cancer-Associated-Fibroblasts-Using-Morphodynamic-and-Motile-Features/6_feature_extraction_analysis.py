# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:15:19 2021

@author: Chan Hong Min & Minwoo Kang, The Shin Lab, KAIST

motility feature extraction part is adpated from https://github.com/cellgeometry/heteromotility
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
pio.renderers.default = "browser" #to open interactive plotly plot on web browswer.
# pip install xarray --upgrade #if Attribute error about pandas Panel shows up, hit the line.
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
from tqdm import tqdm
# from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from natsort import natsort, natsorted, index_natsorted, order_by_index, natsort_keygen
import itertools
from statistics import mean

directory = "C:/Users/user/Desktop/"


df = pd.read_csv('C:/Users/makav/Desktop/features_total_20221022_invitroBreastCAF.csv')

df_raw = df.drop(['Unnamed: 0', 'file_name', 'cell_label', 'x_c', 'y_c', 
                  'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2','zernike_moments_0', 'time_point', 'condition', 'image_stack'], axis = 1)

df_params = pd.DataFrame(df_raw.T.index)
df_params.columns = ['parameter']



#area has larger value than other params, we need to normalize it before performing PCA.
scaler = StandardScaler()
X=scaler.fit_transform(df_raw)
pca = PCA(n_components = df_raw.shape[1], svd_solver= 'full')
Y = pca.fit_transform(X)
df_raw_pca = pd.DataFrame(Y)

col_pc = []
for i in range(1, df_raw.shape[1]+1):
    col_pc.append("PC%d"%i)
    
df_raw_pca.columns = col_pc
df_pca = pd.concat([df['image_stack'],df['time_point'], df['cell_label'],df['condition'], df_raw_pca], axis = 1)



variance_ratio = pca.explained_variance_ratio_ #Percentage of variance explained by each of the selected components.
variance = pca.explained_variance_
cum_sum_variance_ratio = np.cumsum(variance_ratio)

plt.bar(range(0,len(variance_ratio)), variance_ratio, alpha=0.5, align='center', label='Individual explained variance', color='black')
plt.step(range(0,len(cum_sum_variance_ratio)), cum_sum_variance_ratio, where='mid',label='Cumulative explained variance',color='black')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Explained variance ratio', fontsize=15)
plt.xlabel('Principal component index',fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#95%설명하는 PC번호 찾기
for i in range(1, len(cum_sum_variance_ratio)+1):
    if (cum_sum_variance_ratio[i-1] >= 0.95).min():
        print("PC%d"%i)
        PC_95percent = i-1
        break
    
        



df_loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
df_loadings.columns = col_pc
df_loadings = pd.concat([df_params, df_loadings], axis =1)


# fig, ax = plt.subplots(figsize=(100,100)) 
# df_loadings_heatmap = df_loadings.set_index('parameter')
# ax = sns.heatmap(df_loadings_heatmap, annot=True, cmap='coolwarm')
# plt.show()






#Top PC 에서 가장 연관성이 높은 morhplogical features 추출.
top_pc = 5
top_parameter = []
for i in range(1,top_pc+1):
    top_parameter.append(df_loadings['PC%d'%i].abs().idxmax()) #절대값 취해서 음의 상관관계도 나오게
    # top_parameter.append(df_loadings['PC%d'%i].idxmax()) #양의 상관관계만

df_loadings_top_pc=pd.DataFrame()
for i in top_parameter:
    loadings_top_pc = df_loadings.iloc[i,0:top_pc+1].to_frame()
    df_loadings_top_pc=pd.concat([df_loadings_top_pc, loadings_top_pc], axis=1)

df_loadings_top_pc = df_loadings_top_pc.T
df_loadings_top_pc = df_loadings_top_pc.set_index('parameter')
df_loadings_top_pc.index.name = None
#folling lines to ensure the sns.heatmap works   
df_loadings_top_pc.astype(float) 
df_loadings_top_pc.fillna(value=np.nan, inplace=True)

# corr_df_loadings = df_loadings.corr(method="spearman")

plt.figure(figsize=(15,12))
heatmap = sns.heatmap(df_loadings_top_pc, annot = True, annot_kws={"size": 25}, cmap = 'coolwarm')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=30, rotation =0)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=30)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)
# plt.xlabel('Morphology cluster', fontsize = 20, labelpad=10)
# plt.ylabel('Condition', fontsize = 20, labelpad=10)
# plt.title('Distribtion of morphology custer', fontsize = 20)
plt.show()


top_pc = 5
top_parameter = []
for i in range(1,top_pc+1):
    # top_parameter.append(df_loadings['PC%d'%i].abs().idxmax()) #절대값 취해서 음의 상관관계도 나오게
    top_parameter.append(df_loadings['PC%d'%i].idxmax()) #양의 상관관계만

df_loadings_top_pc=pd.DataFrame()
for i in top_parameter:
    loadings_top_pc = df_loadings.iloc[i,0:top_pc+1].to_frame()
    df_loadings_top_pc=pd.concat([df_loadings_top_pc, loadings_top_pc], axis=1)

df_loadings_top_pc = df_loadings_top_pc.T
df_loadings_top_pc = df_loadings_top_pc.set_index('parameter')
df_loadings_top_pc.index.name = None
#folling lines to ensure the sns.heatmap works   
df_loadings_top_pc.astype(float) 
df_loadings_top_pc.fillna(value=np.nan, inplace=True)

# corr_df_loadings = df_loadings.corr(method="spearman")

plt.figure(figsize=(15,12))
heatmap = sns.heatmap(df_loadings_top_pc, annot = True, annot_kws={"size": 25}, cmap = 'coolwarm')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=30, rotation =0)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=30)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)
# plt.xlabel('Morphology cluster', fontsize = 20, labelpad=10)
# plt.ylabel('Condition', fontsize = 20, labelpad=10)
# plt.title('Distribtion of morphology custer', fontsize = 20)
plt.show()





#PC1,PC2 축으로한 scatter plot
# plt.scatter(Y[:,0],Y[:,1],color='blue',edgecolor='k')
# plt.xlabel('PC1',fontsize=16)
# plt.ylabel('PC2',fontsize=16)

##################################################################
fig = px.scatter(data_frame = df_pca, x = 'PC1', y = 'PC2', color = 'condition',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.7,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Morphology Space',
labels = {'PC1':'PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)',
          },
hover_data = {'time_point': True, 'cell_label':True},
hover_name = df_pca.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

# for index, value in df_no_rotation_loadings_final.iterrows():
#     fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = 3*value['PC1'], y1 = 3*value['PC2'], opacity = 0.7,
#                   line = dict(color='black', width = 1, dash = 'dot'))
#     fig.add_annotation(x = 3*value['PC1'], y = 3*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
#                        text = value['parameter'], font = dict(size = 8, color = 'black'), opacity = 0.7)

pio.show(fig)


#%%
fig = px.scatter(
data_frame = df_pca,
x = 'PC1',
y = 'PC2',
color = 'time_point',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Morphology Space',
labels = {'PC1':'PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)',
          },
hover_data = {'image_stack': True, 'cell_label':True},
hover_name = df_pca.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 4),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

for index, value in df_loadings.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = 10*value['PC1'], y1 = 10*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = 3*value['PC1'], y = 3*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                       text = value['parameter'], font = dict(size = 8, color = 'black'), opacity = 0.7)

pio.show(fig)




#%%
'''time distribution of all data points'''

plt.figure(figsize=(25,15))
plt.scatter(df_pca['PC1'],df_pca['PC2'],marker='o',s=100, 
                c=df_pca['time_point'],cmap = plt.cm.get_cmap('viridis'))
plt.xlim(-13, 16)
plt.ylim(-12, 18)
clb = plt.colorbar()
clb.ax.set_title('Time[min]', fontsize=25, pad=20)
clb.ax.tick_params(labelsize=25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)
plt.savefig('time_dist_all_cells.png', dpi=600)


##############################################
df_pca_control = df_pca[(df_pca.loc[:,'condition'] == 'control')]
df_pca_with7 = df_pca[(df_pca.loc[:,'condition'] == 'with7')]
df_pca_with231 = df_pca[(df_pca.loc[:,'condition'] == 'with231')]

plt.figure(figsize=(20,15))
# plt.scatter(df_pca_control['PC1'],df_pca_control['PC2'],marker=',',s=10, 
#                 c=df_pca_control['time_point'],cmap = plt.cm.get_cmap('jet'))
# plt.scatter(df_pca_with7['PC1'],df_pca_with7['PC2'],marker=',',s=10, 
#                 c=df_pca_with7['time_point'],cmap = plt.cm.get_cmap('jet'))
plt.scatter(df_pca_with231['PC1'],df_pca_with231['PC2'],marker=',',s=10, 
                c=df_pca_with231['time_point'],cmap = plt.cm.get_cmap('jet'))

plt.xlim(-13, 16)
plt.ylim(-12, 18)
clb = plt.colorbar()
clb.ax.set_title('Time[min]', fontsize=25, pad=20)
clb.ax.tick_params(labelsize=25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)



#%%
'''by sample type'''
plt.figure(figsize=(20,15))
colors = itertools.cycle(["green", "royalblue", "indianred","c","m"])
groups = df_pca.groupby('condition')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o',alpha=.4, linestyle='', markersize=10, label=name, color=next(colors))
# plt.legend(loc='best', fontsize=20)
plt.xlim(-13, 16)
plt.ylim(-12, 18)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)
plt.grid(False)
plt.axis('off')
plt.savefig('morphology_space_withoutframe.png', dpi=600)


#%%
'''K-means clustering'''

'''elbow method'''
k_range = range(2,20)
sum_squared_error = [] # sum_squared_error object를 list로 정의
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df_pca[['PC1','PC2']])
    sum_squared_error.append(km.inertia_) # inertia_ 자체가 sum of squared error 계산식을 포함
sum_squared_error


plt.xlabel('K')
plt.xticks(np.arange(1,20),fontsize=7)
plt.ylabel('Sum of squared error')
plt.plot(k_range,sum_squared_error)
from kneed import KneeLocator # conda install -c conda-forge kneed
kl = KneeLocator(range(2,20),sum_squared_error,curve='convex',direction='decreasing')
kl.elbow # find point of maximum curvature

# elbow method에 의해 k=4임을 알 수 있음


'''silhouette score'''
from sklearn.metrics import silhouette_score
silhouette_coefficeints =[]
for k in range(2,20):
    km = KMeans(n_clusters=k)
    km.fit(df_pca[['PC1','PC2']])
    score = silhouette_score(df_pca[['PC1','PC2']],km.labels_)
    silhouette_coefficeints.append(score)
silhouette_coefficeints



plt.xlabel('K')
plt.xticks(np.arange(1,20),fontsize=7)
plt.ylabel('Sillhouette Coefficient')
plt.plot(range(2,20),silhouette_coefficeints)
# choose the maximum value
#%%

number_of_clusters = 7
km = KMeans(n_clusters = number_of_clusters,random_state=0)
km

kmeans_predicted = km.fit_predict(df_pca[['PC1','PC2']])# fit하고 동시에 predict하는것
kmeans_predicted



df_pca['kmeans_cluster'] = kmeans_predicted

# SettingWithCopyWarning은 df_no_rotation_scores라는 원본에서 나온 df_no_rotation_control 복사본에 새로운 열을 추가할때 발생하는 에러
# 복사된 dataframe에는 메모리가 할당되지 않기때문에 이 복사된 df에도 메모리 공간 할당을 해줘야됨



new_df = df_pca['kmeans_cluster'].replace({0:'Group 0', 1:'Group 1', 2:'Group 2',
                                                       3:'Group 3', 4:'Group 4', 5:'Group 5',
                                                       6:'Group 6', 7:'Group 7', 8:'Group 8',
                                                       9:'Group 9', 10:'Group 10', 11:'Group 11',
                                                       12:'Group 12', 13:'Group 13', 14:'Group 14',
                                                       15:'Group 15', 16:'Group 16', 17:'Group 17',
                                                         })
new_df = pd.concat([df_pca.drop(['kmeans_cluster'],axis=1),new_df],axis=1) 
new_df



fig = px.scatter(
data_frame = new_df,
x = 'PC1',
y = 'PC2',
color = 'kmeans_cluster',
color_discrete_sequence = px.colors.qualitative.Set1, # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
#color_discrete_sequence = px.colors.qualitative.Bold,   
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, _ne
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Morphology Space',
labels = {'PC1':'PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)',
          },
hover_data = {'cell_label': True,'condition':True,'time_point':True,'image_stack':True},
hover_name = df_pca.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )
constant = 3
#for index, value in df_no_rotation_loadings_final.iterrows():
#    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = constant*value['PC1'], y1 = constant*value['PC2'], opacity = 0.7,
#                  line = dict(color='black', width = 1, dash = 'dot'))
#    fig.add_annotation(x = constant*value['PC1'], y = constant*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
#                      text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

fig.add_scatter(x = km.cluster_centers_[:,0], y = km.cluster_centers_[:,1], 
                mode = 'markers', marker= dict(color = 'purple', size = 10))


pio.show(fig)

#%%

plt.figure(figsize=(20,15))
colors = itertools.cycle(["dodgerblue", "navy","steelblue","mediumblue","blue","royalblue","cornflowerblue"])

# colors = itertools.cycle(["midnightblue", "navy","darkbluev","mediumblue","blue","royalblue","cornflowerblue"])

groups = df_pca.groupby('kmeans_cluster')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o', linestyle='', alpha=.4, markersize=10, label=name, color=next(colors))
    # plt.plot(group.PC1, group.PC2, marker='o', linestyle='', markersize=7, label=name)
# plt.legend(loc='best',prop={'size': 20})
plt.xlim(-13, 16)
plt.ylim(-12, 18)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)
plt.grid(False)
plt.axis('off')
plt.savefig('morphology_space_kmeans_withoutframe.png', dpi=600)


#%%

cell_data_list = []
label_data = pd.DataFrame(df_pca.groupby(['image_stack', 'cell_label']).apply(lambda x : x.name)).reset_index()
label_data.sort_values(by=["image_stack", 'cell_label'],key=natsort_keygen()).reset_index(drop=True)

# label_data

for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_pca.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1]))
    # cell_data = df_pca[(df_pca['file_name'] == label_data[0][i][0]) | (df_pca['file_name'] == label_data[0][i][1])]
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보 
    cell_data.reset_index(inplace=True)
    cell_data = cell_data.drop('index', axis = 1)
    cell_data['number_of_appearance'] = cell_data.shape[0]
    cell_data_list.append(cell_data)


df_cell_data = pd.DataFrame()


for j in range(0, len(cell_data_list)):
    df_cell_data = pd.concat([df_cell_data, cell_data_list[j]], axis=0)
df_cell_data.reset_index(inplace=True)
df_cell_data = df_cell_data.drop('index', axis = 1)


df_fulltime_cell = df_cell_data[df_cell_data['number_of_appearance']==df_cell_data['number_of_appearance'].max()] 
df_fulltime_cell.reset_index(inplace=True)
df_fulltime_cell = df_fulltime_cell.drop('index', axis = 1)
df_fulltime_cell


"""
# Distribution 
"""
#group_condition has multiindex
group_condition = pd.DataFrame(df_fulltime_cell.groupby(['condition','kmeans_cluster']).size())
group_condition['distribution(%)'] = group_condition.groupby(level=0).apply(lambda x:  100*x / x.sum())
group_condition = group_condition.drop([0],axis=1)
group_condition = group_condition.unstack(level=0)
group_condition
#group_condition = group_condition[[('distribution(%)','Lateral(B1)'), ('distribution(%)','Medial(B1)'), ('distribution(%)','Fornix(B1)'), ('distribution(%)','Inferior(B1)'), ('distribution(%)','Superior(B1)')]]
group_condition = group_condition[[('distribution(%)','control'), ('distribution(%)','with7'), ('distribution(%)','with231')]]
#group_condition = group_condition[[('distribution(%)','Inferior(B3)'), ('distribution(%)','Superior(B1)'), ('distribution(%)','Fornix(B3)'), ('distribution(%)','Lateral(B3)'), ('distribution(%)','Medial(B3)'), ('distribution(%)','Superior(B3)'), ('distribution(%)','Lateral(B1)'), ('distribution(%)','Inferior(B1)'), ('distribution(%)','Medial(B1)'), ('distribution(%)','Fornix(B1)')]]

'''
#to change the column order(put control first)
cols = group_condition.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols
group_condition = group_condition[cols]
group_condition
'''

plt.figure(figsize=(15,10))
heatmap = sns.heatmap(group_condition.T, annot = True, annot_kws={"size": 20}, cmap = 'Blues')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=15)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=25)
plt.xlabel('Morphology cluster', fontsize = 30, labelpad=10)
plt.ylabel('Condition', fontsize = 30, labelpad=10)
plt.title('Distribtion of morphology cluster', fontsize = 20)
plt.savefig('heatmap_morphology.eps', format='eps', dpi=600)
plt.show()

'''# Shannon entropy'''
shannon_entropy_list = []
# group_condition.columns -> 2중 index
for condition_name in list(group_condition['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, group_condition['distribution(%)'][condition_name].shape[0]):
        shannon_entropy = shannon_entropy + -group_condition['distribution(%)'][condition_name][i]/100*np.log2(group_condition['distribution(%)'][condition_name][i]/100)
    shannon_entropy_list.append(shannon_entropy)
    print(condition_name)
shannon_entropy_list

plt.figure(figsize=(8,8))
colors = ['green', 'royalblue', 'indianred']
# plt.title('Shannon entropy', fontsize = 30)
plt.bar(np.arange(len(shannon_entropy_list)), shannon_entropy_list, color=colors, 
        width = 0.4, edgecolor = "black", linewidth=3)
# plt.xticks(np.arange(len(shannon_entropy_list)), group_condition['distribution(%)'].columns.tolist(), fontsize=25)
plt.yticks(fontsize=20)
plt.xticks([], [])
plt.ylim(2.5, 2.8)
plt.savefig('Shannon_entropy_morphology.eps', format='eps', dpi=600)
plt.show()


'''# Hierarchical clustering'''
transposed = group_condition.T

hl = hierarchy.linkage(transposed,method='average',metric='euclidean')
hl
# Centroid : 두 군집의 중심점(centroid)를 정의한 다음 두 중심점의 거리를 군집간의 거리로 측정
# Single : 최단 연결법, 두 군집에 있는 모든 데이터 조합에서 데이터 사이 거리를 측정해서 가장 최소 거리(작은 값)를 기준으로 군집 거리를 측정
# Complete : 최장 연결법으로 두 클러스터상에서 가장 먼 거리를 이용해서 측정하는 방식
# Average : 평균 연결법, 두 군집의 데이터들 간 모든 거리들의 평균을 군집간 거리로 정의
# Ward : 연결될 수 있는 군집 조합을 만들고, 군집 내 편차들의 Sum of squared error을 기준으로 최소 제곱합을 가지게 되는 군집끼리 연결

plt.figure(figsize=(2,10))

dendrogram = hierarchy.dendrogram(hl, orientation = 'left',labels = transposed.index)


#%%

'''# Phenotype - time distribution'''
#phenotype_time has multiindex
phenotype_time = pd.DataFrame(df_fulltime_cell.groupby(['time_point','kmeans_cluster','condition']).size())
phenotype_time['distribution(%)'] = phenotype_time.groupby(level=0).apply(lambda x:  100*x / x.sum())
phenotype_time = phenotype_time.drop([0],axis=1)
phenotype_time = phenotype_time.unstack(level=0)
phenotype_time = phenotype_time.fillna(0) # NaN -> 0으로 바꿈
phenotype_time.reindex(index=[(0, 'control'),
            (0, 'with7'),
            (0,   'with231'),
            (1, 'control'),
            (1, 'with7'),
            (1,   'with231'),
            (2, 'control'),
            (2, 'with7'),
            (2,   'with231'),
            (3, 'control'),
            (3, 'with7'),
            (3,   'with231'),
            (4, 'control'),
            (4, 'with7'),
            (4,   'with231'),
            (5, 'control'),
            (5, 'with7'),
            (5,   'with231'),
            (6, 'control'),
            (6, 'with7'),
            (6,   'with231')])

####
shannon_entropy_list_control = []
for time_point in list(phenotype_time['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, phenotype_time['distribution(%)'].query("condition == 'control'")[time_point].shape[0]):
        if phenotype_time['distribution(%)'].query("condition == 'control'").reset_index(drop=True)[time_point][i] == 0:
            continue
        else:
            shannon_entropy = shannon_entropy + -phenotype_time['distribution(%)'].query("condition == 'control'")[time_point][i]/100*np.log2(phenotype_time['distribution(%)'].query("condition == 'control'")[time_point][i]/100)
    shannon_entropy_list_control.append(shannon_entropy)
shannon_entropy_list_control

shannon_entropy_list_with7 = []
for time_point in list(phenotype_time['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, phenotype_time['distribution(%)'].query("condition == 'with7'")[time_point].shape[0]):
        if phenotype_time['distribution(%)'].query("condition == 'with7'").reset_index(drop=True)[time_point][i] == 0:
            continue
        else:
            shannon_entropy = shannon_entropy + -phenotype_time['distribution(%)'].query("condition == 'with7'")[time_point][i]/100*np.log2(phenotype_time['distribution(%)'].query("condition == 'with7'")[time_point][i]/100)
    shannon_entropy_list_with7.append(shannon_entropy)
shannon_entropy_list_with7

shannon_entropy_list_with231 = []
for time_point in list(phenotype_time['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, phenotype_time['distribution(%)'].query("condition == 'with231'")[time_point].shape[0]):
        if phenotype_time['distribution(%)'].query("condition == 'with231'").reset_index(drop=True)[time_point][i] == 0:
            continue
        else:
            shannon_entropy = shannon_entropy + -phenotype_time['distribution(%)'].query("condition == 'with231'")[time_point][i]/100*np.log2(phenotype_time['distribution(%)'].query("condition == 'with231'")[time_point][i]/100)
    shannon_entropy_list_with231.append(shannon_entropy)
shannon_entropy_list_with231


# plt.figure(figsize=(18,9))
plt.plot(shannon_entropy_list_control)
plt.plot(shannon_entropy_list_with7)
plt.plot(shannon_entropy_list_with231)
#plt.xticks(ticks = range(0,49,6), labels = list(np.arange(0, 28, 3)))
plt.xlabel("time_point")
plt.ylabel("Shannon's entropy")
plt.legend()

################################
#y-axis 스케일이 달라서 따로따로 그림.
x = np.linspace(0, phenotype_time['distribution(%)'].columns.max(),73)
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].plot(x, shannon_entropy_list_control)
axs[0].set_ylim([1.34, 1.45])
axs[0].set_title('Control', fontsize=20)
axs[0].set_xlabel('Time[min]')
axs[1].plot(x, shannon_entropy_list_with7, 'tab:orange')
axs[1].set_ylim([1.44, 1.58])
axs[1].set_title('with7', fontsize=20)
axs[1].set_xlabel('Time[min]')
axs[2].plot(x, shannon_entropy_list_with231, 'tab:green')
axs[2].set_ylim([1.24, 1.34])
axs[2].set_title('with231', fontsize=20)
axs[2].set_xlabel('Time[min]')
fig.text(0.08, 0.5, 'Shannon entropy', ha='center', va='center', rotation='vertical', fontsize=15)







#condition 및 cluster 별 시간에 따른 분포.

per_max = 10
plt.figure(figsize=(18,9))
heatmap1 = sns.heatmap(phenotype_time.query("condition == 'control'"),
                       annot = False, cmap = 'rocket_r', cbar_kws={'format': '%.0f%%'}, vmax = per_max)
cbar = heatmap1.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)
plt.xlabel('Time[min]', fontsize=15)
plt.yticks(va='center', fontsize=12)

plt.figure(figsize=(18,9))
heatmap2 = sns.heatmap(phenotype_time.query("condition == 'with7'"),
                       annot = False, cmap = 'rocket_r', cbar_kws={'format': '%.0f%%'}, vmax = per_max)
plt.xlabel('Time[min]', fontsize=15)
plt.yticks(va='center', fontsize=12)


plt.figure(figsize=(18,9))
heatmap3 = sns.heatmap(phenotype_time.query("condition == 'with231'"),
                       annot = False, cmap = 'rocket_r', cbar_kws={'format': '%.0f%%'}, vmax=per_max)
plt.xlabel('Time[min]', fontsize=15)
plt.yticks(va='center', fontsize=12)
# plt.title('Temporal distribution of morphology clusters', fontsize=20, pad=20)
# plt.show()



'''# Hierarchical clustering  ???'''
transposed = phenotype_time.T
transposed

hl = hierarchy.linkage(transposed,method='ward',metric='euclidean')
hl
plt.figure(figsize=(20,14))
dendrogram = hierarchy.dendrogram(hl, orientation = 'bottom',labels = transposed.index)

hc = AgglomerativeClustering(n_clusters=7,linkage='ward') 
# affinity = 'euclidan'으로 하면 ward method 자체가 euclidian base라는 이유로 에러가 발생함
y_predicted = hc.fit_predict(transposed)
transposed['hc_cluster'] = y_predicted
transposed



#%%
"""TIME SERIES CLUSTERING """

df_time_series_all = pd.DataFrame()
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_pca.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    cell_data['data'] = cell_data[['PC1','PC2']].apply(tuple,axis=1)
    df_time_series_all_temp = cell_data[['time_point','data']].T
    df_time_series_all_temp.columns = df_time_series_all_temp.iloc[0]
    df_time_series_all_temp = df_time_series_all_temp.drop('time_point')
    df_time_series_all = pd.concat([df_time_series_all, df_time_series_all_temp], axis = 0)

df_time_series_all.reset_index(inplace=True)
df_time_series_all = df_time_series_all.drop(['index'], axis = 1)
df_time_series_all


#for full-time cells
label_fulltime_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()
df_time_series = pd.DataFrame()
for i in range(0, label_fulltime_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_fulltime_data[0][i][0],label_fulltime_data[0][i][1])).reset_index()
    cell_data['data'] = cell_data[['PC1','PC2']].apply(tuple,axis=1)
    df_time_series_temp = cell_data[['time_point','data']].T
    df_time_series_temp.columns = df_time_series_temp.iloc[0]
    df_time_series_temp = df_time_series_temp.drop('time_point')
    df_time_series = pd.concat([df_time_series, df_time_series_temp], axis = 0)

df_time_series.reset_index(inplace=True)
df_time_series = df_time_series.drop(['index'], axis = 1)
df_time_series

#change the data structure for time-series clustering
time_series_list = []
for cell_index in range(0, df_time_series.shape[0]):
    time_series_list_temp = []
    for time_point in range(0, df_time_series.columns.shape[0]):
        time_series_list_temp.append(df_time_series[df_time_series.columns[time_point]][cell_index])
    time_series_list.append(time_series_list_temp)
#time_series_data_list = np.array(time_series_data_list, dtype=np.dtype('float,float'))
time_series_list # dataframe을 (49,2) np.array가 473개 들어가 있는 list로 바꿈


time_series_list = to_time_series_dataset(time_series_list, dtype = 'object') 
# dtype ='float'가 default인데 그러면 tuple을 float 할 수 없다고 에러 뜸
# 'object'는 nan이나 int, float 등 섞인 datatype
time_series_list.shape # (49,2) np.array가 473개 들어가 있는 list -> (473(sample 수),49(time point 수),2(dimension)) np.array로 바꿈 


silhouette_coefficeints =[]
for k in tqdm(range(2,10), total = len(range(2,10))):
    tskm = TimeSeriesKMeans(n_clusters=k,metric = 'softdtw', random_state = 0, verbose = True)
    tskm.fit(time_series_list)
    score = tslearn.clustering.silhouette_score(time_series_list,tskm.labels_)
    silhouette_coefficeints.append(score)
silhouette_coefficeints

plt.xlabel('K')
plt.xticks(np.arange(1,10),fontsize=7)
plt.ylabel('Sillhouette Coefficient')
plt.plot(range(2,10),silhouette_coefficeints)
# choose the maximum value

#%%
'''n_cluster갯수 구한거 입력'''
tskm = TimeSeriesKMeans(n_clusters = 3, metric = 'softdtw', random_state = 0, verbose = True, max_iter = 100) 
# dtw하면 center trajectory 이상함
tskm

tskmeans_predicted = tskm.fit_predict(time_series_list)# fit하고 동시에 predict하는것
tskmeans_predicted


label_fulltime_data['tskm_cluster'] = tskmeans_predicted
label_fulltime_data



df_fulltime_cell['tskm_cluster'] = ''

for index, value in tqdm(df_fulltime_cell.iterrows(), total = df_fulltime_cell.shape[0]):
    for i in range(0, label_fulltime_data.shape[0]):
        
        if(value['image_stack'] == label_fulltime_data['image_stack'][i] and value['cell_label'] == label_fulltime_data['cell_label'][i]) == True:
            df_fulltime_cell['tskm_cluster'][index] = label_fulltime_data['tskm_cluster'][i]

df_fulltime_cell


#%%

fig = px.scatter(
data_frame = df_fulltime_cell,
x = 'PC1',
y = 'PC2',
color = 'tskm_cluster',
color_discrete_sequence = px.colors.qualitative.Set1,
#color_discrete_sequence = ['blue', 'green', 'yellow', 'red'],
#color_discrete_sequence = ['blue', 'green', 'red'],    
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
symbol = 'tskm_cluster',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Morphology Space',
#labels = {'PC1':'PC1('+str(round(df_variance['PC1'][1]*100,ndigits=1))+'%)',
#          'PC2':'PC2('+str(round(df_variance['PC2'][1]*100,ndigits=1))+'%)',
#          },
hover_data = {'cell_label': True, 'time_point':True, 'condition':True,'image_stack':True},
hover_name = df_fulltime_cell.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 4),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

for index, value in df_loadings.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = 3*value['PC1'], y1 = 3*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = 3*value['PC1'], y = 3*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                       text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

pio.show(fig)
fig.write_html("C:/Users/user/Desktop/" + '2d_tskm_cluster.html')



#%%

import matplotlib.cm as cm

# df_fulltime_cell = pd.read_csv('C:/Users/makav/Desktop/breastcaf_fulltime_cell_120623.csv')
colors = itertools.cycle(["sienna", "darkorange", "peachpuff", "wheat","c","m"])
plt.figure(figsize=(20,15))
groups = df_fulltime_cell.groupby('tskm_cluster')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o', linestyle='', alpha=.3, markersize=7, label=name, color=next(colors))
# plt.legend(loc='best', prop={'size':20})
plt.xlim(-13, 15)
plt.ylim(-12, 18)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)
plt.grid(False)
plt.axis('off')
plt.savefig('morphology_timeseries_kmeans_withoutframe.png', dpi=600)


###################
fig, ax = plt.subplots(figsize=(20,15))
groups = df_fulltime_cell.groupby('tskm_cluster')
for name, group in groups:
    ax.scatter(group.PC1, group.PC2, label=name, alpha=0.4, s=80)
ax.legend()
plt.xlim(-13, 15)
plt.ylim(-12, 18)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)




# plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'],marker=',',s=10, 
#                 c=df_fulltime_cell['tskm_cluster'],cmap = plt.cm.get_cmap('Set2'))
# # plt.legend(loc='best')
# plt.xlim(-13, 15)
# plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=20)
# plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=20)

#%%


""""Morphology trajectories"""
plt.figure(figsize=(20,15))
colorcodes = ['tab:blue', 'tab:orange','tab:green']
colors = itertools.cycle(colorcodes)
color = colorcodes

groups = df_fulltime_cell.groupby('tskm_cluster')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o',alpha=.01, linestyle='', markersize=10, label=name, color=next(colors))
plt.legend(loc='best', fontsize=20)
plt.xlim(-13, 15)
plt.ylim(-12, 18)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('PC1('+str(round(variance_ratio[0]*100,ndigits=1))+'%)', fontsize=40)
plt.ylabel('PC2('+str(round(variance_ratio[1]*100,ndigits=1))+'%)', fontsize=40)
for cluster in range(0,tskm.cluster_centers_.shape[0]):
    plt.quiver(tskm.cluster_centers_[cluster][:-1,0],tskm.cluster_centers_[cluster][:-1,1],
               tskm.cluster_centers_[cluster][1:,0]-tskm.cluster_centers_[cluster][:-1,0],
               tskm.cluster_centers_[cluster][1:,1]-tskm.cluster_centers_[cluster][:-1,1], 
               scale_units='xy', angles='xy', scale=1, color = color[cluster], label = cluster)





# plt.figure(figsize=(15,12))
# color = ['red','green','blue','darkorange', 'magenta','purple','brown', 'aqua', 'olive', 'cyan' ]
# #plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1,c='cornflowerblue')
# plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.2, 
#             c=df_fulltime_cell['tskm_cluster'],cmap = plt.cm.get_cmap('Set3'))
# for cluster in range(0,tskm.cluster_centers_.shape[0]):
#     plt.quiver(tskm.cluster_centers_[cluster][:-1,0],tskm.cluster_centers_[cluster][:-1,1],
#                tskm.cluster_centers_[cluster][1:,0]-tskm.cluster_centers_[cluster][:-1,0],
#                tskm.cluster_centers_[cluster][1:,1]-tskm.cluster_centers_[cluster][:-1,1], 
#                scale_units='xy', angles='xy', scale=1, color = color[cluster], label = cluster)
# # [:-1, 0]은 n개의 x좌표 중 x0부터 x(n-1)까지만 추출
# [:-1, 1]은 n개의 y좌표 중 x0부터 x(n-1)까지만 추출
# [1:,0]은 n개의 x좌표 중 x1부터 xn까지만 추출
# [1:,1]은 n개의 y좌표 중 x1부터 xn까지만 추출
# [1:,0] - [:-1,0]은 따라서 x1-x0, x2-x1, ... xn-x(n-1)을 의미
# [1:,1] - [:-1,1]은 따라서 y1-y0, y2-y1, ... yn-y(n-1)을 의미

######## variable labelling ########

#const = 3
#plt.scatter(const*df_no_rotation_loadings_final['PC1'],const*df_no_rotation_loadings_final['PC2'],color = 'black',s=50,marker='^')

#for i in range(0,df_no_rotation_loadings_final.shape[0]):
#    x_values = [0, const*df_no_rotation_loadings_final['PC1'][i]]
#    y_values = [0, const*df_no_rotation_loadings_final['PC2'][i]]
#    plt.plot(x_values, y_values, alpha = 0.3)

#for parameter, x, y in zip(df_no_rotation_loadings_final['parameter'],const*df_no_rotation_loadings_final['PC1'],const*df_no_rotation_loadings_final['PC2']):
#    plt.annotate(parameter,xy = (x,y),xytext=(-5,5),textcoords='offset points',ha = 'right', va = 'bottom', fontsize = 10
                 #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
#                )

#plt.legend(loc = 'upper right')
# plt.xlim(-15, 15)
# plt.ylim(-15, 15)
# plt.show()


"""random example trajectory of a cell """

aaa = df_fulltime_cell[(df_fulltime_cell['image_stack']==label_fulltime_data[0][0][0]) & (df_fulltime_cell['cell_label']==label_fulltime_data[0][0][1])] 
dot_c1=np.arange(aaa.shape[0]-1) # tskm.cluster_centers_ = (cluster 수, time point 수, dimension)


plt.figure(figsize=(24,24))
for i in range(0,3):
    j = np.random.randint(0,label_fulltime_data.shape[0])
    aaa = df_fulltime_cell[(df_fulltime_cell['image_stack']==label_fulltime_data[0][j][0]) & (df_fulltime_cell['cell_label']==label_fulltime_data[0][j][1])] 
    plt.subplot(3,3,i+1)
    plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=3, alpha = 0.1, c='cornflowerblue')
    plt.quiver(np.array(aaa['PC1'][:-1]),np.array(aaa['PC2'][:-1]),
               np.array(aaa['PC1'][1:]) - np.array(aaa['PC1'][:-1]),
               np.array(aaa['PC2'][1:]) - np.array(aaa['PC2'][:-1]), 
               dot_c1, scale_units='xy', angles='xy', scale=1, cmap = plt.cm.get_cmap('jet'))
    plt.xlim(-13, 15)
    plt.ylim(-12, 18)
    # plt.ylim(-15, 15)
# c = dot_c1하면 안되고 c 위치에 dot_c1을 입력해야됨
    # plt.title('tskms_cluster: '+ str(np.array(aaa['tskm_cluster'])[0]) + '  ' + str(label_fulltime_data[0][j][0])+ '  index: ' + str(j))


""" trajectories of clusters """
dot_c1=np.arange(tskm.cluster_centers_.shape[1]-1) # tskm.cluster_centers_ = (cluster 수, time point 수, dimension)

plt.figure(figsize=(20,15))
for cluster in range(0,tskm.cluster_centers_.shape[0]):
    colors = {0:'cornflowerblue', 1:'cornflowerblue', 2:'cornflowerblue', 3:'cornflowerblue', 4:'cornflowerblue', 5:'cornflowerblue', 
               6:'cornflowerblue', 7:'cornflowerblue', 8:'cornflowerblue', cluster:'indianred'}
    plt.subplot(3,3,cluster+1)
    plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1, 
                c=df_fulltime_cell['tskm_cluster'].map(colors))
    plt.quiver(tskm.cluster_centers_[cluster][:-1,0],tskm.cluster_centers_[cluster][:-1,1],
               tskm.cluster_centers_[cluster][1:,0]-tskm.cluster_centers_[cluster][:-1,0],
               tskm.cluster_centers_[cluster][1:,1]-tskm.cluster_centers_[cluster][:-1,1], 
               dot_c1, scale_units='xy', angles='xy', scale=1, cmap = plt.cm.get_cmap('jet'), label = cluster)
    #plt.xlim(-2, 6)
    #plt.ylim(-5.5, 3.5)
# c = dot_c1하면 안되고 c 위치에 dot_c1을 입력해야됨
#    for i in range(0,df_no_rotation_loadings_final.shape[0]):
#        const = 3
#        x_values = [0, const*df_no_rotation_loadings_final['PC1'][i]]
#        y_values = [0, const*df_no_rotation_loadings_final['PC2'][i]]
#        plt.plot(x_values, y_values, alpha = 0.3)
    
#    plt.legend()
#plt.colorbar()  

# [:-1, 0]은 n개의 x좌표 중 x0부터 x(n-1)까지만 추출
# [:-1, 1]은 n개의 y좌표 중 x0부터 x(n-1)까지만 추출
# [1:,0]은 n개의 x좌표 중 x1부터 xn까지만 추출
# [1:,1]은 n개의 y좌표 중 x1부터 xn까지만 추출
# [1:,0] - [:-1,0]은 따라서 x1-x0, x2-x1, ... xn-x(n-1)을 의미
# [1:,1] - [:-1,1]은 따라서 y1-y0, y2-y1, ... yn-y(n-1)을 의미

"""Distribution of tskm clusters"""

tskm_condition = pd.DataFrame(df_fulltime_cell.groupby(['condition','tskm_cluster']).size())
tskm_condition['distribution(%)'] = tskm_condition.groupby(level=0).apply(lambda x:  100*x / x.sum())
tskm_condition = tskm_condition.drop([0],axis=1)
tskm_condition = tskm_condition.unstack(level=0)
tskm_condition = tskm_condition.fillna(0) # NaN -> 0으로 바꿈
tskm_condition = tskm_condition[[('distribution(%)','control'), ('distribution(%)','with7'), ('distribution(%)','with231')]]


'''
#to change the column order(put control first)
cols_tskm = tskm_condition.columns.tolist()
cols_tskm = cols_tskm[-1:] + cols_tskm[:-1]
cols_tskm
tskm_condition = tskm_condition[cols_tskm]
tskm_condition
'''


plt.figure(figsize=(15,10))
heatmap = sns.heatmap(tskm_condition.T, annot = True, annot_kws={"size": 20}, cmap = 'Oranges')
# plt.xlabel('Mrphodynamic cluster')
# plt.ylabel('Phenotype')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=15)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=25)
# plt.xlabel('Morphodynamic cluster', fontsize = 30, labelpad=10)
# plt.ylabel('Condition', fontsize = 30, labelpad=10)
# plt.title('Distribution of morphodynamic cluster', fontsize = 20)
plt.savefig('heatmap_morphodynamic.eps', format='eps', dpi=600)

plt.show()




"""Shannon Entropy"""

shannon_entropy_list = []
for column in list(tskm_condition['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, tskm_condition['distribution(%)'][column].shape[0]):
        if tskm_condition['distribution(%)'][column][i] == 0:
            continue
        else:
            shannon_entropy = shannon_entropy + -tskm_condition['distribution(%)'][column][i]/100*np.log2(tskm_condition['distribution(%)'][column][i]/100)
    shannon_entropy_list.append(shannon_entropy)
    print(column)
shannon_entropy_list

plt.figure(figsize=(8,8))
colors = ['green', 'royalblue', 'indianred']
# plt.title('Shannon entropy', fontsize = 30)
plt.bar(np.arange(len(shannon_entropy_list)), shannon_entropy_list, color=colors, width = 0.4, edgecolor = "black", linewidth=3)
plt.xticks(np.arange(len(shannon_entropy_list)), tskm_condition['distribution(%)'].columns.tolist(),fontsize=25)
plt.yticks(fontsize=20)
plt.ylim(1.4, 1.6)
plt.xticks([], [])
plt.savefig('Shannon_entropy_morphodynamic.eps', format='eps', dpi=600)

plt.show()



""""Hierarchical clustering"""
transposed = tskm_condition.T
transposed
hl = hierarchy.linkage(transposed,method='average',metric='euclidean')
hl
plt.figure(figsize=(2,10))
dendrogram = hierarchy.dendrogram(hl, orientation = 'left',labels = transposed.index)
hc = AgglomerativeClustering(n_clusters=2,linkage='average') 
# affinity = 'euclidan'으로 하면 ward method 자체가 euclidian base라는 이유로 에러가 발생함
y_predicted = hc.fit_predict(transposed)
transposed['hc_cluster'] = y_predicted
transposed


















"""???Morphology Dynamics - Euclidean distance"""""
clone_number = pd.DataFrame(df_pca.groupby(['time_point','condition']).size())
cclone_number = clone_number.unstack(level=0)
clone_number = clone_number.fillna(0)
clone_number

from math import sqrt
from statistics import mean

cell_distance_list = []
average_distance_list = []
time_list = []

label_data = pd.DataFrame(df_pca.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()

for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_pca.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보
    distance = 0
    distance_list = []
    
    for j in range(0,cell_data.shape[0]-1): # 한 세포 안에서 각 time frame 마다
        distance = sqrt((cell_data['PC1'][j] - cell_data['PC1'][j+1])**2 + (cell_data['PC2'][j] - cell_data['PC2'][j+1])**2)
        # euclidean distance
        distance_list.append(distance)
     
    print(i, label_data[0][i], cell_data.shape[0])
    cell_distance_list.append(distance_list)  
    
    if len(distance_list) == 0: # 한 세포에 1개의 time_frame만 있으면 거리 계산이 안되서 len(distance_list) = 0이 됨
        average_distance = 0 # average_distance = 0을 안해주면 mean(distance_list)의 분모가 0이 되어 오류 발생
        
    else:
        average_distance = mean(distance_list)
    average_distance_list.append(average_distance)
    
    time_list.append(cell_data.shape[0])
    
time_list = np.array(time_list)
average_distance_list = np.array(average_distance_list)

df_average_distance = pd.DataFrame(average_distance_list, columns=['average_distance'])
df_time = pd.DataFrame(time_list, columns=['number_of_time'])
df_average_distance


df_dynamics = pd.concat([label_data, df_average_distance, df_time],axis=1)
df_dynamics = df_dynamics[df_dynamics['number_of_time']==df_dynamics['number_of_time'].max()] 
# 20 time step 동안은 single-cell인 세포만 추출
df_dynamics.reset_index(inplace=True)
df_dynamics = df_dynamics.drop('index', axis = 1)
df_dynamics

df_dynamics.groupby('condition').describe()

import plotly.graph_objects as go
fig = px.box(
data_frame = df_dynamics,
x = 'condition',
y = 'average_distance',
color = 'condition',
points= 'all',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
#title = 'Clone 1-1 Morphology Space',
labels = {'clone_type':'clone type',
          'average_distance':'average euclidean distance between each time step',
          },
hover_data = {'condition': True, 'cell_label':True},
hover_name = df_dynamics.index,

height = 500,
width = 700,

)

fig.update_traces(marker = dict(size = 2),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

pio.show(fig)

fig = px.strip(
data_frame = df_dynamics,
x = 'condition',
y = 'average_distance',
color = 'condition',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
#title = 'Clone 1-1 Morphology Space',
labels = {'clone_type':'clone type',
          'average_distance':'morphological variability',
          },
hover_data = {'condition': True, 'cell_label':True},
hover_name = df_dynamics.index,

height = 500,
width = 700,
)

fig.update_traces(marker = dict(size = 2),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

pio.show(fig)
###################
#%%

"""Cell centroid data for Heteromotility analysis"""

cell_position_list = []
label_data = pd.DataFrame(df.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()

for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_position = df.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1]))[['x_c','y_c','time_point','image_stack','cell_label','condition']]
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보 
    cell_position.reset_index(inplace=True)
    cell_position = cell_position.drop('index', axis = 1)
    cell_position['number_of_appearance'] = cell_position.shape[0]
    cell_position_list.append(cell_position)

df_cell_position = pd.DataFrame()
for j in range(0, len(cell_position_list)):
    df_cell_position = pd.concat([df_cell_position, cell_position_list[j]], axis=0)
df_cell_position.reset_index(inplace=True)
df_cell_position = df_cell_position.drop('index', axis = 1)
df_cell_position

df_fulltime_position = df_cell_position[df_cell_position['number_of_appearance']==df_cell_position['number_of_appearance'].max()] 
df_fulltime_position.reset_index(inplace=True)
df_fulltime_position = df_fulltime_position.drop('index', axis = 1)
df_fulltime_position


label_data = pd.DataFrame(df.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()
df_position_all = pd.DataFrame()
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    cell_data['data'] = cell_data[['x_c','y_c']].apply(tuple,axis=1)
    df_position_temp = cell_data[['time_point','data']].T
    df_position_temp.columns = df_position_temp.iloc[0]
    df_position_temp = df_position_temp.drop('time_point')
    df_position_all = pd.concat([df_position_all, df_position_temp], axis = 0)

df_position_all.reset_index(inplace=True)
df_position_all = df_position_all.drop(['index'], axis = 1)
df_position_all

label_fulltime_data = pd.DataFrame(df_fulltime_position.groupby(['image_stack','cell_label','condition']).apply(lambda x : x.name)).reset_index()
df_position = pd.DataFrame()
for i in range(0, label_fulltime_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_position.groupby(['image_stack','cell_label']).get_group((label_fulltime_data[0][i][0],label_fulltime_data[0][i][1])).reset_index()
    cell_data['data'] = cell_data[['x_c','y_c']].apply(tuple,axis=1)
    df_position_temp = cell_data[['time_point','data']].T
    df_position_temp.columns = df_position_temp.iloc[0]
    df_position_temp = df_position_temp.drop('time_point')
    df_position = pd.concat([df_position, df_position_temp], axis = 0)

df_position.reset_index(inplace=True)
df_position = df_position.drop(['index'], axis = 1)
df_position

position_dict = {}
for cell_index in range(0, df_position.shape[0]):
    position_list_temp = []
    for time_point in range(0, df_position.columns.shape[0]):
        position_list_temp.append(df_position[df_position.columns[time_point]][cell_index])
    position_dict[cell_index] = position_list_temp
#time_series_data_list = np.array(time_series_data_list, dtype=np.dtype('float,float'))
position_dict # dataframe을 (49,2) np.array가 473개 들어가 있는 dict로 바꿈


position_dict_all = {}
for cell_index in range(0, df_position_all.shape[0]):
    position_list_temp = []
    for time_point in range(0, df_position_all.columns.shape[0]):
        position_list_temp.append(df_position_all[df_position_all.columns[time_point]][cell_index])
    position_dict_all[cell_index] = position_list_temp
#time_series_data_list = np.array(time_series_data_list, dtype=np.dtype('float,float'))
position_dict_all # dataframe을 (49,2) np.array가 473개 들어가 있는 dict로 바꿈

#%%
# from application.app.folder.file import func_name
"""Module 어떻게?"""

time_unit = 1/6 # 한 time frame마다의 time unit (hour 기준)
range_for_moving = np.arange(30,80,20) #um/hr
# morphology에 의한 moving이 아닌 실제 migration에 의한 moving이라고 생각되는 속도 범위(for time_moving, avg_moving_speed)
range_for_timedelay = range(3,50,10) # frame 당
# 기준 cell path의 position부터 timedelay만큼 뒤에 path position 고려(for turn_stats,
time_delay = 5 # time_delay for autocorrelation and partial autocorrelation

'''
1. Calculate general motility features (average speed, distance, ...)
2. Calculate features related to mean squared displacement
3. Calculate features related to random walk similarity
'''

#from __future__ import division, print_function
import numpy as np
import math
from scipy import stats
from scipy import interpolate
from statsmodels.tsa.stattools import acf, pacf


def average_xy( coor1, coor2 ):
    '''Finds the average value of two XY coordinates'''
    x1 = float(coor1[0])
    x2 = float(coor2[0])
    y1 = float(coor1[1])
    y2 = float(coor2[1])

    x3 = (x1 + x2) * 0.50
    y3 = (y1 + y2) * 0.50
    return (x3, y3)

def distance( coor1, coor2 ):
    '''Euclidean distance'''
    x1, y1 = coor1
    x2, y2 = coor2
    d = math.sqrt( (x2-x1)**2.00 + (y2-y1)**2.00)
    return d

class GeneralFeatures(object):
    '''
    Calculates features of speed, distance, path shape, and turning behavior.
    Attributes
    ----------
    cell_ids : dict
        keyed by cell ids, values are lists of coordinate tuples.
    moving_tau : int
        threshold for movement
    tau_range : iterable, ints
        range of window sizes to use to establish cell direction by regression.
    interval_range : iterable, ints
        range of time intervals to use between a cell's current and 'next'
        direction when calculating turning features.
    total_distance : dict
        total distance each cell travels.
        keyed by cell id, values are floats.
    net_distance : dict
        net distance each cell traveled, dist(final_pos - init_pos)
        keyed by cell id, values are floats.
    linearity : dict
        Pearson's r**2 linearity metric on cell paths.
        keyed by cell id, values are floats.
    spearmanrsq : dict
        Spearman's rho**2 monotonicity metric on cell paths.
        keyed by cell id, values are floats.
    progressivity : dict
        net_distance/total_distance, metric of directional persistence.
        keyed by cell id, values are floats.
    min_speed : dict
        minimum cell speed.
        keyed by cell id, values are floats.
    max_speed : dict
        maximum cell speed.
        keyed by cell id, values are floats.
    avg_moving_speed : dict
        average moving speed above a certain threshold speed,
        defined as the movement cutoff.
        useful to deconfound the cell's average speed while moving from the
        dwell time in an immotile state.
        keyed by threshold, values are dicts keyed by cell id, valued with floats.
    time_moving : dict
        proportion of time spent moving above a certain threshold speed,
        defined as the movement cutoff.
        from 0 to 1, 0 means no motion, 1 means motion for all frames
        keyed by threshold, values are dicts keyed by cell id, valued with floats.
    turn_stats : dict
        proportion of the time a cell turns to the right of its past direction.
        useful to determine directional bias.
        keyed by `tau` from `tau_range`, values are dicts keyed by `interval` from `interval_range`.
        final dicts have float values, [0, 1].
    theta_stats : dict
        average turning angle magnitude.
        keyed by `tau` from `tau_range`, values are dicts keyed by `interval` from `interval_range`.
        final dicts have float values, [0, 1].
    '''
    def __init__(self, cell_ids, move_thresh):
        '''
        Parameters
        ----------
        cell_ids : dict
            keyed by cell ids, values are lists of coordinate tuples.
        move_thresh : int
            time interval to use when calculating `avg_moving_speed`, `time_moving`
        '''
        self.cell_ids = cell_ids
        if len(cell_ids[list(cell_ids)[0]]) > 19:
            self.moving_tau = move_thresh
            self.tau_range = range_for_timedelay
            self.interval_range = 2
        elif 9 < len(cell_ids[list(cell_ids)[0]]) <= 19:
            self.moving_tau = move_thresh
            self.tau_range = range_for_timedelay
            self.interval_range = 2
        else:
            print("Time series too small to calculate turning features")
        self.total_distance, self.avg_speed, self.all_speed_values = self.calc_total_distance(self.cell_ids)
        self.net_distance = self.calc_net_distance(self.cell_ids)
        self.linearity = self.calc_linearity(self.cell_ids)
        self.spearmanrsq = self.calc_spearman(self.cell_ids)
        self.progressivity = self.calc_progressivity(self.cell_ids, net_distance=self.net_distance, total_distance=self.total_distance)
        self.min_speed, self.max_speed = self.calc_minmax_speed(self.cell_ids)
        self.avg_moving_speed, self.time_moving = self.calc_moving_variable_threshold(cell_ids, thresholds = range_for_moving, tau = self.moving_tau)

        self.turn_stats, self.average_theta, self.min_theta, self.max_theta = self.all_turn_stats(cell_ids, tau_range = self.tau_range, interval_range = self.interval_range)

    def calc_total_distance(self, cell_ids):
        total_distance = {}
        avg_speed = {}
        all_speed_values = {}
        for u in cell_ids: # u는 세포 하나의 index
            cell = cell_ids[u] #  cell은 t=0 ~ t=T 까지
            start_point = cell[0] # start_point = (x0, y0)
            traveled = []
            for coor in cell[1:]: # coor = (x1, y1)
                d = distance( start_point, coor )
                traveled.append(d)
                start_point = coor # start_point = (x1, y1)로 바뀜...
            total_distance[u] = sum(traveled)
            avg_speed[u] = np.mean(traveled)/time_unit
            all_speed_values[u] = [i/time_unit for i in traveled]

        return total_distance, avg_speed, all_speed_values

    # Calculate net distance traveled and save each cell's stat
    # as a float in a list net_distance
    # Simple distance b/w start and end points
    def calc_net_distance(self, cell_ids):
        net_distance = {}
        for u in cell_ids: # u는 세포 하나의 index
            cell = cell_ids[u] #  cell은 t=0 ~ t=T 까지
            start_point = cell[0]
            end_point = cell[-1]
            d = distance( start_point, end_point)
            net_distance[u] = d
        return net_distance

    # Calculate linearity of the traveled path as the R^2 value
    # of a linear regression of all path points
    def calc_linearity(self, cell_ids):
        linearity = {}
        for u in cell_ids:
            cell = cell_ids[u]
            x = []
            y = []
            for coor in cell:
                x.append( coor[0] )
                y.append( coor[1] )
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            r_squared = r_value**2
            linearity[u] = r_squared
        return linearity

    # Calculate Spearman's rho^2 for all XY points in a cell path
    def calc_spearman(self, cell_ids):
        spearmanrsq = {}
        for u in cell_ids:
            xy = np.array( cell_ids[u] )
            if len(np.unique(xy[:,0])) == 1 or len(np.unique(xy[:,1])) == 1:
                # fuzz to avoid bounds issue
                print('Tracks for cell_id %s were invariant in at least one dimension.' % u)
                print('Spearman coefficient is undefined, and should not be used for analysis.')
                rho = 0.
            else:
                rho, pval = stats.spearmanr( xy )
            spearmanrsq[u] = rho**2 # 방향성을 없애기 위해 제곱
        return spearmanrsq

    def calc_progressivity(self, cell_ids, net_distance, total_distance):
        progressivity = {}
        for u in cell_ids:
            progressivity[u] = float(net_distance[u]) / float(total_distance[u])

        return progressivity

    def calc_minmax_speed(self, cell_ids):
        '''
        Calculate max/min speeds as the distance traveled
        across 5 frames and append as a float to list max_speed/min_speed
        Speeds in the unit of pixels per hour
        '''
        max_speed = {}
        min_speed = {}
        for u in cell_ids:
            cell = cell_ids[u]
            speeds = []
            i = 0
            while (i + 1) < len(cell):
                d = distance( cell[i], cell[i+1] ) # t = 0~1 distance, 1~2 distance , ...
                speeds.append( d )
                i += 1
            max_speed[u] = max(speeds) / time_unit
            min_speed[u] = min(speeds) / time_unit
        return min_speed, max_speed

    def calc_moving_stats(self, cell_ids, move_speed, tau):
        '''
        Calculates the proportion of time a cell spends moving_speeds,
        min, max, and average speeds while moving
        Parameters
        ------------
        cell_ids : dict keyed by cell id, containing lists of tupled XY coors
        move_speed : minimum speed to be considered "moving"
        tau : time lag to use when determining movement speed, 
        '''
        avg_moving_speed = {}
        time_moving = {}
        for u in cell_ids:
            cell = cell_ids[u]
            speeds = []
            i = 0
            while (i + tau) < len(cell):  # gf의 move_thresh (= 1) 와 동일
                d = distance( cell[i], cell[i+tau] )
                speeds.append( d / (tau*time_unit) )
                i += 1
            moving_speeds = []
            for val in speeds:
                if val > move_speed:
                    moving_speeds.append( val )
                else:
                    pass

            time_moving[u] = float(len(moving_speeds)) / float(len(speeds)) # 0에서 1값 (1은 항상 움직임, 0은 안움직임)

            if moving_speeds == []:
                moving_speeds.append(0)
            else:
                pass
            avg_moving_speed[u] = np.mean( moving_speeds )

        return avg_moving_speed, time_moving

    def calc_moving_variable_threshold(self, cell_ids, thresholds , tau):

        # dicts keyed by moving speed threshold used for calc
        # each key corresponds to a dict keyed by cell_id, with vals for moving speed
        # and time moving for the given threshold in the top level dict
        avg_moving_speed = {}
        time_moving = {}

        for index, thresh in enumerate(thresholds):
            avg_moving_speed[index], time_moving[index] = self.calc_moving_stats( cell_ids, thresh, tau )

        return avg_moving_speed, time_moving

    def turn_direction(self, cell_path, tau = 3, interval = 2):
        '''
        Returns the proportion of time a cell turns left as float 0.0-1.0.
        Parameters
        -----------
        cell_path : list
            timeseries of tuples, each tuple holding an XY position
            i.e. cell_path = [(x1,y1), (x2,y2)...(xn,yn)]
        tau : int
            desired time lag between a point of
            interest and a point in the distance (p_n+tau) to determine
            a cell's turning behavior. Must be > `interval`.
        interval  : int
            number of points ahead and behind the point of interest to consider
            when find a regression line to represent the cell's direction at p_n.
        Returns
        -------
        turns  : list
            binaries reflecting turns left (0) or right (1)
        thetas : list
            angles of each turn in radians
        Notes
        -----
        (1) Estabish direction of object at p_n:
        Given a time series XY of x,y coordinates, where p_t denotes a point
        at time t, take a given point p_n
        determine the 'direction' of motion at p_n by plotting a linear reg.
        on points [p_n-i, p_n+i] for some interval value i
        This linear regression function is R(x) = slope*x + b
        (2) Determine if p_n+tau is left or right
        For a point p_n+tau, for a variable time lag tau, determine if p_n+tau
        lies left or right of R
            p_n+tau,Y > R(p_n+tau) and initial direction = right : left turn
            p_n+tau,Y < R(p_n+tau) and initial direction = right : right turn
            ...
        (3) Calculate magnitude of turn theta
        Calculate the angle of the turn theta as the angle between
            R(x) from (1)
            R2, line connecting p_n to p_n+tau
            v1 = (1, slope of R(x))
            v2 = (1, slope of R2)
            theta = arccos( v1 dot v2 / ||v1|| * ||v2||)
        N.B. If a cell moves in a perfectly straight line, such that point of
        interest p_n has x coordinate == p_n+tau, the function skips this p_n
        i.e. Skips if the cell has moved in a perfectly linear line, or circled
        back on itself
        '''
        if tau < interval:
            print('tau must be > interval')
            return None
        else:
            pass

        XY = np.asarray(cell_path) # 세포 1개의 time series (x,y) data

        turns = []
        thetas = []
        # do every 3rd point to cut runtime
        for n in range( interval, len(XY)-tau ): # len(XY) = time_frame 수 (=49)
            p_n = XY[n]                        # p_n은 t=n 일때의 x,y 좌표
            p_tau = XY[n+tau]                 # p_tau는 t = n+tau일때의 x,y 좌표
            if p_n[0] == p_tau[0]:           # p_n의 x좌표와 p_tau의 x좌표가 같으면 code skip
                continue
            direction = p_tau[0] - p_n[0]    # direction = p_tau x좌표 - p_n x좌표
            regress_p = XY[n-interval:n+interval+1] # regress_p = t-interval ~ t+interval 까지의 (x,y) 좌표
            if all(regress_p[:,0] == regress_p[0,0]): # skip p_n if all x coors in interval are equal
                continue
            R_slope, b, rval, pval, stderr = stats.linregress(regress_p[:,0], regress_p[:,1]) 
            # regress_[:,0] = x coordinates of one cell's time series trajectory
            # regress_[:,1] = y coordinates of one cell's time series trajectory
            # linregress(x_list,y_list) -> y = R_slope*x + b 형태로 fitting
            if np.isnan(R_slope): # skip if cell moves entirely on a vertical line for given interval period
                continue
            R_tau = R_slope*(p_tau[0]) + b # R_tau = R_slope * (x coordinate of p_tau) + b
            # direction > 0 = +x movement, < 0 = -x
            # turn = 0 : left
            # turn = 1 : right
            if direction > 0:        # direction > 0 = +x movement (->)
                if R_tau < p_tau[1]: # p_tau[1] = y coordinate of p_tau
                    turn = 0
                else:
                    turn = 1
            else:                    # direction < 0 = -x movement (<-)
                if R_tau < p_tau[1]:
                    turn = 1
                else:
                    turn = 0
            turns.append(turn)

            # find line R2 connecting p_n and p_tau
            R2_slope = (p_tau[1] - p_n[1]) / (p_tau[0] - p_n[0])
            R2_intercept = p_n[1] - R2_slope*p_n[0]
            # create vectors v1 : R, v2 : R2
            v1 = np.array( [1, R_slope] )
            v2 = np.array( [1, R2_slope] )
            
            if (np.linalg.norm(v1)*np.linalg.norm(v2)) == 0:
                print("error (np.linalg.norm(v1)*np.linalg.norm(v2)) == 0")
                print(n, p_n, p_tau)
                continue
            costheta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) # cos(theta) = v1 dot v2 / ||v1|| * ||v2||
            #print "v1 = ", v1, "| v2 = ", v2, "| costheta = ", costheta
            if costheta < -1.0 or costheta > 1.0:
                #print "Error calculating turn magnitude"
                #print "costheta = ", costheta
                #print "v1, v2 = ", v1, v2
                continue
            theta = math.acos(costheta) # arccos(cos(theta)) = theta
            thetas.append(theta)

            # If no turns are detected, add neutral values to avoid np.mean Error
            # 0 turn mag, neutral turn direction

        if len(thetas) == 0:
            thetas.append(0)
            turns.append(0.5)
            return turns, thetas
        else:
            return turns, thetas

    def all_turn_stats(self, cell_ids,
                        tau_range = range(3,13), interval_range = 2):
        '''
        Parameters
        ----------
        cell_ids : dict
            keyed by cell id, values are lists of tuple coordinates.
        tau_range : iterable
            range of time lags to try for calculation.
        interval_range : iterable
            range of interval distances to use for calculation of
            the linregress about point of interest p_n.
        Returns
        ----------
        turn_stats : dict
            keyed by tau, containing dict keyed by interval,
            containing dict keyed by cell, containing proportion of
            time a cell turns right
        theta_stats : dict
            keyed by tau, containing dict keyed by interval,
            containing dict keyed by cell, containing a list of
            three elements [avg_theta, min_theta, max_theta]
        '''
        turn_stats = {}
        average_theta = {}
        min_theta = {}
        max_theta = {}
        for index, tau in enumerate(tau_range):
            turn_stats[index] = {}
            average_theta[index] = {}
            min_theta[index] = {}
            max_theta[index] = {}
                
            for u in cell_ids:
                cell_path = cell_ids[u]
                turns, thetas = self.turn_direction(cell_path, tau = tau, interval = interval_range)
                turn_stats[index][u] = np.mean(turns)
                average_theta[index][u] = np.mean(thetas)
                min_theta[index][u] = np.min(thetas)
                max_theta[index][u] = np.max(thetas)
                
        return turn_stats, average_theta, min_theta, max_theta

#%%
class MSDFeatures(object):
    '''
    Calculates mean squared displacement related features.
    Attributes
    ----------
    cell_ids : dict
        keyed by cell ids, values are lists of coordinate tuples.
    msd_distributions : dict
        MSDs for each cell across a range of time lags `tau`.
        keyed by cell ids, values are lists of floats.
    log_distributions : dict
        log transform of `msd_distributions`.
        keyed by cell ids, values are lists of floats.
    alphas : dict
        alpha exponent for each cells motion.
        keyed by cell ids, values are floats.
    Notes
    -----
    MSD(tau) = (1 / tau) * sum( (x(t + tau) + x(t))^2 )
    Plotting a distribution of Tau vs MSD will generate a curve.
    The exponential nature of this curve will describe the type
    of motion the particle is experiencing.
    1 < alpha --> impeded diffusion
    Linear curve (alpha = 1) --> diffusion
    1 < alpha < 2 --> super diffusion
    alpha = 2 --> ballistic motion
    '''
    def __init__(self, cell_ids):
        self.msd_distributions, self.log_distributions = self.msd_dist_all_cells(cell_ids, tau = 31)
        self.alphas = self.calc_alphas(self.log_distributions)

    def calc_msd(self, path, tau):
        '''
        Calculates mean squared displacement for a cell path
        for a given time lag tau
        Parameters
        ----------
        path : list
            tuples of sequential XY coordinates.
        tau  : int
            time lag to consider when calculating MSD.
        Returns
        -------
        msd : float
            mean squared displacement of path given time lag tau.
        '''
        distances = []
        t = 0
        while (t + tau) < len(path): 
            distsq = ( distance(path[ t + tau ], path[ t ]) )**2
            distances.append(distsq)
            t += 1

        msd = sum(distances)/t

        return msd

    def calc_msd_distribution(self, path, max_tau):
        '''
        Calculates the distribution of MSDs for a range of time
        lag values tau.
        Parameters
        ----------
        path : list
            tuples containing sequential XY coordinates for each cell
        max_tau : int
            maximum time lag `tau` to consider for MSD calculation.
        Returns
        -------
        distribution : list
            MSDs indexed by `tau`.
        log_distribution : list
            log transform of `distribution`.
        Notes
        -----
        If a cell is stationary indefinitely, it effectively has
        a MSD of `0`.
        However, this calls a math domain error, so checks are in
        place that ensure the final `alpha` calculation will be
        0 without raising exceptions.
        Here -- returns both distributions as a string 'flat' if
        MSD calc is 0 for a given range tau
        '''
        tau = 1
        distribution = []
        log_distribution = []
        while tau < max_tau:
            msd = self.calc_msd(path, tau)
            distribution.append(msd)
            if msd > 0:
                log_distribution.append( math.log(msd) )
            else:
                distribution = 'flat'
                log_distribution = 'flat'
                return distribution, log_distribution
            tau += 1

        return distribution, log_distribution

    def msd_dist_all_cells(self, cell_ids, tau = 31):
        '''
        Calculates MSD distributions for all cells in dict cell_ids
        Parameters
        -------------------
        cell_ids : dictionary keyed by cell_id, contains lists of
                   tuples of sequential XY coordinates
        tau      : maximum time lag to calculate for MSD distributions
                   n.b. if path < 30 t's long, uses len(path) as max tau
        Returns
        -------------------
        msd_distributions : dict keyed by cell_id containing list
                            of MSD distributions, indexed by tau
        log_distributions : dict with log transformed lists of
                            MSD distributions
        Notable behavior
        -------------------
        checks for distributions == 'flat' string and sets values for
        that cell as 'flat', which is checked to provide an alpha of 0
        '''

        msd_distributions = {}
        log_distributions = {}
        for u in cell_ids:
            cell = cell_ids[u]
            if len(cell) > 30:
                max_tau = tau
            else:
                max_tau = len(cell)
            dist, log_dist = self.calc_msd_distribution(cell, max_tau)
            if dist == 'flat':
                msd_distributions[u] = 'flat'
                log_distributions[u] = 'flat'
            else:
                msd_distributions[u] = dist
                log_distributions[u] = log_dist

        return msd_distributions, log_distributions

    def calc_alphas(self, log_distributions):
        '''
        Calculate the alpha coefficient value as the slope of a
        log(MSD) v log(tau) plot
        Parameters
        ----------
        log_distributions : dict
            keyed by cell_id containing lists of log transformed MSDs,
            indexed by tau
        Returns
        -------
        alphas : dict
            keyed by cell_id, with values as the alpha coeff
            from log(MSD(tau)) = alpha*log(tau)
        Notes
        -----
        Checkes if log_distributions[cell_id] is == 'flat'
        If so, sets alpha at the appropriate "flat" slope of 0
        '''

        alphas = {}

        for u in log_distributions:
            if log_distributions[u] == 'flat':
                alphas[u] = 0
            else:
                tau = np.arange(1, len(log_distributions[u]) + 1 )
                log_tau = np.log(tau)

                slope, intercept, r_val, p_val, SE = stats.linregress( log_tau, log_distributions[u] )
                alphas[u] = slope

        return alphas

    def _plot_msd_dists(self, output_dir, msd_distributions, log_distributions):
        '''
        Plots tau vs MSD and log(tau) vs log(MSD)
        Saves as a PDF in the motility_statistics output folder
        '''

        tau = range(1,31)
        log_tau = []
        for val in tau:
            log_tau.append( math.log(val) )

        plt.figure(1)
        plt.subplot(121) # call subplots, 1 row, 2 columns, plot 1
        for u in msd_distributions:
            plt.scatter(tau, msd_distributions[u])
            plt.plot(tau, msd_distributions[u])
            plt.xlabel(r'$\tau$') # raw text string, latex to call tau char
            plt.ylabel('Mean Square Displacement')

        plt.subplot(122) # call subplots, 1 row, 2 columns, plot 2
        for u in msd_distributions:
            plt.scatter(log_tau, log_distributions[u])
            plt.plot(log_tau, log_distributions[u])
            plt.xlabel(r'log($\tau$)')
            plt.ylabel('log(Mean Square Displacement)')

        plt.subplots_adjust(wspace = 0.2) # increasing spacing b/w subplots

        plt.savefig(str(output_dir + 'MSD_Plots.pdf'))
        return

#%%
time_delay = 5 # time_delay for autocorrelation and partial autocorrelation
class RWFeatures(object):
    '''
    Calculates features relative to a random walk and self-similarity measures.
    Attributes
    ----------
    cell_ids : dict
        keyed by cell ids, values are lists of coordinate tuples.
    gf : hmsims.GeneralFeatures object.
    diff_linearity : dict
        difference in linearity r**2 between observed cell and a random walk.
        keyed by cell ids, values are floats.
    diff_net_dist : dict
        difference in net distance between an observed cell and a random walk.
        keyed by cell ids, values are floats.
    cell_kurtosis : dict
        measured displacement distribution kurtosis.
        keyed by cell ids, values are floats.
    diff_kurtosis : dict
        difference between `cell_kurtosis` and a random walk kurtosis.
        keyed by cell ids, values are floats.
    nongaussalpha : dict
        non-Gaussian parameters alpha_2 of the displacement distribution.
        keyed by cell ids, values are floats.
    disp_var : dict
        variance of the displacement distribution.
        keyed by cell ids, values are floats.
    hurst_RS : dict
        Hurst coefficients of the displacement time series, as estimated by
        Mandelbrot's original Rescaled Range method.
        keyed by cell ids, values are floats.
    autocorr_max_tau : int
        maximum tau for autocorrelation calculation.
    autocorr : dict
        autocorrelation coefficients for displacement time series.
        keyed by cell ids, values are floats.
    '''
    def __init__(self, cell_ids, gf):
        '''
        Calculates features relative to a random walk and self-similarity measures.
        Parameters
        ----------
        cell_ids : dict
            keyed by cell ids, values are lists of coordinate tuples.
        gf : hmsims.GeneralFeatures object.
        '''
        self.cell_ids = cell_ids
        self.gf = gf

        self.diff_linearity, self.diff_net_dist = self.run_comparative_randwalk(cell_ids, gf.linearity, gf.net_distance, gf.avg_speed)
        
        self.disp_kurtosis, _ = self.vartau_kurtosis_comparison(cell_ids, max_tau = 1)
        self.disp_variance, self.disp_skewness = self.displacement_props(cell_ids)
        self.nongaussalpha = self.nongauss_coeff(cell_ids)
        
        self.hurst_RS = self.hurst_mandelbrot(cell_ids)
        self.autocorr_max_tau = time_delay
        self.autocorr, _, _, self.partial_autocorr = self.autocorr_all(cell_ids, max_tau = self.autocorr_max_tau)

    def random_walk(self, origin, N, speed_mu, speed_sigma = None):
        '''
        Parameters
        ----------
        origin : starting point for the model
        N      : number of steps to model
        speed_mu : mean speed for the model
        speed_sigma : stdev of the normal distribution for step sizes
        Returns
        -------
        model_net_dist : float
        model_linearity : float
        model_path : list
        Notes
        -----
        Net distance of random walks is based on
        tau = time_unit
        rate = sigma = step_size = time_unit * velocity_x
        n = number of steps (can be exchanged for t if =)
        <x(t)^2> = n * sigma^2
        <x(t)^2>^0.5 = sigma * sqrt(n) = step_size * sqrt(n)
        <(x,y)^2> = <r^2> = <x(t)^2> + <y(t)^2> = 2 * n * sigma^2
        <r^2>^0.5 = sqrt(2) * step_size * sqrt(n)
        step_size = rate = c = sqrt(x_step^2 + y_step^2) = sqrt(2*step^2)
        x_step = y_step = rate/sqrt(2)
        (Random Walks in Biology, 1992, Howard C. Berg)
        '''
        model_net_dist = math.sqrt(2) * speed_mu * math.sqrt( N ) # speed_mu = 0~T time span 동안 cell 하나의 avg_speed
        model_path = [ origin ] # path의 초기 좌표
        if speed_sigma == None:
            speed_sigma = 0.2*speed_mu # gaussian distribution의 std = 0.2*mu로 둠
        rate = round( np.random.normal(speed_mu, speed_sigma), 3) # 속도를 N(mu, 0.2*mu)인 gauss distribution에서 pick
        vectors = [ (0, rate), (0, -rate), (rate, 0), (-rate, 0) ] 
        
        # Random Walk simulation: +/- x 방향, +/- y 방향 확률 각 25% 씩, N = number of total time frames (=49)
        i = 0
        while i < N:                 
            walk = np.random.random()
            if 0 <= walk < 0.25:
                step = vectors[0] # +y 방향
            elif 0.25 <= walk < 0.50:
                step = vectors[1] # -y 방향
            elif 0.50 <= walk < 0.75:
                step = vectors[2] # +x 방향
            else:
                step = vectors[3] # -x 방향

            new_x = model_path[-1][0] + step[0]
            new_y = model_path[-1][1] + step[1]
            model_path.append( (new_x, new_y) ) # model_path = [(origin), (new_x1, new_y1), (new_x2, new_y2), ...]
            i += 1


        x = []
        y = []
        for coor in model_path:
            x.append( coor[0] )
            y.append( coor[1] )

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            model_linearity = r_value**2
        except:
            return 'error', 'error', 'error'

        return model_net_dist, model_linearity, model_path

    def compare_to_randwalk(self, cell, u, cell_linearity, cell_net_dist, cell_avg_speed):
        '''
        Compares an observed path to a simulated random walk with the
        same displacement mean
        '''
        origin = cell[0]
        N = len(cell)
        rate = cell_avg_speed
        linearitys = []

        i = 0
        while i < 200: # Random_walk simulation을 200번 해보기
            model_net_dist, model_linearity, model_path = self.random_walk(origin, N, rate)
            if model_net_dist == 'error':
                pass
            else:
                linearitys.append( model_linearity )
            i += 1

        avg_linearity = np.mean(linearitys) # 200번 simulation한 Random walk path linearity의 average

        diff_linearity = cell_linearity - avg_linearity # 실제 path 와 random walk path의 linearity 비교
        diff_net_dist = cell_net_dist - model_net_dist # 실제 path 와 random walk path의 net_distance 비교

        return diff_linearity, diff_net_dist

    def run_comparative_randwalk(self, cell_ids, linearity, net_distance, avg_speed): # 바로 위 작업을 모든 세포에 대해 
        '''Runs comparisons to random walks for all cells in `cell_ids`'''
        diff_linearity = {}
        diff_net_dist = {}
        for u in cell_ids:
            cell = cell_ids[u]
            lin, net = self.compare_to_randwalk(cell, u, linearity[u], net_distance[u], avg_speed[u])
            diff_linearity[u] = lin
            diff_net_dist[u] = net

        return diff_linearity, diff_net_dist

    def make_displacement_dist(self, cell_ids, tau):
        '''Generates displacement distributions for all cells in `cell_ids`'''

        displacement_dist = {}
        for u in cell_ids:
            cell = cell_ids[u]
            displacement_dist[u] = []
            i = 0
            while (i + tau) < len(cell): # len(cell) = number of  time frames
                d = distance( cell[i + tau], cell[i] ) # d = cell[tau] ~ cell[0], cell[2tau] ~ cell[tau] , ...(tau=1)
                displacement_dist[u].append( d )
                i = i + tau

        return displacement_dist # time lag = tau 인 각 세포마다의 displacement distribution을 dict 형태로 담고 있음

    def kurtosis_comparison(self, cell_ids, displacements):
        '''Compares cell kurtosis to random walk kurtosis'''
        cell_kurtosis = {}
        diff_kurtosis = {}
        rayleigh_kurtosis = 3.245089
        for u in cell_ids:
            dist = displacements[u] # dist = 세포 1개의 displacement distribution 
            cell_kurtosis[u] = stats.kurtosis(dist) # 
            diff_kurtosis[u] = cell_kurtosis[u] # 결국 여기선 cell_kurtosis = diff_kurtosis

        return cell_kurtosis, diff_kurtosis

    def vartau_kurtosis_comparison(self, cell_ids, max_tau = 1 ):
        '''
        Calculate `cell_kurtosis` and `diff_kurtosis` for a range of possible
        time intervals `tau`.
        '''
        # set up dicts keyed by tau, containing dicts keyed by cell_id
        all_cell_kurtosis = {}
        all_diff_kurtosis = {}

        tau = 1
        while tau <= max_tau: 
            displacement_dist = self.make_displacement_dist(cell_ids, tau) 
            all_cell_kurtosis[tau-1], all_diff_kurtosis[tau-1] = self.kurtosis_comparison(cell_ids, displacements = displacement_dist)
            tau += 1

        return all_cell_kurtosis, all_diff_kurtosis # cell_kurtosis = diff_kurtosis

    def moving_kurt_comparison(self, cell_ids, displacements, speed):
        '''Compare kurtosis only while cell is moving'''
        cell_kurtosis = {}
        diff_kurtosis = {}
        rayleigh_kurtosis = 3.245089 # rayleigh distrubition 의 kurtosis = 3.245089임
        # rayleigh distrubition: 벡터 성분들이 normal distribution을 따를 때 벡터의 크기는 rayleigh distribution을 따름
        for u in cell_ids:
            dist = np.array(displacements[u])
            dist = dist[dist > speed]
            if len(dist) > 0:
                cell_kurtosis[u] = stats.kurtosis(dist)
                diff_kurtosis[u] = cell_kurtosis[u] - rayleigh_kurtosis
            else:
                cell_kurtosis[u] = 0
                diff_kurtosis[u] = 0

        return cell_kurtosis, diff_kurtosis

    def moving_kurt(self, cell_ids, max_tau = 10, moving_range = range_for_moving):
        '''
        Parameters
        ----------
        cell_ids : dict
            keyed by cell_id, values are lists of corrdinate tuples.
        max_tau : int
            maximum time lag to consider for kurtosis calculations.
        moving_range : iterable
            range of speeds to consider as a threshold for movement.
        Returns
        -------
        cell_moving_kurt : triple dict raw kurtosis of all cells in cell_ids
                           keyed by speed threshold, tau, and cell_id
                           cell_moving_kurt = {
                            speed1 : {
                                tau1 :
                                    {cell_id1 : kurt
                                    ...}
                                }
                                ...
                            } ...
                           }
        diff_moving_kurt : kurtosis of all cells, normalized by Rayleigh kurt
        structured as above
        '''

        cell_moving_kurt = {}
        diff_moving_kurt = {}

        for index, speed in enumerate(moving_range):
            cell_moving_kurt[speed] = {}
            diff_moving_kurt[speed] = {}
            tau = 1
            while tau <= max_tau:
                displacement_dist = self.make_displacement_dist(cell_ids, tau) # time lag = tau 인 각 세포마다의 displacement distribution을 dict 형태로 담고 있음
                cell_moving_kurt[index][tau], diff_moving_kurt[index][tau] = self.moving_kurt_comparison(cell_ids, displacements= displacement_dist, speed=speed)

        return cell_moving_kurt, diff_moving_kurt

    def displacement_props(self, cell_ids):
        '''
        Calculates variance and skewness of the displacement distribution
        for each cell in cell_ids
        Parameters
        ----------
        cell_ids : dict of lists containing tuples of sequential XY coordinates
        Returns
        -------
        var : dict keyed by cell_id with variance of displacement distribution
        skew : dict keyed by cell_ids with skew of displacement distribution
        '''
        var = {}
        skew = {}

        allX = self.make_displacement_dist(cell_ids=cell_ids, tau = 1) # time lag = tau 인 각 세포마다의 displacement distribution을 dict 형태로 담고 있음
        for u in allX: # u 는 하나의 세포 index
            X = np.asarray( allX[u] )
            var[u] = np.var(X)
            skew[u] = stats.skew(X)

        return var, skew

    def calc_ngaussalpha(self, X):
        '''
        Calculates the non-Gaussian parameter alpha_2 of a given displacement
        distribution, X
        Parameters
        ----------
        X : array-like
            distribution of displacements.
        Returns
        -------
        alpha_2 : float
            non-Gaussian coefficient, floating point values.
        Notes
        -----
        alpha_2 = <dx^4> / 3*<dx^2>^2 - 1
        effectively, a ratio of coeff* (kurtosis / variance)
        For a Gaussian distribution, alpha_2 = 0
        For non-Gaussian distributions, alpha_2 increases with the length
        of the tails
        Levy-like motion would be expected to have alpha_2 > 0, while diffusive
        motion would be expected to have alpha_2 == 0
        Rayleigh alpha_2 ~= -0.33
        '''

        X = np.asarray(X)
        alpha_2 = np.mean( X**4 ) / (3*np.mean(X**2)**2) - 1

        return alpha_2

    def nongauss_coeff(self, cell_ids):
        '''
        Calculates non-Gaussian coefficient alpha_2 for all cells in cell_ids.
        Parameters
        ----------
        cell_ids : dict
            lists containing tuples of sequential XY coordinates.
        Returns
        -------
        nongauss_coeff : dict
            keyed by cell_id, nonGauss coeff values.
        '''

        ngaussalpha = {}
        allX = self.make_displacement_dist(cell_ids=cell_ids, tau=1) # time lag = tau 인 각 세포마다의 displacement distribution을 dict 형태로 담고 있음
        for u in allX: # u 는 하나의 세포 index
            X = allX[u]
            ngaussalpha[u] = self.calc_ngaussalpha(X)

        return ngaussalpha


    def largest_pow2(self, num): 
        '''Finds argmax_n 2**n < `num`'''
        for i in range(0,10):
            if int(num / 2**i) > 1: # num (number of time frames = 49) > 2^i 이면 다음 i 시도
                continue
            else:
                return i-1 # num < 2^i 이 처음 되는 순간 (i = 6) i-1 (5)을 반환

    def rescaled_range(self, X, n):
        '''
        Finds rescaled range <R(n)/S(n)> for all sub-series of size `n` in `X`
        '''
        N = len(X) # X = time series displacement data of one cell, len(X) = number of time frames - 1 = 48
        if n > N:
            return None
        # Create subseries of size n
        num_subseries = int(N/n)
        Xs = np.zeros((num_subseries, n))
        for i in range(0, num_subseries):
            Xs[i,] = X[ int(i*n) : int(n+(i*n)) ]

        # Calculate mean rescaled range R/S
        # for subseries size n
        RS = []
        for subX in Xs:

            m = np.mean(subX)
            Y = subX - m
            Z = np.cumsum(Y)
            R = max(Z) - min(Z)
            S = np.std(subX)
            if S <= 0:
                print("S = ", S)
                continue
            RS.append( R/S )
        RSavg = np.mean(RS)

        return RSavg

    def hurst_mandelbrot(self, cell_ids):
        '''
        Calculates the Hurst coefficient `H` of displacement time series for each cell in `cell_ids`.
        Notes
        -----
        for E[R(n)/S(n)] = Cn**H as n --> inf
        H : 0.5 - 1 ; long-term positive autocorrelation
        H : 0.5 ; fractal Brownian motion
        H : 0-0.5 ; long-term negative autocorrelation
        N.B. SEQUENCES MUST BE >= 18 units long, otherwise
        linear regression for log(R/S) vs log(n) will have
        < 3 points and cannot be performed
        '''

        hurst_RS = {}
        allX = self.make_displacement_dist(cell_ids, tau = 1) # time lag = tau 인 각 세포마다의 displacement distribution을 담고 있음
        for u in allX:
            X = allX[u]
            RSl = []
            ns = []
            for i in range(0,self.largest_pow2(len(X))): # range(0,5)
                ns.append(int(len(X)/2**i))
            for n in ns:
                RSl.append( self.rescaled_range(X, n) )
            m, b, r, pval, sderr = stats.linregress(np.log(ns), np.log(RSl))
            hurst_RS[u] = m
        return hurst_RS

    # using Dominik Krzeminski's (@dokato) dfa library
    # see
    # https://github.com/dokato/
    # https://github.com/dokato/dfa/blob/master/dfa.py
    def dfa_all(self, cell_ids):
        '''
        Performs detrended fluctuation analysis on cell displacement series.
        Parameters
        ----------
        cell_ids : dict of lists keyed by cell_id
        ea. list represents a cell. lists contain sequential tuples
        containing XY coordinates of a cell at a given timepoint
        Returns
        -------
        dfa_alpha : dict keyed by cell_id, values are the alpha coefficient
        calculated by detrended fluctuation analysis
        References
        ----------
        Ping et. al., Mosaic organization of DNA nucleotides, 1994, Phys Rev E
        '''
        dfa_alpha = {}
        allX = self.make_displacement_dist(cell_ids, tau = 1) # time lag = tau 인 각 세포마다의 displacement distribution을 담고 있음
        for u in allX:
            X = allX[u]
            scales, fluct, coeff = dfa(X, scale_lim = [2,7], scale_dens = 0.25)
            dfa_alpha[u] = coeff
        return dfa_alpha

    def autocorr_all(self, cell_ids, max_tau = 10):
        '''
        Estimates the autocorrelation coefficient for each series of cell
        displacements over a range of time lags.
        Parameters
        ----------
        cell_ids : dict of lists keyed by cell_id
        ea. list represents a cell. lists contain sequential tuples
        containing XY coordinates of a cell at a given timepoint
        Returns
        -------
        autocorr : dict of lists, containing autocorrelation coeffs for
        sequential time lags
        qstats : dict of lists containing Q-Statistics (Ljung-Box)
        pvals : dict of lists containing p-vals, as calculated from Q-Statistics
        Notes
        -----
        Estimation method:
        https://en.wikipedia.org/wiki/Autocorrelation#Estimation
        R(tau) = 1/(n-tau)*sigma**2 [sum(X_t - mu)*(X_t+tau - mu)] | t = [1,n-tau]
        X as a time series, mu as the mean of X, sigma**2 as variance of X
        tau as a given time lag (sometimes referred to as k in literature)
        Implementation uses statsmodels.tsa.stattools.acf()
        n.b. truncated to taus [1,10], to expand to more time lags, simply
        alter the indexing being loaded into the return dicts
        '''
        autocorr = {}
        qstats = {}
        pvals = {}
        partial_autocorr = {}
        allX = self.make_displacement_dist(cell_ids, tau = 1) # time lag = tau 인 각 세포마다의 displacement distribution을 담고 있음
        for index in range(0, max_tau):
            autocorr_temp = {}
            qstats_temp = {}
            pvals_temp = {}
            partial_autocorr_temp = {}
            for u in allX:
                X = allX[u]
                # Perform Ljung-Box Q-statistic calculation to determine if autocorrelations detected are significant or random
                ac, q, p = acf(X, adjusted = True, nlags = (max_tau+1), qstat=True, fft = False) # time lag 0 ~ max_tau 까지 autocorrelation 계산
                pac = pacf(X, nlags = max_tau) # time lag 0 ~ max_tau 까지 partial autocorrelation 계산
                
                autocorr_temp[u] = ac[index+1] # time lag 0 일 때의 autocorrelation = 1이므로 제외
                qstats_temp[u] = q[index]
                pvals_temp[u] = p[index]
                partial_autocorr_temp[u] = pac[index+1] # time lag 0 일 때의 partial autocorrelation = 1이므로 제외
                
            autocorr[index] = autocorr_temp
            qstats[index] = qstats_temp
            pvals[index] = pvals_temp
            partial_autocorr[index] = partial_autocorr_temp

        return autocorr, qstats, pvals, partial_autocorr

#%%

gf = GeneralFeatures(position_dict, move_thresh = 1) # gf는 object, GeneralFeatures는 class

msdf = MSDFeatures(position_dict) # msdf는 object, MSDFeatures는 class

rwf = RWFeatures(position_dict, gf) # rwf는 object, RWFeatures는 class
#%%
list_of_all_speeds = list(np.array(list(gf.all_speed_values.values())).flat)
plt.hist(list_of_all_speeds, bins = 200)


#%%

gf_feature_list = ['total_distance', 'net_distance', 'min_speed', 'max_speed', 'avg_speed', 'time_moving', 
                      'avg_moving_speed', 'linearity', 'spearmanrsq', 'progressivity', 
                   'turn_stats', 'average_theta', 'min_theta', 'max_theta']

msdf_feature_list = ['alphas']

rwf_feature_list = ['diff_linearity', 'diff_net_dist', 'disp_kurtosis', 'disp_skewness', 'disp_variance', 
                    'hurst_RS', 'nongaussalpha', 'autocorr', 'partial_autocorr']

df_motility = pd.DataFrame()
df_motility = pd.concat([df_motility, label_fulltime_data['image_stack'], label_fulltime_data['cell_label'],label_fulltime_data['condition']], axis = 1) 
########## general motility features ##########
for name_idx in range(0, len(gf_feature_list)):
    if len(getattr(gf, gf_feature_list[name_idx])) == df_position.shape[0]:
        df_motility_temp = pd.DataFrame(list(getattr(gf, gf_feature_list[name_idx]).values()), columns=[gf_feature_list[name_idx]])
        df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)
    
    else: 
        for i in range(0,len(getattr(gf, gf_feature_list[name_idx]))):
            df_motility_temp = pd.DataFrame(list(getattr(gf, gf_feature_list[name_idx])[i].values()), columns=[gf_feature_list[name_idx]+'_%s'% str(i)])
            df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)

########## Mean Squre Displacement features ##########
for name_idx in range(0, len(msdf_feature_list)):
    if len(getattr(msdf, msdf_feature_list[name_idx])) == df_position.shape[0]:
        df_motility_temp = pd.DataFrame(list(getattr(msdf, msdf_feature_list[name_idx]).values()), columns=[msdf_feature_list[name_idx]])
        df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)
    
    else: 
        for i in range(0,len(getattr(msdf, msdf_feature_list[name_idx]))):
            df_motility_temp = pd.DataFrame(list(getattr(msdf, msdf_feature_list[name_idx])[i].values()), columns=[msdf_feature_list[name_idx]+'_%s'% str(i)])
            df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)
            
########## Random Walk features ##########            
for name_idx in range(0, len(rwf_feature_list)):
    if len(getattr(rwf, rwf_feature_list[name_idx])) == df_position.shape[0]:
        df_motility_temp = pd.DataFrame(list(getattr(rwf, rwf_feature_list[name_idx]).values()), columns=[rwf_feature_list[name_idx]])
        df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)
    
    else: 
        for i in range(0,len(getattr(rwf, rwf_feature_list[name_idx]))):
            df_motility_temp = pd.DataFrame(list(getattr(rwf, rwf_feature_list[name_idx])[i].values()), columns=[rwf_feature_list[name_idx]+'_%s'% str(i)])
            df_motility = pd.concat([df_motility, df_motility_temp], axis = 1)

df_motility

df_motility['alphas']

df_motility.groupby('condition').describe()['avg_speed']
df_motility.groupby('condition').describe()['time_moving_0']

df_motility.groupby('condition').describe()['progressivity']


fig = px.box(
data_frame = df_motility,
x = 'condition',
y = 'time_moving_0',
color = 'condition',
points= 'all',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
#title = 'Clone 1-1 Morphology Space',
labels = {'condition':'condition',
          'avg_speed':'average speed (μm/h)',
          },
hover_data = {'condition': True, 'cell_label':True},
hover_name = df_motility.index,

height = 500,
width = 700,
)

fig.update_traces(marker = dict(size = 2),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

pio.show(fig)

#%%
fig = px.box(
data_frame = df_motility,
x = 'condition',
y = 'progressivity',
color = 'condition',
points= 'all',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
#title = 'Clone 1-1 Morphology Space',
labels = {'condition':'condition',
          'progressivity':'persistence',
          },
hover_data = {'condition': True, 'cell_label':True},
hover_name = df_motility.index,

height = 500,
width = 700,
)

fig.update_traces(marker = dict(size = 2),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

pio.show(fig)



#%%
""""Motility space"""
df_motility_input = df_motility.drop(['condition', 'image_stack', 'cell_label', 'partial_autocorr_0', 'total_distance'],axis=1)
# time lag 1 일 때의 partial autocorrelation = time lag 1 일 때의 autocorrelation 이므로 partial_autocoor_0 제외
# total_distance/24h = avg_speed라서 제외(correlation = 1)
df_motility_input

df_title = pd.DataFrame(df_motility_input.T.index)
df_title.columns = ['parameter']
df_title

scaler = StandardScaler()
X_mot = scaler.fit_transform(df_motility_input)
pca_mot = PCA(n_components = df_motility_input.shape[1], svd_solver= 'full')
Y_mot = pca_mot.fit_transform(X_mot)
df_raw_pca_mot = pd.DataFrame(Y_mot)

col_pc_mot = []
for i in range(1, df_raw_pca_mot.shape[1]+1):
    col_pc_mot.append("PC%d"%i)
    
df_raw_pca_mot.columns = col_pc_mot
df_pca_mot = pd.concat([df_motility['image_stack'], df_motility['cell_label'],df_motility['condition'], df_raw_pca_mot], axis = 1)



variance_ratio_mot = pca_mot.explained_variance_ratio_ #Percentage of variance explained by each of the selected components.
variance_mot = pca_mot.explained_variance_

cum_sum_variance_ratio_mot = np.cumsum(variance_ratio_mot)

plt.bar(range(0,len(variance_ratio_mot)), variance_ratio_mot, alpha=0.5, align='center', label='Individual explained variance', color ='g')
plt.step(range(0,len(cum_sum_variance_ratio_mot)), cum_sum_variance_ratio_mot, where='mid',label='Cumulative explained variance',color ='g')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Explained variance ratio', fontsize=15)
plt.xlabel('Principal component index', fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#95%설명하는 PC번호 찾기
for i in range(1, len(cum_sum_variance_ratio_mot)+1):
    if (cum_sum_variance_ratio_mot[i-1] >= 0.95).min():
        print("PC%d"%i)
        PC_95percent_mot = i-1
        break



df_loadings_mot = pd.DataFrame(pca_mot.components_.T * np.sqrt(pca_mot.explained_variance_))
df_loadings_mot.columns = col_pc_mot
df_loadings_mot = pd.concat([df_title, df_loadings_mot], axis =1)


#Top PC 에서 가장 연관성이 높은 morhplogical features 추출.
top_pc = 5
top_parameter = []
for i in range(1,top_pc+1):
    top_parameter.append(df_loadings_mot['PC%d'%i].abs().idxmax()) #절대값 취해서 음의 상관관계도 나오게
    # top_parameter.append(df_loadings_mot['PC%d'%i].idxmax()) #양의 상관관계만

df_loadings_top_pc=pd.DataFrame()
for i in top_parameter:
    loadings_top_pc = df_loadings_mot.iloc[i,0:top_pc+1].to_frame()
    df_loadings_top_pc=pd.concat([df_loadings_top_pc, loadings_top_pc], axis=1)

df_loadings_top_pc = df_loadings_top_pc.T
df_loadings_top_pc = df_loadings_top_pc.set_index('parameter')
df_loadings_top_pc.index.name = None
#folling lines to ensure the sns.heatmap works   
df_loadings_top_pc.astype(float) 
df_loadings_top_pc.fillna(value=np.nan, inplace=True)

# corr_df_loadings = df_loadings.corr(method="spearman")

plt.figure(figsize=(15,12))
heatmap = sns.heatmap(df_loadings_top_pc, annot = True, annot_kws={"size": 25}, cmap = 'coolwarm')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=30, rotation =0)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=30)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)
# plt.xlabel('Morphology cluster', fontsize = 20, labelpad=10)
# plt.ylabel('Condition', fontsize = 20, labelpad=10)
# plt.title('Distribtion of morphology custer', fontsize = 20)
plt.show()

# plt.scatter(Y_mot[:,0],Y_mot[:,1],color='blue',edgecolor='k')
# plt.xlabel('PC1',fontsize=16)
# plt.ylabel('PC2',fontsize=16)




#%%

##################################################################
fig = px.scatter(
data_frame = df_pca_mot,
x = 'PC1',
y = 'PC2',
color = 'condition',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Motility Space',
labels = {'PC1':'PC1('+str(round(variance_ratio_mot[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio_mot[1]*100,ndigits=1))+'%)',
          },
hover_data = {'cell_label': True, 'condition': True},
hover_name = df_pca_mot.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

for index, value in df_loadings_mot.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = 3*value['PC1'], y1 = 3*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = 3*value['PC1'], y = 3*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                       text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

pio.show(fig)
fig.write_html(directory + 'motility_space.html')


#%%


'''Kmeans'''
k_range = range(2,20)
sum_squared_error = [] # sum_squared_error object를 list로 정의
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df_pca_mot[['PC1','PC2']])
    sum_squared_error.append(km.inertia_) # inertia_ 자체가 sum of squared error 계산식을 포함

plt.xlabel('K')
plt.xticks(np.arange(1,df_motility_input.shape[1]+1),fontsize=7)
plt.ylabel('Sum of squared error')
plt.plot(k_range,sum_squared_error)
from kneed import KneeLocator # conda install -c conda-forge kneed
kl = KneeLocator(range(2,20),sum_squared_error,curve='convex',direction='decreasing')
kl.elbow # find point of maximum curvature


from sklearn.metrics import silhouette_score
silhouette_coefficeints =[]
for k in range(2,20):
    km = KMeans(n_clusters=k)
    km.fit(df_pca_mot[['PC1','PC2']])
    score = silhouette_score(df_pca_mot[['PC1','PC2']],km.labels_)
    silhouette_coefficeints.append(score)

plt.xlabel('K')
plt.xticks(np.arange(1,20),fontsize=7)
plt.ylabel('Sillhouette Coefficient')
plt.plot(range(2,20),silhouette_coefficeints)
# choose the maximum value

#%%
number_of_clusters = 3
km = KMeans(n_clusters = number_of_clusters,random_state=0)
kmeans_predicted = km.fit_predict(df_pca_mot[['PC1','PC2']])# fit하고 동시에 predict하는것
df_pca_mot['kmeans_cluster'] = kmeans_predicted
df_pca_mot

new_df_mot = df_pca_mot['kmeans_cluster'].replace({0:'Group 0', 1:'Group 1', 2:'Group 2',
                                                       3:'Group 3', 4:'Group 4', 5:'Group 5',
                                                       6:'Group 6', 7:'Group 7', 8:'Group 8',
                                                       9:'Group 9', 10:'Group 10', 11:'Group 11',
                                                       12:'Group 12', 13:'Group 13', 14:'Group 14',
                                                       15:'Group 15', 16:'Group 16', 17:'Group 17',
                                                         })
new_df_mot = pd.concat([df_pca_mot.drop(['kmeans_cluster'],axis=1),new_df_mot],axis=1) 
new_df_mot


#############################
fig = px.scatter(
data_frame = new_df_mot,
x = 'PC1',
y = 'PC2',
color = 'kmeans_cluster',
color_discrete_sequence = px.colors.qualitative.Set1, # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Motility Space',
labels = {'PC1':'PC1('+str(round(variance_ratio_mot[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio_mot[1]*100,ndigits=1))+'%)',
          },
hover_data = {'cell_label': True,'condition':True},
hover_name = df_pca_mot.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )
constant = 3
for index, value in df_loadings_mot.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = constant*value['PC1'], y1 = constant*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = constant*value['PC1'], y = constant*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                      text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

fig.add_scatter(x = km.cluster_centers_[:,0], y = km.cluster_centers_[:,1], 
                mode = 'markers', marker= dict(color = 'black', size = 10))


pio.show(fig)
fig.write_html(directory + 'motility_kmeans.html')


plt.figure(figsize=(20,15))
plt.scatter(df_pca_mot['PC1'],df_pca_mot['PC2'],marker=',',s=40, 
                c=df_pca_mot['kmeans_cluster'],
            cmap = plt.cm.get_cmap('Set1'),
            #color =['red','green','blue','purple']
           )

#%%
'''by sample type'''
plt.figure(figsize=(20,15))
colors = itertools.cycle(["green", "royalblue", "indianred"])
groups = df_pca_mot.groupby('condition')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o',alpha=.7, linestyle='', markersize=20, label=name, color=next(colors))
# plt.legend(loc='best', fontsize=20)
plt.xlim(-9, 9)
plt.ylim(-7, 14)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('PC1('+str(round(variance_ratio_mot[0]*100,ndigits=1))+'%)', fontsize=40)
# plt.ylabel('PC2('+str(round(variance_ratio_mot[1]*100,ndigits=1))+'%)', fontsize=40)
plt.savefig('motility_space.png', dpi=600)
#%%


plt.figure(figsize=(20,15))
colors = itertools.cycle(["navy", "turquoise", "khaki"])

groups = df_pca_mot.groupby('kmeans_cluster')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o', linestyle='', alpha=.7, markersize=20, label=name, color=next(colors))
# plt.legend(loc='best',prop={'size': 20})
plt.xlim(-13, 15)
plt.xlim(-9, 9)
plt.ylim(-7, 14)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('PC1('+str(round(variance_ratio_mot[0]*100,ndigits=1))+'%)', fontsize=40)
# plt.ylabel('PC2('+str(round(variance_ratio_mot[1]*100,ndigits=1))+'%)', fontsize=40)
plt.savefig('motility_space_kmeans.png', dpi=600)


#%%
"""Distribution of motility clusters"""
cluster_type = 'kmeans_cluster'
group_condition_mot = pd.DataFrame(df_pca_mot.groupby(['condition',cluster_type]).size())
group_condition_mot['distribution(%)'] = group_condition_mot.groupby(level=0).apply(lambda x:  100*x / x.sum())
group_condition_mot = group_condition_mot.drop([0],axis=1)
group_condition_mot = group_condition_mot.unstack(level=0)
group_condition_mot = group_condition_mot.fillna(0)
group_condition_mot = group_condition_mot[[('distribution(%)','control'), ('distribution(%)','with7'), ('distribution(%)','with231')]]


plt.figure(figsize=(15,10))
heatmap = sns.heatmap(group_condition_mot.T, annot = True, annot_kws={"size": 20}, cmap = 'YlGnBu')
# plt.xlabel('Motility cluster')
# plt.ylabel('condition')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=15)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=25)
# plt.xlabel('Motility cluster', fontsize = 30, labelpad=10)
# plt.ylabel('Condition', fontsize = 30, labelpad=10)
# plt.title('Distribution of motility cluster', fontsize = 20)
plt.savefig('heatmap_motility.eps', format='eps', dpi=600)

plt.show()





'''# Shannon entropy'''
shannon_entropy_list = []
for condition_name in list(group_condition_mot['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, group_condition_mot['distribution(%)'][condition_name].shape[0]):
        shannon_entropy = shannon_entropy + -group_condition_mot['distribution(%)'][condition_name][i]/100*np.log2(group_condition_mot['distribution(%)'][condition_name][i]/100)
    shannon_entropy_list.append(shannon_entropy)
    print(condition_name)
shannon_entropy_list
plt.figure(figsize=(8,8))
colors = ["green", "royalblue", "indianred"]
# plt.title('Shannon entropy', fontsize = 20)
plt.bar(np.arange(len(shannon_entropy_list)), shannon_entropy_list, color=colors, width = 0.4,edgecolor = "black", linewidth=3)
# plt.xticks(np.arange(len(shannon_entropy_list)), tskm_condition['distribution(%)'].columns.tolist(),fontsize=25)
plt.yticks(fontsize=20)
plt.ylim(1.3, 1.6)
plt.xticks([], [])
plt.savefig('Shannon_entropy_motility.eps', format='eps', dpi=600)
plt.show()



""""Hierarchical clustering"""
transposed_mot = group_condition_mot.T
transposed_mot
# Centroid : 두 군집의 중심점(centroid)를 정의한 다음 두 중심점의 거리를 군집간의 거리로 측정
# Single : 최단 연결법, 두 군집에 있는 모든 데이터 조합에서 데이터 사이 거리를 측정해서 가장 최소 거리(작은 값)를 기준으로 군집 거리를 측정
# Complete : 최장 연결법으로 두 클러스터상에서 가장 먼 거리를 이용해서 측정하는 방식
# Average : 평균 연결법, 두 군집의 데이터들 간 모든 거리들의 평균을 군집간 거리로 정의
# Ward : 연결될 수 있는 군집 조합을 만들고, 군집 내 편차들의 Sum of squared error을 기준으로 최소 제곱합을 가지게 되는 군집끼리 연결

hl_mot = hierarchy.linkage(transposed_mot,method='average',metric='euclidean')

plt.figure(figsize=(2,10))
dendrogram = hierarchy.dendrogram(hl_mot, orientation = 'left',labels = transposed.index)

hc_mot = AgglomerativeClustering(n_clusters=2,linkage='ward') 
# affinity = 'euclidan'으로 하면 ward method 자체가 euclidian base라는 이유로 에러가 발생함
y_predicted = hc_mot.fit_predict(transposed_mot)
transposed_mot['hc_cluster'] = y_predicted
transposed_mot


''''Cell trajecory'''
#클러스터별로 나오도록
plt.figure(figsize=(20,20))
for i in range(0,9):
    cell_idx = np.random.randint(0,label_fulltime_data.shape[0])
    position_map_x = []
    position_map_y= []
    for x, y in position_dict[cell_idx]:
        position_map_x.append(x-position_dict[cell_idx][0][0])
        position_map_y.append(y-position_dict[cell_idx][0][1])
    plt.subplot(3,3,i+1)
    plt.plot(position_map_x, position_map_y, '-',
             color= 'black', linewidth = 2,
            )
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.axvline(0, color='black',linewidth = 1)
    plt.axhline(0, color='black',linewidth = 1)
    plt.title(str(cell_idx)+ '   ' + str(label_fulltime_data[0][cell_idx][0]) + '  ' +'cluster '+str(df_pca_mot['kmeans_cluster'][cell_idx]), pad=20)
#position_map = position_dict[cell_idx] - position_dict[cell_idx][0]



#%%
'''Morphodynamics - motility commposite state space'''


'''Morphodynamics features with 95% PC coverage'''
#for full-time cells with 95%PC
label_fulltime_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()
df_time_series = pd.DataFrame()
for i in range(0, label_fulltime_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_fulltime_data[0][i][0],label_fulltime_data[0][i][1])).reset_index()
    
    cell_data = cell_data.drop('index', axis = 1)
    pc_list =  [] # pc_list = ['PC1', 'PC2', 'PC3', ...]
    for i in range(0, PC_95percent+1): # PC1, PC2, PC3, ...
        pc_list.append(cell_data.columns[i+4]) #index drop 하고 하면 cell_data.columns[4] 부터 PC1 시작
    
    cell_data['data'] = cell_data[pc_list].apply(tuple,axis=1) #95% PC까지 tuple로 저장
    df_time_series_temp = cell_data[['time_point','data']].T
    df_time_series_temp.columns = df_time_series_temp.iloc[0]
    df_time_series_temp = df_time_series_temp.drop('time_point')
    df_time_series = pd.concat([df_time_series, df_time_series_temp], axis = 0)

df_time_series.reset_index(inplace=True)
df_time_series = df_time_series.drop(['index'], axis = 1)
df_time_series


'''Morphodynamics features with 95% PC coverage'''
mean_list = []
for cell_idx in range(0, df_time_series.shape[0]): # for one cell (473 cells in total)
    mean_list_onecell = []
    for pc_num in range(0, PC_95percent+1): # for one PC component (11 components in total)
        temp_list = []
        for time_point in range(0, df_time_series.columns.shape[0]): # for one time point (49 time points in total)
            temp_list.append(df_time_series[df_time_series.columns[time_point]][cell_idx][pc_num])
        mean_list_onecell.append(mean(temp_list))
    mean_list.append(mean_list_onecell)
mean_list = np.array(mean_list)
mean_list.shape # (cell idx, number of PCs)


morphology_features = pd.DataFrame()
for i in range(0, mean_list.shape[1]):
    df_temp = pd.DataFrame(mean_list[:,i], columns=['Mean_morphology_PC%s' % str(i+1)])
    morphology_features = pd.concat([morphology_features, df_temp], axis = 1)
morphology_features


'''Motility features with 95% PC coverage'''

motility_features = pd.DataFrame()
for i in range(0, PC_95percent_mot+1):
    
    df_temp = pd.DataFrame(df_pca_mot.iloc[:,i+3].to_list(), columns=['Motility_PC%s' % str(i+1)])

    # df_temp = df_pca_mot.iloc[:,i+3]
    # df_temp.columns=['motility_PC%s' % str(i+1)]
    motility_features = pd.concat([motility_features, df_temp], axis = 1)
motility_features

'''Motility-morphodynamics composite PCA with 95% PC coverage'''

composite_features = pd.concat([morphology_features, motility_features], axis = 1)
composite_features




#%%

df_composite_features = pd.concat([df_motility['image_stack'], df_motility['cell_label'],df_motility['condition'], composite_features], axis = 1)

df_composite_features


sns.pairplot(df_composite_features, hue='condition',
             vars=["Mean_morphology_PC1","Mean_morphology_PC2","Motility_PC1", "Motility_PC2"])
             





#%%
df_title = pd.DataFrame(composite_features.T.index)
df_title.columns = ['parameter']
df_title

scaler = StandardScaler()
X_composite = scaler.fit_transform(composite_features)
pca_composite = PCA(n_components = composite_features.shape[1], svd_solver= 'full')
Y_composite = pca_composite.fit_transform(X_composite)
df_raw_pca_composite = pd.DataFrame(Y_composite)

col_pc_composite = []
for i in range(1, df_raw_pca_composite.shape[1]+1):
    col_pc_composite.append("PC%d"%i)
    
df_raw_pca_composite.columns = col_pc_composite
df_pca_composite = pd.concat([df_motility['image_stack'], df_motility['cell_label'],df_motility['condition'], df_raw_pca_composite], axis = 1)



variance_ratio_composite = pca_composite.explained_variance_ratio_ #Percentage of variance explained by each of the selected components.
variance_composite = pca_composite.explained_variance_

cum_sum_variance_ratio_composite = np.cumsum(variance_ratio_composite)

plt.bar(range(0,len(variance_ratio_composite)), variance_ratio_composite, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_variance_ratio_composite)), cum_sum_variance_ratio_composite, where='mid',label='Cumulative explained variance')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Explained variance ratio', fontsize=15)
plt.xlabel('Principal component index', fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


df_loadings_composite = pd.DataFrame(pca_composite.components_.T * np.sqrt(pca_composite.explained_variance_))
df_loadings_composite.columns = col_pc_composite
df_loadings_composite = pd.concat([df_title, df_loadings_composite], axis =1)




#%%

##################################################################
fig = px.scatter(
data_frame = df_pca_composite,
x = 'PC1',
y = 'PC2',
color = 'condition',
#color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Morphodynamics-motility composite Space',
labels = {'PC1':'PC1('+str(round(variance_ratio_mot[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio_mot[1]*100,ndigits=1))+'%)',
          },
hover_data = {'cell_label': True, 'condition': True},
hover_name = df_pca_mot.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )

for index, value in df_loadings_composite.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = 3*value['PC1'], y1 = 3*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = 3*value['PC1'], y = 3*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                       text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

pio.show(fig)
fig.write_html(directory + 'motility_space.html')


#%%


'''Kmeans'''
k_range = range(2,20)
sum_squared_error = [] # sum_squared_error object를 list로 정의
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df_pca_composite[['PC1','PC2']])
    sum_squared_error.append(km.inertia_) # inertia_ 자체가 sum of squared error 계산식을 포함

plt.xlabel('K')
plt.xticks(np.arange(1,composite_features.shape[1]+1),fontsize=7)
plt.ylabel('Sum of squared error')
plt.plot(k_range,sum_squared_error)
from kneed import KneeLocator # conda install -c conda-forge kneed
kl = KneeLocator(range(2,20),sum_squared_error,curve='convex',direction='decreasing')
kl.elbow # find point of maximum curvature


from sklearn.metrics import silhouette_score
silhouette_coefficeints =[]
for k in range(2,20):
    km = KMeans(n_clusters=k)
    km.fit(df_pca_composite[['PC1','PC2']])
    score = silhouette_score(df_pca_composite[['PC1','PC2']],km.labels_)
    silhouette_coefficeints.append(score)

plt.xlabel('K')
plt.xticks(np.arange(1,20),fontsize=7)
plt.ylabel('Sillhouette Coefficient')
plt.plot(range(2,20),silhouette_coefficeints)
# choose the maximum value


number_of_clusters = 3
km = KMeans(n_clusters = number_of_clusters,random_state=0)
kmeans_predicted = km.fit_predict(df_pca_composite[['PC1','PC2']])# fit하고 동시에 predict하는것
df_pca_composite['kmeans_cluster'] = kmeans_predicted
df_pca_composite

new_df_composite = df_pca_composite['kmeans_cluster'].replace({0:'Group 0', 1:'Group 1', 2:'Group 2',
                                                       3:'Group 3', 4:'Group 4', 5:'Group 5',
                                                       6:'Group 6', 7:'Group 7', 8:'Group 8',
                                                       9:'Group 9', 10:'Group 10', 11:'Group 11',
                                                       12:'Group 12', 13:'Group 13', 14:'Group 14',
                                                       15:'Group 15', 16:'Group 16', 17:'Group 17',
                                                         })
new_df_composite = pd.concat([df_pca_composite.drop(['kmeans_cluster'],axis=1),new_df_composite],axis=1) 
new_df_composite


#############################
fig = px.scatter(
data_frame = new_df_composite,
x = 'PC1',
y = 'PC2',
color = 'kmeans_cluster',
color_discrete_sequence = px.colors.qualitative.Set1, # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
opacity = 0.9,
template = 'plotly_white', # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
#symbol = 'label',
#symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
title = 'Motility Space',
labels = {'PC1':'PC1('+str(round(variance_ratio_composite[0]*100,ndigits=1))+'%)',
          'PC2':'PC2('+str(round(variance_ratio_composite[1]*100,ndigits=1))+'%)',
          },
hover_data = {'cell_label': True,'condition':True},
hover_name = df_pca_composite.index,

height = 1000,
width = 2000,
)

fig.update_traces(marker = dict(size = 10),
                                #line = dict(width=1, color='DarkSlateGrey')) ,
                  #selector=dict(mode='markers')
                 )
constant = 3
for index, value in df_loadings_composite.iterrows():
    fig.add_shape(type='line', x0 = 0, y0 = 0, x1 = constant*value['PC1'], y1 = constant*value['PC2'], opacity = 0.7,
                  line = dict(color='black', width = 1, dash = 'dot'))
    fig.add_annotation(x = constant*value['PC1'], y = constant*value['PC2'], ax = 0, ay = 0, xanchor='center',yanchor='bottom',
                      text = value['parameter'], font = dict(size = 15, color = 'black'), opacity = 0.7)

fig.add_scatter(x = km.cluster_centers_[:,0], y = km.cluster_centers_[:,1], 
                mode = 'markers', marker= dict(color = 'black', size = 10))


pio.show(fig)
fig.write_html(directory + 'morphodynamics-motility_kmeans.html')


plt.figure(figsize=(20,15))
plt.scatter(df_pca_composite['PC1'],df_pca_composite['PC2'],marker=',',s=40, 
                c=df_pca_composite['kmeans_cluster'],
            cmap = plt.cm.get_cmap('Set1'),
            #color =['red','green','blue','purple']
           )


#%%


plt.figure(figsize=(20,15))
groups = df_pca_composite.groupby('kmeans_cluster')
for name, group in groups:
    plt.plot(group.PC1, group.PC2, marker='o', linestyle='', markersize=7, label=name)
plt.legend(loc='best',prop={'size': 20})
plt.xlim(-13, 15)
plt.xlabel('PC1('+str(round(variance_ratio_composite[0]*100,ndigits=1))+'%)', fontsize=20)
plt.ylabel('PC2('+str(round(variance_ratio_composite[1]*100,ndigits=1))+'%)', fontsize=20)



#%%
"""Distribution of morphodynamics-motility clusters"""
cluster_type = 'kmeans_cluster'
group_condition_composite = pd.DataFrame(df_pca_composite.groupby(['condition',cluster_type]).size())
group_condition_composite['distribution(%)'] = group_condition_composite.groupby(level=0).apply(lambda x:  100*x / x.sum())
group_condition_composite = group_condition_composite.drop([0],axis=1)
group_condition_composite = group_condition_composite.unstack(level=0)
group_condition_composite = group_condition_composite.fillna(0)
group_condition_composite = group_condition_composite[[('distribution(%)','control'), ('distribution(%)','with7'), ('distribution(%)','with231')]]


plt.figure(figsize=(15,10))
heatmap = sns.heatmap(group_condition_composite.T, annot = True, annot_kws={"size": 15}, cmap = 'BuGn')
# plt.xlabel('Morphodynamics-motility cluster')
# plt.ylabel('condition')
heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=15)
heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=15)
# plt.xlabel('Morphodynamics-motility cluster', fontsize = 20, labelpad=10)
# plt.ylabel('Condition', fontsize = 20, labelpad=10)
# plt.title('Distribution of morphodynamics-motility cluster cluster', fontsize = 20)
plt.savefig('heatmap_composite.eps', format =eps,  dpi=600)
plt.show()





'''# Shannon entropy'''
shannon_entropy_list = []
for condition_name in list(group_condition_composite['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, group_condition_composite['distribution(%)'][condition_name].shape[0]):
        shannon_entropy = shannon_entropy + -group_condition_composite['distribution(%)'][condition_name][i]/100*np.log2(group_condition_composite['distribution(%)'][condition_name][i]/100)
    shannon_entropy_list.append(shannon_entropy)
    print(condition_name)
shannon_entropy_list
plt.figure(figsize=(8,8))
colors = ['black', 'grey', 'red']
plt.title('Shannon entropy', fontsize = 20)
plt.bar(np.arange(len(shannon_entropy_list)), shannon_entropy_list, color=colors, width = 0.4)
plt.xticks(np.arange(len(shannon_entropy_list)), group_condition_composite['distribution(%)'].columns.tolist(),fontsize=15)
# plt.ylim(1.1, 1.6)t
plt.show()



""""Hierarchical clustering"""
transposed_composite = group_condition_composite.T
transposed_composite
# Centroid : 두 군집의 중심점(centroid)를 정의한 다음 두 중심점의 거리를 군집간의 거리로 측정
# Single : 최단 연결법, 두 군집에 있는 모든 데이터 조합에서 데이터 사이 거리를 측정해서 가장 최소 거리(작은 값)를 기준으로 군집 거리를 측정
# Complete : 최장 연결법으로 두 클러스터상에서 가장 먼 거리를 이용해서 측정하는 방식
# Average : 평균 연결법, 두 군집의 데이터들 간 모든 거리들의 평균을 군집간 거리로 정의
# Ward : 연결될 수 있는 군집 조합을 만들고, 군집 내 편차들의 Sum of squared error을 기준으로 최소 제곱합을 가지게 되는 군집끼리 연결

hl_composite = hierarchy.linkage(transposed_composite,method='average',metric='euclidean')

plt.figure(figsize=(2,10))
dendrogram = hierarchy.dendrogram(hl_composite, orientation = 'left',labels = transposed.index)

hc_composite = AgglomerativeClustering(n_clusters=2,linkage='ward') 
# affinity = 'euclidan'으로 하면 ward method 자체가 euclidian base라는 이유로 에러가 발생함
y_predicted = hc_composite.fit_predict(transposed_composite)
transposed_mot['hc_cluster'] = y_predicted
transposed_mot

#%%
'''dimensionality reduction of morphodynamics-motility features using UMAP'''
import umap

composite_features


reducer = umap.UMAP(random_state=3) 
# reducer = umap.UMAP()

scaler = StandardScaler()
X_composite = scaler.fit_transform(composite_features)
embedding = reducer.fit_transform(X_composite)
embedding.shape

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],)



col_umap_composite = []
for i in range(1, embedding.shape[1]+1):
    col_umap_composite.append("UMAP%d"%i)

df_raw_umap_composite = pd.DataFrame(embedding)    
df_raw_umap_composite.columns = col_umap_composite
df_umap_composite = pd.concat([df_motility['image_stack'], df_motility['cell_label'],df_motility['condition'], df_raw_umap_composite], axis = 1)


"""
'''Kmeans'''
'''Elbow method'''
k_range = range(2,20)
sum_squared_error = [] # sum_squared_error object를 list로 정의
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df_umap_composite[['UMAP1','UMAP2']])
    sum_squared_error.append(km.inertia_) # inertia_ 자체가 sum of squared error 계산식을 포함

plt.xlabel('K')
plt.xticks(np.arange(1,composite_features.shape[1]+1),fontsize=7)
plt.ylabel('Sum of squared error')
plt.plot(k_range,sum_squared_error)
from kneed import KneeLocator # conda install -c conda-forge kneed
kl = KneeLocator(range(2,20),sum_squared_error,curve='convex',direction='decreasing')
kl.elbow # find point of maximum curvature

'''Silhouette coefficient'''

from sklearn.metrics import silhouette_score
silhouette_coefficeints =[]
for k in range(2,20):
    km = KMeans(n_clusters=k)
    km.fit(df_umap_composite[['UMAP1','UMAP2']])
    score = silhouette_score(df_umap_composite[['UMAP1','UMAP2']],km.labels_)
    silhouette_coefficeints.append(score)

plt.xlabel('K')
plt.xticks(np.arange(1,20),fontsize=7)
plt.ylabel('Sillhouette Coefficient')
plt.plot(range(2,20),silhouette_coefficeints)
# choose the maximum value
"""

number_of_clusters = 3
km = KMeans(n_clusters = number_of_clusters,random_state=0)
kmeans_predicted = km.fit_predict(df_umap_composite[['UMAP1','UMAP2']])# fit하고 동시에 predict하는것
df_umap_composite['kmeans_cluster'] = kmeans_predicted
df_umap_composite


'''by sample type'''
plt.figure(figsize=(20,15))
colors = itertools.cycle(["green", "royalblue", "indianred"])
groups = df_umap_composite.groupby('condition')
for name, group in groups:
    plt.plot(group.UMAP1, group.UMAP2, marker='o', linestyle='', alpha=.7, markersize=20, label=name, color=next(colors))
# plt.legend(loc='best', fontsize=20)
# plt.xlim(-13, 15)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('UMAP1', fontsize=40)
# plt.ylabel('UMAP2', fontsize=40)
plt.savefig('umap_by_sample.png', dpi=600)

'''by cluster'''
plt.figure(figsize=(20,15))
groups = df_umap_composite.groupby('kmeans_cluster')
for name, group in groups:
    plt.plot(group.UMAP1, group.UMAP2, marker='o', linestyle='', alpha=.7, markersize=20, label=name)
# plt.legend(loc='best',prop={'size': 20})
# plt.xlim(-13, 15)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('UMAP1', fontsize=40)
# plt.ylabel('UMAP2', fontsize=40)
plt.savefig('umap_by_compositedescriptor.png', dpi=600)

'''pairplot'''
df_composite_features['kmeans_cluster'] = kmeans_predicted

sns.pairplot(df_composite_features, hue='kmeans_cluster', palette='flare', 
              vars=["Mean_morphology_PC1","Mean_morphology_PC2","Motility_PC1", "Motility_PC2"])
plt.savefig('pairplot.png', dpi=600)


# sns.pairplot(df_umap_composite, hue='kmeans_cluster',
#              vars=["UMAP1"])

# sns.pairplot(df_umap_composite, hue='condition',
#              vars=["UMAP1","UMAP2"])


# sns.pairplot(df_umap_composite, hue='kmeans_cluster',
#              vars=["UMAP1","UMAP2"])



#%%
"""Distribution of morphodynamics-motility clusters"""
cluster_type = 'kmeans_cluster'
group_condition_composite = pd.DataFrame(df_umap_composite.groupby(['condition',cluster_type]).size())
group_condition_composite['distribution(%)'] = group_condition_composite.groupby(level=0).apply(lambda x:  100*x / x.sum())
group_condition_composite = group_condition_composite.drop([0],axis=1)
group_condition_composite = group_condition_composite.unstack(level=0)
group_condition_composite = group_condition_composite.fillna(0)
group_condition_composite = group_condition_composite[[('distribution(%)','control'), ('distribution(%)','with7'), ('distribution(%)','with231')]]


plt.figure(figsize=(15,10))
heatmap = sns.heatmap(group_condition_composite.T, annot = True, annot_kws={"size": 20}, cmap = 'flare')
# plt.xlabel('Morphodynamics-motility cluster')
# plt.ylabel('condition')
# heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), va='center', fontsize=15)
# heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
# cbar.ax.tick_params(labelsize=20)
# plt.xticks(fontsize=25)
# plt.xlabel('Morphodynamics-motility cluster', fontsize = 30, labelpad=10)
# plt.ylabel('Condition', fontsize = 30, labelpad=10)
# plt.title('Distribution of morphodynamics-motility cluster', fontsize = 20)
plt.savefig('heatmap_composite.eps', format = 'eps', dpi=600)
plt.show()



'''# Shannon entropy'''
shannon_entropy_list = []
for condition_name in list(group_condition_composite['distribution(%)'].columns):
    shannon_entropy = 0
    for i in range(0, group_condition_composite['distribution(%)'][condition_name].shape[0]):
        shannon_entropy = shannon_entropy + -group_condition_composite['distribution(%)'][condition_name][i]/100*np.log2(group_condition_composite['distribution(%)'][condition_name][i]/100)
    shannon_entropy_list.append(shannon_entropy)
    print(condition_name)
shannon_entropy_list
plt.figure(figsize=(8,8))
colors = ["green", "royalblue", "indianred"]
# plt.title('Shannon entropy', fontsize = 20)
plt.bar(np.arange(len(shannon_entropy_list)), shannon_entropy_list, color=colors, width = 0.4, edgecolor = "black", linewidth=3)
plt.xticks([], [])
# plt.xticks(np.arange(len(shannon_entropy_list)), group_condition_composite['distribution(%)'].columns.tolist(),fontsize=25)
plt.yticks(fontsize=20)
plt.ylim(1.4, 1.6)
plt.savefig('Shannon_entropy_composite.eps', format = 'eps', dpi=600)

plt.show()


""""Hierarchical clustering"""
transposed_composite = group_condition_composite.T
transposed_composite
# Centroid : 두 군집의 중심점(centroid)를 정의한 다음 두 중심점의 거리를 군집간의 거리로 측정
# Single : 최단 연결법, 두 군집에 있는 모든 데이터 조합에서 데이터 사이 거리를 측정해서 가장 최소 거리(작은 값)를 기준으로 군집 거리를 측정
# Complete : 최장 연결법으로 두 클러스터상에서 가장 먼 거리를 이용해서 측정하는 방식
# Average : 평균 연결법, 두 군집의 데이터들 간 모든 거리들의 평균을 군집간 거리로 정의
# Ward : 연결될 수 있는 군집 조합을 만들고, 군집 내 편차들의 Sum of squared error을 기준으로 최소 제곱합을 가지게 되는 군집끼리 연결

hl_composite = hierarchy.linkage(transposed_composite,method='average',metric='euclidean')

plt.figure(figsize=(2,10))
dendrogram = hierarchy.dendrogram(hl_composite, orientation = 'left',labels = transposed.index)

hc_composite = AgglomerativeClustering(n_clusters=2,linkage='ward') 
# affinity = 'euclidan'으로 하면 ward method 자체가 euclidian base라는 이유로 에러가 발생함
y_predicted = hc_composite.fit_predict(transposed_composite)
transposed_mot['hc_cluster'] = y_predicted
transposed_mot


#%%
"""Quasi potential"""

# D=0.2
D=0.3
# D=0.4
# D=0.5
# D=0.6
bin_num = 100



##################### Generate vector field #########################
import scipy
from math import sqrt
from statistics import mean
import pde

xgrid = np.linspace(-7, 9, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-6, 8, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy') 
# Xgrid , Ygrid = 각각 (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)


transition_mag_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
transition_vec_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        transition_mag_array_temp[row,col] = [0]
        transition_vec_array_temp[row,col] = [(0,0)]        
        
label_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label']).apply(lambda x : x.name)).reset_index()
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보

    transition_mag = 0
    for t in range(0,cell_data.shape[0]-1): # 한 세포 안에서 각 time frame 마다
        x = cell_data['PC1']
        y = cell_data['PC2']
        residual = (x[t]-Xgrid)**2+(y[t]-Ygrid)**2 # residual = (100,100) 2d array
        min_coordinate = np.unravel_index(np.argmin(residual), residual.shape) # residual이 minimum인 array index 반환, (x,y)꼴
        #transition_mag = ((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_mag = sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_vec = (x[t+1]-x[t], y[t+1]-y[t])
        transition_mag_array_temp[min_coordinate].append(transition_mag)
        transition_vec_array_temp[min_coordinate].append(transition_vec)

########### transition magnitude의 list의 element 개수가 2 이상이면 0을 제외(평균 낼 때 평균값을 작게 만듬) #########           
for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        if len(transition_mag_array_temp[row,col]) > 1:
            transition_mag_array_temp[row,col].remove(0)
        if len(transition_vec_array_temp[row,col]) > 1:    
            transition_vec_array_temp[row,col].remove((0,0))

########### 각 element가 transition magnitude의 list인 100x100 array -> 각 list의 평균이 element인 100x100 array #########   
transition_mag_array= np.empty((bin_num,bin_num))
transition_vec_x_array= np.empty((bin_num,bin_num))
transition_vec_y_array= np.empty((bin_num,bin_num))

for row in range(0,transition_mag_array.shape[0]):
    for col in range(0,transition_mag_array.shape[1]):
        transition_mag_array[row,col] = mean(transition_mag_array_temp[row,col])
        x_temp = []
        y_temp = []
        for x, y in transition_vec_array_temp[row,col]:
            x_temp.append(x)
            y_temp.append(y)
        transition_vec_x_array[row,col] = mean(x_temp)
        transition_vec_y_array[row,col] = mean(y_temp)

##################### Generate initial pdf #########################
# initial data coordinates
x = df_fulltime_cell[(df_fulltime_cell['time_point'] >= 0) & 
                     (df_fulltime_cell['time_point'] <= 100)]['PC1'] # (data 수,) 1d vector
y = df_fulltime_cell[(df_fulltime_cell['time_point'] >= 0) & 
                     (df_fulltime_cell['time_point'] <= 100)]['PC2'] # (data 수,) 1d vector
kde_coordinate = np.vstack([x, y]) # (2, data 수) 2d array
kde = scipy.stats.gaussian_kde(kde_coordinate) # kernel 정의(bandwidth by Scott's Rule)

# evaluate on a regular grid
xgrid = np.linspace(-7, 9, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-6, 8, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid) 
# Xgrid , Ygrid = (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()])) 
pdf = Z.reshape(Xgrid.shape)
pdf = pdf/np.sum(pdf) # area under pdf should be 1 (그 전엔 area 34.298 나옴)
# Xgrid.ravel() = 10,000 shape 1d vector (-2,...5, -2, ...5 을 100번 반복한 vector)
# np.vstack() = (2,10,000) 2d array linspaced coordinates
# Z = (10000,) 1d vector 

####################### solve fokker-planck pde ##########################
grid = pde.CartesianGrid([[-7, 9], [-6, 8]], [bin_num, bin_num]) # [[0,x],[0,y],[x쪽 등분 수, y쪽 등분 수]]

field = np.stack((transition_vec_x_array, transition_vec_y_array), axis = 0)

velocity = pde.VectorField(grid, data=field)
initial_state = pde.ScalarField(grid, data=pdf)

#grid.plot(title=f'Area={grid.volume}', fig_style={'dpi': 200})
velocity.plot(kind='vector', 
              #fig_style={'dpi': 300}, 
              action='show',
              #cmap='jet', 
              title='vector field', )
initial_state.plot(title='initial state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
                  )

#print(grid.shape)
#print(np.array_equal(field, velocity.data))

bc_x = ({'value': 0}, {'value': 0}) 
bc_y = ({'value': 0}, {'value': 0})
#bc_x = ({'derivative': 0}, {'derivative': 0}) # 
#bc_y = ({'derivative': 0}, {'derivative': 0}) # natural bc로 하면 오른쪽 아래로 가있음, steady-state도 잘 안감

#D = 0.265
# 5보다 크면 inf로 error뜸
# 3이면 그냥 그림 중간을 평균으로 하는 normal distribution
# 0.08보다 작으면 inf로 error뜸
# D = 0.1, 0.2, 0.3, 0.4 다르고 0.4 이상은 일정

storage = pde.MemoryStorage()
trackers = [ 'progress',  # show progress bar
            'steady_state', # stop when steady state is reached
            #'plot',
            #pde.PlotTracker(show=True), # show images
            #storage.tracker(interval=10), # # store data every t = 10
            #pde.PrintTracker(interval=100),# print output every t = 100
           ]  

eq = pde.PDE({'P': f'- P*divergence(V) - dot(gradient(P),V) + {D} * laplace(P)'}, consts={'V': velocity},bc=[bc_x,bc_y])
result = eq.solve(initial_state, t_range=1e4, dt=1e-3, tracker=trackers)

result.plot(title='final state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
           )

####################### 확률이 - 인것 고치기 ##########################
for row in range(0,result.data.shape[0]):
    for col in range(0,result.data.shape[1]):
        if result.data[row,col] <=0:
            result.data[row,col] = abs(result.data).min()

result.data = result.data/np.sum(result.data) # area under pdf should be 1



dot_c1=np.arange(tskm.cluster_centers_.shape[1]-1) # tskm.cluster_centers_ = (cluster 수, time point 수, dimension)

plt.figure(figsize=(20,15))
plt.imshow(-np.log(result.data), origin='lower',cmap='jet', extent=[-12, 15, -10, 16], aspect='auto', 
            vmax=13, 
            vmin=7.5,
              )

plt.colorbar()

#%%
##################### Generate vector field #########################
from math import sqrt
from statistics import mean

xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy') 
# Xgrid , Ygrid = 각각 (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

condition = 'control'
label_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label','condition']).apply(lambda x : x.name)).reset_index()
label_data = label_data[label_data['condition']==condition].reset_index()
label_data = label_data.drop('index', axis = 1)
    
transition_mag_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
transition_vec_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        transition_mag_array_temp[row,col] = [0]
        transition_vec_array_temp[row,col] = [(0,0)]
            
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보

    transition_mag = 0
    for t in range(0,cell_data.shape[0]-1): # 한 세포 안에서 각 time frame 마다
        x = cell_data['PC1']
        y = cell_data['PC2']
        residual = (x[t]-Xgrid)**2+(y[t]-Ygrid)**2 # residual = (100,100) 2d array
        min_coordinate = np.unravel_index(np.argmin(residual), residual.shape) # residual이 minimum인 array index 반환, (x,y)꼴
        #transition_mag = ((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_mag = sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_vec = (x[t+1]-x[t], y[t+1]-y[t])
        transition_mag_array_temp[min_coordinate].append(transition_mag)
        transition_vec_array_temp[min_coordinate].append(transition_vec)

for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        if len(transition_mag_array_temp[row,col]) > 1:
            transition_mag_array_temp[row,col].remove(0)
        if len(transition_vec_array_temp[row,col]) > 1:    
            transition_vec_array_temp[row,col].remove((0,0))

transition_mag_array= np.empty((bin_num,bin_num))
transition_vec_x_array= np.empty((bin_num,bin_num))
transition_vec_y_array= np.empty((bin_num,bin_num))
for row in range(0,transition_mag_array.shape[0]):
    for col in range(0,transition_mag_array.shape[1]):
        transition_mag_array[row,col] = mean(transition_mag_array_temp[row,col])
        x_temp = []
        y_temp = []
        for x, y in transition_vec_array_temp[row,col]:
            x_temp.append(x)
            y_temp.append(y)
        transition_vec_x_array[row,col] = mean(x_temp)
        transition_vec_y_array[row,col] = mean(y_temp)

##################### Generate initial pdf #########################
# initial data coordinates
x = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC1'] # (data 수,) 1d vector
y = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC2'] # (data 수,) 1d vector
kde_coordinate = np.vstack([x, y]) # (2, data 수) 2d array
kde = scipy.stats.gaussian_kde(kde_coordinate) # kernel 정의(bandwidth by Scott's Rule)

# evaluate on a regular grid
xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid) 
# Xgrid , Ygrid = (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()])) 
pdf = Z.reshape(Xgrid.shape)
pdf = pdf/np.sum(pdf) # area under pdf should be 1 (그 전엔 area 34.298 나옴)
# Xgrid.ravel() = 10,000 shape 1d vector (-2,...5, -2, ...5 을 100번 반복한 vector)
# np.vstack() = (2,10,000) 2d array linspaced coordinates
# Z = (10000,) 1d vector 

####################### solve fokker-planck pde ##########################
grid = pde.CartesianGrid([[-13, 16], [-12, 18]], [bin_num, bin_num]) # [[0,x],[0,y],[x쪽 등분 수, y쪽 등분 수]]

field = np.stack((transition_vec_x_array, transition_vec_y_array), axis = 0)

velocity = pde.VectorField(grid, data=field)
initial_state = pde.ScalarField(grid, data=pdf)

#grid.plot(title=f'Area={grid.volume}', fig_style={'dpi': 200})
velocity.plot(kind='vector', 
              #fig_style={'dpi': 300}, 
              action='show',
              #cmap='jet', 
              title='vector field', )
initial_state.plot(title='initial state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
                  )

#print(grid.shape)
#print(np.array_equal(field, velocity.data))

bc_x = ({'value': 0}, {'value': 0}) 
bc_y = ({'value': 0}, {'value': 0})
#bc_x = ({'derivative': 0}, {'derivative': 0}) # 
#bc_y = ({'derivative': 0}, {'derivative': 0}) # natural bc로 하면 오른쪽 아래로 가있음, steady-state도 잘 안감

#D =0.265
# 5보다 크면 inf로 error뜸
# 3이면 그냥 그림 중간을 평균으로 하는 normal distribution
# 0.08보다 작으면 inf로 error뜸
# D = 0.1, 0.2, 0.3, 0.4 다르고 0.4 이상은 일정

storage = pde.MemoryStorage()
trackers = [ 'progress',  # show progress bar
            'steady_state', # stop when steady state is reached
            #'plot',
            #pde.PlotTracker(show=True), # show images
            #storage.tracker(interval=10), # # store data every t = 10
            #pde.PrintTracker(interval=100),# print output every t = 100
           ]  

eq = pde.PDE({'P': f'- P*divergence(V) - dot(gradient(P),V) + {D} * laplace(P)'}, consts={'V': velocity},bc=[bc_x,bc_y])
result_A = eq.solve(initial_state, t_range=1e4, dt=1e-3, tracker=trackers)

result_A.plot(title='final state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
           )

####################### 확률이 - 인것 고치기 ##########################
for row in range(0,result_A.data.shape[0]):
    for col in range(0,result_A.data.shape[1]):
        if result_A.data[row,col] <=0:
            result_A.data[row,col] = abs(result_A.data).min()
            
result_A.data = result_A.data/np.sum(result_A.data) # area under pdf should be 1



#%%
##################### Generate vector field #########################
from math import sqrt
from statistics import mean

xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy') 
# Xgrid , Ygrid = 각각 (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

condition = 'with7'
label_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label','condition']).apply(lambda x : x.name)).reset_index()
label_data = label_data[label_data['condition']==condition].reset_index()
label_data = label_data.drop('index', axis = 1)
    
transition_mag_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
transition_vec_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        transition_mag_array_temp[row,col] = [0]
        transition_vec_array_temp[row,col] = [(0,0)]
            
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보

    transition_mag = 0
    for t in range(0,cell_data.shape[0]-1): # 한 세포 안에서 각 time frame 마다
        x = cell_data['PC1']
        y = cell_data['PC2']
        residual = (x[t]-Xgrid)**2+(y[t]-Ygrid)**2 # residual = (100,100) 2d array
        min_coordinate = np.unravel_index(np.argmin(residual), residual.shape) # residual이 minimum인 array index 반환, (x,y)꼴
        #transition_mag = ((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_mag = sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_vec = (x[t+1]-x[t], y[t+1]-y[t])
        transition_mag_array_temp[min_coordinate].append(transition_mag)
        transition_vec_array_temp[min_coordinate].append(transition_vec)

for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        if len(transition_mag_array_temp[row,col]) > 1:
            transition_mag_array_temp[row,col].remove(0)
        if len(transition_vec_array_temp[row,col]) > 1:    
            transition_vec_array_temp[row,col].remove((0,0))

transition_mag_array= np.empty((bin_num,bin_num))
transition_vec_x_array= np.empty((bin_num,bin_num))
transition_vec_y_array= np.empty((bin_num,bin_num))
for row in range(0,transition_mag_array.shape[0]):
    for col in range(0,transition_mag_array.shape[1]):
        transition_mag_array[row,col] = mean(transition_mag_array_temp[row,col])
        x_temp = []
        y_temp = []
        for x, y in transition_vec_array_temp[row,col]:
            x_temp.append(x)
            y_temp.append(y)
        transition_vec_x_array[row,col] = mean(x_temp)
        transition_vec_y_array[row,col] = mean(y_temp)

##################### Generate initial pdf #########################
# initial data coordinates
x = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC1'] # (data 수,) 1d vector
y = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC2'] # (data 수,) 1d vector
kde_coordinate = np.vstack([x, y]) # (2, data 수) 2d array
kde = scipy.stats.gaussian_kde(kde_coordinate) # kernel 정의(bandwidth by Scott's Rule)

# evaluate on a regular grid
xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid) 
# Xgrid , Ygrid = (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()])) 
pdf = Z.reshape(Xgrid.shape)
pdf = pdf/np.sum(pdf) # area under pdf should be 1 (그 전엔 area 34.298 나옴)
# Xgrid.ravel() = 10,000 shape 1d vector (-2,...5, -2, ...5 을 100번 반복한 vector)
# np.vstack() = (2,10,000) 2d array linspaced coordinates
# Z = (10000,) 1d vector 

####################### solve fokker-planck pde ##########################
grid = pde.CartesianGrid([[-13, 16], [-12, 18]], [bin_num, bin_num]) # [[0,x],[0,y],[x쪽 등분 수, y쪽 등분 수]]

field = np.stack((transition_vec_x_array, transition_vec_y_array), axis = 0)

velocity = pde.VectorField(grid, data=field)
initial_state = pde.ScalarField(grid, data=pdf)

#grid.plot(title=f'Area={grid.volume}', fig_style={'dpi': 200})
velocity.plot(kind='vector', 
              #fig_style={'dpi': 300}, 
              action='show',
              #cmap='jet', 
              title='vector field', )
initial_state.plot(title='initial state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
                  )

#print(grid.shape)
#print(np.array_equal(field, velocity.data))

bc_x = ({'value': 0}, {'value': 0}) 
bc_y = ({'value': 0}, {'value': 0})
#bc_x = ({'derivative': 0}, {'derivative': 0}) # 
#bc_y = ({'derivative': 0}, {'derivative': 0}) # natural bc로 하면 오른쪽 아래로 가있음, steady-state도 잘 안감

#D =0.265
# 5보다 크면 inf로 error뜸
# 3이면 그냥 그림 중간을 평균으로 하는 normal distribution
# 0.08보다 작으면 inf로 error뜸
# D = 0.1, 0.2, 0.3, 0.4 다르고 0.4 이상은 일정

storage = pde.MemoryStorage()
trackers = [ 'progress',  # show progress bar
            'steady_state', # stop when steady state is reached
            #'plot',
            #pde.PlotTracker(show=True), # show images
            #storage.tracker(interval=10), # # store data every t = 10
            #pde.PrintTracker(interval=100),# print output every t = 100
           ]  

eq = pde.PDE({'P': f'- P*divergence(V) - dot(gradient(P),V) + {D} * laplace(P)'}, consts={'V': velocity},bc=[bc_x,bc_y])
result_B = eq.solve(initial_state, t_range=1e4, dt=1e-3, tracker=trackers)

result_B.plot(title='final state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
           )

####################### 확률이 - 인것 고치기 ##########################
for row in range(0,result_B.data.shape[0]):
    for col in range(0,result_B.data.shape[1]):
        if result_B.data[row,col] <=0:
            result_B.data[row,col] = abs(result_B.data).min()
            
result_B.data = result_B.data/np.sum(result_B.data) # area under pdf should be 1

#%%
##################### Generate vector field #########################
from math import sqrt
from statistics import mean

xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy') 
# Xgrid , Ygrid = 각각 (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

condition = 'with231'
label_data = pd.DataFrame(df_fulltime_cell.groupby(['image_stack','cell_label','condition']).apply(lambda x : x.name)).reset_index()
label_data = label_data[label_data['condition']==condition].reset_index()
label_data = label_data.drop('index', axis = 1)
    
transition_mag_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
transition_vec_array_temp = np.empty((bin_num,bin_num), dtype = 'object')
for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        transition_mag_array_temp[row,col] = [0]
        transition_vec_array_temp[row,col] = [(0,0)]
            
for i in range(0, label_data[0].shape[0]): # 각 세포마다
    cell_data = df_fulltime_cell.groupby(['image_stack','cell_label']).get_group((label_data[0][i][0],label_data[0][i][1])).reset_index()
    # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보

    transition_mag = 0
    for t in range(0,cell_data.shape[0]-1): # 한 세포 안에서 각 time frame 마다
        x = cell_data['PC1']
        y = cell_data['PC2']
        residual = (x[t]-Xgrid)**2+(y[t]-Ygrid)**2 # residual = (100,100) 2d array
        min_coordinate = np.unravel_index(np.argmin(residual), residual.shape) # residual이 minimum인 array index 반환, (x,y)꼴
        #transition_mag = ((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_mag = sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)
        transition_vec = (x[t+1]-x[t], y[t+1]-y[t])
        transition_mag_array_temp[min_coordinate].append(transition_mag)
        transition_vec_array_temp[min_coordinate].append(transition_vec)

for row in range(0,transition_mag_array_temp.shape[0]):
    for col in range(0,transition_mag_array_temp.shape[1]):
        if len(transition_mag_array_temp[row,col]) > 1:
            transition_mag_array_temp[row,col].remove(0)
        if len(transition_vec_array_temp[row,col]) > 1:    
            transition_vec_array_temp[row,col].remove((0,0))

transition_mag_array= np.empty((bin_num,bin_num))
transition_vec_x_array= np.empty((bin_num,bin_num))
transition_vec_y_array= np.empty((bin_num,bin_num))
for row in range(0,transition_mag_array.shape[0]):
    for col in range(0,transition_mag_array.shape[1]):
        transition_mag_array[row,col] = mean(transition_mag_array_temp[row,col])
        x_temp = []
        y_temp = []
        for x, y in transition_vec_array_temp[row,col]:
            x_temp.append(x)
            y_temp.append(y)
        transition_vec_x_array[row,col] = mean(x_temp)
        transition_vec_y_array[row,col] = mean(y_temp)

##################### Generate initial pdf #########################
# initial data coordinates
x = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC1'] # (data 수,) 1d vector
y = df_fulltime_cell[(df_fulltime_cell['condition']==condition) & 
                     (df_fulltime_cell['time_point'] <= 100) &
                     (df_fulltime_cell['time_point'] >= 0)]['PC2'] # (data 수,) 1d vector
kde_coordinate = np.vstack([x, y]) # (2, data 수) 2d array
kde = scipy.stats.gaussian_kde(kde_coordinate) # kernel 정의(bandwidth by Scott's Rule)

# evaluate on a regular grid
xgrid = np.linspace(-13, 16, bin_num) # (100, ) 1d x coordinate
ygrid = np.linspace(-12, 18, bin_num) # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid) 
# Xgrid , Ygrid = (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()])) 
pdf = Z.reshape(Xgrid.shape)
pdf = pdf/np.sum(pdf) # area under pdf should be 1 (그 전엔 area 34.298 나옴)
# Xgrid.ravel() = 10,000 shape 1d vector (-2,...5, -2, ...5 을 100번 반복한 vector)
# np.vstack() = (2,10,000) 2d array linspaced coordinates
# Z = (10000,) 1d vector 

####################### solve fokker-planck pde ##########################
grid = pde.CartesianGrid([[-13, 16], [-12, 18]], [bin_num, bin_num]) # [[0,x],[0,y],[x쪽 등분 수, y쪽 등분 수]]

field = np.stack((transition_vec_x_array, transition_vec_y_array), axis = 0)

velocity = pde.VectorField(grid, data=field)
initial_state = pde.ScalarField(grid, data=pdf)

#grid.plot(title=f'Area={grid.volume}', fig_style={'dpi': 200})
velocity.plot(kind='vector', 
              #fig_style={'dpi': 300}, 
              action='show',
              #cmap='jet', 
              title='vector field', )
initial_state.plot(title='initial state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
                  )

#print(grid.shape)
#print(np.array_equal(field, velocity.data))

bc_x = ({'value': 0}, {'value': 0}) 
bc_y = ({'value': 0}, {'value': 0})
#bc_x = ({'derivative': 0}, {'derivative': 0}) # 
#bc_y = ({'derivative': 0}, {'derivative': 0}) # natural bc로 하면 오른쪽 아래로 가있음, steady-state도 잘 안감

#D =0.265
# 5보다 크면 inf로 error뜸
# 3이면 그냥 그림 중간을 평균으로 하는 normal distribution
# 0.08보다 작으면 inf로 error뜸
# D = 0.1, 0.2, 0.3, 0.4 다르고 0.4 이상은 일정

storage = pde.MemoryStorage()
trackers = [ 'progress',  # show progress bar
            'steady_state', # stop when steady state is reached
            #'plot',
            #pde.PlotTracker(show=True), # show images
            #storage.tracker(interval=10), # # store data every t = 10
            #pde.PrintTracker(interval=100),# print output every t = 100
           ]  

eq = pde.PDE({'P': f'- P*divergence(V) - dot(gradient(P),V) + {D} * laplace(P)'}, consts={'V': velocity},bc=[bc_x,bc_y])
result_C = eq.solve(initial_state, t_range=1e4, dt=1e-3, tracker=trackers)

result_C.plot(title='final state', colorbar=True, cmap = plt.cm.get_cmap('jet'), 
                   #fig_style={'dpi': 300},
           )

####################### 확률이 - 인것 고치기 ##########################
for row in range(0,result_C.data.shape[0]):
    for col in range(0,result_C.data.shape[1]):
        if result_C.data[row,col] <=0:
            result_C.data[row,col] = abs(result_C.data).min()
            
result_C.data = result_C.data/np.sum(result_C.data) # area under pdf should be 1


#%%

# dot_c1=np.arange(tskm.cluster_centers_.shape[1]-1) # tskm.cluster_centers_ = (cluster 수, time point 수, dimension)

# plt.figure(figsize=(20,15))
# plt.imshow(-np.log(result.data), origin='lower',cmap='jet', extent=[-7, 9, -6, 8], aspect='auto', 
#             vmax=13, 
#             vmin=7.5,
#               )

# plt.colorbar()







#%%
vmax = 12
vmin = 7
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
#plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1, )
plt.imshow(-np.log(result_A.data), origin='lower',cmap='jet', extent=[-13, 16, -12, 18], aspect='auto', 
           vmax=vmax, 
           vmin=vmin
              )
plt.colorbar()
plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)

plt.subplot(2,3,2)
#plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1, )
plt.imshow(-np.log(result_B.data), origin='lower',cmap='jet', extent=[-13, 16, -12, 18], aspect='auto', 
           vmax=vmax, 
           vmin=vmin
              )
plt.colorbar()
plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)
plt.subplot(2,3,3)
#plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1, )
plt.imshow(-np.log(result_C.data), origin='lower',cmap='jet', extent=[-13, 16, -12, 18], aspect='auto', 
           vmax=vmax, 
           vmin=vmin
              )
plt.colorbar()
plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)
