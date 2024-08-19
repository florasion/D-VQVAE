import scipy.cluster
from scipy.stats import entropy
import numpy as np
import torch
import json

def diversity(params_list, cls_num=20):
    # params_list = scipy.cluster.vq.whiten(params_list)
    #  # k-means
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences  count: [20]
    print(counts)
    ee = entropy(counts)
    return ee, np.mean(dist)

    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20
    
test_dataset='ho3d'

file_names = []
with open(test_dataset+'.txt', 'r') as file:
    for line in file:
        file_names.append(line.strip()) 
data_list = []
recon_list = []
num = 0
print('start load data')
for file_name in file_names:
    

    with open('./'+test_dataset+'/'+file_name, 'r') as json_file:
        data = json.load(json_file)
    for i in range(0,len(data['recon_params'])):

        try:
            num+=1
            data_list.append(data['recon_params'][i][0])
            recon_param= data['recon_params'][i][0]
            recon_list.append(recon_param)

        except:
            continue
print('sample num : ',num)
print('data loaded')
data_matrix = []  
recon_matrix = []

for data in data_list:
    data_matrix.append(data)


n_clusters = 20  



for data in recon_list:
    recon_matrix.append(data)
#print(len(recon_matrix))
entropy, dist = diversity(recon_matrix, cls_num=n_clusters)
print(" entropy :", entropy)
print(" dist :", dist)