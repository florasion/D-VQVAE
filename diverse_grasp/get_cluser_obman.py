
import json
from sklearn.cluster import KMeans
from scipy.stats import entropy
import numpy as np
file_names = []
test_dataset = 'FHAB'
with open(test_dataset+'.txt', 'r') as file:
    for line in file:
        file_names.append(line.strip())  # 去除行尾换行符

#print(file_names)
data_list = []
print('start load data')
#i=0
for file_name in file_names:
    #i+=1
    #if i == 100:
    #    break
    with open('./'+test_dataset+'/'+file_name, 'r') as json_file:
        data = json.load(json_file)
    for i in range(100):
        try:
            data_list.append(data['recon_params'][i][0])
        except:
            continue
print('data loaded')
# 将数据列表转化为合适的数据结构
data_matrix = []  # 这里假设data是一个字典，需要根据实际数据结构进行修改
print(len(data_list[0]))
for data in data_list:
    data_matrix.append(data)
#print(data_matrix)
# 初始化K-Means模型，假设要分为n_clusters个簇
n_clusters = 20  # 根据需求设置簇的数量
print('start_Kmeans')
kmeans = KMeans(n_clusters=n_clusters)
# 进行聚类
kmeans.fit(data_matrix)
# 获得聚类结果

cluster_labels = kmeans.labels_
cluster_counts = np.bincount(cluster_labels)
print(cluster_labels)
data_entropy = entropy(cluster_counts, base=2)  # 使用底数为2的对数计算熵

print("Entropy:", data_entropy)
from collections import Counter
cluster_sizes = dict(Counter(cluster_labels))
total_size = 0
# 打印每个簇的大小
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: Size {size}")
    total_size  += size
print('mean_size: ',total_size/20)