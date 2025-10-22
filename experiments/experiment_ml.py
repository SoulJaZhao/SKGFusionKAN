import json
import os
import random
import socket
import struct
import warnings

import category_encoders as ce
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import from_networkx
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.utils import class_weight
from torch.optim import Adam
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler, NearMiss, InstanceHardnessThreshold, CondensedNearestNeighbour
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import ipaddress

warnings.filterwarnings("ignore")

dataset = 'NF-ToN-IoT-v2'
classes_count= 10
multiclass_report_file_path = f'./reports/RF_{dataset}_report.json'
binary_report_file_path = f'./binary_reports/RF_{dataset}_report.json'
data = pd.read_csv(f'{dataset}.csv')
data = data.groupby(by='Attack').sample(frac=0.05, random_state=2024)

def ip_to_int(ip):
    return int(ipaddress.IPv4Address(ip))

# 将 IPV4_SRC_ADDR 列中的每个 IP 地址替换为随机生成的 IP 地址
# 这里生成的 IP 地址范围是从 172.16.0.1 到 172.31.0.1
data['IPV4_SRC_ADDR'] = np.vectorize(ip_to_int)(data['IPV4_SRC_ADDR'])

data['IPV4_DST_ADDR'] = np.vectorize(ip_to_int)(data['IPV4_DST_ADDR'])

# 删除不再需要的 Label 列
data.drop(columns=['Label'], inplace=True)

# 将 Label 列重命名为 label
data.rename(columns={"Attack": "label"}, inplace=True)

le_label = LabelEncoder()
data['label'] = le_label.fit_transform(data['label'])

# 将 label 列提取出来，保存到一个单独的变量中
label = data['label']

data = data.drop(columns='label')

# 将数据分为训练集和测试集，按 70% 和 30% 的比例分配，保证 stratify 参数确保按标签分层抽样
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.3, random_state=2024, stratify=label)

# 创建 StandardScaler 对象，用于标准化数据
scaler = StandardScaler()

# 创建 TargetEncoder 对象，用于对分类特征进行目标编码
encoder = ce.TargetEncoder(cols=['TCP_FLAGS', 'L7_PROTO', 'PROTOCOL'])

# 用训练集的特征和标签拟合编码器
encoder.fit(X_train, y_train)

# 对训练集的特征进行编码转换
X_train = encoder.transform(X_train)

# 需要标准化的列，去除掉 label 列
cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns)) - set(['label']))

# 对需要标准化的列进行标准化
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])

# 对测试集进行目标编码转换
X_test = encoder.transform(X_test)

# 对需要标准化的列进行标准化
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

# 获取边标签的唯一值
unique_labels = np.unique(y_test)

# 计算每个类的权重，以处理类别不平衡问题
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=unique_labels,
                                                  y=y_test)

# # 创建 KNN 分类器实例
# knn = KNeighborsClassifier(n_neighbors=classes_count)  # 这里选择了 k=3
#
# # 训练模型
# knn.fit(X_train, y_train)
#
# # 进行预测
# y_pred = knn.predict(X_test)

# # 创建 ExtraTreesClassifier 实例
# etc = ExtraTreesClassifier(n_estimators=100, random_state=42)
#
# # 训练模型
# etc.fit(X_train, y_train)
#
# # 进行预测
# y_pred = etc.predict(X_test)

# # 创建 SVM 分类器
# svm_model = SVC(kernel='rbf', C=1.0, random_state=2024)
#
# # 训练模型
# svm_model.fit(X_train, y_train)
#
# # 进行预测
# y_pred = svm_model.predict(X_test)


# 创建 RandomForestClassifier 实例
rf_model = RandomForestClassifier(n_estimators=20, random_state=2024)

# 训练模型
rf_model.fit(X_train, y_train)

# 进行预测
y_pred = rf_model.predict(X_test)


multi_actual = le_label.inverse_transform(y_test)
multi_test_pred = le_label.inverse_transform(y_pred)

# 打印详细的分类报告
multi_report = classification_report(multi_actual, multi_test_pred, target_names=np.unique(multi_actual),
                                     output_dict=True)
# 保存分类报告为JSON文件
with open(multiclass_report_file_path, 'w') as jsonfile:
    json.dump(multi_report, jsonfile, indent=4)

print(multi_report)

# 将实际标签和预测标签转换为 "Normal" 或 "Attack"
binary_actual = ["Normal" if i == 0 else "Attack" for i in y_test]
binary_test_pred = ["Normal" if i == 0 else "Attack" for i in y_pred]

# 打印详细的分类报告
binary_report = classification_report(binary_actual, binary_test_pred, target_names=["Normal", "Attack"],
                                      output_dict=True)
# 保存分类报告为JSON文件
with open(binary_report_file_path, 'w') as jsonfile:
    json.dump(binary_report, jsonfile, indent=4)

print(binary_report)









