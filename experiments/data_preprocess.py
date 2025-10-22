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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.utils import class_weight
from torch.optim import Adam
from tqdm import tqdm
from efficientKan import KANLinear
from imblearn.under_sampling import RandomUnderSampler, NearMiss, InstanceHardnessThreshold, CondensedNearestNeighbour
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

'''
定义全部参数
数据集名称：
    NF-BoT-IoT
    NF-BoT-IoT-v2
    NF-ToN-IoT
    NF-ToN-IoT-v2
    NF-CSE-CIC-IDS2018-v2
'''


# 获取数据集数据
def get_data(dataset):
    if dataset == 'NF-BoT-IoT':
        data = pd.read_csv('NF-BoT-IoT.csv')
    elif dataset == 'NF-BoT-IoT-v2':
        data = pd.read_csv('NF-BoT-IoT-v2.csv')
    elif dataset == 'NF-ToN-IoT':
        data = pd.read_csv('NF-ToN-IoT.csv')
    elif dataset == 'NF-ToN-IoT-v2':
        data = pd.read_csv('NF-ToN-IoT-v2.csv')
    elif dataset == 'NF-CSE-CIC-IDS2018-v2':
        data = pd.read_csv('NF-CSE-CIC-IDS2018-v2.csv')
    else:
        raise ValueError("Invalid dataset name.")
    return data

# 获取训练保存图的路径
def get_train_graph_file_path(dataset):
    return f'{dataset}_train_graph.dgl'

# 获取测试保存图的路径
def get_test_graph_file_path(dataset):
    return f'{dataset}_test_graph.dgl'

# 获取测试标签保存路径
def get_test_labels_file_path(dataset):
    return f'{dataset}_test_labels.npy'

def get_label_encoder_file_path(dataset):
    return f'{dataset}_le_label.pkl'

# 定义保存 DGL 图的方法
def save_graph(graph, file_path):
    dgl.save_graphs(file_path, [graph])

# 定义加载 DGL 图的方法
def load_graph(file_path):
    return dgl.load_graphs(file_path)[0][0]

def frac_data(dataset, frac):
    data = get_data(dataset)
    data = data.groupby(by='Attack').sample(frac=frac, random_state=2024)
    print("Resampled data shape:", data.shape)
    return data

def resample_nf_bot_iot(data):
    # 将 IPV4_SRC_ADDR 列中的每个 IP 地址替换为随机生成的 IP 地址
    # 这里生成的 IP 地址范围是从 172.16.0.1 到 172.31.0.1
    data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(
        lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

    # 将 IPV4_SRC_ADDR 列中的每个值转换为字符串类型
    data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
    # 将 L4_SRC_PORT 列中的每个值转换为字符串类型
    data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
    # 将 IPV4_DST_ADDR 列中的每个值转换为字符串类型
    data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
    # 将 L4_DST_PORT 列中的每个值转换为字符串类型
    data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)

    # 将 IPV4_SRC_ADDR 和 L4_SRC_PORT 列的值连接起来，中间用冒号分隔
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
    # 将 IPV4_DST_ADDR 和 L4_DST_PORT 列的值连接起来，中间用冒号分隔
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']

    # 删除不再需要的 L4_SRC_PORT 和 L4_DST_PORT 列
    data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)

    # 删除不再需要的 Label 列
    data.drop(columns=['Label'], inplace=True)

    # 将 Label 列重命名为 label
    data.rename(columns={"Attack": "label"}, inplace=True)

    le_label = LabelEncoder()
    data['label'] = le_label.fit_transform(data['label'])

    # 保存 LabelEncoder 对象
    with open(get_label_encoder_file_path(dataset), 'wb') as f:
        pickle.dump(le_label, f)

    # 将 label 列提取出来，保存到一个单独的变量中
    label = data['label']

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

    # 将标准化后的列组合成列表，添加为新的列 'h'
    X_train['h'] = X_train[cols_to_norm].values.tolist()

    # 从 pandas DataFrame 中创建一个无向多重图
    # 边的数据包含 'h' 和 'label' 列
    G = nx.from_pandas_edgelist(X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'], create_using=nx.MultiGraph())

    # 将无向图转换为有向图
    G = G.to_directed()

    # 将 NetworkX 图转换为 DGL 图，边的数据包含 'h' 和 'label' 属性
    G = from_networkx(G, edge_attrs=['h', 'label'])

    # 为每个节点的 'h' 属性赋值，初始值为全 1 的张量，维度与边的 'h' 属性相同
    G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

    # 为每条边添加 'train_mask' 属性，初始值为 True，表示这些边用于训练
    G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

    # 保存图 G 到指定路径
    save_graph(G, get_train_graph_file_path(dataset))
    print("Train graph created and saved to file.")

    print("Test graph file not found. Creating new test graph.")
    # 对测试集进行目标编码转换
    X_test = encoder.transform(X_test)

    # 对需要标准化的列进行标准化
    X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

    # 将标准化后的列组合成列表，添加为新的列 'h'
    X_test['h'] = X_test[cols_to_norm].values.tolist()

    # 从 pandas DataFrame 中创建一个无向多重图
    # 边的数据包含 'h' 和 'label' 列
    G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'],
                                     create_using=nx.MultiGraph())

    # 将无向图转换为有向图
    G_test = G_test.to_directed()

    # 将 NetworkX 图转换为 DGL 图，边的数据包含 'h' 和 'label' 属性
    G_test = from_networkx(G_test, edge_attrs=['h', 'label'])

    # 从 G_test 的边数据中取出 'label' 并删除
    actual = G_test.edata.pop('label')

    # 为 G_test 的每个节点设置 'feature' 属性，初始值为全 1 的张量，维度与训练图中的节点特征相同
    G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[1])

    # 保存测试图 G_test 到指定路径
    save_graph(G_test, get_test_graph_file_path(dataset))
    np.save(get_test_labels_file_path(dataset), actual)
    print("Test graph created and saved to file.")


# 下采样NF-BoT-IoT数据集
def resample_data(dataset):
    data = frac_data(dataset, 0.005)
    resample_nf_bot_iot(data)
    print("Train graph or test graph file not found. Creating new graph.")


if __name__ == '__main__':
    dataset = 'NF-ToN-IoT-v2'
    resample_data(dataset)


