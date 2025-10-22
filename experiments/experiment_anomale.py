import json
import math
import random
import socket
import struct
import time
import warnings

import category_encoders as ce
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import from_networkx
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from torch.optim import Adam
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import sys
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 module 目录的绝对路径（上一级目录的 module 文件夹）
module_path = os.path.abspath(os.path.join(current_dir, '../module'))

# 添加 module 目录到 sys.path
sys.path.append(module_path)
from efficientKan import KANLinear
from CPCA2d import CPCABlock
from SKNet import SKAttention
from SCSA import SCSA
from CBAM import CBAM

from DFF2d import DFF
from SDM2d import SDM
from MSAF2d import MSAF
from SFFusion2d import SqueezeAndExciteFusionAdd
from TIF import TIF
from WCMF import WCMF
from GatedFusion import gatedFusion

import pickle
from data_preprocess import get_train_graph_file_path, get_test_graph_file_path, get_label_encoder_file_path, \
    get_test_labels_file_path

warnings.filterwarnings("ignore")


# 定义加载 DGL 图的方法
def load_graph(file_path):
    return dgl.load_graphs(file_path)[0][0]


# 定义计算准确度的函数
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()


# 定义计算 F1 score 的函数
# def compute_f1_score(pred, labels):
#     pred_labels = pred.argmax(1).cpu().numpy()
#     # 如果 labels 已经是 numpy 数组，则直接使用它
#     if isinstance(labels, np.ndarray):
#         true_labels = labels
#     else:
#         true_labels = labels.cpu().numpy()
#     return f1_score(true_labels, pred_labels, average='weighted')

def compute_f1_score(pred, labels, binary=False):
    if binary:
        # Directly use the string labels for binary classification
        pred_labels = pred
        true_labels = labels
    else:
        # Handle multi-class classification
        pred_labels = pred.argmax(1).cpu().numpy()
        if isinstance(labels, np.ndarray):
            true_labels = labels
        else:
            true_labels = labels.cpu().numpy()

    return f1_score(true_labels, pred_labels, average='weighted')


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
      super(SAGELayer, self).__init__()
      self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      self.activation = F.relu
      self.W_edge = nn.Linear(128 * 2, 256)
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

        # Compute edge embeddings
        u, v = g.edges()
        edge = self.W_edge(th.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        return g.ndata['h'], edge

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, 128, F.relu))

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt:
        e_perm = th.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm]
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        #nfeats = layer(g, nfeats, efeats)
        nfeats, e_feats = layer(g, nfeats, efeats)
      #return nfeats.sum(1)
      return nfeats.sum(1), e_feats.sum(1)


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
      super(Discriminator, self).__init__()
      self.weight = nn.Parameter(th.Tensor(n_hidden, n_hidden))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      features = th.matmul(features, th.matmul(self.weight, summary))
      return features

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,  F.relu)
      #self.discriminator = Discriminator(128)
      self.discriminator = Discriminator(256)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)

      positive = positive[1]
      negative = negative[1]

      summary = th.sigmoid(positive.mean(dim=0))

      positive = self.discriminator(positive, summary)
      negative = self.discriminator(negative, summary)

      l1 = self.loss(positive, th.ones_like(positive))
      l2 = self.loss(negative, th.zeros_like(negative))

      return l1 + l2


# 定义实验参数
dataset = 'NF-BoT-IoT'

if dataset == 'NF-BoT-IoT' or dataset == 'NF-BoT-IoT-v2':
    output_classes = 5
elif dataset == "NF-CSE-CIC-IDS2018-v2":
    output_classes = 15
else:
    output_classes = 10

epochs = 2
binary_best_model_file_path = f'./binary_model/anomale_{dataset}_best_model.pth'
binary_report_file_path = f'./binary_reports/anomale__{dataset}_report.json'
binary_test_pred_file_path = f'./binary_predictions/anomale__{dataset}_test_pred.pth'

multiclass_best_model_file_path = f'./model/anomale__{dataset}_best_model.pth'
multiclass_report_file_path = f'./reports/anomale__{dataset}_report.json'
multiclass_test_pred_file_path = f'./predictions/anomale__{dataset}_test_pred.pth'

# 打印出所有实验配置
print(f'Dataset: {dataset}')
print(f'Epochs: {epochs}')
print(f'Best model file path: {multiclass_best_model_file_path}')
print(f'Report file path: {multiclass_report_file_path}')

# 获取图
G = load_graph(get_train_graph_file_path(dataset))
# 获取测试图
G_test = load_graph(get_test_graph_file_path(dataset))
# 获取测试标签
actual = np.load(get_test_labels_file_path(dataset))
# 获取标签编码器
with open(get_label_encoder_file_path(dataset), 'rb') as f:
    le_label = pickle.load(f)

# 将节点特征重塑为三维张量
# 原始节点特征维度为 (num_nodes, feature_dim)
# 重塑后的维度为 (num_nodes, 1, feature_dim)
G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))

# 将边特征重塑为三维张量
# 原始边特征维度为 (num_edges, feature_dim)
# 重塑后的维度为 (num_edges, 1, feature_dim)
G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))

# 重塑测试图的节点特征为三维张量
# 原始节点特征维度为 (num_nodes, feature_dim)
# 重塑后的维度为 (num_nodes, 1, feature_dim)
G_test.ndata['feature'] = th.reshape(G_test.ndata['feature'],
                                     (G_test.ndata['feature'].shape[0], 1, G_test.ndata['feature'].shape[1]))

# 重塑测试图的边特征为三维张量
# 原始边特征维度为 (num_edges, feature_dim)
# 重塑后的维度为 (num_edges, 1, feature_dim)
G_test.edata['h'] = th.reshape(G_test.edata['h'], (G_test.edata['h'].shape[0], 1, G_test.edata['h'].shape[1]))

# 从图的边属性中提取标签并转换为 numpy 数组
edge_labels = G.edata['label'].cpu().numpy()

# 获取边标签的唯一值
unique_labels = np.unique(edge_labels)

# 计算每个类的权重，以处理类别不平衡问题
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=unique_labels,
                                                  y=edge_labels)

# 首先，根据是否有 CUDA 可用来设置设备
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 将 class_weights 转换为浮点张量，并根据设备选择移动到 GPU 或保持在 CPU 上
class_weights = th.FloatTensor(class_weights).to(device)

# 初始化损失函数，使用计算的类权重
criterion = nn.CrossEntropyLoss(weight=class_weights)

G = G.to(device)

# 获取节点特征和边特征
node_features = G.ndata['h']
edge_features = G.edata['h']

# 获取边标签和训练掩码
edge_label = G.edata['label']
train_mask = G.edata['train_mask']

# 将模型移动到设备上（GPU 或 CPU）
model = dgi = DGI(ndim_in=G.ndata['h'].shape[2], ndim_out=128, edim=G.edata['h'].shape[2], activation=F.relu).to(device)

# 将节点特征和边特征移动到设备上
node_features = node_features.to(device)
edge_features = edge_features.to(device)
edge_label = edge_label.to(device)
train_mask = train_mask.to(device)

# 定义优化器
dgi_optimizer = th.optim.Adam(dgi.parameters(),
                lr=1e-3,
                weight_decay=0.)

# 变量用于保存最高的 F1 score
best_f1_score = 0.0
binary_best_f1_score = 0.0

# 将测试图移动到设备（GPU 或 CPU）
G_test = G_test.to(device)

import timeit

# 记录开始时间
start_time = timeit.default_timer()

# 获取测试图的节点特征和边特征
node_features_test = G_test.ndata['feature']
edge_features_test = G_test.edata['h']

# 训练循环
cnt_wait = 0
best = 1e9
best_t = 0
dur = []

for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
    dgi.train()

    if epoch >= 3:
        t0 = time.time()

    dgi_optimizer.zero_grad()
    loss = dgi(G, node_features, edge_features)
    loss.backward()
    dgi_optimizer.step()

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        th.save(dgi.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == 200:
      print('Early stopping!')
      break

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 50 == 0:

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur),
              loss.item(),
              G.num_edges() / np.mean(dur) / 1000))

# 进行前向传播，获取测试预测
# 将模型移动到设备上（GPU 或 CPU）
dgi.load_state_dict(th.load('best_dgi.pkl'))

training_emb = dgi.encoder(G, G.ndata['h'], G.edata['h'])[1]
training_emb = training_emb.detach().cpu().numpy()

testing_emb = dgi.encoder(G_test, node_features_test, edge_features_test)[1]
testing_emb = testing_emb.detach().cpu().numpy()

# 初始化随机森林分类器
classifier = RandomForestClassifier(n_estimators=30, max_depth=None, random_state=2024)

# 使用训练集的嵌入和标签进行训练
classifier.fit(training_emb, edge_labels)

# 对测试集进行预测
test_pred = classifier.predict(testing_emb)
binary_test_pred = test_pred


# 计算并打印前向传播所花费的时间
elapsed = timeit.default_timer() - start_time
print(str(elapsed) + ' seconds')

multi_actual = le_label.inverse_transform(actual)
multi_test_pred = le_label.inverse_transform(test_pred)

# 打印详细的分类报告
multi_report = classification_report(multi_actual, multi_test_pred, target_names=np.unique(multi_actual),
                                     output_dict=True)
# 保存分类报告为JSON文件
with open(multiclass_report_file_path, 'w') as jsonfile:
    json.dump(multi_report, jsonfile, indent=4)

print(multi_report)


# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# 绘制混淆矩阵
# cm = confusion_matrix(multi_actual, multi_test_pred)
# plot_confusion_matrix(cm=cm,
#                       normalize=False,
#                       target_names=np.unique(multi_actual),
#                       title="Confusion Matrix")

# 将实际标签和预测标签转换为 "Normal" 或 "Attack"
binary_actual = ["Normal" if i == 0 else "Attack" for i in actual]
binary_test_pred = ["Normal" if i == 0 else "Attack" for i in test_pred]

# 打印详细的分类报告
binary_report = classification_report(binary_actual, binary_test_pred, target_names=["Normal", "Attack"],
                                      output_dict=True)
# 保存分类报告为JSON文件
with open(binary_report_file_path, 'w') as jsonfile:
    json.dump(binary_report, jsonfile, indent=4)

print(classification_report(binary_actual, binary_test_pred, target_names=["Normal", "Attack"]))


# 定义绘制混淆矩阵的函数
def binary_plot_confusion_matrix(cm,
                                 target_names,
                                 title='Confusion matrix',
                                 cmap=None,
                                 normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}\n'
               f'Precision={binary_report["weighted avg"]["precision"]:0.4f}; Recall={binary_report["weighted avg"]["recall"]:0.4f}; F1-Score={binary_report["weighted avg"]["f1-score"]:0.4f}')
    plt.show()

# 绘制混淆矩阵
# binary_cm = confusion_matrix(binary_actual, binary_test_pred)
# binary_plot_confusion_matrix(cm=binary_cm,
#                       normalize=False,
#                       target_names=np.unique(binary_actual),
#                       title="Confusion Matrix")

