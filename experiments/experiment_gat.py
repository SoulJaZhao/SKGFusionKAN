import json
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
import dgl.function as fn
from dgl import from_networkx
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from torch.optim import Adam
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


class TwoLayerGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout):
        super(TwoLayerGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads=4, feat_drop=dropout)
        self.gat2 = GATConv(hidden_feats, out_feats, num_heads=1, feat_drop=dropout)
        self.dropout = dropout

    def forward(self, g, nfeats, efeats):
        # Apply first GAT layer
        h = self.gat1(g, nfeats)
        h = th.mean(h, 2)  # Flatten the output of multi-head attention
        h = th.relu(h)

        # Apply second GAT layer
        h = self.gat2(g, h)
        h = h.flatten(1)  # Flatten the output of multi-head attention
        return h


# 定义一个MLPPredictor类，继承自nn.Module
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes, fusion_name, mlp_name):
        super().__init__()
        # 初始化MLPPredictor类
        # 定义线性层，输入维度为两倍的节点特征维度，输出维度为指定的类别数
        self.mlp_name = mlp_name
        self.fusion_name = fusion_name

        # Explicitly set fusion to None initially
        self.fusion = None

        if fusion_name == "DFF":
            self.fusion = DFF(in_features)
        elif fusion_name == "MSAF":
            self.fusion = MSAF(channels=in_features, r=4)
        elif fusion_name == "SDM":
            self.fusion = SDM(in_channel=in_features, guidance_channels=in_features)
        elif fusion_name == "SFF":
            self.fusion = SqueezeAndExciteFusionAdd(channels_in=in_features)
        elif fusion_name == "TIF":
            self.fusion = TIF(dim_s=in_features, dim_l=in_features)
        elif fusion_name == "WCMF":
            self.fusion = WCMF(channel=in_features)
        elif fusion_name == "GATE":
            self.fusion = gatedFusion(in_features)
        else:
            self.fusion = None

        if mlp_name == "KAN":
            # 第一层 KANLinear，输入维度为两倍的节点特征维度
            self.fc1 = KANLinear(in_features * 2, in_features)
            # 激活函数
            self.relu = nn.ReLU()
            # 第二层 KANLinear，输出维度为指定的类别数
            self.fc2 = KANLinear(in_features, out_classes)
        elif mlp_name == "MLP":
            self.W = nn.Linear(in_features * 2, out_classes)

    # 定义边应用函数，edges是DGL中的边数据
    def apply_edges(self, edges):
        # 获取源节点特征
        h_u = edges.src['h']
        # 获取目标节点特征
        h_v = edges.dst['h']

        # 执行特征融合
        if self.fusion is not None:
            if self.fusion_name == "GATE":
                input1 = h_u.view(h_u.size(0), 1, 1, -1)
                input2 = h_v.view(h_v.size(0), 1, 1, -1)
            else:
                input1 = h_u.view(h_u.size(0), -1, 1, 1)
                input2 = h_v.view(h_v.size(0), -1, 1, 1)
            h_u = self.fusion(input1, input2).view(h_u.size(0), -1)

        # 将源节点和目标节点的特征连接起来，通过线性层转换
        if self.mlp_name == "KAN":
            # 将源节点和目标节点的特征连接起来，通过线性层转换
            combined_features = th.cat([h_u, h_v], 1)

            # 通过两层 KANLinear 进行特征转换
            x = self.fc1(combined_features)
            x = self.relu(x)
            score = self.fc2(x)
        elif self.mlp_name == "MLP":
            score = self.W(th.cat([h_u, h_v], 1))

        # 返回包含预测得分的字典
        return {'score': score}

    # 定义前向传播函数
    def forward(self, graph, h):
        with graph.local_scope():
            # 使用local_scope保护当前图数据不被修改
            # 设置节点特征
            graph.ndata['h'] = h
            # 应用边上的计算函数，将预测得分存储在边数据中
            graph.apply_edges(self.apply_edges)
            # 返回边数据中的预测得分
            return graph.edata['score']


# 定义一个Model类，继承自nn.Module
class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, attention_name, fusion_name, mlp_name,
                 output_classes):
        super().__init__()
        # 初始化Model类
        self.gnn = TwoLayerGAT(ndim_in, edim, ndim_out, dropout)
        # 创建一个MLPPredictor模型，用于边的预测
        self.pred = MLPPredictor(ndim_out, output_classes, fusion_name, mlp_name)

    # 定义前向传播函数
    def forward(self, g, nfeats, efeats):
        # 使用SAGE模型进行节点特征的计算
        h = self.gnn(g, nfeats, efeats)
        # 使用MLPPredictor模型进行边的预测，并返回预测结果
        return self.pred(g, h)


# 定义实验参数
dataset = 'NF-BoT-IoT'
'''
attention 方法：
    - SE: Squeeze-and-Excitation
    - SK: Selective Kernel
    - CPCA: Channel-wise PCA
    - SCSA: Spatial and Channel Squeeze & Excitation
    - CBAM: Convolutional Block Attention Module
'''
attention_name = None

'''
fusion 方法:
    - DFF: DFF
    - SDM: SDM
    - SFF: SFFusion
    - TIF: TIF
    - WCMF: WCMF
    - GATE: GatedFusion
'''
fusion_name = None

'''
mlp 方法：
    - KAN: KAN
    - MLP: MLP
'''
mlp_name = "MLP"

if dataset == 'NF-BoT-IoT' or dataset == 'NF-BoT-IoT-v2':
    output_classes = 5
elif dataset == "NF-CSE-CIC-IDS2018-v2":
    output_classes = 15
else:
    output_classes = 10

epochs = 3000
binary_best_model_file_path = f'./binary_model/GAT_{dataset}_best_model.pth'
binary_report_file_path = f'./binary_reports/GAT_{dataset}_report.json'
binary_test_pred_file_path = f'./binary_predictions/GAT_{dataset}_test_pred.pth'

multiclass_best_model_file_path = f'./model/GAT_{dataset}_best_model.pth'
multiclass_report_file_path = f'./reports/GAT_{dataset}_report.json'
multiclass_test_pred_file_path = f'./predictions/GAT_{dataset}_test_pred.pth'

# 打印出所有实验配置
print(f'Dataset: {dataset}')
print(f'Attention: {attention_name}')
print(f'Fusion: {fusion_name}')
print(f'MLP: {mlp_name}')
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
model = Model(ndim_in=G.ndata['h'].shape[2], ndim_out=128, edim=G.ndata['h'].shape[2], activation=F.relu, dropout=0.2,
              attention_name=attention_name, fusion_name=fusion_name, mlp_name=mlp_name,
              output_classes=output_classes).to(device)

# 将节点特征和边特征移动到设备上
node_features = node_features.to(device)
edge_features = edge_features.to(device)
edge_label = edge_label.to(device)
train_mask = train_mask.to(device)

# 定义优化器
opt = Adam(model.parameters())

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
for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
    # 前向传播，获取预测值
    pred = model(G, node_features, edge_features)

    # 计算损失，只考虑训练掩码内的边
    loss = criterion(pred[train_mask], edge_label[train_mask])

    # 清零梯度
    opt.zero_grad()

    # 反向传播，计算梯度
    loss.backward()

    # 更新模型参数
    opt.step()

    # 每 100 轮输出一次训练准确度和 F1 score
    if epoch % 100 == 0:
        accuracy = compute_accuracy(pred[train_mask], edge_label[train_mask])
        f1 = compute_f1_score(pred[train_mask], edge_label[train_mask])
        print(f'Epoch {epoch}: Training acc: {accuracy}, F1 score: {f1}')

    # 计算当前模型的 F1 score，如果高于最高的 F1 score，则保存模型和图
    model.eval()  # 切换到评估模式
    with th.no_grad():  # 禁用梯度计算
        test_pred = model(G_test, node_features_test, edge_features_test)
        current_f1_score = compute_f1_score(test_pred, actual)

        # 获取预测标签
        binary_test_pred = test_pred.argmax(1)

        # 将预测结果从 GPU 移动到 CPU，并转换为 numpy 数组
        binary_test_pred_numpy = binary_test_pred.cpu().detach().numpy()

        # Create binary labels for prediction and actual labels
        binary_actual = ["Normal" if i == 0 else "Attack" for i in actual]
        binary_test_pred = ["Normal" if i == 0 else "Attack" for i in binary_test_pred_numpy]

        # Compute binary F1 score
        # 打印详细的分类报告
        binary_report = classification_report(binary_actual, binary_test_pred, target_names=["Normal", "Attack"],
                                              output_dict=True)
        current_binary_f1_score = binary_report["weighted avg"]["f1-score"]

        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            th.save(model, multiclass_best_model_file_path)
            print(f'New best model and graph saved at epoch {epoch} with F1 score: {best_f1_score}')

        if current_binary_f1_score > binary_best_f1_score:
            binary_best_f1_score = current_binary_f1_score
            th.save(model, binary_best_model_file_path)
            print(f'New best binary model and graph saved at epoch {epoch} with F1 score: {binary_best_f1_score}')

# 进行前向传播，获取测试预测
# 将模型移动到设备上（GPU 或 CPU）
best_model = th.load(multiclass_best_model_file_path)
best_model = best_model.to(device)
best_model.eval()
test_pred = best_model(G_test, node_features_test, edge_features_test).to(device)

binary_best_model = th.load(binary_best_model_file_path)
binary_best_model = binary_best_model.to(device)
binary_best_model.eval()
binary_test_pred = binary_best_model(G_test, node_features_test, edge_features_test).to(device)

# 保存test_pred到本地
th.save(test_pred, multiclass_test_pred_file_path)
th.save(test_pred, binary_test_pred_file_path)

# 计算并打印前向传播所花费的时间
elapsed = timeit.default_timer() - start_time
print(str(elapsed) + ' seconds')

# 获取预测标签
test_pred = test_pred.argmax(1)
binary_test_pred = binary_test_pred.argmax(1)

# 将预测结果从 GPU 移动到 CPU，并转换为 numpy 数组
test_pred = test_pred.cpu().detach().numpy()
binary_test_pred = binary_test_pred.cpu().detach().numpy()

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

# 将实际标签和预测标签转换为 "Normal" 或 "Attack"
binary_actual = ["Normal" if i == 0 else "Attack" for i in actual]
binary_test_pred = ["Normal" if i == 0 else "Attack" for i in binary_test_pred]

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

