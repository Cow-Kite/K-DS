import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

import time
import torch
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# CSV 파일 경로 설정 및 데이터 로드
file_path = r'D:\128.치매 고위험군 라이프로그\ver1.csv'  # 실제 경로로 교체하세요
df = pd.read_csv(file_path)

file_path = r'D:\128.치매 고위험군 라이프로그\label.csv'  # 실제 경로로 교체하세요
labels_df = pd.read_csv(file_path)
print(labels_df.head(5))
y = torch.tensor(labels_df['label'].values, dtype=torch.long)


# 피어슨 상관계수 행렬 계산
pearson_corr = df.T.corr(method='pearson')

# 피어슨 상관계수 행렬을 Numpy 배열로 변환
corr_array = pearson_corr.values

# 대각선 요소 제외 (자기 자신과의 상관관계는 1이므로)
np.fill_diagonal(corr_array, np.nan)

# 행렬을 1차원 배열로 변환하고 NaN 값을 제외
flattened_corr = corr_array.flatten()
valid_corr = flattened_corr[~np.isnan(flattened_corr)]

# 상위 95%에 해당하는 값 계산
top_95_percent_value = np.percentile(valid_corr, 95)

# 상위 95% 이상인 값들의 인덱스 구하기
edge_indices = np.argwhere(corr_array >= top_95_percent_value)

# 엣지 인덱스를 리스트로 변환
edge_list = [[int(edge[0]), int(edge[1])] for edge in edge_indices]

# edge_list를 PyTorch 텐서로 변환
edge_tensor = torch.tensor(edge_list, dtype=torch.long)
edge_tensor = edge_tensor.T

# 데이터프레임을 Numpy 배열로 변환
numpy_array = df.values

# Numpy 배열을 PyTorch 텐서로 변환
tensor_data = torch.tensor(numpy_array, dtype=torch.float32)

# 노드의 수 계산 (예: y의 길이로부터)
num_nodes = y.size(0)

# 모든 노드를 True로 설정
train_mask = torch.ones(num_nodes, dtype=torch.bool)
test_mask = torch.ones(num_nodes, dtype=torch.bool)

# 결과 출력
print(f"PyTorch 텐서로 변환된 데이터프레임:\n{tensor_data[:5]}")  # 데이터의 첫 5개 행 출력
print(f"PyTorch 텐서로 변환된 엣지 리스트:\n{edge_tensor[:5]}")  # 엣지 리스트의 첫 5개 출력

data = Data(x=tensor_data, edge_index=edge_tensor, y=y, train_mask=train_mask, test_mask=test_mask)
graph = data
print(data)

device = "cpu"
num_classes = y.unique().size(0)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)
        return output

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    
    for epoch in range(1, n_epochs + 1):
        model.train() 
        optimizer.zero_grad() 
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask]) 
        loss.backward() 
        optimizer.step() 

        pred = out.argmax(dim=1) 
        #acc = eval_node_classifier(model, graph, graph.val_mask) 

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval() 
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    #print(pred[mask] + graph.y[mask])

    # 정확도 계산
    acc = int(correct) / int(mask.sum())

    return acc


gcn = GCN().to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)
