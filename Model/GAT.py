import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn
from torch.nn import ModuleList

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
top_95_percent_value = np.percentile(valid_corr, 98)

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

# Data 객체 생성
data = Data(x=tensor_data, edge_index=edge_tensor, y=y)

# RandomNodeSplit을 사용하여 데이터셋을 80% 트레인, 20% 밸리데이션으로 분할
split = T.RandomNodeSplit(num_val=0.2)
graph = split(data)

print(graph)  # 분할된 그래프 정보 출력

#device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 데이터와 레이블을 GPU로 이동
graph = graph.to(device)
num_classes = y.unique().size(0)

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = ModuleList([
            GATConv(data.num_node_features, 8, heads=8, dropout=0.6),
            GATConv(8 * 8, 8, heads=8, dropout=0.6),
            GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(x, dim=-1)

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=500):
    
    for epoch in range(1, n_epochs + 1):
        model.train() 
        optimizer.zero_grad() 
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask]) 
        loss.backward() 
        optimizer.step() 

        # 학습 정확도 계산
        train_acc = eval_node_classifier(model, graph, graph.train_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Train Acc: {train_acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval() 
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    
    # 정확도 계산
    acc = int(correct) / int(mask.sum())

    return acc


# 모델 학습
gcn = GAT().to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)

# 학습 완료 후, 검증 데이터에 대한 정확도 측정
val_acc = eval_node_classifier(gcn, graph, graph.val_mask)
print(f'Validation Accuracy: {val_acc:.3f}')
