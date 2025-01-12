---
title: LightGCN pytorch 原始碼筆記
mathjax: true
date: 2022-04-28 11:23:08
tags: 
    - Vision Transformer
    - Source Code
categories: 電腦視覺整理
---

研究所社群媒體探勘的期末作業，分析 PTT 的推薦系統，使用 LightGCN 作為主網路

keywords: LightGCN
<!--more-->

### 程式簡介
使用 MovieLens (small) 資料集。由 9,000 個電影及 600 個使用者建立出 100,000 個評價 (edge) 

### import data
這一步在把所有會用到的套件 import 進來，一共四個部份。雜七雜八套件、sklearn 分資料集、Pytorch、Pytorch Geometric
```python
# import required modules
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import one line split data set
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

# import Pytorch Geometric
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
```

### 資料集前處理

一共有四個處理：下載資料、讀取 node 資料、讀取 edge 資料、分 Training、Testing、Validation 資料、轉換為 SparseTensor

```python
# download the dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

# 定義資料位置
movie_path = './ml-latest-small/movies.csv'
rating_path = './ml-latest-small/ratings.csv'
```

```python
# load user and movie nodes
def load_node_csv(path, index_col):
    """Loads csv containing node information

    Args:
        path (str): path to csv file
        index_col (str): column name of index column

    Returns:
        dict: mapping of csv row to node id
    """
    # 把 user item 的 node id 給讀出來
    # (1) 先用 panda 讀取 csv 檔，且把 index_col 設為目標欄位 (指定為 index 的用意)
    # (2) 用 df.index 讀取 index 資料，再用 .unique() 使得 index 唯一 (不重覆)
    # (3) 建立一個 mapping 為 {原 user item id: 從 0 開始的編碼}
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping

# index_col 為欄位名稱
user_mapping = load_node_csv(rating_path, index_col='userId')
movie_mapping = load_node_csv(movie_path, index_col='movieId')
```

```python
# load edges between users and movies
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    """Loads csv containing edges between users and items

    Args:
        path (str): path to csv file
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    df = pd.read_csv(path)
    edge_index = None
    # 把每一個 user item id 做一對一對應
    # 轉換為從 0 開始的編碼，edge 的起點：src、終點：dst
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    
    # (1) 利用 rating 的高低作為 edge_attr (0.5 ~ 5)
    # (2) 把 numpy 格式轉為 pytorch tensor
    # (3) 把 list 的維度從 (1, n) 變為 (n, 1) 
    # (4) 且轉換為 long 整數型態
    # (4) 如果 rating 超過 4 才會計算
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    # 合併 src dst 為 COO 格式
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    # 從 python list 轉為 torch tensor 格式
    return torch.tensor(edge_index)


edge_index = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    link_index_col='rating',
    rating_threshold=4,
)
```

```python
# split the edges of the graph using a 80/10/10 train/validation/test split
num_users, num_movies = len(user_mapping), len(movie_mapping)

# 做一個 edge 的「編碼」表，用來做 Traning Testing Validation 資料集打亂對應的
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]

# 使用 sklearn 中一個很好用的套件，可以只使用「一行」程式完為分資料集的工作
# https://clay-atlas.com/blog/2019/12/13/machine-learning-scikit-learn-train-test-split-function/

# 隨機取樣 edge 的「id」，再依 80/10/10 的比例劃分
# 先分 80/20，再分 20 中的 50/50
train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, random_state=1)
val_indices, test_indices = train_test_split(
    test_indices, test_size=0.5, random_state=1)

# 用 [:, [id]] 可以保留仍意位置上的 edge，且回傳成一個 list (神奇)
train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]
```

```python
# convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

# 因記憶體效能原因，把 COO 換成稀疏矩陣會好多
# 使用 SparseTensor 這個包把 COO 的 [2, n] ，轉為 [len(u) + len(i), len(u) + len(i)] 的方型矩陣
train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
```
![Image](https://i.imgur.com/7eaavGC.png)
轉換為最右的表示法

```python
# function which random samples a mini-batch of positive and negative samples

# 因為推薦系統的任務是：edge prediction
# 所以我們要用 transductive 的方式來切分資料集
# 也就是說 Traning Testing Validation 三個資料集，彼此的邊都不一樣，但是三個加在一起等於原 Graph 的邊
def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """

    # 為了使後續計算 BPR loss (有相關的邊分數高、沒相關的邊分數低)
    # 所以我們這邊要先手動生出一些「沒相關」的邊來
    # 使用 Pytorch Geometric 中的 structured_negative_sampling 函式
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices
```
![Image](https://i.imgur.com/CMQIQUJ.png)


### LightGCN Model
```python
# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        # 加入 l = 0，這是整個網路唯一個可學習參數
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        # 初始化 Embedding
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        # 使用 gcn_norm 來簡化 A=DAD 的運算過程
        # 且加入了 self_loops
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        # 這一步是在實作 E_0
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        # 從中間 node 往外 hop K 個步，並計算 propagation (Wx+b)
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        # 把最後不同 K 層的結果，用 mean Aggregation0
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        # 把 user item 的特徵向量分開
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)

model = LightGCN(num_users, num_movies)
```
![Image](https://i.imgur.com/BiT5LYW.png)
![Image](https://i.imgur.com/UGp6lJU.png)

```python
def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss
```
$$
\begin{equation}
L_{BPR} = -\sum_{u = 1}^M \sum_{i \in N_u} \sum_{j \notin N_u} \ln{\sigma(\hat{y}_{ui} - \hat{y}_{uj})} + \lambda ||E^{(0)}||^2 
\end{equation}
$$
### Reference

[sklearn train_test_split 使用方法](https://clay-atlas.com/blog/2019/12/13/machine-learning-scikit-learn-train-test-split-function/)