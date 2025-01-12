---
title: 你所不知道的 Pytorch 大補包(八)：訓練小技巧 DDP 透過多機多卡來訓練模型
mathjax: false
date: 2022-12-29 00:33:39
tags: Pytorch
categories: Pytorch 大補包
---

DDP 的全文是 Distributed Data Parallel，是一種可以透過多機多卡來訓練模型的一種方法，它的本質上就是一個像 Map-Reduce 的東西，把訓練資料、Gradient、Loss 等資訊平均分配給每一個 GPU，達成多工處理的目的

DDP 也可以就看成，提高 batch-size 來提高網路效果

下面我們直接先來看 code 吧：

keywords: DDP
<!--more-->

```python
################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 使用 DDP 最主要 import 的兩個包
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 網路架構 (Module) ### 
# 隨便設計的模型
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 假設會用到的資料集
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    # 我們要給 DataLoader 提供 DPP 的 sampler，使用下面的程式實現
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # 在 DataLoader 中加入 sampler
    # 這裡的 batch_size 指的是一個 rank 中 (一個程序) 的 batch_size
    # 也就是說總 batch_size 是 batch_size x world_size (總程序數量)
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化模型、數據、各種配置  ####
# 要從外面手動新增 local_rank 參數
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP backend 初使化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl 由 Nvidia 用 C++ 寫的 Map-Reduce 後端

# 準備資料集
trainloader = get_dataset()

# 建立模型
model = ToyModel().to(local_rank)
# 要 Load 預訓練的模型，需要在建立 DDP 模型之前，且只需要在 rank=0 (主要程序) 上 Load 就可以了
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# 建立 DDP 模型 (這一句是精隨 XD)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 要在建立 DDP 模型之後，才能設定 optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 設定 Loss function
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 網路訓練  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # 設定 sampler 的 epoch
    # DistributedSampler 需要利用這個方式統一 shuffle
    # 使每個程序之間的亂數 seed 都是一樣的，使不同程序有相同的 shuffle 效果
    trainloader.sampler.set_epoch(epoch)
    # 後面就與沒有用 DDP 的部份一樣了
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 與原相同，使用 torch.save torch.load 就可以了
    # 要只在 rank=0 上儲存，不然會存到很多遍
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "%d.ckpt" % epoch)


################
## 在 command line 中執行程式
# 使用 torch.distributed.launch 來啟動 DDP 模式
# 使用 CUDA_VISIBLE_DEVICES，來決定使用哪些 GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```

## Ring-Reduce

但是這種有關 Thread 的東西，就不得不請出我們的 Python GIL 啦，Python GIL 是一個全區鎖，可以看成是使 python 多執行緒效果非常差的兇手

可見以下網站更詳細的講解：

[http://cenalulu.github.io/python/gil-in-python/](http://cenalulu.github.io/python/gil-in-python/)

而 DDP 為了減少 Python GIL 的限制，因而使用而 Ring-Reduce 架構來使 GPU 內互相溝通

![](https://i.imgur.com/B3Sslcd.png)

每個執行緒都只會接收來自上一個節點，並且把結果只丟給下一個節點，這種「圓圈圈」的做法可以大大減少互相通訊的複雜度 (如果假設是每個節點相互連接的話)

進一步詳細的做法可以參考下面的知乎大神：

[https://zhuanlan.zhihu.com/p/69797852](https://zhuanlan.zhihu.com/p/69797852)

## 並行計算


![](https://i.imgur.com/drp22sg.png)

一般來說神經網路的並行模式有一下三種：

1. Data Parallelism

   這是最常見的模式，換局話來說就是「增加 Batch-size」

   DP DDP 剛剛講的那些 trick 都是屬於這一種的

2. Model Parallelism

   把模型放在不同 GPU 上，是平行運算 (綠、黃)

   看通訊效率，加速效果可能不明顯

3. Workload Partitioning

   把模型放在不同 GPU 上，是串聯運算 (綠、藍)

   不能加速

## DDP 的一些基本名詞

* group

  * 程序組，一般只有一個組

* world size

  * 表示「全部」的程序總數
  * 例如有 2 個 server ，每一台每面有 2 張 GPU，world size 為 2x2 = 4

  ```python
  # world size 在不同程序中，得到的值都是相同的
  torch.distributed.get_world_size()
  ```

* rank

  * 表示目前的程序編號，0, 1, 2, 3, ...
  * 其中 rank=0 代表 master 程序

  ```python
  # 每個程序有它自己的 rank 編號
  torch.distributed.get_rank()
  ```

* local rank

  * 同樣表示目前的程序編號，0, 1, 2, 3, ...
  * 但特指「一個機器內的 GPU 編號」
  * (以 2 個 server ，每一台每面有 2 張 GPU 為例，rank：0~3，local_rank：0, 1, 0, 1)
  * 目的是在執從 torch.distributed.launch 時，機器會自動去分配對應的 GPU

## DDP 原理

假設我們有 N 張 GPU

* 減少 GIL 的限制
  * 總共 N 張 GPU 就會有 N 個程序被啟動
  * 每一個 GPU 都執行同一個模型，參數的數值一開始也是相同的
* Ring-Reduce 加速
  * 在訓練模型時，使用 Ring-Reduce，彼此交換各自的梯度
  * 藉此來得到所有運行程序中的梯度
* Data Parallelism
  * 把每個程序的梯度平均後，各自做 backpropagation 更新權重值
  * 因為各程序的初始參數、更新梯度是一樣的，所以更新後的參數值也是完全一樣的

## DDP vs Gradient Accumulation

* 上面有提到 DDP 其實也就是「增加 Batch Size」而已
* 而 Gradient Accumulation 也是變像的增加 Batch Size
* 那兩者有什麼差別呢？
* 效能上
  * 在沒有 Buffer 參數 (像是 Batch Normalization) 下，理論效能是一樣的
  * 程序數 8 的 DDP 與 Step 8 的 Gradient Accumulation 是一樣的
  * (因為 Buffer 參數，理論上要每兩步才更新一次，但因是每個 epoch 都會更新的緣故，BN 的分母會有對不上正確數字的問題)
* 效率上
  * DDP 因使用平行化處理
  * 會比 Gradient Accumulation 快超多

## DDP 調用方式

與原本使用 python3 main.py 的使用方不同，需要用 torch.distributed.launch 來啟動訓練

torch.distributed.launch 有幾個參數：

* --nnodes
  * 有多少台機器
* --node_rank
  * 目前是在哪個機器？
* --nproc_per_node
  * 每個機器有多少個程序
* --master_address
  * master (rank=0) 的程序在哪一台 server 上
* --master_port
  * 要用哪一個 port 進行通訊？

單機下的例子：

```bash
# 假設只有一台機器
# 且一台機器內有 8 張 GPU
python3 -m torch.distributed.launch --nproc_per_node 8 main.py
```

多機下的例子：

```bash
# 假設有兩台機器
# 且每一台機器內有 8 張 GPU
# 需每個機器都執行一次程式

# 機器一
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 --master_adderss $address --master_port $port main.py

# 機器二
python3 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 --master_adderss $address --master_port $port main.py
```

如果我們要求只使用機器內特定的 GPU 呢？像是機器一共有 8 張卡，但只使用 4, 5, 6, 7 

```bash
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 main.py
```

## Reference

[DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)

