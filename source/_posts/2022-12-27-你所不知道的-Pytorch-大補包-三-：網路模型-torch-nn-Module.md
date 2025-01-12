---
title: 你所不知道的 Pytorch 大補包(三)：網路模型 torch.nn.Module
mathjax: false
date: 2022-12-27 14:16:02
tags: Pytorch
categories: Pytorch 大補包
---

設定好訓練 SOP、設定好自定義的資料集後，接下來我們要來設計自己的網路模型，會使用到 Pytorch 中 torch.nn.Module 這個物件

keywords: torch.nn.Module
<!--more-->

## torch.nn.Module

* 有三種創建 module 的方法
  * 繼承 nn.module 的普通方法
  * nn.sequential
  * nn.ModuleList

### nn.Module

* 基本款
* 有一個 \_\_init\_\_ 設定各個神經層的設定，命名好後在下一個 forward 來使用，通常是放「需要學習的的層」
* 另一個 forward 來設定各個層的連接以及參數設定，通常放的是「不需要學習的層」，像 activate function
* 在 pytorch 中 backward 會自動實現，使用的是 Autogard
* 以及在 pytorch 中 nn.Module 只支持 mini-batch 的輸入方式 N x C x H x W (1 x 3 x 128 x 128)

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(conv1(x))
        return F.relu(conv2(x))
```

### nn.Sequential

* nn.Sequential 的模組是按照順序排列的，需要確保輸出大小與輸入大小一致

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class net_seq(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.seq = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
        )      
    def forward(self, x):
        return self.seq(x)
net_seq = net_seq()
print(net_seq)

#net_seq(
#  (seq): Sequential(
#    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (1): ReLU()
#    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (3): ReLU()
#  )
#)
```

* nn.Sequential 中也採用 OrderedDict 来指定 module 的名字，而非 index (0, 1, 2, ...)

```python
from collections import OrderedDict

class net_seq(nn.Module):
    def __init__(self):
        super(net_seq, self).__init__()
        self.seq = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
    def forward(self, x):
        return self.seq(x)
net_seq = net_seq()
print(net_seq)

#net_seq(
#  (seq): Sequential(
#    (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (relu1): ReLU()
#    (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (relu2): ReLU()
#  )
#)
```

### nn.ModuleList

* nn.ModuleList 也是一個存不同 module 的 list，可任意得把 nn.Module 加到 list 中
* 與 python 的 list 操作相同，可以 extend append...
* 但它會自動把 module 的 parameters 自動加入網路中

```python
class net_modlist(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = nn.ModuleList([
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU()
        ])

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

net_modlist = net_modlist()
print(net_modlist)
#net_modlist(
#  (modlist): ModuleList(
#    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#    (1): ReLU()
#    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#    (3): ReLU()
#  )
#)

for param in net_modlist.parameters():
    print(type(param.data), param.size())
#<class 'torch.Tensor'> torch.Size([20, 1, 5, 5])
#<class 'torch.Tensor'> torch.Size([20])
#<class 'torch.Tensor'> torch.Size([64, 20, 5, 5])
#<class 'torch.Tensor'> torch.Size([64])
```

### nn.Sequential vs nn.ModuleList

* nn.Sequential 內部自動實現 forward 所以不用再一個一個加，但 nn.ModuleList 沒有，任需一個一個加入
* 且 nn.Module 中沒有一定的順序，可用 index 來指定

```python
# 不在 nn.Module 的方法
seq = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)
print(seq)
# Sequential(
#   (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (3): ReLU()
# )

# nn.Sequential
# 繼承 nn.Module 的方法，就要寫出 forward 
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.seq = nn.Sequential(
             nn.Conv2d(1,20,5),
             nn.ReLU(),
             nn.Conv2d(20,64,5),
             nn.ReLU()
        )      
    def forward(self, x):
        return self.seq(x)
 
# nn.ModuleList 的方法
class net2(nn.Module):
   def __init__(self):
      super(net2, self).__init__()
      self.modlist = nn.ModuleList([
          nn.Conv2d(1, 20, 5),
          nn.ReLU(),
          nn.Conv2d(20, 64, 5),
          nn.ReLU()
      ])

   # 注意：只能按照下面利用 for 的方式
   def forward(self, x):
       for m in self.modlist:
           x = m(x)
       return x
```

### Reference

* https://blog.csdn.net/u012609509/article/details/81203436
* https://zhuanlan.zhihu.com/p/75206669