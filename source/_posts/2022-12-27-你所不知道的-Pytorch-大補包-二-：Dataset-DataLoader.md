---
title: 你所不知道的 Pytorch 大補包(二)：Dataset DataLoader
mathjax: false
date: 2022-12-27 13:59:29
tags: Pytorch
categories: Pytorch 大補包
---

如果今天開發需求不是像 Mnist 這樣，別人已經幫你準備好的資料集，而是自己的影像資料集，那要怎麼放進 DataLoader 裡面訓練呢？

keywords: DataLoader
<!--more-->

## Dataset DataLoader

* 使用繼承 Dataset 可以自定義 data，再放進 DataLoader 中
* 一個 Dataset 繼承後要 override 的 function 如下：

```python
from torch.utils.data.dataset import Dataset

class customDataset(Dataset):
    def __init__(self):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
```

* \_\_init\_\_ 負責初使化 path、img list、label list、transform
* \_\_getitem\_\_ 負責讀取圖片，並做 transform，回傳 img 以及 label
    * 回傳值也可以回傳不只兩個，也可根據需求回傳想要的資料
* \_\_len\_\_ 回傳 imgs 的長度 len(self.imgs)