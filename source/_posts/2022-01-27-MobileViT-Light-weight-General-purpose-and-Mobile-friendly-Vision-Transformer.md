---
title: >-
  MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision
  Transformer
mathjax: true
date: 2022-01-27 15:14:00
tags: Vision Transformer
categories: 電腦視覺整理
---

2021 10 月 Apple 基於 Transformer 提出 MobileViT 架構，其主要目的是把 Transformer 輕量化，以達到能在移動設備上部署。

本篇最主要的方法為結合 MobileNet 與 Transformer，得到效果好、效率也不錯的架構

[https://arxiv.org/abs/2110.02178](https://arxiv.org/abs/2110.02178)

keywords: MobileViT
<!--more-->

## Introduction

作者開頭就提到，CNN 有 Inductive bias，ViT 則沒有，因此需要更多的資料或是用 L2Norm 來達到類似效果，在 ViT 使用 Self-Attention 運算量大以及資料量需求大下，自然是沒辨法輕易的部署到移動設備上了。

而本篇作者借鏡了 2019 年的 MobileNet v3 架構，提出 MobileViT 架構。MobileViT 試著結合了 CNN 與 Transformer 各自的優點，達成在相同低參數量下效果比 CNN 好

MobileViT 在 ImageNet 上 top-1 準確率為 78.4%，參數使用量為 6M，比 MobileNet v3 高出 3.2%，比 DeiT 高出 6.2%。也在偵測 MS-COCO 上比 MobileNet v3 高出 5.7%

## 網路架構

MobileViT 有三個目標：Light-weight 輕量化、Genral-purpose 歸納能力強、Low latency 低延遲

而作者在設計架構時認為：CNN 的特色為：有 Inductive Bias 歸納能力強，Transformer 的特色為：可以關注到全局的資訊，但計算量大，因此採用：「以 CNN 為主，把 Transformer 融合到 CNN 架構中」

首先來上整體架構圖：

![image-20220127160654393](https://i.imgur.com/W2Q0mH6.png)

整體架構圖不難發現，粉紅色的地方為 MV2 (MobileNet v2) 塊，是 CNN 的部份，而且占網路大多數，而綠色才是 MobileViT Transformer 的部份，只占了三格而已

借著 MobileNet v2 Block 減少圖片解析度，獲得多重解析度，再經過 MobileNet v2 達到類似 Attention 的效果

### Unfold -> Matmul -> Fold

在介紹一個 MobileViT Block 內部架構前，先來了解一下 CNN 的運算，通常我們在 pytorch 中設計 CNN 會使用到許多卷積運算 (Conv)

```python
torch.nn.functional.conv2d(inp, w)
```

但根據這篇作者以及 [pytorch 官網](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)上的敘述，一個 Conv 可以拆分成三個部份：Unfold、Matmul、Fold

```python
inp = torch.randn(1, 3, 10, 12)
w = torch.randn(32, 3, 4, 5)
# Unfold
inp_unf = torch.nn.functional.unfold(inp, (4, 5))
# Matmul
out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
# Fold
out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
```

上面那三個東西是什麼呢？下面這張圖可以清楚的表示：圖片來源：[https://blog.csdn.net/u010087338/article/details/113666140](https://blog.csdn.net/u010087338/article/details/113666140)

![image-20220127162459106](https://i.imgur.com/NsWeQJX.png)

我們一般認識的卷積運算就是一張圖片 $\mathbb{R}^{w_0\times h_0 \times c_0}$ 對一個 kernel  $\mathbb{R}^{w_k\times h_k\times c_1}$ 做矩陣乘法後，得到結果 $\mathbb{R}^{w_1\times h_1\times c_1}$ 的結程。但是我們也可以手動的設計上面這一系列步驟。

首先是 Unfold，它可以將輸入「切成」對應 kernel 大小的塊，並把塊「轉換維度」至序列。程式如下：

```python
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
```

可以發現跟 Conv 很像，它也有 kernel_size、stride …等等參數設定，但與 Conv 最大的不同在於，Unfold 只有「切」而已，不負責「運算」。

可以參考上圖中下部份，假設原圖 1x3x10x12 (Batch, Channel, H, W)，kernel 大小 32x3x4x5 (Channel_out, Channel_in, H, W)，在 stride 為 1 下，我們可以「切」出 (10-4+1)x(12-5+1) = 56 個 3x4x5 塊，再把這個塊「轉化維度」至序列得到 1x56x(3x4x5) = 1x56x60

再來是 Matmul 也就是矩陣乘法，把得到的 1x56x60 乘上 kernel (3x4x5)x32 = 60x32，最後會得到 1x56x32 的結果

再來是 Fold，這一步的用意是把序列「轉換維度」回塊，把 1x56x32 轉回成 1x32x7x8，當然也可以直接用 view 直接設定維度來達成

根據 pytorch 官網，CNN 與 Unfold 三部曲的運算是等價的。而本篇 MobileViT 正是利用這個特性，把原本的 CNN 拆成三個步驟，並且把中間的 Matmul 核心運算層，更改為 Transformer 運算。就是這麼剛好，中間那層是一個序列維度的資料，正好適合放進 Transformer 中。


### MobileViT Block

先上圖

![image-20220127160530959](https://i.imgur.com/GX6A0uO.png)

流程為：

1. 首先會做一個 nxn conv 運算 (論文 n=3) 得到局部特徵
2. 再來做一個 1x1 conv 放大特徵圖數量
3. 接著進行：Unfold -> Transformer -> Fold 得到全局特徵
4. 再用一個 1x1 conv 回到原特徵圖數量
5. 用一個 shortcut 把原輸入與剛剛經 Transformer 的結果相加
6. 最後用一個 nxn conv 調整回原圖大小，使得輸入與輸出維度不變

理論上「單層」 MobileViT 的運算複雜度為 $O(N^2Pd)$ 而 ViT 的則是 $O(N^2d)$，看起來運算量反而變高了，但是作者解釋，MobileViT 因有 1 2 步的 CNN 得到局部特徵，加上模仿 CNN 的 Unfold Fold 架構，使得 MobileViT 有更強的 Inductive Bias 能力，在網路設計上可以使用效少的層數得到相同的效果。

以 ViT based 的 DeiT 為對照組，DeiT 需要 L=12, d=192，而 MobileViT L={2, 4, 3}, d={96, 120, 144} 均少於 DeiT

### MobileViT vs ViT

ViT 網路中有一步 Patch Embedding 實質上就是在把一影像，分成一個 Patch 一個 Patch 彼此不 overlap 的序列，而 MobileViT 套用了 CNN stride 的概念，每個 Patch 之間是會 overlap 的，並且彼此間距為 stride 1

但是這樣看起來 MobileViT 分 Patch 的數量比 ViT 多上不少，運算量應該會更大，但 MobileViT 藉由優秀的特徵提取，可以比 ViT 少了非常多層，變向減少計算量

另外因 MobileViT 結合了 CNN 與 Transformer 的優點，可以達到全域局部特徵都可觀察到的特色，如下圖所示，中心紅色點會達距離的藍色點計算 (Transformer)，藍色點也會和周邊其它的點計算 (CNN) 

![image-20220129003704991](https://i.imgur.com/89weffQ.png)

## Experiments

### 網路家族

作者設計了三個不同大小的網路，特別的是，網路是設計的越來越小

![image-20220127172148824](https://i.imgur.com/X8NtFGR.png)

以及三種不同大小網路的效果

![image-20220129012458883](https://i.imgur.com/uRodGGI.png)

### 參數量 vs 分類效果

作者與 Transformer 做比較，不清楚為什麼沒有與 Swin 作比較

![image-20220129012108102](https://i.imgur.com/pUnohgN.png)

### 偵測上的結果

![image-20220129012312539](https://i.imgur.com/FgcgUeb.png)

### 分割上的結果

![image-20220129012434838](https://i.imgur.com/WLRikce.png)

## 結論

MobileViT 又是一篇整合了 CNN 與 Transformer 的論文，比較創新的地方在是以「減少運算量為目標」

網路架構主要還是以 MobileNet 為主，以 Transformer 為輔，並且利用 Unfold -> Matmul -> Fold 的方法巧妙融合 CNN 與 Transformer。這種方法使得與 MobileNet 在相同參數下效果好上了不少

## Reference

[網路上參考的筆記1](https://aijishu.com/a/1060000000243736)

[網路上參考的筆記2](https://blog.csdn.net/u014546828/article/details/120741293)

[图解卷积计算原理与pytorch中fold和unfold函数的使用 (圖) ](https://blog.csdn.net/u010087338/article/details/113666140)

[pytorch手动实现滑动窗口操作，论fold和unfold函数的使用 (解說)](https://blog.csdn.net/LoseInVain/article/details/88139435)

[pytorch transpose() 和 permute()](https://blog.csdn.net/xinjieyuan/article/details/105232802)

[pytorch view()](https://blog.csdn.net/york1996/article/details/81949843)