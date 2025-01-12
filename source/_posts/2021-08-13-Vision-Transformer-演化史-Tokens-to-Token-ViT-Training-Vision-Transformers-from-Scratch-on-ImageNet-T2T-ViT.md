---
title: >-
  Vision Transformer 演化史: Tokens-to-Token ViT: Training Vision Transformers from
  Scratch on ImageNet - T2T-ViT
mathjax: true
date: 2021-08-13 11:29:02
tags: Vision Transformer
categories: 電腦視覺整理
---

這篇論文發表在 CvT、CeiT 之前，但想要解決的問題是一樣的 (解決分 patch、運算量大等…)。CvT、CeiT 是使用 CNN 來解決問題，而 T2T-ViT 則是使用 Token-to-Tokens 來解決問題。

[https://arxiv.org/pdf/2101.11986.pdf](https://arxiv.org/pdf/2101.11986.pdf)

keywords: T2T-ViT
<!--more-->

## 1. Introduction

T2T-ViT 相要改進 ViT 在 ImageNet 上訓練時不如傳統 CNN 的兩個缺點：

### 1. ViT 分 patch 的方法會使得圖片之間的訊息流失

在 ViT 中分 Patch 的公式算單來說是長以下這樣的：

$$
\begin{gathered}
H\times W\times C \rightarrow N\times (P^2\cdot C) \rightarrow (N, D)\\
\mathrm{where}, N=HW/P^2
\end{gathered}
$$

在原 source code 中是長以下這樣：

```python=
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

可發現它其實就是一個 kernel, stride 皆為 16 的一個卷積運算而已，這樣子的做法無法很有效的去表達 16x16 中的局部特徵訊息

### 2. ViT 的 backbone (self-attention)，在特徵的提取上有點冗餘

這裡作者直接用實驗結果來證明，作者把 ResNet 與 ViT 的其中一層特徵向量取出來做視覺化，如以下：

![Image](https://i.imgur.com/UH57D4h.png)

可發現 ResNet 隨著網路越深 (圖片越往右)，特徵的紋理越來越多樣，但是 ViT 隨著網路越深，基本上特徵圖沒有什麼多樣的地方 (大部分都還是狗)，而且還會有全白全黑的問題 (用紅框的部份)。

實驗可證明 ViT 在特徵截取上的確不如 CNN 來的好

## 2. 網路架構

### 整體架構

![Image](https://i.imgur.com/gdo0RUu.png)

首先原圖會做一次 Unfold 成二維向量，與 ViT 的 patch embedding 不太一樣的是，T2T-ViT 中所有的 Unfold 都有 overlap，增加相關性。

接著經過一層 Transformer，再接回 T2T module，這個步驟重覆兩次。

接著會加上 cls token 以及 PE ，最後放到 Backbone 去

$$
\begin{gathered}
T_i = \mathrm{MLP(MSA(T_i))}\\
T_{i+1} = \mathrm{T2T\_module(T_i)}
\end{gathered}
$$

### Tokens-to-Token module

為了解決以上兩個問題作者設計了一個 Tokens-to-Token module，以下介紹：

![Image](https://i.imgur.com/lDqEniJ.png)

#### 第一步：Restructurization

把二維向量 reshape 成三維，如下公式所示：

$$
\mathbb{R}^{l\times c} \rightarrow \mathbb{R}^{h\times w\times c}
$$

#### 第二步：Soft Split (SS)

剛剛把二維 reshape 成三維，現在我們又要把三維 reshape 回二維，只是做法不太一樣。這一步是為了進一步提取 local information 的。

為了達成可以提取 local information，作者使用了 pytorch 中的 Unfold 函式來達成。特別的是 Unfold 中每個 kernel 是有重疊的，增加 local information。其實原理與一個 Conv 差不多，如以下公式：

$$
\begin{gathered}
(B, C, H, W) \rightarrow (B, Ck^2, HW)\\
k, \mathrm{kernel\_size}
\end{gathered}
$$

### T2T-ViT Backbone

為解決 ViT Backbone 很多特徵是多餘沒用的，T2T-ViT 參考 CNN 的做法，一共試了 5 種做法：

* 參考 DenseNet：使用 Dense 連接
* 參考 Wide-ResNets：Deep-narrow vs. shallow-wide 結構比較
* 參考 SE module：使用 Channel attention 結構
* 參考 ResNeXt：在 attention 中使用更多的 heads
* 參考 GhostNet：使用 Ghost module

經過了大量的實驗後，作者得出使用 CNN 的 Deep-narrow 深窄結構效果最好，可以增加特徵的多樣性

所以作者設談的 T2T backbone 它的 Embedding dimension (二維序列長度) 比較小，同時層數比較多

## 3. Experiments

### 與 ViT 比較

不論在參數量、運算量、效能上，皆比 ViT 好

![Image](https://i.imgur.com/KJGikvO.png)

### 與 ResNet 比較

與 CNN 的對比則當然比較好啦，效能好一些，不過計算量高一些些

![Image](https://i.imgur.com/bt9DepO.png)

### 與 MobileNet 比較

與小小模型比較，在相同參數量的前提下，效能提高，但運算量高一些些

![Image](https://i.imgur.com/8yQ4IwG.png)

### 各種不同 backbone 的比較

可以直接看結論：使用 DN (Deep-Narrow) 深窄結構效果最好

![Image](https://i.imgur.com/end7cto.png)

## 結論

T2T-ViT 是在 2021 1月發表的文章，比 CvT、CeiT 還早，但已經有想要使用 CNN 來結決問題的大方向。整體網路架構印象最深的地方是 T2T 的 Unfold 運算，不知道這樣子的做法是不是真的會比較好…

## Reference

https://zhuanlan.zhihu.com/p/386955720

https://zhuanlan.zhihu.com/p/348055832