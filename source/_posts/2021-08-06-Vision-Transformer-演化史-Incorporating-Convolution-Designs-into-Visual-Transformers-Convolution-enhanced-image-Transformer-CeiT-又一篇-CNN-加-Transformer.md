---
title: >-
  Vision Transformer 演化史: Incorporating Convolution Designs into Visual
  Transformers - Convolution-enhanced image Transformer (CeiT) 又一篇 CNN 加
  Transformer
mathjax: true
date: 2021-08-06 23:42:28
tags: Vision Transformer
categories: 電腦視覺整理
---

Convolution-enhanced image Transformer (CeiT)，與 CvT 的想法相同，都是想要藉助 CNN 的力量來改善 Transformer 的效能，而這兩篇論文提出的時間差不多，基本上思路也差不多，以下會簡單帶過

[https://arxiv.org/pdf/2103.11816.pdf](https://arxiv.org/pdf/2103.11816.pdf)

keywords: CeiT
<!--more-->

## 1. Introduction

這篇作者認為 CNN 最重要的兩大特色就是：Invariance 平移不變性，以及 Locality 局部性，與 CvT 強調的 (local receptive fields 局部感受視野, shared weights 權重共享, spatial subsampling 空間下採樣) 概念相同。因此提出把 CNN 結合 Transformer 來解決以上問題。

網路架構有以下的改進：

1. 提出 Image-to-tokens 的方法，來改善原本 ViT 16x16 patch 的做法
2. 為了強化特徵的提取，CeiT 把 MLP 層的全連接層 (Feed-Forwardnetwork)，換成了 Locally-enhanced Feed-Forward layer，加強 token 之間的關聯性
3. 在 Transformer 最後一步加上了 Layer-wise Class token Attention 進一步提升性能

## 2. 網路架構

### Image-to-tokens

![Image](https://i.imgur.com/hDpjVX4.png)

為了解決 ViT 把圖片依照 patch 16x16 過於粗糙的做法，因此改成先經 CNN 再來分 patch，簡單來說原圖會先做一次卷積，加上一個 BN 再加上一個 Maxpolling，最後再用 ViT 的分 patch 的方法來分塊。公式如下：

$$
x' = \mathrm{I2T(x)} = \mathrm{MaxPool(BN(Conv(x)))}
$$

I2T 利用了 CNN 在取得低階特徵的優勢，來縮小 patch 的大小減少訓練難度

與 CvT 不同的地方：CvT 是在每一個 Stage 都會做一次 CNN 卷積，而 CeiT 只會在網路的最一開始做一次而已

### Locally-enhanced Feed-Forward layer

![Image](https://i.imgur.com/MgIMIcc.png)

作者把原本的 MLP 層中的 FF 換成 LeFF (Locally-enhanced Feed-Forward layer)，加強 token 之間全局特徵提取的能力

1. 首先會先把 input token 分成 patch token 以及 class token
2. class token 不動，而 patch token 則會經過以下步驟
3. 經 Linear Projection 放大維度
4. 再 reshape 成三維圖片
5. 再做一次 Depth-wise 卷積運算
6. reshape 成二維序列
7. 再做一次 Linear Projection
8. 把 class token 加回來

比較特別的是在每一個 Linear Projection 以及 Convolution 之後都會加上一個 BN 以及 GELU

公式如下：

$$
\begin{gathered}
  \mathrm{x_{cls}, x_{patch} = Split(x)}\\
  \mathrm{x_{patch} = GELU(BN(Linear1(x_{patch})))}\\
  \mathrm{x_{patch} = SpatialRestore(x_{patch})}\\
  \mathrm{x_{patch} = GELU(BN(DWConv(x_{patch})))}\\
  \mathrm{x_{patch} = Flatten(x_{patch})}\\
  \mathrm{x_{patch} = GELU(BN(Linear2(x_{patch})))}\\
  \mathrm{x = Cancat(x_{cls},x_{patch})}\\
\end{gathered}
$$

與 CvT 不同的地方：其實這一步與 CvT 基本上差不多，只是 CvT 是作用在 MSA 層上，而 CeiT 是作用在 MLP 層上。以及最重要的，CeiT 在網路中使用了 GELU

#### GELU

關於 GELU 這裡不多做介紹。可以參考以下文章：我自己的大意是，GELU 與 ReLU 很像，都是把值乘上 0 或 1，只是 GELU 會根據當下值的機率來決定要乘 0 還是 1。

而 CeiT 之所以會使用 GELU 是因為之前在 NLP 流行的 GPT-2、BERT 都使用上了 GELU，並且在 語音辨識上取得不錯的成積。嗯…可以理解成 NLP 專用的 Activate funtion 吧哈哈

[https://www.jiqizhixin.com/articles/2019-12-30-4](https://www.jiqizhixin.com/articles/2019-12-30-4)

### Layer-wise Class token Attention

![Image](https://i.imgur.com/8On4Qq8.png)

Layer-wise Class token Attention 是加在整個 Transformer 的最後面的，由圖可以看出，作者提出的這個新的 LCA 層是加在 Encoder 之外。

作者認為隨著網路不斷的加深，希望能在 layer 與 layer 層與層之間加深彼此的關系，因此把每一層的 class token 都拿出來，經過一次 self-attention 得到 Layer-wise 的 attention，也就是每個層的 class token 之間的關系，而最後的 output 也是整個 CeiT 的 output

## 3. Experiments

### 網路架構

![Image](https://i.imgur.com/7RxBVHO.png)

### 實驗一、SOTA 比較

效果沒有比 EfficientNet 來得好，但是運算量及參數使用量比較少

![Image](https://i.imgur.com/XPzxExY.png)

### 實驗二、Transfer Learning 比較

雖然還是沒有超過 EfficientNet ，但在 ImageNet 上的 Transfer Learning 超過了 ViT，證明 CNN + Transformer 是有潛力的

![Image](https://i.imgur.com/3MOfyr3.png)

### 實驗三、I2T 的比較

使用了不同卷積的 kernel size、stride，以及 maxpooling BN，看看哪一個排列組合效果最好：

![Image](https://i.imgur.com/Ps4TVZH.png)

### 實驗四、LeFF 的比較

同樣比較了 kernel size 以及是否使用 BN 來找出最好的排列組合

![Image](https://i.imgur.com/8ZERYDf.png)

## 結論

CeiT 基本與 CvT 想法一模一樣，都是把 CNN 加上了 Transformer 來改善運算量、資料夾大小、效能等等問題。

而 CeiT 我自己認為比較特別的點在於，使用到了 GELU 這個 NLP 才在用的 Activate funtion，以及在最後加上了 LCA，把每一個不同 stage 的 class token 拿出做一個 self attention，找出一個橫跨 Layer 之間的關系。

## Reference

https://zhuanlan.zhihu.com/p/361112935