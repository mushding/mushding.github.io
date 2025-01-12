---
title: >-
  Vision Transformer 演化史: Visual Transformers: Transformer in Transformer - 使用雙層
  Transformer 來重新思考 Patch Embedding
mathjax: true
date: 2021-07-27 12:14:51
tags: Vision Transformer
categories: 電腦視覺整理
---

很多人會覺得 (包括我 XD) ViT 的方法實在太神奇了，直接把圖片表示在 16x16 的字串？！然後竟然還可以 work？這篇論文覺得直接把二維轉換成一維流失了太多空間上的資訊了，包括圖片像素與像素之間的關系，提出了 TNT Transformer in Transformer 架構，希望可以以內外兩層 Transformer 來加強圖片轉序列的可解釋性及可行性。

[https://arxiv.org/pdf/2103.00112](https://arxiv.org/pdf/2103.00112)

keywords: TNT、Transformer in Transformer、word embedding

<!--more-->

## 1. Introduction

在之前的研究中包含 ViT、DeiT 都沒有去探討一個問題：patch embedding 的可行性。倒底這種直接把圖片用 16x16 個區塊來表示，並且直接經過一個 linear transform 的做法可行嗎？這種方法真的是最合適的方法嗎？

作者認為 ViT、DeiT 等直覺 (intuitive) 的做法會有一個大問題：**會乎略掉每個 patch 之間的訊息**，也就是說在 16x16 之一的小區塊，經過了一個 linear 的轉換後會破壞像素與像素之間的關聯性。

因此作者提出了 TNT (Transformer iN Transformer)，**試圖在 patch 內再新增一個 Transformer 來取得 patch 內的訊息**，保留外部的 Transformer 的同時也新增一個內部的 Transformer，利用內外不同視野的獲取資訊，來使網路有更好的效果。

## 2. Approach

整體架構如下：

![Image](https://i.imgur.com/V102VCT.png)

### Patch Embedding & Pixel Embedding

首先依照 ViT、DeiT 的方法把一張 $H\times W\times C$ 的圖片分割成大小為 $p$ 數量為 $n$ 的 patch

$$
\mathcal{X} = [X^1,X^2,...,X^n] \in \mathbb{R}^{n\times p\times p\times c}
$$

接著把得到的 patch 再做一次一模一樣的操作得到更小的 patch，把 $p\times p\times C$ 的圖片分割成大小為 $p'$ 數量為 $m$ 的 patch

$$
\mathcal{Y_0} = [Y^1_0,Y^2_0,...,Y^n_0] \in \mathbb{R}^{n\times p'\times p'\times c}
$$

![Image](https://i.imgur.com/YHc678K.png)

而比較大的 patch 稱為 **Patch Embedding**
比較小的 patch 稱為 **Pixel Embedding**

接著各別不同大小的 Embedding 會經過不同的 Transformer

Patch Embedding 經過 Outer Transformer，負責 patch 之間的全局 (Global) 資訊
Pixel Embedding 經過 Inner Transformer，負責 pixel 之間的局部 (Local) 資訊

![Image](https://i.imgur.com/CWKy2QU.png)

### Outer Transformer & Inner Transformer

Inner Transformer 的公式，先做 MAT 再做 MLP，與 ViT 相同：

$$
\begin{gathered}
  Y'^i_l=Y^i_{l-1} + MSA(LN(Y^i_{l-1}))\\
  Y^i_l=Y'^i_{l-1} + MLP(LN(Y'^i_{l}))
\end{gathered}
$$

Outter Transformer 的公式，與上述差不多：

$$
\begin{gathered}
  X'^i_l=X^i_{l-1} + MSA(LN(X^i_{l-1}))\\
  X^i_l=X'^i_{l-1} + MLP(LN(X'^i_{l}))
\end{gathered}
$$

那兩個不同視野的 Transformer 要怎麼合併資訊呢？作者這邊是使用在進入 Outter Transformer 前 會與 Inner Transformer 的結果 concat 起來。

首先 Inner Transformer 的結果會先 flattern，接著經過一層 linear 層把維度轉換成與 Outter Transformer 相同，再與 Outter Transformer 相加，做為下一時間點的輸入。公式如下：

$$
Z^i_{l-1}=Z^i_{l-1}+Vec(Y^i_{l-1})W_{l-1}+b_{l-1}
$$

既：原 Outter Transformer 加上 flattern 後 乘上 $W$ 轉維度，再加上一個 b 權重值 (這裡不知怎麼多出來的…)

### Positional Encoding

與 ViT 不同，TNT 使用的是 1D 的 Positional Encoding，公式如下：

$$
\mathcal{Z} \leftarrow \mathcal{Z} + E_{patch}
$$

$$
E_{patch} \in \mathbb{R}^{(n+1)\times d}
$$

剛剛的 Patch Embedding & Pixel Embedding 在運算前都分別加上去。

一樣 Patch Positional Encoding 負責全局空間的訊息 (global spatial information)
而 Pixel Positional Encoding 負責局部相對的訊息 (local relative information)

### 運算量分析

看起來 TNT 的運算量是 ViT 的兩部之多，因為整整多做一次 Transformer，但其實不然，如果仔細去分析複雜度 (論文有細詳推論過程這邊不多說)，會發現 Pixel Embedding 的部分因為圖片太小而 (Pixel 的大小遠小於 Patch)，因此複雜度並不會多很多，多一點點而已 (1.09倍) 並沒有想像中的大。

### 網路架構

設計了大小 (B-S) 模型，一律：patch size 設為 16×16，小 patch size 設為 4×4

![Image](https://i.imgur.com/OVAy35n.png)

## 3. Experiment

嗯…不錯呢，超越了 ViT 及 DeiT！

![Image](https://i.imgur.com/ahN4tGk.png)

### 一定要 Positional Encoding 嗎？

作者有試著把兩個 Encoding 都拿掉看看效果有沒有影響，結論是在做 attention 之前的 flattern 步驟，如果沒有位置的話，flattern 後的結果不管怎麼排都沒差。因此實驗也證明加上 Encoding 效果比較好。

![Image](https://i.imgur.com/7wUmjOf.png)

### head 數量

2 或 4 為最佳

![Image](https://i.imgur.com/OTUQ9cf.png)

### 小 patch size 的大小設定

大 patch size 是 16x16，那小的呢？
實驗證明 4x4 為最佳

![Image](https://i.imgur.com/h83UkUz.png)

### 可視化

Patch Embedding 可視化，兩個 Transformer 的結果好處有特徵抓取的能力更強了，比 DeiT 相比，特徵分佈的更寬廣

![Image](https://i.imgur.com/KH3M8tZ.png)

Pixel Embedding 可視化，隨著網路越深越抽象

![Image](https://i.imgur.com/SYveZi7.png)

## 結論

如何把三維圖片表示成二維字串真的是一大難題，也是研究的熱門話題阿，而 TNT 提出了雙重 Transformer 的解法，雖然運算量大了一咪咪，但效果不錯，且有試著往解釋神奇的 16x16 前進了一小步，相信未來一定有更好的做法來解釋 16x16。

## Reference

https://zhuanlan.zhihu.com/p/354913120