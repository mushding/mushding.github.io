---
title: 'Vision Transformer 演化史: SwinIR: Image Restoration Using Swin Transformer'
mathjax: true
date: 2021-12-02 22:38:49
tags: Vision Transformer
categories: 電腦視覺整理
---

論文網址：[https://arxiv.org/pdf/2108.10257.pdf](https://arxiv.org/pdf/2108.10257.pdf)

這是基於 Swin Transformer 應用在 Super Resolution 的研究，網路稱 SwinIR，實驗證明 Backbone 使用 Transformer 也能達到不錯的效果

最後效果甚至成為當時的 SOTA，改進了 0.14∼0.45dB，且參數使用量相較下少了 67% (拜層數不深所賜)

keywords: Swin Transformer、SwinIR
<!--more-->

## Introduction

### CNN vs Transformer

以前 CNN-based 的 SR 網路，重點常常聚焦在 Residual connection 上的改良，以及深層網路的堆疊

代價就是參數使用量偏高

而已 Transformer-based 的 SwinIR，因層數偏少的因素，相同效果下參數使用量明顯小了一些，如下圖：

![Image](https://i.imgur.com/9is3YjZ.png)

除了參數變少外，當然一定要提一下的…就是 Transformer 的 Global Recepitve Filed，當然有了這個資訊一定是補足了 CNN 一些比較不足的地方

### ViT vs SwinT

如果直接把 ViT 拿來做 SR 會發生什麼事？由於 ViT 各 Patch 之間互相獨立，互相不做運算，因此在 patch 邊邊的像素會出現邊界現象

而如果 patch 彼此有 overlaping 的話，運算量會增加

要怎麼在不增加運算量的前提下解決這個問題呢？解答就是 Swin Transformer 所提出的 Shifted windows 方法

### SwinIR

因此作者提出以 SwinT 為基準的 SR 網路

分為三個階段：

1. 淺層特徵提取
2. 深層特徵提取
3. 影像 upsampling 至高解析度 (image reconstruction)

## 網路架構

![Image](https://i.imgur.com/Y7GR1Oi.png)

### 淺層特徵提取

$$
F_0 = H_{SF} (I_{LQ})
$$

$I_{LQ}$ 代表 input 一張 Low Quality 的影像

$H_{SF}$ 代表一個 3x3 conv 負責 Shallow Feature

### 深層特徵提取

$$
F_{DF} = H_{DF}(F_0)
$$

$H_{DF}$ RSTB Block 負責 Deep Feature

### RSTB

RSTB 的全名是 residual Swin Transformer blocks，由 $K$ 個 Swin Transformer 以及一個 3x3 conv 所組成

$$
\begin{gathered}
    F_i = H_{RSTBi}(F_{i-1}), i=1,2,...,K\\
    F_{DF} = H_{CONV}(F_K)
\end{gathered}
$$

![Image](https://i.imgur.com/qOjA61X.png)

STL 代表 Swin Transformer Layer，與原論文架構相同，這邊就不多講了

### image reconstruction

$$
I_{RHQ} = H_{REC}(F_0+F_{DF})
$$

$I_{RHQ}$ 代表 reconstruct high-quality image

$H_{REC}$ 會接兩個參數：淺層特徵與深層特徵，兩個不同特徵一起當 input

而 upsampling 使用的方法則是 pixelshuffle

另外對於一些圖片不需要 upsampling 的應用，像是去噪、去雨…公式改成以下：

$$
I_{RHQ} = H_{SwinIR}(I_{LQ}) + I_{LQ}
$$

### Loss function

損失函數則是簡單的 L1 loss

$$
\mathcal{L} = ||I_{RHQ} - I_{HQ}||_1
$$

## Experiments

### channel 數、RSTB 層數、STL 層數數量實驗

![Image](https://i.imgur.com/6OIFF9o.png)

最後選擇 channel 180 個 (Source code 上好像是 96 個)

RSTB、STL 各 6 層，使得網路相對小

### patch size 的影響、訓練集大小的影響

可發現 patch size 越大，SwinIR 效果越好

![Image](https://i.imgur.com/6XiySR4.png)


### RSTB 中的 residual connection 以及 最後一個 CNN 的選擇

![Image](https://i.imgur.com/TAg6zhx.png)

有 residual 比沒 residual 好、3x3 比 1x1 來得好

普通 3x3 與 inverted-bottlenect 3x3 差不多，後者參數少，效果差一些些 (合理)

### SOTA 比較

![Image](https://i.imgur.com/SXPUFfs.png)

### 一些實驗結果

![Image](https://i.imgur.com/HhoTifa.png)

![Image](https://i.imgur.com/Lo8cuiy.png)

![Image](https://i.imgur.com/12DaV9G.png)

## 結論

Transformer 應用在 SR 上，因著 Swin Transformer 的成功，也應用的非常順利

本篇論文其實沒什麼特別的貢獻，大概就是側面證明了 Swin 的厲害

## Reference

[https://arxiv.org/pdf/2108.10257.pdf](https://arxiv.org/pdf/2108.10257.pdf)

[https://arxiv.org/pdf/2103.14030.pdf](https://arxiv.org/pdf/2103.14030.pdf)

