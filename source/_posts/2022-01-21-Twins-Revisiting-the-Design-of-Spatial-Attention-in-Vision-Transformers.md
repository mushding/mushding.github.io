---
title: 'Twins: Revisiting the Design of Spatial Attention in Vision Transformers'
mathjax: true
date: 2022-01-21 18:28:28
tags: Vision Transformer
categories: 電腦視覺整理
---

在 2021 年 3 月提出 Swin Transformer 後，4 月澳洲 Adelaide 大學提出 Twins-PCPVT 以及 Twins-SVT 兩個新架構來改進 Swin Transformer Backbone 上的一些問題。

本篇論文比較像一個工程報告書，比較 PVT、Swin 以及作者提出的 Twins 之間的優缺點

[https://arxiv.org/abs/2104.13840](https://arxiv.org/abs/2104.13840)

keywords: Twins-PCPVT、Twins-SVT
<!--more-->

## Introduction

作者認為 Swin Transformer 有以下的優缺點：

pros：

* 提出 window 單位，解決了 Transformer 運算量過大的問題

cons：

* shifted window 設計，雖然解決了 windows 之間缺少相關性的問題，但程式其中所使用的 `torch.roll()`，對運算量非常不友好。一些部署優化的規範 (如：ONNX、TensorRT)，並不支援這一指令

因此提出了 Twins-SVT 來改善 Swin 的缺點

另外作者認為 PVT 與 Swin 的網路架構想法相近，皆是在 Transformer 架構中加入多重解析度的概念。

也提出了 Twins-PCPVT 類似技術報告的方法，來使 PVT 的效果更好一些

## 網路架構

### Twins-PCPVT

作者比較 PVT 與 Swin 的網路架構的差別，發現：

* PVT 沒有使用 window 為單位，不同解析度的特徵圖整張做 Self-Attention 運算，在運算量上比 Swin 大
* PVT 中的 Positional Encoding 是使用如同 ViT 中的 APE (Absoute Positional Encoding 絕對位置)，而 Swin 則是使用 RPE (Reletive Positional Encoding 相對位置)

明明兩個網路架構都使用到了多重解析度的概念，那為什麼 PVT 的效果不及 Swin 呢？作者認為問題是出在位置編碼上

因此作者參考了 CPVT 這篇論文所提出的 CPE (Conditional Position Encoding)，並把原本 PVT 的 APE 替換成 CPE。PVT 與 CPVT 兩篇論文相互結合，作者稱這個新的混合方法為 Twins-PCPVT

![Image](https://i.imgur.com/r65k5FV.png)

PEG (Positional Encoding Generator) 為 CPVT 中提出的架構，詳細可以參考我之前寫過的文章：

[Vision Transformer 演化史: Conditional Positional Encodings for Vision Transformers - 可變序列長短的 Positional Encoding](https://mushding.space/2021/07/28/Vision-Transformer-%E6%BC%94%E5%8C%96%E5%8F%B2-Conditional-Positional-Encodings-for-Vision-Transformers-%E5%8F%AF%E8%AE%8A%E5%BA%8F%E5%88%97%E9%95%B7%E7%9F%AD%E7%9A%)

### Twins-SVT

本篇架構目標是把 Swin 改進，Twins-SVT 與 Swin 相同，同樣使用 window 作為一個計算單位，但是不使用 shifted window 來解決各 windos 不相關的問題，而是提出了 SSSA (Spatially Separable Self-Attention) 架構

#### SSSA (Spatially Separable Self-Attention)

模仿 CNN 中的 Separable-Convolution 將卷積運算分成 Depth-wise + Point-wise，目的為了減少運算量

而 SSSA 也把 Self-Attention 分成兩個步驟：LSA (Locally-grouped self-attention) 與 GSA (Global sub-sampled attention)

#### LSA

LSA 簡單說與 Swin 一模一樣，就是把張圖片分成 window 們，每一個 Self-Attention 只會發生在一個 window 內

設定 window 大小為 $mn$，則 window 個數為 

$$
k_1k_2=\frac{H}{m}\frac{W}{n}
$$

#### GSA

與 Shifted window 不同，作者使用的方法更直接一些，就是在 LSA 後直接再做一次整張圖的 Self-Attention

也可以說先做一次 LSA 代表局部資訊，再做 GSA 代表全局資訊，兩兩結合就是全部圖片的相關性了

但如果是這樣，計算量就又一樣了，又是整張圖片去做運算。

作者的解法為：把每個 window 中選一個最重要的值，代表這個 window 的主要特徵，於是我們可以拼出一個 $mn$ 大小的新 window。我們將這個新 window 看成是 Key 一般，去對原圖中每一個 window 做 Self-Attention

換句話說：LSA 是 window 中自己與自己計算相關性，而 GSA 是 window 中自己與「全局重要特徵」計算相關性

詳細流程可參考下圖：

![Image](https://i.imgur.com/eqJueht.png)

每一個 Transformer Block 的流程為：`LSA -> FFN -> GSA -> FFN`

## Experiments

### ImageNet 分類上的結果

![Image](https://i.imgur.com/DEYK7L4.png)

Twins-PCPVT 實驗證實，PVT 架構是很有潛力的，將其中 APE 替換成 CPE 後，效果比原 PVT 好 1.4%。

同時在效果與 Swin 差不多的前提下，Twins-PCPVT 運算量比 Swin 少了 18%，而 Twins-SVT 更是少了 35%

### ADE20K 分割上的結果

為了公平比較，分割方法皆為使用 UpperNet

![Image](https://i.imgur.com/9nUPKrx.png)

Twins-PCPVT 比 PVT 高 4.3% mIoU

Twins-SVT 比 Swin 高大約 1.7% mIoU

## 結論

這篇論文提出了兩個新架構，一是改進 PVT 的位置資訊編碼，加入了 CPVT，側向證明位置編碼的重要性

二是改進 Swin `torch.roll()` 的問題，在效果參數量不變的前提下，運算量又再一步下降

## Reference

[Twins 論文](https://arxiv.org/abs/2104.13840)

[PVT 論文](https://arxiv.org/abs/2102.12122)

[CPVT 論文](https://arxiv.org/abs/2102.10882)

[我的 PVT 筆記](https://mushding.space/2021/08/17/Vision-Transformer-%E6%BC%94%E5%8C%96%E5%8F%B2-Pyramid-Vision-Transformer-A-Versatile-Backbone-for-Dense-Predictionwithout-Convolutions-%E6%8A%8A%E9%87%91%E5%AD%97%E5%A1%94%E7%B6%B2%E8%B7%AF%E6%87%89%E7%94%A8%E5%9C%A8-Transformer/)

[我的 CPVT 筆記](https://mushding.space/2021/07/28/Vision-Transformer-%E6%BC%94%E5%8C%96%E5%8F%B2-Conditional-Positional-Encodings-for-Vision-Transformers-%E5%8F%AF%E8%AE%8A%E5%BA%8F%E5%88%97%E9%95%B7%E7%9F%AD%E7%9A%84-Positional-Encoding/)