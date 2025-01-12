---
title: >-
  Global-Local Temporal Representations For Video Person Re-Identification - 融合
  Dilated 與 Self-Attention 的空間時間注意力
mathjax: true
date: 2022-07-18 21:32:08
tags: 
  - 3D image
  - Attention
---

本篇論文目標同樣為：在一影片序列中，找出不同幀影像間的注意力，來區分重要與不重要的特徵幀，藉此來強化網路的結果

[https://arxiv.org/abs/1908.10049](https://arxiv.org/abs/1908.10049)

keywords: Dilated Convolution、Self-Attention、GLTR
<!--more-->

## Introduction

作者認為在一個影片序列中有兩個重要特徵：short-term temporal 短期的關連性，目標在相鄰幀中找到相似的行人；long-term temporal 長期的關連性，目標在兩較遠的幀中找出關連，使得可以解決行人遮擋或影片雜訊等問題

為了達成上述目標，作者提出了 Global-Local Temporal Representation (GLTR) 架構，其中包含了兩個子架構，Dilated Temporal Pyramid (DTP) 架構來達成 short-term temporal 短期的關連性； Temporal Self-Attention (TSA) 架構來達成 long-term temporal 長期的關連性，藉著結合：Dilated Conv 以及 Self-Attention 作者在結果上取得了不錯的成績 (MARS 87.02% Rank-1 Accuracy)

作者提到在這之前有人使用 3D Conv 的方法來解決影片資料的問題，作者認為這樣子的方法有幾個缺點：運算時間大、沒有很有邏輯的去分析空間中的注意力

## 網路架構

### Backbone

作者使用 ResNet-50 作為主架構，先將影片拆分出所有的幀，將二維的影像 $H\times W\times d$ 先經骨幹網路學習，再把結果 reshape 成類似三維的 $H\times W\times d\times T$  

![image-20220718143905285](https://i.imgur.com/kII3QqO.png)

詳細的做法為：一幀影像大小為 $H\times W\times d$ ，一共有 N 個幀 $N \times H\times W\times d$ ，再加上 Batch，最後再 reshape 一下得到 $BN\times D\times H\times W$ 的輸入表示，這個維度可以理解為把 Batch 與幀數視為相同一個維度，Batch 假設是 10，影片假設有 10 幀，則一次放進網路的二維影像總數就是 10x10 = 100 張。最後再把維度 reshape 回原 $B\times N\times D\times H\times W$。利用這個方法就不會因多一個維度需要 3D Conv 了。

### Dilated Temporal Pyramid Convolution

作者在架構中引入 Dilated Convolution 擴張卷積，利用其 rate $r$ 的特色，可以在不改變解釋度、不增加運算量的前提下，增加網路的 receptive field 視野。當 $r$ 越大除了可看為視野越大外也可理解為「兩相鄰幀時間隔離變遠」，越大越 long-term

同時也引入了 FPN 金字塔網路的概念，設計不同的 rate 最後用 concat 融合在一起，也就是把 short-term 與 long-term 合併特徵，使得網路有更豐富的資訊

### Temporal Self Attention

將剛剛得到不同視野 (時間長短) 的特徵圖做 Self-Attention，作者設計的很剛好，FPN 的金字塔層數是 3 剛好對應 Self-Attention 要切成 QKV 三份。三個不同視野 (時間長短) 的特徵圖彼此做重要度分析，最後經一 average pooling 得到最後的結果

## 實驗結果

DTP TSA 的一些 Ablation study，可發現兩個都加上效果最好

![image-20220718212848428](https://i.imgur.com/maZMfUA.png)

SOTA 表，其中 STA 為上一篇的論文架構名稱

![image-20220718213016057](https://i.imgur.com/FLP2FaR.png)