---
title: >-
  STA: Spatial-Temporal Attention for Large-Scale Video-based Person
  Re-Identification 影片空間時間注意力
mathjax: true
date: 2022-07-18 12:05:43
tags: 
  - 3D image
  - Attention
categories: 電腦視覺整理
---

本篇論文雖然目的是在做 Re-ID 任務，但是「手勢辨識」「影片中的行人辨識」，這類任務與我現在的題目都有相似之處：輸入資料並非單純的二維影像。要怎麼利用多一個「時間、空間」維來完成任務，是這些題目所要解決的重點

[https://arxiv.org/abs/1811.04129](https://arxiv.org/abs/1811.04129)

keywords: Re-ID、Spatial-Temporal Attention (STA)
<!--more-->

Person re-identification (Re-ID) 行人辨識，所要解決的任務是在一個影片做行人偵測，這個任務最困難的地方在輸入不是一張影像，而是由許多幀數構成的影片。

這種帶有時間維度的資料，在處理上有幾個困難的點：

1. 最直接的做法是把影片每一幀當成影像放進網路訊練，最後用一個 maxpooling 統整所有結果。缺點是遇到 occlusions 遮擋時效果會很不好
2. maxpooling 還有一個缺點是，會破壞影片時間軸的資訊，之後有任務在模型中加入 attention 區別各個幀的重要性，但是本篇作者認為，現有的 attention 方法都沒辨很好的做到：找到最重要的那一幀，以及大部份 attention 因全連接層的關系輸入長度是要固定的。

作者基於上面兩個理由提出 **Spatial-Temporal Attention (STA)** 空間-時間注意力機制

![image-20220717151834910](https://i.imgur.com/lKF4lp7.png)

## 網路架構

### Backbone

作者使用 ResNet-50 作為網路主幹，小小修改的地方在把最後一層的 average pooling 及全連接層去掉，改接入到作者自己設計的 STA 中

### Spatial-Temporal Attention Model

作者認為：現有的 Attention 機制有以下三個問題

1. 因經過更多的 Conv 層，代表著更多的計算
2. 不同空間中的 attention 彼此是互相獨立的，這會使得前景的目標行人網路關注的地方並非完整的人，而是零散的區域
3. 空間、時間注意力也是彼此獨立的，權重不共享

因此設計出 Spatial-Temporal Attention (STA) 架構，流程如下：

首先一影片 $V$ 由許多幀 $I$ 構成，$V={I_1,...,I_N}$，會先在其中隨機取 n 個幀做運算得出輸入 $f_n$ ，對 $f_n$ 做 $l_2$ 正規化，再用平方和來除以它。作者利用平方和的 $l_2$ 對選出的 n 個幀做空間注意力。
$$
g_n(h,w)=\frac{||\sum^{d=D}_{d=1}f_n(h,w,d)^2||_\mathbb{2}}{\sum^{H,W}_{h,w}||\sum^{d=D}_{d=1}f_n(h,w,d)^2||_\mathbb{2}}
$$
對每個幀做完 $l_2$ 後，再將每個幀水平的切成 $K$ 個相同大小的塊
$$
\begin{aligned}
g_n = [g_{n,1},...,g_{n,K}]\\
f_n = [f_{n,1},...,f_{n,K}]
\end{aligned}
$$
接著對「每一塊」都做一次 $l_1$ 正規化，作者說這樣可以達到區塊中的空間注意力
$$
s_{n,k}=\sum_{i,j}||g_{n,k}(i,j)||_\mathbb{1}
$$
計算完一幀內的「全局」「局部」空間注意力後，再合併剛剛的 n 幀，對所有 n 個空間注意力結果再做 $l_1$ 而非複雜的 Conv 層，作者說得到的結果可看成時間注意力分數
$$
S(n,k)=\frac{s_{n,k}}{\sum_n||s_{n,k}||_\mathbb{1}}
$$
最後我們就可以得到一個二維 $n\times k$ 的注意力矩陣，n -> 幀數、k -> 塊數

![image-20220718001928912](https://i.imgur.com/vNdPVZs.png)



### Inter-Frame Regularization

作者為了避免網路過度依賴單一區塊的權重，設計了 Inter-Frame Regularization 來正規化彼此差異

作者會在做完第一次 $l_2$ 得到幀的空間注意力後，隨機選出兩個幀，彼此做 Frobenius Norm，公式如下：(我個人覺得就是 pixel-wise 的像素平方差而已…)
$$
\begin{align}
Reg&=||g_i=g_j||_F\\
&=\sqrt{\sum^H_{h=1}\sum^W_{w=1}|g_i(h,w)-g_j(h,w)|^2}
\end{align}
$$
為了不要兩幀差異太大，所以這個 Reg 值越小越好，作者並且加到 Loss 裡面變成：($\lambda$ 為控制比例超參數)
$$
\min(\mathcal{L}_{total}+\lambda Reg)
$$

### Feature Fusion 合併方法

做完上述 Attention 的得到一個 $n\times k$ 的 $s_{n,k}$ 分數(注意力)矩陣，先會把特徵圖也分為 K 塊，接著做兩種不同的合併方法

1. 對所有幀中的塊，直接選擇分數最高的那一塊。例如 n=4 ，我們要在 k 中選一個最大的值為結果，一共會選 4 次，稱為 Pick max index
2. 對每一個幀、每一個塊做 element-wise 的乘法，把分數加權在特徵圖中，稱為 Weighted sum
3. 最後把 Pick max index 、 Weighted sum 兩矩陣 concat 起來

最後就是一連串的：GAP -> FC -> FC -> 分類…

### Loss

作者使用 triplet loss + softmax 的混合 loss

triplet loss 更詳細的介紹可以參考下面網站：[triplet loss 损失函数](https://zhuanlan.zhihu.com/p/171627918)

## 實驗結果

作者有做了一些 Ablation 實驗來證實他們的方法是有用的，先來看修改效果圖

![image-20220718005006372](https://i.imgur.com/EqYKmMG.png)

對於隨機在影片中選 N 幀的實驗

![image-20220718005041227](https://i.imgur.com/UtHPsk4.png)

對於一幀中切 K 個塊的實驗，作者發現切太多塊反而不好

![image-20220718005050352](https://i.imgur.com/3LWUCn7.png)