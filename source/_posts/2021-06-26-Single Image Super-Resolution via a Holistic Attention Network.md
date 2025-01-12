---
title: Single Image Super-Resolution via a Holistic Attention Network
date: 2021-06-26 11:30:30
tags:
  - super resolution
categories: 論文
description: ""
mathjax: true
---

在 super resolution 超高解析度影像問題中，此篇作者認為 LR 至 HR 小細節常常會「平滑化」的原因是，在超深的網路中，各個 layer 之間的資訊並不流通，因而提出了 HAN 架構。

keywords: HAN, LAM, CSAM

<!--more-->

Single Image Super-Resolution via a Holistic Attention Network
===

## Abstract

作者認為 channel attention 把每一個 channel 都分別來開單獨計算，而乎略掉了各 channel 之間的關聯性。

為了解決這個問題，作者提出了 HAN (Holistic attention 全局 attention)，由LAM 如 CSAM 所組成

LAM (Layer attention module)，可以去找各 layer 間的垂直關系

CSAM (Channel-Spatial attention module)，嗯…就是 Spatial attention ? 去找各特徵圖上重要的像素自

## Introduction

* Single image super-resolution (SISR) 就是指單一影像變高解析度
  * 給定一個 LR low-resolution 生成一張 HR high resolution 影像，以上問題程為 SR super-resolution 問題

* SRCNN 則是 SR 領域的開山始祖
  * 現今大部份成功的 SR model 都是建立在 CNN 上，且使用很深的網路以及 Residual 
  * 超深的網路好處是，在尋找 LR 與 HR 之間複雜的對應關系非常厲害，而多虧了 Residual 的幫忙，太深的網路才不會發生梯度消失的問題
* 作者發現在 LR 圖上的細節部份，像素間常常會變平滑掉，作者認為是因為乎略掉中間特徵層之間的關系所導致
  * 雖然在有些地方用上了 channel attention 但還是乎略掉 feature 與 feature 之間的關系
  * channel attention 不能計算出各 layer 之間的權重，尤其是在淺層網路中的資訊很容易因網路深度而慢慢消失，雖然在設計中會有一個 long skip connetion 使淺層資訊得以流動到下層去，但這會使重要的下層資訊與上層資訊權重相同 (越深的網路應該越重要才對)
* 作者提出了 HAN Holistic attention network
  * 包含了 LAM 以及 CSAM
  * LAM 在尋找 multi-scale layers 之間的關系
  * CSAM 則找 channel spatial 之間的關系

## Related Work

* 作者說 SR 領域有兩種做法
  * 一是傳統演算法
  * 二是使用 CNN
* SRCNN -> DRCN -> DRRN -> LapSR ....

## HAN

<img src="https://i.imgur.com/dvcu3qi.png" alt="image-20210625135046099"  />

作者 backbone 的部份使用的是 RCAN，RCAN 的特色就是使用到了 RIR (Residual in Residual) 一共包含了兩個 skip connection 一個 long skip 一個 short ，目的就是為了能使各 layer 之間的訊息能更有效的流動，不會因為深度太深的問題導致梯度消失…，以及在 RG (Residual Group) 裡加上了 CA (channel attention) ，嗯…這個 attention 有沒有幫助嗎…是有到一點點啦

### 網路架構

整個網路架構如下：
與 RCAN 相同，有兩個 skip connection ，不同的地方在於，在每個 RG 的 output 層拉出了一條線連接到 LAM 去，去尋找各 Layer 的重要性，有用的 layer 會被加強，多餘的則會被壓制，而 CSAM 的部份作者只有做最後一層，是在效果與正確率所做出的選擇 (當然可以每一層都做啦…就很慢就是了)

https://zhuanlan.zhihu.com/p/65469586

![image-20210625161718604](https://i.imgur.com/TiuFecI.png)

最後把 LAM CSAM 與 long skip connection 三個相加，再經由一個 Upsample 層，這邊使用的是 sub-pixel conv，又稱作 pixel shuffle，如果要將原圖放大 3 倍，我們會先需要生出 3^2 個特徵圖，全部是經過 conv 轉換，最後把這些特徵圖按照順序放到原 pixel 中 (從 1 個像素變成 9 個像素)

![image-20210625145232838](https://i.imgur.com/2XUg1us.png)

### loss funtion

Loss function 的部份為了與 RCAN 做比較，與原論文同樣是使用 L1 loss，把原圖 (SR) 與生成的高解析圖 (HR) 像素相減求平均

### LAM

以下介紹 LAM
![image-20210625150435548](https://i.imgur.com/c4oUpQk.png)

LAM 的想法與 self-attention 有些類似，一樣特徵圖分為三分，兩分做 correlation ，得出的結果做線性加權，只是乘出來的 feature 是 NxN ，這個就是本篇論文最大的特色，是找出一個 layer 之間的 correlation matrix，以下是用數學式子來表達

$\delta$ 為 softmax
$\varphi$ 為 reshape
T 為轉至

$$
\begin{gather}
w_{j,i} = \delta (\varphi(FG)_i \cdot (\varphi(FG))^T_i)\\
i, j=1,2,...N,\\
\end{gather}
$$

結果會用線性加權的方式回原圖，最後與 short cut 來的原圖相加，作者多設計了一個 $\alpha$ 原始為 0 ，是通過機器自己去學習出來的，也可以代表一個 layer 的重要性

$$
F_{L_j}=\alpha\sum^N_{i=1}w_{i,j}FG+FG_j
$$


### CSAM

<img src="https://i.imgur.com/bUIMvHo.png" alt="image-20210625155307457" style="zoom:67%;" />

與傳統的方法不同，作者為了增加 channel 與 spatial 之間的相關性，直接把特徵圖做 3 維卷積，直接把 channel spatial 看成一個大整體，最後與 self-attention 相同，與自己做 element-wise product ，最後加上原圖得到最後結果，作者認為使用 3 線卷積可以使 CSAM 學到 inter-channel 還有 intra-channel 之間的關系，也就是層與層，與 spatial 的綜合關系


