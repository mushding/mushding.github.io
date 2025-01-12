---
title: >-
  Vision Transformer 演化史: CSWin Transformer: A General Vision Transformer
  Backbone with Cross-Shaped Windows
mathjax: true
date: 2021-12-03 13:22:46
tags: Vision Transformer
categories: 電腦視覺整理
---

論文網址：[https://arxiv.org/pdf/2107.00652.pdf](https://arxiv.org/pdf/2107.00652.pdf)

Swin 原班人馬在 2021 7 月提出更進一步的網路架構 CSWin Transformer，提出全新的 **C**ross-**S**haped **Win**dow self-attention 有著更好的特徵截取能力，以及更少的網路運算量

更提出新的位置資訊架構 LePE (Locally-enhanced Positional Encoding)，相較於原本的絕對位置 (APE) 或是相對位置 (RPE) 有著更好的表現

keywords: CSwin、LePE
<!--more-->

## Introduction

Self-Attention 的運算量過大，這是眾所皆知的事實，因此 Swin Transformer 藉由把 Patch 再切成更小的 Window 嘗試減少運算量，同時為了使 window 與 window 之間有關聯，Swin 把整個流程切成兩步 W-MSA 與 SW-MSA，藉由**兩次**不同位置的 window 來達成像素的關聯

而 CSwin 再進一步減少運算量的同時還加強了截取特徵的能力，使用有別於原本 Self-Attention 的 Cross-Shaped Window Self-Attention 

![Image](https://i.imgur.com/74I4gl5.png)

如上圖，CSwin 分成垂直、水平 Attention 來取得像素間的關聯，且是利用**把 multi head 分成兩半**來達成，一半負責垂直部份，一半負責水平部份。這樣做的好處是可以在**一步**就完成不同 patch 像素間的關聯，而作者後續的實驗也證明 CSwin 相比 Swin 可以在使用更少的層達到相同的效果

上圖 b 則是類似 ViT 的方法全部圖片都做 Self-Attention，c 則是 Swin 的方法，e 與本文的 CSwin 有點類似，不同的點在於 e 是先做水平再做垂直的，與本文利用 head 一次做兩步有些許的差別

## 網路架構

網路架構圖如下圖所示：

![Image](https://i.imgur.com/h6GmDDy.png)

與 Swin 架構類似，首先會經過 convolutional token embedding，也就是利用 7x7 conv stride 4 來得到 W/4 H/4 個 Patch。其實 ViT 也是利用 conv 來達來劃分 Patch 的目的，但是 ViT 的 conv 沒有 overlap，而 CSwin 這邊則有，有 overlap 的效果比沒有要好上一些

網路主架構分為四個 Stage，每個 Stage 會使用 3x3 conv stride 2 像 CNN 一樣不斷的減少圖片大小，同時增加特徵圖數量

本論文最特別的地方提出了 CSwin Self-Attetion，與傳統的 Self-Attetion 有著以下兩點的不同：
 
1. 把 Self-Attention 換成了 Cross-Shaped Windows Self-Attention
2. 為了增強 local inductive bias (局部的歸納偏置能力)，提出了全新的 LePE 架構

### Cross-Shaped Window Self-Attention

![Image](https://i.imgur.com/iSPamsH.png)

為了提高局部像素之間的關系 (增加 Window 的大小)，同時顧及到運算量不要過大 (像 ViT 那樣與圖片大小呈平方關系)，CSWin 所使用的方法是**利用水平及垂直的 stripe window 來做 Self-Attention**

先來看水平的 stripe

每個 window 可表示成 $X$，而 $X$ 的大小定義為 $sw \times W$，$sw$ 代表為水平 window 的寬度，$W$ 即為圖片的總寬度

每張圖片可以分割成相同大小的 $M$ 個 $X$，且每個 $X$ 不重疊，所以 $M=H/sw$

$$
\begin{gathered}
X=[X^1,X^2,...,X^M] \quad \mathrm{where}\quad X^i\in \mathbb{R}^{(sw\times W)\times C}\quad \mathrm{and} \quad M=H/sw
\end{gathered}
$$

同時假設這些特徵來自第 $k$ 個 head

接著把每個 $X$ 也就是每個 window 彼此之間做 Self-Attention。

$$
\begin{gathered}
Y^i_k = \mathrm{Attention}(X^iW^Q_k,X^iW^K_k,X^iW^V_k),\quad \mathrm{and} \quad i=q,...,M\\
W^Q_k,W^K_k,W^V_k \in\mathbb{R}^{C\times d_k}
\end{gathered}
$$

最後就得到的水平 (Horizontal) 方向的 CSwin 了

$$
\mathrm{H-Attention_k}(X)= [Y^1_k,T^2_k,...,T^M_k]
$$

而垂直 (Vertical) 方向也是同理，公式與上面基本一樣，只有 $M$ 的部份改為 $M=W/sw$

把 multi-head 的數量 $K$ 分成兩半，一半給水平，一半給垂直，得到最後下列式子：

$$
\begin{gathered}
\mathrm{CSWin-Attention}(X) = \mathrm{Concat}(\mathrm{head}_1,...,\mathrm{head}_K)W^O
\end{gathered}
$$
$$
\mathrm{where} \quad \mathrm{head}_k =\left\{
  \begin{aligned}
    \mathrm{H-Attention}_k(X) \quad k &= 1,...,K/2\\
    \mathrm{V-Attention}_k(X) \quad k &= K/2+1,...,K
  \end{aligned}
\right.
$$

### 計算複雜度與 sw 的變化

CSwin 的計算複雜度如下：

$$
\Omega(\mathrm{CSWin-Attention}=HWC\times(4C+sw\times H+sw\times W))
$$

詳細推導過程可以參考以下這個網頁：

[https://zhuanlan.zhihu.com/p/388165447](https://zhuanlan.zhihu.com/p/388165447)

而複雜度的結論為：複雜度與 sw 有關，如果 sw 遠小於 HW，則呈一次方關系，如果 sw 大，則呈兩次方關系

因此，CSwin 一共分為四個階段，每當網路越來越深的時候，sw 的值也會隨之變化：**淺層的 sw 比較小，深層的 sw 的比較大** (論文中提出的變化為：[1, 2, 7, 7] 皆為輸入圖片 224 的倍數)

會這麼做的用意是原圖的解析度大，如果 sw 大的話，計算量會非常大，而在 CSwin 中每個 Stage 結束後都會用 conv 來解少圖片的解析度，因此到了深增時圖片解析度相對小，就可以用比較大的 sw 來做計算了

這麼做的第二個優點是淺層的關注度比較偏局部，而深層的關注度就比較全局，這一點與 CNN 非常類似，但與 ViT 的想法相反。

我覺得…自從 Transformer 從關注一個 Patch 到關注一個 Window 後，Transformer 的初始關注並沒像 ViT 的那麼全局了，轉而像 CNN 一樣從局部再慢慢的到全局

### LePE

![](https://i.imgur.com/C1dtNEi.png)

作者比較了 APE (絕對位置)、RPE (相對位置) 整理如上表，APE 是加在 Self-Attention 前，RPE 是加在 Self-Attention 之中

而作者提出的 LePE 如圖最右邊，將位置訊息加到 Value 中，再將結果加到 Self-Attention 的結果中，公式如下：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{SoftMax}(QK^T/\sqrt{d})V+\mathrm{DWConv}(V)
$$

作者提到這邊使用 Depth-wise Conv 的原因有二：

1. 相較於 Conv 計算量較少
2. 位置編碼只會和當前同一張圖周圍有像素有關聯，不會與其它特徵圖之間有關聯

結論來看 CSWin Transformer block 是由一個十字形的 Attention window，以及一個 Depth-wise conv，兩個分支合併而成的

### CSWin Transformer Block

![](https://i.imgur.com/W6cJ8aF.png)

一個 Block 與 ViT 相同，這邊就不再多解釋了

## Experiments

### 網路模型種類

一種分為 4 個不同大小的模型

![](https://i.imgur.com/EuSDNIH.png)


### 相同模型大小比較

在參數量差不多的情況下做比較，發現當網路模型越大，Transformer-based 的效果比 CNN-based 好上一些些

![](https://i.imgur.com/WbonUTn.png)

### ImageNet-1K 分類比較

個人覺得分類的榜快刷不動了…大概也就好那一點點

![](https://i.imgur.com/jZBUDW8.png)

### COCO 偵測比較

偵測的結果主要是和 Swin 來比，可發現效果好上 1.5 個點，好上不少

![](https://i.imgur.com/9nINvYM.png)

### ADE20K 語意分割比較

可發現 CSwin 在分割項目上超強，直接超過了 2 個點以上

![](https://i.imgur.com/5HQaFQr.png)

### 其它一些技巧的相互比較實驗

動態調整 $sw$、同時算平行垂直、網路初期卷積 kernel 重疊、Deep-Narrow

以上四個 Tricks 是 CSwin 效果好的主因

![](https://i.imgur.com/RD5zQgy.png)

## 結論

CSwin 在 Swin 的成功下進一步增加效果且減少運算量，算是 Swin 家族的一個重大優化

提出的 LePE 也很值得讓人研究，倒底如何加入 PE 才是最好的做法呢？

水平垂直平行化處理的觀念也很創新，那是不是可以再把 head 多切分成更多塊呢？

不論如何，雖然分類的榜已經快刷不動了，但看起來 Transformer 的強項是在分割阿

## Reference

### Arxiv
[https://arxiv.org/pdf/2107.00652.pdf](https://arxiv.org/pdf/2107.00652.pdf)

### 知乎大神們
[https://bbs.cvmart.net/articles/5075](https://bbs.cvmart.net/articles/5075)

[https://zhuanlan.zhihu.com/p/388165447](https://zhuanlan.zhihu.com/p/388165447)

