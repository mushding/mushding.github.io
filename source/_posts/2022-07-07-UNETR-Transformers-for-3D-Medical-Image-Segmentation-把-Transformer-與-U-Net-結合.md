---
title: >-
  UNETR: Transformers for 3D Medical Image Segmentation: 把 Transformer 與 U-Net
  結合
mathjax: true
date: 2022-07-07 13:52:32
tags: 
  - 3D image
  - Segmentation
categories: 電腦視覺整理
---

而本篇作者將 Transformer 架構與 U-Net 架構融合，提出混合架構 **UNE**t **TR**ansformers (UNETR)，其中最重要的特色與 V-Net 相同，UNETR 的輸入同樣是三維的 volumetric (3D) medical image

keywords: UNETR、volumetric medical image
<!--more-->

## Abstract

在影像分割任務上 FCNNs (Fully Convolutional Neural Networks) 也就是全部都是卷積的架構取得了相當不錯的成績，其中又以 U-Net 效果提升最為顯著。不過卷積雖然有很好的 Inductive bias，可以很有效的去學習局部注意力，但在全局上；例如距離很遠的兩像素；效果就不是很好。

在 2020 年 Google 將 Transformer 架構轉移到影像處理領域上後，引入 Self-Attention，靠著 Self-Attention 全局的特色，卷積不夠全局的詬病得以解決。而本篇作者將 Transformer 架構與 U-Net 架構融合，提出混合架構 **UNE**t **TR**ansformers (UNETR)，其中最重要的特色與 V-Net 相同，UNETR 的輸入同樣是三維的 volumetric (3D) medical image

## Introduction

作者在這邊快速介紹了一下影像分割的進展：

首先是 U-Net 的提出，U-Net 的 downsampling-upsampling 先提取特徵再從特徵中回歸原圖，這種作法在當時取得了巨大的成功。再來是因卷積的 Long-range dependencey 被 localized receptive field 限制著了，後續有人提出了加入 atrous convolutional layers 來加大 receptive field

後在因 Transformer 在 NLP 界大放異彩，以及 ViT 實驗的提出，Transformer 應用在 CV 界上似乎是個可行的方法，後續也有大量的論文在針對這個做研究。

而作者在搭上了一輛順風車，提出以 Transformer 為基礎的 3D U-Net 分割網路，有其以下三個特色

1. UNETR 可以直接使用 3D volumetric data 當輸入
2. UNETR 的 Encoder 使用 Transformer 架構，並加入了多層的 skip-connection 可以把不多層的特徵圖融合在一起，達成類似 FPN 的效果
3. UNETR 可以直接把 3D volumetric data 切成不同的 patch 放進 Transformer，不需經過任何卷積

## 網路架構

先直接上架構圖

![image-20220702120847005](https://i.imgur.com/U5VCY7v.png)

網路架構分為三個部分：資料前處理、encoder、decoder，整體網路與 U-Net 想法相同，皆是使用多個特徵截取器，並且引入 FPN 的相法，將不同層數的特徵圖那來一個一個做 upsampling，最後全部特徵圖做成一樣，用一個 concat 全部加起來，就是最後的結果。

資料前處理及 Encoder 的部份主要是參考了 ViT 的做法，同樣經過了一些「經典」步驟：

![image-20220703002701164](https://i.imgur.com/qsEIpRY.png)

### 資料前處理

1. 切分 Patches。將網路輸入的 3D 影像 $x \in \mathbb{R}^{H\times W\times D\times C}$ ，切成一塊一塊的 Patch。這裡作者拓展二維影像的邏輯，將三維影像 $(H, W, D)$ 看為一張影像的解析度，而 $C$ 為特徵圖數，並超參數 $P$ 代表 Patch 的大小。維度變化見下式：
   $$
   \begin{gathered}
   x \in \mathbb{R}^{(H\times W\times D)\times C} \rightarrow x_v\in\mathbb{R}^{N\times (P^3\cdot C)}
   \end{gathered}
   $$
   我們把 $H\times W\times D\times C$ 的影像，依照一個 Patch 為一正方形 $P\times P\times P$，將原影像切成 $N$ 個特徵圖維度為 $P^3 \times C$  Patch 的一維序列，表示為：$N\times (P^3\cdot C)$ 。其中 $N=(H\times W\times D)/P^3$

2. Patch Embedding。接著會做一個 Linear layer，將一維序列的特徵維度改為固定的超參數 $K$。維度變化如下：
   $$
   \begin{gathered}
   x_v\in\mathbb{R}^{N\times (P^3\cdot C)} \rightarrow x_v\in\mathbb{R}^{N\times K}
   \end{gathered}
   $$

3. Positional Embedding。由於不管在二維影像或三維空間中，前面有 reshape 破壞影像結構的動作，所以這裡要加上位置資訊，確保網路在學習的時候是有序的，而不會錯亂彼此的相對位置，變成無序的像素集合。Positional Embedding 維度為 $x_v\in\mathbb{R}^{N\times K}$，加在 Patch Embedding 之後。整體網路前處理的公式如下 (與 ViT 相同)、公式中的 $\mathrm{E}$ 代表 Linear layer：
   $$
   \mathrm{z}_0=[\mathrm{x}_v^1\mathrm{E};\mathrm{x}_v^2\mathrm{E};...;\mathrm{x}_v^N\mathrm{E}]+\mathrm{E}_{pos}
   $$
   值得注意的是，在 UNETR 本篇論文中所引用的 ViT 架構並未加入 class token (cls token)，作者說這是因為分割網路後面會有 upsampling 來處理，因此不需要有分類的結果

### Encoder

這裡 Encoder 與 ViT 就一模一樣了，一樣是由兩個模組組成：multi-head self-attention (MSA) 及 multilayer perceptron (MLP)。小小不一樣的地方是，UNETR 重疊了 12 層 Transformer。公式如下：
$$
\begin{gathered}
\mathrm{z}'_i=\mathrm{MSA}(\mathrm{Norm}(\mathrm{z}_{i-1}))+\mathrm{z}_{i-1},\quad i=1...L,\\
\mathrm{z}_i=\mathrm{MLP}(\mathrm{Norm}(\mathrm{z}'_i))+\mathrm{z}'_i,\quad i=1...L,\\
\end{gathered}
$$
Norm 是做 Layer Norm，MLP 層中間會有 activate function GELU

self-attention 也會分 qkv，也有做一個 softmax 規一化數值，其中 $K$ 為 q 或 k 的一維長度，用來當作一個平衡 qk 乘積的除數因子，再經過一個 softmax 平滑化 feature map，方便訓練。接著再乘上 v，得到 self-attention 最後的結果。
$$
\begin{gathered}
A=\mathrm{Softmax}(\frac{qk^T}{\sqrt{K_h}})\\
SA(\mathrm{z})=Av
\end{gathered}
$$
接著經過一個全連接層 MSA
$$
\mathrm{MSA}(z) = [\mathrm{SA}_1(z);\mathrm{SA}_2(z);...;\mathrm{SA}_n(z)]\mathrm{W}_{msa}
$$
![image-20220703011929113](https://i.imgur.com/CSZTaDR.png)

### Decoder

藉由 U-Net 的起發，本架構同樣會在第 (3, 6, 9, 12) 層拉出不同層數的特徵圖，藉以達到類似 FPN 多重解析度的功能，而各階的維度變化如下：由一維序列乘上 Patch Embedding 特徵數，變為三維空間乘上 Patch Embedding 特徵數
$$
\frac{H\times W\times D}{P^3}\times K \rightarrow\frac{H}{P}\times\frac{W}{P}\times\frac{D}{P}\times K
$$
接著會經過許多 3x3x3 卷積做 deconvolution，把 Patch 的大小一步步放大，同時特徵圖數也一步步縮小。換句話說，作者作者例用 deconvolution 作者類似 swin transformer 中的「合併 window」，把深層的特徵圖一步步回覆成原輸入影像大小

最後用一個 1x1x1 卷積把特徵圖變成目標分類數量的特徵圖數，再接上一個 softmax 把值距離放大，就可以對每個像素做分類任務得到最後的分割結果。



### Loss Function

Loss 的部份作者是使用 Dice Loss 加 Cross-entropy Loss 多任務 Loss 來達成，式子如下：前一項為 Dice Loss 後一項為 Cross-entropy Loss
$$
\mathcal{L}(G,Y)=1-\frac{2}{J}\sum^J_{j=1}\frac{\sum^I_{i=1}G_{i,j}Y_{i,j}}{\sum^I_{i=1}G^2_{i,j}+\sum^I_{i=1}Y^2_{i,j}}+\frac{1}{I}\sum^I_{i=1}\sum^J_{j=1}G_{i,j}\log Y_{i,j}
$$
Dice Loss 詳解。Dice Loss 是從 V-Net 這篇論文所提出來的想法，它是從 Dice coefficient 改編而來的，是一種計算集合相似度的函數，公式表示如下：
$$
s=\frac{2|X\bigcap Y|}{|X|+|Y|}
$$
其中 $|X\bigcap Y|$ 代表；$|X|$ 和 $|Y|$ 分别表示 X 和 Y 的元素個數。 其中，分子中的系數為 2，是因为分母重複計算了 X 和 Y 之間的共同元素的原因，Dice Coefficient 值越大代表兩集合越相似

而如果我們要表示成 Loss 勢比要「越小越好」，有兩種做法，一、直接加負號，二、1 - Dice Coefficient，第一種做法會是負的 Loss 看起來很怪，因此比較人使用第二種，同時值也會落在 0~1 之間，也就是：
$$
d=1-\frac{2|X\bigcap Y|}{|X|+|Y|}
$$
為什麼要使用 Dice Loss？Dice Loss 尤其應用在分割任務上特別多，為什麼不使用一般的 Cross-entropy 就好了呢？原提出論文 V-Net 作者給了一個解釋：在醫學影像中分割目標通常都極小一塊，例如腫瘤，這個特性造成網路訓練資料正負樣本不均，使得既使網路全猜負樣本也會有非常高的正確率。而由於 Cross-entropy 是「每一個像素都會參與計算」，去算出所有像素的 Loss 總合，加大了正負樣本不均的問題。作者提出的 Dice Loss 由於只會與「目標集合」做運算，可以省下許多與負樣本的計算誤差，改善正負樣不均的問題。

但是因 Dice Loss 的 Backpropagation 式子較為複雜，原式子與其一次微分：其中 p 為預測輸出、t 為 GT 輸出
$$
f'(\frac{2pt}{p^2+t^2})dp\rightarrow \frac{2t^2}{(p+t)^2}
$$
當在極端狀態下，當 p 與 t 都超小時，Loss 無限大，相較於 Cross-entropy 一次微分做 Backpropagation，Dice Loss 不太好訓練，這會使得網路不好收斂。所這 UNETR 這篇作者採用兩個都來的做法



## 實驗結果

以下簡單貼一些實驗結果：

BTCV 醫學資料集上的結果

![image-20220704013205692](https://i.imgur.com/5QY087k.png)

MSD dataset 上的結果

![image-20220704013244526](https://i.imgur.com/4G8Q33R.png)

最終效果視覺圖：

<img src="https://i.imgur.com/kns3IEf.png" alt="image-20220704013310139" style="zoom:50%;" />

一些 Ablation 實驗，作者倒是有特別強調他們的 Inference Time 特別小

![image-20220704013440354](https://i.imgur.com/3gSBLfq.png)

## 結論

這篇特別之處在二：一、直接使用 volumetric 當作網路 input；二、使用 Transformer 模仿 U-Net，如果真的照作者說的：在參量數量運算量上升的情況下，Inference Time 依舊低是真的話，那這篇論文可以參考一下

## Reference 

### CSDN 筆記

[[深度学习论文笔记]UNETR: Transformers for 3D Medical Image Segmentation](https://blog.csdn.net/weixin_49627776/article/details/123831261)

[Transformer论文阅读（三）：UNETR: Transformers for 3D Medical Image Segmentation](https://blog.csdn.net/qq_38296005/article/details/119830386)

### Dice Loss

[图像分割中的Dice Loss](https://blog.csdn.net/longshaonihaoa/article/details/111824916)

[医学图像分割之 Dice Loss (大推詳細！)](https://www.aiuai.cn/aifarm1159.html)

[01.医学影像分割LOSS](https://zhuanlan.zhihu.com/p/362935363)



