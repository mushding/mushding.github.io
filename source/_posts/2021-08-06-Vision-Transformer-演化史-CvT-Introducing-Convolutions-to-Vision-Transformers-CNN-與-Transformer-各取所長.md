---
title: >-
  Vision Transformer 演化史: CvT: Introducing Convolutions to Vision Transformers -
  CNN 與 Transformer 各取所長
mathjax: true
date: 2021-08-06 10:41:10
tags: Vision Transformer
categories: 電腦視覺整理
---

作者提出了新架構：Convolutional vision Transformer (CvT)，試著把 CNN 與 Transformer 做結合，並各取所長。

CvT 同時擁有了 CNN 的優點 (local receptive fields 局部感受視野, shared weights 權重共享, spatial subsampling 空間下採樣)

以及 Transformer 的優點 (dynamic
attention 動態的注意力機制, global context fusion 更關注全局訊息的整合, better generalization 更好的歸化能力)

[https://arxiv.org/pdf/2103.15808.pdf](https://arxiv.org/pdf/2103.15808.pdf)

keywords: CvT
<!--more-->

## 1. Introduction

我們已經知道 CNN 在局部的空間內提取特徵的能力非常強，並且藉著不斷的 downsample 來把圖片越縮越小、特徵向量越來越長，使得 CNN 關注到的特徵越來越複雜。而相反的 Trasnformer 更在乎的是全局的關系，藉由整個圖片的 attention 來取得每個 pixel 與每個 pixel 之間的關系。

在之前介紹 self attention 的文章中也有提到，CNN 可以看成是 self attention 的一種特例，因此也可以來解釋全局與局部的關系。

![Image](https://i.imgur.com/HY2RA43.png)

於是作者在這篇論文中提出 CvT，除了試著把 CNN 與 Transformer 各取所長，建立出一個效果更好的模型之外，也拜 CNN 所賜改進 ViT 訓練資料集過大的問題，同時在執行的效率上也有不錯的降低。

下圖為作者比較 ViT T2T TNT PVT DeiT 等等新模型的效能：

![Image](https://i.imgur.com/6kdnMTl.png)

## 2. 網路架構

![Image](https://i.imgur.com/vsBhvUE.png)

作者提出了兩個新的模組：**Convolutional Token Embedding** 以及 **Convolutional Projection**，輸入的圖片會依序經過這兩個步驟，如 CNN 一樣會不斷的把特徵圖大小縮小，同時增加 channel 特徵圖的數量，在最後一個 Stage 才加上 cls token 做為分類的輸出，以上就是 CvT overall 的架構，接下來細講兩個新模組的做法：

### Convolutional Token Embedding

![Image](https://i.imgur.com/oOSrjYQ.png)

首先我們先做一個正常的 Conv 卷積，kernel size 為 $s$ ，使得維度上的變化如下列公式：

$$
\begin{gathered}
x_{i-1} \in \mathbb{R}^{H_{i-1}\times W_{i-1}\times C_{i-1}}\\
f(x_{i-1}) \in \mathbb{R}^{H\times W\times C}
\end{gathered}
$$

接著再經過一個 reshape，把卷積出來的三維圖片，轉換維度至二維序列

$$
\begin{gathered}
f(x_{i-1}) \in \mathbb{R}^{H\times W \times C}\\
H_iW_i\times C_i
\end{gathered}
$$

最後再經過一層 layer normalization

到目前為止就是 Convolutional Token Embedding 全部的架構了，而 Convolutional Token Embedding 架構則是在模擬 CNN 會把圖片大小 ($HW$) 不斷的減少同時增加特徵 ($C$) 的數量，只是最後我們會把三維的結果轉換成二維序列，因此上面的步驟也可以想成：序列的長度會越來越短，同時序列的特徵數會越來越多。

藉由這個模擬 CNN 的方法，可以使用二維的 (patch token) 會學習到更複雜的特徵。

而與原本 ViT 的 Patch Embedding 不同的是，Patch Embedding 是把圖片使用 16x16 來表示成 token，而 Convolutional Token Embedding 則是使用卷積運算來變成 token

### Convolutional Projection

![Image](https://i.imgur.com/crlPDW1.png)

接著我們把 Convolutional Token Embedding 做完的二維序列放到 Convolutional Projection 中進行下一個步驟，而 Convolutional Projection 架構如上同所示

為什麼叫做 Convolutional Projection 呢？其實這個名詞是從 ViT 中的 Linear Projection 而來的，在 ViT 中我們為了做 self attention 於是把輸入序列 (patch token) 經過三個 Linear Projection (線性轉換) 得到三個不同的新序列，各有對應的新名稱 (query key value)。在原本 ViT 中的做法就只是單純的使用不同的線性組合來達成而已。而在 CvT 中作者改用 Conv 卷積的方法來實作。如下圖：

![Image](https://i.imgur.com/AJyXO58.png)

而具體 Convolutional Projection 的方法為：

先將 Convolutional Token Embedding 的結果 reshape 成回三維，接著做 Depthwise-separable Convolution，得到三種不同的 token map 分別對應 (Query Key Value)

具體流程公式如下：先經一個 Depth wise Conv 以及一個 Batch Norm，最後再經 Point wise Conv

$$
\mathrm{Depth \ wise \ Conv2d\rightarrow BatchNorm2d \rightarrow Point \ wise \ Conv2d}
$$

而 Depthwise-separable Convolution 是由 Depth-wise Conv 和 Point-wise Conv 所組成的，如下圖所示：

Depth-wise Conv：

![Image](https://i.imgur.com/IWs8Hp1.png)

Point-wise Conv：

![Image](https://i.imgur.com/Oeu5jat.png)

Depthwise-separable Convolution 是普通的卷積運算的子集合，線性組合的數量比較少，因此在執行上速度比較快，但是效果可能差一些些。

這邊特別注意在卷積運算時加上了 zero padding，這篇論文使用到了 CVPR 的概念，也就是使用 zero padding 來取代 positional encoding

其餘的部份皆與原版 ViT 的 Encoder 相同

### 在效率上更進一步

作者提出 Convolutional Projection 後又更進一步減少網路運算量，作者把生成的 Key 和 Value 的卷積運算改成 stride 2，使得出來的 Key 和 Value 比原本的做法大小少 4 倍，整體的運算量也同樣少了 4 倍，但根據作者的實驗，網路的效能不會下降太多

![Image](https://i.imgur.com/6gQDZGJ.png)

## 3. Experiments

### 網路架構

設計了三種不同大小的網路，數字代表使用了多少 Transformer Block

![Image](https://i.imgur.com/Ga9TISk.png)

### 與 SOTA 的比較

![Image](https://i.imgur.com/7qwl5sx.png)

### 與 Transfer Learning 的比較

![Image](https://i.imgur.com/ojWf8MX.png)

### 實驗一、位置編碼的影響

CvT 中並沒有使用位置編碼，而是使用 zero padding，作者設計了一系列的實驗來看看哪一種方法效果最好，以及 zero padding 是否有給 CvT 位置的訊息。

發現 CvT 特別加上了位置訊息效果不會變更好，效果反而是差不多，證明了 zero padding 的功效

![Image](https://i.imgur.com/6JU4Xxi.png)

### 實驗二、Convolutional Token Embedding 的影響

作者比較了 ViT 16x16 的 Patch Embedding 以及 Convolutional Token Embedding。發現不做位置資訊的 Convolutional Token Embedding 效果最好，其次是做位置資訊的 Patch Embedding

![Image](https://i.imgur.com/PLHKeOf.png)

### 實驗三、Convolutional Projection 的 Stride 1 Stride 2

究竟把 Key Value 的大小縮小 4 倍對效能影響有多大呢？可看到運算量少 1.5 倍，但是效果只少一些些

![Image](https://i.imgur.com/ulfsKV5.png)

### 實驗四、Convolutional Projection 的影響

實驗證明把全部的 Linear Projection 換成 Convolutional Projection 效果最好，證明了 Convolutional Projection 是個有用的測略

![Image](https://i.imgur.com/D9U4SAH.png)

## 結論

CvT 嘗試把 CNN 與 Transformer 結合，各取一點好處來使效果更好外，也有試著往運算量更少的方向進前。

比較特別的兩個點是，使用 zero padding 來當作位置資訊，以及把 cls token 放到最後一個階段才加上去 (原文沒有特別著墨在這裡，不知道這麼做的用意是…？)

總之新增的兩個模組都把 Transformer 往 CNN 的地方又更像了一點

## Reference 

https://zhuanlan.zhihu.com/p/361112935