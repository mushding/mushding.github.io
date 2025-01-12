---
title: >-
  Vision Transformer 演化史: Visual Transformers: Token-based Image Representation
  and Processing for Computer Vision - 使用 visual token 來強化傳統 CNN 的結果
mathjax: true
date: 2021-07-26 14:24:44
tags: Vision Transformer
categories: 電腦視覺整理
---

這是一篇來自 UC Berkeley 的論文，論文提出了基於 Transformer 的一個類似強化的模組 Visual Transformer (visual token)，可以加在任何現有的 Backbone 或是 FPN 上，可以比原架構效果好一些些，重要的是大大減少了參數運算量。

[https://arxiv.org/pdf/2006.03677](https://arxiv.org/pdf/2006.03677)

keywords: Visual Transformer、Tokenizer
<!--more-->

## 1. Introduction

作者發現傳統的 CNN 有以下三大缺點：

1. 圖上的每一個 pixel 重要性都是不一的

在 CNN 的每個 kernel 中的像素，它們的權重 (重要性) 是視為相當的，均匀排列的像素矩陣 (uniformly-arranged pixel arrays)。但這個就會產生一個問題，如果今天的 task 是語意分割，CNN 會把物體、背景視為一樣重要的東西，使我們更難分離出偵測物體與背影

2. 不是每一個圖片都可以表達「整體」

這句話的意思是說，CNN 在處理小物件上非常強，像是直線、角落等等…，但是當這個小組件共同組合成大物件時，例如車子、房子，受限於 CNN 的 kernel size 處理起來並不直覺 (常見的做法有：增加 kernel size、增加深度、放大倍率…)

3. CNN 對於遠距離的關聯性很弱

同樣是受限於 kernel size 的問題，在一張圖片中距離相隔遙遠的兩點，CNN 是做不太到計算兩者之間的相關性的。同樣可以使用增加 kernel size、增加深度來解決，但是付出的代價就是計算量上升。

因此作者提出了 Visual Transformer (VT) 架構 (嗯…這個名字好容易跟 Vision Transformer 搞混阿 XD)，使用 Visual Token 來描述高階圖片的特徵，並且用到了類似 Spatial attention 的概念來生成 Visual Token。接著把 Visual Token 放入 Transformer 中，透過 Transformer 可以找到 Token 與 Token 間的重要性。

這樣子 VT 就可以改進以下三點：

1. 關注到重要的地方，而不像 CNN 一樣，全部視為一樣重要
2. 多了一個 Visual Token 類似多了一個語意編碼 semantic tokens 的資訊來加強結果
3. 使用 Transformer 建立 Token 之間的關系

以上是論文原文的話，我個人的讀後想法是：CNN 在小物件上的偵測效果很強，但是隨著物件的放大，CNN 雖然說可以透過加廣加深來解決這個問題，但是付出的運算量也是非常可觀的。而 Transformer 在處理大物件上之間的關聯很強，因此本論文的作者試著將兩者結合，既有 CNN 小物件的強處，到了中後段改使用 Transformer 進一步分析結果。

## 2. Visual Transformer (VT)

以下細講 Visual Transformer (VT) 的架構流程

![Image](https://i.imgur.com/6cBVuly.png)

給定一張圖片，先對它做 CNN 層層卷積找到 low-level 低階特徵，輸出一個 feature map，接著通過一個 tokenizer 把 feature 轉換成 visual tokens，其中這每一個 visual token 都代表一個 semantic concept。再把 visual token 放進 Transformer 中輸出也是 visual token，而這些 visual token 可以直接當成分類的結果，或是可以再經過一個 Projector 變成語意分割任務。

如果我上述結論：作者先讓 CNN 處理低階特徵，再來用 Transformer 來處理高階特徵。

接著來依序講講

### Tokenizer

**Filter-based Tokenizer**

![Image](https://i.imgur.com/BgG41c0.png)

先上公式：

$$
T=\mathrm{SOFTMAX}_{HW}(XW_A)^TX
$$

feature map $X$ 會先做一個 1x1 conv 從 $HW \cdot C$ 變成 $HW \cdot L$ 得到一個 Spatial attention A ，$XW_A$
接著把結果轉至 $A^T = (XW_A)^T$ 
再與原圖相乘 $(XW_A)^TX$ ，就得到最後的 Visual tokens 了

因為是透過 1x1 conv 來找尋特徵，所以稱為 Filter-based Tokenizer

**Recurrent Tokenizer**

為了加強 Filter-based Tokenizer 的不足，作者又提出了 Recurrent Tokenizer 方法，簡單來說就只是把：第一次生成出來的 Visual tokens 拿來當成第二次生成的依據

![Image](https://i.imgur.com/grv288k.png)

公式如下：

$$
\begin{gathered}
W_R = T_{in}W_{T\rightarrow R} \\
T=\mathrm{SOFTMAX}_{HW}(XW_R)^TX
\end{gathered}
$$

所有有變動的地方就是從 $W_A$ 變為 $W_R$ 了。
首先上一次生成的 Visual Token 會先乘生一個神奇的 $W_{T\rightarrow R}$ ，維度大小為 $W_{T\rightarrow R}\in\R^{c\times c}$ (我真的看不出來這個 $c \times c$ 倒底從哪裡生出來？)
後序步驟與上面一致

### Transformer

與原版 Transformer 有一點點不同，公式如下：

$$
T_{out} = T_{in} + \mathrm{softmax}_L((T_{in}Q)\cdot(T_{in}K)^T)\cdot T_{in}
$$

$$
T_{out} = T_{out}' + \sigma(T_{out}'F_1)F_2
$$

其中 Query 與 Key 互做運算後，乘上的 Value 並沒有經過 1x1 conv 分割，而是乘上整體。
接下來就是 Add & Norm 的部分了

### Projector

如果要把結果進一步成語意分割任務的話，作者認為再經過一步 Projector 效果會比較好，而 Projector 最主要的目的是把 Visual token 轉回用像素的方式來表達，這樣在以像素分割時效果較好。

公式如下：

$$
X_{out} = X_{in} + \mathrm{softmax}_L((X_{in}W_Q)\cdot(X_{in}W_K)^T)\cdot T_{in}
$$

其中 $X_{in}$ 為 CNN 生出的最後一層特徵圖
可以看到 attenion 公式中，$X_{in}$ (Query) 與結果 $T$ (Key) 互相做運算，最後再乘上全部的 $T$ (Value)
也就是說 Projector 的重點是 原 feature map 與 visual token 互做運算的結果

最後把得到的重點特徵加會原圖，就是最後的結果了。

## 3. 用法

這個 Visual Transformer 最強的地方在於，它是一個「模組」，因此可以安插在任何現有的網路模型之中。

### 放在 ResNet 中

把最後一個 Stage 直接改為 Visual Transformer，可看到效果好了一些些，運算量也少了一些些

![Image](https://i.imgur.com/0MFnupv.png)

### 放在 FPN 中

![Image](https://i.imgur.com/KYuX8xA.png)

![Image](https://i.imgur.com/Emxn71Z.png)

## 結論

這是一篇試著把 CNN 與 Transformer 結合的一篇論文，提出了一個基於 Transformer 的「模組」，而可以達到效果好一些些，同時運算量也下降一些些的優勢。(但我個人覺得…這篇論文在 VT 的部分有一些地方沒說清楚…，那個 $c \times c$ 倒底怎麼來的阿…)

## Reference

https://zhuanlan.zhihu.com/p/349315675