---
title: >-
  Vision Transformer 演化史: Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows - 打破各項 SOTA 的新網路
mathjax: true
date: 2021-10-07 12:41:08
tags: Vision Transformer
categories: 電腦視覺整理
---

微軟提出 **S**hifted **Win**dows，簡稱 Swin Transformer，目的是要解決 Transformer 在處理文本與處理影像差異的問題。然而效果卻出奇的好，甚至達到各項領域的 SOTA，在未來的幾篇論文介紹中也會繼續以 Swin 做為出發點。

[https://arxiv.org/pdf/2103.14030.pdf](https://arxiv.org/pdf/2103.14030.pdf)

keywords: Swin Transformer、Shifted Windows
<!--more-->

## 1. Abstract

作者提出了 **S**hifted **win**dows 網路架構 (**Swin** Transformer)，通過這個「移動視窗」架構可以使原本 Transformer 的架構有以下兩個優點：

1. 可以偵對不同圖片的大小 (Scale) 處理
2. 在時間複雜度上不受圖片大小而成平方關系 $O(n^2)$，而是成線性關系 $O(n)$

在各種電腦視覺領域上，Swin Transformer 也都刷新了各項 SOTA，其中尤其以 semantic segmentation (ADE20K) 最為顯著

## 2. Introduction

### 對偵對不同圖片的大小 (Scale) 解釋

作者認為 CNN 與 Transformer 兩個架構最大的不同在於「Scale」，也就是訓練資料的大小。在 NLP 中一個「patch」就非常固定得為一個字詞 (ex：I am student -> 分為 3 個 patch)，而在最一開始的 Transformer 論文中，像是 ViT，所選擇的做法也同像為固定 patch，把圖片固定切成 $H/4 \cdot W/4$ 個 patch，而每個 patch 的大小不會隨著網路深度而改變，全部皆為 16x16。

### 時間複雜度上的解釋

與 NLP 相比，在影像上尤其是高解析度影像，如果單純使用 Transformer 計算量會與圖片大小呈平方關系。而且如果是要全高解釋度的語意分割領域，使用 Transformer 的效果一定不好。

### Overall in SwinT

於是作者提出了 Swin Transformer，改進了以上兩點：

1. Patch size 會隨著網路越深，由小慢慢放大，而這一步的用意是為了模仿 CNN 的全局視野會隨著網路越深視野越大 (在 CNN 中 kernel size 不變，但圖片大小會變、在 Transformer 中 patch size 會變，但圖片大小不變)。同時也因為這樣加入了多重解析度，可以應付更多電腦視覺的領域 (其中以分割效果最好)

2. 會把圖片分割成 non-overlapping windows (不會重疊的視窗)，只單純在 window 裡面做 self-attention，而非在整張圖片中做。因為一個 window 中所包含的 patch number 遠遠的小於圖片的大小，所以時間複雜度可以降到與圖片大小呈線性關系 (在文章後續會細講)

![Image](https://i.imgur.com/AZdaM1E.png)

上圖灰色小格為一個 patch，紅色格子為一個 window。每個 window 中包含固定數量的 patch，且 self-attention 只會在一個 window 中做計算。同時也可發現 Swin Transformer 的 patch 與 window 大小會隨著網路深度而變大，而且也有多重解析度的觀念在裡面。

### shifted window

Swin Transformer 最核心的觀念就是 shitfed window。為了使 window 和 window 之間也能學到彼此相關性，每做完一次 self-attention 後，window 會往斜角的方向移動。

![Image](https://i.imgur.com/cYvXzn0.png)

## 3. 網路架構

先上完整架構圖，接下來慢慢由左至右一塊塊介紹

![Image](https://i.imgur.com/aCJj2hV.png)

### Patch Partition

與 ViT 一樣，會先經過一個 Patch embadding (SwinT 稱這一步為 Patch Partition) 的步驟，把三維 $H\times W \times C$ 的圖片表示成二維序列 $N \times (P^2 \times C)$

![image-20210710135026339](https://i.imgur.com/mG3JoYk.png)

在 SwinT 中，$P$ 預設是 4，輸入圖片大小為 $H\times W\times 3$，所以網路的輸入維度是 

$$
\begin{gathered}
\frac{HW}{4^2} \times (4^2 \times 3) \\
= \frac{H}{4}\times \frac{W}{4}\times 48
\end{gathered}
$$

### stage 1

#### Linear Embedding

在 stage 1 中會先經過一層 Linear Projection (SwinT 中稱 Linear Embedding)，簡單說就是 1x1 conv，把 48 維轉換成 C 維 (C 會依照網路設計的大小而改變)

#### Swin Transformer Block

接著會經過 Swin Transformer Block

![Image](https://i.imgur.com/kdOG9Ve.png)

由上圖可以發現 SwinT 與 ViT 最大的差別就在於，把 ViT 中的 MSA 改成 W-MSA (Window-based MSA) 與 SW-MSA (Shifted Window-based MSA)

其餘部份與 ViT 大部份相同，有一不同的地方在 MLP 的 activation function 從 ReLU 改為 GELU (嗯…可能受到了 BERT 的啟發吧…)

#### Window-based MSA

主要設計是為了解決原 self-attention 計算複雜度為 $O(N^2)$ 的問題

以下簡單介紹原 self-attention 計算量之算法

計算出 $QKV$ 的公式：$x\times W^Q$、$x\times W^K$、$x\times W^V$ 一個需要 $hwC^2$，三個就為 $3hwC^2$

計算 $QK^T$ 需要 $(hw)^2C$

再計算乘以 $V$ 完整公式：$(QK^T)V$ 也需要 $(hw)^2C$

最後得到的 Multi-Head 還要再乘上一個 $W^Z$ 需要 $hwC^2$

所以總得來說原版 MSA 的計算量為

$$
\Omega(MSA) = 4hwC^2 + 2(hw)^2C
$$

在 SwinT 中 self-attention 只會在一個 window 中做

所以 $QK^T$ 變成 $\frac{h}{M}\frac{w}{M}$ 再乘上 $(M^2)^2C$ ，得到

$$
\Omega(W-MSA) = 4hwC^2 + 2M^2hwC
$$

而一個 window 所含的 patch size 遠小於圖片大小，所以計算量就可以與圖片大小呈線性的關系了

#### Shifted Window-based MSA

![Image](https://i.imgur.com/cYvXzn0.png)

先前有提到為了使不同 window 間也能有關系，所以會把 window 往斜上方移動，但移動後會產生幾個問題：

1. window 的數量變多了
2. 每個 window 的大小還不一樣

因此我們沒辨法直接對移動過的 window 做self-attention。

##### cyclic shift

而作者提出了 cyclic shift 來解決這個問題，把因位移而多出來的右上角，把它用搬的方法搬到了左下角，使得一張圖片中的 window 數量維持一致，如下圖

![Image](https://i.imgur.com/Sx5Q2ya.png)

參考了知乎大神上更詳細的圖片，從左邊移成右邊

![Image](https://i.imgur.com/zb57ex9.png)

##### masked MSA

但這又沿生出另一個問題，一個 window 內有來自不同地方的區塊阿，像是上圖的右上角，一個 window 裡同時包含了 6 和 4，如果直接做 self-attention 會…非常的不合理…，於是作者又提出了 masked MSA，通過適當的遮罩使得來自不同區塊不會互相運算到

![Image](https://i.imgur.com/kYiySYr.png)

再次參考知乎大神的詳解，舉個例子來說明會更清楚。

![Image](https://i.imgur.com/3AXzEGu.png)

我們再次以右上角為例子，這個 window 內同時有 6 和 4。要怎麼設計 mask 使得計算 attention 不會發生交疊呢？

答案如下圖：

![Image](https://i.imgur.com/OiKCryC.png)

以此類推右下角的例子，有 4 個同時存在呢？

![Image](https://i.imgur.com/fhJUFja.png)

答案如下：

![Image](https://i.imgur.com/jn2x2Um.png)

按照上面的邏輯可以推出所有 window 內的狀況所對應的 mask 設計。透過 0 1 的 mask 設計，可以使不同的區塊不會相互計算 attention 而影響。

最後附上完整流程圖

![Image](https://i.imgur.com/HQLqm3c.png)

### stage 2 ~ 4

#### Patch Merging

接著到了新的 stage，在 stage 2 ~ 4 中都做著重複的動作，首先會經過一個 Patch Merging，把剛剛 stage 1 所產生的 $\frac{H}{4}\times \frac{W}{4}\times C$，以每一個 patch 階與相鄰的其它 2x2 patch concat 起來，得到新的 $\frac{H}{8}\times \frac{W}{8}\times 4C$ ，如下圖

![Image](https://i.imgur.com/VTqWFKm.png)

再經過一層線性轉換，把 4C 變為 2C，得到 $\frac{H}{8}\times \frac{W}{8}\times 2C$，就是 stage 2 的輸入了，如下圖

![Image](https://i.imgur.com/SmBBECI.png)

會做 Patch Merging 的理由是模仿 U-Net 或是一般 CNN 中的多重解析度，藉由不斷的合併相鄰 patch 使得越深的網路，patch 的視野越大。

最後再做跟 stage 1 一樣的 Swin Transformer Block，就是完整的網路了。

### Relative position bias

Swin Transformer 在設計 Self-Attention 時，參考了 ViT 的設計之外，還額外加入了一個 Bias $B$，使得式變為以下：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{SoftMax}(\frac{QK^T}{\sqrt{d}}+B)V
$$

作者稱這個 Bias 為 relative position bias，為網路填加了額外的值置資訊

而這個 $B$ 不是隨機生成出來的，而是透過一系列算法生出來的，詳細可見以下知乎大神的 source code 解釋：

[https://blog.csdn.net/weixin_42364196/article/details/119954379](https://blog.csdn.net/weixin_42364196/article/details/119954379)

後面作者有用實驗比較一些 position 加入的方法，包含 absolute position、reletive position

發現：
* 有 shifted window 比沒有 shifted 效果來得好
* 加了 absolute position 效果不是最好的
* 使用 reletive position bias 效果是最好的

(下圖中的 w/o app. 指的是第一個 Attention 公式沒有 Attenion，只有 B)

![Image](https://i.imgur.com/jmUlfwe.png)

## 4. Experiments

### 分類 ImageNet 上的實驗

![Image](https://i.imgur.com/Vz82yqF.png)

### 偵測 COCO 上的實驗

![Image](https://i.imgur.com/cm1G7Bj.png)

![Image](https://i.imgur.com/zWJO8Dd.png)

### 語意分割 ADE20K 上的實驗

![Image](https://i.imgur.com/YOaTVpZ.png)

可看到不管在哪一個領域上 SwinT 皆打敗了所有目前的 Transformer 網路，而與傳統 CNN 網路相比 (EfficientNet) 效果我認為是不相上下

值得注意的是 SwinT 在語義分割的題目表現特別好，好超過 3 個 ticks，連在 SOTA 的網站中都可以看到明顯的差距

![Image](https://i.imgur.com/AkLOwMM.png)

## 結論

我自己讀完乍看之下，好像與直接前幾篇論文的 proposal 差不多，不外乎就是模仿 CNN 或是減少運算量。但如果真的照著論文給的實驗結果，SwinT 的效果也好太多，真的有點神奇。

如果現在在 Google 上搜尋 SwinT 也常常會找到什麼屠榜之類的標題，我自己是覺得有點過頭了啦，論文實驗歸實驗，真正要在現實實作上發揮功能才是最重要的部份。(例如測資多寡的問題)

不過這個論文倒是開了一個「Swin」風潮，希望再過個半年到一年，能把「Swin」的觀念再發揮的成熟一些，讓大家知道 Transformer 的厲害哈哈

## Reference

https://www.youtube.com/watch?v=SndHALawoag

https://zhuanlan.zhihu.com/p/360513527

https://zhuanlan.zhihu.com/p/404001918