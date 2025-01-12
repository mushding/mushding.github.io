---
title: >-
  Vision Transformer 演化史: An Image is Worth 16x16 Words:Transformers for Image
  Recognition at Scale - 正式開始 Transformer 元年
mathjax: true
date: 2021-07-09 17:22:11
tags: Vision Transformer
categories: 電腦視覺整理
---

如果說之前的 DETR 是 Transformer 系列的開山始祖的話，那 ViT 就一定是發揚光大的人了。2020 Google 提出了 Vision Tranformer，提一個完全不用 CNN 只使用 Transformer 的網路架構，整體來說網路架構並不複雜，但對後來的影響力可不小，從 ViT 之後的論文名字都會變成 …T 什麼什麼 Transformer 的意思，而我系列的文章也改名為：「Vision Transformer 演化史」。

[https://arxiv.org/pdf/2010.11929.pdf](https://arxiv.org/pdf/2010.11929.pdf)

keywords:
<!--more-->

## Introduction
這篇論文的名字：An Image is Worth 16x16 Words，彷彿就在告訴我們如果把圖片切成一塊一塊的，是不是就能變成一串 Sequence 呢？這樣就可以放近 Transformer 中訓練了。而這篇論文提出的方法非常簡單，完全沒有使用任何 CNN 架構，以使用「最原汁原味」的 Transformer 為主要目標。告訴大家：其實只用 Transformer 在分類上效果很不錯喔！

## 網路架構
以下為 ViT 網路架構，可發現只使用了 Transformer 的 Encoder，沒有使用 Decoder，(我個人理解為，Transformer 設計初衷是要 Seq2Seq，但分類問題不需要輸出 Sequence 所以把 Decoder 取消掉了)。以下一一介紹。

![image-20210710132412756](https://i.imgur.com/xGVHCtB.png)

### 圖片預處理：由圖片變為 Patch
這一步最大的精神就是，想辨法把三維圖片 $(HWC)$ 表示成二維 sequence $(ND)$。sequence 中的每一塊稱作為一個 Patch。而這篇文提出「切塊 (Patch)」的方法。具體做法如下：

把 $x\in H\cdot W\cdot C$ 根據切塊圖片大小 $P$ 變成一個 $x_p\in N \cdot(P^2 \cdot C)$ 的二維向量，而 $N$ 等於 $HW/P^2$，也就是說 squence 可表示成：特徵數長度為 $C$，一塊大小為 $P^2$，sequence 長度為 $N = HW/P^2$ 的 sequence。嗯…用文字好像不好描述，看圖。

左手邊原圖大小為 $H\cdot W\cdot C$ 記為 $x$，而切塊後大小為 $N \cdot(P^2 \cdot C)$ 的向量記為 $x_p$

![image-20210710135026339](https://i.imgur.com/mG3JoYk.png)

### Patch Embedding

得到 $x_p$ 後，要再把維度 $N \cdot(P^2 \cdot C)$ 轉換成 $(N\cdot D)$ ，而 $D$ 是自定義的參數，目的是做維度的整理 (或說降低維度) (假設從 3072 變成 1024)

做法是經過一個可學習的 Linear 層 $E$ 來得到

公式如下：

$$
z_0 = [x_{class}; x^1_pE; x^2_pE;...;x^N_pE] + E_{pos}
$$

把預處理得到的 Patch ，再經過一個可學習 Linear $E$ ，得到最後輸進網路的 $D$。這一步稱作 Patch Embedding

### class token

接著把 $N$ 加上 1，多了一個 $x_{class}$ 輸出。為什麼要加上這個東西呢…？我們可以回想上一章提到的 Object query ，假設我們 $N=9$ 代表我們有 9 個表示 Object 大小位置一些特性的向量，新增一個格子有點像新增一個 query 去和其它 9 個向量做 self attention 的感覺，而這新增的格子就是用來輸出分類結果，運算中會與其它 9 個格子做 self attention 計算相似度，找出最有可能的結果。而 $x_{class}$ 是一個可學習的向量，通常是加在 0 這個地方。

### Position Embedding

最後照著 Transformer 的傳統，加上代有位置訊息的 Position embedding，只不過這裡 ViT 使用的不是 sincos 那樣固定的編碼，而是使用可自行訓練的變量。以下為視覺化的 Position embedding 發現好像有那麼一點規律可循。

![image-20210710141000186](https://i.imgur.com/6pEJ6Rh.png)

### Encoder
Encoder 的地方真的是什麼也沒動，頂多最後輸出的 $(N, B, C)$ 向量經過一個全連接層變成 $(N, B, class\_num)$ 而已。

![image-20210710141119408](https://i.imgur.com/mscGLxB.png)

## 訓練方式
這篇論文使用 Transfer Learning 的方法，先在大數據集上預訓練，在放到小數據集上 fine tune。(後面會講效果)

同時設計了三種大小不同的模型：

![image-20210710142220216](https://i.imgur.com/yvlYUpo.png)

## Experiments
實驗用到數據集有：(越往下越難)

* ImageNet -> 1000 classes
* ImageNet-21k -> 21k classes
* JFT -> 18k classes

**實驗一、對比 CNN**
這篇論文因為使用 Transfer Learning (等等會提到更深入) 所以選用 Big Transfer (BiT) 以及 Noisy Student 來做比較。

![image-20210710141942775](https://i.imgur.com/vLeP9Hx.png)

可以發現比 BiT 效果好一些，重點是參數的使用量！少非常非常多！

**實驗二、對比數據集**
作者對比了不同大小的數據集，以及不同架構的網路，得出以下圖片：

![image-20210710142532275](https://i.imgur.com/ucDF7IS.png)

發現一件重要的事情：

**在小預訓集上訓練時效果不比 CNN 好，但在大預訓集上 Transformer 的強大顯現出來了**
**在小預訓練集上 Residual 還是比較強，在大預訓練集上 attention 才發揮能力**

### 細看 Transformer

作者把 patch embedding 中的 $E$，做可視化分析，發現特別的地方是，patch embedding 學到的東西與 CNN 有幾分相似，都是一些基本的特徵組合

![Image](https://i.imgur.com/Eq2nJx9.png)

接著作者分析了在 self-attention layer 中 各個 attention head 與各層之間的關系，以 Mean attention distance 作為分析目標。

Mean attention distance 的意思指的是，一個 pixel 能最遠與附近的其它 pixel 做相關性運算，也可以理解為就是 CNN 中的 receptive field (空間感知域)

依據實驗結果可看到在網路第一層，假設網路中有 16 個 head，這 16 個 head 它們的 receptive field 有的大有的小，有些 head 天生就可以有比較 Global 的感知域，而有些則是比較 Local 的感知域。

隨著層數的增加，每個 head 的 receptive field 也隨之增加，意謂著層數越深越能看到更全局 Global 的資訊

與 CNN 不一樣的是，CNN 在一開始並不會出現全局的感知域，而是像底下藍線一樣，隨著層數而呈線性關性，但 Transformer 能做到的是紅色圈圈部份，這些早期全局資訊是 CNN 所沒有的。

![Image](https://i.imgur.com/spEkO2q.png)

## 結論
這篇論文實作出一個完全 Transformer based 的方法解決分類問題，由於 Transformer 在大訓練集上的效果比較好，因此如果要使用的話，會要使用 Transfer learning 最後在 fine tune 這樣。

## Reference

https://zhuanlan.zhihu.com/p/356155277

https://zhuanlan.zhihu.com/p/342261872

https://www.youtube.com/watch?v=j6kuz_NqkG0&t=1173s

https://www.youtube.com/watch?v=TrdevFK_am4

https://www.youtube.com/watch?v=DVoHvmww2lQ