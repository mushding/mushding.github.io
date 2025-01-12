---
title: 'Swin Transformer V2: Scaling Up Capacity and Resolution'
mathjax: true
date: 2021-12-23 00:29:33
tags: Vision Transformer
categories: 電腦視覺整理
---
論文網址：[https://arxiv.org/pdf/2111.09883.pdf](https://arxiv.org/pdf/2111.09883.pdf)

Swin 原班人馬在 2021 11 月提出 Swin Transformer 的改良版 Swin Transformer V2。主要是優化 Swin 在 scale up 大參數模型上的能力

改進了 Swin 架構中的三個小地方：

* post normalization：在 self-attention layer 和 MLP block 後做 layer normalization
* scaled cosine attention approach：使用 cosine 相似度來計算 token pair 之間的關系
* log-spaced continuous position bias：設計全新的相對位置編碼

keywords: Swin v2
<!--more-->

## Introduction

在 NLP 的領域中，自 Transformer 提出以來，一路提出更多新架構：BERT、GPT-3，而使用的參數量也呈指數上升。這個現象叫做 scaling up 是 NLP 領域為了提升更好的效能所做的方法 (白話的說叫：巨量資料集、瘋狂疊參數)。

但是在 CV 領域中，很少聽到有人用 scaling up 達到很好的效果，而且實作經驗也告訴我們，一昧的增加參數效果不見得好，所以目前 CNN 最多的參數量 (1B 億)，與 NLP 相比單位完全在不同的量級上 (GPT-3 的參數可是到 1700B 億了…)

那為什麼會有這樣的現象？這篇作者認為是 CNN 的 inductive bias 限制了效果，而最近流行的 Transformer 並沒有這個限制

因此本篇作者提出 Swin V2 是為了之後 Scaling up 做準備，並且同時實驗分類任務與分割任務，看看效果如何

## 網路架構

![](https://i.imgur.com/3iR0LCB.png)

作者為了把 Swin Transformer Scaling up 做了以下三個小技巧

### Post normalization

作者第一個小技巧是把 LN 放到 Self-Attention Block 後

作者經由下圖實驗發現當 Swin 做 Scale 後，越深層的 activate function 之間的差就越大，使得網路變得非常難以練訓

紅色是最大的網路架構，有 658M 個參數量，可發現上下相差非常大

![](https://i.imgur.com/6d3m9YE.png)

會使得 activate function 極端化的原因是：在經過超多次的 Self-Attention 後，兩像素之間，相似會變超相似，不相關的會超不相關

作者還提出 Scaling up 後 Pre-Norm 與 Post-Norm 的差別，可看到 Pre-Norm 甚至還跑到一半就爆了

![](https://i.imgur.com/7qenyIF.png)

作者還每 6 個 Transformer Block 又額外加一個 LN 層，為了使網路更穩定

### Scaled cosine attention approach

在最原本的 Transformer 論文中，query 與 key 的運算子是使用 dot product (內積運算)

作者發現當把模形做 Scaling up 後，Attention map 中的某些 Patch 某些 Head，權重往往會變過大，變成只有它最重要，特徵不平衡了

於是作者改使用 Scaled cosine attention (cosine 相似度) 來代替

$$
\mathrm{Sim}(q_i,k_i)=cos(q_i,k_i)\tau+B_{ij}
$$

$\tau$ 是一個可學習參數，head layer 之間不共享

$B_{ij}$ 是指相對位置

因為 cosine 本身的取值範圍本身就相當於是被正歸化後的結果，因此可以平均差距的問題

### Log-spaced contiguous position bias（log spaced CPB)

作者直接把模形 Scaling up 發現效果越來越差，推論可能是因為相對位置沒有一併放大的問題，因此提出來 Log-spaced contiguous position bias 來減少因放大而產生的差距

舉個例子，假設我們要把 8×8 window size fine-tuned 到 16 × 16 window size，使用原本 Swin 定義，相對位置座標會從 [−7, 7] × [−7, 7] 到 [−15, 15]×[−15, 15]，放大倍率約為 1.14x

因此作者試著轉換相對位置的座標，把單位從整數，改為以 log 為單位，公式如下：

$$
\begin{gathered}
\hat{\Delta x} = \mathrm{sign}(x) + log(1+ |\Delta x|)\\
\hat{\Delta y} = \mathrm{sign}(y) + log(1+ |\Delta y|)
\end{gathered}
$$

經由上面的座標轉換從  [−2.079, 2.079] × [−2.079, 2.079] 變成 [−2.773, 2.773] × [−2.773, 2.773]，放大倍率為 0.33x

相比之前的方法差距小了不少

作者在把座標換成 log，又新增了個叫 Continuous relative position bias，簡單來說就是把上面得出的相對位置座標，再經 2 層 MLP 層

加入 2 層可學習的 MLP 後，使得未來在 Scaling up 輸入圖片大小不同時，網路彈性大一些，而不是像以前一樣固定的死死的

結論可看下圖，最上面使用 ViT 的原作法，中間則是 Swin 的整數做法，最下面是 Swin v2 的 log 座標做法。可發現 ViT 與 Swin 的效果相差最多，Swin v2 提出的 log 只好一些些

![](https://i.imgur.com/5GRgOEM.png)

## Experiment

實驗方面鐵定頂級，這裡就不再多說了，有興趣可自己去看原論文，下面就放一張 SOTA 的比較表

![](https://i.imgur.com/8e4vFQs.png)

## 結論

這篇論文並未提出什麼新架構，僅僅是把 Swin 改成更好 Scaling up 模形的工程報告書而已

不過我們也可以從中看到一個趨勢：CV 開始往 Scaling up 方向前進了

由 NLP 成功的例子我們知道：「大力出奇蹟」，更多的資料，更大的模形，勢必是 CV 界下一步的方向

因此 Self-Supervised Learning、Trasfer Learning、Scaling up 想必是末來研究的重點

而 CV 是否真的能 copy paste NLP 的經歷並成功打出一片天？我們就靜觀其變吧！

## Reference 

[https://zhuanlan.zhihu.com/p/435210138 知乎大神筆記](https://zhuanlan.zhihu.com/p/435210138)

[知乎大神們對 CV 未來的爭論，裡面有很有趣的觀點，大推](https://www.zhihu.com/question/500004483)