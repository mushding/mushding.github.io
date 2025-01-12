---
title: >-
  Vision Transformer 演化史: Conditional Positional Encodings for Vision
  Transformers - 可變序列長短的 Positional Encoding
mathjax: true
date: 2021-07-28 16:02:03
tags: Vision Transformer
categories: 電腦視覺整理
---

論文提出 Conditional Positional Encoding (CPE) 模組，以及應用 CPE 模組的 Conditional Position encoding Vision
Transformer (CPVT) 網路架構，負責來解決 Transformer 輸入圖片大小要固定的問題。

keywords: CPVT、CPE、PEG、zero padding
<!--more-->

## 1. Introduction

論文指出雖然最原始的 Transformer 是可以支持不同長度向量的輸入，但由於在把向量放進 Encoder 前會多做一步 Positional Encoding，不管是公式解或是機器自行學習解，都會遇到這一層長度無法修改的問題，因為在大部份的實作中會設置一層可學習層 (pytorch 中的 nn.Parameter)，網路在學習時不斷修改其中的參數，因此不行處理不同長度的輸入。雖說在之前的研究有人使用雙三次插值 bicubic 來補充遺失的位置資訊。

下圖為各種 Encoding 方式的效果：可看到有 Positional Encoding 效果還是會比較好。

![Image](https://i.imgur.com/c820JAY.png)

再來作者提到 CNN 有平移不變性 (translation-invariance)，即圖中特徵點的位置不影響分類任務的效果，如果加上了絕對位置編碼 (absolute positional encoding) 會破壞 CNN 混然天成平移不變性 (translation-invariance) 的優點，而如果使用相對位置編碼 (relative positional encodings) 則會有更多運算、需修改 Transformer 架構、與絕對位置編碼衝突等問題。

因此作者提出一個完全不同於絕對、相對位置編碼的想法，不是在 input 上「加上」位置資訊，而是在 Encoder 中用「算」出來的，這邊劇透一下，這邊用到了 CNN zero-padding 會加上位置資訊的特點來達成 (關於這方面的議題，下面會細講，而且之後會專開一系列來討論這個話題)

提出了 Conditional Positional Encoding (CPE) 模組，中間使用了 Positional Encoding Generator (PEG) 小模組。以及應用 CPE 設計出的 Transformer 架構 Conditional Position encoding Vision Transformer (CPVT)。

![Image](https://i.imgur.com/Z7GY1H0.png)

## 2. 網路架構

基於以下三點來設計架構：

1. 效果好
2. 避免排列不變性 (permutation equivariance)，也就是：輸入序列順序變化時，结果也不同。且隨著輸入圖片 size 的改變要也可以有對應變化
3. 能直接套用在現成 Transformer 架構上

下圖為架構圖：

### Positional Encoding Generator (PEG)

![Image](https://i.imgur.com/O5dfLIf.png)

1. 把 input token，(class token 以及 patch token (這裡論文稱為 feature token)) 的 patch token reshape 成原圖片的二維大小 (也就是一種回去原維度的感覺)，公式如下：

$$
X\in\mathbb{R}^{B\times N\times C}
$$

$$
\rightarrow X\in\mathbb{R}^{B\times H\times W\times C}
$$

2. 接著經過一個 transform 定義為 $\mathcal{F}$，而這個 $\mathcal{F}$ 其實就是一個 conv 做卷積運算，其中 kernel size $k\ge3$，**$\frac{k-1}{2}$ 的 zero padding**，而這裡的 zero padding 正是網路獲得位置資訊的重要來源。
3. 再把三維圖片 reshape 至二維序列

$$
X\in\mathbb{R}^{B\times H\times W\times C}
$$

$$
\rightarrow X\in\mathbb{R}^{B\times N\times C} 
$$

4. 而 class token 的部份則不參與 PEG 計算，直接加回二維序列中
5. 最後一步把新算出來帶有位置資訊的二維序列，「加」concat 回原二維序列中，再當成下一個時間點的 Encoder 輸入

### Conditional Position encoding Vision Transformer (CPVT)

![Image](https://i.imgur.com/Z7GY1H0.png)

而 CPVT 的做法也很直覺，不像 ViT DeiT 一樣在輸入 Encoder 前加上位置資訊，而是選擇在第一個 Encoder 做完後執行 PEG 模組，藉此加上位置資訊，再完成乘下的 Encoder。

而 CPVT-GAT 想要解決的是 class token 視為額外 token 的問題，因為 class token 是不能隨便與 patch token 順序亂混的。但作者認為 GAP (global average pooling) 在垂直上是無序的 (inherently translation-invariant)，因此現在就可以直接把 token 們視為一個整體放入 PEG 中做計算，最後再經一個 GAP 得到最後結果。作者發現 CPVT-GAT 是效果最好的方法。

## 3. Experiment

### 與 SOTA 的比較

作者主要與相同架構的 DeiT 做比較，可以發現效果好個 **1%** 上下

![Image](https://i.imgur.com/Tw8krlq.png)

### 與其它 Positional Embedding 的比較

LE 代表 learnable encoding，RPE 代表 relative positional encoding，sin-cos 代表 absolute positional encoding。

結論：sin-cos 和 LE 差別不大，作者提出的 PEG 優於所有的方法

![Image](https://i.imgur.com/58eHqTn.png)

### PEG 插入 Encoder 位置的比較

發現在第一個 Encoder 到第四個 Encoder 之間插入效果最好

![Image](https://i.imgur.com/yqi3PIT.png)

那每一個 Encoder 後面都加呢？作者發現不會越多越好，運算量增加但效果基本不變

![Image](https://i.imgur.com/m333C3N.png)

### 神奇的 Padding 比較

在 CNN 加上了一個 zero padding，真的有必要嗎？

結論：zero padding 真的學到了位置的資訊，知道哪裡是角、哪裡是邊。也側面證實了絕對位置編碼的作用。

![Image](https://i.imgur.com/3f96HYb.png)

### CNN vs Padding

作者在網路中加上了一層 CNN，那…倒底效果變好是因為 CNN 學習的關系，還是單純有了 zero padding 呢？

* 如果是 PEG 位置表示能力起了作用，那我把 conv 換成 FC 層，效果應該會差一點
* 如果是 PEG 的 CNN 運算 (representative power) 起了作用，那讓 conv 參數固定不更新 (不學習)，效果應該會差一點

而作者實驗的結論是：就算把 conv 參數固定不訓練，效果依舊好，證明了是 zero padding 起了作用，而非 CNN 起了作用

![Image](https://i.imgur.com/km1k7zF.png)

## 結論

這篇論文討論了位置編碼這個議題，而使用的方式竟是 CNN 神奇的 zero padding 特性來達成。關於 CNN 的特性我後面會開系列文章解釋 (因為已經有不少論文討論過相關話題了)。總之對我而言，透過這篇論文學到一種新的位置資訊獲得方法，以及了解到原來 CNN 有絕對位置的特性。

## Reference

https://zhuanlan.zhihu.com/p/354913120

https://zhuanlan.zhihu.com/p/99766566