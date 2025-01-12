---
title: AutoML - NAS 與 NASNet 回頭閱讀
mathjax: true
date: 2022-01-22 14:57:12
tags: AutoML
categories: 電腦視覺整理
---

在深度學習發展的今天，要設計一個網路要一件非常簡單的事情，但是要設計出「符合硬體需求」的網路非常困難，要怎麼在硬體的限制下求得最佳效能的網路呢？我們可以加深網路深度、加寬網路…等等技巧，但要怎麼在效能與效果間取一個最佳平衡？於是 NAS 誕生了。NAS 旨在透過一個「非人工」「自動化」的方法去尋找最佳的網路組合。

keywords: NAS、NASNet
<!--more-->

## 什麼是 NAS ？

那倒底什麼是 NAS 呢？NAS 的目的就是希望以一套演算法能**自動的根據我們的需求找到效果最好的網路架構**，而一個 NAS 一共可以包含以下三個部份：Search Space、Search Strategy、Performance Estimation Strategy

### Search Space

也就是我們在尋找網路架構時由「基本單位」所組成的尋找空間。一個「基本單位」可以是一個 conv 層、一個 kernel size、chennel size、一個 normalization…等等，透過像堆積木的方式把所有在尋找空間中的「基本單位」**拼**成一個網路

### Search Strategy

在一定的 Search Space 中，我們總不可能所有排列組合都試一邊，可能性太多種了，因此我們需要一個演算法來找出最好的排列組合。可以是 Greedy Search、Evolution Algorithm 等等…

### Performance Estimation Strategy

透過 Search Strategy **拼**出來的網路架構後需要一個評斷這個網路好壞的方法，可能是分類的 Accuracy、偵測的 mIoU 等等…，最高的 Performance Estimation 即為我們要的網路

## 2016 Google NAS

在 2016 年 11 月 Google 是最先提出 NAS (Neural Architecture Search) 的想法的論文 [https://arxiv.org/abs/1611.01578](https://arxiv.org/abs/1611.01578)，整個流程如下：

![image-20220123151846095](https://i.imgur.com/oIyFEH7.png)

使用一個由 RNN 構成的 controller (左粉紅色區塊)，會以一個機率 $p$ 隨機找出一個網路 $A$ (上線流程)，把網路 $A$ 在 CIFAR-10 小資料夾上做預訓練並得到 Accuracy $R$，接著把 $R$ 當作是 Backbropagation 去更新 RNN 中的參數權重，重覆以上動作直接 RNN 網路收斂。以上的做法是使用到強化式學習 (Reinforcement Learning) 的精神，把 RNN 當成是 controller 去不斷自我訓練。

在最原始的 NAS 中，Search Space 中都是一些超基本的參數：CNN Filter (Kernel) 的長寬、數量、Stride 數量：等等，每一個變量都當做是 RNN 中的一個神經元，透過強化式學習找出效果最好的組合

![image-20220123155242514](https://i.imgur.com/hzO2Z5m.png)

至於 Search Strategy 這邊就不細講，有興趣的人可以自己去參考下面這篇知乎的文章：

[NAS 知乎詳解](https://zhuanlan.zhihu.com/p/52471966)

## NASNet

以上 NAS 會有一個問題：付出的運算成本太大了，又要在一大堆 Search Space 中找排列組合，又要評估好壞，因此只能實作在較小的資料集如 CIFAR-10 上，以 Google 的例子光 CIFAR-10 就要 500 台 GPU 運行 28 天才有結果，如果要應用在大資料集如 ImageNet 上那更不知需要多大的 GPU 了。

而為了能將 NAS 應用在大資料集上 Google 又在 2017 年提出 NASNet，其最大的不同在於把 Search Space 裡中的單位「變大」，也就是稍微變得複雜了點，不像 NAS 中一個基本元件是 kernel 的長、寬…等等，NASNet 的基本元件是已經組裝好的，例如下圖所示：

![image-20220123161850799](https://i.imgur.com/4EV40JP.png)

而 NASNet 的基本元件又分為兩種：Normal Cell、Reduction Cell，簡單來說前者不會降維、後者則會，用意是模仿 ResNet 會慢慢降低解析度。如下圖：

![image-20220123162428688](https://i.imgur.com/3GXfKxF.png)

不管是 Normal Cell、Reduction Cell，一個 Cell 都由五個步驟所組成，分別對應下圖的「灰色 x2」「黃色 x2」「綠色」

![image-20220123162908703](https://i.imgur.com/HxpXrU5.png)

NASNet 的詳細流程為：

1. 「灰一」從之前已經訓練好的第 $h_i$ 層中，選擇一個 Feature Map 作為 Hidden Layer A 的輸入 
2. 「灰二」再從第 $h_{i-1}$ 層中，再選擇一個 Feature Map 作為 Hidden Layer B的輸入
3. 「黃一」從 Search Space 中選擇一個運算給 Hidden Layer A 
4. 「黃二」從 Search Space 中選擇一個運算給 Hidden Layer B
5. 「綠」把 Hidden Layer A、B，用 Concat 或 Element-wise 的方法加起來

為了能讓 RNN controller 同時找出 Normal Cell 與 Reduction Cell，RNN 會有 $2\times 5\times B$ 個輸出，其中前 $2\times 5$ 作為 Normal Cell，後 $2\times 5$ 作為 Reduction Cell

找出來效果最好的網路架構如下，可以看看參考一下

![image-20220123163851151](https://i.imgur.com/hIYg2lO.png)

同樣，對於 NASNet 的 Search Strategy 這邊不做多述，有興趣的人可以參考下面知乎文章：

[NASNet 知乎詳解](https://zhuanlan.zhihu.com/p/52616166)

## 結論

NASNet 解決了 NAS 無法應用在大資料集上的問題，同時也因把「基本單位」變複雜了，也可以換句話說：「基本單位」由人工設計組合給定死了，雖然網路的彈性下降，但減少了 Search Space 所代來最大的好處就是運算時間變少了。

而 NASNet 這一篇論文也有達到當時的 SOTA，也開起了後續 AutoML 領域的發展，也更是為後面另一個強大的論文 EfficientNet 奠定基礎

## Reference

[medium NAS 大全筆記](https://medium.com/ai-academy-taiwan/%E6%8F%90%E7%85%89%E5%86%8D%E6%8F%90%E7%85%89%E6%BF%83%E7%B8%AE%E5%86%8D%E6%BF%83%E7%B8%AE-neural-architecture-search-%E4%BB%8B%E7%B4%B9-ef366ffdc818)

[NAS 知乎詳解](https://zhuanlan.zhihu.com/p/52471966)

[NASNet 知乎詳解](https://zhuanlan.zhihu.com/p/52616166)
