---
title: >-
  Vision Transformer 演化史: Bottleneck Transformers for Visual Recognition - BoT 把
  Bottleneck 加上 Transformer
mathjax: true
date: 2021-09-02 15:59:50
tags: Vision Transformer
categories: 電腦視覺整理
---

2021 年 1 月 Google 提出了 BoTNet 架構，其最核心的思想就是替換 ResNet 中的 Bottleneck，把最後幾層的卷積層 (Conv) 替換為 Multi-Head Self-Attention (MHA)。實驗證實在僅僅只修改幾層網路下，BoTNet 在實例分割任務上取得了 44.4% 的 Mask AP 與 49.7%的 Box AP，與純 ResNet 相比，**在分類、分割任務上皆有效能上的提升，同時還可以降低參數量**。

[https://arxiv.org/pdf/2101.11605.pdf](https://arxiv.org/pdf/2101.11605.pdf)

keywords: BoT、Bottleneck
<!--more-->

## 1. Introduction

本篇論文主要是在討論實例分割的改進，因而作者以「應用 Transformer 在實例分割上」，以及像 ViT 一樣使用純 Transformer 為出發點下，提出了兩大的問題：

1. 通常分割的圖片 (1024x1024) 相較於分類的圖片 (224x224) 大小還有來得大。
2. 在圖片大的情況下，attention 的計算量會呈現指數的上升 ([可參考以前的文章](https://mushding.space/2021/07/09/NLP-%E8%88%87-CV-%E7%9A%84%E7%B5%90%E5%90%88%EF%BC%9ADeformable-DETR-Deformable-Transformer-For-End-To-End-Object-Detection-%E6%AD%A3%E9%9D%A2%E5%B0%8D%E6%B1%BA-DETR-%E7%9A%84%E7%BC%BA%E9%BB%9E%EF%BC%81/))

為了解決以上的問題作者提出了改進方法：

1. 先使用普通的卷積運算的強項，從大解析度的圖中截取低階特徵
2. 在最後一層卷積，圖片被 downsampling 成小解析度後，再做 Transformer 運算

因此作者直接引用了非常成熟的 ResNet-50 來滿足低階特徵截取的部分，並將其最後幾層改為 Transformer。由於 ResNet-50 使用了 Bottleneck，於是作者將這個 ResNet 與 Transformer 結合的網路稱為 Bottleneck Transformer，簡寫為 BoT。

BoT 的特色為網路架構非常單純的僅僅把 bottleneck 中的卷積層替換成 Self-Attention 層。如下圖所示：

![Image](https://i.imgur.com/KUAnnj9.png)

而我個人認為，BoT 雖然在創新上占比不多 (而且也只是加個 Self-Attention 就說自己是 Transformer…)，但是嘗試著把 Transformer 與 CNN 結合，以及最後的實驗證明結果，都可以當成研究 Transformer 與 CNN 互利共生的好的出發點。

## 2. 網路架構

前面也提到了，作者直接把 ResNet-50 拿來用，所以 BoT 在網路架構上並不複雜。下圖左邊為傳統的 Transformer Block，而中間則是作者提出的 BoT Block。作者將其視為與 Transformer Block 同階級的模組

![Image](https://i.imgur.com/erOrnzF.png)

而其中的 MHSA 為 Multi-Head Self-Attention 的簡寫 (也可寫為 MSA)，其詳細架構如下圖：

可看到 BoT 是有做 positional encoding 的，作者提到這邊作的是 Relative Position Encodings，而且是與 Query 做矩陣乘法而非加法

![Image](https://i.imgur.com/zfvcaNS.png)

而這樣子的 MHA 架構，只會套用在 ResNet-50 的最後一層 (c5 層)，其餘架構**全部**與 ResNet-50 一致。這是為了達成減少運算量這個需求。

![Image](https://i.imgur.com/Nu1aVpt.png)

## 3. Experiment

作者在實驗上沒有特別與其它網路或者是 sota 互相比較，而是單純比較 R50 與 BoT50 之間的差別。

### 在實例分割上的比較

資料集選用 COCO，可看到在不同 epochs 下 BoT50 皆比 R50 優秀

![Image](https://i.imgur.com/Wx1Bw0o.png)

### 位置編碼的比較

單 attention 增加 0.6，而加上「相對位置」後的效果明顯好了一些

![Image](https://i.imgur.com/n7JbimZ.png)

## 4. BoTNet-S1

在論文的最後，作者把 BoTNet 改成分類任務，並改稱作為 BoTNet-S1。

作者發現如果單純的把 BoTNet 直接放進 ImageNet 分類的話，效果與 ResNet50 不相上下，但是如果參考 ViT，只把圖片 downsample 到 1/16 的大小 (換算成 ResNet 為第四個階段 (c4))，而非 ResNet 的1/32 的話效果會變好。

於是作者把 BoT 的最後一層 (c5) 的 stride 2 給取消，稱為 S1 (少一個 stride 的意思…？)，網路稱為 BoTNet-S1

實驗比較結果：

![Image](https://i.imgur.com/Wj9yHNC.png)

BoTNet-S1 與其它 sota 比較

![Image](https://i.imgur.com/lcC7ebU.png)

## 結論

本篇論文架構簡單，改進的地方並不多，許多實驗也並未與 sota 比較。

但是如果換個角度來看：只把 ResNet 最後一層改成 attention 效果就可以好上 1% 這點來說，還是挺有意思的，以最低的更改成本就可以達到效果好且參數少。

且這篇論文也在 CNN 與 Transformer 的結合，不管是效能還是運算量，都給出了一些見解，使這兩個網路往各取所長之路前進。

## Reference

https://zhuanlan.zhihu.com/p/347602463

https://bbs.cvmart.net/articles/4142