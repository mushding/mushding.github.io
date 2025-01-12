---
title: 'ConvNeXt: A ConvNet for the 2020s - 模仿 Swin 的 CNN 網路'
mathjax: true
date: 2022-03-15 14:38:52
tags: Vision Transformer
categories: 電腦視覺整理
---

2021 年是 Transformer 發揚光大的一年，短短的一年間推出了許多新的架構，其中尤其又以 Swin Transformer 效果最為突出，其效果甚至超越了當前 CNN 的 SOTA。

2022 年 FAIR 重新探討了 CNN 與 Transformer 之間的關系，試著建立出一個「很像 Transformer 的 CNN 網路」，提出了基於 ResNet 魔改的 ConvNeXt。經實驗得知只使用 CNN 架構的效果就超越了 Swin Transformer。

keywords: ConvNeXt
<!--more-->

## 前言

![image-20220316194754349](https://i.imgur.com/8jMGxop.png)

自從 2020 10 月 ViT 提出後，Vision Transformer based 的方法開始屠榜 SOTA，CNN 好像被遺棄了一般，大家一窩峰的去研究 Transformer。而後 2021 3 月 Swin 被提出，Swin 基於「計算量過大」「缺少多重解析度」以上兩個理由把 CNN 的一些想法引進到 Transformer 中。從這之後，更多的論文試著把 CNN 的獨有的想法融入到 Transformer 中。有趣的是到了 2022 的現在，FaceBook 這篇論文反過來思考：**如果我們是把 Transformer 的特色融入到 CNN 中呢？**於是誕生了這篇論文

這篇論文的核心理念是：現在 Transformer 架構之所以好的原因可能不只是在 Self-attention 上而已，**Transformer 特有的訓練技巧也是效果好的原因之一**，我們能不能不斷的優化 CNN 的訓練來取得更好的效果呢？

其實早在 2021 10 月著名的 [pytorch image model - timm](https://github.com/rwightman/pytorch-image-models) 套件作者 Ross Wightman 就提出了篇論文：[ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476)，它的核心想法是把經典的 ResNet-50 用新的訓練想法來練 (Mixcut 資料擴增、LARS optimizer)，在相同網路架構下成功的把 ImageNet 分類問題準確率提升到 80.4%

可以參考以下的文章有更詳細的說明：

[如何看待timm作者发布ResNet新基准：ResNet50提至80.4，这对后续研究会带来哪些影响？](https://www.zhihu.com/question/492966803/answer/2176330600)

[ResNet Strikes Back! | Patches Are All You Need? | Papers Explained](https://www.youtube.com/watch?v=Gl0s0GDqN3c&ab_channel=TheAIEpiphany)

以上實驗也間接說明了一件事情：其實 CNN 架構還有很好的優化空間，在適當的優化下是可以提升不少點的

## Introduction

如同前面提到的，這篇論文試著參考 Transformer 的訓練流程來套用到 CNN 上，優化 CNN 使其能得到更好的效果。並把結果網路取名叫 ConvNeXt，且是一個 pure-CNN 架構。

## 魔改 ResNet

作者使用 ResNet-50 以及 ResNet-200 作為 baseline 當作網路魔改的起始點。

ResNet 一共經過四次修改最後成為了 ConvNeXt，不管是最後效果或是 FLOPs 都對標了當前最強的 Swin，在相同運算量下取得更好的效果

下面是修改流程圖：以下一步一步來解說

![image-20220316193217265](https://i.imgur.com/Jdm64HD.png)

## 訓練技巧 Traning Techniques

與 CNN 不同的是 Transformer 使用的一些訓練方法都比較新穎，ResNet 畢竟也是 2015 年的產物了，這個實驗想試試看，如果 ResNet 使用了 DeiT 與 Swin 的方法是不是效果會有所改變？

詳細的改變有：epochs 變 300、使用 AdamW optimizer、使用 Mixup Cutmix RandAugment 等資料擴增、Stochastic Depth、Label Smoothing

最後結果把 ResNet-50 的效果從 76.1% 提升到 78.8% (+2.7%)

詳細的訓練參數：

![image-20220316201155238](https://i.imgur.com/ljMhHe2.png)

## 大架構修改 Macro Design

### 改變各 Stage 的層數比例

在 Swin-T 中一共有 4 個 Stage，分別做 Self-attention 的比例是 1:1:3:1；更大一點的架構則是 1:1:9:1。而原始 ResNet-50 層數比例也從 3: 4: 6: 3 修改成 3: 3: 9: 3

最後結果把 ResNet-50 的效果從 78.8% 提升到 79.4% 

(不過這個性能提升也可能是來自於 FLOPs 的增加…)

### 修改網路最初架構 (stem) 的運算

Swin-T 在網路最一開始做 Patch Embedding，把三維影像轉換為二維序列，而其核心的運算其實是用到了一個 4x4 的大 kernel 來實現的

而 ResNet 的最初的運算稱做 stem 它較為複雜一些，是用一個 7x7 kernel with stride 2，再一個 max pool 來達成

作者直接把 Swin 的做法放到 ResNet 上面，也就是 4x4 kernel with stride 4 (也可看成不重疊的 kernel)。把 Patch Embedding 的想法套用到 ResNet 上面。

最後結果把 ResNet-50 的效果從 79.4% 提升到 79.5% (提升了一點點點而已)

## ResNeXt 化

ResNeXt 引入了 Grouped Convolution，利用**增加網路寬度**的方法來提升效果，而 Grouped Convolution 的極端就是一個 channel 一個 Grouped，而這就是 Depthwise Convolution 的想法。

作者把 ResNet 的卷積層全部換為 Depthwise Convolution，理所當然的因為計算量的下降，最後的效果也下降了，但同時也把經 stem 後的 channel 數量從 64 提升至 96，與 Swin-T 一模一樣

這一加一減的操作下，最後結果把 ResNet-50 的效果從 79.5% 提升到 80.5%

作者在論文中提到：Depthwise Convolution 與 Self-attention 的比較。與之前我有寫過的 [MobileViT]() 有相同的想法，其實這兩個東西是相似的。Depthwise Convolution 是對 kernel 裡面的特徵算加權和，可看成是 local attention，而 Self-attention 則沒有 kernel 的限制，是 global attention。這兩個最的區別在於：Depthwise Convolution 就是固定學習 kernel 中權重，而 Self-attention 因一次看整張圖片，因此權重是動態的。可以參考 Microsoft 的論文 [Demystifying Local Vision Transformer: Sparse Connectivity, Weight Sharing, and Dynamic Weight](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.04263) 有深入的分析。

## Inverted Bottleneck

作者把 Depthwise Convolution 當成是 Self-attention Layer 看，並模仿 ViT 的整體架構。首先採用 inverted bottleneck 用兩個 1x1 conv 放大 4 倍再縮小 4 倍 (下圖 b)，接著把 Depthwise Convolution 層移到第一層輸入處 (下圖 c)，模仿下下圖 ViT 的 Self-attention -> MLP 形式

![image-20220317011954779](https://i.imgur.com/X2jp898.png)

![image-20220317012304797](https://i.imgur.com/Wgpj9Ii.png)

## 增大 kernel size

因為 Swin-T 的 window size 是 7x7，但是自 VGG 提出以來都是使用 3x3，因為有著更低的運算量，及更多非線性轉換。繼然要模仿那就要模仿到底阿，於是作者設計了不同的 kernel 大小 3x3 5x5 7x7 11x11

經實驗發現效果為 79.9% (3×3) -> 80.6% (7×7)，使用 7x7 會使效果變更好 (這是當然在相同層數下使用 7x7 運算複雜度較高)。但是這高計算量被上一步的 inverted bottleneck 圖c 設計兩兩相互抵消

## 其它小改動

### 把 ReLU 換成是 GELU

也把 activation function 換成 NLP 常用的 GELU，作者經實驗發現，在 ConvNeXt 架構下效果差不了多少

### 更少 activation function、normalization 層

以前 CNN 每一個 conv 後都會接一層 BN、ReLU 層，而現在只會在 Depthwise Convolution 後加 LN，在 inverted bottleneck 中加入 GELU。如下圖：

這個操作把效果提高到 81.4% 已經超越了 Swin-T 的效果了

![image-20220317014901729](https://i.imgur.com/XuEdpUH.png)

### 把 BN 換成 LN

BN 的種種缺點我在 NFNet 這篇論文中有提過了，但是在影像上 BN 仍然有它優勢在。把 BN 替換成專為 NLP 設計的 LN，在這篇論文實驗下效果差不多，從 81.4% -> 81.5%

### 修改 downsampling 下採樣的策略

ResNet 中是使用 3x3 with stride 2 來達成減少特徵圖維度，而在 Swin 中是 2x2 conv with stride 2。於是 ConvNeXt 完全模仿 Swin 使用 2x2 conv with stride 2。經實驗證明效果從 81.5% 提升至 82.5%，是個大提升呢

而這個就是最後魔改 ResNet 後的架構 ConvNeXt 了，最後再來一張總表整理一下所有 trick 對應的分類效果與計算量的改動：

![image-20220317015934289](https://i.imgur.com/d8oxmVq.png)

## Experiment

設計了 5 種不同大小的架構，彼此差別僅在於 channal 數的不同及層數重覆的不同。其中 ConvNeXt-T ConvNeXt-B 與 Swin-T Swin-B 計算量是對標的。

![image-20220317020218628](https://i.imgur.com/9kEf89T.png)

ImageNet-1K 分類的 SOTA 表

![image-20220317020417267](https://i.imgur.com/ONCJwTA.png)

ImageNet-22K 分類的 SOTA 表

![image-20220317020434857](https://i.imgur.com/9fXZ457.png)

可發現 ConvNeXt Swin 不管在參數使用量及運算量上都差不大多，但是效果就是好了一些些

## 結論

可以發現 ResNet 經魔改後竟然能與流行的 Transformer 相提並論了， 可謂捲土重來，也可觀察到 CNN 網路還有優化的可能，會不會其實這還不是 CNN 的完全體呢？

另外雖然 ConvNeXt Swin 不管在參數使用量及運算量上都差不大多，兩方面都算不上少了很多，但是在應用工業部署上，大家對於 CNN 的優化及接收度仍效高，已經是很成熟的技術了，相對於 Transformer 大家還沒有一定的優化部署方案，我想在應用上應該還是 CNN 占了不少優勢在

## Reference

[ConvNeXt：全面超越Swin Transformer的CNN (知乎 大推)](https://zhuanlan.zhihu.com/p/458016349)

[The AI Epiphany youtube 解說影片](https://www.youtube.com/watch?v=idiIllIQOfU&t=1783s&ab_channel=TheAIEpiphany)
