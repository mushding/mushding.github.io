---
title: RCNN 全家桶速讀：Mask R-CNN
mathjax: true
date: 2022-06-22 19:41:47
tags: 
  - Object detection
categories: 電腦視覺整理
---

接下來來看看由 FAIR 何愷明大神在 2017 改進 Faster R-CNN 提出的 Mask R-CNN，實現了 Instance segmentation 實例分割。除了可以有分割的效果外，也可以知道同類別的不同物體 (例如兩隻不同的狗狗)

原論文：[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)

keywords: Mask R-CNN
<!--more-->

## 網路架構

Mask R-CNN 大部份是架構都是由 Faster R-CNN 改造而來的，簡單的說 Mask R-CNN 在後面的流程中多加了一個分支 mask，下圖為原論文的圖，可以看到中間做完 RoIAlign 後分了兩個分支，上面 class box 是原 Faster R-CNN 架構，而下面則是 mask R-CNN 新設計的架構

![Image](https://i.imgur.com/NGpTdVC.png)

而更詳細的架構圖可參考 [圖解兩階段物件偵測算法_Part09 : Mask R-CNN](https://www.youtube.com/watch?v=5VLI_gbpocE&ab_channel=WilsonHo) 影片簡報中的附圖

圖中上半部分支與 Fast R-CNN 一致，會產生 box 及 cls 結果，不同的部份在多了下面的 FCN (Fully Convolutional Network)，經過一串的 CNN 組合最後得到一個 14x14x80 的 feature map (80 的原因是 COCO Dataset 會分 80 類)

![Image](https://i.imgur.com/02ABxWS.png)

那最後分割的結果是怎麼來的呢？首先會從 cls 中知道物體是第 k 個類別，再從對應的 14x14x80 中選出第 k 個 14x14 feature map，再依據 RolAlign 的變形大小把 box 放回原本的影像大小，最後就是結果了。

而以下是原論文提供的 FCN (ResNet)

![Image](https://i.imgur.com/IS4qaxp.png){ width=50% }

作者也有做使用 FPN 的實驗，效果比 FCN 好上 4 個百分點

![Image](https://i.imgur.com/rJIWyAr.png){ width=50% }

![Image](https://i.imgur.com/R5Yz7D6.png){ width=50% }

## RoIAlign

另一個改動較大的地方是把 RoI Pooling 更換成更複雜的 RoIAlign。而它的核心概念其實很簡單，用一句話帶過：取整的部份從去小數點變成雙線性內差。

![Image](https://i.imgur.com/tXZveNN.png){ width=50% }

原本的 RoI Pooling (假設是 2x2 pooling) 做法是直接把長寬的小數點去掉，得到一個大框框，接著再分割出 4 塊 pool，如果還是不整除，再把小數點去掉。這一來一往去掉了兩次小數點，與原框框的誤差變得非常大

而 RoIAlign 則是使用雙線性內差來解決小數點全丟棄的問題，一、大框框的小數點不去掉，二、一個 pool 的值是從 pool 取等分的 4 個點，而每個點都是從附近像素做雙線性內差而來的，最後再對這個 4 個點找 max 

![Image](https://i.imgur.com/nlgD6OF.png){ width=50% }

作者在後續做實驗比較，可看到做了 RoIAlign 提升的百分比非常多，可見去掉小數點是很傷的事情。(不過要記得，當運算量上升時，理論效果好是應當的，但這個上升的比例好像很值得呢 XD)

![Image](https://i.imgur.com/8txHiM0.png){ width=50% }

## 網路訓練

那網路的 Loss 如果設計呢？除了原本 Faster R-CNN 的 $\mathcal{L}_{cls}$ $\mathcal{L}_{box}$ 外，加上了來自 mask 的 $\mathcal{L}_{mask}$，它的算法精神在：把 Ground truth 的像素與由 feature map 生成的像素做簡單的 L2 loss

## 一些網路效果

![Image](https://i.imgur.com/D0LKUG1.png)

## 結論

Mask R-CNN 改進了 Faster R-CNN 許多小細節，同時也引入了分割的想法，對於未來的偵測分割網路功不可沒

## Reference

[圖解兩階段物件偵測算法_Part09 : Mask R-CNN](https://www.youtube.com/watch?v=5VLI_gbpocE&ab_channel=WilsonHo)