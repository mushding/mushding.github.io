---
title: >-
  CNN 與絕對位置資訊 - CNN 倒底學到了什麼？
mathjax: true
date: 2021-07-30 14:13:44
tags: Vision Transformer
categories: 電腦視覺整理
---

在上一篇 Transformer 中我們提到作者使用 zero padding 來當作位置資訊的考量，在這一篇文章中我引用了兩篇論文來更進一步了解一下，CNN 與絕對位置之間的關系。分別是 Uber 提出的 coordConv 以及一篇專門解釋 zero padding 的文章。

[https://arxiv.org/abs/1807.03247 (coordConv)](https://arxiv.org/abs/1807.03247)

[https://arxiv.org/pdf/2101.12322.pdf (zero padding)](https://arxiv.org/pdf/2101.12322.pdf)

keywords: zero padding、coordConv
<!--more-->

## Introduction

我們都知道 CNN 卷積神經網路有一大特性：平移不變性 (translation invariant)，也就是說，圖中的特徵點不管位置在哪、大小多大，對於機器而言都是一樣的特徵。在分類任務上這個特徵似乎帶給我們很大的幫助，因為不管目標物出現在圖中任何位置，機器都會把視為相同特徵 (例如：不同地方的兩台車)

![Image](https://i.imgur.com/onTxnsC.png)

而在另一個分割領域情況就稍微比較複雜一點了，分割分為三種：語意分割 (Semantic segmentation)、實例分割 (Instance segmentation)、全景分割 (Panoramic Segmentation)。如下圖：

![Image](https://i.imgur.com/UsG2YNH.png)

簡單來說語意分割 (Semantic segmentation) 是依照「像素」級別來分割的，把每一個像素對應一個類別，就可以達成類似分割的效果了。語意分割比較簡單

但實例分割 (Instance segmentation) 就比較困難了，是把每個「物件」都分離出來，就算是同類別也是一樣，概念有點類似分類 + 語意分割的結合。

全景分割 (Panoramic Segmentation) 更困難，是加上了背影的實例分割。

## coordConv

首先 Uber 在 2018 提出相關問題：CNN 在絕對位置上的能力沒有很好。而論文中真正實驗討論的問題是將直角座標系轉換成 one-hot 的能力。如以下影片所述

[https://www.youtube.com/watch?v=8yFQc6elePA&t=1s](https://www.youtube.com/watch?v=8yFQc6elePA&t=1s)

可以看到論文中最原始的想法就是要實作出一個可以在直角座標系轉換成 one-hot 的網路

![Image](https://i.imgur.com/GUmUB3h.png)

但是做出來的效果非常差，於是 coordConv 的核心想法就是：在特徵層多加兩層，分別為 x 軸座標以及 y 軸座標

![Image](https://i.imgur.com/xRe9gdk.png)

這兩層座標想法非常的直接，就是直接加入了 0 ~ 1 之間的數字，新增在最後兩層特徵圖中，而當這兩層特徵圖全為 0 時，就等同於原始的 CNN 網路。

藉由人工加入了「絕對座標」資訊，網路在「生點點 one-hot」的能力上有顯著的進步

![Image](https://i.imgur.com/s0fai2C.png)

## zero padding

而另外一篇論文則是在討論，其實不用像 coordConv 那像人工加上絕對位置資訊，CNN 本身就好像自帶有這種能力了，只是以前大家都不是很清楚倒底是怎麼來的，反正「it just works！」

有在做實例分割的人心中應該都有一個疑問：那就是 CNN 倒底是怎麼知道同一類不同位置的物件？還可以成功得把它分割出來呢？如同實例分割始使論文中提到的觀念：可參考以下論文

[Semantic Instance Segmentation with a Discriminative Loss Function (CVPR2017)](https://arxiv.org/abs/1708.02551)

論文中寫到透過一個 Loss function 使得「同一個實例的像素更加靠近、不同的實例像素盡可能地遠離」。嗯…這下子就神奇了，CNN 是怎麼知道它是不同的實例阿，不是有平移不變性嗎？那為什麼效果還不錯呢？會不會是…CNN 透過某種神秘的方法學習到了有關位置的資訊，使得機器知道同一類不同實例的物體？

因此本篇論文設計了一串實驗來證實：是的！以前大家都沒有想到，但是 CNN 是天生具有學習位置的能力的！而關鍵就發生在 zero padding 上面！以下介紹論文：

## Experiment

使用的方法是輸入一個雜訊圖，目標要先出對應的座標圖 (像是下圖中的黃綠圖)，可看到 VGG、ResNet 輸出效果皆帶有位置的資訊

![Image](https://i.imgur.com/Dm1mawX.png)

作者認為 CNN 之所以會有絕對位置資訊 (absolute position) 的原因是因為 zero padding。zero padding 最初是用來使 CNN 的輸入輸出維度相同而設置的，但在不經意間 zero padding 會透露出邊邊、角等資訊，為了證實這一件事情，作者設計了有做 padding 以及沒做 padding 的實驗看看誰效果比較好：

![Image](https://i.imgur.com/b7hqfCG.png)

## 結論

其實這篇論文設計了非常多實驗，我也沒有很認真的把每一個看完，但最重要的結論就是：zero padding 這一步使得 CNN 有了微微的絕對位置資訊能力。而在往後的其它論文中提出其它更好的方法來解決絕對位置的問題中，也可發現加上絕對位置資訊後的效果大概是從 80 分到 100，而非 0 分到 100 分，這也是 zero padding 在幕後默默的推了一把的關系吧！

## Reference

https://medium.com/ching-i/%E5%BD%B1%E5%83%8F%E5%88%86%E5%89%B2-image-segmentation-%E8%AA%9E%E7%BE%A9%E5%88%86%E5%89%B2-semantic-segmentation-1-53a1dde9ed92

https://medium.com/ching-i/%E5%BD%B1%E5%83%8F%E5%88%86%E5%89%B2-image-segmentation-%E5%AF%A6%E4%BE%8B%E5%88%86%E5%89%B2-instance-segmentation-1-2a796c4fa738

https://www.codenong.com/cs105241864/

https://zhuanlan.zhihu.com/p/99766566

https://zhuanlan.zhihu.com/p/39919038

https://blog.piekniewski.info/2018/07/14/autopsy-dl-paper/