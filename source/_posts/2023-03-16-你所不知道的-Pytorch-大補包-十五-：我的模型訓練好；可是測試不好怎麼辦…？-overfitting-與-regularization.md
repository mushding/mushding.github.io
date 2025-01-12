---
title: 你所不知道的 Pytorch 大補包(十五)：我的模型訓練好；可是測試不好怎麼辦…？- overfitting 與 regularization
mathjax: false
date: 2023-03-16 16:56:28
tags:
categories:
---

overfitting、underfitting  
這兩個詞相信有在碰深度學習人一定都不陌生，學校裡有都有教。但是在實作中，遇到什麼樣子的情況可以稱作 overfitting？網路會有怎樣的表現？下一步要怎麼來解決？

以下文章會把目光放在 overfitting 上來講解

keywords: Overfitting、Regularization、Weight Decay、Label Smoothing、Warmup
<!--more-->

## 什麼是 overfitting、underfitting

在深度學習中會使用 Loss 表示網路找到的迴歸區線與現實資料分佈的差異，並且利用 Loss 進一步算出梯度後更新參數，使網路更符合現實資料的分佈。

在實作中會把資料集分為三種：訓練集 Training Set、驗證集 Validation Set、測試集 Testing Set，不同的資料集會有著不同的資料分佈，但理論上因為是從同一筆資料分出來的，所以彼此之間應該不會差太多。

Underfitting 的意思是：訓練得很不好 (訓練 Loss 高)。  
Overfitting 的意思是：訓練得很好 (訓練 Loss 低)，可是測試時不好 (測試 Loss 高)。

如下圖：左圖是 underfitting，中圖是正常，右圖是 overfitting。

<img src="https://i.imgur.com/i0fDKv1.png" alt="Image" />

用下面的網站來進一步解釋 (這是一個簡單的迴歸線視覺化網站，裡面有很多東西可以自定義，可以解釋很多深度學習的一些現象)：  
[Tinker With a Neural Network Right Here in Your Browser.](http://playground.tensorflow.org/)

Underfitting 的意思是，訓練 Loss 還太高，網路迴歸的能力還沒有很好，常發生在：

- 網路訓練初期
- 網路架構太淺

<img src="https://i.imgur.com/92QDjeE.png" width="50%" height="50%" />

而 Overfitting 的意思是，訓練 Loss 很好、網路在訓練資料集有著很強的能力，可是面對新的驗證資料分佈時，反而效果變很差，驗證 Loss 很高。

最明顯的特徵是：網路訓練到後期，驗證 Loss 與訓練 Loss 有一段小差距，甚至這個差距還會越來越大，驗證 Loss 不斷的在上升。

<img src="https://i.imgur.com/wjlyCLJ.png" width="50%" />

再舉一個我的親身經驗：下面是我其中一個實驗訓練與驗證 Loss 的曲線圖：紅框的部份很明顯訓練跟驗證間隔拉大了。

<img src="https://i.imgur.com/h3nhJmv.png" alt="Image" />

## 如何解決 Overfitting

相較於 Underfitting，Overfitting 的成因複雜很多，不過可以總結成一句話：**網路泛化能力 Generalization 不好的時候會發生**，也就是網路只要換一個資料集就沒用了，完全沒有自行推論未見資料的能力。

...

## Reference

- [IBM What is overfitting? (總覽)](https://www.ibm.com/topics/overfitting)
- [ML | Underfitting and Overfitting (GeekforGeeks)](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/)
- [Regularization 方法 : Weight Decay , Early Stopping and Dropout (weight decay 公式推導)](https://hackmd.io/@allen108108/Bkp-RGfCE)
- [[Day27] Weight Decay Regularization](https://ithelp.ithome.com.tw/articles/10306518)
- [[Day25] Label Smooth](https://ithelp.ithome.com.tw/articles/10305524?sc=iThelpR)
- [好玩的網路訓練模擬網站](http://playground.tensorflow.org/)
