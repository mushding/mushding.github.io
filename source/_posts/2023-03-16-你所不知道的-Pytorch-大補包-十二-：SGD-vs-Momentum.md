---
title: 你所不知道的 Pytorch 大補包(十二)：一切的開端 - SGD vs Momentum
mathjax: true
date: 2023-03-16 16:43:39
tags: Pytorch
categories: Pytorch 大補包
---

在以前第九章中，有很 ~ 淺的列舉了一些優化器 optimizer，在第十二 ~ 十四章中，會更詳細一點去介紹，這些 optimizer 的原理，以及當初提出是要改進什麼事？

keywords:SGD、Momentum
<!--more-->

### 梯度下降 Gradient Desent

還記得在第十章中有介紹了什麼是損失 Loss、什麼是梯度，以及網路是如何利用梯度來找到最佳解嗎？如果忘記的話可以來這邊複習喔 [你所不知道的 Pytorch 大補包(十)：Pytorch 如何實做出 Backpropagation 之什麼是 Backpropagation](https://mushding.space/2022/12/29/%E4%BD%A0%E6%89%80%E4%B8%8D%E7%9F%A5%E9%81%93%E7%9A%84-Pytorch-%E5%A4%A7%E8%A3%9C%E5%8C%85-%E5%8D%81-%EF%BC%9APytorch-%E5%A6%82%E4%BD%95%E5%AF%A6%E9%A9%97-Backpropagation-%E4%B9%8B%E4%BB%80%E9%BA%BC%E6%98%AF-Backpropagation/)

在裡面提到了網路中的函數非常複雜，複雜到我們沒辦法用一般多項式的方法來求解，所以我們將損失函數對權重做一階微分，得到網路的梯度。利用梯度下降法，一步步縮小 Loss，像在山坡地上滑溜滑梯一樣，滑到最低點，就可以找到最接近真實答案的結果了。

梯度公式如下，在深度學習中我們稱這個符號 $\nabla$ ，代表梯度的意思

$$
\nabla g = \frac{\partial\mathcal{L}}{\partial w}
$$

而梯度下降法則是利用梯度的值來修改權重 $w$，其中 $w_t$ 代表目前的權重，$w_{t-1}$ 代表上一次的權重，公式如下：

$$
w_{t}=w_{t-1}-\nabla g
$$

### 什麼是優化器 optimizer

所謂優化器指「優化」網路做梯度下降的「速度」或「效果」，也就是說剛剛介紹的梯度下降其實還存在著許多的缺點，例如：收斂時間久、效果不穩定…等

而一個最最簡單概念的優化器 (這個概念是我自己想的，有些人可能不這麼覺得…) 就是學習率 learning rate，符號通常表示 $\eta$

學習率設計用來控制梯度大小用，因為通常梯度算出來都很大，所以學習率會設介於 0.1 ~ 0.0001 的區間來縮小梯度計算結果，詳細可看第十章實驗，實驗結果可知如果不加學習率，梯度會超大，網路永遠都不可能會收斂

加入學習率的梯度下降公式如下：

$$
w_{t}=w_{t-1}-\eta\nabla g
$$

像這種找到梯度下降的缺點，並加以改進的方法，就可以稱作為一種優化器。

### SGD

快速複習完梯度下降 (Gradient Desent, GD) 後，緊接來介紹應用最廣、最穩定，也最元老的優化器：SGD

SGD 全名為 Stochastic Gradient Descent，中文稱：隨機梯度下降法

其實它跟剛剛上面我們介紹的加入學習率後的公式一模一樣，只是在「計算對象」及「方法」做了一點點的小修改，而這個故事中間有一點點關於歷史淵源，下面做簡單的介紹：

理論上的梯度下降會把**全部**的資料都看過一遍之後，用**全部**的資料去計算梯度，並更新一次參數。

但理想很豐滿；現實很骨感，在現實中我們的資料集又大又多，動輒幾 G 甚至幾 T 起跳的，實作上沒有辦法暫存下這麼多資料，然後再一次更新的，於是有人提出 Mini-Batch Gradient Desent，我們不看完整個資料集更新一次參數，而是設定一個 mini batch 的數值 (其實就是現在說的 batch，可能以前的人覺得 256、512 這些數字相對於全部的資料集來說，數字小了不少)，