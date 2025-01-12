---
title: Big Transfer (BiT) - Transfer Learning 的總結
mathjax: true
date: 2021-07-15 11:55:33
tags: pre-training
categories: 電腦視覺整理
---

在 2020 同樣熱門的研究主題還有 pre-training、fine tune 這一個領域，一個 Google 大神又再次以 BiT 這篇論文，結出了一個簡單全面的結論，來看看 pre-training 可以做到什麼程度，效果如何

keywords: Big Transfer、pre-training
<!--more-->

## 介紹
這篇論文非常簡單，全部網路架構使用了 ResNet 50 做為實驗對照，分別在三種不同大小的資料集上實驗，分別是：ILSVRC-2012 (就是 ImageNet 最原始的版本)、ImageNet-21k (ImageNet 加強版)、JFT-300M (傳說是 Google 內部的資料集，未開放)。來看看不同大小的資料集對 pre-training 有什麼影響

## 改進
有三個不同方向的改進：

* scale，不同大小的資料集就用不同大小的網路來訓練，也就是大資料集用大模型、小資料集用小模型
* 使用 Group Norm 以及 Weight Standardization 來替換掉 Batch Normalization。作者認為 BN 對 Transfer Learning 有很大的影響，因為模型太大，所以一次的 Batch 不能設太大，所以 BN 取平均下來反而效果不好
* 使用 BiT-HyperRule，設定了 training schedule, resolution, whether to use MixUp regularization ，三個方面

![image-20210715145932148](https://i.imgur.com/9JvaQln.png)

## pre-training
真的 pre-training 在效果上會比較好嗎？真的會比從頭開始 (Train from scratch) 好嗎？簡單來說結論如下：

自己的模型使否有公開 pre-training 好的參數。若有，選pre-training 任務最接近的參數來用。
若沒有，觀察自己的dataset數量是否夠、是否有足夠的 variance。若足夠全面，可以直接train from scratch，訓練過程的中前段可以留一個 checkpoint 加速未來實驗。
若沒有，挑選 domain 盡量接近的公開 dataset 自己pre-training。

## 結論
pre-training 的確有用，而且在越複雜的模型上要使用越大的資料集來訓練。最後再使用自己的 data 來 fine tune ，不僅效果可能會好一點，但肯定的是，訓練起來的時間一定少了不少。

## Reference
https://medium.com/%E8%BB%9F%E9%AB%94%E4%B9%8B%E5%BF%83/deep-learning-%E4%BD%BF%E7%94%A8pre-training%E7%9A%84%E6%96%B9%E6%B3%95%E8%88%87%E6%99%82%E6%A9%9F-b0ef14e777e9

https://zhuanlan.zhihu.com/p/142864566

https://blog.csdn.net/qq_14845119/article/details/106825219