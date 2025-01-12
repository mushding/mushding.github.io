---
title: 你所不知道的 Pytorch 大補包(十三)：我可以在 optimizer 中動態的調整學習率嗎？- RMSProp、AdaGrad
mathjax: true
date: 2023-03-16 16:44:52
tags: Pytorch
categories: Pytorch 大補包
---

在上章我們介紹了 SGD 與 Momentum，接下來進一步介紹可以自己調整學習率的 RMSProp 與 AdaGrad

keywords: RMSProp、AdaGrad
<!--more-->

## 為什麼要自己調整學習率？

剛剛加入 Momentum 的 SGD 似乎看起來很完美，收斂又快，又有跨過小山丘的能力，那…還有什麼地方可以改進的呢…？

我們一起來看下面這張圖，來源自：[Setting the learning rate of your neural network.](https://www.jeremyjordan.me/nn-learning-rate/)

假設我們的網路是一個類似二次多項式的曲線

> 當我們學習率設太小：收斂太慢 (左圖)
> 學習率設太大：完全找不到最低點，一直跳來跳去 (右圖)
> 學習率設得剛剛好：完美！(中圖)

可見如何選擇學習率是一個重要的課題，其影響程度甚至可以使你的網路永遠不收斂，效果就是比別人差。

那倒底要選擇多大的學習率呢？答案是：我也不知道…，每一個網路有著他自己的特性，所以每個網路最佳的學習率都不太一樣，所以最好的做法就是：學習率不是固定的！而是一個從大慢慢變小的過程。

由剛剛的圖可以得知通常網路在訓練初期梯度較大因此可以設較大的學習率，而隨著網路訓練慢慢的收斂，學習率也要隨之調整變小，以適應較緩的梯度。

而至於從什麼時候開始變小，則是根據網路自己的權重來動態的決定，自己決定自己的學習率最合理，人為定的都猜不準

以下兩個優化器就是按著這個思維來設計，希望可以利用網路自身的權重值來自己決定學習率要如何變小。

## 最簡單的想法

介紹前我們先來看最最簡單的想法，由此出發，更能體會到下面的優化器想要解決什麼事情

**學習率會隨著 epoch 增加而變小**，這是核心中的核心概念，那既然是跟時間有關，我們可以把學習率與時間成反比就好了呀，可以得到下面的公式：

$$
w_{t+1} = w_t + \frac{\eta}{t} \nabla g
$$

直接把學習率除以時間 t，這樣學習率就會隨著時間慢慢的變小了！不過這樣子真的就好了嗎？式子中的時間 t 好像跟網路一點關聯也沒有，不同的網路學習率的變化基本是一模一樣，所以剛剛的那一句要稍微改一下

**學習率會根據網路權重且隨著 epoch 增加而變小**，AdaGrad 及 RMSProp 就是在討論網路權重對於學習率的影響。以下介紹兩種優化器

## AdaGrad

AdaGrad 全名 Adaptive Gradient，其想法是在網路初期干預不多因此學習率大；網路後期干預多因此學習率小。

公式如下：

$$
\begin{gather}   
w_{t+1} = w_t - \frac{\eta}{\sigma_t} \nabla g\\
\sigma_t = \sqrt{G_t+\epsilon}\\
G_t = \sum^t_{n=1}g_n^2
\end{gather}
$$

$G_t$ 代表權重值，累積到第 t 時刻的梯度平方和，$\epsilon$ 是平滑項 (smooth term) 用於避免 $\sigma$ 為 0 否則會除 0，一般設為 $10^{-8}$

AdaGrad 使用**網路加權到 t 時刻的權重平方和**來做為除以學習率的分母，因為會隨時間加權的原因，學習率這一項會越來越小，直到接近 0。也可理解為網路越後期優化器干預的越多，學習率因此降低

AdaGrad 的優點是不需人工調整學習率；而缺點是收斂到最後，調整多，學習率幾乎降為 0，而無法再改進參數值

在 Pytorch 中 AdaGrad 可以很方便的直接呼叫函式庫就可以囉，基本上沒有什麼超參數要特別調

<img src="https://i.imgur.com/65ZEpjx.png" alt="Image" />

## RMSProp

RMSProp 是 Hinton 教授在上課的講義中提定的一個優化器，並沒有正式發表在論文當中。

公式如下：

$$
\begin{gather}
w_{t+1} = w_t - \frac{\eta}{\sigma_t} \nabla g\\
\sigma_t = \sqrt{\alpha(\sigma_{t-1})^2+(1-\alpha)g_t^2+\epsilon}
\end{gather}
$$

RMSProp 與 AdaGrad 基本上差不多都是學習率 $\eta$ 除上一個由權重決定的分母 $\sigma$，$\sigma$ 同樣是由當前梯度平方來決定，但是多了一個超參數 $\alpha$。

分母的意思為：除了加總當前梯度平方和之外，也考慮前一個時刻的梯度平方和。

實作上 $\alpha$ 會設為 0.9，代表當網路後期，優化器干預學習率越多時，偏好使用舊梯度做平方和運算。

這樣做相比於 AdaGrad 計算 1 ~ t 時刻的梯度平方和，每一個時刻的權重值都會加起來，RMSProp 因設定 $\alpha=0.9$ 偏好使用舊梯度，做到類似加權平均的概念，可以避免 $\sigma$ 值過大的問題。

同時 RMSProp 這個概念也很像動量 Momentum，在更新權重前除了當前的權重值外也考量前一時刻的權重值，使得 RMSProp 相比 AdaGrad 在梯度曲面較複雜的情況也有著比較好的表現。

在 Pytorch 上有 RMSProp 的實作函式：其中 alpha 參數預設 0.99 代表高度依靠歷史梯度來更新參數

<img src="https://i.imgur.com/zoCLYKc.png" alt="Image" />

## Reference

[推！。Setting the learning rate of your neural network.](https://www.jeremyjordan.me/nn-learning-rate/)

[Adagrad、RMSprop、Momentum and Adam – 特殊的學習率調整方式](https://hackmd.io/@allen108108/H1l4zqtp4)
