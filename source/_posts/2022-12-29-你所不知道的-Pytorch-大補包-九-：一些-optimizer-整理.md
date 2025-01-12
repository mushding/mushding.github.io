---
title: 你所不知道的 Pytorch 大補包(九)：一些 optimizer 整理
mathjax: true
date: 2022-12-29 00:35:50
tags: Pytorch
categories: Pytorch 大補包
---

本篇筆記主要參考以下網路文章：[https://zhuanlan.zhihu.com/p/22252270](https://zhuanlan.zhihu.com/p/22252270)

整理了一些常用 optimizer 的數學原理，及其重點特色

keywords: optimizer
<!--more-->

## SGD

* stochastic gradient descent

$$
g_t = \nabla_{\theta_{t-1}}f(\theta_{t-1}) \\
\Delta\theta_t = - \eta * g_t
$$

* $\eta$ 是 learning rate
* SGD 完全依賴目前梯度的斜率大小
* 遇到鞍點等地方會不容易達到最優
* 且 SGD 整體更新速度慢

## Momentum

* 模仿物理中的動量
* 把之前算出來的梯度大小一起放到這一次的運算

$$
m_t = \mu*m_{t-1} + g_t \\
\Delta\theta_t = -\eta * m_t
$$

* 相較於 SGD 更新速度快
* 在梯度改變方向的時候，$\mu$ 可以減少更新，抑制振盪

## Nesterov

* nesterov 在梯度更新時做一個校正，避免前進太快，同時提高靈敏度，與 momentum 有點像
* 由公式可以看出 momentum 沒有更改當前梯度 $g_t$
* 於是在 Nesterov 中就是透過修改 $g_t$ 來達到修改的目的

$$
g_t = \nabla_{\theta_{t-1}}f(\theta_{t-1}-\eta*\mu*m_{t-1}) \\
m_t = \mu * m_{t-1} + g_t \\
\Delta\theta_t = -\eta*m_t
$$

雖然 momentum nesterov 都是為了增加梯度更新時的彈性，但人工設定還不如用機器自己來學習

以下介紹機器自己學習的方法

## Adagrad

* 是對 learning rate 設定了一項限制
* $\epsilon$ 用來保證分非 0
* 把 $\eta$ 除上一個值使得
* 前期 $g_t$ 較小的時候，regularizer 比較大，能夠放大梯度
* 後期 $g_t$ 較大的時候，regularizer 比較小，能夠約束梯度
* 缺點：
* 仍要人工設定 learning rate
* $\eta$ 設太大的話，會讓 regularizer 過於敏感，對梯度改變太大

$$
n_t = n_{t-1} + g^2\\
\Delta\theta_t = -\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t
$$

## Adadelta

* 是 Adagrad 的進階版
* 只累加固定大小的項

$$
n_t = v*n_{t-1} + (1-v) *g^2_t \\
\Delta\theta_t = -\frac{\eta}{\sqrt{n_t+\epsilon}} * g_t
$$

* 在經過作者一系列，近似牛頓迭代法的方法後
* 可以實現機器自動學習 learning rate

## RMSprop

* 算是 Adadelta 的變形

$$
E|g^2|_t = \rho * E|g^2|_t-1 + (1 - \rho) * g^2_t \\
RMS|g|_t = \sqrt{E|g^2|_t + \epsilon} \\
\Delta\theta_t = -\frac{\eta}{RMS|g|_t} * g_t
$$

* RMS 均方根，作為 learning rate 的約束
* 仍然是人工固定的 learning rate

## Adam

* 就是帶有 Momentum 的 RMSprop
* 因為 m n 是變數，所以梯度可以動態調整

$$
m_t = \mu * m_{t-1} + (1-\mu)*g_t \\
n_t = \mu * n_{t-1} + (1-v)*g_t \\
\hat{m}_t = \frac{m_t}{1-\mu^t} \\
\hat{n}_t = \frac{n_t}{1-v^t} \\
\Delta\theta_t = -\frac{\hat{m}_t}{\sqrt{\hat{n}_t} + \epsilon} * \eta
$$