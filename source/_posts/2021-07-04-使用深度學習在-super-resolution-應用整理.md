---
title: 使用深度學習在 super resolution 整理 (一)
mathjax: true
date: 2021-07-04 21:42:24
tags: super resolution
categories: 電腦視覺整理
---

因為實驗室最近有人在報 SR 領域相關的論文，於是我也來研究研究一下，倒底深度學習在 SR 發展到什麼地步，以及目前最新的技術是什麼。以下這篇文章會從一些基本的 loss funtion、metrics 開始講起，接著會講從 2016 一直到現在論文倒底改進了哪些地方。這個系列我會分兩篇文章來說，首先先來看看 SR 領域的概論。

keywords: super resolution
<!--more-->

## 問題定義 Problem Definition

在 SR 領域中，我們所要探討的問題是要把 LR 影像 (low resolution) 轉換成 HR 影像 (high resolution)。

通常在實作中實驗順序會先把高解析度 ground truth 經過模糊化得到 LR。如以下的公式：($I_x$ 為 LR，$I_y$ 為 ground truth，D 為模糊公式，$\theta$ 為參數)

$$
I_x = D(I_y; \theta)
$$

接著會經過一個網路，最後生成一張 HR 圖片，公式如下：($I_x$ 為 LR，$I'y$ 為生成的 HR，F 為一個神經網路，$\alpha$ 為參數)

$$
I'_y = F(I_x; \alpha)
$$

最後一步把生成出的 HR ($I'_y$) 與 ground truth ($I_y$) 做比對，或是做 loss funtion 就可以知道網路生成的圖片與原圖相不相近了，可用公式描述如下：($\lambda\Phi(\theta)$ 為正歸項)

$$
\theta' = argmin_\theta(I_y, I'_y) + \lambda\Phi(\theta)
$$

## 評估方法 Image Quality Assessment
要如何用客觀、數學的方式來評斷兩張高解析度圖片相不相近呢？(ground truth vs HR)，有以下三種方法：PSNR，MSE，SSIM。

### MSE
也就是 mean square error，均方差的意思，把兩張圖 pixel py pixel 把每個像素的誤差開平方相加，公式如下：$I_i - I'_i$ 為兩圖相減。

$$
MSE=\frac{1}{N}\sum^N_{i=1}(I_i - I'_i)^2
$$

### PSNR
又稱峰值訊噪比，與 MSE 有相關，做法是把圖片最高的 pixel 除以圖片全部的均方差，數字越大越好，最後再因為人眼的關系再取 log，使得數值變化縮小，通常峰值訊噪比值在 30dB 到 50dB 之間，越接近 50dB 越好。公式如下：($L$ 為圖片中最大的像素值)

$$
PSNR=10*log(\frac{L^2}{MSE})
$$

### SSIM
改進 PSNR 數值太大，人眼反而不準的問題，與 PSNR 一樣，數值越大越相似，SSIM 由以下三項定義出來：
* 兩張影像灰階平均值的差異，沒有一張亮一張暗 (由平均來看)
* 兩張影像的顏色種類分佈 (由標準差來看)
* 兩張影像一致性的變化 (由共變異數來看)

公式如下：(c 的目的是避免除 0)

$$
\begin{gathered}
l(f, g)=\frac{2\mu_f\mu_g+c_1}{\mu_f^2+\mu_g^2+c_1}
\end{gathered}
$$

$$
\begin{gathered}
c(f, g)=\frac{2\sigma_f\sigma_g+c_2}{\sigma_f^2+\sigma_g^2+c_2}
\end{gathered}
$$

$$
\begin{gathered}
s(f, g)=\frac{2\sigma_fg+c_3}{\sigma_f\sigma_g+c_3}
\end{gathered}
$$

## 上採樣方法 Upsampling Methods

在 SR 領域中最重要的就是 upsampling，目的是要把經過 CNN 截取出的特徵圖，慢慢的放大，使特徵可以放近 LR 中，回恢成 HR。而 upsampling 的方法有好幾種，目前有也研究目標在朝向更好的 upsampling 方法，因為目前主流的 upsampling 都會有細節模糊等缺點。

### bicubic
中文可叫做雙三次插值，所謂的插值就是在已知的兩數之間再找出一個新值。bicubic 的計算量大，相比其它的插值法效果比較好，詳細公式參考底下網站。
https://www.codenong.com/cs106567714/

### deconv 反卷積
反卷積就是卷積的相反，把原圖的像素之間補上 0 ，這樣在做完反卷積時圖片就會放大了。反卷積的效果不好，在細節的處理上也很鋸齒。

### sub-pixel convolution
又叫做 pixel shuffle，它的核心思想就是如果今天要把圖片放大 3 倍，我們就要將原圖的 channel 數量從 1 -> 9，經過幾層 conv 的訓練後，將每一層的 channel 有規律的放回原圖，就可得到比原圖大的圖片了
![image-20210706142249603](https://i.imgur.com/QTsyr9h.png)

接下來會來講講從 2016 開始的 SRCNN 一路到現在的網路改進重點。