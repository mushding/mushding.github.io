---
title: >-
  Vision Transformer 演化史: Pyramid Vision Transformer: A Versatile Backbone for
  Dense Prediction without Convolutions - 把金字塔網路應用在 Transformer
mathjax: true
date: 2021-08-17 09:57:19
tags: Vision Transformer
categories: 電腦視覺整理
---

這篇論文是南京大學、香港大學在 2021 2 月提出的，這篇論文提出了 Pyramid Vision Transformer (PVT) 架構，其實就是把 CNN 已經非常廣泛使用的概念搬到 ViT 上面來。主要創新點包含兩點：Progressive shrinking stategy 加入金字塔網路、Spatial Reduction Attention 減少運算量。

[https://arxiv.org/pdf/2102.12122.pdf](https://arxiv.org/pdf/2102.12122.pdf)

keywords: PVT、Progressive shrinking stategy、Spatial Reduction Attention
<!--more-->

## 1. Introduction

![Image](https://i.imgur.com/vrcjsZQ.png)

有鑒於 CNN 在電腦視覺的成功，PVT 提出的動機希望能把已經在 CNN 非常成功的概念 Feature Pyramid Network (FPN) 應用在 Transformer 上面，藉此更善 ViT 的一些缺點：

1. **加上多重解析度**：不同於 ViT 低解析度輸出、高運算複雜度，PVT 可以得到更高解析率的輸出
2. **減少運算**：如同 FPN 一樣會慢慢減少特徵圖數量，減少運量，改善 ViT 遇到解析度大圖片時運算量會爆增的問題
3. **增加應用範圍**：傳統 ViT 只能用在分類任務上，PVT 不但也能分類，也因為有多重解析度，因此也能運用在辨識、分割任務上

下圖為不同網路架構能做的電腦視覺任務比較圖：

![Image](https://i.imgur.com/O69770a.png)

## 2. 網路架構

### 整體架構

整體架構圖如下：

![Image](https://i.imgur.com/iJ2sJ1d.png)

作者為了模仿 FPN 的多重解析度，因此本論文的 PVT 架構設計了四個階段用於生成不同解析度的特徵，每個階段的操作都相同，包含兩個步驟：**Patch Embedding、Transformer Encoder**，步驟相同但是圖片的解析度會隨著網路而慢慢加深

整體架構文字流程如下：

* 首先會輸入一張 $H\times W\times 3$ 的影像
* 與 ViT 的 Patch 大小為 16x16 不同，PVT 的 Patch 大小設為 4x4
* 接著把三維的圖片 $H\times W\times 3$ reshape 至二維 $HW/4^2 \times C_1$
* 把二維序列放進 Transformer 中
* Transformer 輸出的結果 $HW/4^2 \times C_1$ reshape 回 $H/4 \times W/4 \times C_1$

$$
\begin{gathered}
H\times W\times 3\\
\rightarrow HW/4^2 \times C_1\\
\rightarrow H/4 \times W/4 \times C_1
\end{gathered}
$$

作者在論文提出 Feature Pyramid for Transformer 以及 Transformer Encoder 來詳細介紹架構

### Feature Pyramid for Transformer

與 ViT 提出的 Patch Embedding 不同，ViT 中的 Patch Embedding 只有在網路的一開始出現，而 PVT 中的 Patch Embedding 會在每一個 Stage 中出現 (在這篇論文舉的例子一共出現 4 次)。

而在 PVT 中這些 Patch Embedding 擔任了 progressive shrinking 重要的責任，負責把 Transformer 中的特徵圖慢慢減少圖片大小、增加特徵圖

透過這樣在每個 Encoder 前做一次 Patch Embedding 的方法，就可以人為的控制我們想要的各種不同解析度了

主要網路公式如下：

輸入前一網路特徵圖 $F_{i-1}$ ，經 reshape 至二維，經 Transformer 得到二維結果，接著 reshape 回三維。特別注意是在這篇論文中 Patch $P$ 設為 4 or 2 (可參考 Experiment)。最後經過一個 LayerNorm 即為最後結果

$$
\begin{gathered}
F_{i-1} \in \mathbb{R}^{H_{i-1}\times W_{i-1} \times C_{i-1}}\\
\rightarrow \frac{H_{i-1} W_{i-1}}{P^2_i}\times C_i\\
\rightarrow \frac{H_{i-1}}{P_i}\times \frac{W_{i-1}}{P_i} \times C_{i-1}
\end{gathered}
$$

與 ViT 的 patch embedding 中的 source code 相同，在 pytorch 中的實作方法就是使用 kernel size 、 stride 皆為 Patch $P$ (4) 的 Conv2d

```python=
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

### Transformer Encoder

對於每一層的 Encoder PVT 也有做一些調整。由於圖片的解析度會越來越大，需要的運算也會放大，為了解決運算量的問題，作者提出了 SRA(spatial-reduction attention) 代替原本的 MHA(multi-head-attention)

![Image](https://i.imgur.com/gDRAYaY.png)

解決方法也很簡單，把網路中的 K 、 V 的維度縮小，再放進 MHA 中做計算。

這一步與 TNT 中的做法相同，一樣是通過減少 K V 的長度來減少運算量，且效能不會減少太多 (可看 TNT 實驗)。至於為什麼減少 K V 對效果不會影響太多嗎…目前我還不清楚 XD

公式如下：

$$
\mathrm{head}_j = \mathrm{Attention}(QW^Q_j,\mathrm{SR}(K)W^K_j,\mathrm{SR}(V)W^V_j)
$$

## 3. Experiments

### 網路架構

參考 ResNet 設計了四個 Stage，特徵圖放大了 32 倍，並且也有四個不同大小的網路 (好玩的是 Stage 重複次數跟本與 ResNet 一模一樣 XD，同樣用到了越深重覆次數越多的概念)

![Image](https://i.imgur.com/d4R63cK.png)

由於有多重解析度，Transformer 系列終於不只能做分類了，於是作者與分類、偵測、語義分割、實例分割都來比較了一下

### 與分類比較

使用 ImageNet 來做比較，實驗發現效果比 CNN 好，比 ViT 好，沒有 TNT T2T-ViT 好，但是參數量與運量少非常非常多，證明 CNN 的多重解析度可以非常有效率的截取特徵

![Image](https://i.imgur.com/JDMZ5tR.png)

### 與偵測比較

![Image](https://i.imgur.com/QQbp295.png)

### 與實例分割比較

![Image](https://i.imgur.com/50I4xEw.png)

### 與語音分割比較

![Image](https://i.imgur.com/WbXqa14.png)

## 結論

由以上實驗可證實，在不同參數設置下，PVT 的效果皆比 ResNet、ResNeXt 還要來得好，尤其在分割上面 Transformer 更關注全局，這個特性對分割來說是個非常有效的，因此效果比想像中好，未來也可以試著往這方向結合。

總而言之，PVT 試著把 FPN 與 Transformer 結合，並且把 Transformer 能完成的任務大大的拓展了，不再只能用來分類。且套用了 CNN 的概念參數量有大大下降的趨勢。

## Reference

https://mp.weixin.qq.com/s/LCLQltmBxL9f1XzV4Ci-iw

https://www.jianshu.com/p/d2a878723af4

https://blog.csdn.net/P_LarT/article/details/114157235