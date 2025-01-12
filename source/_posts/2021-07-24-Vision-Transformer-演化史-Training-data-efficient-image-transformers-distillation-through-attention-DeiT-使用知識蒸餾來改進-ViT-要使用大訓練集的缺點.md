---
title: >-
  Vision Transformer 演化史: Training data-efficient image transformers &
  distillation through attention - DeiT 使用知識蒸餾來改進 ViT 要使用大訓練集的缺點
mathjax: true
date: 2021-07-24 16:15:49
tags: Vision Transformer
categories: 電腦視覺整理
---

讀完 Google 發表的 ViT 論文後，不禁讓人覺得：哇塞這樣也行！，直接把圖片用一個字串來表示放進 Transformer 中。然而在原論文中也明確提到了：「that transformers do not generalize well when trained on insufficient amounts of data.」，意思即是在資料集不大的情況下 Transformer 的效果是比 CNN 還是來得差的，因此 Google 大神使用了 JFT-300 這個資料集做 pre-training ，但…Google 沒跟你說的是，這個資料不公開阿。因此 Facebook 提出 DeiT 模型，使用 distillation 的方法只需要使用 ImageNet 就可以有不錯的效果。

https://arxiv.org/pdf/2012.12877.pdf

keywords: DeiT, distillation
<!--more-->

## 1. Introduction
作者提出了一新架構叫 Data-efficient image Transformers，簡稱 DeiT。旨在用更少的資料集也能達到 CNN sota 的效果。

DeiT 一共有下列幾項特色：

1. 整個架構不使用任何 CNN，全為 Transformer，在同樣都是使用 ImageNet 來訓練的前提下，與純 CNN 的 sota 效果不相上下。
2. 提出了一個根據 Transformer 設計的 distillation 流程，用上了叫 distillation token 的東東。distillation token 會與 class token 在 Transformer 中不斷的交互做計算。使結果更好 (後續會細講)

下圖為論文中使用 ImageNet 做訓練 ImageNet 做測試的結果，可發現 ViT 在小資料集的效果確實不太出色。

![Image](https://i.imgur.com/r083nae.png)

## 2. Distillation through attention

與一般我們認識的 Distillation 不一樣的是，DeiT 提出了一個新的 Distillation token 流程，並且比較了 Soft Distillation 與 Hard-label Distillation 的差別、及傳統 Distillation 與 Distillation token 的差別

讓我們先回想一下 ViT 的做法…，ViT 在 Encoder 輸入字串長度 $N$ 的地方改成 $N+1$ 並取名叫 class token，而其餘的 $N$ 取名叫 patch token，class token 目的在於輸出分類結果，最後直接經過一個 softmax 就是最後輸出了。

![Image](https://i.imgur.com/w1oU12e.png)

### Soft Distillation vs Hard-label Distillation
Distillation 分為兩種一是「軟蒸餾」Soft Distillation、一是「硬蒸餾」Hard-label Distillation

**Distillation**：首先最原始的蒸餾指的是：Student model 經 softmax 後的結果與 ground truth 做 cross entropy：$\mathcal{L}_{CE}$ 為 cross entropy、$\psi$ 為 softmax、$y$ 為 ground truth、$Z_s$ 為 student 的 logits

$$
\mathcal{L}_{Distillation} = \mathcal{L}_{CE}(\psi(Z_s), y)
$$

**Soft Distillation**：簡單說就是把蒸餾式子多加上一個與 Teacher model 的 logits 互做 KL Divergence：$Z_t$ 為 teacher 的 logits、$Z_s$ 為 student 的 logits ($\lambda$ $\tau$ 為超參數)

$$
\mathcal{L}_{Soft} = (1-\lambda)\mathcal{L}_{CE}(\psi(Z_s), y) + \lambda \tau^2KL(\psi(Z_s / \tau), \psi(Z_t / \tau))
$$

**Hard Distillation**：這個方法是這篇論文所提出來的，加上的部份改成為 student model 的結果與 teacher model 的結果做 Cross Entropy，可理解為：同時與真正的 Ground truth 與 把 Teacher model 產生的結果當成 Ground truth 各做一次 Cross Entropy：$y_t$ 定義為 $y_t = \mathrm{argmax}_cZ_t(c)$ 即 teacher model 經 softmax 的最後結果

$$
\mathcal{L}_{Hard} = \frac12\mathcal{L}_{CE}(\psi(Z_s), y) + \frac12\mathcal{L}_{CE}(\psi(Z_s), y_t)
$$

我個人的理解為 Soft 加上了 student logits 與 teacher logits 之間的差，多了一個兩模型結果的比較項。而 Hard 的做法則是與 teacher 經過 softmax 最後處理得到的結果做 CE 。感覺一個是與前資料做比較、一個是與處理後資料做比較的概念。

### Distillation token

與 ViT Transformer 不同的地方是，DeiT 在 token 的地方新加了一個 Distillation token，型成一個 $N + 2$ 的字串，與 Class token 計算方法一樣，會與所有的 token 一起做 attention。唯一的區別在於：

class token 目標與 GT 一樣、而 distillation token 目標與 teacher 結果一樣

![Image](https://i.imgur.com/Qo7IYYS.png)

而這個 Distillation token 最後對應到的就是 Distillation loss (蒸餾損失)，可以選擇使用 hard distillation loss 或者 soft distillation loss，加上這一項的 Loss funtion 可以讓我們在調整 Loss 時，可以多根據 teacher model 的結果來調整，也就是說最後的 Loss funtion 如下式：

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{teacher(Distillation)}}
$$

以上就是 DeiT 全部的核心概念了，其實不難理解，就只是單純在 Transformer 後多加一個 token ，但目標與 class token 不一致。以下實驗來看看是不是有加新 token 的必要性。

## 3. Experiment

### DeiT 架構參數

DeiT 分三個架構，依照參數量由小排至大，其中 DeiT-B 是參數最大，且與 ViT-B 參數量相同的架構

![Image](https://i.imgur.com/kdyA0Dt.png)

### 實驗一、哪種 Teacher model 更好？

作者選用 FB 之前提出近似 NAS 想法的網路 RegNet 來做 Teacher model，結果如下：

![Image](https://i.imgur.com/FvdgxsR.png)

嗯…當然，teacher model 架構越大效果越好。⚗ 代表蒸餾的意思 (也太可愛)

### 實驗二、哪種 Distillation 方法更好？

作者提出三種不同的蒸餾方法，普通、軟蒸餾、硬蒸餾

![Image](https://i.imgur.com/C1bIoGz.png)

可以看到硬蒸餾效果最好

### 實驗三、哪種 token 的組合效果最好？

一共有：只使用 class token、只使用 distillation token、以及兩個都使用

![Image](https://i.imgur.com/SVDMCCG.png)

這個實驗證明加上 distillation token 真的會讓結果好一些些，大概 0.6 %

### 實驗四、與 SOTA 的對比

![Image](https://i.imgur.com/PjBA2DI.png)

### 實驗五、性能對比

當然要來與 ViT 做一下比較啦，DeiT 用上了 distillation 
因此在模型的運算速度上面非常的有優勢

![Image](https://i.imgur.com/A1kvC5m.png)

## 結論

DeiT 使用 distillation 解決了 ViT 一定要用大資料集訓練效果才好的問題，好訓練、執行速度也快是它的一大特色

## Reference

https://zhuanlan.zhihu.com/p/349315675

https://zhuanlan.zhihu.com/p/51431626

https://zhuanlan.zhihu.com/p/102038521
