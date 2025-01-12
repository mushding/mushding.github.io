---
title: >-
  Vision Transformer 演化史: Going deeper with Image Transformers - CaiT 引入
  LayerScale 及 class-attention layers 優化 DeiT
mathjax: true
date: 2021-09-08 15:37:38
tags: Vision Transformer
categories: 電腦視覺整理
---

本篇論文是 Facebook AI 團隊在 2021 3 月所提出，作者 Hugo Touvron 與 DeiT 是同一個人。論文主要的貢獻有二：提出了 LayerScale 優化了 Transformer 的網路，以及 class-attention layers 進一步使得 class token 的使用變得更合理。

CaiT 沿用了 DeiT ViT 的核心精神，並再加入新概念加以改進，在 ImageNet 上取得了 86.3% 的 Acc1 performance，比原本的 DeiT 多了不少。

keywords: CaiT、LayerScale、class-attention layers
<!--more-->

## 1. Introduction

這篇論文的核心在於優化 Transformer 的網路架構，使得網路更好訓練，不因網路越深就越難收斂。作者的核心思想就是：**網路架構 (architecture design) 與 optimization (優化) 是互相呼應的**，ResNet 就是一個非常經典的例子。

$$
x_{l+1} = g_l(x_l) + R_l(x_l)
$$

加上 Residual 後並沒更改太多架構，但是變得非常好訓練，網路效果也因而上升了一個層級。這種明明沒改什麼網路效果確超乎想像的例子，證明了網路優化的重要性。

那 Transformer 呢？每個 Block 內的公式可寫成以下：$\eta$ 為 LayerNorm

$$
\begin{gathered}
x_l' = x + SA(\eta(x_l))\\
x_{l+1} = x'_l + FFN(\eta(x'_l))
\end{gathered}
$$

作者經由實驗給出的答案，有下列兩項的改進：

1. LayerScale 使加深後的 Transformer 更容易收斂，更好訓練
2. class-attention layers 更合邏輯的來處理 class token 的問題

## 2. 網路架構

### LayerScale

作者提到 ViT DeiT 與原 Transformer 的 Encoder 不同，原始 Transformer 的實作方法是把正規化放在後面 (post-norm)，而 ViT DeiT 等實作方法則為把正規化放在前面 (pre-norm)

因此作者設計了四種不同正規化的排列組合，來試試看哪一種對於網路的優化較高，優化高後下一步就可以往把網路加深的方向改進。

![Image](https://i.imgur.com/VOPovED.png)

對應上圖分別為：(a) ViT DeiT 原始作法、(b) ReZero and Fixup、(c) ReZero and Fixup 加上正規化、(d) LayerScale 

**(a) ViT DeiT 原始作法**：經典的 pre-norm 作法，先做一次 LayerNorm 再進行 FFN 或者是 SA 運算。

**(b) ReZero and Fixup**：取消了 LayerNorm，並新增了一個可學習的參數 $\alpha$ 作用在 Residual 上，用來決定網路中 Residual 與運算 Block 各所占的比例。而 ReZero 為 $\alpha$ 初始為 0、Fixup 為 $\alpha$ 初始為 1。作者在後續實驗中證明這個方法不會使網路訓練時收斂

**(c) ReZero and Fixup 加上正規化**：就是 (a) (b) 的結合，實驗證實有效

**(d) LayerScale**：這是本篇論文提出效果最好的方法，也是 CaiT 使用的方法。把 (c) 乘上的 $\alpha$ 改為乘上一個對角矩陣，公式如下：

$$
\begin{gathered}
  x_l' = x_l + \mathrm{diag}(\lambda_{l,1},...,\lambda_{l,d}) \times \mathrm{SA}(\eta(x_l))\\
  x_{l+1} = x_l' + \mathrm{diag}(\lambda_{l,1},...,\lambda_{l,d}) \times \mathrm{FFN}(\eta(x_l'))
\end{gathered}
$$

矩陣中的 $\lambda$ 是可學習參數，一般預設值都會設成很小，而且預設值會隨著網路的加深越來越小。論文提供的初始參數為：0 層時 -> $0.1$、18 層時 -> $10^{-5}$、24 層時 -> $10^{-6}$

作者使用一個對角矩陣是為了可以**各別調整各 Layer 中的重要度**，而非像 $\alpha$ 一樣每個 Layer 一視同仁，一起乘上某個值。比起 $\alpha$，LayerScale 更能增加網路的多樣性，進一步調整及優化 Residual 與 Block 的關系。

而值一開始設定小的原因，是為了在學習時更能專注在自己的 Block 上，讓大部份的資訊向 shortcut 流，使得與 Identity Map 比較接近

### class-attention layers

除了優化 Transformer Block 之外，作者對於 ViT 中使用的 class token 抱持懷疑。作者認為 ViT 在引入 class token 時，是直接放進網路一開始，與 patch token 一同訓練，這使得 class token 要在網路中起到以下兩個作用：

1. 引導 patch token 一同截取出網路特徵 attention map
2. 最後把 patch token 的訓息總合，得到最後分類的結果

class token 要同時達到這兩個目的看似有些自我矛盾，因此作者提出 class-attention layers 把以上兩個目標分成兩個 stage 來實作。

![Image](https://i.imgur.com/ZRJ2cRP.png)

如上圖所示，作者試者把 class token (CLS) 移到最後一個階段才做運算。因而網路分成兩大部份：

1. patch token 之間的 self-attention，沒有 class token 來參與
2. class-attention，加入 class token 

**patch token**：這個部份與 ViT 差不多，只是沒有 class token 進來參數運算

**class-attention**：加入 class-attention 後，**patch token 會被 freeze 起來，不更新權重**，而 class token 會從 patch token 那提取特徵，也不會把訊息反向回傳給 patch token。簡單來說 class token 單向的從 patch token 得到特徵訊息，接著再傳給 FFN 做最後的分類。

個人理解為，class token 有點像 student model。把前面 patch token 辛苦學到的特徵，用簡單的一兩層來吸收在自己身上。全程 class token 不參與運算，最後兩個 token 資訊是單向流動的，且最後 patch token 不參與分類，全由 class token 來負責。

詳細的 class-attention 公式為：

參與運算的有二：$z=[x_\mathrm{class}, x_\mathrm{patches}]$ 與 $x_\mathrm{class}$。首先先分三組，注意的地方是 Q 只有 class token，而 K, V 是 class token + patch token 

$$
\begin{gathered}
  Q=W_qx_{\mathrm{class}}+b_q\\
  K=W_kz+b_k\\
  V=W_vz+b_v
\end{gathered}
$$

Q 乘上 K 的轉置，並 scale-dot

$$
a=\mathrm{Softmax}(Q\cdot K^T/\sqrt{d/h})
$$

最後乘上 V，並接上一個 Residual，把計算後的結果與原 class token 相加

$$
\mathrm{out}_\mathrm{CA} = W_oAV+b_o
$$

經作者實驗以上步驟做兩次就好了，太多效果不好。

## 3. Experiments

### 與 SOTA 相比

與 DeiT 比效果好很多，與最大的 NFNet 比，差一點點

![Image](https://i.imgur.com/cEGfrYl.png)

### 不同大小網路架構

分 XXS XS S M 來代表 attention map 的數量，**不是深度！**，深度階為 24 或 36 層，相較於 ViT 的 16 層的確深了不少

![Image](https://i.imgur.com/QKQzmKd.png)


### 實驗一、不同使訓練更穩定的方法

作者除了試 LayerScale 外，還嘗試其它方法，結論如下圖：

![Image](https://i.imgur.com/RkG7cTR.png)

**調整不同深度的 drop rate**：越深越大，結論：沒用

**正規化**：比較 (b) 與 (c) 發現加上了 LayerNorm 後，網路就可以收斂了。單純使用 Fixup ReZero 沒什麼用

**LayerScale**：橘色為沒加 LayerScale、藍色為有加 LayerScale。數值越大代表 Residual 的作用越大，代表模型離 Identity 越遠。作者發現加上 LayerScale 後每一層變得更 uniform 了，證明更能專注在每一個 Block 中

![Image](https://i.imgur.com/Zj2hSq5.png)

### 實驗二、class-attention layers 的作用

先是證實了加上 class-attention 會比沒加的 DeiT 好上一點點

再來得到 class attention layer 最好的層數是 2 層

![Image](https://i.imgur.com/NaTTaY7.png)

## 結論

CaiT 在模型優化上有兩個貢獻：引入 LayerScale 使得 Residual 在 Transformer 中更能專注在一個 Block 上。引入 class-attention 使得 class token 操作變得合理一些些。

同時也因為網路優化的關系，CaiT 在層數方面，從 ViT 的 16 層到達了 36 層 (最高還有 48 層的…)

本論文成功的證明了網路優化的重要性，Transformer 整體架構的合理性又往前了一小步…

## Reference

https://zhuanlan.zhihu.com/p/363370678