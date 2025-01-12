---
title: >-
  Vision Transformer 演化史: DeepViT: Towards Deeper Vision Transformer - 試著把 Transformer 變深
mathjax: true
date: 2021-09-07 15:49:56
tags: Vision Transformer
categories: 電腦視覺整理
---

如果 CNN 可以透過增加網路深度來使效果更好，那 Transformer 呢？此篇作者發現，如果 Transformer 想要仿照 CNN 一樣加深度的話，效果不增反減，作者稱為注意力坍塌 (attention collapse)，因而提出了 Re-attention 機制，來取代原本的 Self-Attention。

[https://arxiv.org/pdf/2103.11886.pdf](https://arxiv.org/pdf/2103.11886.pdf)

keywords: attention collapse、Re-attention
<!--more-->

## 1. Introduction

ViT 的模型根據不同的層數一共有三種 ViT-B、ViT-H、ViT-L，分別對應的層數為 12、24、32。此篇作者發現，如果 Transformer 想要仿照 CNN 一樣加深度的話，神奇的事是 32 層的效果並沒有 24 層來的得好。

![Image](https://i.imgur.com/30Rltc0.png)

於是作者深入研究了一下，發現 Transformer 當層數越深時，同一層內的 Attention Map 特徵會越來越相近。也就是說如果我們只單純的把 Transformer 加深，就會因 Attention Map 之間的相似度越來越近，作者稱作這個現象為：注意力坍塌 (attention collapse)。

![Image](https://i.imgur.com/F6uggiB.png)

而作者使用計算的公式如下：如果有興趣的話可以自己去看原文，簡單解釋就是計算 $p$, $q$ 兩層在 $h$ (head) $t$ (token) 下的 cos 相似度。

$$
M_{h,t}^{p,q} = \frac{(A^p_{h,:,t})^TA^q_{h,:,t}}{||A^p_{h,:,t}||||A^q_{h,:,t}||}
$$

## 2. 網路架構

為了解決注意力坍塌 (attention collapse)，作者提出 Re-Attention 架構，把原本的 Self-Attention 的地方取代掉了。如下圖 (左邊為 ViT，右邊為 DeepViT)：

![Image](https://i.imgur.com/BgC7d0v.png)

### Re-Attention

作者進一步發現，雖然 Transformer 越深時層與層之間的 Attention Map 差距很小沒錯，但是同層不同 head 間的差距卻很大。因此作者提出一個想法，如果能將在計算 head 時，把不同 head 的訊息結合起來，再利用它們來產生 Attention Map

所以 Re-attention 是一種使用「可學習」方式，來**整合不同 attention heads 的資訊**，使得生成的 Attention Map 內有更多樣的特徵資訊。

詳細的作法為，在 attention 經過 softmax 後，再經過一個轉置矩陣做一次的 Linear Transformeation $\Theta$，公式如下：

$$
\mathrm{Re-Attention}(Q,K,V)=\mathrm{Norm}(\Theta^T(\mathrm{Softmax(\frac{QK^T}{\sqrt{d}})}))V
$$

## 3. Experiments

### 與 ViT Attention Map 的比較

作者把 Re-Attention 與 Self-Attention，兩者皆做一次上述的 cos 相似公式得到下圖：

![Image](https://i.imgur.com/jkdEgfS.png)

可發現 Re-Attention 可以有效的把 Attention Map 相似的層數往後移了不少，但是神奇的是，仍然會在層數約 30 的地方相似度急速上升 (像魔咒一樣…)

### 與 SOTA 比較

單單的把 Self-Attention 換成 Re-Attention 效果就可以好很多… (神奇)

![Image](https://i.imgur.com/mlKDMP0.png)

###

## 結論

這篇論文提出了一個有趣的想法：如果把 Transformer 加深會發生什麼事呢？並且發現以 ViT 來說，不能單單的加深深度，不然會發生注意力坍塌 (attention collapse)。並用 Re-Attention 來解決它。

我個人認為，這篇文是用實驗的方法來找到這個問題，而非從理論基礎上找到真正的問題，Re-Attention 結合 Attention head 是一個不錯的做法，但這讓我有種治標不治本的感覺，一定還有什麼背後原因使用 Transformer 不能單單加深，不然為什麼這篇論文的實驗結果在層數 30 附近相似度又上升了呢？

## Reference

https://zhuanlan.zhihu.com/p/363370678

https://zhuanlan.zhihu.com/p/359601694

https://zhuanlan.zhihu.com/p/359191305