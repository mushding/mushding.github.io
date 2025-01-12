---
title: 'Contrastive Learning 對比學習: BYOL 與 SimSiam'
mathjax: true
date: 2022-02-22 20:18:23
tags: Contrastive Learning
categories: 電腦視覺整理
---

本篇接續上篇文章，依照時間順序介紹有關對比學習的論文：BYOL -> SimSiam

keywords: BYOL、SimSiam
<!--more-->

前篇文章介紹了 MoCo、SimCLR 兩篇優秀的自監督式學習，它們有個共通點：**都在負樣本的尋找上動手腳**。MoCo 用一個 Queue 來儲存之前的負樣本、SimCLR 直接把 Batch 設超大來解決

## 我們一定要負樣本嗎？

在回答這個問題前，先來了解為什麼 MoCo SimCLR 視負樣本為重。以 SimCLR 的想法為例：一張圖片做兩種不同的資料擴增，經過網路找出特徵後，在最後結果的向量空間內，兩向量距離應該會非常接近。但如果我們只使用正樣本來這樣訓練的話，那網路是不是每次只要輸出**一個等於自己的常數**就會永遠得到最大的相似度？，大家稱這種現象叫 collapsing output。

![image-20220224165354961](https://i.imgur.com/fQ3PXrc.png)

其中一種解決 collapsing output 的方法就是引入負樣本，使得樣本間存在一定的負雜度，不會讓網路往奇怪的地方收斂

但 MoCo、SimCLR 也同時證明了，不管使用哪種增加負樣本的方法，訓練起來都非常的麻煩，或是對硬體要求非常高。而後來的 BYOL、SimSiam 就在這個出發點上以提出一個「簡單、直覺」的自監督學習方法，來嘗試去掉負樣本。

## BYOL

[Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf)

以下是 BYOL 的架構圖，網路的流程為：

一張影像 x 經過兩個不同的資料擴增得到 $t$ $t'$ ，其中上面的分支稱為 **online**，下面的分支稱為 **target**。online 會依序經過三次線性轉換 (view -> representation -> projection -> prediction)，而 target 只會做兩次線性轉換 (view -> representation -> projection)。最後 online 的 preduction $q_\theta(z_\theta)$ 與 target 的 projection $sg(z_\xi')$ 會做相似度的 loss。

![image-20220225143219347](https://i.imgur.com/ep8f3g8.png)

而相似度的公式的流程：把最後兩個結果 $q_\theta(z_\theta)$ $sg(z_\xi')$ 做 L2 Loss
$$
\mathcal{L}_{\theta,\xi}\triangleq\,\mid\mid\bar{q_\theta}(z_\theta)-\bar{z'_\xi}\mid\mid^2_2\quad=2-2*\frac{\langle q_\theta(z_\theta),z'_\xi\rangle}{\mid\mid q_\theta(z_\theta)\mid\mid_2\cdot\mid\mid z'_\xi\mid\mid_2}
$$
計算完 Loss 後 online 會照 Loss 做 Backpropagation，而 target 則是透過 momentum 來更新。(上圖中的 sg 代表為 stop-gradient 的意思)
$$
\begin{gather}
\theta \leftarrow \mathrm{optimizer}(\theta, \triangledown\theta\mathcal{L}^{BOYL}_{\theta, \xi}, \eta)\\
\xi \leftarrow\tau\xi+(1-\tau)\theta
\end{gather}
$$
整體網路架構與 MoCo 不同的點在於去掉了 memory bank 的設計，整個網路只會使用正樣本來訓練。而與 SimCLR 最大的不同在加上了一個新 prediction 層，換句話說又多加了一層線性轉換層

以上就是 BYOL 整體架構，可以看到網路只使用正樣本來訓練，但是網路並沒有提出任何顯著的方法來避免 collapsing output 的發生，而且我們從 Loss function 就可以發現，當存在一個特殊解：online 與 target 皆輸出一恆定常數時，Loss 為零。可以說在 Loss function 中可以發現 collapsing output 的存在，但是這篇論文是解釋是說，經實驗證明，加入 prediction 層可以把發生的機率降到最低，從而使網路穩定。

## SimSiam

[Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566v1.pdf)

SimSiam 為 Simple Siamese 的縮寫，先開始介紹什麼是 Siamese 網路。Siamese 的原意是孿生的意思，應用在神經網路的的意思為：**有兩個網路，它們有各自的輸入，但是擁有相同的參數權重**。

![image-20220225160653482](https://i.imgur.com/WYaWLL9.png)

而這一篇 SimSiam 論文中，作者把自監督學習的這種架構看作 Siamese 網路，把輸入圖片做兩種不同的擴增後放進「參數共享」的網路中，最後再比較兩網路輸出的相似度。網路架構如下圖：

![image-20220225155643647](https://i.imgur.com/q31nKpw.png)

等等…是不是有一個地方怪怪的…，「參數共享」的網路？不就是同一個網路嗎？沒錯在原論文中作者說 ` In a nutshell, our method can be thought of as "BYOL without the momentum encoder"` 也就是在說： SimSiam 與 BYOL 的最大差別在有沒有做 momentum 更新。

作者提出的 SimSiam 主要的核心概念是：提出一個超極直白的自監督式學習的架構，沒有負樣本、沒有超大 Batch Size、沒有 momentum。除了效果很簡單外，也很神奇的避免了 collapsing output 的發生。

作者經實驗發現 BYOL 的三項改進 momentum encoder、predictor 和 stop gradient 中，真正能避免 collapsing output 發生的是 stop gradient

但是在論文原文 4.7 Summary 章節中作者自己也提到了：`but we have seen no evidence that they are related to collapse prevention` 。簡單來說現在大家還不知道為什麼 prediction 層效果這麼好、為什麼 stop gradient 可以避免 collapsing output 

## 結論

BYOL 以及 SimSiam 都是在把自監督式學習往更簡單更直覺的方向前進，去除掉了之前論文較複雜的部份，只是目前還沒有人搞懂為什麼這種架構效果這麼的好…

我自己也認為是如此，論文中大部份都是先有實驗結果才有理論證明，就…看起來不是很能說服人呢…希望後續有更多論文可以提出新架構來解釋這一切

## Reference

[BYOL csdn](https://blog.csdn.net/dhaiuda/article/details/117897881)

[孿生網路](https://iter01.com/581069.html)

[MoCo SimCLR BYOL 大整理 (英文)](https://generallyintelligent.ai/blog/2020-08-24-understanding-self-supervised-contrastive-learning/)

[MoCo SimCLR BYOL SimSiam 大整理 (極市平台)](https://www.gushiciku.cn/pl/gLs8/zh-tw)

[MoCo SimCLR BYOL SimSiam 大整理 (軟體之心)](https://www.gushiciku.cn/pl/gLs8/zh-tw)