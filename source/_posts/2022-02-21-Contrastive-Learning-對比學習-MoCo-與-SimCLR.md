---
title: 'Contrastive Learning 對比學習: MoCo 與 SimCLR'
mathjax: true
date: 2022-02-21 20:57:09
tags: Contrastive Learning
categories: 電腦視覺整理
---

本篇接續上篇文章，依照時間順序介紹有關對比學習的論文：MoCo -> SimCLR -> MoCo v2

keywords: MoCo、SimCLR
<!--more-->


## MoCo

由 FaceBook 何凱明大神團隊提出 [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf)

上個 Memory Bank 中有個問題，就是 memory bank 中的資料不同步，q 中會隨著每一次 Batch 而更新，隨後放進 bank 中，而 k 因為是去記憶體中直接拿資料所以不參與更新，如果 q 訓練速度快的話，久而久之 memory bank 資料的表示就會出現訓練差異。一個比較直觀的想法是在 k 中也加入一個 encoder 去學習，但是 memory bank 會隨著時候而增加，要做 BackPropagation 的話計算量會越來越大

**momentum encoder的輸出會被一個queue儲存起來，取代原本的memory bank**

改進方法是使用了兩個不同的 encoder，q 的 encoder 是從自監督學習學來的特徵，而 k 的 encoder 是基於動量來更新的，會一點一點的更新 k，確保放進 memory bank 中的資料之間不會相差太多
$$
\theta_k = m\theta_k + (1-m)\theta_q,\,\mathrm{where}\,m=0.999
$$


![image-20220220172014505](https://i.imgur.com/1pVnv8m.png)

## SimCLR

由 Google Hinton 團隊提出 [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)，提出了 CL 訓練的一個大框架

與 MoCo 最大的不同在於，SimCLR 不使用計算過程複雜的 memory bank，memory bank 在實作上最大的困難在於需要使用兩個不同的 encoder 訓練，實作成本相對複雜，而是改使用增大 Batch size 來達到更多負樣本的效果

並且得到了以下三個結果：

* 資料擴增，以及方法的選擇，在自監督式學習中有著相當重要的角色
* 在網路中新增非線性層，使效果變超好
* Batch Size 的增大，在自監督式中提升的效果比監督式提昇更多

單一流程是：x 是輸入圖片，會做兩個不同的資料擴增 $t$ $t'$ (下面狗狗為例子，選兩種)，經過一個特徵提取層 (ResNet) 得到 $h$ 結果，接著在 $h$ 後面加一個 non-linear 層 (2 層 MLP) 得到 $z$ 。**大個embedding網路執行特徵抽取得到**`h`，接下來使用一個**小的網路投影到某個固定為度的空間得到**`z`。本篇論文發現這個 non-linear 層會有顯著的增加效果

![image-20220220174550589](https://i.imgur.com/QqFzOn3.png)

![image-20220220175013533](https://i.imgur.com/JRjgGk4.png)

實際流程是：取一個 Batch 大小為 N，每一個圖片都做兩個不一樣的資料擴增，總數量變為 2N，而 2N 中其中 2 個目標資料為正樣本，其餘 2(N-1) 為負樣本，我們定義正資料的相似度為 cos 相似度
$$
  \mathrm{sim}(u,v)=\frac{u^Tv}{||u||||v||}
$$
而目標正樣本的 loss 函數定義為，本篇論文稱之 NT-Xent (the normalized temperature-scaled cross entropy loss)，分子為正樣本相似度、分母為正樣本與負樣本相似度之合：
$$
l_{i,j}=-\mathrm{log}\frac{\mathrm{exp}(\mathrm{sim(z_i,z_j)/\tau})}{\sum^{2N}_{k=1}\mathbb{1}_{[k\neq i]}\mathrm{exp}(\mathrm{sim(z_i,z_k)/\tau})}
$$
而以上的式子，其實與 infoNCE 非常相似，後面也很像是一個 softmax 表示，不同的地方在 infoNCE 相似度是用交差熵，NT-Xent 是使用 cos 相似度。希望後式越大越好，前面加個負號符合 loss 的定義。

最後把 2N 中每兩兩一對做上面的計算後，除以總數量得出總平均
$$
\mathcal{L}=\frac{1}{2N}\sum^N_{k=1}[(l(2k-1, 2k), l(2k, 2k-1))]
$$


SimCLR 的結果表，其中 2x 4x 代表最後一層 linear 的倍數，可發現 SimCLR (4x) 已經與監督式學式媲美

![image-20220220175852964](https://i.imgur.com/wgBlabt.png)

SimCLR 中做了非常非常多的實驗，大概簡單的說一下：

#### 資料擴增結論一

比較各資料擴增的好壞，作者兩兩對比尋找效果最好的前兩個資料擴增方法，結論是 crop + color distribution 效果最好，作者還發現如果只做 crop 機器可能只關注顏色的大概分佈就好了，被挖掉的內容不重要，這時如果加入顏色改變可以很有效的解決這個問題

![image-20220220180327444](https://i.imgur.com/horypbk.png)

作者同時也提到了 crop 的妙用，crop 可以同時達到一般 crop 以及鄰近 crop 的應用，如同前面的論文「拼圖任務」中，作者覺得這個 crop 的方法可以同時包含這兩個擴增

![image-20220220180744507](https://i.imgur.com/sQ6GHKH.png)

最後作者選擇使用隨機 crop、隨機改變顏色、隨機高斯模糊化，來把圖片做擴增

前面也有提到要選 $h$ 而不是 $z$ 這樣放在下游任務的效果好。作者給的解釋是：雖然經過 loss 的是 g()，可以最小化 loss，但真正要應用的資料是一些特徵，可能資料經過兩個非線性的 g() 後已經沒有資料擴增的特徵在裡面了。

![image-20220220181123989](https://i.imgur.com/WUtUUAy.png)

## MoCo v2

同樣由 Facebook 提出，有趣的是本篇改版論文緊接著 SimCLR 提出，而改進的部份也大都來自 SimCLR 的 non-linear 概念，論文內容也只有短短的兩頁，火藥味意外的濃厚。[Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/pdf/2003.04297.pdf)

MoCo v2 參考了 SimCLR 的三個優點

* 超大 Batch
* 多了 non-linear 層
* 有效的資料擴增方法

超大 Batch 的部份 MoCo 已經有 Memory Bank 了所以不用，所以作者把 non-linear 層應用在 MoCo 上面，看看這個框架對於自監督式學習的可行性

### 增加 non-linear MLP 層

MoCo v2 模仿 SimCLR 在最後面加上一層 MLP 層，可以發現在任何 $\tau$ (溫度) 下，效果皆好出超多的

![image-20220222195315386](https://i.imgur.com/gP7PlXP.png)

### 模仿 SimCLR 的資料擴增

MoCo v2 還模仿 SimCLR 在顏色變化中加入高斯雜訊，並且也模仿 SimCLR 的 cosine (half-period) learning rate schedule。不管加入哪一項皆有明顯的提升，而且不管是 batch 或是 epochs 上都有比 SimCLR 優秀

![image-20220222200530916](https://i.imgur.com/dmhbTYb.png)

## Reference

[bilibili 講得很好的對比學習影片](https://www.bilibili.com/video/BV1v5411x7rD?share_source=copy_web)

[bilibili 自監督式學習 Loss 公式講解 (前半段)](https://www.bilibili.com/video/BV1Sa4y1x7Am?share_source=copy_web)

[自監督學習文章 (英文)](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)

[科技猛獸大神文章 (知乎)](https://zhuanlan.zhihu.com/p/378953015)