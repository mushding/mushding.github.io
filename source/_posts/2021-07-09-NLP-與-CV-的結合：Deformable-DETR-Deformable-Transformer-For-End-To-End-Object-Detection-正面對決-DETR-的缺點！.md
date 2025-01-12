---
title: >-
  NLP 與 CV 的結合：Deformable DETR: Deformable Transformer For End-To-End Object
  Detection - 正面對決 DETR 的缺點！
mathjax: true
date: 2021-07-09 02:33:30
tags: Vision Transformer
categories: 電腦視覺整理
---

Deformable DETR 的提出是為了解決 DETR 的兩個缺點：

* 訓練時間超長
  * 因為 CNN 是 Attention Map 的一種特例，也就是說 Attention Map 的組合性多，效果效好，但是複雜度高
* 計算複雜度高
  * 同上 Attention Map 是 $N_q \cdot N_k$ 維的，而 CNN 是 $HW$

論文中使用了 Deformable conv 的觀念來達成減少運算量及加入多重解析度。

[https://arxiv.org/pdf/2010.04159.pdf](https://arxiv.org/pdf/2010.04159.pdf)

keywords: Deformable DETR
<!--more-->

### 計算時間長詳解
假設 Batch 為一，圖片經 CNN 後的維度會變成 $(H \cdot W \cdot C)$ 的特徵向量，後經 reshape 變成 $(HW \cdot C)$ 再加上 Positional Enbedding 後放進 Transformer。其中 $(HW \cdot C)$ 可看成長度為 $HW$ 大小為 $C$ 的 sequence

![image-20210709115659980](https://i.imgur.com/6pAB8h3.png)

以下 $N_q N_k$ 其實就是 $HW$，則輸入向量 $(N \cdot C)$，乘上一個 $W$ 轉換矩陣 $(C \cdot 1)$ 則計算 self attention 的時間複雜度為：

$$
O(N_qC^2 + N_kC^2 + N_qN_kC)
$$

分別對應

$O(N_qC^2)$ 計算 Query 的複雜度

$O(N_kC^2)$ 計算 key 的複雜度

$O(N_qN_kC)$ Attention 的複雜度 $(N_qC \cdot CN_k) = (N_qN_k)$

![image-20210709120129060](https://i.imgur.com/gr4y6sI.png)

透過以上可以發現當圖片的解析度越大，Attention 的計算複雜度為所有像素數量的平方，也就是 $(HW)^2 = N^2$ ，這就導致了圖片越大，模型越不好收斂的原因。

## 網路架構
作者引用了 Deformable conv 這篇論文，最大的觀念就是突破以往 conv 固定 size 的卷積核 (3x3) ，而是改用一個 (3x3) + 偏移量的方式來做，如下圖：(每一個原 conv 的點都會加上一偏移量)

![image-20210709153338566](https://i.imgur.com/7x5XPIH.png)

而這個偏移量是透過一層的 conv 來自己學出來的，如下圖：(注意綠色 conv 的深度為 2N，代表 x 軸與 y 軸的偏移量)

![image-20210709153515728](https://i.imgur.com/VuXpX2L.png)

原 Deformable 作者認為這個變型 conv 的好處有：

* 對物體的形變能力更強 (超畸形都沒在怕)
* 對圖片的視野更廣擴，因為不受矩型 conv 的限制，可以自由奔放的去找特徵點。

本論文 Deformable DFTR 的作者就把這個觀念放到網路中的…任何地方，(基本上想到的地方都加上了)，包含 CNN 層、Encoder

![image-20210709154053491](https://i.imgur.com/WXO4wG5.png)

## Deformable Attention Module
於是作者提出 Deformable Attention Module 來解決 DETR 的問題，與原 Attention 公式對比如下：(上式為原 Attention、下式為 Deformable Attention)

$$
\mathrm{MultiHeadAttn}(z_q,x_k)={\sum^M}_{m=1}W_m[\textcolor{purple}{\sum_{k\in\Omega_k}}A_{mqk}\cdot W'_m\textcolor{red}{x_k}]
$$

$$
\mathrm{DeformAttn}(z_q,p_q,x)={\sum^M}_{m=1}W_m[\textcolor{purple}{\sum^K_{k=1}}A_{mqk}\cdot W'_mx\textcolor{red}{(p_q+\Delta p_{mqk})}]
$$

用非常白話來講兩個最大的不同點就是：

* key 的數量不同：
  * 原本的 self attention 「每個」 query 會與「每個」 key 做計算，如上一節提到的 $(N_qN_k)$
  * 而 Deformable 則是使用一個自定數 $K$ ，來限制 query 只與 $K$ 個 key 做計算，變成 $N_qK$ (作者的 K 取 4，很小喔…)
* key 的意義不同：
  * 原本的 self attention 就是單純計算第 i 個 query 與第 j 個 key 之間的關系
  * 而 Deformable，則是引入了 Deformable 的觀念，把原本點上 $(p_q)$ 做一個位移偏差 $\Delta p_{mqk}$ ，總偏移點的數量為 $K$，如下圖所示：
  * 意義就變為「只與 $p_q$ 點附近的其它點做 query key 的計算了」

![image-20210709155840876](https://i.imgur.com/lhqVF0k.png)

* Attention 做法小不同：
  * 在 Deformable DETR 中的 Attention 塊並不是把 key 與 query 做內積，而是直接做線性轉換，之後再乘上 $K$ 個偏差特徵點就可以了。完整的 Deformable Attention Module 如下圖：

![image-20210709160943065](https://i.imgur.com/IJQwucJ.png)

改用這個架構時間複雜度算出來為：結果就會與圖片大小的 $WH$ 無關啦啦

$$
O(NKC^2)
$$

## Multi-scale Deformable Attention Module
在這一章作者要來解決 DETR 中沒有使用 FPN 使得在小物件偵測效果不好的問題。公式如下：

$$
\mathrm{MSDeformAttn}(z_q,\hat{p_q},\{x^i\}^L_{l=1}) = \sum^M_{m=1}W_m[\sum^L_{l=1}\sum^K_{k=1}A_{mlqk}\cdot W'_mx^l(\phi_l(\hat{p_q})+\Delta p_{mlqk})]
$$

簡單來說就是在每一個 CNN 的特徵向量中，假設有 $L$ 層，每一層各取 $K$ 個點的意思，因此 key 可以表示成 $K\cdot L$，在乘上 query 後，這個意義其實就融合了各層的特徵，所以作者認為不需要再做 FPN。$K\cdot L$ 乘上 query 天生就有相加的效果了。下圖為完整架構：

![image-20210709154053491](https://i.imgur.com/WXO4wG5.png)

下圖則為 CNN 特徵向量到 Encoder 的架構圖：可以發現 Encoder 的 C 皆為 256，因此要對不同解析度的特徵圖做 1x1 conv，以及多做一層卷積層得到放大 6 倍的特徵圖。

所以 Encoder 中為 CNN 第 3, 4, 5, 6 層的特徵向量。

![image-20210709162927599](https://i.imgur.com/TEeAUU8.png)

## Decoder
Decoder 中有兩個 Block：cross-attention、self-attention。兩個 attention 的三個 input 彼此都不太一樣。由於 Deformable attention 只能用在與 CNN 相關的層上，所以 cross-attention 可以做修改，而 self-attention 就維持原樣了。

**self attention**

Query 來自 Object query
Key 來自 Object query
維持原做法，不做任何調整

**cross attention**

Query 來自 Object query
Key 來自 Encoder 的輸出
使用的是 Deformable Attetion Module

另外最後一層的 FFN 預測 BBox 的輸出有一點點的不一樣，變成預測出相對於 $p_q$ 參考點的偏移量 $b_{q\{x,y,w,h\}}$ (x軸 y軸 長 寬)，(嗯…好像有那麼一點點 YOLO 的味道)，公式如下：

$$
\hat{b_q}= \{\sigma(b_{qx}+\sigma^{-1}(\hat{p_{qx}})), \sigma(b_{qy} + \sigma^{-1}(\hat{p_{qy}})), \sigma(b_{qw}), \sigma(b_{qh}))\}
$$

## Experiments
與 DETR 的效果相比：在 epoch 與 Traning GPU hours 上與 DETR 少很多

![image-20210709170526824](https://i.imgur.com/ob3Smaa.png)

與目前的 SOTA 比較：

![image-20210709170657400](https://i.imgur.com/46THxYA.png)

## 結論
Deformable DETR 透過使用 Deformable 的方法來使 Transformer 中的運算數減少許多，不再 depend on 圖片大小，而且因運算減少所以可以加上類似 FPN 的多重解析度。效果比 DETR 好一點點

比較神奇的是不知道為什麼 BBox 的預測又跑回去 YOLO 那一套了，說是比較好收斂啦…

## Reference

https://zhuanlan.zhihu.com/p/342261872

https://blog.csdn.net/irving512/article/details/109713148

https://www.jianshu.com/p/8524abf10018