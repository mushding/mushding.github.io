---
title: >-
  Vision Transformer 演化史: CoAtNet: Marrying Convolution and Attention for All
  Data Sizes - 使用 Depthwise Conv 來結合 CNN 與 Transformer
mathjax: true
date: 2021-10-21 13:24:39
tags: Vision Transformer
categories: 電腦視覺整理
---

Google 繼提出 BotNet 後又提出新的 Transformer 網路 CoAtNet，並且在數學的公式上發現，Depthwise Convolution 是一個很好結合 CNN 與 Transformer 的點，將兩者公式結合得到刷新「分類」項目上的 SOTA，值得注意的是這篇論文目前並未開源。

[https://arxiv.org/pdf/2106.04803.pdf](https://arxiv.org/pdf/2106.04803.pdf)

keywords: CoAtNet、Depthwise Convolution
<!--more-->

## 1. Introduction

作者發現在**相同資料量及運算量下**所有 Transformer based 的方法都不及 CNN 的效果，也可以換句話說 Transformer 只有在資料量多的前提下才能發揮它的強處。

作者認為這個問題最關鍵的點在於 Transformer 缺乏 inductive bias 的能力，因此才需要使用大量的資料來補足這個問題。

---

什麼是 inductive bias？下面這個討論區大家的回答都蠻可以參考的，而以下是我個人的想法

[https://www.zhihu.com/question/264264203](https://www.zhihu.com/question/264264203)

Inductive Bias 這個詞可以分成兩個部份來看：Induction (歸納、推理)，指的是在資料中尋找共同性、尋找一個通用的規則。Bias (偏見、誤差) 指的是對規則的偏好

在日常生活中的 Inductive Bias，以颱風路徑預報為例子可為：觀察雲向、氣壓、衛星圖等資料來**歸納**出明颱風最有可能的路徑為何，但每個不同國家的氣象局對於每個因素都會有不同的**偏好**，進而使得每個國家預報出來的路徑都不相同。

對於深度學習 CNN 來說，透過 kernel 可以使 CNN 有**歸納**出 locality、spatial invariance 的特性，也就是局部特徵提取，和空間不變性 (特徵不管在圖的哪個地方都是同個特徵)。而對於找出來的不同特徵，會再給它們**權重**來選擇重要及不重要的特徵

因此 Inductive Bias 可以簡單的說是找特徵的能力。

---

回到論文，作者認為之前論文提出結合的方法 (像是 CvT、Levit、T2T-ViT) 都過於生硬，都有點像直接把 CNN 的某個區塊直接併上 Transformer，因此作者提出 CoAtNet，試著從深度學習的兩個角度來考量：Generalization (歸化能力)、Model Capacity (模組的擬合能力)，看能不能找一個平衡點使合併後的網路最佳化。

## 2. 網路架構

作者把如果最佳化合併兩網路分成兩個問題：

1. 要怎麼最佳化合併
2. 要怎麼最佳化堆疊合併後的網路

### 最佳化合併

作者要合併的 CNN 網路選擇同為 Google 提出的 MobileNet，理由有二：

1. MobileNet 與 Trasnformer 中的 FFN 都是使用了 inverted bottleneck，也就是會先把維度放大再縮回原 size
2. MobileNet 中使用到了 Depthwise Conv，與 Transformer 相同的部份，兩者皆是**一層一層的在定義空間中找出經權重的加總**，只是 Transformer 定義為整張圖，而 Depthwise Conv 定義為一個 kernel size。原文如下：
>> a per-dimension weighted sum of values in a pre-defined receptive field

![Image](https://i.imgur.com/K6d7uoC.png)

MobileNet 公式可表式如下：

$$
y_i = \sum_{j\in\mathcal{L}(i)}w_{i-j}\odot x_j
$$

$x_i$ $y_i$ 代表第 $i$ 個位置的輸入和輸出，$\mathcal{L}(i)$ 表示一個 kernel size 的大小，$\odot$ 表示在定義空間中的內積

Self Attention 的公式如下：

$$
y_i = \sum_{j\in\mathcal{G}} \underbrace{\frac{\mathrm{exp}(x_i^Tx_j)}{\sum_{k\in\mathcal{G}}\mathrm{exp}(x_i^Tx_k)}}_{A_{i,j}}x_j
$$

$\mathcal{G}$ 代表整張圖。把 MobileNet 中的 $w_{i-j}$ 替換成 $A_{i,j}$，意義為找出 $x_i, x_y$ 之間的相關性 (co-relation)

在融合兩公式前，來對比一下各自的優缺點：

1. Input-adaptive Weighting 輸入權重比較
   * MobileNet 中的 Depthwise Conv 權重計算是用 kernel ($w_{i-j}$)，特色是 kernel 中的值不會隨著不同層數的圖片而改變，也可說 kernel 是靜態的 (static)，與輸入圖片無關
   * Self-attention 是根據整張圖的 $QK^T$ 做計算，每一個特徵層中的權重都不一樣，也可說 Self-Attention 是動態的 (dynamically) 尋找特徵。正是因為比 kernel 還要自由的原因，Self-Attention 更適合尋找空間中彼此的關系，同時也需要比較大的資料才能發揮，不然會很容易 overfitting
2. Translation Equivariance 平移不變性
   * 在 Conv 中，只關心 kernel 中的局部特徵，因此 Conv 有 translation equivalence 平移不變性，而這個特性可以幫助 CNN 在小資料集中有更好的 Generalization 泛化能力，也就是尋找特徵的能力
   * 而在 Transformer 中，以 ViT 為例子，ViT 使用了 absolution positional embedding 絕對位置編碼，平移不變性消失了，這也是 Transformer 需要更大資料集的其中一個原因
3. Global Receptive Field 全局感知野
   * CNN 中的感知野只限於 kernel 中，也可以說是局部感知野，或是也有人說這是 CNN 的 locality 特色
   * 而 Transformer 一次看一整張圖片，屬於全局感知野，可以更有彈性的去尋找特徵，但代價就是運算量高，與圖片的大小呈平方關系 

下圖是上述各優點的整理

![Image](https://i.imgur.com/pgCjUSs.png)

因此作者要把上面三點優點相互結合成一個新的網路，提出的方法為把 CNN 以及 Transformer 的局部感知野與全局感知野相加，也就是 kernel 以及 attention matrix 兩個部份相加，又分為在 softmax 前後相加得到下列兩個式子：

$$
y_i^{post} = \sum_{j\in\mathcal{G}}(\frac{\mathrm{exp}(x_i^Tx_j)}{\sum_{k\in\mathcal{G}}\mathrm{exp}(x_i^Tx_k)}+w_{i-j})x_j
$$

$$
y_i^{pre} = \sum_{j\in\mathcal{G}}\frac{\mathrm{exp}(x_i^Tx_j+w_{i-j})}{\sum_{k\in\mathcal{G}}\mathrm{exp}(x_i^Tx_k+w_{i-j})}x_j
$$

式子中的 $\sum_{k\in\mathcal{G}}\mathrm{exp}(x_i^Tx_k)$ 指的就是 softmax，而來自 Conv 的 kernel $w_{i-j}$，分別加在 softmax 後，及 softmax 中

作者最後選擇 $y_{pre}$ 做為網路架構，原因是在 softmax 前加上 $w_{i-j}$ 的意思更能符合，**Self-attneion 除了考慮全局感知野外，還加上了 $w_{i-j}$ 局部感知野**的感覺，也可以看成在 Self-attention 中加入了來自 $w_{i-j}$ 的平移不變特性。

### 最佳化堆疊合併後的網路

設計好網路核心後，接下來要討論如何有效的堆疊 CNN 與 Transformer。由於 Self-Attention 的計算量偏大，要在網路效果及效能間做出取捨，因此作者提出了以下三種解決方案

1. 先用 CNN 做幾次 downsampling，再把比較小的特徵圖丟給 Transformer
2. 只使用 local attention，把 Self-Attention 中的 $\mathcal{G}$ 改成跟 kernel 一樣大小
3. 把原本的 Self-Attention 改成線性 Self-Attention，使時間複雜度變為線性

作者經實驗證實，2 3 的方法會影響到網路效能，因此最終方案採用 1，詳細流程如下：

downsampling 的做法可分為兩種：

1. 像 ViT 一樣直接切成 16x16，記做 $ViT_{REL}$
2. 使用 CNN 的 stride 2 兩倍兩倍往下

整個網路分為 4 個 stage，又 Conv 找特徵的能力比較強一定要在 Transformer 之前，所以一共有以下五種情況：

1. $ViT_{REL}$
2. $C-C-C-C$
3. $C-C-C-T$
4. $C-C-T-T$
5. $C-T-T-T$

比較以上五種網路的指標分別為

1. 歸化能力 (generalization)
   * 在比較訓練損失 (training loss) 與驗證集正確率 (evaluation accuracy) 之間的差距，在兩模型有相同訓練損失的前提下，有比較高的驗證集正確率代表有更好的歸化能力
   * 可理解成網路遇到**未看過資料**尋找重點特徵的能力
2. 模型擬合能力 (model capacity)
   * 給一個超大的訓練集，確保網路絕對不會出現 overfitting 的現象，看哪一個網路**收斂**的速度最快，也就是學習力最好的網路

#### 歸化能力 (generalization) 實驗

![Image](https://i.imgur.com/TkpnkrN.png)

直接用實驗得出以下結果

$$
C\textrm{-}C\textrm{-}C\textrm{-}C \approx C\textrm{-}C\textrm{-}C\textrm{-}T \geq C\textrm{-}C\textrm{-}T\textrm{-}T \gt C\textrm{-}T\textrm{-}T\textrm{-}T \gg ViT_{REL}
$$

#### 模型擬合能力 (model capacity) 實驗

![Image](https://i.imgur.com/Nr0QVt4.png)

直接用實驗得出以下結果

$$
C\textrm{-}C\textrm{-}T\textrm{-}T \approx C\textrm{-}T\textrm{-}T\textrm{-}T \gt ViT_{REL} \gt C\textrm{-}C\textrm{-}C\textrm{-}T \gt C\textrm{-}C\textrm{-}C\textrm{-}T
$$

綜合以上兩個實驗結果發現 $C\textrm{-}C\textrm{-}T\textrm{-}T \approx C\textrm{-}T\textrm{-}T\textrm{-}T$ 兩個結果相當，於是作者最後再把 ImageNet-1K 加上 30 個 epochs 看看誰比較好

![Image](https://i.imgur.com/67DFOUq.png)

最後選擇 $C\textrm{-}C\textrm{-}T\textrm{-}T$ 作為 CoAtNet 的主架構

### 網路架構

![Image](https://i.imgur.com/ucvJj3k.png)

網路架構包括 5 個 stage

* stage S0：兩層簡單的 CNN 做低階特徵選取
* stage S1：使用 MobileNet with SE (Squeeze-Excitation)
* stage S1-S4：照 $C\textrm{-}C\textrm{-}T\textrm{-}T$ 依序堆疊

## 3. Experiments

### 實驗一、CoAtNet 家族

依照每個 stage 重複層數不同來區分

![Image](https://i.imgur.com/PEvvF1s.png)

### 實驗二、與 SOTA 的比較

較大型的 CoAtNet 有超越 NFNet 一點點

![Image](https://i.imgur.com/BRGLUys.png)

### 實驗三、FLOPs 運算量的比較

![Image](https://i.imgur.com/X0RNPtu.png)

### 實驗四、Params 參數量的比較

![Image](https://i.imgur.com/owvIlTA.png)

## 結論

本文提出 Self-Attention 可以自然的與 Depthwise Conv 結合在一起，以更數學的角度來結合兩公式。

其次就是找到適合的堆疊方法，大概上就是 Conv 與 Transformer 各占一半是效果最好的，且 Conv 要先於 Transformer 做運算

注意！這篇論文目前沒開源，網路上找到的這個 github [https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CoAtNet.py](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CoAtNet.py) 是有一些 issue 的，畢竟不是官方的 code…，在使用前可能要多多留意一下 XD

## Reference

https://www.zhihu.com/question/264264203

https://zhuanlan.zhihu.com/p/385106095

https://jishuin.proginn.com/p/763bfbd5eae9

https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CoAtNet.py