---
title: 'NFNet: High-Performance Large-Scale Image Recognition Without Normalization'
mathjax: true
date: 2022-02-12 13:54:04
tags: CNN
categories: 電腦視覺整理
---

DeepMind 在 2021 年 2 月提出一篇以 CNN based 的 NFNet，旨在把深度學習中已經使用已久的 Batch Normalization 去掉，希望能藉此建構出 Normalize-Free 的網路架構 (正是 NFNet 的名稱由來)

並提出代替 BN 的 AGC (自適應梯度修剪 Adaptive Gradient Clipping)，在調整梯度大小上有著不錯的效果

在手動選用 SE+ResNeXt 網路下，並加上 AGC 的加持，NFNet 成功達到了當前的 SOTA

[https://arxiv.org/pdf/2102.06171.pdf](https://arxiv.org/pdf/2102.06171.pdf)

keywords: NFNet、AGC
<!--more-->

## Batch Normalization 的缺點

首先來看看為什麼這篇論文要把 BN 給去掉，BN 做為算是深度學習中的基石元件，倒底發生了什麼事情呢？

本篇論文給出了以下三點缺點：

1. **BN 需要額外的計算及記憶體資源**。在計算一個 mini batch 之間的 $\mu$ 及 $\sigma$ 會需額外保存它的臨時變量
2. **BN 會使得網路在訓練及測試時會有差異 (discrepency)**。也就是 pytorch 中的 `model.train()` 和 `model.eval()` 的差異，資料在進網路參數時會因為 BN 而有不同的行為模式，會需要用一個隱藏參數來調整
3. **BN 破壞了資料樣本之間的獨立性**。BN 與 Batch Size 有絕對的關系，當 Batch Size 越大越能反應真實資料的分佈，效果越好，反之越差。換句話說，網路訓練的好壞會與資料的選擇有關

另外還有一點 (我自己多認為的)，在 pytorch DDP (分佈式訓練) 中，BN 的存在會使得不同機器上的資料分佈不同，各個機器最後在整合資訊時，會出現一定程度上資料不合的問題

## Batch Normalization 的優點

雖然 BN 有上述種種小問題，但是在先把 BN 去掉之前也要先認識一下 BN 倒底有哪些優點，才會讓它在深度學習獨霸一時

**BN 會降低 Residual Branch 的隱參數權重**。所謂 Residual Branch 就是 ResNet 中的「主要網路塊」，如圖最中間網路塊

![image-20220214132956188](https://i.imgur.com/LPHRZr4.png)

而在 Residual Branch 中加入 BN 可以有效的使 ResNet 中大部份的資料流，流向 Skip Connection，使得資料得以往網路深層前進，加深網路的層數。也可看成加入 BN 後，會使得主支線的輸出非常小，經 $\mathcal{F}(x)+x$ 公式後，網路下一層的輸入的初始值會與上一層網路差不多 $x$ 

**BN 會減少資料分佈**。如果資料分佈非常鬆散，會使得網路非常難收斂，非常難訓練。也可說 BN 可以有效的平滑 landscape。

**BN 有正規化的效果、BN 在訓練大 Batch Size 比較有效果**。與上一點相當，可以使網路的 landscape 平滑化，進而在設定 learning rate 時可以調大一點，再進而可以加速網路訓練 (但不會加強太多效果)

## NF-ResNet

其實早在 2018-2019 年，就有人陸陸續續提出不含 BN 的網路架構了，但基本上都沒辨法達到當前的 SOTA。2021 年同樣也是由 Deepmind 提出 NF-ResNet，而本篇的 NFNet 正是由「自家」的網路加以修改而來。

而 NF-ResNet 最核心的理念如下圖：

![image-20220214144159705](https://i.imgur.com/BpN2e3j.png)

將 ResNet 的公式，新增了兩個超參數 $\alpha\beta$ ，修改如下：
$$
\begin{gather}
x=\mathcal{F}(x)+x \rightarrow\\
h_{i+1} = h_i+\alpha f_i(h_i/\beta_i)
\end{gather}
$$
簡單來說 $\alpha$ 為經網路後的加權值，通常都設得很小 0.2，$\beta$ 為預測輸入的標準差。加入以上兩個超參數的用意是為了模仿 BN 使大部份資料流向 Skip Connection

## Adaptive Gradient Clipping (AGC)

而本篇 NFNet 網路是架構在 NF-ResNet 之下的，並且提出了 Adaptive Gradient Clipping，將 NFNet 可訓練的 Batch Size 進一步增大

首先什麼是 Gradient Clipping？Gradient Clipping 白話來說就是：為了使網路訓練穩定，當梯度下降的量值過大於一定設定值時，而強迫它改為一固定常數。公式如下：
$$
G\rightarrow \left\{
\begin{array}{ll}
\lambda\frac{G}{||G||} &\mathrm{if} ||G||>\lambda,\\
G&\mathrm{otherwise.}
\end{array}
\right.
$$
![image-20220214152848254](https://i.imgur.com/k0UiJNV.png)

但以上公式會有一個問題：$\lambda$ 的值非常敏感，太大太小效果都不好

於是本篇作者提出：可自適應調整 $\lambda$ 的 Gradient Clipping，公式如下：
$$
G^l_i\rightarrow\left\{
\begin{array}{ll}
\lambda\frac{||W^l_i||^*_F}{||G^l_i||_F}G^l_i & \mathrm{if} \frac{||G^l_i||}{||W^l_i||^*_F}>\lambda,\\
G^l_i&\mathrm{otherwise.}
\end{array}
\right.
$$
其函意如下：

$G^l$ 為一層中算出來的梯度，$W^l$ 為一層中目前的權重值

當「算出來的梯度」與「目前權重值」的比值大於 $\lambda$ 就進入 Clipping

Clipping 多少呢？Clipping 「算出來的梯度」乘上剛剛比值的「倒數」

意義為：我們的上限決定值加入當前權重變量，如果前一時刻權重變化大，梯度計算也大，那比值小代表還算合理；如果前一時刻權重變化小，梯度計算大，比值就會超大，進入 Clipping

利用這個方法就可以自動的來調整 $\lambda$，那為什麼加入 AGC 可以改善沒有 BN 的問題呢？前面也有提到了，少了 BN 網路中的 landscape 非常崎嶇 (以下為示意圖，非本例子)，在非常崎嶇下梯度常常算一算就跑掉了，因此才需加入 AGC 穩定訓練

![image-20220214163945964](https://i.imgur.com/pRc9S3z.png)

## 效果

下圖為：使用 BN (藍)、使用 NF-ResNet 沒有 AGC (橘)、使用 NFNet 有 AGC (綠)，在不同 Batch Size 下的 Top1 準確率

![image-20220214154715148](https://i.imgur.com/QvmDKAQ.png)

可以發現未加 AGC 以及 未加 BN 的 NF-ResNet 在 Batch Size 超過 2048 後就爆掉了，而加入 AGC 的 NFNet 可以很好的解決不使用 BN Batch Size 不能設太大的問題

## 網路架構

相較之下本篇的網路架構比較不是重點，作者為了能夠把本架構刷到 SOTA 而使用了 SE-ResNeXt-50。詳細的架構圖如下：

![image-20220214191340591](https://i.imgur.com/YrN8TP5.png)

左手為 Transition Block，右手為 Non-Transition Block

## 實驗結果

首先來看 SOTA 表

![image-20220214191548689](https://i.imgur.com/PyChVBf.png)

雖然說使用了 SE-ResNeXt-50 所以效果才那麼好，但作者還是有做 Ablation 實驗，看看使用 AGC 是不是真的有比較好？

![image-20220214191855754](https://i.imgur.com/komL4Bd.png)

實驗證明在相同架構下，使用 AGC 效果有好大約 1%

## 結論

本篇成功的提出把 BN 去掉的網路架構 NFNet，並且也刷上了當前的 SOTA (力抗 Transformer 架構 XD)

但是我覺得是不是有達到 SOTA 到不是其次，那是因為作者使用了 SE-ResNeXt-50 (沒有為什麼) 才有可能的

而是這篇論文提出了另一個不用 BN 效果也不差的方法 AGC，使得 AGC 網路也有 BN 網路的「Batch Size 大」「訓練快」「landscape 平滑」等優點

至於 BN 真的有沒有必要去掉呢？我認為在普通情況下其實差不多，但在 DDP 分佈式訓練上，或許就是 AGC 的天下了

## Reference

[Yannic Kilcher 論文圖解 (英文) (推)](https://www.youtube.com/watch?v=rNkHjZtH0RQ)

[csdn 文章 (我覺得這篇很詳細)](https://blog.csdn.net/zhouchen1998/article/details/113824617)

[medium 文章](https://medium.com/ching-i/nfnet-normalizer-free-resnets-%E8%AB%96%E6%96%87%E9%96%B1%E8%AE%80-ce7235d1b123)

[知乎大神](https://zhuanlan.zhihu.com/p/358228383)

[NF-ResNets arxiv](https://arxiv.org/pdf/2101.08692.pdf)

[Visualizing the Loss Landscape of Neural Nets (landscape 圖)](https://arxiv.org/pdf/1712.09913.pdf)
