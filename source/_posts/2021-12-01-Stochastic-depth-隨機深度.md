---
title: Stochastic depth 隨機深度
mathjax: true
date: 2021-12-01 15:34:56
tags: 網路模組
categories: 電腦視覺整理
---

論文地址：

[https://arxiv.org/pdf/1603.09382v3.pdf](https://arxiv.org/pdf/1603.09382v3.pdf)

Stochastic depth 這篇論文是在 ECCV 2016 所出的方向，這個時候是介於 ResNet 提出後，及 DenseNet 之前

而提出的作者 Gao Huang 也正是 ResNet 同一個作者


keywords: Stochastic depth、ResNet
<!--more-->

## 目的

ResNet 提出 shortcut 的目的就是為了解決當網路過深時，可以有效的學習特徵，把每一個 block 加上 Residual line 使得每個 block 只學到網路上下的「差值」而已

而 Stochastic depth 則是進一步拓展這個想法，除了跳過一個 block 之外，直接跳過網路中的一層

利用一個隨機變數來控制網路中的某一層，是不是要直接省略不訓練，其機率會隨著網路越深而越大

作者發現利用這個方法可以進一步提高 ResNet 的 Generalization 的能力，並使網路更 robust

## 架構

![](https://i.imgur.com/Y95rNJS.png)


實際的公式如下：

$$
H_l = \mathrm{ReLU}(b_lf_l(H_{l-1})+id(H_{l-1}))
$$
$H_l$ $H_{l-1}$ 代表 Residual block 的結果，以及前一層的結果

$b$ 的值只有 0 或 1，是一個隨機變數，代表這一個 block 是不是要 activate

$f$ 代表經過 conv 層、BN、ReLU… 等的運算方向

$id$ 代表 identity line，也就是 shortcut

架構如下圖：最後兩個方向的分流會合併，並且再經過一層 ReLU

當 $b = 0$ 時，公式就會變成：

$$
H_l = \mathrm{ReLU}(id(H_{l-1}))
$$
公式中的 $b$ 有一定的「生存機率」，使得 $b$ 在此機率下為 1，也就是「通過 Block」

生存機率依以下公式生成：

$$
p_l = 1-\frac{l}{L}(1-p_L)
$$
![](https://i.imgur.com/Oja6Aqt.png)


$p_l$ 為第 $l$ 層的機率

$L$ 為 block (或稱層數) 的總數量

$p_L$ 就代表最後一層的機率

注意的是 $p_L$ 為自定變數，以及第一層 $l$ 為 0，代表第一層一定不會省略掉，機率照著一定比例下降到最後一層

這樣子做的目的是因為淺層的網路所抓取到的特徵會一直被深層網路所使用，相比之下重要性大得許多，所以這些淺層不應該有太大的機率跳過

## 結論

經由隨機的把 ResNet 中的一些層省略後，實驗證明，效果竟然變好一點點了

實驗結果如下圖：

![](https://i.imgur.com/3Z4Tx2r.png)


個人結論為，相較於 ResNet 多提供一個 shortcut 可以走的解法，Stochastic depth 更像是強制網路選擇 shortcut 的方法

同時經過實驗也證實，在 ResNet 中眾多層的網路中，有一些層數是沒學到任何東西、是多餘的

## Transformer

主要會寫這一篇的原因是，在 timm 開源的程式中，ViT 及 Swin Transformer 都使用到了這個方法

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
```

且在下面這一篇 2019 年的論文中 [REDUCING TRANSFORMER DEPTH ON DEMAND WITH STRUCTURED DROPOUT](https://arxiv.org/pdf/1909.11556.pdf) 同時也提出類似的 LayerDrop 架構，透過實驗來證明 stochastic depth 的方法同樣可以應用在深層的 Transformer 上面

## Reference

### arxiv

[Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382v3.pdf)

[REDUCING TRANSFORMER DEPTH ON DEMAND WITH STRUCTURED DROPOUT](https://arxiv.org/pdf/1909.11556.pdf)

### 其餘心得文章

[论文阅读：Reducing Transformer Depth On Demand With Structured Dropout](https://www.cnblogs.com/zyxxmu/p/12788051.html)

[深度学习模型之——Stochastic depth（随机深度）](https://blog.csdn.net/comway_Li/article/details/82228348)

