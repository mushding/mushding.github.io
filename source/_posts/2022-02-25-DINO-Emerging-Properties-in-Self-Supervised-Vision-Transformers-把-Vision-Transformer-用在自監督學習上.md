---
title: >-
  DINO: Emerging Properties in Self-Supervised Vision Transformers - 把 Vision
  Transformer 用在自監督學習上
mathjax: true
date: 2022-02-25 18:27:32
tags: 
  - Contrastive Learning
  - Vision Transformer
categories: 電腦視覺整理
---

2021 年 4 月，正是 Transformer 熱潮發揚光大的時候，而 Facebook 這時也趁熱出了一篇把 Transformer 應用在自監督式學習上面，並藉著 distillation 的概念，把網路架構稱作 DINO。得益於 Transformer 的強大，基於 ViT based 的架構成功刷到了當前的 SOTA。

[https://arxiv.org/pdf/2104.14294.pdf](https://arxiv.org/pdf/2104.14294.pdf)

keywords: DINO
<!--more-->

## Introduction

DINO 全名為 self-**di**stillation with **no** labels (嗯…就是這麼硬湊 XD)。翻成中文是：沒有標記的「自知識蒸餾學習」，這篇論文把自監督學習架構看成是一種 student 與 teacher 的 knowledge distillation 方法，所以才會這麼叫它 (就像 SimSiam 中把網路架構看成是一個 Siamese network 一樣)。

DINO 與其它自監督學習的架構相同的是：沒有使用任何負樣本，保留了來自 MoCo 的 momentum

DINO 與其它自監督學習的架構不同的是：它沒有用任何 predictor (用來預測的 MLP 層)、normalization (online-target 網路結果經過一個 L2 Loss)、contrastive loss (像是 infoNCE)，把 Loss function 改為 cross entropy，除之還加入了新的 centering 與 sharpening 架構來避免 collapse

本篇論文發現如果把 Transformer 應用在 DINO 上，最後的輸出特徵空間有著非常強的「邊界」資訊，相較於傳統卷積網路的效果好上非常多，對於應用在分割任務上有很大的前途。

## 網路架構

在詳細介紹前，特別注意這篇文同樣是一篇沒有負樣本訓練的網路架構。以下是網路架構圖：

![image-20220228005007483](https://i.imgur.com/C58wnFx.png)

流程為：

* 輸入影像 x ，會做兩個不同的資料擴增 (使用從 SwAV 來的 multi-crop stategy，後面會細說)
* 分別輸入到 student $g_{\theta_s}$ 與 teacher $g_{\theta_t}$ 網路中，兩者可為卷積層或是 Transformer 層
* student 經特徵提取後，經一個 sharpening (銳利化) 的 softmax 得到結果 p1
* teacher 經特徵提取後，分別經 centering (中心化) 以及 sharpening (銳利化) 的 softmax 得到結果 p2
* p1 與 p2 做 cross entropy 得到網路 loss 
* loss 只會在 student 網路中做 backpropagation，teacher 則有個 sg (stop-gradient) 則不會做 backpropagation
* student 會經由一個 EMA (exponential moving average)，其實就跟 momentum 的概念一模一樣，一點一點的慢慢更新 teacher 網路

網路中所有輸出前都會經過 softmax，而公式如下，特別的是用到了 $\tau$  temperature 這個參數來控制銳利化的大小
$$
P_s(x)^{(i)}=\frac{\exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{\sum^K_{k=1}\exp(g_{\theta_s}(x)^{(k)}/\tau_s)}
$$
而最後的 Loss 要表示的是 teacher 與 student 兩網路學出來的特徵表示在空間中的距離**越近越好**，且使用的是二元 cross entropy loss
$$
\begin{gather}
\min_{\theta_s}H(P_t(x), P_s(x))\\
H(a,b)=-a\log b
\end{gather}
$$
本篇論文有給 pesudo code，一目了然 XD

```python
# gs, gt: student 和 teacher 網路
# C: centering 中心化
# tps, tpt: student 和 teacher 的溫度參數
# l, m: centering 中心化的比率、momentum 的比率
gt.params = gs.params
for x in loader: # 一次讀取一 minibatch 影像
    x1, x2 = augment(x), augment(x) # 把影像擴增成兩個不同的 view
    s1, s2 = gs(x1), gs(x2) # student 的輸出
    t1, t2 = gt(x1), gt(x2) # teacher 的輸出
    loss = H(t1, s2)/2 + H(t2, s1)/2  # 理論上 t1 s2、t2 s1 越近越好
    loss.backward() # back-propagate
    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)
def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # centering 中心化 + sharpening (softmax) 銳利化
    return - (t * log(s)).sum(dim=1).mean()
```

### 自監督學習與知識蒸餾 (Knowledge Distillation)

為何稱知識蒸餾 (Knowledge Distillation)？與 SimSiam 最大不同的點是，SimSiam 是兩個「相同」的網路，所以那篇作者才會歸納為一種「孿生網路 Siamese network」。

而 DINO 是兩個「不同」的網路，加上中間還有一條 EMA 參數傳導鍊，但其實就是 MoCo 中的 momentum，公式如下也一樣就是了
$$
\theta_t \leftarrow\lambda\theta_t+(1-\lambda)\theta_s
$$
所以本篇作者認為是一個 teacher 教導 student 的知識蒸餾網路。(只是不知為何本篇是 student 教 teacher 就是了…)

### multi-crop strategy

接下來說說論文中提到的 multi-crop strategy，簡單來說就是 crop 的定義升級版。假設影像大小為 224x224，則定義：

* 當 crop 的長寬**大**於影像大小的 50%，稱為 Global view
* 當 crop 的長寬**小**於影像大小的 50%，稱為 Local view

![image-20220228023326112](https://i.imgur.com/DqvnJ5s.png)

作者把 student 放 Global view + Local view 的擴增，而 teacher 只放 Local view 的擴增，作者認為可以達到 local-to-global 的效果。也就是 teacher 學習到的內容遠比 student 來得複雜，或說 teacher 學習的只是 student 的一個子集合而已，藉由知識蒸餾的觀點來解釋：student 會學習複雜的參數，並把整理後的結果放到較簡單的 teacher 中做整理與歸納。等等…這個觀念是不是相反了阿… (正常來說不是要 teacher 教 student 嗎？怎麼反過來了呢？我也不知道反正論文中是這麼起名的就是了…)

### centering 中心化與 sharpening 銳利化

前面也有提到這篇論文沒有使用到負樣本訓練，那要怎麼避免 collapse 的發生呢？本篇作者提出 centering 與 sharpening 的概念。

centering 中心化的目標是：避免特徵維度由單一維獨大控制。做法為在 teacher 的特徵提取層 $g_t(x)$ 後加上一個 bias $c$
$$
g_t(x)\leftarrow g_t(x)+c
$$
而這個 $c$ 的更新與 EMA (momentum) 類似，由一個參數 $m$ 來控制，大部份為上一刻算出來的 $c$，小部份為 下一個 Batch 內的結果
$$
c\leftarrow mc+(1-m)\frac{1}{B}\sum^{B}_{i=1}g_{\theta_t}(x_i)
$$
sharpening 銳利化的目標是：加強相近的特徵，減弱較遠的特徵，簡單說就是 softmax 在做的事

![image-20220228025442182](https://i.imgur.com/TaACCBY.png)

作者經實驗發現加入這個兩東西可以一定的避免 collapse 的發生，且發現 centering 容易 collapse 而 sharpengin 則相反，兩者正好互相抵消

### 卷積與 Transformer

DINO 的 backbone 是可以替換的，作者發現使用 Transformer 的效果非常好，對於找出物體的邊界有著顯著的效果。特別提一下，原卷積 based 的網路 MLP 層中有 BN 層，而改為 Transformer 因架構關系不能用 BN，所以作者特別提了一下 Transformer 版本的架構是 **BN-free** 架構。

## 實驗

### SOTA 表

使用 ViT 作為 backbone 的效果明顯於使用 ResNet-50 的效果

![image-20220228025928799](https://i.imgur.com/5i9dz5c.png)

### 應用在分割上

作者發現相較於監督式學習，自監督學習更能找到目標**真正想要關注的位置**，更集中更接近人類對於物體的定義

![image-20220228030120395](https://i.imgur.com/HBJFLEJ.png)

![image-20220228030200153](https://i.imgur.com/Epprwlk.png)

### 一些架構的 ablation 實驗

發現 momentum 的重要性，以及經實驗發現 DINO 架構下 CE 比 MSE 好

![image-20220228030504626](https://i.imgur.com/jmDdBXU.png)

### centering 與 sharpening 實驗

作者發現有兩種 collapse 的發生，一是網路不管 input 只往一大參數做為輸出，而 centering 就是避免這個情況發生的解法，但是 centering 同時也會讓特徵向量過於平均，明顯的特徵被平滑化了，而 sharpening 就是在避免這個情況的解法。兩者互補，缺一不可

![image-20220228030700498](https://i.imgur.com/OG431IX.png)

## 結論

DINO 是一篇把 Transformer 應用到自監督學習的論文，並且也不是單單搬過來而已，同時也修改了一些地方，像是加入了 centering 與 sharpening 來避免 collapse。

雖然說實驗結果的資料告訴我們：自監督的強項是找到「符合人類認為的物體邊界」，但這也同時告訴我們：**選擇資料的重要性**，網路會慢慢的頃向我們所認為的資料收斂。如果資料今天不是想 ImageNet 分佈的那麼集中呢？它會關注到哪些部份呢？我想這也是一個有趣的議題來討論

## Reference

[(Youtube) Yannic Kilcher 大神講解影片](https://www.youtube.com/watch?v=h3ij3F3cPIk&t=1010s&ab_channel=YannicKilcher)

