---
title: Knowledge Distillation 知識蒸餾 & Noisy Student
mathjax: true
date: 2021-07-15 10:52:15
tags: Knowledge Distillation
categories: 電腦視覺整理
---

2020 由於 BERT 在 NLP 的成功，Active Learning 與 Semi-supervised Learning 研究是相當熱門的一年，Google 提出的 Noisy Student 藉由 Teacher Student model 彼此之間的相互訓練，以及在 Student 加中雜訊來得到更好的結果。

https://arxiv.org/pdf/1503.02531
https://arxiv.org/abs/1911.04252

keywords: Knowledge Distillation、Noisy Student
<!--more-->

## Knowledge Distillation
中文翻作知識蒸餾，屬於模型壓縮的一種方法。最大的核心想法就是找出一個模型簡單，但一樣能處理複雜問題的模型。

核心的做法是：使用 Teacher Student model (師徒模型)，先把 Teacher 訓練好後，再從中選精華作為 student 訓練的目標，使得 student 也能達到 teacher 一樣的效果。

詳細：

先預訓練好一個 Teacher model，把 Teacher model 所生成的結果 $q$，當成 student 訓練的目標 $p$，使的 $p$ 與 $q$ 越接近越好

但是直接使用 teacher 的輸出 q 可能會不太合適，因為經過 softmax 的結果，通常對正確答安非常肯定 (機率非常高)，而對非答案的選項非常不肯定 (機率非常低)，這樣會造成 student 在訓練時很快就收斂了，什麼也沒有學到。

因此我們在 Label 上的機率動手腳，讓彼此之間相距近一點。把 softmax 的公式改今 softmax-T 如下：

$$
q_i = \frac{exp(z_i/T)}{\sum_jexp(z_j/T)}
$$

當 $T=0$ 時與 softmax 公式相同，通常會把 $T$ 設成 3 以上，當 $T$ 的值越大，Label 機率分佈就會比較均勻。

## Noisy Student

Google 在 2020 提出了 Noisy Student 終結了 Knowledge Distillation 的系列討論，提出了一個全盤的分析與實驗。Noisy Student 完整運用了 Knowledge Distillation 的特性來訓練網路，最大的重點則是將 student 的輸入加上雜訊的部份 (與 Knowledge Distillation 的概念類似，只是方法不同)，以下為步驟：


### Training
有兩個重點：
慢慢使用架構增大的網路 Iterative Training，讓 student 的潛力至少大於等於 teacher。
在訓練 student 加上 noisy，這裡的 noidy 代表 data augmentation、dropout、stochastic depth

* 第一步：
  * train teacher model 使用 GT 標記來訓練
* 第二步：
  * generate pseudo labels (假標籤)
  * 直接相信 teacher model 所判斷的結果，當成 pseudo labels ，當然也有可能有誤
  * 相信網路的能力，把 confidence 大於一定的百分比樣本留下，做為 student 的訓練構本
* 第三步：
  * train student model
  * 把 pseudo label 來訓練 student model 且同時加入雜訊
* 第四步：
  * 整理過程重複 N 次，直到網路收斂。

### Iterative Training
論文使用 Efficientnet-B7 當 teacher model，Efficientnet-L0 當 student model
再把 Efficientnet-L0 當 teacher，Efficientnet-L1 當 student model，它比 L0 更寬
最後用 Efficientnet-L1 當 teacher，Efficientnet-L2 當 student model，為最後結果

整個訓練起來喔…我覺得真的超複雜，模型大到一個不可思議，難怪後來的人很難訓練成功

自我訓練完後使用 FixRes 的觀念，train 圖片縮小，test 圖片放大

### 結論
Google 的這篇論文簡單的在 Knowledge Distillation 系列上總結了一下：基本上 knowledge distillation 能用，而效果最好的一步是在 student 加上雜訊，使用在訓練時有更好的表現。但是整體訓練方法過於複雜，難以復刻。

## Reference

https://medium.com/%E8%BB%9F%E9%AB%94%E4%B9%8B%E5%BF%83/deep-learning-noisy-student-knowledge-distillation%E5%BC%B7%E5%8C%96semi-supervise-learning-4e0c2d11520a

https://zhuanlan.zhihu.com/p/102038521

https://zhuanlan.zhihu.com/p/81467832

https://zhuanlan.zhihu.com/p/164597142

https://chtseng.wordpress.com/2020/05/12/%E7%9F%A5%E8%AD%98%E8%92%B8%E9%A4%BE-knowledgedistillation/