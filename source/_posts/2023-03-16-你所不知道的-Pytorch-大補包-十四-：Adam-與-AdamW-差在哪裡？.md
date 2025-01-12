---
title: 你所不知道的 Pytorch 大補包(十四)：後起之秀 - Adam 與 AdamW 差在哪裡？
mathjax: true
date: 2023-03-16 16:46:45
tags: Pytorch
categories: Pytorch 大補包
---

問：為什麼剛剛前幾個介紹的優化器最近都不怎麼出現過，反而較近期的 BERT、最近流行的 Transformer 架構 ViT，都是使用 Adam 優化器，是…因為新潮所以使用它嗎？還是 Adam 真的有什麼可取之處？

keywords: Adam
<!--more-->

## Adam

Adam 優化器在 2014 年提出，相較於 SGD、RMSProp 來說是相對比較新的優化器。論文連結：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

Adam 名稱來自：Adaptive Moment Estimation，直翻就是「動態動量預估」，其特色是融合了 AdaGrad 與 RMSProp 各自的優點，並且在這之上額外加入了 bias-correction。

以下是論文原文：

> the name Adam is derived from adaptive moment estimation. Our method is designed to combine the advantages of two recently popular methods: AdaGrad (Duchi et al., 2011), which works well with sparse gradients, and RMSProp (Tieleman & Hinton, 2012)

公式有點複雜，先來看核心公式：

$$
w_{t+1} = w_t-\eta\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}
$$

Adam 公式中可分為兩個部份，一個長得像 Momentum 記作 $\hat{m_t}$ 稱做第一動量 (first moment estimate)，一個長得像 RMSProp 記作 $\hat{v_t}$ 稱做第二動量 (second raw moment estimate)，最後分母的 $\epsilon$ 是平滑項避免除以 0。

$\hat{m_t}$ 如同 Momentum 有歷史梯度平均的資訊，優點為更新速度快，$\hat{v_t}$ 如同 RMSProp 有歷史梯度平方的平均，優點為動態調整學習率，但又不會因數值太使更新值接近 0。

而各別的 $\hat{m_t}$ $\hat{v_t}$ 公式列記在下面：

$$
m_{t} = \beta_1\cdot m_{t} + (1-\beta_1)\cdot \nabla g_{t-1}
$$

$$
v_{t} = \beta_2\cdot v_{t} + (1-\beta_2)\cdot (\nabla g_{t-1})^2
$$

Adam 有兩個超參數可調整，$\beta_1$ 控制 $m_t$ 預設 0.9，$\beta_2$ 控制 $v_t$ 預設 0.999，兩個超參數超接近 1 目的是使權重更新傾向參考**歷史梯度**而非目前梯度，使網路在遇到較複雜的曲面時有比較穩定的表現 (不會因為目前梯度變化大而「三心二意」的)。

$\beta_2$ 又比 $\beta_1$ 更靠近 1，因為 $\beta_2$ 負責控制**權重的平方和**，使網路非常非常以歷史權重值為依據更新，如果太傾向考量當前權重值的話 ($(1-\beta_2)$)，容易使 $v_t$ 過大，進而使 $m_t/\sqrt{v_t}$ 接近 0 更新不了參數了 (就這與 AdaGrad 的老毛病一樣)。

眼睛尖的人可能已經發現了，為什麼在核心公式中 $\hat{m_t}$ $\hat{v_t}$ 頭上會有一頂帽子 hat 呢？

這頂帽子代表的是 bias-correction，經由前時刻的梯度計算出來的 $m_t$ $v_t$ 還會再經過一個偏差估算的步驟，校正式子中的計算誤差，使得最後正式參與更新的是 $\hat{m_t}$ $\hat{v_t}$。

$$
\hat{m_t} = \frac{m_t}{1-\beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1-\beta_2^t}
$$

### 在 Pytorch 實作中有 Adam 套件可以直接呼叫使用：

<img src="https://i.imgur.com/Be7rCpb.png" alt="Image" />

### 相關參考

[Why is it important to include a bias correction term for the Adam optimizer for Deep Learning?](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)

[Adagrad、RMSprop、Momentum and Adam – 特殊的學習率調整方式](https://hackmd.io/@allen108108/H1l4zqtp4)

[With Adam optimizer, is it necessary to use a learning scheduler?](https://discuss.pytorch.org/t/with-adam-optimizer-is-it-necessary-to-use-a-learning-scheduler/66477)
