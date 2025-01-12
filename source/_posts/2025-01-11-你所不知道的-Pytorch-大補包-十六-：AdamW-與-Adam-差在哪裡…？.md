---
title: 你所不知道的 Pytorch 大補包(十六)：AdamW 與 Adam 差在哪裡…？
mathjax: true
date: 2025-01-11 15:58:21
tags: Pytorch
categories: Pytorch 大補包
---

AdamW 在 2017 年提出，它與在 2014 年提出的 Adam 差在哪裡，而 AdamW 又是發現了 Adam 有什麼可以改進的地方嗎？

keywords: AdamW、Adam
<!--more-->

## 一句話總結

簡單用一句話總結 AdamW，因為 Adam 加上 Weight decay 實作方法不合理，所以微微修改 Weight decay 加上去的地方，使得 AdamW 有計算量少、數學公式較合理等特色。

## Weight decay 發生什麼事？

在前一章介紹了 Weight decay，它是由 L2 Regularization 延伸出來的概念，當在損失函數中加入權重的平方項，將損失函數值對權重值作偏微分得到 \(2\lambda\eta w\) 這一項，這一大坨就是 Weight decay（更詳細的推導過程可以參考：[你所不知道的 Pytorch 大補包(十五)：我的模型訓練好；可是測試不好怎麼辦…？- overfitting 與 regularization](https://mushding.space/2023/03/16/%E4%BD%A0%E6%89%80%E4%B8%8D%E7%9F%A5%E9%81%93%E7%9A%84-Pytorch-%E5%A4%A7%E8%A3%9C%E5%8C%85-%E5%8D%81%E4%BA%94-%EF%BC%9A%E6%88%91%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4%E5%A5%BD%EF%BC%9B%E5%8F%AF%E6%98%AF%E6%B8%AC%E8%A9%A6%E4%B8%8D%E5%A5%BD%E6%80%8E%E9%BA%BC%E8%BE%A6%E2%80%A6%EF%BC%9F-overfitting-%E8%88%87-regularization/)）

$$
\mathcal{L} = \mathcal{L_{\mathrm{class}}(f(x,w),y)} + \lambda \sum_{i=0}^n w_i^2
$$

$$
w_{t+1} = w_t - \eta \frac{\partial \mathcal{L}_\mathrm{class}}{\partial w_t}-2\eta\lambda w_t
$$

然而在這篇文章中有一個假設，假設我們的優化器是用最原始的 SGD，連動量 Momentum 都沒有，才會推導出 \(2\lambda\eta w\) 這一項。

那如果是 Adam 會變成怎樣呢？首先是 Adam 的公式：

$$
w_{t+1} = w_t-\eta\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}
$$

$$
m_{t} = \beta_1\cdot m_{t} + (1-\beta_1)\cdot \nabla g_{t-1}
$$

$$
v_{t} = \beta_2\cdot v_{t} + (1-\beta_2)\cdot (\nabla g_{t-1})^2
$$

再把 \(\nabla g_t\) 拆開：

$$
\begin{aligned}
m_{t} &= \beta_1\cdot m_{t} + (1-\beta_1)\cdot \nabla g_{t-1}\\
&=\beta_1\cdot m_t + (1-\beta_1) \cdot \nabla g_ {t-1} + \color{red}(1-\beta_1) \cdot 2\lambda w
\end{aligned}
$$

$$
\begin{aligned}
v_{t} &= \beta_1\cdot v_{t} + (1-\beta_1)\cdot \nabla (g_{t-1})^2\\
&=\beta_1\cdot v_t + (1-\beta_1) \cdot \nabla (g_ {t-1})^2 + \color{red}(1-\beta_1) \cdot (4w\nabla g+4\lambda w^2)
\end{aligned}
$$

可以看到在公式後面紅紅的地方就是因 Weight decay 而多產生的常數項。

AdamW 這篇作者認為，在 SGD 時，因為優化器額外項不多不複雜，所以最後的常數項數值都會是 \(2\lambda w\)。但後來的優化器加上動量、加上動態學習率的分母，早早就加在損失函數上的 L2 Regularization，會隨著各種微分，數值不僅會散掉，同時還會增加不少額外的計算量。

因此作者提出 **Adam with decoupled weight decay (AdamW)**，如果要在 Adam 中使用 Weight decay，不會使用 L2 Regularization 加在損失函數上的概念，而是直接加在優化器上，如圖（論文原圖）：  

<img src="https://i.imgur.com/1SoW9fl.png" alt="Image" />

也就是剛剛 Adam 一大坨看不懂的東西會直接變成這樣：

$$
w_{t+1} = w_t-\eta\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}-\color{red}2\lambda w
$$

直接套在優化器後面，就不會因經過很多層微分運算而有計算量大、數值分散等問題，而且從數學式子角度來看，也比較直白好理解。

至於 AdamW 真的會比 Adam 好嗎？論文中當然會是說效果比較好啦，但真正情況就要看各個實驗的資料集。不過可以確定的是 AdamW 的運算量比 Adam 小。

當然最重要的是，如果實驗中沒有使用到 Weight decay 的話，那 Adam 與 AdamW 是一模一樣的！

## Reference

- [AdamW and Super-convergence is now the fastest way to train neural nets (fast.ai)](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html)
- [Adam和AdamW的区别 (一句話總結)](https://blog.csdn.net/weixin_45743001/article/details/120472616)
