---
title: Self-supervised Learning 與 Contrastive Learning 速讀
mathjax: true
date: 2022-02-17 12:05:50
tags: 
    - Self-supervised Learning
    - Contrastive Learning
categories: 電腦視覺整理
---

2018 Google 提出 BERT，給 NLP 下了一個定心丸，同時也證明了無監督學習以及預訓練的潛力

但是在坐穩監督式學習的 CV 中，似乎不論如何無監督學習始終超越不了有監督式學習，但是收集資料以及標記資料所花的成本也偷偷在告訴我們無監督的強項

其實早在 2006 年，AI 大佬 LeCun 就曾提過類似的想法了，並且在日後還說出：`self-supervised learning is the future of ai` 這番很有野心的話，可見大家對於它的期望還是還高的

隨著時間的進展，無監督式學習被 NLP 玩得走火入魔，也慢慢誕生出了新名詞：Self-supervised Learning 自監督學習，這個詞是 LeCun 自己這麼叫的，目的是為了和無監督式學有個區分，但本質上又有哪一點點類似

後來有了 Contrastive Learning、以及 MoCo SimCLR 的提出，應用在 CV 上的自監督學習似乎也在慢慢成長起來…

keywords: Self-supervised Learning、Contrastive Learning
<!--more-->

## 什麼是自監督學習 (Self-Supervised Learning)

自監督學習是無監督學習的一種分支，主要是利用輔助任務 (pretext)，先使用一大堆無標記的資料中挖掘自身的資訊，再來把得到的資訊放到下游任務中做進一步的分析 (Pretrain -> Finetune)

與無監督學習最大的差別在於「挖掘自己的資訊」，最後的特徵結果是從自己與自己相互比較得出來的

![v2-8d077a997287e6fc7f9b5576b3e16f00_720w](https://i.imgur.com/9VsiOqR.jpg)

大致上來說自監督學習可為兩類：生成式以及判別式

**生成式**的代表任務有：GAN、VAE、ELMo、BERT、GPT…。期望能利用數據重新生成一張新的數據。目前在 NLP 上非常流行，但是在影像的本質上不像語言，「理解」後就可以「實行」出來，語言理解了後我們都會說，但理解了一張圖片我們不一定能「畫」出來。如下圖：能知道什麼是鈔票但是畫不出來

![image-20220218173741384](https://i.imgur.com/zLU4fD7.png)

**判別式**的代表任務有：MoCo、SimCLR…。利用無監督數據，自行建立學習任務以及樣本，最後得到數據的向量表示。而判別式 SSL 應用在 CV 又可分為三種方法：基於背景的輔助任務、基於時序的輔助任務、基於對比學習

聽不懂上面在講什麼嗎 XD，用兩句話來解釋的話就是。**生成式**，輸入一張圖片，通過 Encoder Decoder 還原輸入圖片資訊。**判別式**，輸入兩張圖片，通過 Encoder，判斷兩張圖是否相似 0 or 1

![image-20220221205043036](https://i.imgur.com/3wJlaGs.png)

## 基於背景的輔助任務 (pretext)

也可說是基於上下文 (context based) 的方法，在 NLP 中已經玩得非常成熟了，像 Word2Vec 就是基於前後文的順序來預測。而 CV 中也有非常多的論文也提出了相關的做法，下面就來簡單的掃過一遍：

### 拼圖任務

目的是預測兩 Patch 之間的順序關系，流程如下圖：給定一藍色 Anchor 周圍的 9 個紅色 Patch，則藍色與紅色的相對位置關系是什麼？[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)

![image-20220218210930235](https://i.imgur.com/lThZLyw.png)

也可以如下下圖：隨機給兩個綠色 Patch 看看彼此的相對位置如何？[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf)

![image-20220218211619868](https://i.imgur.com/l6CMosM.png)

第一個方法一共有 8 種可能，第二個方法一共有 64 種可能，而且方法二效果好於方法一，於是得到了一個啟發：**使用更強的監督訊息，或說更難的輔助任務，最後網路學到的東西更多，效果更好**

### 挖空任務

目的是要預測被挖去的內容是什麼？如下圖。這件事也啟發**自監督學習不僅可以學習到特徵，還同時也得到一些神奇的效果**。[Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379.pdf)

![image-20220218212025594](https://i.imgur.com/Yqxpyky.png)

### 顏色預測

也可以輸入灰階圖，要預測圖片的顏色。[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)

![image-20220218214019298](https://i.imgur.com/reOSCRu.png)

### 圖片旋轉預測

也可以把圖片轉成各種角度，並預測出對應的角度。[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/pdf/1803.07728.pdf)

![image-20220218214640977](https://i.imgur.com/yi0dvNc.png)

### 解耦特徵互相學習

把原始的數據分成兩個部份，各做一個圖片的修改，並使它們互相學習，就可以達到自監督學式的目標。[Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction](https://arxiv.org/pdf/1611.09842.pdf)

![image-20220218214941924](https://i.imgur.com/0yD1bpB.png)

### 與任務相關的自監督學習 (Task Related Self-Supervised Learning)

在以上種種選擇自監督學習的輔助任務，會希望越接近下游任務的目標越好，如果差太多的話，效果可能會不盡其想。所以開始有了「與下游任務結合」的自監督學習想法。像下圖的做法，也很直覺，把圖片的旋轉也看成是下游分類任務的其中一類，要同時預測物體以及旋轉角度

[Self-supervised Label Augmentation via Input Transformations](https://arxiv.org/pdf/1910.05872.pdf)

![image-20220218215439870](https://i.imgur.com/4V7YB2j.png)

## 基於時序的輔助任務

除了一張一張圖片之外，我們也可以對有「時間相關」的資料做輔助任務，例如影片、音樂、聲音…。

### 影片上的時序

我們可以把一個影片中相近的 frame 看成是有相關的樣本、相遠的 frame 是不相關的樣本，或是放多個攝影機，同個角度拍出來的相關樣本，不同角度拍的是不相關樣本

![image-20220218221520253](https://i.imgur.com/oaCxxIg.png)

## 對比學習 Contrastive Learning

通過數據之間的對比來學習特徵，就好像以面這句話一樣：We don't know something is blue until we see red，沒有比較我們就永遠不知道類別差在哪裡。而核心理念很簡單：**相似的影像結果也要相似，不相似的影像結果也要不相似**，用數學公式來表達會是：其中 \+ 是指正樣本，相似的樣本，- 是指負樣本，不相似的樣本

$$
\mathrm{score}(f(x),f(x^+)) >> \mathrm{score}(f(x), f(x^-))
$$
![gif](https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif)

以下介紹而常見的損失函數

### Noise-contrastive Estimation (NCE)

[Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

把正樣本看成一個類別、負樣本看成一個類別，列出來一個正樣本以及負樣本的 cross entropy，跟二元 cross entropy 其實差不多。其中第一項 $v^+$ 代表正樣本越大代表越相近，第二項 $v^-$ 代表負樣本越小代表越不相近，加個負號代表越小越好
$$
log\,\sigma(u^Tv^+/\tau)+log\,\sigma(-u^Tv^-/\tau)
$$

### infoNCE

[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)

像是在 NCE 中加入 Softmax，使得上式子中的大小分佈差距加大
$$
\mathcal{L}_N = -\mathbb{E}_X[\log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}]
$$

$$
I(x_{t+k}, c_t) \geq\log(N)-\mathcal{L}_N
$$
而比較有趣的是，這個 infoNCE 是 MI (Mutual Information) 的 lower bound。MI 其實表示的是「期望值」，是 Pointwise Mutual Information (PMI) 的期望值，PMI 是個像件機率，公式可定義為：設隨機變數 $(X,Y)$ 是空間 $X\times Y$ 中的一對隨機變數。他們的聯合分布是 $p(x,y)$，邊緣分布分別是 $p(x)$ $p(y)$

$$
PMI = \log(\frac{p(x,y)}{p(x)p(y)})
$$

而再把機率乘上自己就可以得到期望值 MI

$$
MI=I(X;Y)=\int_Y\int_Xp(x,y)\log(\frac{p(x,y)}{p(x)p(y)})
$$

而 MI 也可以用 KL 來列式，兩者的想法正好相同 -> 兩機率分佈之間的關系。對上兩機率的乘積的 KL 差距。當兩集合獨立時，因聯集為 0、乘積也為 0，所以 KL 差距也為 0，[相互資訊 wiki](https://zh.wikipedia.org/wiki/%E4%BA%92%E4%BF%A1%E6%81%AF)
$$
I(X;Y)=D_{KL}(p(x,y)\,||\,p(x)\otimes p(y))
$$

![image-20220220161230417](https://i.imgur.com/LU0Scoo.png)

同時這個 MI 也可以與條件機率有一些公式，如同貝氏圖那樣

$$
\begin{aligned}
I(X;Y) &=\\
&= H(X) - H(X|Y)\\
&= H(Y) - H(Y|X)\\
&= H(X) + H(Y) - H(X,Y) ...
\end{aligned}
$$

那為什麼 infoNCE 是 MI 的一個下界呢？因為在 infoNCE 公式中的 $f$，會正比於剛剛提到的 MI

$$
\mathcal{L}_N = -\mathbb{E}_X[\log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}]
$$

而 $f$ 展開會得到

$$
\begin{aligned}
f_k(x_{t+k},c_t) &=\\
&= p(d=i|X,c_t)\\
&= \frac{p(x_i|c_t)\prod_{l\neq i}p(x_l)}{\sum^N_{j=1}p(x_j|c_t)\prod_{i\neq j}p(x_l)}\\
&= \frac{\frac{p(x_i|c_t)}{p(x_i)}}{\sum^N_{j=1}\frac{p(x_j|c_t)}{p(x_j)}}
\end{aligned}
$$

$c$ 代表 context 正確的目標 (原論文是應用在 NLP 上，所以名稱這樣取)

$p(x_i|c_t)$ 代表從正確目標出選出正樣本的機率分佈、$p(x_l)$ 代表從其它與 c 無關的地方「亂」取的負樣本

給定大 $X=\{x_1,...,x_N\}$ 其中包含 $1$ 個從 $p(x_i|c_t)$ 選出來的正樣本，與 $N-1$ 個從 $p(x_l)$ 選出來的負樣本

$p(d=i|X,c_t)$ 的意思是：給定一目標正確 context，與從 $X$ 中選一個 $x$ 分佈，是正樣本 $i$ 的機率為何

所以 $p(x_i|c_t)\prod_{l\neq i}p(x_l)$ 的意思是：從 c 中選了一個正樣本，其餘選了負樣本的意思

分母的部就是全部的正樣本跟全部的負樣本

最後可以發現我最後推導出來的式子，分子的部分與 PIM 相似，是一個正比的關系，也就是說我們只要去優化這個 infoNCE 就可以順便也優化了 MI

下式 $N$ 指的是負樣本的大小，所以下式可理解為，增加負樣本效果越好，而 MI 也是越大越好 (代表兩者越近)，同時要最小化 infoNCE 的 loss 才能使 MI 最大化
$$
I(x, c)\ge log(N)-\mathcal{L}_N
$$

## 應用在 CV 的對比學習

從上面的結論可知：**負樣本的數量越多，效果越好**，我們一共有兩種方法可以增加負樣本：

* 把之前訓練過的負樣本存起來下次再用 -> Memory Bank、MoCo
* 直接增大 Batch Size -> SimCLR

看開始介紹之前，先來看看最基本的訓練過程。我們要把網路分成兩個部份 q k、其中 q 是 anchor 錨點、k 是正負樣本，兩個樣本獨立訓練，最後再用一個 loss 函數統一在一起，query 會不斷的去與 key 相比，看看是不是正樣本或負樣本

![image-20220220164453375](https://i.imgur.com/75CHVCI.png)

### Memory Bank

[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/pdf/1805.01978.pdf)

在一個 Batch 下，負樣本的數量是一定的 (除非啦…你的 Batch 可以設超大)，在錢不夠的條件下，要如何增大負樣本數呢？

一個做法是把上次訓練的負樣本「向量表示法」存起來，下一階段訓練時再隨機從 memory bank 中拿取一定數量的負樣本，而因為 memory bank 是一個記憶體的概念，所以沒有 Backpropagation 去更新參數

![image-20220220165121350](https://i.imgur.com/WyTTGI4.png)

## 結論

以上快速的帶過自監督學習的歷史，一路從自監督學習 -> 應用在 NLP 上 -> 提出對比學習 -> 應用在 CV 上演進，而其效果也不斷的往監督式學習逼近。我認為對比學習還有很大的進步空間，尤其是看到了 NLP 的成功，大家也不免俗的想要在 CV 上複製一份嘛 XD

下一篇繼續來看看 2020 年由 FaceBook、Google 兩大巨頭所提出對比學習的方法，兩篇都把對比學習往前推了一大步

## Reference

[bilibili 講得很好的對比學習影片](https://www.bilibili.com/video/BV1v5411x7rD?share_source=copy_web)

[bilibili 自監督式學習 Loss 公式講解 (前半段)](https://www.bilibili.com/video/BV1Sa4y1x7Am?share_source=copy_web)

[知乎大神自監督學習文章 (本心得大部份都是參考它的，大推)](https://zhuanlan.zhihu.com/p/108906502)

[自監督學習文章 (英文)](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)