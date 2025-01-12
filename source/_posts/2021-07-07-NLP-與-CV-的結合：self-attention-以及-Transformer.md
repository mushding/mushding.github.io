---
title: NLP 與 CV 的結合：self attention 以及 Transformer
mathjax: true
date: 2021-07-07 16:10:44
tags: Vision Transformer
categories: 電腦視覺整理
---

2020 是個 Transformer 在 CV 界大放異彩的一年，在大學時期不知為何的學了一堆 NLP 領域的東西、但是因著興趣研究所選擇念 CV 的我，一聽到這個消息我有點小開心阿，竟然有一天可以把我學到的這兩個東西結合在一起，真是太神奇啦啦。於是打算在未來研究所試試看往這個方向研究…。這篇是 Transformer 系列文的第一篇，會來先了解最基本也是一切的開始：self attention 以及 Transformer，這兩個開山始祖。

keywords: self attention, Transformer
<!--more-->

## Transformer 架構初看
這系列的重點是在 Transformer 上，所以我們先來看看 Transformer 的架構長什麼樣，再來一步一步拆解其中的區塊。(一關一關來過 w)

![image-20210707231927724](https://i.imgur.com/gLA1fus.png)

## self attention
在研究所上課期間，有聽到老師介紹 self attention GAN 這篇論文，因此我對 self attention 的第一印象是來自 CV 的觀念，attention 是用來尋找圖片中重要的像素值，並且加以放大。但是在 NLP 領域中，self attention 的觀念有那麼一點點的不同，但基本的大觀念是相通的，以下是我對在 NLP 中 self attention 意義的理解：

self attention 的誕生是為了解決 RNN 不太能「平行處理」的問題 (parallel problem)，什麼是平行處理呢？通常在 RNN 中對定一個 input 會對應一個 output，接著把 output 當做是下一個值的 input，接著重複以上的動作。可以發現一個問題我們沒有辨法一次性的把所有 input 一口氣放到 RNN 中，一口氣生出一串 output，而這個就是 self attention 所以解決的問題。

![image-20210707163517881](https://i.imgur.com/LwKMHeF.png)

self attention 提供的解法最核心的想法是：用算的！把每一個 input 與 input 之間的關系都算一遍！在 NLP 中 self attention 拆成的三個 vector 都有它對應的名字：query、key、value。

* query 指的是 -> 要去與其它配對的
* key 指的是 -> 被配對的
* value 指的是 -> 放大縮小配對關系

以下是計算步驟
![image-20210707163950107](https://i.imgur.com/wvp9Wv1.png)

* 第一步：
  * 將 query 和 key 計算相似度得到一個共變異數矩陣
  * 可以是內積，cosine 相似度，MLP
* 第二步：
  * 使用 softmax 把權重歸一化
* 第三步：
  * 不像 SE 是直接把值乘回原圖
  * 這邊的做法是使用「加權求和」
  * 把 attention map 中的每一行，與原圖的每一行做線性組合

與 CV 的不太相同，CV 做 self attention 是要加強特徵圖中重要的地方，而 NLP 中做 self attention 是為了可以得到一個類似 RNN 提取特徵的網路架構。

## Multi-head Self attention
在 Transtormer 中使用的 Self attention 是 Multi-head Self attention，它的觀念也很簡單，就是把 query、key、value 再多用一個矩陣分為 $q1, q2, k1, k2, v1, v2$，因此最後的 $b$ 會是兩個結果，如下圖：

![image-20210707165016203](https://i.imgur.com/WJbnu3a.png)

而最後的結果 $b$ 會把 $b1, b2$ 維度相加，再經一個調整維度的 $W$ 使回複成與輸入相同的維度。如下圖：

![image-20210707165317159](https://i.imgur.com/i6mAG9s.png)

使用 Multi-head Self attention 最直覺得差異就是多了一個 $b$ 在這裡稱為一個 head，每多一個 head 等同於多一個訓練不同側重點的 attention，例如 2 個 head 的話，可能一個訓練是全域訊息，一個訓練是局部訊息，越多的 head 線性組合的空間也就越大。

## Positional Encoding
在 self attention 中會發現一個問題，就是當輸入字串是「A 打了 B」與「B 打了 A」機器會把它們當成是同一個輸入，因為 self attention 並沒有考慮 sequence 之間的順序。因此我們在輸入前要加上一個與 $a^i$ 同維度的 $e^i$，而這個 $e^i$ 就代表位置資料，在原論文中 $e^i$ 是人工設計的，不是學習出來的。

那會有一個小問題，為什麼 $a^i$ 與 $e^i$ 之間是相加呢？這邊提出一個想法，假設有一個 one hot encoding $p^i$ ，它會與最原使的輸入 $x^i$ 相加，一同乘以 $W$ 矩陣，根據線性代數的原理 $W$ 可看作 $W^I, W^P$ 的組合，公式如下：

$$
\begin{gathered}
W \cdot x^i_p = [W^I, W^P] \cdot  \begin{bmatrix}x^i\\p^i\end{bmatrix} = \\
W^I \cdot x^i + W^P \cdot p^i = \\
a^i + e^i
\end{gathered}
$$

而乘開後得到 $W^I \cdot x^i + W^P \cdot p^i$ ，但其實 $W^I \cdot x^i$ 就是 $a^i$ ，$W^P \cdot p^i$ 就是 $e^i$，得證是可以直接相加的。

![image-20210707233810364](https://i.imgur.com/UHazpwA.png)

那又一個問題來了，$e^i$ 倒底是怎麼設計的呢？它如果用圖畫出來會長這個樣子…

![image-20210707235902004](https://i.imgur.com/tGK6dQb.png)

嗯…看不懂 ww，不過它是根據一個神奇的公式所生成出來的，叫做 Sinusoidal，以下以 $PE$ (Position Embedding) 代稱：

$$
\begin{gathered}
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model}) \\
PE_{(pos, 2i + 1)} = cos(pos/10000^{2i/d_model})
\end{gathered}
$$

$pos$ 代表輸入值在 sequence 中的位置，舉個例子：當 $pos$ 為 1 時，對應的 Positinal Encoding 可以寫成：

$$
PE(1) = [sin(1/10000^{0/512}),cos(1/10000^{0/512}),sin(1/10000^{2/512}),cos(1/10000^{2/512}),...]
$$

至於為什麼是 10000 嘛…沒人知道 w，總之這個奇怪的式子可以有以下的好處：

* 使每一個位置都有唯一的 Positional Encoding
* 當輸入是長度會變動時，是可以單單修改公式中的 $i$ 來達成目的
* 因為是三角函數的關系，可以讓 model 容易計算出相對的位置 (角和公式)

當然還有更多奇奇怪怪的編碼方式…

![image-20210708001248746](https://i.imgur.com/tsn4mMZ.png)

## Transformer
接下來就到重頭戲啦，我們終於可以來仔細看看裡面倒底藏了什麼東東。一共分成兩半，左半稱為 Encoder，右半稱為 Decoder。每一個「綠色」的 Block 都可以重複 N 遍，Encoder 的資料會送給 Decoder。接下來細解說各個部份：

![image-20210707231927724](https://i.imgur.com/gLA1fus.png)


### Encoder
先來說說左半邊的 Encoder，Encoder 的前半段 Multi-Head Attention 就是上面提到的 Attention，比較不一樣的地方是有拉一條 Residual 直接與結果相加，(這個部份與 self attention GAN 觀念相同)。Add & Norm 則是兩個東西的合稱，Add 就是 Residual，而 Norm 則是做完 Residual 後會經過一個 Normalization，而這裡選用的是 Layer Norm，與 Batch Norm 不同的是，Layer 看重的是 channel 與 channel 之間的標準化

![image-20210708002338795](https://i.imgur.com/ASydltz.png)

接著把結果再放進一個 FFN 中做進一步訓練，並且也加上了 Residual 及 Layer Norm。以上為 Encoder 的整體架構。

### Decoder
我們慢慢由下往上講起，Decoder 中一共有兩個 Attention。

Decoder 的輸入就比較有趣一點了，與 RNN 相同，Decoder 的輸入為每一個時間點產生的結果合，也就是說，在 $t-1$ 的 output 就是在 $t$ 的 input 。

也因為這樣，在 Decoder 的「第一個」 self attention，換了個名字：叫做 Masked Multi-Head Self Attention，其實道理也很簡單：因為 Decoder 的 input 是隨著時間變化了增加的，因此在做 attention 的時候我們不能像時空旅人一樣，直接預知到未來的輸出一起做運算。解決的方法就是在 query 乘上 key 後多乘上一個 Mask ，這個 Mask 負責把後面的值給蓋住，不讓 attention 算到它。(先做 Masked 再做 Softmax)，下圖為 Masked Multi-Head Self Attention 的流程圖：

![image-20210708003720186](https://i.imgur.com/ECf9hPH.png)

Decoder 的輸入第一個字符會是一個 \<Begin\>，而輸出最後一個字節會是一個 \<End\>，下圖解釋 Decoder 的 input 以及 output 以及它是「依序」生出結果來的。

![image-20210708004319820](https://i.imgur.com/RYXM1sz.png)

值得注意的是 Decoder 的訓練與測式的方法不同：
**測試時：**
如果 RNN 一樣，上一個時間點的 output 就為下一個時間點的 input，接著照著 Transformer 的架構走，一個一個的生成出結果。
**訓練時：**
這裡就比較特別了，用了一個叫做 Teacher Forcing 的方法，簡單來說就是直接把 Ground Truth 當成輸入，直接去訓練 Decoder，因為是直接輸入「整串」GT，所以可以平行化加速。(但依然會被 Mask 給蓋掉後面的值 w，不然真的就是時空旅人了)

「第二個」Attention，也有變化，它其實不能稱為完全的 Attention，「第二個」Attention 的 Query 來自 Decoder，Key Value 來自 Encoder (仔細看看圖)

最後經過 Softmax 得到 one Hot encoding，預測出下一個時間點的字詞。

以下就是 Transformer 的完整架構啦啦，比較特別是它的 Seq2Seq 的感覺吧，輸入是 Sequence 輸出也是 Sequence，這種東西要怎麼放到 CV 中去實作呢…？所以接下來要來討論 DETR 這篇論文，他成功的把 Transformer 放進 Object detection 的問題應用中。

## Reference

為什麼要使用 LayerNorm
https://www.zhihu.com/question/395811291

為什麼要提出 scaled dot product attention
https://blog.csdn.net/qq_37430422/article/details/105042303