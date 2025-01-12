---
title: >-
  BERT 速讀 - Pre-training of Deep Bidirectional Transformers for Language
  Understanding - 為自監督式學習打下標竿
mathjax: true
date: 2022-05-09 01:09:05
tags: 
  - Self-supervised Learning
  - Vision Transformer
categories: 電腦視覺整理
---

最近 CV 流行自監式學習，目前一共分成兩個支派，一是之前介紹的 Contrastive Learning，一是以 BERT 為首魔改的系列，接下來幾個文章會來談談最近兩篇比較熱門的「類 BERT」論文

既然這麼說了，當然就要先從什麼是 BERT 開始說起啦！(怎麼很像以前看 Transformer 前介紹 Self-Attention 呢…) (CV 界一直在往 NLP 界靠，而我要補的論文也越來越多了…)

[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)

keywords: BERT
<!--more-->

## Introduction

先來介紹一下 BERT 囉，這邊先直接附上李宏毅的連結：

[https://youtu.be/gh0hewYkjgo](https://youtu.be/gh0hewYkjgo)

BERT 是一個自監督式學習的架構，其利用兩種不同的任務來 pre-train 模型，最後再利用 transfer learning 把下遊任務做 fine-tune 微調結果。其 backbone 為一大堆 Transformer 的 Encoder 堆疊而成的。

![image-20220509005826295](https://i.imgur.com/6OymkLb.png)

BERT 模型的 pre-train 過程有點像在模仿我們學語言的過程，一是利用「克漏字」挖空任務，二是利用「段落重組」來達成訓練。

第一個「克漏字」任務，英文稱 Masked LM。給定一個句字，接著把它其中 15% 的「方格字」給去掉替換特別的符號，其中 80% 會替換成 mask 代表什麼都沒有，10% 換成隨機完全跟上下文無關的字，10% 不進行改動 (照原本的字當輸入)。

![Image](https://i.imgur.com/KdbjTcz.png)

```
ex:
我今天心情不錯

我「今」天「心」情不「錯」

我 mask 天小情不錯
```

替換後的完整句字會全部送進 Transformer 的 Encoder 中編號學習，最後只有「被修改過」的字把它的向量拿出來，做一層簡單的全連接層後做一個分類任務，對應預測分類的字是不是等於標籤中的字。

這樣子做的原因是，利用在句子中挖 mask，網路會更依賴上下文的關系去推論出字詞。且除了放 mask 之外額外增加的兩種改動 (錯字、正確字)，使得網路除了有從未知詞推論出詞的能力外 (mask -> 詞)，還多了糾錯能力 (錯 -> 詞)，使得 BERT 更 Robust，放在各種下游任務中也更靈活。

第二個「段落重組」任務，英文稱 Next Sentence Prediction：給定一篇文章中的兩句話，判斷第二句話在文章中是否緊跟在第一句話之後

![Image](https://i.imgur.com/UdGRQfw.png)

在 BERT 最前面加入 CLS token，這個 CLS token 初始值為亂數，會隨著 BERT 中 Encoder 的進行而學習到**其它 token 的綜合特徵表示**，也可理解成其它 token 的精華重點整理，最後進入一個全連接層做一個 Yes/No 的二分類任務 -> 是不是第二句在第一句的後面

這種通過把原文打亂順序再還原出正確順序的做法，需要對文章的大意有充分、準確的理解才做得出來，因此加入了「段落重組」任務後 BERT 加強了在文意理解上的能力

BERT 模型通過對 「克漏字」 Masked LM 和 「段落重組」 Next Sentence Prediction 兩項任務，使模型對每個字詞的向量特徵表示都能盡可能的全面、準確的描述輸入文章的整體訊息，為後面下游任務的 fine-tune 微調給了很好的參數初始值，打下很好的基礎

### Reference

[BERT Neural Network - EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo&t=502s&ab_channel=CodeEmporium)

[知乎 关于BERT中的那些为什么 (十萬個為什麼 XD)](https://zhuanlan.zhihu.com/p/360343071)