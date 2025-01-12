---
title: NLP 與 CV 的結合：End-to-End Object Detection with Transformers DETR
mathjax: true
date: 2021-07-08 00:59:19
tags: Vision Transformer
categories: 電腦視覺整理
---

本篇文章要來看看 Facebook 是怎麼把 Transformer 運用在 Object Detection 上，也因為這篇論文的成功，CV 界吹起了一陣 Transformer 熱…

[https://arxiv.org/pdf/2005.12872.pdf](https://arxiv.org/pdf/2005.12872.pdf)

keywords: DETR
<!--more-->

## Abstract
DETR 為 Detection Tranformers 的簡寫，這篇論文提出了一個 end-to-end 且 based on Tranformer 的方法來解決 Object Detection，而最後的準確率以及運行時間與改良後的 Faster R-CNN 相當。

### 特色
下圖為 DETR 的架構圖： DETR 架構分為兩個主要的部份：CNN 以及 Transformer。
![image-20210708144748759](https://i.imgur.com/ViOBka4.png)

由於因為使用了 Transformer，因此作者把 Object Detection 的問題看成一個 set prediction problem，並且訓練時要求 predict set 與 ground truth set 間的bipartite matching。看不懂嗎 ww 沒關系以下詳細介紹各個名詞意思。

### set prediction problem
有一些 Set，彼此之間做 matching 的問題，通常會包含兩種 Set：predict set 與 ground truth set。
![image-20210708163216274](https://i.imgur.com/B2nWwSK.png)

### bipartite matching
而 bipartite matching 是 set prediction problem 中的一種特例：Set 數為 2，且所有的對應關系皆為「一對一關系」，如下圖所示：

![image-20210708165853197](https://i.imgur.com/R7ObEmW.png)

比較特別的地方是如果已經沒對應的話，例如上圖的 2，那麼它的對應關系就會是 $\emptyset$，在 Object Detection 中就代表背景。之後會來詳細介紹例子來進一步解釋…

### No anchor, No NMS, No receptive field
因為使用 set prediction 使得 DETR 有以下的特色：

* 不用 NMS 因為所有的集合關系為「一對一」，不像以前 anchor based 的方法會有多對一的問題
* 整體的網路架構非常簡單，不需要因為領域的不同而做對應的細調

## 細看網路架構圖
![image-20210708171653143](https://i.imgur.com/unsItRh.png)

DETR 可分為四個部份：backbone、encoder、decoder、FFN，以下分別解釋：

### backbone
處理的問題非常簡單，輸入為圖片，輸出則為 $(B , C , H , W)$ 的特徵圖。負責找出特徵用的，在 DETR 中會把特徵圖壓縮成 $(B , 2048 , H/32 , W/32)$ 張，也就是放大 5 倍，特徵圖數量為 2048。

接著經過一個 1x1 conv 降維減少運算量使 $(B , 2048 , H/32 , W/32)$ 變成 $(B , 256 , H/32 , W/32)$

但因為要把特徵圖放進 Transformer 的原因，我們要轉換維度 (從 3d 變成 2d)，有點像把圖片用 sequence 來表示的感覺。把$(B , 256 , H/32 , W/32)$ 變成 $(B , 256 , (H/32 \cdot W/32))$

原論文中使用的是 ResNet-50 或 ResNet-101

### encoder
**1. Positional Encoding**
把 backbone 產生的特徵圖，先加上 positional encoding，再放進 encoder，其中 positional encoding 也有做修改，變成二維的編碼了，為了符合圖片是二維的關系。公式改為以下：

$$
\begin{gathered}
PE_{(pos_x,2i)} = sin(pox_x/10000^{2i/128})\\
PE_{(pos_x,2i+1)} = cos(pox_x/10000^{2i/128})\\
PE_{(pos_y,2i)} = sin(pox_y/10000^{2i/128})\\
PE_{(pos_y,2i+1)} = cos(pox_y/10000^{2i/128})
\end{gathered}
$$

小細節的地方是原本特徵數 256 的部份，會平分一半給 x 軸的編碼，一半給 y 軸的編碼，所以各是 128。

把生成的 Positional Encoding 加上 CNN 生成的特徵圖就是 Encoding 的輸入了。如下圖所示：

![image-20210708180309801](https://i.imgur.com/X6K8Cf7.png)

**2. Encoder**
底下是 encoder 與 decoder 的架構：

![image-20210708172344944](https://i.imgur.com/J90hZNT.png)

總結與原 Transformer 不一樣的地方：

* Positional Encoding 改成可考慮二維的編碼
* 且每一個 Block 的輸入都要加上 Positional Encoding (原始是只加再最一開始而已)
* 且 Positional Encoding 只與 Query、Key 相加，不與 Value 相加

最後的輸出維度為 $(B, 256, HW)$，且會把結果送給 Decoder。

### Decoder
Decoder 的變化更大了，他的 input 是一個叫做 Object query 的東東，通常維度設為 $(N, b, 256)$，而這個 $N$ 在原來 Transformer 代表輸出句子的長度，在這裡指的是「要生出多少個 BBox」，這個 $N$ 設越大越好，越大的 $N$ 可以有更多的 BBox 組合可能性，同時付出的計算代價也沒有很大。(因為 Object query 是一個矩陣，其中一維變大而已)

在這裡 Object query 擔任的是一個類似 Positional Encoding 的角色，它會與第一個 self attention 的 query key 相加，與第二個 self attention 的 query 相加。只不過它是一個可以自我學習的 Positional Encoding，不像前一個是人工設定的，可以理解為 Object query 在學習這 100 個 BBox 之間的全局關系。

![image-20210708172344944](https://i.imgur.com/J90hZNT.png)

### FFN
最後 FFN 的地方會分成兩個不同維度的輸出

* 一個是維度 $(B, 100, class + 1)$ 的分類輸出
* 一個是維度 $(B, 100, 4)$ 的 BBox 輸出，4 分別代表的是 $(c_x, c_y, w, h)$

## Loss function
到目前為止我們已經得到了兩個結果：一共 N 個 BBox set 以及預測分類結果。那接下來我們來看 Loss function，問題來了，這些輸出都是無序的阿，完全不知道哪一個 BBox 對應到那一個 Class，在這篇論文使用了一個經典的演算法 **Hungarian Algorithm 匈牙利演算法**，可以來專門解決一對一分配問題。

### Hungarian Algorithm 匈牙利演算法
匈牙利演算法是一個專門來解決指派問題，假設今天有三位工人以及三份工作，每一位工人作工作都有不同的成本，今天在**每一個工作都被分配到的前提下**，找出一個成本最小的組合。例如：

![image-20210708215043817](https://i.imgur.com/D5Z4QeY.png)

可以發現 -> 讓吉姆清潔浴室、史提夫打掃地板、艾倫清洗窗戶時，可以達到最小成本 $6，匈牙利演算法就是在解決這個問題，詳細算法不在這多做說明，有興趣可參考維基百科 [維基百科](https://zh.wikipedia.org/wiki/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95)

對應到我們的例子中，工人就是 100 個 BBox，而工作就是圖中的 Ground truth 類別。假設有一張圖中有 Dog、Horse、Car，我們的矩陣就是一個 $100 \cdot 3$ 的矩陣，如下圖：

![image-20210709011410151](https://i.imgur.com/mk6a73F.png)

選擇一個矩陣中值總合為最小的組合，即為 BBox 對應的分類別了，其中所有對應不到的就算在空集合中，也就是背景類別。有個小地方要注意(這是我個人的理解)，在背景 Background 的部份在第一個出來的版本是會選擇出一個框來框它的，而會在後來的調整中把屬於背景的 BBox 去掉。如原論文下圖最後一步表示：綠色的框不見了。

![image-20210708144748759](https://i.imgur.com/ViOBka4.png)

### Loss 定義
那矩陣中所代表的值就是我們的 Loss 啦，在 DETR 中 Loss 定義為以下：首先是「匈牙利演算法」 -> 總合為最小 Loss 的數學定義：

$$
\hat{\sigma} = arg\underset{\sigma\in\sum_N}{min} \sum^N_iL_{match}(y_i,\hat{y}_{\sigma(i)})
$$

意思為某一真值 $y_i$ 以及 一預測值 $\hat{y}_{\sigma(i)}$ 的所有可能的排列，經過 $L_{match}$ 使 $y_i$ 與 $\hat{y}_{\sigma(i)}$ 的距離為最小。

而 $L_{match}$ 也就是上圖矩陣中的數值一共包含兩個部份：

* class 分類的 cross entropy loss
* BBox 的 loss

$$
\mathcal{L}_{Hungarian}(y,\hat{y}) = \sum^N_{i=1}[-log\hat{p}_{\hat{\sigma}(i)}(c_i)+\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}}(i))]
$$

BBox 的 loss 又包含兩個部份：

* L1 loss
* GIoU

其中的 $\lambda_{iou} \lambda_{L1}$ 為超參數，可調整，代表 BBox Loss 所佔的比重

$$
\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}}(i)) = \lambda_{iou}\mathcal{L}_{iou}(b_i,\hat{b}_{\hat{\sigma}}(i)) + \lambda_{L1}||b_i-\hat{b}_{\hat{\sigma}}(i)||
$$

## 模型訓練方法
所以模型的訓練就可以用白話解釋成：
已知圖上有 Car Dog Horse 三類別，先使用匈牙利算法算出 Loss 最低的組合，再把這個組合與 GT 的 BBox 計算出 Loss，接著 backpropagation 回模型訓練

## Object query 詳解以及 Decoder 詳解
看完了架構，我們來回頭看 Object query，倒底這個 $(100,B,256)$ 的向量做了什麼事，而 Decoder 中的 query key value 又為什麼這樣設計呢？

**Object query (query)**
我們可以把 Object query 看成是有 100 個格子，每個格子有 256 維的向量，每個格子中的 256 維包含了某個類別的訊息，例如 Car 的位置、編碼特徵等等，可理解為這個格子就是在找 Car 的，所以稱為不同 Object 的訊息

**Key Value**
而 Key 和 Value 則是從 Encoder 而來，是經過 Encoder 找出的「圖像全局訊息」，嗯…就是一個綜合特徵感。把 Query 與 Key 計算像是在尋找「某個位置附近有沒有 Car (Object)」，而如果有就經 Value 加權輸出，如果沒有…就什麼也沒有啦 w 就輸出為 0 

**最後**
最後會發現如果與 Fast R-CNN 比較的話，其實 Object query 與 anchor 非常像，只是這個 Object query 的維度為 $(100,B,256)$ 非常高，優點為能夠通過訓練來尋找，且因維度高能表示的特徵也多，缺點為維度太高，訓練時間長，不好訓練。

## Experiments
與 Faster RCNN 對比，在效果上不相上下。

![image-20210709021214812](https://i.imgur.com/z1Effh9.png)

缺點：

* 沒有引入 FPN 所以在小物件上效果不好
* 訓練時間真的太久啦

## 結論
這篇 DETR 可說是 Transform 熱門的先趨，用了非常多的概念，希望能把圖片表示像是 sequence 一樣的來訓練。

使用 Transformer 的好處是可以學習到更多的特徵點，並且輸入輸入概念全部不一樣，全都是變成 Sequence 了，有點像 Seq2Seq 那樣，也因此引申出不用 NMS 的算法，給出了一個全新的思考方向。

這篇論文雖然 AP 與 Faster R-CNN 相當，但帶出的觀念給後來的人非常多的想像，究竟 Transformer 可以到什麼程度呢，讓我們繼續往下看吧 XD

## Reference

https://zhuanlan.zhihu.com/p/340149804

https://medium.com/%E8%BB%9F%E9%AB%94%E4%B9%8B%E5%BF%83/detr%E7%9A%84%E5%A4%A9%E9%A6%AC%E8%A1%8C%E7%A9%BA-%E7%94%A8transformer%E8%B5%B0%E5%87%BAobject-detection%E6%96%B0pipeline-a039f69a6d5d

https://zhuanlan.zhihu.com/p/326647798