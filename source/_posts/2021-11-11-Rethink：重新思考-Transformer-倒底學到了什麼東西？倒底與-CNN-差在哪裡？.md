---
title: Rethink：重新思考 Transformer 倒底學到了什麼東西？倒底與 CNN 差在哪裡？
mathjax: true
date: 2021-11-11 19:22:34
tags: Vision Transformer
categories: 電腦視覺整理
---

前面看了這麼多不同的 Transformer 網路架構，不仿現在稍微停下腳步，回頭看看一些最基本的概念及問題：**倒底 Transformer 比 CNN 好在哪裡？**。究竟是什麼原因使得現在 Transformer 可以在各大題目上刷新 SOTA，而究竟 Transformer 創新的地方在哪裡？

keywords: Transformer、CNN
<!--more-->

## 回過頭來看看 ViT

![Image](https://i.imgur.com/abtdhSj.png)

總的來說 ViT 的架構流程可以用以下方式來表達：
1. 把 $x\in H\cdot W\cdot C$ 圖片依照 Patch 大小切塊成 $N\cdot (P\cdot P \cdot C)$ 的三維 Patch
2. 再把三維 Patch reshape 成二維向量 $N\cdot (P^2 \cdot C)$
3. 再把 $x_p\in N \cdot(P^2 \cdot C)$ 經可學習 Linear $E$ 後得到 $N\cdot D$，這一步叫 Patch Embedding
4. 再加上 Positional Embedding
5. 再加上 Class token
6. 經過好幾層 Transformer 
7. 輸出分類結果

### Self-Attention Layer 代表著什麼？

首先我們來回顧始組 CNN

在 CNN 中我們都知道會有 kernel 去一層層的提取特徵，同時我們也會加入 pooling 或是 stride 去縮小解析度，提取出來的 feature map 會一層層大縮小解析度並增加特徵。

這一步的用意除了減少運算量，還可以使 CNN 能有更大的感知域 (receptive field)，如下圖所示。使得 CNN 除了在一個 kernel size 中有很強的關聯性外，也能夠與 kernel size 外的點，也有關聯性

![Image](https://i.imgur.com/g1LIhgr.png)

同時因為卷積的特性使 CNN 還有 Locality 與 Spatial Invariance 兩種特性，也就是局部注意力與空間不變性。因為只會加強關注 kernel 內的訊息而有 Locality 特性、因為 kernel size 會平移，使得同樣特徵不管出現在圖片的任何地方都可以偵測出，而有了 Spatial Invariance 的特性

以上這些特性又可合併成為 Inductive Bias (歸納偏置)，意思為綜合以上特性，使得 CNN 在尋找圖片中的特徵有非常強的歸納能力 (Inductive)，同時面對沒看過的資料也能作出正確的推論 (Bias)

---

接下來來看看 Transformer

Transformer 與 CNN 最大的不同在於
1. 沒有卷積，而改用 Self-Attention
2. 圖片解析度不會隨層數縮小

Self-Attention 的核心公式如下所示：

$$
A=softmax(\frac{QK^T}{\sqrt{d}})V
$$

![Image](https://i.imgur.com/Bmql5qV.png)

如果以瀏覽器為例子的話：Q 代表 Query，可代表使用者下的關鍵字，K 代表 Key，可代表瀏覽器後台的 Database，V 代表 Value，可代表結果的權重值。

因此 Self-Attention，可看成使用者下關鍵字 Q，瀏覽器會去資料庫中與每個資料 K 做相關性運算，得出最有相關的資料後，再乘上 V 權重值，使結果想搜尋列一樣越有關的排在越前面。

如果換成影像也是同理，Self-Attention 會去計算 每個 Patch 與 Patch 中的每個像素與像素之間的關系，最後得出的值再乘上權重。最後得到一個特徵圖 (feature maps)

![Image](https://i.imgur.com/vAzIUSO.png)

因為 Self-Attention 沒有使用 kernel，因此它的 receptive field 是整張圖片。同時也因沒有使用 kernel 因而失去了 Locality 與 Spatial Invariance 兩種特性，進而導致 Transformer 沒有 Inductive Bias 的能力。

沒有 Inductive Bias 的重點是沒有了「Bias (偏置)」的能力，什麼意思呢？意思就是少了遇到沒見過的資料的選擇能力。在相同資料訓練的前提下 CNN 看到沒見過的資料有更強的推論能力，而 Transformer 無法做到這一點

為了彌補 Transformer 這個缺點，需要使用大量的資料去訓練它，使它看過更多的資料，直接把可能會遇到沒看過的圖片的可能性去掉，就是再也不需要 Bias(推論) 這項能力。

---

最後的結論可整理為下列表格：

CNN 的優點：
* Locality
* Spatial Invariance 
* Inductive Bias

Transformer 的優點：
* Global Receptive Field

### Self-Attention 學到了什麼？

由上面結論可以看出，Transformer 少了一些 CNN 的優點，同時多了一個 Global Receptive Field，接下來來仔細看看到底 Transformer 的特徵圖中學到了什麼？

根據 ViT 論文後面實驗的部份，作者有把 Patch Embedding 中可學習的向量 $E$ 拿來做可視化，如下圖：

![Image](https://i.imgur.com/PHXp4bp.png)

可以發現經過一層 MLP 後，裡面學習到的東西與 CNN 非常的類似，都是一個像特徵域的東西。

以下是我的想法：MLP 是 Transformer 最主要提取特徵的主要來源

![Image](https://i.imgur.com/DSnccCE.png)

不管是 Patch Embedding 中的 MLP 或是 Encoder 中的 MLP，它們都含有類似像 CNN 一樣找特徵的能力，找到特徵後再送至 Self-Attention 做特徵的強化，隨著層數的增加，在 MLP 與 Self-Attention 的共同努下找到的特徵就越來越清晰了

其實在 CV 最一開始發展的時候，那時候還沒有 CNN，大家都是使用 MLP (多層感知機) (或是稱全連接層 FFN)，但是因為 MLP 的運算量太大，因而才有了後續 CNN 的誕生。而且在 CNN 發展的過程中，最後用來輸出分類結果的全連接層也慢慢被淘汰掉了

雖然 MLP 計算量大但是它更 General、網路更有彈性，能計算到的特徵數更多。而 CNN 算是 MLP 的一個特例，透過 kernel 的限制使得網路計算量少且更好訓練，但也因為限制範圍而使 CNN 的效果有一定的上限。

但隨著科技的進步、計算力的增加，這些計算量比較大的方法有慢慢回歸的趨勢，大家慢慢使用更多的資料集去訓練，更多的計算力去計算。

---

接著來看看 Global Receptive Field 的部份

同樣是根據 ViT 的論文作者分析了在 self-attention layer 各層中各個 attention head 之間的關系，以 Mean attention distance 作為分析目標


Mean attention distance 的意思指的是，一個 pixel 能最遠與附近的其它 pixel 做相關性運算，也可以理解為就是 CNN 中的 receptive field (空間感知域)

依據實驗結果可看到在網路第一層，假設網路中有 16 個 head，這 16 個 head 它們的 receptive field 有的大有的小，有些 head 天生就可以有比較 Global 的感知域，而有些則是比較 Local 的感知域。

隨著層數的增加，每個 head 的 receptive field 也隨之增加，意謂著層數越深越能看到更全局 Global 的資訊

與 CNN 不一樣的是，CNN 在一開始並不會出現全局的感知域，而是像底下藍線一樣，隨著層數而呈線性關性，但 Transformer 能做到的是紅色圈圈部份，這些早期全局資訊是 CNN 所沒有的。

![Image](https://i.imgur.com/spEkO2q.png)

---

參考 Google 在 2021 年 8 月發表的論文

[https://arxiv.org/pdf/2108.08810.pdf](https://arxiv.org/pdf/2108.08810.pdf)

論文裡面使用了實驗數據來比較 Transformer 與 CNN 的差別

下圖也是一個 Transformer Global Receptive Field 很好的例子，橫縱軸分別代表為網路兩兩層之間相互的關系，座標值越大代表層數越深，可以看到 Transformer 在每個層上兩兩都有關系，但 ResNet50 最淺的層與最深的層兩層的關系非常弱，由圖可知 ResNet50 暗色的部份正是 Transformer 所彌補的強項

![Image](https://i.imgur.com/BlozGy1.png)

下圖則是 Receptive Field 的比較，可看到Transformer 則是在第六層就有全局的資訊出來了，而 CNN 要等到 16 層才慢慢有全局的資訊出來

不過也可以在這張圖中發現，CNN 的局部關注力真的很強，非常大且非常黑

![Image](https://i.imgur.com/zhNvCxd.png)

---

結論：

Transformer 所謂的 Global Receptive Field 指的是：「在網路淺層時期就能有全局的概念」，並非指 CNN 就沒有全局的概念，而是 CNN 需要到網路非常深的時候才能顯現出這項特點

### 為什麼一定要 Reshape？

也許你也注意到了，Transformer 的輸入是二維的，在輸入到 Transformer 前會做分 Patch 以及 reshape 這兩步，那不禁讓人懷疑

1. 直接把圖片分塊，塊與塊之間彼此互不相關，合理嗎？
2. 直接把三維圖片 reshape 成二維，在數學的角度上意義為？

在目前為止我認為這部份的解答只有一個

**大家就只是很直覺把 Transformer 好奇的拿到 CV 上來試試看效果，沒想到效果竟然還不錯**

所以最一開始的 ViT，基本上都是從 NLP 的觀點來出發的，網路輸入要二維？那我就直接 reshape 你；網路輸入是字詞？那我就直接把圖片切塊

以上這些操作目前我認為沒有任何實際上的意義，但是！實驗證明 Transformer 就是有它厲害的地方！它好是一定有道理的，我們不仿試著去想想看這些操的合理性，以及想一下文字與圖片倒底差在哪裡？

在以下的影片中有提到觀念：

[https://www.youtube.com/watch?v=aH7s6qXEUcc](https://www.youtube.com/watch?v=aH7s6qXEUcc)

在 NLP 中一串句字中的每一個字都有相對應的編碼，可能是 one-hot encoding 也可能是 token based encoding，總之句子中的每個字都有對應的編碼用來表示特徵向量

![Image](https://i.imgur.com/l3huXd8.png)

而在圖片中資訊是用長寬來表示的，但是如果我們換個角度想：想想電腦是怎麼理解一個二維矩陣的？是從左上角為原點一列一列的往右掃，直到最右下角的點。

![Image](https://i.imgur.com/6A0TNnn.png)

這時我們再以每一列拿出來排成一排，像是把圖片中的一列當成是句字中的一個詞，圖片中的行代表句字的長度，我們就可以得到用文字的方式來表達的圖片了。如下圖所示：

![Image](https://i.imgur.com/x8PLYi7.png)

可以想像一列就是一個詞，每一列都會送到 Transformer 中與其它的每一列做 Self-Attention 相關性運算。到目前為止皆與文字相同，但是圖片有一個最大不同的點

除了每一列之間要計算相關性外，列之間的每一個像素也應該要計算相關性，因為文字中的向量表示是這個「詞」的特徵，但是在圖片中一個列的向量的函意不只是一個特徵，更是其中的一個個像素。

而這正是 Transformer 擅長與其它人做相關性計算 (全局 Global)，而比較不擅長與自己內部做相關性計算 (局部 Local) 的另一個方面的解釋

### Transformer 有什麼缺點？

綜觀以上與 CNN 之間的比較，我們接下來來看看除了與 CNN 之間的差異，Transformer 本身有什麼樣的問題？

[https://bbs.huaweicloud.com/blogs/298123](https://bbs.huaweicloud.com/blogs/298123)

參考以上文章 Transformer 的缺點可分為以下五大類：

1. 資料需求量大
2. 計算量大
3. 堆疊的層數的限制
4. 模型本身無位置編碼
5. 局部注意力較弱

#### 1. 資料需求量大

這點在上面有仔細分析過了，Transformer 一是缺少了 Inductive Bias 需要更多資料去彌補。而造成沒有 Inductive Bias 的主要原因是不使用 CNN 而使用 MLP 做為特徵提取器，MLP 本身更 General 也因此需要使用到更大的資料集去訓練

#### 2. 計算量大

假設 Batch 為一，圖片經 CNN 後的維度會變成 $(H \cdot W \cdot C)$ 的特徵向量，後經 reshape 變成 $(HW \cdot C)$ 再加上 Positional Enbedding 後放進 Transformer。其中 $(HW \cdot C)$ 可看成長度為 $HW$ 大小為 $C$ 的 sequence

![image-20210709115659980](https://i.imgur.com/6pAB8h3.png)

以下 $N_q N_k$ 其實就是 $HW$，則輸入向量 $(N \cdot C)$，乘上一個 $W$ 轉換矩陣 $(C \cdot 1)$ 則計算 self attention 的時間複雜度為：

$$
O(N_qC^2 + N_kC^2 + N_qN_kC)
$$

分別對應

$O(N_qC^2)$ 計算 Query 的複雜度

$O(N_kC^2)$ 計算 key 的複雜度

$O(N_qN_kC)$ Attention 的複雜度 $(N_qC \cdot CN_k) = (N_qN_k)$

![image-20210709120129060](https://i.imgur.com/gr4y6sI.png)

透過以上可以發現當圖片的解析度越大，Attention 的計算複雜度為所有像素數量的平方，也就是 $(HW)^2 = N^2$ ，這就導致 Transformer 參數使用量及計算量特高的原因

#### 3. 堆疊的層數的限制

根據以下這篇論文的實驗 (DeepViT)
[https://arxiv.org/abs/2103.11886](https://arxiv.org/abs/2103.11886)

發現隨著網路層數的增加，各個 attention head 所關注的資料也漸漸靠攏全局，而之間的相關性也會漸漸上升。

![Image](https://i.imgur.com/nG54UWa.png)

因此如果單純的疊加層數，效果反而不一定會變更好

#### 4. 模型本身無位置編碼

可以參考我之前寫過的文章

[CNN 與絕對位置資訊 - CNN 倒底學到了什麼？](https://mushding.space/2021/07/30/CNN-%E8%88%87%E4%BD%8D%E7%BD%AE%E8%B3%87%E8%A8%8A-CNN-%E5%80%92%E5%BA%95%E5%AD%B8%E5%88%B0%E4%BA%86%E4%BB%80%E9%BA%BC%EF%BC%9F-Position-Padding-and-Predictions-A-Deeper-Look-at-Position-Information-in-CNNs-CNN-%E8%88%87%E7%B5%95%E5%B0%8D%E4%BD%8D%E7%BD%AE%E8%B3%87%E8%A8%8A/)

[Vision Transformer 演化史: Conditional Positional Encodings for Vision Transformers - 可變序列長短的 Positional Encoding](https://mushding.space/2021/07/28/Vision-Transformer-%E6%BC%94%E5%8C%96%E5%8F%B2-Conditional-Positional-Encodings-for-Vision-Transformers-%E5%8F%AF%E8%AE%8A%E5%BA%8F%E5%88%97%E9%95%B7%E7%9F%AD%E7%9A%84-Positional-Encoding/)

Transformer 因為解決了 RNN 訓練時間過長的問題提出平行化訓練，但同時也捨棄了 RNN 的 Inductive Bias 也就是時間上的假設。因此 Transformer 無法了解各字詞之間的順序關系。

而 CNN 根據實驗，因 kernel 及 padding 的關系，使網路學習到了相對的位置資訊。

為了補足 Transformer 沒有如同 CNN 一樣的位置資訊，因此加上了 Positional Encoding。

但也有一些實驗試著結合 CNN 使 Positional Encoding 用更自然的方式實現。

#### 5. 局部注意力較弱

一個 Patch 內部像素之間的相關性，不如 CNN 來得強
未來實驗可以朝前半段使用 Transformer 後半段使用 CNN 來互補

![Image](https://i.imgur.com/HoSmP7V.png)

## Reference

### ViT
[(Youtube) An image is worth 16x16 words: ViT | Is this the extinction of CNNs? Long live the Transformer?](https://www.youtube.com/watch?v=DVoHvmww2lQ)

[(Youtube) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4)

[(Youtube) Vision Transformer (ViT) - An image is worth 16x16 words | Paper Explained](https://www.youtube.com/watch?v=j6kuz_NqkG0)

[Vision Transformer 超详细解读 (原理分析+代码解读) (二)](https://zhuanlan.zhihu.com/p/342261872)

[目前Vision Transformer遇到的问题和克服方法的相关论文汇总](https://bbs.huaweicloud.com/blogs/298123)

### ViT vs CNN

[(arxiv) Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/pdf/2108.08810.pdf)

[(arxiv) Transformers in Vision: A Survey](https://arxiv.org/pdf/2101.01169.pdf)

### ViT vs MLP

[歸納偏置多餘了？靠“資料堆砌”火拼Transformer，MLP架構可有勝算？](https://www.gushiciku.cn/pl/gXcq/zh-tw)

[CNN vs Transformer、MLP，谁更胜一筹？](https://zhuanlan.zhihu.com/p/405295929)

### Other

[(Youtube) Transformers can do both images and text. Here is why.](https://www.youtube.com/watch?v=aH7s6qXEUcc)

[如何理解Inductive bias？](https://www.zhihu.com/question/264264203)

[(超猛視覺化 Self-Attention) Getting meaning from text: self-attention step-by-step video](https://peltarion.com/blog/data-science/self-attention-video)

[Attention and Transformers](https://theaisummer.com/topics/attention/