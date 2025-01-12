---
title: AutoML - MobileNet v1~v3 與 EfficientNet v1 回頭閱讀
mathjax: true
date: 2022-01-23 16:48:12
tags: AutoML
categories: 電腦視覺整理
---

時間來到 2017 ~ 2019 年，在這期間 Google 依序提出基於「輕量化」的神經網路 MobileNet v1~v3，在相同效果的條件下，運算量少了非常之多。而 2019 年 EfficientNet 則繼承了這項重責大任，把 NAS 應用在 MobileNet 上，找出最佳的排列組合。結果是非常驚人的，在效率及效果均刷新 SOTA 好幾個百分點，並為 CNN 的發展打下了非常牢固的基礎。

keywords: EfficientNet、NAS、MobileNet
<!--more-->

## 什麼是 MobileNet ？

在介紹 EfficientNet 之前，先來很簡單的說一下什麼是 MobileNet。

MobileNet 是 Google 2017 首次提出的網路架構，目的是在降低網路的運算量及參數使用量，使得深度學習可以應用在日漸發展的物聯網、移動平台上

以下依時間順序，簡單講解：改進了什麼？以及為什麼這樣改？

如果有想要更進一步了解更多東西的話，可以參考下列文章：

[MobileNet 演變史](https://chihangchen.medium.com/%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98-mobilenetv3%E6%BC%94%E8%AE%8A%E5%8F%B2-f5de728725bc)

### 2017 MobileNet v1

[https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

MobileNet v1 主要應用 **Depth-wise Separable Convolution**，把一個 Convolution 運算拆解成 Depthwise Convolution 以及 Pointwise Convolution。雖然這個概念不是 MobileNet 原創，而是由 [Xception](https://arxiv.org/abs/1610.02357) 這篇論文提出的，但 MobileNet 仍把它發揮得淋漓盡致。網路架構圖如下：

![image-20220124145825257](https://i.imgur.com/zjQuFO3.png)

而為什麼 Depthwise Separable Convolution 可以降低運算量呢？假設我們要把一張大小為 32x32x3 的圖片，經過一個 3x3 kernel 特徵圖放大為 64，則原 Convolution 總運算為：
$$
(32\times32)\times3\times64\times(3\times3)=1769472
$$
而使用 Depthwise Separable Convolution 的運算則為：
$$
\begin{gather}
[(32\times32)\times3\times1\times(3\times1)]+
[(32\times32)\times3\times64\times(1\times1)]=205824
\end{gather}
$$
兩個運算相差近 10 倍，可說 Depthwise Separable Convolution 省下了非常非常多的運算，然後省下運算也代表著網路的「彈性」變小，理論上限效果會變差，但根據這篇論文其實效果並為減少太多。

### 2018 MobileNet v2

[https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

MobileNet v2 與 v1 最大的差別在於，v2 多引入了 bottleneck block 與 inverted-bottleneck block 架構，再運算量又進一步減少。架構如下：

<img src="https://i.imgur.com/kn5AjKz.png" alt="image-20210426130521052"/>

bottleneck block 的核心在於，輸入特徵圖會先經過一個 1x1 conv 做放大/縮小的運算，接著做一 3x3 的 Depthwise Convolution，再用一個 1x1 conv 變回原本維度。這種把特徵圖放大再縮小就是 bottleneck 的特色了。

因為每一個 conv 後都會做 ReLU 的關系，作者經實驗發現 inverted bottleneck 的效果最好。因為如果 3x3 特徵圖數太小，很有可能大部份的特徵值都會被 ReLU 化為 0，網路就學不到東西了。

除了使用 inverted bottleneck block 外，作者也在最後一個 1x1 conv 使用 linear activation 線性的函數，避免太多的 ReLU 非線性 block 破壞了網路的結構

另外在 MobileNet v2 中，開始以 stride 2 取代 2x2 pooling 達成降維操作

### 2019/5 MobileNet v3

[https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)

MobileNet v3 與 v2 最大的差別在於加入了 SENet 以及使用 NAS。

SENet 全名 Squeeze and Excitation 是一個類似 Attention 想法的網路，放大重要的特徵，縮小不重要特徵，並加入了 GAP Global Average Pooling 計算每個 Feature Map 的權重。

![image-20220124155129171](https://i.imgur.com/yMSANwl.png)

並且透過 NAS 找出了一個最佳排列組合的網路

## 什麼是 EfficientNet ？

就在 MobileNet v3 提出的同月，Google 發表了 EfficientNet v1，照抄了 MobileNet v3 的架構，在修改 NAS 的 Search Strategy 後，結果好到直接把 MobileNet v3 甩到牆上

### 2019/5 EfficientNet v1

[https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

EfficientNet 的核心想法認為：在以往設計網路時，常常加強網路的三個面向以得到更好的效果：深度、寬度、解析度，如下圖所示：

![image-20220124204200111](https://i.imgur.com/qX2iVPC.png)

加寬代表 (圖 b)：增加 Feature Map 也就是 Channel 的數量，可以得到更多的特徵組合

加深代表 (圖 c)：使網路學習到更多更複雜的特徵

加解析度代表 (圖 d)：在做 Object Detection 時，有時影像中的小物件效果不好，可以增加解析度來得到更好的效果

但是作者認為這三個東西並非是三個獨立的參數，不應該每次只調整其中一個而已 (如深度)，應該是三個參數一起找一個最佳組合才對，而作者稱這種方法叫 Compound Scaling (圖 e)

如果以上想法用數學公式來表達的話，如下式：

假設 input 是 $X$ 經一層卷積運算 $\mathcal{F}_i()$ 得到 output $Y$ ，而 $i$ 代表的是第 $i$ 層卷積運算

如果今天網路有很多卷積運算，則可得到下列表示：
$$
\begin{align}
\mathcal{N}&=\mathcal{F}_k\odot...\odot\mathcal{F}_2\odot\mathcal{F}_1(X_1)\\
&=\bigodot_{i=1...s}\mathcal{F}_i^{L_i}(X_{(H_i,W_i,C_i)})
\end{align}
$$
在以上式為基準之下，調整 $d,w,r$ 參數，使得準確率為最大：
$$
\begin{align}
\max_{d,w,r}&\quad\mathrm{Accuracy}(\mathcal{N}(d,w,r))\\
s.t.&\quad\mathcal{N}(d,w,r)=\bigodot_{i=1...s}\hat{\mathcal{F}}_i^{d\cdot \hat{L}_i}(X_{(r\cdot \hat{H}_i,r\cdot \hat{W}_i,w\cdot \hat{C}_i)})
\end{align}
$$
並且額外加入兩個條件式，在記憶體使用量及運算量都要小於一定值：
$$
\begin{align}
&\mathrm{Memory}(\mathcal{N})\leq\mathrm{target\_memory}\\
&\mathrm{FLOPS}(\mathcal{N})\leq\mathrm{target\_flops}
\end{align}
$$
作者在規劃調整參數有時兩個發現：

發現一：各個參數在加大時，準確率「提升程度」越來越小，白話說：付出的計算成本與效果 cp 值越來越低

![image-20220125151315306](https://i.imgur.com/A5E4Z6o.png)

發現二：參數不能只調整單個，要整體來考慮。下圖為固定 w 下調整 d, r 的結果，發現調哪個參數對網路影響最大都不是一定的

![image-20220125151722668](https://i.imgur.com/A3fF45y.png)

根據上列發現，作者設計了以下限制式：
$$
\begin{align}
\mathrm{depth}&:d=\alpha^\phi\\
\mathrm{width}&:w=\beta^\phi\\
\mathrm{resolution}&:r=\gamma^\phi\\
\mathrm{s.t.}&\quad\alpha\cdot\beta^2\cdot\gamma^2 \approx2^\phi\\
&\quad\alpha\ge1,\beta\ge1,\gamma\ge1
\end{align}
$$
透過調整 compound coefficient $\phi$ 設計出不同大小的網路，且要找出 $\alpha\beta\gamma$ 三者相乘後最接近 $2^\phi$ 的組合。

至於為什麼 $\beta\gamma$ 要加個平方項呢？因為當我們放大網路寬度、解析度時，是同時對圖片的「長寬」同時放大，因此運算量也呈平方關系。而深度因圖片數量一樣，只是多做幾次而已，與運算量程線性倍數關系。

至於為什麼是要小於 $2^\phi$ 呢？嗯…可能是作者經實驗或經驗得來的吧，論文中並未明確給出解答，是個神奇的 magic number 呢。但總之作者經上述式子令 $\phi$ 時，找到當 $\alpha=1.2$ $\beta = 1.1$ $\gamma = 1.15$ 時效果最好，並把此倍大倍率套回 MobileNet v3 的架構中，得到 EfficientNet-B0 架構

![image-20220125155242575](https://i.imgur.com/nxWry7B.png)

最後放大 $\phi$ 得到 EfficientNet-B1~B7 不同大小的架構

![image-20220125155355722](https://i.imgur.com/ehgVs2g.png)

最後這是 EfficientNet 與 SOTA 的比較，在相同效果下，運算量少了近 5 倍以上

![image-20220125155500013](https://i.imgur.com/wCldkxc.png)

## Reference

[cdsn 文章](https://blog.csdn.net/qq_37541097/article/details/114434046)

[講得很棒的 EfficienDet Youtube](https://www.youtube.com/watch?v=qeCi-Qo1OcA)
