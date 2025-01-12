---
title: RCNN 全家桶速讀：R-CNN、Fast R-CNN、Faster R-CNN
mathjax: true
date: 2022-06-20 20:20:33
tags: 
  - Object detection
categories: 電腦視覺整理
---

碩論題目下來了，是有關於 3D 的瑕疵辨識，根據前面學長姐的題目來看，看起來這個題目偏向分類任務，可能是瑕疵的二分類吧…，不管題目是什麼，還是多看看一些論文為未來鋪路吧 XD，搞不好哪一天真的用上了。

接下來就先來看看 Object Detection 目標偵測的元老：RCNN 系列吧

keywords: R-CNN、Fast R-CNN、Faster R-CNN、Object Detection
<!--more-->

## 什麼是 Object detection

在介紹之前我們先來看看傳統的深度學習電腦視覺任務，可大概分成下列 4 個任務：分類、定位、偵測、分割
![Image](https://i.imgur.com/OsA93iB.png)

所謂分類是指：找出影像中的目標物體類別；定位是指要找出目標物體所在的空間範圍；而偵測是分類 + 定位的結合，且有時還會加上多類別的任務；最後一個分割同樣也是分類 + 定位，它會描出物體的輪廓，不過它比較像「像素」級別的分類

分類的目標為輸出 k 個不同的物體的類別，通常使用 cross entropy 算出 0 ~ 1 之間的機率，來表示某類別的可能性；定位的目標為輸出 4 個值 (x, y, w, h) 用來表示框框的 (起始點、長寬)，通常使用 IoU 來算出交集的條件機率 0 ~ 1

IoU 的定義為：兩框框的 聯集/交集，如下圖。通常我們定義 IoU > 0.5 就是一個不錯的效果
![Image](https://i.imgur.com/uWG7Izw.png){ width=50% }
<p align = "center">
IoU 定義
</p>

我們已經定義好了 IoU 後，但是我們要怎麼在一張影像上選出框呢？一個最直覺的方法是窮舉暴力，把所有 pixel 的框框排列組合一遍…？聽起來效率就超低，於是就有了後續的 Object detection 來解決這一系列問題…

## R-CNN
原 paper 連結：[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)

### 網路架構

2013 年提出的 R-CNN 是第一個使用 CNN 來實作 Object detection 的網路架構

![Image](https://i.imgur.com/BPQPRjE.png)
<p align = "center">
R-CNN 完整架構圖
</p>

### Selective Search

提出 Selective Search，它是一個改善窮舉的演算法，一張影像經過 Selective Search 後會生成出 2k 個 Region Proposal。這個 Selective Search 因為是傳統演算法的緣故，不能放進 GPU 中加速。

![Image](https://i.imgur.com/ONnVaHs.png)
<p align = "center">
Selective Search 示意圖，在影像上有規律的找出許多框框
</p>

接著再把圖片 warped (大小弄成一樣)，好放進後續 CNN(AlexNet) 做訓練。這個 CNN 相當於特徵提取器 backbone，負責找出 wraped 過後的框框的特徵，再用 SVM 做分類 (跟現在直接全連結不大一樣)，雖然這些步驟現在看起有點過時，但在那個年代效果超好。

### Problems of R-CNN
1. 每一個 Region Proposal 都要經過一次 CNN 運算 -> 超極慢
2. 可以很明顯發現網路是兩階段：(先用 Selective Search、再用 CNN 找特徵)，不是 end-to-end 架構
3. 分類、BBox 是分開的網路，Loss 也是個別計算
4. 找出來的 Region Proposal 要事先存在本地，浪費硬碟空間


## Fast R-CNN
原 paper 連結：[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)


### 網路架構

2015 年又提出 Fast R-CNN，目的在改善 R-CNN 太慢的問題，首先把 Region Proposal 改名為 RoI (Region of Interest 感興趣的範圍)。及提出 RoI Pooling 使得 Fast R-CNN 在後半提取特徵的部份皆是 CNN 架構。

![Image](https://i.imgur.com/1u91AnU.png)

### CNN 權重共享

網路先把整張影像放進「全」CNN 網路 (VGG) 中找特徵，接著找出 RoI 的範圍 (這裡還是用特別的演算法獨立先算出來的)，由於 CNN feature map 的特性，我們可以先找出 RoI 在原圖的座標，接著直接映射到 feature map 上，如圖左邊的紅框框，這麼做的好處是：CNN 權重是共享的，且只需做一次 CNN，不需要有幾個 RoI 就要做幾次 CNN

再找一次 RoI -> RoI Pooling 作用在於把大小不同的影像變成一樣大小
multi-task loss，把分類、BBox 的 Loss 合併在一起
end-to-end model

### RoI Pooling
RoI Pooling 的目的與 R-CNN warped 相同，皆是把影像變成相同大小，只是原本的做法是直接 scaling，缺點也很明顯：影像比例嚴重失真。改進的方法是使用 max pooling 來取代 scaling。

詳細做法：pooling 取的範圍不再是一個正方型，(例 2x2 pooling，就是在 2x2 選一個最大值替代)，改成一個長方型，它的長寬是：RoI 除以目標大小，去小數。對，其實 RoI Pooling 簡單說就是在去小數而已，也因去了小數，會讓資訊不準 (多框一點，少框一點)。

![Image](https://i.imgur.com/R6L3Y4N.png)

更多詳細介紹可參考以下這篇文章 [Understanding Region of Interest (RoI Pooling)](https://erdem.pl/2020/02/understanding-region-of-interest-ro-i-pooling)。也可直接記結論：RoI Pooling 就是利用不規則的 pooling 來達到輸出影像大小皆相同

### multi-task loss
把下游的分類、BBox 迴歸任務合併在一起，設計出多任務的 Loss，其實就是把兩個不同任務的 Loss 直接相加，這樣做的好處是速度快，只需跑一次網路兩個一起訓練，缺點就是理論上會比分別算的網路不精確一些

### Problems of Fast RCNN
還是用 selective search 來決定 RoI，這個還是很花時間…

## Faster RCNN
原 paper 連結：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)

### 網路架構

![Image](https://i.imgur.com/cLGmQkc.png){ width=50% }

提出 RPN (region proposal network) 專來解決 selective search 無法加速的問題。雖然說 selective search 及 RPN 在運算複雜度上並沒相差太多，但 RPN 因是神經網路形式所以可用 GPU 加速。

網路架構大多改動前半段，新增 RPN 用來找出框框，且這個框有個特別的名字：Anchor (中文可翻錨點、先驗框)。後半段基本上沒什麼變動了

### RPN
RPN 的輸入是來自 CNN 去掉全連接層的 Feature map，輸出同樣為 RoI

Anchor 先驗框，可看成預先列出幾個，事先設計好的框 (不同長寬比例、不同大小比例)，再依據 Anchor 微調，找出真正的框

![Image](https://i.imgur.com/pK8N6lx.png){ width=50% }

RPN 的流程如下：會先用預先設計好的 Anchor 來當 window，依據一定的 stride 在 feature map 上移動 (設 stride 是為了更有效率的灑網)，接著經過 RoI Pooling 把影像變為一樣再放進 CNN 中，最後一樣有兩個任務，一是 2k 的分類任務 (背景、目標物)、一是 4k 的 BBox (x, y, h, w)

![Image](https://i.imgur.com/jJpTWSr.png){ width=50% }

這還沒結束喔，記得以上步驟都只是 RPN 而已，接著會把 4k 的 RoI 結果先做一遍 IoU，如果 IoU > 0.7 當做正樣本、如果 IoU < 0.3 當做負樣本，其餘區間直接捨棄。

接著會把 RPN 所生出來的 RoI 再用 Fast R-CNN 一模一樣方法做下去 (經 VGG 再有 k 分類任務及 4k BBox 任務)

### Loss

$$
\begin{align}
\mathcal{L}(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_{i}L_{cls}(p_i,p_i^*)+\lambda \frac{1}{N_{reg}}\sum_{i}p_i^*\mathcal{L}_{reg}(t_i,t_i^*)
\end{align}
$$

式子分為左右兩邊，左邊為分類任務 Loss，簡單做了一個 cross entropy，而右邊為 BBox 的迴歸 loss，迴歸 loss 中的 $t$ 可再展開如下：

其中 $x_a$ 為 Anchor、$x$ 是 predict 預測 Box、$x^*$ 是 ground truth 

$$
\begin{align}
t_x = (x-x_a)/w_a&,\quad t_y = (y-y_a)/h_a \\
t_w = \log(w/w_a)&, \quad t_h = \log(h/h_a) \\
t_x^* = (x^*-x_a)/w_a&,\quad t_{y^*} = (y^*-y_a)/h_a \\
t_w^* = \log(w^*/w_a)&, \quad t_h^* = \log(h^*/h_a)
\end{align}
$$

所以 $t$ 的意思就是「預測」的 BBox 與「Anchor」 的 BBox 的誤差，而 $t^*$ 的意思就是「Anchor」 的 BBox 與 「Ground truth」 的誤差，而 $t$ 與 $t^*$ 做 Loss 就代表它們兩個越像越好，我理解為有點像在 Anchor 與 Ground truth 中間找個中間框，距離到它們兩個剛好相等

最後還加入了 smooth label 來平滑化標籤，避免網路難以收斂

## 論語

以上大概介紹了 R-CNN 家族演變史，可看到網路架構不斷的往加速發展，且最後也實驗了全部是神經網路的 end-to-end 架構，不需要再分兩個不同的網路來訓練了。

## Reference
[Introduction of RCNN,Fast RCNN,Faster RCNN 中文，講得很清楚，大部份是參考這個影片](https://www.youtube.com/watch?v=M1mN03REGU8&t=356s&ab_channel=AshingTsai)

[某線上課程 ppt](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)

[RoI Pooling 超圖解](https://erdem.pl/2020/02/understanding-region-of-interest-ro-i-pooling)