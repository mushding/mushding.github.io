---
title: 使用深度學習在 super resolution 整理 (二)
mathjax: true
date: 2021-07-06 14:26:34
tags: super resolution
categories: 電腦視覺整理
---

繼上一篇的文章，本篇文章將會重點式的整理各個網路所要解決的問題，以及提出的改進方法。

keywords: SRCNN, FSRCNN, DRRN, EDSR
<!--more-->

## SRCNN
這是 SR 領域的開山第一篇論文，架構非常簡單，只包含了三層 CNN 

![image-20210706143720694](https://i.imgur.com/CrLEPCq.png)

SRCNN 首先把 LR 使用 bicubic 將圖片放大至目標大小，接著作者提出了三步驟：Patch extention、Non-linear mapping、Reconstruction。簡單來說就是經過：找特徵、組合特徵、upsampling 三個步驟，而這三個步驟也是 SR 最重要的核心想法。

卷積層使用的 kernel size 大小分為9x9，1x1 和 5x5。用 Timofte資料集，和 ImageNet pre-train。使用 MSE 作為 loss function。

## FSRCNN
主要是針對 SRCNN 改進了三點：一、在最後一層使用了 upsampling，解決了 SRCNN 輸入圖片還要經過 bicubic 放大的問題。二、改變卷積 kernel size 有更快的效果。三、共享 mapping 層，使如果要更改放大倍率的話可以直接 fine tune。

FSRCNN 不用先把圖片放大，也把 kernel size 變小，速度上加速了不少。
![image-20210706145159922](https://i.imgur.com/G0J1tLA.png)

FSRCNN 可分為五個步驟：一、找特徵，與 SRCNN 一樣只是 kernel size 變小了。二、降維，經過一個 1x1 conv 把特徵圖減少，使得網路速度加快。三、mapping，也有做特徵圖組合的部份。四、Expanding。作者發現如果特徵圖太少的話，upsampling 的效果不太好，所以多加上了個 1x1 conv 增維。五、deconv。FSRCNN 的 upsampling 層使用的是 deconv。

## ESPCN
此篇作者認為如果在高解析的圖片上做 upsampling 會增加計算複雜度，所以提出一個新方法可以直接在低解析的圖 upsampling 至高解析度
![image-20210706145659734](https://i.imgur.com/VFFYWCu.png)

作者提出 sub-pixel convolutional layer 又稱 pixel shuffle，可參考上一篇文章

## VDSR
在介紹 VDSR 之前，先來介紹 ResNet 對 SR 領域的影響，這篇作者提到，其實低解析度與高解析圖片之間的差異度是非常小的，全部一起訓練會學到很多不必要的資訊，如果可以只學習「高解析與低解析之間的差」，那效率一定會提高。因此 VDSR 引入了 ResNet 的觀念，加上了殘差。

![image-20210706151158349](https://i.imgur.com/InBFF4e.png)

VDSR 主要有四個貢獻：一、加深網路的層數到 20 層，為了能提取更多的特徵。二、同時也因為加入了 Residual 的概念，網路不會出現梯度消失/爆炸的問題。三、在每次 conv 之前都會對圖片做 padding 補 0，可以使每次 conv 完的圖片不會慢慢縮小，而是維持原大小，作者有在文後實驗補充 padding 的效果會比較好。四、將不同解析度倍數的圖片放在一起訓練，這樣 model 就可以一次處理不同的解析度問題了。

## DRCN
使用了 RNN 的概念去提取特徵
![image-20210706152238830](https://i.imgur.com/qBcuVIz.png)

作者把網路分為三個部份，提取特徵、特徵組合、upsampling，最大的特包就是特徵組合用一個 RNN 來實作，彼此共享權重，但我個人認為這個就只是神經元固定的好多層的 conv 而已

![image-20210706152501440](https://i.imgur.com/coc8uMP.png)

作者還在 RNN 每一層中加上一個 Recursive-Supervision 為了來解決梯度消失/爆炸的問題，與 Residual 的概念相近。

## DRRN
作者提出了新的概念：局部 vs 全局，概據 Residual 加入的長度來作為劃分：
![image-20210706152939520](https://i.imgur.com/1QJbC3P.png)

VDSR 在各個地方都加上了 Residual 不管是小區域，或是大區域，作者認為這樣殘差的學習可以更全面。

與前面其它的架構做比較：VDSR 是全局殘差學習。DRCN 是全局殘差學習 + 每一個權重的殘差學習。DRRN是多路徑模式的局部殘差學習 + 全局殘差學習 + 每一個權重的殘差學習

## LapSRN
作者總結了一下之前論文所遇到的問題：一、圖片先預先做放大，增加計算時間開銷，而且作者認為 deconv 或是 sub-pixel conv 在學習上的架構都過於簡單，在低解析度到高解析之間的 mapping 效果並不好。二、使用 L2 當做 loss 會有細節平滑化的問題 (smooth)。三、當要 upsampling 成很大的圖片時，例如直接放大 8 倍，效果一定不會很好。因此作者提山 LapSRN ，一個慢慢增加圖片大小的做法。

![image-20210706154107539](https://i.imgur.com/DsfRAju.png)

LapSRN 一次只會放大 2 倍，如果要把圖片放大 8 倍的話，就會經過 3 次的運算。

LapSRN 網路架構分為兩部份：
一是 Feature Extraction Branch，負責先找出「此放大倍率」下的特徵，再經一個 deconv 放大倍率，後接兩個 conv 層，一個用於繼續放大圖片特徵，一個用於計算出不同倍率間的殘差。
二是 Image Reconstruciton Branch，會先把 LR 做 deconv 得到 2 倍放大的圖片，接著與在 Feature Extraction Branch 計算出的殘差數值相加，就會最後結果。

而 LapSRN 的 loss funtion 設計為：

$$
L(\hat{y}; y;\theta) = \frac{1}{N} \sum^N_{i=1}\sum^L_{s=1}\rho((\hat{y}^i_s - x^i_s) - r^i_s)
$$

其中， $\rho$ 叫作 Charbonnier 的懲罰函數 $\sqrt{x^2 + \varepsilon^2}$ ( $L1$ 的變形 )， $\varepsilon$ 大小設為 0.001。$x$ 表示低解析度圖，y 表示高解析度圖，r 表示殘差，s 表示對應的放大級數。N 表示訓練的 batch size 大小，L表示網路一共有多少放大層數。

可以看到這個 loss funtion 在每一個不同放大倍率下對有對應的計算，因此每一級都有一個 loss，訓練時就是要把每一級的 loss 和減少。

## SRDenseNet
因為 DenseNet 是 CVPR 2017 Best Paper，所以就把 DenseNet 拿到 SR 領域來做啦

![image-20210706155926611](https://i.imgur.com/LCajMLV.png)

SRDenseNet 一共分為四個部份：一、經過一個 conv 學到低解析的特徵。二、經過一大堆的 Dense Block 學習到高解析的特徵。三、經過一個 deconv upsampling。四、最後再經過一個 conv 得到最終的影像

這篇論文的作一共做了三種不同的實驗：一、只把 Dense Block 最後一層拿來用。二、把最後一層以及低解析度的那一層拿來用。三、每一個 Dense Block 以及低解析度的都拿來用。經實驗得知效果是 3>2>1，而計算量也是 3>2>1。經過這個實驗可以證明另一個有趣的事情，就是各個不同值置的 Residual 是有互補的關系的，也就是說低解析高解析是可互相提供訊息的

## SRGAN(SRResNet)
使用了 GAN 來做 SR 的問題，而 Generator Network 則是使用叫 SRResNet，基本上就是 ResNet

![image-20210706160800084](https://i.imgur.com/1c2FsEi.png)

使用 GAN 來解決問題，基本上與傳統 GAN 沒什麼大不同，loss function 也就是 content loss + adversarial loss。

adversarial loss 為 GAN 的 Discriminator 的 loss
content loss 為 Generator 的 Loss，可使用 MSE 或是預訓練的 VGG loss

## EDSR
![image-20210706161259395](https://i.imgur.com/nlM5Cz1.png)

EDSR 的最大貢獻是把 Residual 中的 BN 給去掉了，本篇作者認為 CNN 中的分類、偵測是屬於「高層」應用，而 SR 屬於「低層」應用，直接把 Residual 搬過來用是不合適的，作者認為 BN 耗費時間及記憶體，所以如果把 BN 拿掉的話，就可以疊加「更多」層進去了。

並且使用分批訓練，先預訓練低解析度的網路，再把低解析度的網路參數初始化下一階的訓練，實驗證明這樣做的效率好很多，神奇的是效果也提升了。

下一個是同論文的架構 MDSR

![image-20210706162159565](https://i.imgur.com/81kYY7P.png)

在網路前面加上了不同解析度預訓練好的模型來減少不同倍數輸入圖片之間的差異。

## Reference
https://zhuanlan.zhihu.com/p/31664818