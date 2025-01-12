---
title: 'MAE: Masked Autoencoders Are Scalable Vision Learners - 模仿 BERT 且更簡單的自監督學式'
mathjax: true
date: 2022-05-08 10:23:21
tags: 

  - Self-supervised Learning
  - Vision Transformer
categories: 電腦視覺整理

---

繼上一篇 BEiT 後，2021 11月 FAIR 也提出了一個基於 BERT 改造且應用在電腦視覺上的自監督式學習，其最核心的想法，就是建構出一個更「直覺」「簡單」的模型。模型取名叫做 masked autoencoders (MAE)，相較於上一篇 BEiT 效果上差不多，但是整體的訓練流程卻相對簡單許多。

[https://arxiv.org/pdf/2111.06377.pdf](https://arxiv.org/pdf/2111.06377.pdf)

keywords: Self-supervised Learning、BERT、MAE
<!--more-->

## Abstract

作者提出一個 scalable(可自由放大縮小) 的網路架構 MAE，想法為：**利用 mask 遮罩隨機的把 patch 給遮掉，經過網路後再重建回原影像**。其核心的實做辨法有二：

1. 提出一個「不對稱」的 encoder-decoder 架構
2. 被 mask 遮蓋掉的 patch 高達 75%

以上兩個做法不僅在「效率 efficient」上減少相當多，同時在「效能 accuracy」上也是增加的。作者證明使用了最陽春的 ViT-H 做為 backbone，且只在 ImageNet-1K 上訓練，就可以得到 (87.8% accuracy) 的效果。說明 MAE 提取特徵之厲害的地方。

![Image](https://i.imgur.com/plhkvt8.png)

## Introduction

作者說：BERT (如果不清楚可以看我上一篇文章的介紹) 的觀念很直覺，藉由移除掉訓練資料中的一部份再把它預測回來來訓練網路，這種方法因為「移除」資料的原因，因此它的**訓練資料集**及**模型參數量**也是異常的大。不過以上兩個缺點卻完全沒有影響到 BERT 成功的亮光，BERT 在 NLP 界大放異彩，對後續自監督式學習起到了重要作用。

也因此在 CV 界的大家同時開始在想：要是…我們把 BERT 移到影像上呢？要是…我們今天 mask 蓋住的不是字詞而是一個 patch 呢？因此這篇論文就是在做這件事情：把 BERT 應用在影像上面

而作者開始研究的第一步不是直接想一個網路出來，而是先問自己：如果我們要設 mask 蓋住東西的話，蓋住一個字詞跟蓋住一個圖片的差別倒底在哪裡呢？`what makes masked autoencoding different between vision and language?`，作者提出了以下三點回覆：

1. 直接把影像放在 BERT 上的第一個困難就是：資料維度的不同，一個是二維影像、一個是一維序列，而且 BERT 中還有 positional enbedding 這些 CV 中都沒有的特色，是要怎麼融合在一起呢？多虧了 ViT 論文的提出，我們已經知道直接把影像丟到 Transformer 中訓練不僅是一個可行的方法，同時效果可望還能突破傳統 CNN 架構，所以這已經不是一個困難的點了
2. 資料複雜度非常的不同。對於一個句字來說，裡面包含了非常非常多的資訊：文法、字詞、上下文關系，如果把其中一個字詞挖掉可能會影響到整句話的意思，對於人類來說因為有著很多的「先備知識」所以可能會覺得很簡單，但對機器來說並非如此；那如果是一張影像呢？因為影像有 heavy spatial redundancy 的特色，多一少一個 pixel 對影像的影響不大 (看看 stride pooling 的影響，其實很小)，所以如果跟原本 BERT 一樣只挖 15% 是不夠的，網路會因為訓練難度不夠而效果不好。因此作者提出挖掉非常高 **75%** 的 patch 來解決這項問題
3. 最後是 Decoder 的複雜度。在原本 BERT 中最後 mask 的部份會做分類任務，因為「詞」這個本身已經有很多函意在裡面了，所以只使用了簡簡單單的一層全連接層就搞定了；在影像上為了要 by pixel 的重建回影像，在以往分割的經驗中，我們會需要多層的卷積及 upsampling 才能提取其中的特徵。所以作者有別於 BERT 的一層全連接，設計了相對複雜的 Decoder

綜合以上三點作者提出的 MAE 有著以下兩個特色：利用高達 75% 的 mask 來訓練網路、以及「不對稱」的 Encoder-Decoder 架構

MAE 蓋掉 75% 的 patch 重建回原影像的結果，發現網路對圖片的理解非常可怕，蓋掉一大堆還大概知道原圖長什麼樣…

![image-20220509110908182](https://i.imgur.com/A0wboRG.png)

什麼叫做「不對稱」呢？在 MAE 中，Encoder 是「短而厚」，輸入的 patch 不長 (75% 被蓋住了)，但是網路較深；而 Decoder 是「長而薄」輸入全部的 patch 但是網路較淺，稍後網路架構會有更深入的說明。但可知道的是作者藉著這種操作大量減少了運量，作者在論文中稱：與正常的 Encoder-Decoder 相比減少了近 3 倍的運算量，且可以把省下來的運算量拿去利用給 Encoder 的編碼，更加強網路的效果。

作者利用 ViT-L/16，ViT-H/16 兩個模型，僅在 ImageNet-1K 上面做預訓練，最後再 fine-fune 就可得到 87.8% 的正確率。在其它模型下要取得這種正確率，網路的參數量可要非常大才行。(ViT 可用了 JFT-300M 才有這個效果)

## Related Work

其中在 Autoencoder 的地方作者做了一個有趣的比較，作者說：MAE 也算是一種 AutoEncoder，有著三大要素：Encoder、Decoder、以及中間的 latent space。MAE 的 mask 作法尤其更像 2008 年的 [Denoising AutoEncoder](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)，同樣都是在圖片上加入雜訊，同樣都有 Encoder-Decoder 架構，作者認為 MAE 「架構」算是某種程度上是 Denoising Autoencoder 的特例，但整體訓練「思維」及訓練「方法」有著相當大的不同

另外還提到了 Self-supervised Learning。作者說最近自監督式學習在電腦視覺上越來越流行，目前有兩種支派、一是 Contrastive Learning 代表做有 MoCo、SimCLR 等等，而 MAE 與 BEiT 是屬於修改 BERT 類，作者認為最大的不同在於資料預處理上，Contrastive Learning 非常依賴資料擴增。

## Approach

網路流程如同 Autoencoder 一樣，會經過 Encoder -> latent space -> Decoder，MAE 特別的地方是：一、進 Encoder 前會用 mask 隨機的把影像上一部分 patch 給遮掉；二、也因為被遮掉後輸入影像 patch 數量改變，所以是一個「非對稱」的 Autoencoder

### Masking

以下架構與 ViT 一樣，首先輸入影像做 Patch embedding 分成許多 16x16 的 patches，接著用一個隨機分佈亂數，隨機的取其中 25% 的 patches，也可反過來理解為把 75% 的 patch **移除**掉。等等**移除**…那 mask 呢？在 BERT 中 mask 會是一個可學習的變量乀，在 MAE 中又代表什麼呢？其實 MAE 中的做法與 BERT 不一樣，**MAE 不會把帶有 mask 的 patch 放進 transformer 中訓練**，也可換句話說 MAE 中並沒有 mask 這個概念更像是一個被移除 patch 的標籤而已，在移除掉「大量」的 patch 下，進 transformer 訓練的向量數少非常多。而下面 Encoder 會講更清楚

![image-20220509141211862](https://i.imgur.com/bnfzznr.png)

### MAE Encoder

Encoder 的部份也與 ViT 一模一樣，是原汁原味的 transformer 架構。上面有提到**只有不帶 mask 的 patch 才會進 Encoder**，且在「大量 (75%)」的 patch 移除下，計算量也更是直接少了 75%

那為什麼要移除掉這麼多 patch 呢？理由是影像資訊非常的 redundancy 「冗」XD，加一少一個 pixel 基本對整體理解並未差太多，如果我們今天照著 BERT 一樣只移除掉 15% 的輸入，那網路可能強度不強學不太到什麼有用的資訊，甚至網路可能只是在學內差而已 (單純用內差也可以解決這個問題)，所以才會提出移掉 75% 的想法。那為什麼是 75% 呢，後面作者有做詳細的實驗，不過我們可以先來看上面的圖，可以發現當移除達 95% 時，好像資料少太多了，重建的圖開始與輸入相差甚遠，而 75% 正剛剛好，不多也不少，重建的影像品質也是裡面最好的一組

再來因為在輸入序列長度上變「短」了，多出來的計算量正好可以補在「厚度」的地方，我們可以把使用較大的 Transformer 架構 (ViT-L, ViT-H) 來訓練，但參數量不會上升太多，因此說 MAE 的 Encoder 「短而厚」。這個特性後續對於網路的 scaling up 放大實現起來非常容易

### MAE Decoder

Decoder 就回歸正常操作了，輸入是「全部」的 patch (mask + 進 Encoder 的部份)，網路也是做 transformer 運算。MAE Decoder 的 mask 同樣是一個可學習向量，透過在 Decoder 與其它 patch 計算相關性，最後得出一個特徵向量表示

在 Decoder 的地方也對每個 mask 加上 position embedding 不然重建網路時會不知道彼此的絕對關系

如同 BERT 一樣，MAE 的 Decoder 只在訓練 (pre-train) 時存在，在測試 (fine-tune) 的時候只會使用到 Encoder 的特徵層而已，所以 Decoder 的層數就可以不用設計像 Encoder 那麼深，論文中提到計算量大約是 Encoder 的 <10%。因此說 MAE 的 Decoder「長而淺」

### Reconstruction target

MAE 重建影像的評估是建立在：每個有 mask 的 patch 與原影像之間 pixel 級別上的關系。Decoder 最後輸出的特徵向量，會經過一個全連接層維度轉換成 256 = 16x16，再把這 256 維透過位置訊息重建回 16x16 的影像，這個 16x16 的向量也不再做什麼分類任務了，它最後直接就是表示成一張影像。最後與原圖 pixel 做 mean squared error (MSE) 得到這個 patch loss，加總所有 masked patches 得到網路整體的 loss

## Experiments

作者全部實驗都是 fine-tune 過後的 (先用一大堆無標記 per-train，再用少量有標記 fine-tune) 作為實驗依據，又分別有兩種做法，分別是 end-to-end fine-tuning (全部 encoder 參數都可以修改) 以及 linear probing (固定前 N - 1 層參數，只修改最後第 N 層的參數)，代號分別是 ft、lin。理所當然 fine-tune 因為動到的參數多計算量大所以效果一定比 linear probing 好

### ImageNet 橫向比較

同樣都是使用 ViT-L、ImageNet-1K，左邊是 ViT 原始效果，中間是作者在原 ViT 超參數中加了一些 regularization 規則項，右邊是 MAE

![image-20220509152108923](https://i.imgur.com/OkYUMZF.png)

發現：一、ViT-L 經調教過後還是可以有比較好的表現的，二、僅管如此 MAE 效果還是比較好。且在 fine-tune 上所需要的計算成本非常小 (50 epochs vs 200 epochs)

### Masking ratio 比例

作者同時比較了 ft 與 lin 在不同 masking ratio 下的表現，發現不管是哪一個 fine-tune 做法，橫軸比例縱軸正確率的圖表下都成一個倒 V 型，太多太少比較效果都不好，中間值落到 75% 時效果最好

![image-20220509152557788](https://i.imgur.com/Nu566Mv.png)

### Decoder 的一些設計實驗

因為 Encoder 直接抄 ViT 所以沒什麼好說的 XD，以下簡單看一下 Decoder 的 ablation 實驗

![image-20220509153109437](https://i.imgur.com/llbiPES.png)



圖 a、發現 Decoder 深度不用深，就有不錯的效果了 (在 ft 更明顯、用一層也行)。

圖 b、Decoder 在特徵維上的大小實驗，發現不用維度也不用大，比起 encoder 的 1024 小了不少。

圖 c、encoder 要不要放入 mask。發現：一、放了效果不好 (84.9 -> 84.2, 73.5 -> 59.6)，二、運算量還多了 3.3 倍。那…幹麻放它進去 XD

圖 d、重建的依據。pixel 代表 by pixel 的 MSE (一個一個算)，發現在做 loss 前做一個 patch 內的 normalization 會使效果更穩定 (合理)。同時與 BEiT 使用的 dVAE 做比較，發現效果其實差不多，但是在觀念和算法複雜度上差很多，那既然如此為什麼不選用簡單直覺的做法呢？

圖 e、MAE 對資料擴增的敏感度。當然有做效果一定會比較好，但是提升不多，可理解為 MAE 對資料擴增不敏感 (我覺得作者刻意提這個是為了與 Contrastive Learning 比較)

圖 f、挖 mask 的方法。隨機挖效果最好，一塊一起挖會不平均 (沒辨法保證重要特徵集中在邊上)，固定挖法 (太簡單了，網路跑去偷吃步去了，不知道學到了啥)

![image-20220509154309795](https://i.imgur.com/TDLLFxo.png)

### Training schedule

發現 MAE 的方法不容易使網路 overfitting，epoch 都已經調到 1600 了，測試效果還在上升 (當然前提是你的 $$ 足夠你這樣做 XD)

![image-20220509154618345](https://i.imgur.com/vqmDapZ.png)

### SOTA 表

分類：

結論：與 BEiT 差不多，但架構簡單很多。與 Contrastive Learning 還有得比，事後才知道誰是大贏家

![image-20220509154835970](https://i.imgur.com/vURKilX.png)

偵測：

![image-20220509155018121](https://i.imgur.com/fj9yAtb.png)

分割：

結論同偵測，在更複雜的影像任務上 MAE 開始展示了它的強悍，提升許多百分點

![image-20220509155029106](https://i.imgur.com/4oi5W7y.png)

## 結論

MAE 與 BEiT 相比：效果差不多，但架構更簡單，更直覺。好像大家都比較喜歡直覺簡單的網路架構呢哈哈。MAE 同樣是學 BERT 來實作，又把 CV 往自監督學習推了一步，這篇感覺可以殿定很好的基礎 (愷明大神的加持？！)，希望後續有更多類似論文的提出 (會不會下一個換 GPT XD)

## Reference

[MAE 论文逐段精读【论文精读】(中文超極詳讀，還帶了很多寫論文的私貨，大推 XD)](https://www.youtube.com/watch?v=mYlX2dpdHHM&ab_channel=MuLi)

[(AI Coffee Break) Masked Autoencoders Are Scalable Vision Learners - Paper explained and animated!](https://www.youtube.com/watch?v=Dp6iICL2dVI&ab_channel=AICoffeeBreakwithLetitia)



