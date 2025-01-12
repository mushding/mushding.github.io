---
title: 你所不知道的 Pytorch 大補包(六)：訓練小技巧 Gradient accumulation
mathjax: false
date: 2022-12-29 00:26:47
tags: Pytorch
categories: Pytorch 大補包
---

你是不是常常覺得 GPU 顯存不夠用？是不是覺得自己太窮買不起好的顯卡、也租不起好的機台？覺得常常因為東卡西卡關就放棄深度學習？

沒關系！接下來有幾招可以在預算不太足的情況下，還是可以讓訓練跑得起來！

keywords: Gradient accumulation
<!--more-->

## Gradient accumulation

第一招叫做 Gradient accumulation

這是利用 pytorch 每一次在 backpropagation 前都會把梯度清零

正常一個訓練部份程式會這樣寫：

```python
for i, (image, label) in enumerate(train_loader):
    # 1. input output
    pred = model(image)
    loss = criterion(pred, label)

    # 2. backward
    optimizer.zero_grad()   # 把梯度清零
    loss.backward()					# backpropagation 計算當前的梯度
    optimizer.step()        # 拫據梯度更新網路參數
```

而 Gradient accumulation 會這麼寫

```python
for i,(image, label) in enumerate(train_loader):
    # 1. input output
    pred = model(image)
    loss = criterion(pred, label)

    # 2.1 loss 要除以累積的總步數，正規化 loss 的值
    loss = loss / accumulation_steps  
 
    # 2.2 計算梯度的值
    loss.backward()

    # 3. 當累積的步數到一定的程度後，梯度中的值也會不斷累加，才會更新網路的參數
    if (i+1) % accumulation_steps == 0:
        # optimizer the net
        optimizer.step()        # 更新網路參數
        optimizer.zero_grad()   # 清空以前的梯度
```

這樣子這的理由是，每次的 epoch 不把梯度清零，而是用累加到一定的程度後才會更新網路參數值

目的是要在不增加 RAM 的條件下，變向增加 batch size (窮人做法阿 QQ)

有兩個要注意的小地方：

1. learning rate

   因為變每兩步為一個單位計算梯度了，而 learning rate 的設定依舊是以一步為一個單位

   所以要適當的調大一些些

2. batch normalization

   原理同上

   BN 的分母為全部 Batch szie 的值，但因單位的改變，而 BN 卻沒跟上

   所以 BN 的分母的值並非全部的 batch size (但根據下面文章好像又沒差多少…我不是很清楚 XD)

   但可確認的是，效果一定比單純的增加 batch size 來的差一些些

   或是可以調低 BN 的 momentum 參數