---
title: 你所不知道的 Pytorch 大補包(十)：Pytorch 如何實做出 Backpropagation 之什麼是 Backpropagation
mathjax: true
date: 2022-12-29 00:38:57
tags: Pytorch
categories: Pytorch 大補包
---

常常我們初學 pytroch 的時候都一定會看過下面的程式碼：

```python
for epoch in range(1, epochs+1):
  output = model(dataset)
  loss = criterion(output, target)
  
  # wtf
  optimizer = zero_grad()
  loss.backward()
  optimizer.step()
```

好不容易跨出第一步，並剛接觸程式碼的你，一看到這坨鬼東西一定心裡有三個問號…(至少我是這樣啦哈哈。

keywords: Backpropagation
<!--more-->


而大部份網路上的教學都會強調：這個就是 Backpropagation 喔！也不用太了解它，知道在寫程式時記得要加上它就好了！…

更進階一點，你是從學校修神經網路相關的課程，也知道 Backpropagation 背後的數學原理，甚至還用 mathlab python 手刻了一個陽春 Backpropagation，只是當你轉換到 pytorch 上來看到程式碼時，不禁覺得…這程式也太簡潔了吧…，只要一行 `loss.backward()` 就可以了，這真的可靠嗎？

而這篇文章就會從最一開始的脈絡，來慢慢解釋：什麼是 Backpropagation、要怎麼用程式來實作 Backpropagation、pytorch 倒底幫我們做了什麼？不管你是初心者或是小有經驗的開發者，這些底層冷知識可以幫助你加深對 pytorch 的感情喔！

### 什麼是 Backpropagation？

在理解什麼是 Backpropagation 之前，先來複習一下訓練一個神經網路一定要經過的四個步驟：

1. Forward Pass 前傳導
2. Calculate Gradient 計算梯度
3. Backpropagation 後傳導
4. Weight update 權重更新

用一個非常非常簡單的例子來舉例，假設我們要訓練一個可以把輸入資料都都 x2 的網路，並且定義以下參數

```python
import torch

# x 定義為輸入資料 = 2
x = torch.tensor(2, dtype=torch.float32)

# y 定義為目標 Ground Truth 
# y = x*2 = 2*2 = 4
x = torch.tensor(4, dtype=torch.float32)

# w 定義為網路中的一個權重值，初始為 0
w = torch.tensor(0, dtype=torch.float32)
```

把上面的文字及變數定義換成更白話一點的說法就是：我們今天有一個式子 `w * x = y` 要找到一個適合的 w 值，使得 `2x = y`。

畫成樹狀圖可以長成如下：

![未命名绘图.drawio](https://i.imgur.com/Y7vRHHO.png)

當然在現階段我們可以清楚的一看就知道 `w = 2` 就是答案了，只是在一般的深度學習中，x 的多項式可是會達到幾千甚至幾萬的維度，跟本不能用多項式求解的方式來知道答案。那該怎麼辨呢？就使用漸近求解的方式吧！因此才會多一個計算 Loss 的步驟，Loss 可以得知，網路輸出的結果與真實的結果倒底相差多遠，透過這個相差多遠的資訊，可以進一步得知網路是否有在往正確的方向學習。

我們可以把最後的結果套上 Mean Square Error (均方誤差)，就是一個相減後平方的公式：$\mathcal{L}=(\hat y -y)^2$

例如當 `w = 1` 時，我們算出來的 Loss 為 $(1-2)^2=1$，可解讀為我們離正確解答的距離還有 1 (單位)，而隨著 w 的數值越來越接近 2，Loss 也會越來越小，直到趨近於 0。

因為我們在網路中加入 Loss，對應的運算樹狀圖也要修改如下：

![未命名绘图.drawio](https://i.imgur.com/vY3QdQ8.png)

接著我們實際把 xyw 輸入到網路中，並經過一串多項式運算，如同下圖求出 Loss 為 16，這個步驟就是 Forward Pass 前傳導

![未命名绘图.drawio](https://i.imgur.com/ixuxvfW.png)

---

那計算出來的 Loss 是要做什麼的呢？其實這個 Loss 除了看網路訓練的好不好之外，還可以用來計算梯度並更新每個節點上的權重。

什麼是梯度呢？高中數學我們會學到，在一個二維曲線上畫一條切線，就代表它的斜率；如果是在物理上，在 v-t 圖的一個時間點上找切線斜率，則是代表瞬時速度。只是在深度學習中，我們習慣稱為梯度，英文為 Gradient，數學符號為 $\nabla$

那梯度在深度學習中代表的函意又是什麼呢？代表在多維的空間中，某一點的斜率。以三維空間為例子，三維空間就是一個大曲面，這個曲面有凹有凸，就像一個山脈，一個山脈有山頂、有山谷、有平原、有懸崖…，而梯度類比山脈的例子就相當於是當下等高線的坡度

<img src="https://i.imgur.com/aDZmUmo.jpg" alt="image-20220901184131815" style="zoom: 50%;" />

那我們算梯度要做什麼…？還記得前面我們有說過 Loss 的值是要…越小越好對吧？代表網路預測的結果跟真實的結果距離越近，我們要怎麼知道如何修改 w 值才可以使得 Loss 最小？這個問句可以用山脈的例子同等於：我們怎麼走才可以下山？甚至也可以說：我們怎麼走才可以最快的到山下？

答案當然是用滑的阿，有爬過山的都知道上山易下山難，下山時多希望自己有個鋼鐵屁屁可以一口氣滑下山 XD，而深度學習也是利用一模一樣的方法：了解哪裡坡度/梯度最大，就可以快速的滑下山，取得最小的 Loss，同時也取得預測效果最準的結果

---

而要求得梯度只有一種辨法：微分，更詳細的說是偏微分，我們最想要了解最後網路的**Loss 與變量 w 之間的梯度關系**，因此只要求得 loss 對 w 的偏微分，就可以知道 w 要怎麼調整效果會最好了，公式如下：
$$
\nabla g = \frac{d\,\mathrm{loss}}{dw}
$$
但仔細看會發現…上面這個式跟本算不出來阿，為什麼呢？如果我們把 loss 解壓縮的話：
$$
\nabla g=\frac{d\,\mathrm{loss}}{dw}=\frac{d(\hat y-y)}{dw}
$$
loss 中間完全沒有 w 變量阿，倒底要怎麼對 w 做偏微分呢？我們可以先停下腳步來別想要一簇登天直接求得對 w 做偏微分，我們可以先算出 Local Gradient，也就是每一個權重先對自己的變量求梯度 (loss 先對 s、s 先對 $\hat y$、$\hat y$ 先對 w)：
$$
\frac{d\,\mathrm{loss}}{ds} = \frac{ds^2}{ds}=2s\\
\frac{ds}{d\hat y}=\frac{d(\hat y-y)}{d\hat y}=1\\
\frac{d\hat y}{dw}=\frac{d(x\cdot w)}{dw}=x
$$
更詳細如下圖：

![test](https://i.imgur.com/tOz7cAL.png)

再仔細看看上面的 Local Gradient，咦…好像怎麼有規律？！這不就是傳說中的 chain rule 嗎？
$$
\nabla g=\frac{d\,\mathrm{loss}}{dw}=\frac{d\,\mathrm{loss}}{ds}\frac{ds}{d\hat y}\frac{d\hat y}{dw}
$$
也就是說當我們在做 Local Gradient 的時候，其實就是在幫我們最感興趣的**loss 對 w 的偏微分**在計算它的 chain rule，而這個透過 chain rule 一層一層慢慢的找到值的方式，就是 Backpropagation
$$
\nabla g=\frac{d\,\mathrm{loss}}{dw}=\frac{d\,\mathrm{loss}}{ds}\frac{ds}{d\hat y}\frac{d\hat y}{dw}=2s\cdot 1 \cdot x=-16
$$
![test](https://i.imgur.com/bh1nSlm.png)

接著我們再將算出來 Backpropagation 的值，簡單的利用以下的公式去更新每一個權重，其中 $\nabla g$ 為 Backpropagation 的結果、$\eta$ 為 learning rate 控制大小用：
$$
w_{t+1}=w_t-\eta\nabla g
$$

### 要怎麼用程式來實作 Backpropagation？

以下的例子會回歸最原本的初心，不用高級的 pytorch 工具，而是使用 numpy 來達成以上四個基本操作

```python
import numpy as np

# 定義：輸入 x=2，目標 y=4，變量 w=0
x = np.array([2], dtype=np.float32)
y = np.array([4], dtype=np.float32)
w = 0.0

# 定義網路前傳導的式子
def forward(x):
    return x * w

# 使用 MSE 均方差來做為 Loss
def loss(y_hat, y):
    return ((y_hat - y)**2).mean()

# 計算 loss 對變數 w 的偏微分
# 這是是直接把式子展開，直接計算偏微分 (沒有用到 chain rule 的概念)
# dloss/dw = d(w*x - y)^2/dw = 2x (w*x - y)
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat-y).mean()

epochs = 100
lr = 0.01

# 開始訓練
print("Start Training...")
for epoch in range(1, epochs+1):
    # (1) Foward Pass 前傳導
    y_hat = forward(x)
    # (1.5) 計算 Loss
    l = loss(y_hat, y)

    # (2, 3) 計算 Local Gradient 以及 Backpropagation
    dw = gradient(x, y, y_hat)

    # (4) 更新權重 w
    w -= lr * dw

    print(f"epoch: {epoch}, loss: {l:.8f}, w: {w:.3f}, dw: {dw:.3f}")
```

```
# 印出來的結果
Start Training...
epoch: 1, loss: 16.00000000, w: 0.160, dw: -16.000
epoch: 2, loss: 13.54240036, w: 0.307, dw: -14.720
epoch: 3, loss: 11.46228790, w: 0.443, dw: -13.542
epoch: 4, loss: 9.70168018, w: 0.567, dw: -12.459
epoch: 5, loss: 8.21150303, w: 0.682, dw: -11.462
epoch: 6, loss: 6.95021534, w: 0.787, dw: -10.545
epoch: 7, loss: 5.88266134, w: 0.884, dw: -9.702
epoch: 8, loss: 4.97908545, w: 0.974, dw: -8.926
epoch: 9, loss: 4.21429777, w: 1.056, dw: -8.212
epoch: 10, loss: 3.56698155, w: 1.131, dw: -7.555  <- w=1.131
```

從印出來的結果可以看到，在 epoch=1 的時候，dw 也就是我們剛剛算出來的值 -16 是完全正確的，接著 dw 值會往 0 靠近，而 Loss 的數值也慢慢降低，以及 w 的值，從 0 慢慢的往答案 2 靠近

但可能會覺得這個訓練效果也太不好了吧…搞了這麼多數學的東西，結果訓練出來的 w 竟然離 2 還很遠！沒關系！這個網路還有很多可以優化的地方：像是增加 epoch 數量、或是增加資料集，都可以使網路效果變更好喔！

```python
# 定義：輸入 x，目標 y，新增資料集至 4 組
x = np.array([2, 4, 6, 8], dtype=np.float32)
y = np.array([4, 8, 12, 16], dtype=np.float32)
...
# 因為資料不只一組，由於 Gradient 不能是一個 Vector (向量) 必需要是一個 Scalar (純量)，所以要取 mean 平均
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat-y).mean()  <- 這裡 mean 的作用
```

```
Start Training...
epoch: 1, loss: 30.00000000, w: 1.200, dw: -120.000
epoch: 2, loss: 4.79999924, w: 1.680, dw: -48.000
epoch: 3, loss: 0.76800019, w: 1.872, dw: -19.200
epoch: 4, loss: 0.12288000, w: 1.949, dw: -7.680
epoch: 5, loss: 0.01966083, w: 1.980, dw: -3.072
epoch: 6, loss: 0.00314570, w: 1.992, dw: -1.229
epoch: 7, loss: 0.00050332, w: 1.997, dw: -0.492
epoch: 8, loss: 0.00008053, w: 1.999, dw: -0.197
epoch: 9, loss: 0.00001288, w: 1.999, dw: -0.079
epoch: 10, loss: 0.00000206, w: 2.000, dw: -0.031  <- YA, w=2 了！
```