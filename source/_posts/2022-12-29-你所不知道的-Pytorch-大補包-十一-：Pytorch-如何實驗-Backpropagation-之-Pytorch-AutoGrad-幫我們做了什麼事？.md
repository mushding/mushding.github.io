---
title: >-
  你所不知道的 Pytorch 大補包(十一)：Pytorch 如何實驗 Backpropagation 之 Pytorch AutoGrad
  幫我們做了什麼事？
mathjax: true
date: 2022-12-29 00:41:11
tags: Pytorch
categories: Pytorch 大補包
---

本文接續著上一篇 [你所不知道的 Pytorch 大補包(十)：Pytorch 如何實做出 Backpropagation 之什麼是 Backpropagation] 繼續更深入的了解 Pytorch 的底層

keywords: AutoGrad
<!--more-->

## pytorch AutoGrad 幫我們做了什麼事？

以上我們成功的用 numpy 手刻了一個超~簡單的神經網路出來，並且訓練它，還取得了 100% 正確率的成果，但剛剛 Foward 的函式很簡單，簡單到它的梯度甚至不用到 chain rule 就算得出來，如果今天是一個 100 層深的網路，那我們的算式就會變得超長，跟本沒有辨法像剛剛直接用一條算式表達出來

這個時候幫我們實作 Backpropagation 的 pytorch 的派上用場了，pytorch 使用 AutoGrad 自動的幫我們把所有梯度都算出來，並且也做完 Backpropagation，在解釋一切程式碼之前，我們再來回頭看看用 pytorch 寫出來的程式會多麼的簡潔

```python
import torch
import torch.nn as nn

# 定義：輸入 x=2，目標 y=4，變量 w=0
x = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
y = torch.tensor([4, 8, 12, 16], dtype=torch.float32)

# 這個後面有 requires_grad 等等會介紹
w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

epochs = 100
lr = 0.01

def foward(x):
    return w * x

# 定義使用 SGD 作用最佳化演算法
optimizer = torch.optim.SGD([w], lr=lr)
# 定義 MSE Loss 
loss = nn.MSELoss()

for epoch in range(1, epochs+1):
    # (1) Foward Pass 前傳導
    y_hat = foward(x)
    # (1.5) 計算 Loss
    l = loss(y_hat, y)

    # (2, 3) 計算 Local Gradient 以及 Backpropagation
    l.backward()

    # (4) 更新權重 w
    optimizer.step()
    # (4.1) 淨空 dw (Gradient) 值
    optimizer.zero_grad()
    
    print(f"epoch: {epoch+1}, w: {w:.3f}, loss: {l:.8f}, w.grad: {w.grad}")
```

### 什麼是 requires_grad？

一般在建立一個新的 tensor 時，我們會使用 `torch.tensor` 來達成，但是如果 tensor 想要實作自動計算梯度的話，我們必需在後面加一個參數 requires_grad

```python
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# tensor([1., 2., 3., 4.])

w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
# tensor(0., requires_grad=True)    <- 這裡多一個屬性
```

打開這個 requires_grad 屬性後，pytorch 會幫我們打開更多的屬性，而這些屬性是只有在做 Backpropagation 時才會用到的，所以平常把它關起來減少記憶體的消耗。還記得前面有提到訓練網路的四大步驟嗎？等等會依照這四個步驟的順序來介紹

### Foward Pass 前傳導

前傳導其實就是由一堆運算式所組成，輸入一個值，經過這個複雜的運算式後得到一個結果就稱為 Foward Pass 前傳導，那如果我們在前傳導的式子中加入 required_grad 會發生什麼事情呢？以下用兩個程式來對比

```python
a = torch.tensor(2.0)
b = torch.tensor(3.0)

c = a * b
```

![PyTorch Autograd-A 1.drawio](https://i.imgur.com/NuB11Wn.png)

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)

# Foward Pass
c = a * b
```

![PyTorch Autograd-A 2.drawio](https://i.imgur.com/SvpuwfT.png)

可以由上圖分析出幾個重點：

1. 打開 requires_grad 後，tensor 的所有運算操作都會畫成一個 Graph
2. 一個運算當中只要有一個變量 requires_grad=True，未來運算所新增的 tensor 一樣會是 requires_grad=True
3. 有三個新的屬性：grad、grad_fn、is_leaf

* grad 值在前傳導時為 None，要等到做 Backpropagation 時才會把值填上去

* grad_fn 是在前傳導時 pytorch 自動幫我們加上去的，意思是「對應運算符微分後的算式」pytorch 提供了一大堆的 grad_fn 以應付各種微分運算，加速 Backpropagation 的流程
* is_leaf 為了要表示這個權重節點是不是在葉節點上，為什麼這個這麼重要呢？因為 **pytorch 只會在葉結點上儲存 grad 資訊**，目的是為了保留記憶體

### Backpropagation

前傳導完，記錄了許多數值後，接著就利用這個數值來做 Backpropagation，程式以及對應的 Graph 如下：

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
 
# Forward Pass 
c = a * b
 
# Backpropagation
c.backward()
```

![PyTorch Autograd-A 4.drawio](https://i.imgur.com/YlfmW2o.png)

可以看到，簡簡單單的一行 `c.backward()` pytorch 竟然幫我們做了這麼多事情…。宏觀上來看這一行指令幫我們算出來 a 的 grad 值 3，微觀上來看這一行指令幫我們畫了超多的圖…同時也吃掉了不少記憶體

首先當呼叫 `c.backward()` 時，pytroch 會先去尋找 grad_fn，接著根據 grad_fn 裡面的微分運算計算 grad，然後放進 AccumulateGrad，累計不同次運算的 grad，最後再放到對應節點權重的 grad 屬性中

以數字的例子為：

```
假設一開始初始值為 1.0 ->
找到 grad_fn 為 MulBackward ->
計算 dc/da 的偏微分 (dc/db 因為 requires_grad=False 所以不參與計算) ->
dc/da = d(a*b)/da = b = 3.0 ->
用一個累計暫存器存起來 ->
放到 a.grad 中
```

再用一個更複雜的例子來舉例：

```python
# 假設 pytorch 的運算變成這個樣子：
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
 
# Foward Pass 
c = a * b
 
# 再定義一個 d
d = torch.tensor(4.0, requires_grad=True)

# 再多一個 Foward Pass
e = c * d

# Backpropagation
e.backward()
```



![PyTorch Autograd-Simple 5.drawio](https://i.imgur.com/2DCyMTA.png)

覺得圖片變太複雜嗎 XD，沒關系我們一起從最下面慢慢算上去：

```
假設一開始初始值為 1.0 ->
找到 grad_fn 為 MulBackward ->
計算 de/dc、de/dd 的偏微分 ->
de/dc = d(c*d)/dc = d = 6.0 ->
de/dd = d(c*d)/dd = c = 4.0 ->

de/dc 因為 c 不是葉結點，所以不用寫回 c.grad，直接把值傳給 c.grad_fn，繼續做下一個偏微分 ->
de/dd 因為 d 是葉結點，用一個累計暫存器存起來，再把結果寫回 d.grad ->

計算 dc/da 的偏微分 (dc/db 因為 requires_grad=False 所以不參與計算) ->
dc/da = d(a*b)/da = b = 3.0 ->
再乘上傳進來的 4，4*3 = 12
用一個累計暫存器存起來 ->
放到 a.grad 中
```

我們可以印印看結果是不是正如我們所計算的？

```python
# 假設 pytorch 的運算變成這個樣子：
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
 
# Foward Pass 
c = a * b
 
# 再定義一個 d
d = torch.tensor(4.0, requires_grad=True)

# 再多一個 Foward Pass
e = c * d

# Backpropagation
e.backward()

print(a.grad)
print(c.grad)
print(d.grad)
print(e.grad)
```

```
a.grad -> 12.0
/opt/conda/lib/python3.7/site-packages/torch/_tensor.py:1083: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at  /opt/conda/conda-bld/pytorch_1656352464346/work/build/aten/src/ATen/core/TensorBody.h:477.)
  return self._grad
c.grad -> None
d.grad -> 6.0
e.grad -> None
```

咦…怎麼跟想像中的答案不一樣…，`a.grad` `d.grad` 都是對的，`c.grad` `e.grad` 發生了什麼事…？其實剛剛也有提到，在 pytorch 中，**只有葉結點 (is_leaf=True) 才會把 .grad 存起來**，目的是為了節省不必要的記憶體，而且在圖中也可看到，資料流動 flow 如果不是指向葉結點，會直接把值放到下一個 grad_fn 中，不會有 AccumulateGrad 把值存到 .grad 中，因此 pytorch 才會跳提醒說這個操作不合理，並且回傳 None

那如果我們真的真的想要得到不是葉結點的 .grad 值呢？那我們就要在**前傳導的時候先把它註冊下來**，使用 `retain_grad()` 這個函式：

```python
import torch

# 假設 pytorch 的運算變成這個樣子：
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)

# Foward Pass 
c = a * b
# 用 .retain_grad() 來註冊，告訴 pytorch 要把這個值存起來
c.retain_grad()

# 再定義一個 d
d = torch.tensor(4.0, requires_grad=True)

# 再多一個 Foward Pass
e = c * d
# 用 .retain_grad() 來註冊，告訴 pytorch 要把這個值存起來
e.retain_grad()

# Backpropagation
e.backward()

print(f"a.grad -> {a.grad}")
print(f"c.grad -> {c.grad}")
print(f"d.grad -> {d.grad}")
print(f"e.grad -> {e.grad}")
```

詳細 `.retain_grad()` 的說明可以看下面這個影片，裡面很詳細的介紹為什麼這樣就可以，以及一個新概念：hook 的用法：[https://www.youtube.com/watch?v=syLFCVYua6Q](https://www.youtube.com/watch?v=syLFCVYua6Q)

另外也可以發現，在圖中有一個 AccumulateGrad 的方塊，這是用來儲存每一次 Backpropagation 的結果，並把新算出來的 grad 與之前的相加存到 .grad 中間，也就是說它不會自動淨空！

所以通常在程式中我們會手動淨空 .grad 值，以確保每一次訓練時 .grad 都是最新的

```python
for epoch in range(1, epochs+1):
    # (1) Foward Pass 前傳導
    y_hat = foward(x)
    # (1.5) 計算 Loss
    l = loss(y_hat, y)

    # (2, 3) 計算 Local Gradient 以及 Backpropagation
    l.backward()

    # (4) 更新權重 w
    optimizer.step()
    # (4.1) 淨空 dw (Gradient) 值    <- 如果不清空 .grad 會累加！
    optimizer.zero_grad()
```

至於為什麼要這樣設計，因為如果我們使用的設備記憶體不足，沒辨法一次結太多資料訓練，我們就可以使用 Gradient accumulation 的技巧，改成每訓練兩次更新一次參數，變向放大 Batch size

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

### 要怎麼去掉 requires_grad？

從上面的圖來看，只要我們在 tensor 中加入 require_grad 參數，pytorch 就會一直記錄追縱未來所有的運算，一直更新那一張大圖，記憶體開銷非常可觀，那我們要怎麼樣把圖中藍藍的那一堆 Backpropagation 專用的圖給去掉呢？一共有三個做法：

```python
# (1) x.requires_grad_(False)
# 這個程式可以 inplace 把 x 的 requires_grad 拿掉
# 在 pytorch 中，所有 xxx_ <- 這個底線的意思代表 inplace 操作的意思，不會回傳任何值

x = torch.tensor(3.0, requires_grad=True)
print(x)
# tensor(3., requires_grad=True)

x.requires_grad_(False)
print(x)
# tensor(3.)
```

```python
# (2) x.detach()
# 這個程式一樣可以把 .requires_grad 去掉
# 但是它會建新一個新的且不帶 requires_grad 的 tensor，並回傳

x = torch.tensor(3.0, requires_grad=True)
print(x)
# tensor(3., requires_grad=True)

y = x.detach()
print(y)
# tensor(3.)
```

```python
# (3) torch.no_grad()
# 會搭配 with 一起使用，也是最常使用的一個方法
# 在 with 的縮排範圍內，任何 tensor 都不帶 .requires_grad

x = torch.tensor(3.0, requires_grad=True)
print(x)
# tensor(3., requires_grad=True)

with torch.no_grad():
  y = x + 2
	print(y)
# tensor(5.)
```

最後一個 `torch.no_grad()` 最常看到，通常會在驗證、測試的程式碼中會出現，理由有兩個。

* 一、結省記憶體。有時候 nvidia 會噴記憶體不夠，不一定是 Batch size 設太大的問題，也有可能是驗證、測試時忘記寫到 `torch.no_grad()` 把那一大堆 Backpropagation 的圖都載入了
* 二、不會更新參數。因為 `torch.no_grad()` 把所有 Backpropagation 剛掉了，排除了所有會更新到參數的因素，因此可以放心的驗證、測試，而不用擔心動到網路的參數

以上就是全部的內容了！希望看完這篇文章可以更了解 pytorch 倒底背後幫我們做了什麼事情！

### Reference

本篇文章大量參考了以下兩個 youtube：

[(系列教學影片) PyTorch Tutorial 03 - Gradient Calculation With Autograd](https://www.youtube.com/watch?v=DbeIqrwb_dE)

[(系列教學影片) PyTorch Tutorial 04 - Backpropagation - Theory With Example](https://www.youtube.com/watch?v=3Kb0QS6z7WA)

[(系列教學影片) PyTorch Tutorial 05 - Gradient Descent with Autograd and Backpropagation](https://www.youtube.com/watch?v=E-I2DNVzQLg)

[PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE)

官方 Document 永遠是你最好的朋友

[A GENTLE INTRODUCTION TO `TORCH.AUTOGRAD`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

[(iT邦幫忙) Day 2 動態計算圖：PyTorch's autograd (裡面有提到 in-place 的問題)](https://ithelp.ithome.com.tw/articles/10216440)

[(Backpropagation 參考文章) 神经网络的传播（权重更新）](https://blog.csdn.net/weixin_41417982/article/details/81393917)

[(為什麼我看不到 .grad？) Why cant I see .grad of an intermediate variable?](https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94)

