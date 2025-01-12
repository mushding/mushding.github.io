---
title: 你所不知道的 Pytorch 大補包(七)：訓練小技巧 AMP 混合精度
mathjax: false
date: 2022-12-29 00:30:59
tags: Pytorch
categories: Pytorch 大補包
---

用一串話簡單解釋什麼是 AMP：

在 2017 Nvidia 提出了用於「混合精度的訓練方法」，是一種可使用不同精度來運算 cuda tensor 運算，Nvidia 很貼心的用 python 整理成 apex 套件讓大家方便使用 https://github.com/NVIDIA/apex。而在之後 pytorch 1.6 的更新中，在 Nvidia 的幫忙下，開發了 torch.cuda.amp 函式 (AMP 全稱 Automatic Mixed Precision)，使得混合精度訓練可以在 pytorch 中直接引入並使用。

keywords: AMP
<!--more-->


相信大家看完一定還是霧颯颯，那接下來依照下列順序介紹 AMP，更詳細的了解背後的歷史演進：

* 什麼是精度？
* 為什麼要混合精度？
* 如何使用 AMP？

### 什麼是精度？

一般我們在使用 pytorch 時，如果簡單的初始化一個 tensor，如下：

```python
import torch

tensor1 = torch.zeros(20)
print(tensor.type())   # 'torch.FloatTensor'

tensor2 = torch.Tensor([1,2])
print(tensor.type())   # 'torch.FloatTensor'
```

可以看到 pytorch 中，新增預設的精度就是 FloatTensor，習慣上中文會稱它叫：單精度浮點運算 (single)

小小複習一下，通常 float 會用 32 個 bit 來存資料；double 稱雙精度浮點則用 64 bit

而在 pytorch 中一共支援 10 種不同資料型態的 tensor：

```
torch.FloatTensor    (32-bit floating point)
torch.DoubleTensor   (64-bit floating point)
torch.HalfTensor     (16-bit floating point 1)
torch.BFloat16Tensor (16-bit floating point 2)
torch.ByteTensor     (8-bit integer (unsigned))
torch.CharTensor     (8-bit integer (signed))
torch.ShortTensor    (16-bit integer (signed))
torch.IntTensor      (32-bit integer (signed))
torch.LongTensor     (64-bit integer (signed))
torch.BoolTensor     (Boolean)
```

可以發現在 DoubleTensor 下方多了一個 HalfTensor 「半精度浮點」，而這個就是今天的主角，也是為什麼要使用 AMP 的最大理由。

### 為什麼要混合精度？

剛剛上面介紹各種型態的 Tensor 最後都會整理到 Nvidia GPU 中做運算，而在 GPU 負責運算的單元稱 cuda 核心(**C**ompute **U**nified **D**evice **A**rchitecture 統一計算架構)，一個 cuda 核心由一個 ALU (Integer arithmetic logic uint 整數運算單元) 及一個 FPU (Floating point unit 浮點運算單元) 所組成，也就是說一個 CUDA 核心專門來做**乘法**及**加法**，而 cuda 核心中還有一個特別的指令：FMA (Fused multiply add) 可以用一個指令完成加乘融合的操作。

![image-20220820113513902](https://i.imgur.com/IO8GIwY.png)

一般我們在深度學習中最常看見的算式是這個：
$$
x_{l} = x_{l-1}w+b
$$
這種又加又乘的操作藉由 cuda 核心的幫忙，可以在不改變精度下，把原本要兩個指令完成的事縮減成一個指令，大輻減少運算時間。以上 cuda 預設支援 Float32 的運算，也正好與 pytorch 相符。自 2006 年的 Tesla 架構推出以後，cuda 核心就一直內建在 Nvidia GPU 中了。

不過這時有一個聲音悄悄的跑出來：我們能不能再加速呢？如果還要加速的話有以下兩個地方可以改進：

* 設計新的核心，可以硬體加速更高級的運算，例如一個指令完成 Tensor 運算
* 藉由把浮點數的精度降低，再做乘法，達到減少運算複雜度的加速，但同時又不能失去太多的精度

如果你是 Nvidia 工程師會怎麼呢？小朋友才選擇嘛 XD 當然是兩個都做阿！所以 2017 年年底 Nvidia Volta 架構上提出了新的 Tensor 核心單元，完美達成上面兩件事情：在不損失太多精度下，減少整體的運算時間。接下透過以下兩個 GIF 動畫可以了解到 Tensor 核心的力量

![1fd55a3c-9362-11eb-a595-1278b449b310](https://i.imgur.com/LSa0CvU.gif)

![5a1ec0e0-7e84-11eb-aca1-aa09f3df2eff](https://i.imgur.com/21VdRyt.gif)

上面兩動畫還隱含了兩個資訊：

* Tensor 核心可以做到使用一個指令完成一個 Tensor 運算
* 當資料精度越小時 (FP32 -> FP16 -> INT8)，同一時間下完成的運算量更高

所以整個又回到最一開始的問題，為什麼要使用「混合精度」？因為更低的精度意味著更快的運算，但為了資料不能丟失太多細節，所以有必要使用高精度運算的還是維持 FP32，但是有一些沒那麼重要的運算就可以改使用 FP16，這樣在一個 Tensor 運算中，又有 FP32 又有 FP16 的操作，就是混合精度的原由。

### 如何使用 AMP

剛剛上述提到的 FP32 對應 pytorch 中的 `torch.FloatTensor`，而 FP16 則是對應 `torch.HalfTensor`，這兩種不同的精度各自有什麼優缺點呢？

HalfTensor 的優缺點：

精度低，運算快，但消失精度的代價是算出來的值失去很多細節，這個現象會導致，overfitting/underfitting 的發生。因為在做 Backpropagation 時根據數值不斷的往後計算，越算越小，小到超出 FP16 所能表示的最小數值 $2^{-14}$，會使得更先前的層參數無法更新

另一個問題也是因為 FP16 最小的數值間距為 $2^{-13}$ 如果有小於這個數字的算式都會被當誤差而省略掉了

因此要如何甚選要什麼運算使用 FP16 來加速可是個大問題，好加在 pytorch 已經幫我們整理好了，以下的操作都是可以用 FP16 來加速，因此 pytorch 會自動這型態轉換成 HalfTensor 來計算，而其它則維持 FloatTensor：

```
__matmul__
addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
conv1d
conv2d
conv3d
conv_transpose1d
conv_transpose2d
conv_transpose3d
linear
matmul
mm
mv
prelu
```

那實際上程式碼要怎麼去寫呢？其實也非很簡單，只需引用 torch.cuda.amp 包，再進行以下操作就行了：

```python
# 利用 amp 中的 autocast 來實現，自動判哪些運算要用 HalfTensor 哪些運算維持原樣用 FloatTensor
from torch.cuda.amp import autocast as autocast

# 建立新 model，預設是 torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # 使用 with 關鍵字，把前傳遞 forward 及算 loss
    # 的部份用 autocast() 包起來
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # Backpropagation 不必用 autocast 包起來
    # 理由是 Backpropagation 會依據 Forward 的資料型態直接沿用來做
    loss.backward()
    optimizer.step()
```

是不是很簡單呢？簡簡單單的一行就可以用 Tensor 核心幫你加速訓練/測試的時間，與此同時還有一個好的副作用：顯存下降了！也很合理，因為要存的浮點精度變少了嘛

不過如果只單純這樣用的話，在訓練時會多發生一個問題，訓練會 over/underfitting！，精度的下降果然還是使用在 Backpropagation 時，參數傳不到前面去更新了，因此要再使用 amp 中的另一個黑科技：GradScaler

GradScaler 實際精神在於，把網路算出來的 Loss 用一個倍率放大，在 Backpropagation 存著 .grad 的值也一並放大，但最後用 optimizer 更新參數時還是要把值縮小回原本的大小，這樣子的做法就不會有因為精度損失而導致更新不到前面的參數了

實驗程式碼的實作方式也不困難，如下：

```python
# 利用 amp 中的 autocast 來實現，自動判斷哪些運算要用 HalfTensor 哪些運算維持原樣用 FloatTensor
from torch.cuda.amp import autocast as autocast

# 建立新 model，預設是 torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # 使用 with 關鍵字，把前傳遞 forward 及算 loss
    # 的部份用 autocast() 包起來
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

        # Scales loss. 用一定的倍率放大 Loss，並計算出各個 node 的 .grad 值
        scaler.scale(loss).backward()

        # 這一步詳細流程見下面
        scaler.step(optimizer)

        # 準備著，看下一次是否有要做 scaler 放大 Loss
        scaler.update()
```

這個 scaler 放大倍數也是動態調整的，為什麼呢？理應放大倍率越大越好，保留越多的數字，但現實很骨感，如果真放超大會直接 overfitting 出現 infs，但是放大太小又會出現 NaNs，所以這個 scaler 會自動的去調整放大倍率大小，在不發生仍何 over/underfitting 下找到最合適的放大倍率

以上就是 torch.cuda.amp 的完整詳細介紹及用法啦！要再更進階的話還有一個小細節要注意：如果是有使用 DDP 訓練的方法，在加入 autocast() 要特別注意

除了在 train 的 forward 時要加入 autocast() 前文，同時也要記得在 繼承 nn.module 的 forward() 函式中，也要加上 autocast() 的前文，或是使用 decorator 也可

```python
# 方法一：使用 decorator
MyModel(nn.Module):
    @autocast()
    def forward(self, input):
        ...
        
# 方法一：使用 with 前文
MyModel(nn.Module):
    def forward(self, input):
        with autocast():
            ...


model = MyModel()
dp_model=nn.DataParallel(model)

# 除了訓練 forward 要加，model 中的 forward 也要加
with autocast():
    output = dp_model(input)
    loss = loss_fn(output)
```

那實際效果跑起來如何呢？基本上網友們的反應是：一、顯存下降；二、時間變長，咦…等等等，怎麼用了混合精度時間變慢，不是說精度越小速度越快嗎？後來發現原因出現在 GradScaler 上面，Loss 及梯度在經過一個 scaler 放大縮小一來一回下，增加了不少時間損耗，至於這個功能最後要不要加上去呢…？這個就見人見智囉！ 

### Reference

[cuda core vs tensor core 知乎](https://www.zhihu.com/question/451127498)

[Pytorch自动混合精度(AMP)介绍与使用](https://www.cnblogs.com/jimchen1218/p/14315008.html)

[PyTorch的自动混合精度（AMP）](https://zhuanlan.zhihu.com/p/165152789)