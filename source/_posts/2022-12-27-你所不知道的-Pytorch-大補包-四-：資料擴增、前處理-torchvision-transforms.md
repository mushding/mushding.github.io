---
title: 你所不知道的 Pytorch 大補包(四)：資料擴增、前處理 torchvision.transforms
mathjax: false
date: 2022-12-27 14:42:41
tags: Pytorch
categories: Pytorch 大補包
---

在處理 dataset，有時候我們會遇到需要資料前處理、資料擴增…等等對影像處理的步驟：像是希望可以透過旋轉、鏡像做到資料擴增；希望可以利用影像裁切統一輸入影像大小

而 pytorch 也很貼心的提供給我們一個套件使用：torchvision，torchvision 裡面提供了非常多的影像處理方法：像是旋轉、鏡像、裁切

以下這篇文章整理了大部份常用到的函式：[Pytorch提供之torchvision data augmentation技巧](https://chih-sheng-huang821.medium.com/03-pytorch-dataaug-a712a7a7f55e)

keywords: torchvision
<!--more-->

## 如何使用？

torchvision 所有函式輸入只支持 PIL Image，也是就使用 PIL Image 這個套件打開圖片的格式，才會被 torchvision 接受，其它像是 torch.tensor 或是 np.array 都是沒有辦法的

以我們要把影像轉為灰階圖片 -> 然後從 PIL Image 轉成 torch.tensor，PIL 的 SOP 如下：

```python
import PIL.Image as Image
from torchvision import transforms
    
# read image with PIL module
img = Image.open(imagepath)
img = transforms.Grayscale()(img)
img = transforms.ToTensor()(img)
```

如果覺得這樣寫太多行的話，torchvision 也有提供 transforms.Compose 的方案，可以把很多的 transform 打包在一起，就可以統一呼叫便於管理

```python
import PIL.Image as Image
from torchvision import transforms

# 定義 compose    
transfrom = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# read image with PIL module
img = Image.open(imagepath)
# 使用 compose
img = transform(img)
```

而在 transforms.Compose 中，我們也可以放入自定義的影像處理函式，假設我們要定義一個「自定義 Padding」的處理，程式碼如下：


```python
import PIL.Image as Image
from torchvision import transforms

# 定義 compose    
transfrom = transforms.Compose([
    transforms.Grayscale(),
    # 自定義的處理函式
    SuarePad(),
    transforms.ToTensor()
])

# read image with PIL module
img = Image.open(imagepath)
# 使用 compose
img = transform(img)
```

```python
class SquarePad:
    def __init__(self, targetW, targetH):
        self.targetW = targetW
        self.targetH = targetH

    def __call__(self, image):
        # 2003 308
        w, h = image.size
        w_fix = self.targetW - w
        padding = (0, 0, w_fix, 0)
        return F.pad(image, padding, 0, 'constant')
```

對於上面自定義 class 的補充說明：

`__init__` 的目的在類似 constructor，只在當一個類別 (class) 實作成物件 (object) 時會呼叫

`__call__` 是可以把 class 也模擬有著函式一樣的特色，用傳入參數並呼叫的方式使用