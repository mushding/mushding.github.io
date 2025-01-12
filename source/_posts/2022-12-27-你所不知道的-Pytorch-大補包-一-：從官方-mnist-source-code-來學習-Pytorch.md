---
title: 你所不知道的 Pytorch 大補包(一)：從官方 mnist source code 來學習 Pytorch
mathjax: true
date: 2022-12-27 13:45:48
tags: Pytorch
categories: Pytorch 大補包
---

Pytorch 已經成為深度學習的主流框架了，以下系列是我從大學時期，一路學習 Pytorch 筆記整理下來的心得，後來再經整理整理決定放到網路上給大家參考，希望可以幫助到更多從 0 開始想自學的人 ~

整個系列會由淺到深，一路從最基本的 SOP，到 Pytorch 底層的實作邏輯，以下先以深度學習界的 Hallo World mnist 來開始學習

keywords: mnist
<!--more-->

* 首先來看看 mnist 在 pytorch 上實作的程式

```python
  import torch
  import torch.utils.data as Data
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torch.optim.lr_scheduler import StepLR
  
  from torchvision import datasets, transforms
  
  
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, 3, 1)
          self.conv2 = nn.Conv2d(32, 64, 3, 1)
          self.dropout1 = nn.Dropout(0.25)
          self.dropout2 = nn.Dropout(0.5)
          self.fc1 = nn.Linear(9216, 128)
          self.fc2 = nn.Linear(128, 10)
  
      def forward(self, x):
          x = self.conv1(x)
          x = F.relu(x)
          x = self.conv2(x)
          x = F.relu(x)
          x = F.max_pool2d(x, 2)
          x = self.dropout1(x)
          x = torch.flatten(x, 1)
          x = self.fc1(x)
          x = F.relu(x)
          x = self.dropout2(x)
          x = self.fc2(x)
          output = F.log_softmax(x, dim=1)
          return output
  
  def train(model, device, train_loader, optimizer, epoch, log_interval, is_dry_run):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = F.nll_loss(output, target)
          loss.backward()
          optimizer.step()
          if batch_idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()
              ))
          if is_dry_run:
              break
  
  
  def test(model, device, test_loader):
      # evaluate mode
      model.eval()
      test_loss = 0
      correct = 0
  
      # stop gradient calculation & decrease GPU processing
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              pred = output.argmax(dim=1, keepdim=True)
              correct += pred.eq(target.view_as(pred)).sum().item()
  
      test_loss /= len(test_loader.dataset)
  
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
  
  
  def main():
      # variables
      batch_size = 64
      epochs = 14
      learning_rate = 1
      log_interval = 5
  
      # is variables
      is_save_model = True
      is_dry_run = False
  
      # is use cuda?
      use_cuda = torch.cuda.is_available()
  
      # set seed
      torch.manual_seed(5000)
  
      # set device
      device = torch.device("cuda" if use_cuda else "cpu")
  
      # set train/test dict
      train_kwargs = {'batch_size': batch_size}
      test_kwargs = {'batch_size': batch_size}
  
      # if use cuda?
      if use_cuda:
          cuda_kwargs = {
              'num_workers': 1,
              'pin_memory': True,
              'shuffle': True
          }
          train_kwargs.update(cuda_kwargs)
          test_kwargs.update(cuda_kwargs)
  
      # mnist transforms
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
  
      # load mnist set
      train_dataset = datasets.MNIST(
          root='./data',
          train=True,
          transform=transform,
          download=True
      )
      test_dataset = datasets.MNIST(
          root='./data',
          train=False,
          transform=transform,
          download=True
      )
  
      # set loader
      train_loader = Data.DataLoader(train_dataset, **train_kwargs)
      test_loader = Data.DataLoader(test_dataset, **test_kwargs)
  
      # start init net
      model = Net().to(device)
  
      # set optimizer
      optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
  
      # set lr decrease scheduler
      scheduler = StepLR(optimizer, step_size=1)
  
      # start training for epoch
      for epoch in range(1, epochs + 1):
          train(model, device, train_loader, optimizer, epoch, log_interval, is_dry_run)
          test(model, device, test_loader)
          scheduler.step()
  
      if is_save_model:
          torch.save(model.state_dict(), "mnist_cnn.pt")
  
  
  if __name__ == '__main__':
      main()
```

* 一個 pytorch 一定會有下面幾個部份

* device

* DataLoader

* optimizer

* scheduler (*)

* init & define Net class

  * \_\_init\_\_
  * forward

* for epoch ->

  * train
  * test
  * save model

### cuda 的設置

* 可以根據狀況來使用 "cpu" 或者是 "gpu"

```python
# is use cuda?
use_cuda = torch.cuda.is_available()

# set seed
torch.manual_seed(5000)

# set device
device = torch.device("cuda" if use_cuda else "cpu")
```

### seed 的設置

* 在初使化參數時是選擇用隨機的方式，所以要選擇用哪一個 seed？

```python
# set seed
torch.manual_seed(5000)
```

### MNIST 的設置

* 使用 torchvision 來用 MNIST
  * root = 儲存的地方
  * train = 設定是否訓練集
  * transform = 是否要做資料前處理轉換
  * download = 是否要下載

```python
# load mnist set
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
```

### DataLoader 的設置

* 在 MNIST 的例子中
  * dataset = 要輸進去的 data
  * batch_size = 一次要多少個 batch
  * shuffle = 每個 epoch 是否要洗牌
  * num_workers = 一次有多少 CPU 執行緒來處理
  * pin_memory = 會使用 GPU 處理
  * collate_fn = 是怎麼處理樣本的，可以自定義來實現自己想要的功能
  * drop_last = 如果總資料除以 batch 大小有餘數的話，還乘下不滿 batch 的資料，要不要丟棄
* 在以下的例子中 ** 代表把 dictionary 裡的資料按照 key: value 的方式拿出來

```python
cuda_kwargs = {
    'num_workers': 1,
    'pin_memory': True,
    'shuffle': True,
    'batch_size': batch_size
}
# set loader
train_loader = Data.DataLoader(train_dataset, **train_kwargs)
test_loader = Data.DataLoader(test_dataset, **test_kwargs)
```

### optimizer

* 在 pytorch 有很多 optimizer 可以用
* 所有的 optimizer 都有 step()
  * 當計算好 loss 之後就用來更新所有的參數

```python
loss.backward()
optimizer.step()
```

### scheduler

* 是用來管理 learning rate，合理的 learning rate 可以快速收斂，隨著訓練的進行 training rate 應該要越來越小
* 而 pytroch 提供了 6 種方式來使用

#### StepLR

* 等間隔的調整 learning rate
* step 以一個 epoch 為一個單位

#### MultiStepLR

* 不一定要等間隔，按設定的間隔調整 learning rate

#### ExponentialLR

* 按指數來調整 learning rate
* lr = lr * gamma ** epoch

#### CosineAnnealingLR

* 以 cosine 為週期，在每個週期最大的時候富新設置 learning rate

#### ReduceLROnPlateau

* 當某個指標不再變化(下降或升高)，就調整 learning rate

#### LambdaLR

* 每一組 epoch 就是一個學習的策略
* 每個 epoch 裡都是酪 lambda function

### model save & load

#### state_dist

* 是比較推的做法，只存模型裡面的參數，可以加速載入的速度
* 使用 torch.save() 保存 state_dict()
* 記住一定要用 model.eval() 來固定 drop 以及歸一化層，不然每次的結果都不一樣

```python
torch.save(model.state_dict(), PATH)
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

#### 整個 model

* 比較直觀的做法，但因為是整個 model 比較大

```python
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()
```

#### Checkpoint

* 除了 model 的 state_dict() 之外，有時也可以同時存一些其它的參數
* 像是 epoch 數，optimizer 也會有 state_dict() 的值，loss 值等等…
* 值都是用 dict 來存，所以要存取參數也要用 checkpoint['something'] 的方法

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
            
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)
 
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
 
model.eval()
```

### batch 裡的設置

* optimizer.zero_grad()
  * 初始化梯度為 0
* output = Net(model)
  * 也就是求出 forward propagation
* loss = critertion(output, target)
  * 算出 loss
* loss.backward()
  * 也就是求出 backward propagation
* optimizer.step()
  * 更新參數

```python
optimizer.zero_grad()
output = Net(model)
loss = critertion(output, target)
loss.backward()
optimizer.step()
```

### loss function

* 常見的有以下幾種

#### log_softmax

* 就是 log 和 softmax 一起做
* 而 softmax 的公式如下：
  * 把每個值做 exponential ，再除以全部 exponential 加總的值
  * 值會在 0 ~ 1 之間

$$
softmax(x_i) = \frac{exp(x_i)}{\Sigma exp(x_i)}
$$

* 那 log_softmax 就只是再加上個 log
* 值為 $-\infin$ ~ 0

$$
softmax(x_i) = log(\frac{exp(x_i)}{\Sigma exp(x_i)})
$$

* dim = 0 代表列總合為 1
* dim = 1 代表行總合為 1

#### nll_loss

* negative log likelihood loss
* 把 softmax 的結果中每一個 label 的數值拿出來  
* 取絕對值相加求平均就是 nll_loss

#### cross_entropy

* 在計算兩個向量的相似度，計算期望向量與實際向量的相似度
* 與內積相同，內積也可以看做在做相似度
* 在二元的世界中公式可以表示成：
$$
y . p(f(x)) + (1-y) . (1 - p(f(x)))
$$

* 接著要取 log

$$
y . ln(p(f(x))) + (1-y) . ln(1 - p(f(x)))
$$

* 而因為算出來是機率，越大越相似
* 所以加個負號才能代表 loss 的精神 -> 越小越好