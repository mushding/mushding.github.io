---
title: 你所不知道的 Pytorch 大補包(五)：網路一層模型 Parameter vs Buffer
mathjax: false
date: 2022-12-29 00:18:03
tags: Pytorch
categories: Pytorch 大補包
---

有時候我們在看別人的論文時會發現：常常會有一些「超參數」的出現，像是 ResNet shortcut 進入的權重值等等

這個時候就可以用 Pytorch 提供的 Parameter 和 buffer 來實作，想知道詳細差在哪裡就繼續往下看吧 ~

keywords: Parameter、buffer
<!--more-->

## Parameter 和 buffer

有時候我們想要在網路中新增一層或是一個參數時，就可以使用 Parameter 或是 buffer

* Parameter 在反向傳播時「會」隨著網路更新權重值
* Buffer 在反向傳播時「不會」隨著網硬更新權重值

建立方向：

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        buffer = torch.randn(2, 3)  # tensor
        self.register_buffer('my_buffer', buffer)     # buffer 的定義方式 (str：定義名字，tensor：傳入權重)
        self.param = nn.Parameter(torch.randn(3, 3))  # Parameter 的定義方式 (tensor)
        self.register_parameter("param", param)       # 另一種定義 Parameter 的方式 (與上行程式等價)，看你習慣，好處是可自定義名稱

    def forward(self, x):
        # 可以通过 self.param 和 self.my_buffer 访问
        self.my_buffer(x)     # 使用剛剛定義的 str 名字
        self.param(x)	
```

兩者的共同點就是，在使用 `model.state_dict()` 的方法來保存、讀取網路模型時，都會被存入到 OrderDict 中

```python
# save
torch.save(model.state_dict(), PATH)

# load
model = MyModel(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# get buffer
model = MyModel()
for param in model.parameters():
    print(param)
    
# get param
for buffer in model.buffers():
    print(buffer)
```

在 ViT 的 Patch Embedding 中有使用到，用在 reletive positional encoding 上，因為相對位置編碼不會隨著網路而更新

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1, max_len=5000):
        super(Embeddings, self).__init__()
        self.embs = nn.Embedding(vocab_size, d_model) # word embedding， 需要 backprop 更新
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # pe shape: (0, max_len, d_model)
        pe = self._build_position_encoding(max_len, d_model)  
        self.register_buffer("pe", pe)  # position encoding，不需 backprop 更新
```

### reference
[https://zhuanlan.zhihu.com/p/89442276](https://zhuanlan.zhihu.com/p/89442276)