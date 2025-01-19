---
title: 'Golang 細節研究：Go 是怎麼處理 Allocations: The Stack and the Heap'
mathjax: false
date: 2025-01-19 15:32:30
tags: 
  - 工作技術筆記
  - Golang 筆記
categories: Golang 筆記
---

最近在工作時，在寫公司專案的 unit test 時，遇到了一個神奇的 error：unit test 有會機會跑不過，這種「有機會」的 Bug，絕大部份原因都是出來 thread 上，因為不同 thread 彼此搶記憶體所導致的 error，同事立即發現有可能是 pointer 的問題，在找到倒底是什麼原因產生這個 bug 前，讓我們來看看 Go 是怎麼分配記憶體的…？先來來底層的實作方法，再來看看是不是真的是由 pointer 所所發這個問題。

keywords: Golang、Stack、Heap
<!--more-->

## Stack and Heap

大部份的程式有兩種記憶體儲存方式：Stack and Heap

![Image](https://i.imgur.com/zRtDf3W.png)

簡單歸納一下 stack 跟 heap 彼此的差異，可以參考下面的表：

| 特性            | Stack                              | Heap                              |
|-----------------|------------------------------------|-----------------------------------|
| **使用情況**    | 局部變數、函數呼叫                 | 動態記憶體分配、大型資料          |
| **速度**        | 快                                 | 慢                                |
| **管理**        | 自動管理                           | 手動管理                          |
| **大小限制**    | 有限（通常較小）                   | 幾乎無限制（取決於系統可用記憶體）|
| **常見問題**    | Stack overflow                     | 記憶體洩漏、碎片化                |
| **優點**        | 分配和釋放速度快，自動管理         | 靈活性高，大小無限制              |
| **缺點**        | 大小有限，生命周期受限             | 分配和釋放速度慢，容易出現記憶體洩漏和碎片化問題 |

接下來我們來看看 Go 是怎麼幫我們實作這一塊：

## Stack in Go

假設我們有下面這行範列程式，我們來一步步看一下記憶體分配圖，看看葫蘆裡賣什麼藥

```go
func main() {
    n := 4
    n2 := square(n)
    println(n2)
}

func square(x int) int {
    return x * x
}
```

![第一步：main func 初始化兩個變數，並依序放進 stack 中](https://i.imgur.com/p0VpxQa.png)
<p style="text-align: center;">第一步：main func 初始化兩個變數，並依序放進 stack 中</p>

![第二步：跳進 square func，把參數 x 放進 stack 中，最後 return 回去並修改 x2 的值](https://i.imgur.com/UjR91NF.png)
<p style="text-align: center;">第二步：跳進 square func，把參數 x 放進 stack 中，最後 return 回去並修改 x2 的值</p>

![第三步：這時因為 stack 回到 n2 那一行，下半段因 func square 所產生的記憶體空間就變成 invalid 的](https://i.imgur.com/exiNykz.png)
<p style="text-align: center;">第三步：這時因為 stack 回到 n2 那一行，下半段因 func square 所產生的記憶體空間就變成 invalid 的</p>

![第四步：println 這一行又會新增一段記憶體，這時會自動把下面的記憶體直接覆蓋掉](https://i.imgur.com/xO8r9PB.png)
<p style="text-align: center;">第四步：println 這一行又會新增一段記憶體，這時會自動把下面的記憶體直接覆蓋掉</p>

## Stack with pointer in Go

接下來我們來看看，如果程式中使用 pointer 又會發生什麼事呢？

```go
func main() {
    n := 4
    inc(&n)
    println(n)
}

func inc(x *int) {
    *x++
}
```

![第一步：初始化變數 n](https://i.imgur.com/dtK9RGV.png)
<p style="text-align: center;">第一步：初始化變數 n</p>

![第二步：初始化一個指向 n 的 pointer，並且傳到 int() 去](https://i.imgur.com/Oc3gUzt.png)
<p style="text-align: center;">第二步：初始化一個指向 n 的 pointer，並且傳到 int() 去</p>

![第三步：int() 會把 pointer dereference 並且 += 1，修改掉原本上面 n 的值 (4 + 1= 5)](https://i.imgur.com/6c2nMv5.png)
<p style="text-align: center;">第三步：int() 會把 pointer dereference 並且 += 1，修改掉原本上面 n 的值 (4 + 1= 5)</p>

![第四步：回到 main() println 又會再往下新增一段記憶體，把之前新增過的都蓋掉](https://i.imgur.com/QCEWfS5.png)
<p style="text-align: center;">第四步：回到 main() println 又會再往下新增一段記憶體，把之前新增過的都蓋掉</p>

這時我們可以得到一個結論：Sharing down **typically** stays on the stack，什麼是 sharing down？像是把參數、pointer 往下傳到其它 func 去

## What if returning a Pointer?

那如果我們的程式是回傳一個 pointer 呢？以下面的程式為例子

```go
func main() {
    n := answer()
    println(*n/2)
}

func answer() *int {
    x := 42
    return &x
}
```

![第一步：main() 初始化 n，並假定給它 nil zero value](https://i.imgur.com/qba9HY9.png)
![第二步：進到 answer() 初始化 x 變數](https://i.imgur.com/SILJPkZ.png)
![第三步：回去到 main()，注意此時 answer() 裡剛剛新增的變數，變成 invalid 的](https://i.imgur.com/i62E8Ky.png)
![第四步：println 往下把其它記憶體蓋掉，這時出問題了，它把我想要 reference 的 x 給蓋掉了，我們永遠拿不到 x 的值](https://i.imgur.com/XSPuY01.png)

我們從第四步可以發現，當執行 println 時，它會把 x 的記憶體位置蓋掉，發生 memory leak 了…。好在 Go compiler 其實會主動幫我們解決這一類型的問題，我們來看看 Go compiler 怎麼做到的

![修改第二步：進到 answer() 初始化 x 變數時，不是在 stack 中新增，而是到 heap 中新增](https://i.imgur.com/qbpBVp1.png)

發現修改後的留程，Go compiler 會「**自己**」 知道，這時在 stack 中新增 x 變數記憶體是一件不安全的事情，它會自動把 x 改放到 heap 中，這樣未來 println 在往下蓋掉時，就不會蓋掉我們剛剛剛新增的變數 x。

所以我們又可以得出一個結論：Sharing up **typically** escapes to the heap，這裡的 sharing up 像是 returning pointer, returning reference


## Escape Analysis

注意！我們都用 typically 這個詞，因為倒底是 sharing up or down 完全是靠 compiler 動態決定的，沒有一定的答案，而人類也沒有辦法只看程式就可以一眼看出來這裡是 sharing up or down

我們來看 go 官方 document 怎麼說

> When possible, the Go compilers will allocate variables that are local to a function in that function’s stack frame. 
> However, if the compiler **cannot prove** that the variable is **not referenced after the function returns**, then the compiler must allocate the variable on the garbage-collected heap to avoid dangling pointer errors. 
> 
> [How do I know whether a variable is allocated on the heap or the stack?](https://go.dev/doc/faq#stack_or_heap) 

而這種由 compiler 判斷一個變數要不要變成 heap 的分析稱作：**Escape analysis**

雖然說人類沒辦用肉眼一眼看清，什麼時候要把變數丟到 heap 去，但可以藉助工具的力量，透過下面的指令，Go 就會印出過來說，這個變數有沒有被 escaped to heap

```
go build -gcflags "-m=2"
```

## The general escape timing

以下幾種情況，大部份會 trigger compiler 的 escape to heap

1. 當 function 內的創建的一個變數**可能**會在離開後被 referenced 時
2. 當 compiler 覺得這個 object 太大，以致於塞不進 stack 中
3. 在 compile time 時，compiler 不知道這個 object 的大小

還是看不太懂嗎？這裡直接整理一些比較好理解的講法，直接從有哪些變數有可能會被丟進 heap 去的角度來看

1. 變數值有多個 pointer 指向時
2. Interface 裡的變數
3. Func literal 變數：像是 closure、lambda function
4. Maps, Channels, Slices, Strings 這種長度可能不固定的變數

## Conclusion

好，講到這邊，我們下幾個結論：

1. 程式商業邏輯正確性永遠排第一，其次才是效能 (不要為了這一點點的效能，而忽略了正確性)
2. Sharing down typically stays on the stack.
3. Sharing up typically escapes to the heap.
4. 多去問 compiler，因為人類也不太知道什麼時候會發生 escape

## Reference

[Understanding Allocations: the Stack and the Heap - GopherCon SG 2019](https://youtu.be/ZMZpH4yT7M0?si=ueiibWHjhaHh6ddk)