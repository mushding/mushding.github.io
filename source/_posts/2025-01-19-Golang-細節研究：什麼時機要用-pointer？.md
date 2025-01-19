---
title: Golang 細節研究：什麼時機要用 pointer？
mathjax: false
date: 2025-01-19 18:18:30
tags: 
  - 工作技術筆記
  - Golang 筆記
categories: Golang 筆記
---

在上一篇我們提到了 Go 的 Escape Analysis，接下來我們再來談什麼時候要用 pointer

一般而言，用 pointer 就是可以讓程式加速的代名詞，因為可以避免複製不必要的資料，但在上一篇我們也提到了，Go 在大部份 Sharing up 的場景中，會把 pointer 的變數丟到 heap 裡面去，也就是 Go compiler 會自動幫你 escape 到 heap 去，到 heap 反而又會因為 garbage collection 性能下降…

那倒底什麼場合會用到 pointer 呢？其中的理由又是什麼呢？

keywords: Golang、Pointer
<!--more-->

## 前言

老話一句，使用 pointers 要小心，雖然在 Go 的世界中，Go compiler 已經幫你處理了大部份危險的 case，但還是不能亂用，不是說用了一定就會效能變好，有可能還是會發生：不小心修改到同一份 reference 的 bug、用了 pointer 結果發生 escape to heap 造成後續 Go 還要額外幫你 garbage collect 掉

那有哪些情況是可以用 pointer 的呢？大概可以分成下面三種情況：

## 1. 拿到大 struct 需回傳時，且途中不做修改

假設我們有一個全球天氣 GET api service，它的 call path 是 API -> Backend service -> Database，當我們從 Database 拿到資料後，會一路把資料往前傳，直到前端

在這個使用情型中，因為我們不會對 Database 拿來的資料做修改，這個情況適合用 pointer 來回傳資料，這樣子就可以避免倒處複製一份 struct 出來，從而減少記憶體的使用量

但要注意，如果今天的使用情形，在回傳 pointer 後我們會對他做修改的話，這種 use case 就不建議用 pointer 回傳，因為一但有做修改，就有機會被 Go compiler 偵測到，進而執行 escape to heap

## 2. 修改 Receiver function 的值

receiver func 前面可以有加 pointer 與不加 pointer 兩種選擇

```go
func (s *MyStruct) pointerMethod() { } // method on pointer
func (s MyStruct)  valueMethod()   { } // method on value
```

這兩種差在哪裡呢？其實官方的網站已經寫的很明確了：[Should I define methods on values or pointers?](https://go.dev/doc/faq#methods_on_values_or_pointers)。整理了一下大概可以分成下面幾點討論：

1. 如果你的 struct 很大，使用 pointer 可以減少記憶體使用
2. 如果你的 receiver function 會修改到 struct 本身成員的值，一定要用 pointer receiver，而且一定要大家統一用
3. 為了 coding convention，如果一下子有加 pointer 一下子沒加，可能會很混亂，於是也可以直接全部 default 用 pointer receiver，就可以避免日後遇到要修改 struct 本身值，還要特別把全部的 receiver 改成 pointer receiver 的工

## 3. 表達真正的空值

這個 use case 比較常與 struct tag 一起出現，例如要把 JSON payload 轉成 go struct type 時，又或是把 database type 轉成 go struct type 時。舉下面的例子：

假設我有一個使用者資料的 JSON payload，其中 age 的欄位是 optional 的，可有可無

```json
{
    "email": "mushding@gmail.com",
    "password": "12345"
}

{
    "email": "mushding@gmail.com",
    "password": "12345",
    "age": 14
}
```

再寫一個 struct 去接 JSON payload

```go
type User struct {
    Email    string `json:"email"`
    Password string `json:"password"`
    Age      *int   `json:"age"`
}
```

如果這裡的 age 不加 pointer 的話，在做 json.Unmarshal() 時，go 會自動把 age 塞 0 進去，也就是 int 的 zero value

如果我們想要有多一個型態，特別表示 nil value 的話，這個時候 pointer 就派上用場了，如果使用者沒有給這個欄位，go 會給 pointer 的 zero value 也就是 nil，就可以做出與 0, int 不同的第三種狀態表示


## Reference

[When to use pointers in Go | Medium](https://medium.com/@meeusdylan/when-to-use-pointers-in-go-44c15fe04eac)

[When to use Pointer in Go | Youtube](https://youtu.be/06tOx08sye0?si=uaC3Ti_pvEbB6zcf)