---
title: 什麼是 python Pass by Assignment？
mathjax: false
date: 2022-12-27 12:06:11
tags: python
categories: 雜開發心得
---

這幾天在刷題寫 leetcode，用最方便的 python 語言來刷，在寫到 [39. Combination Sum](https://leetcode.com/problems/combination-sum/) 這題的時候，為了要滿足遞迴的條件，所以在遞迴函式中加入當前答案 `res=list()` 當做參數，結果發現…這個 List 面試的數值怎麼一直變來變去啦…明明我沒有去動到它的阿…

keywords: Pass by Assignment
<!--more-->

以下是犯人 code：

```python
from typing import List
class Solution:
    def __init__(self):
        self.ans = list()
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        res = list()
        res.append(candidates[0])
        self.findSum(res, candidates, target)

        candidates = candidates[1:]
        res.pop()
        res.append(candidates[0])
        self.findSum(res, candidates, target)
        
        return self.ans
        
    def findSum(self, res, candidates, target):
        if not candidates:
            return
        if sum(res) == target:
            self.ans.append(res)
            return
        if sum(res) > target:
            return

        res.append(candidates[0])
        self.findSum(res, candidates, target)

        candidates = candidates[1:]
        res.pop()
        res.append(candidates[0])
        self.findSum(res, candidates, target)

        return
```

發生事情當下我心想：該不會是發生了 call by reference 吧…，所以特別花了一點時間來研究一下，python 中傳值的方式倒底是怎麼傳

### Call by Value vs Call by Reference

在開始講 python 是怎麼傳值前，我們要先來了解在 python 中是如何定義資料型態的，以下的例子我會舉 JS 當成是另一個語言來做比較

在 JS (以及一部份語言) 中所有的資料型態分為兩種：primitive type 以及 object type

所謂的 primitive type (中文翻原始型別)，在 JS 中有以下類別：

* String
* Number (int, float, ...)
* Boolean
* Null
* Undefined

而大部份你看得到的「其它」類別，都是 object type，也就是都是由一個 object 定義出來的

- Object
- Function
- Array
- Set

而在 JS 中：primitive type 全部都是 call by value，也就是傳進函式前，complier 會先幫你在記憶體中新增一塊不同位置，但是值相同

```js
function call_by_value(x) {
  console.log(x);    // 5 
  x = 1;
  console.log(x);    // 1
}

let x = 5;
console.log(x);      // 5 (進入函數前)
call_by_value(x);
console.log(x);      // 5 (雖然進入函數後 x 有更動到，但因在 function 內是其它記憶體，所以外部值沒變)
```

其它的 Object type 則是 call by reference，也就是不管是在函式內外，變數指向的記憶體位置都是一樣的，所以在 call_by_reference() 函式內的改動，會影響到函式外的變數：

```js
let person = {
  name: 'John',
  age: 25,
};

function call_by_reference(obj) {
  obj.age += 1;
}

console.log(preson);    // { name: 'John', age: 25 }
increaseAge(person);
console.log(person);    // { name: 'John', age: 26 }
```

### Mutable vs Immutable

而在 python 中比較不一樣的是，python 中所有的型別都是一個物件 (object)，每一個資料都會有一個 \__class__ 屬性來看看是由哪一個 class 所生成的

```python
x = 100
print(x.__class__)   # <class 'int'>
print(type(x))       # <class 'int'>     // 與上面的程式等價
```

同時也可以使用 id 來看看這個變數是放在記憶中哪一個位置

```python
print(id(x))         # 4342041840
```

因為每一個型別都是一個物件，所以不會像其它語言的分法一樣，在 python 中是透過是不是 mutable 來區分的。根據網路上別人留言的定義 mutable 的意思是

```
// a mutable object is an object that can be changed
// while an immutable object can't be changed
```

下表是 python 中常見型別是不是 mutable/immutable 的表格。[圖片參考自 Mutable vs Immutable Objects in Python](https://medium.com/@meghamohan/mutable-and-immutable-side-of-python-c2145cf72747)

<img src="https://i.imgur.com/UPYZbbs.png" alt="image-20221206110751263" style="zoom:50%;" />

接下來詳細介紹這兩個的差別：

immutable 物件一旦被創造出來，它的值就永遠不可以再更改，像是 int、float、string、bool、**tuple**。用下面簡單的例子來舉例：

```python
x = 100
print(id(x))    # 4334161232

x = 200
print(id(x))    # 4334161264
```

可以發現因為 x 是 int 型別，是屬於 immutable，一旦值發生改變 python 是直接會再找一塊新的記憶體來存 x 變數，而非修改原本記憶體的值

不知道大家看到這邊的時候有沒有覺得很奇怪，為什麼 python 要這麼沒有效率的一直新增記憶體空間阿？我們不訪試著想想看，如果今天是在 C 中我們重複定義了一個變數會發生什麼事：

```c
int x = 100;
int x = 200;    // error: redefinition of ‘x’
```

它會噴重複定義的錯，因為 x 所在的記憶體位置已經被使用了，當我們想再定義一次時，complier 會提醒我們不能這麼做。但是有沒有想過，為什麼 python 可以這麼做呢？python 雖然少了型別 (int) 的部份，但還是可以達成下面程式。

```python
x = 100
x = 200
```

其實我們之所以可以在 python 裡面執行這種操作，就是因為每當使用 `=` 去 assign 一個變數時，python 都會在記憶體中新增一塊位置存放它，也就是說其實這兩個 x 根本是不一樣的東西，而這也是 immutable 的精神所在：值絕對不會被更改

如果用圖片的方式來表達的話，python 中的執行方式，就會如下圖所式：

<img src="https://i.imgur.com/Mu0Wonv.png" alt="image-20221227121240682" style="zoom:67%;" />

到這邊就引出這篇文章最重要的想法：這種每當有 assignment 發生時，immutable 型別所指向的記憶體都會改變，這種方法在 [python official document](https://docs.python.org/3/faq/programming.html#how-do-i-write-a-function-with-output-parameters-call-by-reference) 中稱作：**Pass by Assignment**。

我們現在來看看如果把 Pass by Assignment 的想法加上 function 會發生什麼事，以下範例我們將 immutable 變數當成參數傳進 function：

```python
def fun(x, y):
  x = 5
  y = y + 1
  print(x, y)    # 4 24
  
a = 10
b = 20
print(a, b)      # 10 20

fun(a, b)
print(a, b)      # 10 20
```

當 a b immutable 變數傳進 function 時 python 會像其它的語言一樣，新增一塊記憶體並且 copy 變數的值到這個記憶體中，不管在 function 中的任何操作都不會影響到外面 a, b 變數。

也就是說 python 的 immutable 型別在 Pass by Assignment 中，很像在其它語中稱作的 Call by Value，記憶體操作如下圖表示：

<img src="https://i.imgur.com/HMMzAbJ.png" alt="image-20221227121526401" style="zoom:50%;" />

接下來是換 mutable 的部份。mutable 型別的有：list、set、dict，這些型別如同 mutable 的意義一樣：是可以在宣告後修改的，也就是在同一個記憶體位置中修改存的值，我們用以下的範例來看看：

```python
my_list = [1, 3, 6]
print(id(my_list))     # 4337133824

my_list[0] = 100
print(id(my_list))     # 4337133824

my_list.append(900)
print(id(my_list))     # 4337133824
```

<img src="https://i.imgur.com/oL5rrrg.png" alt="image-20221227121550340" style="zoom:50%;" />

可以發現不管我們對 my_list 做：修改值、append 等操作，id(my_list) 都是不會變的，而正是 mutable 的主要表現：可以在相同記憶體下修改其中的值

那如果 mutable 型別遇上 function 會發生什麼事呢？看看下面程式的例子：

```python
def fun(x, y):
  x.append(30)
  y['id'] = 10
  print(x, y)    # [10, 20, 30]
  							 # {'name': 'John', 'age': 16, 'id': 10}
  
a = [10, 20]
b = {
  'name': 'John',
  'age': 16,
}
print(a, b)      # [10, 20]
								 # {'name': 'John', 'age': 16}

fun(a, b)
print(a, b)      # [10, 20, 30]
  							 # {'name': 'John', 'age': 16, 'id': 10}
```

可以發現，當 a b 傳到 function 後，當 function 內部的 x y 修改值後，外部的 a b 同時也會一起修改

當 a b mutable 變數傳進 function 時，python 會將 function 內的變數 x 指向 a 所指的記憶體位置，使得在 function 內修改值時，function 外也會同時被修改 (因為就是同一個東西)

也就是說 python 的 mutable 型別在 Pass by Assignment 中，很像在其它語中稱作的 Call by Reference，記憶體操作如下圖表示：

<img src="https://i.imgur.com/gFS3AVQ.png" alt="image-20221227121609941" style="zoom:50%;" />

但是與正常 Call by Reference 不一樣的是，如果我們在 function 內是用 assignment 重新給定一個 mutable 變數值時，python 會像 Call by Value 一樣重新找一塊新的記憶體放，而 function 內外的值互不相影響，如下圖：

```python
def fun(x):
  x = [0]
  print(id(x)) # 4337133860
  
a = [10, 20]
print(a)      # [10, 20]
print(id(a))  # 4337133824

fun(a)
print(a)      # [10, 20]
print(id(a))  # 4337133824
```

<img src="https://i.imgur.com/nQ3pNSh.png" alt="image-20221227121638838" style="zoom:50%;" />

可以發現當 `x = [0]` 後 python 竟然是重新找一個記憶體去存放，所以當然也不會動到 a 裡面的值，而正是 python Call by Assignment 最要留意的一個點，它並非「完全的」Call by Reference 喔 ~

### Reference

[JS基礎：Primitive type v.s Object types](https://medium.com/@jobboy0101/js%E5%9F%BA%E7%A4%8E-primitive-type-v-s-object-types-f88f7c16f225)

[(圖片主要參考來源) [Python 基礎教學] 什麼是 Immutable & Mutable objects](https://www.maxlist.xyz/2021/01/26/python-immutable-mutable-objects/)

[(推薦 說得很清楚！)【 Python 教學 】什麼是 Pass By Assignment？](https://luka.tw/Python/%E5%9F%BA%E7%A4%8E%E6%95%99%E5%AD%B8/past/2021-09-21-is-python-call-by-sharing-122a4bf5a956/)

[官方 Call by Assignment Document](https://docs.python.org/3/faq/programming.html#how-do-i-write-a-function-with-output-parameters-call-by-reference)

 