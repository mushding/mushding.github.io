---
title: 為什麼 BLoC 中要使用 Equatable 呢？
mathjax: false
date: 2021-09-23 00:43:54
tags: BLoC
categories: Dart & Flutter 開發
---

這幾天在學 flutter ，看到大家說當程式大起來的時候，state 會不好整理及控制。而 React 中有 Redux ，在 flutter 中大家最受歡迎的方法是 flutter_bloc ，以下簡單筆記我學 BLoC 的一些心路歷程

keywords: BLoC、Equatable
<!--more-->

## 為什麼 BLoC 中要使用 Equatable 呢？

在了解為什麼要使用 Equatable 之前，我們先來看看什麼是 equals、hashCode

### == vs equals vs hashcode

最初在使用這些觀念的語言是 Java，而 Java 對於以上三個值有不同的定義

所謂「==」是指符號兩邊的「記憶體位值」是否相等，兩對象是不是參考同一個位置

而 equal 則是 Java 提供的一個 Override 方法，如果我們沒有特別去 Override 它的話，功能就與 「==」一致。那什麼時候我們會用到 equal 呢？當我們今天要比較的資料是自定義的 class 時，如：

```java
class People {
	String shirt_color;
	int age;
}
```

要怎麼比較兩個不同的 People 呢？不太可能直接用 == 來比較吧，這個時候我們就會用到 equal ，把原本的定義 override 加上自己的定義

```java
People Tom = new People("blue", 20);
People Alex = new People("red", 15);

Tom == Alex // ??!!
```

像上面這個例子中 People 中有兩個 member，shirt_color 以及 age，因此在實作 equal 時，要特別去比較這兩個 member 的值是否相等，完整程式如下：

```java
public class EqualsDemo {
	private String shirt_color;
	private int age;
 
  @Override
  public boolean equals(Object o) {
    
    // (一) 
  	if (this == o) return true;
  	
    // (二)
    if (o == null || getClass() != o.getClass()) return false;
  	
    // (三)
    EqualsDemo that = (EqualsDemo) o;
  	if (name != null ? !name.equals(that.name) : that.name != null) 
      return false;
  	
    return info != null ? info.equals(that.info) : that.info == null;
  }
  
  @Override
  public int hashCode() {
  	int result = name != null ? name.hashCode() : 0;
  	result = 31 * result + (info != null ? info.hashCode() : 0);
  	return result;
  }
}
```

可以看到 (三) 的地方，我們自己多自定義比較 shirt_color 與 age 是否相同，來符合我們的需求

那 (一) 與 (二) 呢？這裡就要先提到 equals() 的 4 個特性

1. 反射性：`x.equals(x)` 必需是 True
2. 非空性：`x.equals(null)` 必需是 False
3. 對稱性：`x.equals(y)` 與 `y.equals(x)` 必需同時成立
4. 類推性：如果 `x.equals(y)` True、 `y.equals(z)` True  則 `x.equals(z)` 也必定 True

而式中的 (一) (二) 正是實作了 equals() 的前兩個特性，確保不違反定律

但是最下面為什麼還要再 override hashCode 呢？所謂的 hashCode 是 **Java 把變數所存的實體記憶體位置經過一個 hashmap 後得到的值**，在 Java 中每一個變數都會有一個獨一無二的 hashCode ，如果它們是同一個變數，則 hashCode 會相同

那為什麼要這樣設計呢？假設我們今天有 1000 個變數，今天新增第 1001 個變數，我們要怎麼知道這第 1001 變數是不是與前 1000 的其中一個相同呢？當然最笨的方法就是一個一個找，可是太沒效率了。於是 hashCode 就來解決這個問題，hashCode 利用 hashmap 的特性來達到：只要是同一個變數，則 hashCode 就會相同

注意 hash 的小細節喔！

* 兩對象相等，所產生的 hashCode 一定一樣
* 兩 hashCode 一樣，不一定代表這兩個對象相等喔 (因為 hash 的 collide)

總結：如果要在 Java 中 override 「==」的話，除了要 override equals 比較其它自定變數，也要 override hashCode 記這這個為了加速了誕生的東西，不然會產生兩相同對象但 hashCode 不同的事情發生

### BLoC 與 Equatable

經過上面的解釋可以了解了 equal 以及 hashCode，而 Dart 與 Java 類似也有相同的概念，於是有了 Equatable 這個套件讓我們不用再手動 override equal 以及 hashCode 了，它會自動幫我們做這一件事情

只是…為什麼 BLoC 中要使用到它呢？

當我們建立一個 class 繼承 Equatable 時，我們可確保 LoginStates 是唯一的，當這個 state 發生兩次以上時，不會再一個一模一樣的呼叫，也不會再重建裡面全部的 Widget

```dart
abstract class LoginStates extends Equatable{}
```

或者是 Stream 與 Equatable 之間，當 Stream 中有兩個一模一樣的 state 被呼叫時，第二個會自動省略

```dart
@override
Stream<LoginStates> mapEventToState(MyEvent event) async* {
  yield LoginData(true, 'Hello User');
  yield LoginData(true, 'Hello User'); // This will be avoided
}
```

更詳細的解說可以到以下網址：[https://medium.com/flutterworld/flutter-equatable-its-use-inside-bloc-7d14f3b5479b](https://medium.com/flutterworld/flutter-equatable-its-use-inside-bloc-7d14f3b5479b)

### 結論

Equatable 省下了我們 override equals 與 hashCode 的時間

而 BLoC 中加入 Equatable 可以避免重覆 state 不必要的重覆呼叫及重建 Widget，優化了速度以及記憶體