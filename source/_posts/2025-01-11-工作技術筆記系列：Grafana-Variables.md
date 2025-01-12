---
title: 工作技術筆記系列：Grafana Variables
mathjax: false
date: 2025-01-11 17:10:26
tags: 工作技術筆記
categories: 工作技術筆記
---

接續上一篇 Grafana 的文章，接下來繼續更深入介紹 Grafana Variable

工作開發心得 - Grafana variable

keywords: Grafana
<!--more-->

![Image](https://i.imgur.com/KNBBC1z.png)

在 Grafana 左上角有時可以看到下拉式選單，可以設定一些環境變數，每改變變數時，底下的 Dashboard 也會做出對應的改變，這個功能稱為 Variables。

就像是程式中的變數一樣，我們可以在 UI 中設定 Variables 來動態改變 metric query 中的值

例如可以把 region 設成變數，這樣就可以方便的在同一個頁面下，橫向對比不同 region 的 Dashboard

而在官網中舉了三個 use case：

* repeated panels Grafana
  * 可以重複相同的 panel
* dynamic dashboard Grafana
  * 依據變數動態地調整 Dashboard，可以不用開一樣的 Dashboard
* nested variables Grafana
  * Variable 是可以巢狀疊在一起的喔

## 怎麼使用

Grafana 稱，含 Variable 的 Query 叫做 Templates，而一個 Templates 長這樣：

```
wmi_system_threads{instance=~"$server"}
```

其中 $server 就是調用 server 變數的意思

## 怎麼新增變數？

找到右上角的齒輪

![Image](https://i.imgur.com/BR7kVSE.png)

會有 Variables 的 Tab

![Image](https://i.imgur.com/gHskEV0.png)

* variable type
  * 等等下面會介紹
* Name
  * template 中的變數名稱
    * $pod
* Label
  * 顯示在 Grafana dashboard 上的名稱，預設跟 Name 相同
* selection options
  * 可以起用 All or 多選

![Image](https://i.imgur.com/pfQ0R4P.png)

## 一些新增變數的方法

### Query variable

下 PromQL query 找到想要變數清單，例如：k8s 中 pod 的清單

![Image](https://i.imgur.com/7GOEOJ7.png)

### Custom variable

完全自定義的變數，格式為 `key1 : value1,key2 : value2` (key 是顯示在下拉式選單的名稱；value 是實際帶到 template 的變數)

### Text box variable

與其給下拉式選單，不如直接給一個輸入框，可以輸入任意字串

### Interval variable

時間間隔變數，用法：`1m, 10m, 1h, ...`

### Ad hoc variable

與其給一個定好的變數，不如給使用者自己定義 query，可以自己選 `XXX="XXX"`

## 如果想把一串 query 當變數？

可以參考下面文章：

[Change Grafana query dynamically based on user inputs](https://community.grafana.com/t/change-grafana-query-dynamically-based-on-user-inputs/26416/4)

主要總結為：用 custom variable

`CPU: usage{...}, MEM: ram{freq="", ...}`

## Reference

[Variables |  Grafana documentation](https://grafana.com/docs/grafana/latest/dashboards/variables/)

[How to Create and Work with Variables | Grafana](https://www.youtube.com/watch?v=mMUJ3iwIYwc&ab_channel=Grafana)