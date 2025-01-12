---
title: 使用 vscode 遠端 docker 開發使用教學
mathjax: false
date: 2022-07-07 14:30:13
tags: 
    - remote ssh
    - docker
    - vscode
categories: 雜開發心得
---

vscode 真是勘稱一代開發神器，上面一大堆好用套件，不僅讓你的開發環境變得美美的，同時還提供了相當便捷的功能。以下是我這兩三年來，在實驗室進行遠端開發，所記錄下來的一些使用心得。

keywords: vscode
<!--more-->

對於新加入 vscode 的人最大的困難就在於熟悉介面，光這一步就不知道勸退多少人了，即然大家不想學習如何使用介面…那我們就先來步置介面吧 XD，把介面弄的美美的眼睛看起來非常舒服，自然就會有想使用 vscode 的動機啦 XD

## 美美的套件 (事前準備)
我的 vscode 美美之路受以下文章的起萌：[小克的 Visual Studio Code 必裝擴充套件（Extensions）私藏推薦](https://blog.goodjack.tw/2018/03/visual-studio-code-extensions.html)，大家可以點進去多看看，裡面有超多推薦的套件，其中我會再整理出幾個「一定」要裝的套件

One Dark Pro。一個裝下去眼睛就會得到解放的套件
![Image](https://i.imgur.com/5EFRD1R.png)

Material Icon Theme。側邊檔案目錄變得清楚明瞭，檔案類型清清楚楚
![Image](https://i.imgur.com/BijJLgI.png)

GitLens。有在用 Git 的話必裝，可以方便切換 commit，merge 版本，還自帶全自動 Git blame 讓你簡單找到 bug 戰犯
![Image](https://i.imgur.com/NlflEAd.png)

CodeSnap。程式碼截圖神器，就算使用 window 電腦，也可以讓你截出 mac 的味道
![Image](https://i.imgur.com/ytZdWhk.png)

Path Intellisense。在程式中輸入檔案位置神器，可以讓你用 Tab 輸入目錄
![Image](https://i.imgur.com/FFBYp62.png)

## SSH 遠端工作系列

以往我們在使用 SSH 遠端工作時，如果是 mac 使用者可以很方便的用 Terminal 打指令進入，如果是 windows 使用者也可以用 mobaXTerm 作為替代。但是如果遇到需要修改程式碼，又不想用 vi、nano 來開起時，這時 vscode Remote 就是一個很好的選擇了

首先先安裝
![Image](https://i.imgur.com/chTKjOK.png)

完成後左邊選單就會多一個螢幕的圖示，這個就是 SSH 連線的地方
![Image](https://i.imgur.com/ZFv3Hhr.png)

點進 SSH 圖示後，選擇上方齒輪，進入設定
![Image](https://i.imgur.com/jK0oRxR.png)

選擇在使用者/Users 底下的 .ssh/config 檔案，vscode 會自動在電腦對應的位置新增空白檔案，以後所有連線的設定都會存在這裡面。下面講解各個欄位用途
```
Host server_自定名稱
	HostName IP_位置
	User 使用者名稱
	Port 埠號
	IdentityFile RSA 非對稱式私鑰位置
```

設定好後就會在左邊出現自定名稱的電腦圖示，按下右邊的「加資料夾」圖示後，就可以直接用 vscode 連 ssh 啦
![Image](https://i.imgur.com/ZQDVhvy.png)

## Docker
很多時候我們連線到遠端，還會在遠端 server 再建立 docker 虛擬環境，很多時候會需要下到 docker 指令，但是…如果說有 GUI 可以操作呢…？

vscode 可以在遠端 server 上再安裝套件，這個套件是與本地分開的，專屬於 server。我們先來下載 docker 吧

![Image](https://i.imgur.com/jKjX7v8.png)

左手邊就會出現小鯨魚的圖案

![Image](https://i.imgur.com/3a8suR4.png)

按下去之後就可以看到 server 各種 image、container、volume...，非常的視覺化
![Image](https://i.imgur.com/tuGdJp0.png)

對任何 image 可以按右鍵把它 run 起來；對任何的 container 也可以按右鍵進到 bash 裡面

![Image](https://i.imgur.com/tkgVDXw.png)

這個時候 vscode 下方的 terminal 就會更改成 docker 虛擬環境裡的 terminal 囉！

## Remote - Container
有時在開發的時候會發生以下的問題：

![Image](https://i.imgur.com/xpsA9vL.png)

vscode 沒有讀到虛擬機裡的 python 環境位置，導致不管我們在 docker 裡面裝了什麼套件，vscode 都不會知道，這個問題會導致 python 套件無法給你提示，例如打 torch 後面會很多很多其它 function 之類的

為了要解決這個問題，我們需要再安裝一個套件：Remote - Container

![Image](https://i.imgur.com/APw02LC.png)

安裝完後再按 ssh 的電腦圖示會發現，最上面多了一個下拉式選單，裡面有兩個選項：ssh 舊的連線，以及 containers 新的連線

![Image](https://i.imgur.com/laq1Qwm.png)

這個 container 的作用類似 ssh，有點像是「用 ssh 的方式連線進 docker 的環境中」，所以以我們的這個例子，如果我們要到遠端的 docker 開發，我們先要：一、連線到遠端，二、連線到 docker，兩層的連線 XD

如果遠端已經建立好的 container 時，選 containers 下面會多出許多可連線的 server 選項，這些選項都是 docker container (也是這個套件名字的由來嘛)，在對應的 container 下按「加分頁圖示」

![Image](https://i.imgur.com/VLkKv28.png)

它會安裝一些東西，這個步驟可能要跑一陣子，我自己的經驗大約是 3 ~ 5 分鐘，跑完之後就成功的進到了 docker container 的環境中啦！在這裡面所有的 python 套件都正常運作，自然也就不會有黃黃底線的問題了！
