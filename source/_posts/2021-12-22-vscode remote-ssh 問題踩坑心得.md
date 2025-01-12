---
title: 'vscode remote-ssh 問題踩坑心得'
mathjax: false
date: 2021-12-22 19:17:34
tags: 
    - remote ssh
    - vscode
categories: 雜開發心得
---

vscode 的 remote ssh 真的超好用的，但是就是有時候連線進去會有下面的問題：


keywords: remote ssh、vscode
<!--more-->

## Downloading with wget

### 問題原因

vscode 的 remote ssh 真的超好用的，但是就是有時候連線進去會有下面的問題：

```
[10:50:31.984] > Acquiring lock on /home/remoteuser/.vscode-server/bin/9df03c6d6ce97c6645c5846f6dfa2a6a7d276515/vscode-remo
> te-lock.9df03c6d6ce97c6645c5846f6dfa2a6a7d276515
> Installing to /home/remoteuser/.vscode-server/bin/9df03c6d6ce97c6645c5846f6dfa2a6a7d276515...
> Downloading with wget
```

然後就…卡住了…

經過了一陣爬文才知道，原來是 vscode 在連線前會去下載一個 vscode-server-linux-x64.tar.gz 包，目的…(我不太想知道 XD)

但是！重點來了！

不知道大家有沒有下載過 vscode 的經驗，有時候去 microsoft 的官網下載，會常常出現網路錯誤中斷下載，而且下載的速度超慢

這裡的問題是一樣的

當 vscode 要去 microsoft 下載 vscode-server-linux-x64.tar.gz 時也出現了網路錯誤，所以才常常卡在 `Downloading with wget` 不動

### 解決辨法

既然 vscode 的載點不能用，那我們就自己手動下載吧

首先去剛剛 terminal 的錯誤畫面中，把 `.vscode-server/bin` 後面的亂碼 copy 下來

```
// 也就是 9df03c6d6ce97c6645c5846f6dfa2a6a7d276515，注意每個人都不一樣
Installing to /home/remoteuser/.vscode-server/bin/9df03c6d6ce97c6645c5846f6dfa2a6a7d276515...
```

接著把下面的 commit_id 修改成剛剛 copy 的亂碼

這個就是手動下載 vscode-server-linux-x64.tar.gz 包的載點了 (看網址好像是中國的 azure 來的！？)

```
https://vscode.cdn.azure.cn/stable/${commit_id}/vscode-server-linux-x64.tar.gz
```

> 在 Google 的時候也有發現另一個載點：
> 但這個好像就是原本 vscode 官方的載點，超極慢…千萬不要用它
> `https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable`

下載後，把 vscode-server-linux-x64.tar.gz 利用 scp 或是隨身碟 copy 到遠端 server 中

```
scp -i /Users/user/.ssh/${your_key} ~/vscode-server-linux-x64.tar.gz ${your_server}:/home/user/...
```

進入 ~/.vscode-server/bin 資料夾

```
cd ~/.vscode-server/bin
```

創立與亂數同名的資料夾，並進去裡面

```
mkdir ${你的亂數}
cd ${你的亂數}
```

把剛剛 copy 過來的 vscode-server-linux-x64.tar.gz 包 mv 過來

```
mv ~/vscode-server-linux-x64.tar.gz .
```

解壓縮它，並且把裡面的檔案全部 copy 到現在的位置

```
tar -zxf vscode-server-linux-x64.tar.gz
mv vscode-server-linux-x64/* .
```

最後再新增一個 vscode-scp-done.flag 檔案

```
touch vscode-scp-done.flag
```

最後就可以重新整理 remote-ssh ，按 retry 就可以正常連線進去囉！

當然以上超極複雜的做法，只有在以下兩個條件下滿足才會用到它

1. 你第一次連到遠端 server 去
2. 今天 microsoft 網路連線超不好

如果你很幸運的兩個條件都符合，恭禧你要全都照做一邊，或是…隔天等網路比較順後再處理吧 XD

## cat /home/user/.vscode-server/...log: Permission denied

### 問題原因

.vscode-server 連線時會去看一份 log 檔，有時候 log 的權限設錯就會出現問題

### 解決方法

去 terminal 中看看連線的 commit id 是多少

接著去到 .vscode-server 中

```
cd ~/.vscode-server
```

把所有 commit id 相關的檔案權限都改成最大 777 (應該有更安全的做法…)

```
chmod 777 ${commit_id}.*
```

## flock: 99: 錯誤的檔案敘述項

完整錯誤訊息：

```
[14:19:42.047] > main: 列 243: /home/user/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba52050e3
0ba3/vscode-remote-lock.user.899d46d82c4c95423fb7e10e68eba52050e30ba3: 拒絕不符 
權限的操作
[14:19:42.061] > Acquiring lock on /home/user/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba520
50e30ba3/vscode-remote-lock.user.899d46d82c4c95423fb7e10e68eba52050e30ba3
flock: 99: 錯誤的檔案敘述項
Installation already in progress...
If you continue to see this message, you can try toggling the remote.SSH.useFloc
k setting
2da976fb3cde: start
```

### 問題原因

好像是在 /bin 資料夾有一個 vscode-remote-lock.user.${commit_id} 的檔案，是它權限設定不對

### 解決方法

一個治標不治本的方法是刪掉這個檔案，然後重連，可以一次性的解決，但是下一次要再連的時候又不行了

更外一個解法是

1. 進入 vscode 後，按下 ctrl+, 進入偏好設定頁面
2. 上面搜尋列打 useFlock
3. 把選項關掉
4. 再重新連線

## Reference

### 問題一

[Github 上的正解](https://github.com/microsoft/vscode-remote-release/issues/4743)

[csdn 上的誤解](https://blog.csdn.net/zhuzixiangshui/article/details/103680328)

[stackoverflow 上的誤解](https://stackoverflow.com/questions/56671520/how-can-i-install-vscode-server-in-linux-offline)

### 問題三

[Github 上的解法](https://github.com/microsoft/vscode-remote-release/issues/2518)