---
title: 如何在 docker container 中 matplotlib 顯示中文？
mathjax: false
date: 2022-12-27 11:58:39
tags: 
    - docker
    - matplotlib
categories: 雜開發心得
---

一般來說 matplotlib 在產生 figure 時，所套用的字體並未包含中文，所以如果要在 figure 中顯示中文，我們勢必要特別指定一個字體給它

keywords: docker、matplotlib、中文
<!--more-->

### 下載字體

首先要來下載喜歡的字體，選一個自己喜歡的就可以了，這裡我是選 google 開源的 Noto 繁中字體

[https://fonts.google.com/noto/specimen/Noto+Sans+TC](https://fonts.google.com/noto/specimen/Noto+Sans+TC)

將下載後的檔案解壓縮，選一個自己喜歡的字體組細，這邊我是選 `NotoSansTC-Medium.otf`

### 加到 matplotlib 裡面

進入到 docker container 中 (使用 vscode ssh container 或是指令 docker exec -it ... 都可以)，到 container 的根目錄中 `/` 

找到並進入以下路徑：`/opt/conda/lib/python3.7/site-packages/matplotlib`。上述路徑是 matplotlib 存放在 docker container 中的位置

接著再找到以下資料夾：`mpl-data/fonts/ttf/`，這是字體存放的地方，把剛剛下載好的 `NotoSansTC-Medium.otf` 上傳至這個資料夾中

### 修改 matplotlib 設定檔

在剛剛的 `mpl-data` 資料夾中，找到一個名稱叫：`matplotlibrc` 的設定檔，打開它

找到一下程式，把以下兩行的註解拿掉，並在 font.serif 的第一個 `,` 前加入剛剛上傳的字體名稱

```yaml
font.family:  sans-serif    # <- 拿掉註解
#font.style:   normal
#font.variant: normal
#font.weight:  normal
#font.stretch: normal
#font.size:    10.0

                # 加入上傳字體名稱
font.serif:     NotoSansTC-Medium, DejaVu Serif, Bitstream Vera Serif, Computer Modern Roman, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif    # <- 拿掉註解

#font.sans-serif: DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
```

### 修改 python 程式

接著回到程式中，我們要在建立一個 figure 物件後，調整一些字體設定，使用 `plt.rcParams` 來修改字體及字體大小

```python
# create font
fig, ax = plt.subplots()

# plt font setting
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['font.sans-serif'] = ['NotoSansTC-Medium']
```

接著就可以跑原本寫的程式啦 ~ ……嗎？

```python
plt.figure('save.png')
```

這個時候會發現…跑了上面存圖片的程式，生出來的中文字還是顯示不出來……為什麼呢？

### 加到 cache 中

原來 matplotlib 會在 `/root/.cache/matplotlib` 中新增 cache，所有的設定優先會從邊尋找，所以我們剛剛這麼大費周章的修改，結果對 matplotlib 來講跟本沒差…

所以現在我們要手動修改 cache 中的檔案，打開 `/root/.cache/matplotlib/fontlist-v330.json` 檔案

在程式的最下面新增這一些東西：

```json
    ...
		...
		{
      "fname": "fonts/ttf/NotoSansTC-Medium.otf",
      "name": "NotoSansTC-Medium",
      "style": "italic",
      "variant": "normal",
      "weight": 400,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"
    },
		{
      "fname": "/usr/share/fonts/truetype/dejavu/NotoSansTC-Medium.otf",
      "name": "NotoSansTC-Medium",
      "style": "normal",
      "variant": "normal",
      "weight": 400,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"
    }
  ],
  "__class__": "FontManager"
}
```

### 重新整理

最後最後，也是最重要也最容易忘記的一步，就是 **重開 container！**，剛剛新增了這麼多東西如果不給它重新整理一下，這個設定是不會生效的！

下 `docker container restart` 重起 container 後就大功告成啦！！！