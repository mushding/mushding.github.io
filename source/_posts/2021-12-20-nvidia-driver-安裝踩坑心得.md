---
title: nvidia driver、cuDNN、CUDA 安裝踩坑心得
mathjax: false
date: 2021-12-20 23:25:59
tags: 
    - nvidia driver
    - docker
categories: 雜開發心得
---

實驗室電腦 CUDA 版本各種跑掉，花了一個禮拜的時間，才上網找出下面的心得…

keywords: nvidia driver、docker
<!--more-->

### 移除舊版本的 driver
* 當時用 apt-get 安裝驅動程式
```
sudo apt-get remove --purge nvidia*
apt-get autoremove
```

* 如果是用 CUDA 來安裝的話
```
nvidia-uninstall
```

### 前置作業
* 加入顯卡 ppt
```
sudo add-apt-repository ppa:graphics-drivers
```

* 慣例的套件更新
```
sudo apt-get update
sudo apt upgrade
```

### 安裝 Nvidia driver
* 在開始前：重開機治百病
```
sudo reboot
```
* 找出目前支援的 GPU driver 版本
```
ubuntu-drivers list
```

* 安裝對應版本的 driver (2022/07/07 為 nvidia-driver-510)
```
sudo apt install nvidia-driver-VERSION_NUMBER_HERE
```

* 檢查是否安裝成功
```
nvidia-smi
```

### 安裝 CUDA
* 到下面的網址找到對應的版本
* https://developer.nvidia.com/cuda-toolkit-archive

* 照著以下選項選擇
![Image](https://i.imgur.com/BddnTAH.png)

* 照著生出來的指令一行一行 copy paste
![Image](https://i.imgur.com/seCnIG7.png)

### 安裝 cuDNN
* 先去官網下載對應的版本 (麻煩的是還要註關 nvidia 帳號，並登入…)
* https://developer.nvidia.com/rdp/cudnn-download

* 選對應的 cuDNN 版本，並選擇對應的作業系統
![Image](https://i.imgur.com/rSH7ZKg.png)

* 有桌面環境，下載完成後直接按兩下就安裝了
* 如果沒有，輸入指令
```
sudo dpkg -i YOUR_VERSION.deb
```

### container-toolkit
* 可以來在 docker container 中使用 CUDA
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
* 安裝 nvidia-docker2
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
* 重新起動 docker
```
sudo systemctl restart docker
```
* 來使用 nvidia/cuda 來測試是不是成功
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 在 docker 中使用 cuda
* 先去 dockerhub pull 下來 pytorch/pytorch 的 image
* 通常以 xxx-runtime 為主
* https://hub.docker.com/r/pytorch/pytorch/tags
* 需特別注意 cuda 及 cuDNN 的版本是否相符
```
docker pull pytorch/pytorch:...
```

### container command
* 把 image run 起來
```
docker run -itd -v /home/ipvr/mushding:/workspace --name mushding_container --gpus all pytorch/pytorch:...
```
```
參數介紹：
-d 背景執行
-it attach mode
-v 設定 volume
--name container 名字
--gpus all 重要！！！不然裡面不會有 CUDA
```
```
-v volume 是可以把你主機本地的資料，鏡像到 docker 的虛擬環境上面
分為兩個部分，由 「:」 冒號分開
左邊是本地的資料夾名稱
右邊是 docker 虛擬環境的資料夾名稱
```

### 給 docker root 權限
* 通常 docker 安裝完以後，如果想要下 docker 的指令都要搭配 sudo 才可使用
* 因為 docker 的服務基本上都是以 root 的身分在執行的，所以目前的使用者身分沒有權限去存取 docker engine

```
sudo groupadd docker
sudo usermod -aG docker $USER
```
* 最後需要退出重新登錄後才會生效

* 如果都還是不行：手動修改權限

```
sudo chmod 777 /var/run/docker.sock
```

### 如果重開機遇到 NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

* 安裝 DKMS
```
sudo apt-get install dkms
```
* 接著去通過以下方法找到對應的 nvidia 版本
* (在最後一行 nvidia-...)
```
cd /usr/src
ls
```
* 重新生成對應的 nvidia 驅動
```
sudo dkms install -m nvidia -v <你的版本路>
```
* 就成功可以下 nvidia-smi 囉

### could not select device driver "" with capabilities: [[gpu]]

首先要重新安裝 nvidia-container-runtime 或 nvidia-docker2，版本跑掉了

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y nvidia-docker2
```

如果在安裝 nvidia-docker2 出現 `Unable to locate package nvidia-docker2 when installing using apt-get`

跑以下指令：

```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
```

記得要重新起動 docker

```
sudo systemctl restart docker
```

[could not select device driver "" with capabilities: gpu](https://github.com/NVIDIA/nvidia-docker/issues/1034#issuecomment-520282450)

[出現 Unable to locate package nvidia-docker2 when installing using apt-get](https://nvidia.github.io/nvidia-docker/)