---
title: 工作技術筆記系列：Prometheus
mathjax: false
date: 2025-01-11 16:05:08
tags: 工作技術筆記
categories: 工作技術筆記
---

工作開發心得 - Prometheus

keywords: Prometheus
<!--more-->

## What is Prometheus?

![Image](https://i.imgur.com/IGrv8Gj.png)

> Prometheus collects and stores its metrics as time series data, i.e. metrics information is stored with the timestamp at which it was recorded, alongside optional key-value pairs called labels.
> 
> [Overview | Prometheus](https://prometheus.io/docs/introduction/overview/)

Prometheus 是一個 TSDB（Time Series Database）時間序列資料庫，提供以下功能：

* 蒐集來自不同 container 的 log 並建立 metrics 
* 使用 PromQL query，找出想要的 log
* 提供簡單的 WebUI 供視覺化 (也可以用 Grafana)
* 自帶監控預警系統

Prometheus 的優點：

* 使用 pull metrics 減少流量
* 資訊視覺化的監控工具
* 支援雲端服務和 container 

##  A Diagram of Prometheus

以下是 Prometheus 運作流程圖，大概可分為四個部份：

* Retrieval
* TSDB
* HTTP server
* Alertmanager

![Image](https://i.imgur.com/hC4eh3D.png)

### Prometheus server

包含 Retrieval、TSDB、HTTP server 三個物件

* Retrieval
  * 會透過 pull metrics 定期向所有 Jobs/exporters 要資料，預設是用 /metrics 這個路徑要資料。也可以是其它 prometheus server
  * 如果有一些時間週期很短的 Jobs 沒辦法等到 Prometheus 去 pull 它，可以先推送到 Pushgateway 上，再等 Prometheus 一段時間 pull 下來
  * 用 pull metrics，可以避免有太多 endpoint 同時向 Monitor 發送資料，造成 Monitor 效能不好，甚至成為 buttleneck 的問題
  * 使用 discover targets 可以自動發現 k8s 中的 log 
* TSDB
  * 將時間序列資料 (sample) 存至硬碟中
* HTTP Server
  * 使用 PromQL (Prometheus Query Language) 查找特定的 log
  * Prometheus 自己有 WebUI
  * 也可以用更精美的 Grafana 

### Alertmanager

當值超過 alert rule 設定的 threshold，alert manager 就會將訊息送出，可以透過 Email、Slack 通知

## Sample & Metric

Sample 是 Prometheus 資料存儲在 HDD 中的格式，包含三種屬性：metric, timestamp, value

```
<--------------- metric ---------------------><-timestamp -><-value->
http_request_total{status="200", method="GET"}@1434417560938 => 94355
http_request_total{status="200", method="GET"}@1434417561287 => 94334
```

Metric 是由 metric name 以及描述特徵的 labelset 兩部份組成

```
<metric name>{<label name>=<label value>, ...}

node_cpu{cpu="cpu0",mode="idle"} 362812.7890625
```

Exporter 回傳給 Prometheus 的資料格式統一包含：metric、前面的註解

```
# HELP node_cpu Seconds the cpus spent in each mode.
# TYPE node_cpu counter
node_cpu{cpu="cpu0",mode="idle"} 362812.7890625
```

註解有 HELP, TYPE 兩類

* help: 包含 metric name 以及 metric 的敘述
* type: 包含 metric name 以及 metric type (有四種)

| 欄述   | Counter           | Gauge               | Histogram/Summary          |
| ------ | ----------------- | ------------------- | -------------------------- |
| 主要功能 | 只增不減的數字    | 可增可減的數字       | 一段時間內的數值統計        |
| 主要功能 | how many times X happened? | what is the current value of X now? | how log or how big? |
| 例子   | 服務重新啟動次數  | 當前硬碟使用容量     | API request 回傳時間統計分佈 |

## Histogram vs Summary

大多數情況會使用「平均值」值為量化的標準，例如平均 API request 回傳時間。但平均容易受到極端誤差影響，例如因網路不好回傳時間 >10s，這種情況下的平均值會比較沒參考價值。

一種解決方式是「分組」，例如回傳時間 0~10ms 一組、10~100ms 一組…。

Histogram 和 Summary 都是為了解決平均值誤差而設計的「分組」指標

### Histogram

自己設計 bucket 範圍，Prometheus 只會計算 bucket 出現次數，原始資料會不見

資料包含：

* bucket 範圍，對每個點進行統計 [metric_name]_bucket{le="上邊界"}
* 計算累計和 [metric_name]_sum
* 計算次數[metric_name]_count

優點：

* 對客戶端計算低，只是一對一加總而已
* 可以做 aggregation 運算 (sum, avg…)

缺點：

* 不會存原始數值，bucket 範圍大小會影響誤差
* 伺服器端可使用 histogram_quantile 計算四分位數，但是在下 query 時計算

```
# HELP prometheus_tsdb_compaction_chunk_range Final time range of chunks on their first compaction
# TYPE prometheus_tsdb_compaction_chunk_range histogram
prometheus_tsdb_compaction_chunk_range_bucket{le="100"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="400"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="1600"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="6400"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="25600"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="102400"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="409600"} 0
prometheus_tsdb_compaction_chunk_range_bucket{le="1.6384e+06"} 260
prometheus_tsdb_compaction_chunk_range_bucket{le="6.5536e+06"} 780
prometheus_tsdb_compaction_chunk_range_bucket{le="2.62144e+07"} 780
prometheus_tsdb_compaction_chunk_range_bucket{le="+Inf"} 780
prometheus_tsdb_compaction_chunk_range_sum 1.1540798e+09
prometheus_tsdb_compaction_chunk_range_count 780
```

### Summary

在一定時間內統計資料，計算成四分位圖 (quantile) 後儲存

資料包含：

* 四分位數值 (quantiles)  [basename]{quantile="[φ]"} value 
* 計算累計和 [metric_name]_sum
* 計算次數 [metric_name]_count

優點：

* 數值精準
* query 可以直接拿到四分位值

缺點：

* 在客戶端計算四分位數，計算大
* 不能做 aggregation

```
# HELP prometheus_tsdb_wal_fsync_duration_seconds Duration of WAL fsync.
# TYPE prometheus_tsdb_wal_fsync_duration_seconds summary
prometheus_tsdb_wal_fsync_duration_seconds{quantile="0.5"} 0.012352463
prometheus_tsdb_wal_fsync_duration_seconds{quantile="0.9"} 0.014458005
prometheus_tsdb_wal_fsync_duration_seconds{quantile="0.99"} 0.017316173
prometheus_tsdb_wal_fsync_duration_seconds_sum 2.888716127000002
prometheus_tsdb_wal_fsync_duration_seconds_count 216
```

## About PromQL

PromQL 是一種 query language，是 Prometheus 專門設計來 query 時間序列資料的語法

* 取得 vector

```
http_requests_total
```

* 用 tag 去指定 {tag_name=""}

```
http_requests_total{job="apiserver", handler="/api/comments"}
```

* 一些邏輯運算符

```
http_requests_total{job!="apiserver"}
```

* 用 [] 指定時間範圍，[Querying basics | Prometheus](https://prometheus.io/docs/prometheus/latest/querying/basics/#range-vector-selectors)

```
http_requests_total{job="apiserver", handler="/api/comments"}[5m]
# 1d, 2s, 3h
```

* 用 =~ 支援 regex

```
http_requests_total{job=~".*server"}
```

* 內建很多 function 可以用，[Query functions | Prometheus](https://prometheus.io/docs/prometheus/latest/querying/functions/)

```
rate(http_requests_total[5m])
```

* 也內建很多 aggregation function，[Operators | Prometheus](https://prometheus.io/docs/prometheus/latest/querying/operators/#aggregation-operators)

```
#      tag list             matric_name/vactor 
sum by (application, group) (http_requests_total)

sum by (job) (
  rate(http_requests_total[5m])
)
```

## Deploy & Setting

設計 Exporter → Prometheus

### Python flask

開一個簡單的後端，設計 /metrics API

設計 Counter metric 每 5s 會 +1

```py
import prometheus_client
from prometheus_client import Counter
from flask import Response, Flask, jsonify

app = Flask(__name__)

total_requests = Counter('total_requests', 'Total Requests')

@app.route("/metrics")
def requests_count():
    total_requests.inc()
    return Response(prometheus_client.generate_latest(total_requests), mimetype="text/plain")

@app.route("/")
def index():
    total_requests.inc()
    return jsonify({
        'status': 'ok'
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

### prometheus.yaml

設定 prometheus 的地方

scrape_configs 裡面定義不同 job 設定，來源 ip、名稱…

```yaml
global:
  scrape_interval: 5s
  external_labels:
    monitor: 'demo-monitor'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'api_monitor'
    scrape_interval: 5s
    static_configs:
      - targets: ['web:5000']
```

### docker-compose.yaml

flask 開在 5000 port

prometheus console 開在 9090 port

```yaml
version: '3.7'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - .:/code

  prometheus:
    image: prom/prometheus:v2.48.0
    volumes:
      - ./prometheus.yaml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    ports:
      - "9090:9090"

volumes:
  prometheus_data: {}
```

### console

打開 9090 port: http://localhost:9090

進到 Prometheus 自帶的 WebUI

![Image](https://i.imgur.com/Wzqykum.png)

在上面打 PromQL 就可以查找 log

![Image](https://i.imgur.com/v0rvYKh.png)