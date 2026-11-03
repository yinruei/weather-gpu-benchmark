# 5 分鐘專案展示與練習

這份展示只使用本 repo 的公開資料與保存結果。目標是讓聽眾理解：問題怎麼定義、如何公平比較、從證據做了什麼決定。先讀[英文案例](performance_case_study.md)與[中文圖解](physicsnemo_gather_nsight_zh.md)，再練口說。

## 展示順序

| 時間 | 畫面 | 要讓聽眾帶走的重點 |
|---|---|---|
| 0:00–0:40 | 英文案例的流程圖 | 真實資料輸入，研究完整流程效能；模型是固定未訓練 FNO |
| 0:40–1:30 | 流程圖、量測邊界 | 相同輸入、權重、輸出與容許誤差；主要時間與診斷分開 |
| 1:30–2:20 | Reader 圖上半部 | 少選 frames 仍讀相同 chunks；精確索引反而慢 |
| 2:20–3:20 | Reader 圖下半部、配對結果 | Gather 接近原版；P50 比值與配對比值不同；保留預設 |
| 3:20–4:20 | Nsight 時間線 | CPU launch、GPU kernels 與 return 等待的關係 |
| 4:20–5:00 | 英文案例的結論 | 用完整流程與正確性證據決定是否採用優化 |

以下英文可照讀；秒數是練習目標，請用錄音實際計時。較難的數字可以指著圖說，不必全部背起來。這是專案展示稿；個人分工與 AI 工具使用請依實際情況說明。

### 0:00–0:40：問題

> This project studies GPU performance in a weather data workflow. It uses public ECMWF data in Zarr format, a fixed, untrained PhysicsNeMo FNO, and an RTX 3050. The question is how much the complete workflow benefits from acceleration. The model provides a repeatable computation workload. The focus is data access, execution, and numerical agreement.

中文提示：先界定研究問題。真實氣象資料不代表模型已訓練，也不代表做了預報能力評估。

### 0:40–1:30：方法

> Each request reads new input frames, prepares FP32 arrays, runs the model, returns the full output, and writes and closes an NPZ file. CPU, GPU eager, and CUDA Graph paths use the same model state and output format. Model setup, warmup, reference calculation, and output checks are outside the primary timer. Outputs are checked against a CPU FP64 calculation using the same stored FP32 weights and inputs cast to double. Coordinates, times, and ensemble identities are checked as well.

中文提示：指著流程圖沿著箭頭說。FP64 是同一個計算的數值參考；檔案 close 邊界沒有要求 fsync。

### 1:30–2:20：第一個發現

> Three readers were compared. Bounding loads a rectangular block. Exact selects individual frames through vectorized indexing. Gather loads a basic slice, then copies the required frames with NumPy. For batch 64 with a reused dataset handle, exact reading took about thirty-nine milliseconds. Bounding and gather took about five milliseconds. All three still fetched the same four forecast chunks. Selecting fewer frames did not reduce chunk access, and the exact path had much higher read latency.

中文提示：圖中 exact 柱子較高。能確認的是相同 chunk 覆蓋與較慢的讀取；尚未把 exact 路徑的每項額外成本完全分離。

### 2:20–3:20：為什麼沒有換預設

> Two sessions retained 960 read-only observations and 2,160 complete-file requests. Methods were shuffled within repetitions, and each matched comparison used the same input variant. Gather avoided the large exact-reader regression, but it did not show a consistent workflow advantage over bounding. In the graph path, the paired bounding-to-gather median ratios were 0.972 and 1.038. One session favored bounding and the other favored gather. Gathering also creates an extra array copy. The original bounding reader therefore remains the default. The reports retain tail latency and slower results.

中文提示：比值大於 1 才是 gather 較快。兩輪各自的 P50 看似變好，也不等於同 repeat 配對結果一致；這裡沒有做顯著性檢定。

### 3:20–4:20：Nsight 告訴我們什麼

> An independent Nsight Systems capture shows the CPU launch APIs, GPU kernels, and memory copies. Eager execution issued ninety-nine kernel-launch APIs per request. Graph execution used one graph launch, while both paths still contained ninety-nine kernel events. The CPU return interval overlaps preceding GPU work, so that interval includes waiting and cannot be interpreted as transfer cost alone. Two initial captures lacked device activity and remain marked partial. The complete capture contains 2,376 kernel events. Its diagnostic timings are kept separate from the primary benchmark.

中文提示：橙色 CPU forward 縮短，不代表灰色 GPU 工作同幅度縮短。先指出每個 panel 的橫軸不同，再說紫色 return 的重疊。

### 4:20–5:00：工程決定

> The main outcome is a validated way to make performance decisions. It connects source data, complete output files, numerical checks, and device traces. The measurements apply to this local dataset, fixed workload, and hardware. Caches were warmed, and model setup was measured separately. Larger gains would need a new, controlled experiment. For this experiment, the evidence supports keeping the simpler default and closing the reader comparison.

中文提示：收在「如何做決定」，不需要把每個實驗都說成成功加速。若追問下一個優化，先說明欲改善的邊界、測法與採用條件。

## 可實際執行的展示

在 repo 根目錄執行。這會使用目前 Python 的 NumPy／xarray／Zarr／Matplotlib，重驗保存的 GPU 輸出、原始量測 CSV、Store probes 與 Nsight 選定事件，再產生圖表與摘要。這個展示不需要 CUDA、PhysicsNeMo 或 Nsight；不會重跑 GPU 量測或重算模型的 FP64 forward。

WSL 可使用既有環境：

```bash
cd ~/weather-gpu-benchmark-github
source ~/venvs/weather-gpu/bin/activate
python scripts/demo_performance_case.py --output results/latest/demo01
```

目前 Windows workspace 已有環境時：

```powershell
.\.venv\Scripts\python.exe scripts/demo_performance_case.py --output results/latest/demo01
```

乾淨的 CPU 環境先安裝 repo 的 `requirements-zarr.txt`。每次選新的輸出路徑，例如下一次用 `results/latest/demo02`。已有路徑會拒絕覆寫；驗證失敗會停止，沒有完成摘要。失敗目錄保留供檢查。

成功後開啟：

```text
results/latest/demo01/overview.md
results/latest/demo01/summary.json
results/latest/demo01/gather/analysis.md
results/latest/demo01/nsight/analysis.md
```

`overview.md` 有兩張重建圖與來自 CSV 的摘要。`summary.json` 方便核對筆數、每輪 P50、配對比值與每次 Nsight 的完整／部分狀態。讀取到檔案的正式 benchmark 重現方法仍在[原始報告](../results/rtx3050_physics_gather_comparison/analysis.md)。

實際展示時先開好流程圖與兩張圖，再執行指令。驗證花費的時間是這台機器的證據重驗時間，請把它與保存的 GPU 毫秒數分別說明。若現場時間有限，先前產出的圖與報告也可用於講解。

## 第一次練習：45 分鐘

| 時間 | 要做的事 | 自己驗收 |
|---|---|---|
| 10 分鐘 | 執行展示，開啟 overview 與兩張圖 | 說出資料、模型、硬體、量測起訖 |
| 10 分鐘 | 關掉稿，用中文解釋 reader 圖 | 說出「相同 chunks」「額外 copy」「配對比值」 |
| 10 分鐘 | 對著圖錄一遍英文 | 記下實際總長與卡住的三句，改短再錄 |
| 15 分鐘 | 回答[技術問答](technical_interview_zh.md)第 2、4、6、8 題，再打開程式 | 每題能指到實際函式或保存證據 |

個人經驗、分工與工具使用可以另外準備；這份公開稿只放專案證據。先能用自己的話解釋，再把它放入履歷或面試回答。
