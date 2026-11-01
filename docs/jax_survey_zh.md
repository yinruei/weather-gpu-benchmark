# JAX 導入評估：Weather GPU Benchmark

查核日期：2026-11-01（Asia/Taipei）。本文件是官方資料調查與現有程式的適用性分析，**尚未安裝或實測 JAX**，沒有 JAX/RTX 3050 加速倍數可報告。

## 適用的地方

JAX 提供接近 NumPy 的陣列介面，能透過 XLA 編譯純數值函式。`jax.jit` 適合包住這個專案的誤差、權重乘法與空間加總；編譯器有機會整合運算，減少暫存陣列與派送成本。這是需要實測的潛力，並不代表已比現有 PyTorch 快。[JAX JIT 官方說明](https://docs.jax.dev/en/latest/jit-compilation.html)

`jax.vmap` 可把單筆計算轉為批次計算，適用於起報日期、時效或成員等批次軸。不過本專案目前已用張量批次運算，共用參考場與權重；導入 `vmap` 本身不能當作新增加速的證據。[自動向量化](https://docs.jax.dev/en/latest/automatic-vectorization.html)

適合先試的核心是 `weather_pipeline.py` 的 `torch_shared`：輸入 FP32 Kelvin 預報、共享參考場、緯度權重與選用遮罩，輸出每個起報／時效／成員的 Bias、MAE、RMSE。Zarr 開檔、xarray 座標對齊與來源檢查仍是獨立的資料準備階段。依現有分階段量測，讀取與單位轉換占相當多時間；換核心函式後的完整流程收益仍需量測。

## 導入時的程式限制

JIT 適合純函式：輸入決定輸出，檔案操作、輸出紀錄與會改變的外部狀態應放在編譯區外。不同形狀／型別或不同靜態參數可能觸發新的編譯；完整 64×64 網格與 32×32 分區應分別記錄編譯成本。不要在計時迴圈內反覆建立新的 JIT 函式。[JIT 與快取](https://docs.jax.dev/en/latest/jit-compilation.html)

早期 `weather_verify.py` 的 `pack` 使用 `values[mask]`，會產生依有效資料數量改變的形狀，不適合直接搬入 JIT。可採目前共享表示法的固定形狀加 `jnp.where`，把無效誤差與權重設為零，再用有效權重和正規化；要測成員各自缺值、共同參考缺值及零有效權重，避免靜默改變驗證定義。[動態形狀與遮罩限制](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes)

## 目前這台機器的相容性

| 項目 | 實際狀態／文件要求 | 判斷 |
|---|---|---|
| 既有 CUDA Python | WSL Python 3.10.12，PyTorch 環境已能完成本專案測試 | 保留為既有基線 |
| 當前 JAX 套件 | 查核時 PyPI 為 0.12.0.1，要求 Python >=3.12 | 新版需要另外的 Python 環境 |
| 作業系統 | Windows 原生 NVIDIA GPU 無官方套件支援；WSL2 GPU 標示 experimental | GPU 試驗應在獨立 WSL 環境進行 |
| 驅動與 CUDA | 本機驅動 560.94；文件列 CUDA 12 的 Linux 基本驅動門檻 >=525，CUDA 13 >=580 | CUDA 12 是可先評估的方向；版本編號符合基本門檻不等於 WSL 組合已驗證 |

套件與 Python 要求來源：[JAX PyPI](https://pypi.org/project/jax/)。平台、驅動與安裝選項來源：[官方安裝文件](https://docs.jax.dev/en/latest/installation.html)。文件提供 `jax[cuda12]`，但亦建議未來遷往 CUDA 13；本次未更動 Python、驅動、PyTorch 或任何 CUDA 套件。

若進行試作，先確認 WSL 有合適的 Python >=3.12，再建立新的虛擬環境。安裝後第一個驗收是 `jax.devices()` 確實列出 GPU；CPU fallback 不能當成 GPU 實驗。依最終解決的套件版本鎖定依賴，不直接在現有 `weather-gpu` venv 執行升級。

## 公平測試必須拆開的成本

| 成本 | 要如何記錄 |
|---|---|
| 首次載入與編譯 | 新程序內記錄首次呼叫；明確說明持久編譯快取是否啟用 |
| 暖機後重算 | 資料預先上裝置，輸出仍須完整取回／等待完成 |
| 每次上傳 | 相同主機輸入，包含裝置傳輸與回傳 |
| 完整本機流程 | 同一份 Zarr、相同座標與單位規則，包含讀取與前處理 |
| 重複任務 | CPU、PyTorch、JAX 都允許相同的輸入快取，區分一次性成本與可攤提成本 |

JAX 非同步派送，計時必須等待結果完成，例如 `block_until_ready()`；若要與目前程式一致，也要把結果取回主機的時間算入。FP32 基線應使用明確的同型別輸入；FP64 真值繼續由 NumPy 計算，避免 JAX 預設 32-bit 讓「參考值」悄悄變成 FP32。這些成本不能靠只量函式回傳就推定。[JAX 官方 benchmarking 指南](https://docs.jax.dev/en/latest/benchmarking.html)

除了與現有 PyTorch eager 比較，也應比較編譯後的 PyTorch 核心，並分開記錄兩者的首次編譯成本。這是本專案的公平比較設計，尚未執行。每個框架都應維持相同區域、成員輸出、遮罩／面積加權、容許誤差及完整結果回傳方式。

## 記憶體與穩定性

JAX 官方文件指出，預設在首次執行時預配置約 75% GPU 記憶體，可用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 或 `XLA_PYTHON_CLIENT_MEM_FRACTION` 調整。記憶體策略會影響保留量、配置成本及碎片風險。未來比較應優先分程序執行各框架，記錄配置策略；不能把預配置／保留的容量直接當作演算法有效輸入大小。[GPU 記憶體配置](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

本專案正在測量連續與 CPU/GPU 交錯工作下的延遲分布。JAX 試驗也應保留這兩種情境，報告 P50、P95、最大值、樣本數與數值誤差；速度改善和偶發慢樣本是不同問題，編譯本身不保證消除系統層面的抖動。

## 建議的最小試作範圍

1. 在獨立 WSL Python >=3.12/CUDA 12 環境確認 GPU 可用，保存確切套件與裝置資訊。
2. 先移植單一固定形狀的加權 Bias/MAE/RMSE 純函式，以現有真實資料與缺值測試對照 NumPy FP64。
3. 同時比較 NumPy、PyTorch eager、PyTorch compiled、JAX JIT；分開首次編譯、暖機重算、上傳及完整流程。
4. 以完整區域與四分區的真實工作驗收，保留 P50/P95 和所有原始樣本，再決定是否整合正式 CLI。

目前結論：**值得做小範圍概念驗證；最可能受益的是純數值核心與批次派送。現有資料讀取、座標檢查與前處理成本仍需處理，尚無證據可以宣稱 JAX 在這個專案比 PyTorch 快。**
