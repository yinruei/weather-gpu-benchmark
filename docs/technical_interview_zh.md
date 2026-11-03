# 這個專案的技術問答

配合[5 分鐘展示](project_demo_zh.md)使用。每題先用自己的話答，再看程式或保存結果。英文短答是練習用句型；描述的是本 repo 的實作與證據，不代表其他模型、資料集或硬體也有相同結果。

## 1. 這個專案解決什麼問題？

研究真實 Zarr 資料進入固定模型，到完整輸出檔關閉的效能。它把 reader、CPU／GPU eager／CUDA Graph 與數值驗證放在同一套可核對的流程。模型是未訓練 FNO，作為固定計算負載；這裡的正確性是計算與資料身分一致。

> It measures a complete weather data workflow, from local Zarr input to a closed output file. A fixed, untrained FNO provides the computation workload.

證據：[案例](performance_case_study.md)、[FNO 設定與 prepare_input](../physicsnemo_benchmark.py)。

## 2. 完整流程究竟計時到哪裡？

從每次 read/select/decode 與來源座標檢查開始，包含 FP32 packing、host tensor／輸出 metadata、GPU upload、forward、完整 host return、NPZ write/close。CPU 路徑沒有 upload。模型建立、graph 建立與暖機、FP64 參考、檔案 readback／hash／數值驗證及 cleanup 在主要計時外。完整流程有來源檢查，寫完檔的檢查則在計時外。Close 沒有要求 fsync，不能據此宣稱資料已持久寫入媒體。

> The timer includes reading, preparation, execution, full return, and file close. Setup and output verification are outside the primary timer.

實作：[FnoFilePipeline.execute](../physicsnemo_graph_pipeline.py)、[write_output 與 verify_output](../physicsnemo_pipeline.py)。

## 3. 怎麼避免測到不同的工作？

固定來源 hash、模型權重、原生格點、轉換、FP32 與輸出 schema。A/B 是兩段真實且不重疊的 frames；每個 repeat 的各方法使用相同 variant，每個 request 都重新讀取與 packing。方法和 case 的順序按 seed 混排。GPU eager 與 graph 用固定輸入 buffer；每次更新內容並完整取回輸出。這降低比較上的混淆，仍不能控制所有作業系統、cache 與硬體狀態。

> Matched requests use the same genuine input variant, model state, and output contract. Method order is shuffled, and each request reads and prepares its input again.

實作：[request_order、read_variant 與 execute](../physicsnemo_graph_pipeline.py)；證據：[正式結果](../results/rtx3050_physics_gather_comparison/analysis.md)。

## 4. 為什麼 GPU 沒有讓完整流程快很多？

GPU forward 只是一部分。資料開啟、索引、decode、packing、傳輸、等待、metadata 與寫檔都會影響請求時間。這個實驗還顯示，reader 換成較慢的 exact 路徑會拖慢整體；graph 減少 launch APIs，卻沒有消除讀取與實際 GPU kernels。應用完整流程數據判斷改善，再以獨立 profile 找原因。

> Faster dispatch improves only part of the request. Data access, preparation, device execution, return, and writing still contribute to latency.

追問練習：假設一個**純串行、可相加**的工作只有 40% 可加速，該部分變快 10 倍，總加速為 `1 / (0.6 + 0.4/10) = 1.5625` 倍。這是說明上限的假設例子；本 repo 的 CPU／GPU 重疊區段不能直接套成已量到的比例。

## 5. 為什麼少選 frames 反而較慢？

邏輯選取量和 Zarr chunk 覆蓋不同。B64 的 bounding、exact、gather 在此都取同樣四個 forecast chunks。Exact 回傳較小陣列，但透過 xarray/Zarr vectorized indexers；觀測到其完整 read/select/decode/check 路徑約 39 ms，原版與 gather 約 5 ms。能確認的是相同 Store 覆蓋與整段讀取延遲差異；尚未把 exact 的索引、後端存取與配置等各項成本逐一隔離。

> Fewer selected frames did not mean fewer fetched chunks. The exact path returned a smaller array but had much higher read latency in this dataset.

實作：[read_exact_from_dataset、read_gather_from_dataset、read_coverage](../physicsnemo_reader.py)；證據：[chunk 與 byte 表](../results/rtx3050_physics_gather_comparison/analysis.md)。

## 6. Gather 已經比較快，為什麼不改預設？

Gather 明顯避開 exact 的回歸，但相對原版沒有穩定整體收益。B64 graph 的配對 bounding/gather 中位數兩輪為 0.972×／1.038×，gather 較快 11/30／17/30。還有額外 copy，其他引擎也有變慢結果。這組證據支持保留原版；沒有宣稱 gather 永遠沒用。

> Gather avoids the exact-reader regression, but its complete-workflow gain over bounding is inconsistent. The evidence supports keeping the current default.

證據：[全部配對、P95 與最大值](../results/rtx3050_physics_gather_comparison/analysis.md)。

## 7. P50 都下降，怎麼配對比值還可能小於 1？

`median(A) / median(B)` 與 `median(A_i/B_i)` 是不同統計量。前者各自排序後取中位數；後者保留同 repeat／variant 的對應。舉例，**以下是假設資料**：`A=[2,100,101]`、`B=[3,1,102]`，各自中位數比值是 `100/3`；逐對比值的中位數卻約 `0.990`。因此報告同時保留個別分位數與配對摘要。配對統計仍是描述，沒有取代顯著性檢定或更完整的系統狀態控制。

> A ratio of medians is different from the median of paired ratios. Pairing preserves the relationship between matched requests.

實作：[paired 與報表](../scripts/analyze_physicsnemo_reader.py)、[gather 報表](../scripts/analyze_physicsnemo_gather.py)。

## 8. Nsight 的 return 很長，代表 PCIe 慢嗎？

不能只看 CPU range 下結論。程式在 `result.cpu().numpy()` 取得完整結果；時間線顯示這段 CPU 呼叫與先前 GPU kernels 重疊，包含等待運算與資料取回。應對照 CUDA runtime correlation、GPU kernels 與實際 memcpy events。這輪沒有把 return 全段當成獨立傳輸成本，也沒有把 CPU range 與 GPU 活動重複相加。

> The return call can wait for earlier GPU work. Its CPU duration is not the transfer time alone.

實作：[execute](../physicsnemo_graph_pipeline.py)、[summarize_events](../nsight_trace.py)；證據：[Nsight 圖與限制](../results/rtx3050_physics_nsight_comparison/analysis.md)。

## 9. CUDA Graph 到底改善了什麼？

這個 runner 先暖機並 capture 固定 shape、固定 input/output storage 的模型 forward。後續把新值 copy 進既有 input，再 replay。完整捕捉中 eager 每次 99 個 kernel-launch APIs，graph 每次一個 graph launch，但兩者仍各有 99 個 kernel events。證據是派送方式改變；沒有證明算術量減少、所有 kernels 融合或 GPU occupancy 提高。Graph setup 成本另外保存。

> Graph replay changes dispatch in this workload. Both paths still execute the model kernels, and graph setup is recorded separately.

實作：[CudaGraphRunner](../physicsnemo_graph.py)。追問：runner 的 output storage 會在下次 replay 覆寫；若要跨請求保存結果，必須管理它的生命週期。本流程每次完整取回並寫檔後才處理下一請求。

## 10. FP64 參考、FP32 與 FP16 要怎麼說？

這輪模型與輸入是 FP32；數值參考是把保存的同一組 FP32 權重與輸入轉成 double，在 CPU 執行。它不會恢復先前 FP32 表示已丟掉的資訊。逐元素條件為 `abs(y-ref) <= 1e-5 + 1e-4*abs(ref)`，另外保留最大絕對誤差、RMSE 與最大 tolerance fraction。這裡的 RMSE 是兩種數值計算結果的差異。這輪沒有對 FNO 做 FP16 比較，也沒有據此聲稱精度取捨。

> FP64 is a numerical reference for the same stored weights and inputs. The model execution here is FP32, with an explicit elementwise tolerance.

實作：[prepare_input、double_reference、error_summary](../physicsnemo_benchmark.py)。

## 11. 回傳陣列較小，就比較省 RAM／VRAM 嗎？

要先說清楚是哪種 memory。B64 gather 的回傳陣列為 1 MiB，但和約 1.594 MiB 來源 slice 在索引時短暫共存，邏輯合計約 2.594 MiB。這不含 decoder、packing、allocator 或程序其他記憶體。它是 CPU 陣列帳，不是程序 peak RAM 或 GPU VRAM。GPU allocator probes 也有自己的配置與資源共存邊界，不能直接等同顯示卡總用量。

> A smaller returned array does not establish a lower process peak. The source slice and gathered copy briefly coexist.

證據：[陣列帳與範圍](../results/rtx3050_physics_gather_comparison/analysis.md)。

## 12. 沒有 GPU 的 CI 究竟驗證了什麼？

CI 執行 CPU 測試，並從保存資料重驗來源選取、順序、量測 CSV／分位數、輸出檔 hash／陣列／metadata／FP64 agreement、Store probes 與選定 Nsight 事件。它能檢查保存證據是否自洽。實際 GPU 測量、模型 FP64 forward 的獨立重算，以及 Nsight 擷取在原 GPU 環境完成；CPU CI 沒有重新執行它們。示範指令沿用同一組證據驗證器。

> CPU CI validates archived outputs and traces. It does not rerun the GPU benchmark or collect new device activity.

實作：[CI](../.github/workflows/test.yml)、[展示入口](../scripts/demo_performance_case.py)。

## 13. 為什麼保留失敗的 Nsight 擷取？

第一、二次只有 CPU NVTX／CUDA API，缺 GPU kernels／copy tables，明確標為 partial。第三次用已記錄的 WSL 時戳設定得到實際 GPU 活動，再透過 correlation IDs 對回請求並核對全表數量。公開選定事件，原始含程序環境的報告留在本機。這讓讀者能分辨嘗試過什麼，以及哪份證據真的支持 GPU 時間線。

> Partial captures remain explicitly partial. Only a capture with validated device activity supports the GPU timeline.

證據：[三次擷取狀態與時戳精度限制](../results/rtx3050_physics_nsight_comparison/analysis.md)。

## 14. 下一個優化要怎麼選？

先指定使用情境與欲改善指標。例如要研究讀取，就控制相同資料內容並比較 chunk layout 的讀取成本與完整請求；要研究重疊，就驗證獨立 buffer、正確依賴、輸出身分與實際時間線，並分開報告吞吐與單次延遲。先寫驗收條件：同輸出／容許誤差、完整流程有可重現收益、tail latency 和記憶體在可接受範圍。這些是未來實驗提案，目前尚未實作或取得成效。

> I would define the target workload and acceptance criteria first. A new optimization needs repeatable workflow gains and the same correctness contract.

## 自己動手的三個檢查

1. 打開 `FnoFilePipeline.execute`，逐行指出計時開始、CPU／GPU 分支、full return 與 file close，再找出外部做 readback 的位置。
2. 用 Python 讀兩輪 `timing_samples.csv`，依同 repeat／variant 配對 `graph_open_once` 和 `graph_gather_once`。重算約 0.972／1.038 的中位數比值與 11／17 個 gather-faster pairs；與展示摘要核對。
3. 開啟 Nsight 圖，用 60 秒說明 CPU forward、CPU return、GPU kernels 與 copies。再說出兩個圖不能支持的推論，以及還需要什麼證據。

若被問到 AI 協作，照實說明使用工具、自己能解釋與驗證的部分，並以以上實作練習支持回答。若某段程式目前還不能解釋，把它當成下一個練習題；個人能力與分工敘述需反映實際情況。
