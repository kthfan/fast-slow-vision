# 仿皮層遞迴網路 × Pathfinder：評估與計畫

分支：`claude/cortical-pathfinder`（自 `claude/charming-ritchie-qycdtj` 0.8.3 分出，沿用訓練、校準、停止策略與 benchmark 工具）。
主線改為**仿皮層遞迴網路**；bViT 保留為對照。調查日期：2026-11-12。

## 1. 三種仿皮層網路的開源程式與實作難度

| 模型 | 官方 / 主要程式 | 框架 | 授權 | 權重 | 核心規模 | 評估 |
|---|---|---|---|---|---|---|
| **CORnet-RT**（Kubilius et al., NeurIPS 2019） | [dicarlolab/CORnet](https://github.com/dicarlolab/CORnet) `cornet/cornet_rt.py` | PyTorch | **GPL-3.0** | ImageNet（自動下載） | 約 110 行 | **很容易**。4 個區（V1/V2/V4/IT），每區：輸入卷積 + GroupNorm + ReLU，把上一時間步的狀態加回輸入後再過 3×3 卷積；權重跨時間共用；預設 5 步。空間大小寫死為 224 px 輸入。 |
| **BLT**（Spoerer et al. 2017 / 2020） | [cjspoerer/rcnn-sat](https://github.com/cjspoerer/rcnn-sat)（原作者，2020 論文） | Keras / TF 1.13 | MIT | ImageNet、ecoset（OSF） | — | 原始程式環境已停止支援，不建議直接用。 |
| | [KietzmannLab/BLT-pytorch-CCN23](https://github.com/KietzmannLab/BLT-pytorch-CCN23) | PyTorch 1.12 | MIT | miniecoset（OSF） | 小 | **容易**。可直接參考：每層 = 由下而上卷積 + 側向卷積 + 由上而下轉置卷積，相加後正規化。 |
| | [KietzmannLab/BLT-VS](https://github.com/KietzmannLab/BLT-VS) | PyTorch | MIT | ImageNet（HuggingFace） | 數百行，8 區 | 較大的腹側流版本（Retina…LOC），可作為之後的大模型參考。 |
| **hGRU**（Linsley et al., NeurIPS 2018） | [serre-lab/hgru-pytorch](https://github.com/serre-lab/hgru-pytorch) | PyTorch（舊版） | **未標示** | 無 | 約 190 行 | **中等**。單層 hConvGRU：Gabor 初始化的 7×7 前端、9×9 水平卷積（抑制 + 興奮兩階段）、閘門；含 Pathfinder 訓練腳本。程式有幾個錯誤（`log()` 結果被丟棄、非 BN 分支用到不存在的屬性、空間大小寫死）。 |
| | [serre-lab/hgru_share](https://github.com/serre-lab/hgru_share) | TF1 | 未標示 | 有 `weights/` | — | 原始 TF 版，只作為對照。 |
| 相關 | [serre-lab/rnn_rts](https://github.com/serre-lab/rnn_rts)（Goetschalckx et al., NeurIPS 2023） | PyTorch 1.8 | MIT | — | — | hGRU + 不確定性反應時間指標（TODO.md 引用的 RT 指標）；任務是 Coco Dots 與迷宮。可直接作為 H3 的方法參考。 |

**結論：三個都在我們的程式架構內自行實作（各約 100–200 行），不直接引入外部程式。**

- CORnet-RT 是 GPL-3.0，直接複製程式碼會讓整個專案受 GPL 約束；結構很簡單，自行實作即可。
- hGRU 的 PyTorch 版沒有標示授權，而且有錯誤；依論文方程式重寫，並參考它的超參數。
- BLT 的 PyTorch 參考實作是 MIT，可以參考。
- 三者都改成與 `RecurrentViT` 相同的介面（`embed` / `forward_step` / `readout` / `features`），
  每步讀出、溫度縮放、停止策略、O3、benchmark 全部沿用。
- 正規化依 TODO.md 2.1 一律用 GroupNorm（hGRU 原版用逐時間步的 BatchNorm，這裡不採用）。
- 空間大小不寫死，支援 64 / 128 px 輸入。

## 2. Pathfinder 資料

- 生成程式：[drewlinsley/pathfinder](https://github.com/drewlinsley/pathfinder)（MIT，Junkyung Kim 2018），`snakes2.py` 為 Python 3（需 Pillow ≥ 9.1、OpenCV、SciPy）。
  - 輸出 PNG 與 `metadata/*.npy`（每列：路徑、檔名、標籤、continuity、contour_length…）。
  - 標籤：50% 機率兩個圓點在同一條蛇的兩端（正例），否則分別在兩條蛇上（負例）。
  - `contour_length` = 每條路徑的線段（paddle）數；難度由它控制（6 / 9 / 14）。
  - 已知問題：生成失敗時無限重試、某些分支有未定義變數、預設輸出路徑寫死。
- LRA 的 Pathfinder（[google-research/long-range-arena](https://github.com/google-research/long-range-arena)，已封存）：
  32 / 64 / 128 / 256 px，三個難度 `easy`（curv_baseline）/ `intermediate`（contour_length 9）/ `hard`（contour_length 14），
  各解析度的生成參數寫在 TFDS builder 的說明裡。預先生成的 `lra_release.gz` 在這個沙箱回 403（可能是代理或已下架），要在使用者機器上再試。
- **計畫：把生成器移植進專案**（MIT，保留著作權聲明），修掉已知問題，用 LRA 公布的各解析度參數生成。
  好處是可以產生任意長度（例如訓練時沒看過的 18、20），用於外推測試。

## 3. 資料集選擇：簡單、困難、第三個測試集

### 建議（主線）：同一個任務、不同難度

| 角色 | 資料集 | 用途 |
|---|---|---|
| 簡單 | Pathfinder，contour_length 6 | 訓練 + 測試 |
| 困難 | Pathfinder，contour_length 14 | 訓練 + 測試 |
| 第三（只測試） | Pathfinder，contour_length 9（內插）與 18、20（外推，訓練時沒看過） | 測試遞迴次數會不會隨難度自動增加 |

- 影像統計、任務格式、讀出頭完全相同，所以遞迴次數的差異可以直接歸因於難度，不會被「換了資料領域」混淆。
- 一個模型同時在 6 + 14 的混合資料上訓練，停止策略必須逐張分配步數：預期簡單圖少步、困難圖多步、更長的路徑更多步。
- 指標：各長度的平均停止步數、步數與路徑長度的相關、準確率–計算量曲線、外推長度上「多跑幾步」能否救回準確率（對照 AdRNN 2023）。
- 解析度：先用 64×64 快速迭代，確認可行後再上 128×128（LRA 的 Path-128 參數；每步計算量也比 CIFAR 大，有利於實際加速）。

### 替代方案：跨任務

| 角色 | 資料集 | 說明 |
|---|---|---|
| 簡單 | CIFAR-10（或 MNIST） | 一般物體辨識，預期 1–2 步就夠 |
| 困難 | Pathfinder-14 | 長距離輪廓追蹤 |
| 第三 | cABC 或 digit clutter | 見下 |

- 能展示架構的通用性，但領域不同，遞迴次數的差異無法單純歸因於難度。
- **cABC**（[drewlinsley/cabc](https://github.com/drewlinsley/cabc)，標題為 "The Cluttered NIST challenge"）：26 個字母、350×350 畫布、PNG + metadata；
  README 寫「Code still in the works」，程式用到 NumPy / SciPy 已移除的函式（`np.int`、`scipy.misc.imsave`），要移植才能跑，也沒有明確的難度參數（由字母數與重疊程度控制）。
  它測的是「由上而下」的分組；Kim et al. ICLR 2020 發現 Pathfinder 靠水平連接、cABC 靠由上而下回饋，
  正好可以比較 hGRU（水平）、BLT（含由上而下）、CORnet-RT（區內遞迴），是仿皮層主線的好題目，建議作為第二階段。
- **digit clutter**（[cjspoerer/digitclutter](https://github.com/cjspoerer/digitclutter)）：BLT 原論文的遮擋數字刺激，需要 ImageMagick。
- **迷宮**（[aks2203/easy-to-hard-data](https://github.com/aks2203/easy-to-hard-data)，MIT，pip 可裝）：9×9 到 59×59，但輸出是逐像素路徑，讀出頭要改成密集預測，暫不採用。

## 4. 下一步（待使用者確認資料集方案後）

1. 移植 Pathfinder 生成器，量生成速度，產生 64×64 的 6 / 9 / 14 / 18 / 20 長度資料。
2. 實作 CORnet-RT、BLT、hGRU（共用介面、GroupNorm、每步讀出），加入 `fsv/experiments.py`。
3. 第一輪：三個仿皮層網路 + bViT 在 Pathfinder 64（6 + 14 混合）上訓練，評估各長度的準確率與遞迴次數；停止策略研究沿用現有工具。
