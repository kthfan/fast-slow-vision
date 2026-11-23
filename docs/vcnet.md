# VCNet 實作說明

論文：Hill, Zhang & Prasetio (2025). *VCNet: Recreating High-Level Visual Cortex Principles for Robust Artificial Vision*
（arXiv 2508.02995；v3 改名 *The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet*，
NeurIPS 2025 NeurReps / CogInterp workshop）。

論文**沒有程式碼**，也沒有通道寬度、步長、正規化方式、分類頭定義、epoch 數，以及回饋迴路的接法。
本實作照論文**有寫的部分**做；**沒寫的部分**用明確的假設補上，列在下表。程式在 `fsv/models/vcnet.py`。

## 研究問題

**原始 VCNet**（固定計算量）與 **VCNet + 自適應步數**（同一架構把預測編碼迴路展開成多步、每步讀出、
以停止策略決定每張圖跑幾步）相比，在準確率、計算量、實際延遲、校準度上的差別。

## 架構

```
影像 ─► V1（3/5/7 DS conv 串接 → 側向交互 → CBAM → 神經調制）= embed，每張圖只算一次
          │
          │  z_t = V1 + W·ε_t      ε_t = ReLU(V1 − 上採樣(P(AIT_{t-1})))；t = 0 時 z_0 = V1
          ▼
   ┌─ 腹側：V2(interstripe ∥ thin) ─► V4 ◄─ MT（背側→腹側連接）─► PIT ─► CIT ─┐
   │                                                                        ├─► AIT_t ─► 讀出 p_t
   └─ 背側：V2(thick) ─► MT（遞迴區塊 ×3、CBAM、神經調制）─► MST（遞迴區塊 ×3）─┘
                                                                   │
                         AIT_t 預測 V1（top-down），進入下一步 ──────┘
```

空間解析度（CIFAR 32×32）：V1 32 → V2 16 → V4 / MT / PIT 8 → CIT / MST / AIT 4。

**一「步」= 一次預測編碼迴圈**：AIT 預測 V1 → 算誤差 ε → 修正 V1 輸入 → 重跑 V2…AIT。
V1 每張圖只算一次。MT / MST 內部的 3 次遞迴是論文原有的固定迭代，留在每一步裡面，不是自適應的對象。

| 變體 | 步數 | 讀出 | 損失 |
|---|---|---|---|
| 原始 VCNet（`c100_vcnet_orig`） | 2（由下而上 1 次 + 回饋修正 1 次） | 只在最後一步 | 只有 CE |
| 原始，不含回饋（`c100_vcnet_orig_ff`） | 1 | 最後一步 | CE |
| 自適應 VCNet（`c100_vcnet_adapt`） | 最多 8 | 每一步（共用讀出頭 + 步數嵌入） | Stage A（CE + Brier + Gatekeeper + 自我蒸餾，二次步權重） |
| 自適應 + AIT 狀態延續（`c100_vcnet_adapt_carry`） | 最多 8 | 每一步 | Stage A |

參數量（CIFAR-100）：`vcnet` 1.02M（3.9 MB，論文 full VCNet 為 3.52 MB）、`vcnet_mini` 0.07M、`vcnet_wide` 4.0M。
FLOPs（每張圖）：V1 0.053 G，第 0 步 0.108 G，之後每步 0.113 G；原始（2 步）共 0.275 G，自適應跑滿 8 步共 0.95 G。

## 論文有寫的部分（照做）

- 腹側流 V1 → V2（interstripe、thin stripe）→ V4 → PIT → CIT → AIT；背側流 V1 → V2（thick stripe）→ MT → MST；
  兩流互相連接；AIT 接收匯聚的輸入，再接分類層。
- V1：三條平行的 depthwise-separable 卷積（3×3、5×5、7×7），結果串接。
- 側向交互（Lateral Interaction）：卷積接 channel-wise self-attention，外面包殘差連接。
- CBAM 與神經調制（可學習、逐通道的乘法縮放）放在 V1、MT、V4。
- MT、MST 的遞迴區塊：共用權重的卷積迭代 t = 3 次，z_{i+1} = f(z_i) + z_0。
- 預測編碼：AIT 預測 V1 的活動，ε = ReLU(V1_bottom-up − AIT_top-down)。

## 假設（論文沒寫）

| # | 項目 | 本實作的選擇 | 理由 |
|---|---|---|---|
| A1 | 通道寬度 | V1 48、V2 64（腹側）/ 32（背側）、V4 80、PIT 80、CIT 128、AIT 160、MT 64、MST 80 | 讓參數量（3.9 MB）接近論文的 3.52 MB |
| A2 | VCNet Mini 寬度 | full 的 ×0.25 | 論文只說「縮小隱藏層寬度」 |
| A3 | 降採樣 | V2、V4/MT、CIT/MST 各以 stride-2 卷積降一半 | 論文未提；對應 32 → 4 的層級 |
| A4 | 正規化與活化函數 | GroupNorm（最多 8 組）+ ReLU | 論文未提；TODO.md 禁止每步 BatchNorm；論文的式子只出現 ReLU |
| A5 | 各區的內部結構 | 每區 1–2 個 3×3 conv-GN-ReLU；PIT、AIT 用殘差 | 論文只有區名 |
| A6 | 背側 → 腹側連接 | MT → V4（1×1 conv 相加）；MST + CIT 串接進 AIT | 論文只說兩流「在多個層級互連」與「匯聚進 AIT」 |
| A7 | channel-wise self-attention | DANet 的 CAM：A = softmax(XXᵀ/√HW)，輸出 γ·AX，γ 初始為 0 | 論文只寫名稱 |
| A8 | CBAM 超參數 | reduction 8、空間 kernel 7 | 沿用 CBAM 原論文 |
| A9 | top-down 預測器 | AIT 經 1×1 conv 後雙線性上採樣到 V1 大小 | 論文未寫 |
| A10 | **誤差 ε 如何使用** | 修正下一次由下而上的輸入：z_t = V1 + W·ε_t，W 是 1×1 conv、初始為 0 | 論文只說 ε 是「additional learning signal」。把 ε 直接當輔助損失會退化（預測值一直加大，ReLU 誤差就會變 0），所以改為往前送 |
| A11 | 原始模型的回饋次數 | 1 次（共 2 步），分類只讀最後一步 | 論文的回饋是用來「refine」表徵，至少要修正一次 |
| A12 | 分類頭 | AIT 全域平均池化 → LayerNorm → 加步數嵌入 → Linear | 與 bViT 共用同一種讀出頭 |
| A13 | 訓練設定 | 沿用本專案的 CIFAR 設定：AdamW lr 1e-3、wd 0.05、batch 256、200 epochs、RandAugment、mixup / CutMix | 論文是 Adam lr 1e-3、batch 16，未寫 epoch；兩個變體用同一套設定才公平 |
| A14 | 輸入 | 3 通道 32×32 CIFAR | 論文的資料是 Spots-10（灰階 32×32）與光場；這裡改用 CIFAR-100 |

## 實驗計畫

第一輪（每個 3 個種子）：

```bash
python scripts/run_experiment.py c100_vcnet_orig c100_vcnet_orig_ff --seeds 0 1 2
```

```bash
python scripts/run_experiment.py c100_vcnet_adapt c100_vcnet_adapt_carry --seeds 0 1 2 --halting
```

```bash
python scripts/benchmark_adaptive.py c100_vcnet_adapt c100_vcnet_adapt_carry --seeds 0 1 2
```

比較項目：

1. 原始 VCNet 的準確率 / ECE / FLOPs / 延遲，與自適應 VCNet 在**相同平均 FLOPs**（約 0.275 G，即平均 2 步）下的準確率。
2. 自適應 VCNet 的準確率–計算量曲線：固定深度、max-prob / 熵門檻、學習式停止策略（S0、O3），以及 oracle。
3. 自適應 VCNet 固定跑 2 步（t = 1）時與原始模型的差距：每一步都讀出、都算損失的代價或好處。
4. `orig_ff` 對 `orig`：回饋修正本身有沒有用。
5. `adapt_carry` 對 `adapt`：步與步之間保留 AIT 狀態，後段步數是否更有用。

## 跨步狀態（0.10.0 起，論文以外的延伸）

第 1 輪發現預測編碼迴圈修正 1 次後就飽和。原因推測是：每一步唯一的新資訊只有 V1 輸入的小修正，
V2…AIT 每步從零重算，不記得上一步。使用者同意加入跨步狀態，並可融合其他仿皮層方法。
示意圖：<https://claude.ai/artifact/8JwLgaThQ8iG12xfgeMn5X>

| 開關 | 做法 | 來源 |
|---|---|---|
| `--lateral` | 每一區 A ∈ {V2, MT, V4, PIT, CIT, MST, AIT}：A_t = F(這一步的輸入) + R_A(A_{t−1})，R_A 為 1×1 卷積 | CORnet-RT（Kubilius et al., 2019）的區內遞迴 |
| `--topdown` | 每一區再加上一層上一步的輸出：A_t += T_A(上採樣(B_{t−1}))；V2←V4、V4←PIT、PIT←CIT、CIT←AIT、MT←MST、MST←AIT | BLT 網路（Spoerer et al., 2017、2020）的 top-down 連線 |

- R、T 都初始為 0：新模型一開始和原本的迴圈完全相同，用不用、用多少由訓練決定。
- AIT→V1 的預測編碼回饋保留；兩個開關都在它之外加連線。
- 第 0 步沒有上一步，計算不變。
- 狀態包含 7 個腦區的特徵圖，打包成一個張量（`VCNet.pack` / `unpack`），所以停止策略、線上推論、CUDA graph 都不用改。
- 成本（CIFAR-100，`vcnet`）：

| 變體 | 參數 | 每步 FLOPs（第 1 步起） |
|---|---|---|
| 原本的迴圈 | 1.02M | 113.4 M |
| `--lateral` | 1.09M（+6.8%） | 119.2 M（+5.1%） |
| `--lateral --topdown` | 1.15M（+12.7%） | 122.2 M（+7.8%） |

### 第 2 輪實驗

```bash
python scripts/run_experiment.py c100_vcnet_adapt_lat c100_vcnet_adapt_blt --seeds 0 1 2 --halting
```

```bash
python scripts/benchmark_adaptive.py c100_vcnet_orig c100_vcnet_adapt_carry c100_vcnet_adapt_lat c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3
```

- benchmark 從 0.10.0 起也會量沒有停止策略的模型（固定深度），所以原始 VCNet 和自適應版能在同一個行程裡比延遲。
- λ 格點加入更小的值：第 1 輪 `adapt` 的策略在 0.001–0.3 全部停在第 0 步。

要回答的問題：

1. 第 3 步以後準確率是否繼續上升（第 1 輪在第 2 步飽和）。
2. 自適應版與原始 VCNet 在同樣準確率下的 FLOPs 與延遲。
3. `blt` 對 `lat`：由上而下連線是否再多帶來增益。

### 第 2 輪結果與第 3 輪

結果見 [`results/vcnet_round2_h100`](../results/vcnet_round2_h100/summary.md)。跨步狀態解除了飽和（`blt` 到第 4 步仍在進步），
停止策略也第一次在同一個模型內勝過固定深度。但相對原始 VCNet 的增益，大部分來自架構。

第 3 輪把「架構」和「每步讀出 + 停止策略」分開：同樣的 `blt` 架構照原始 VCNet 的方式訓練（固定步數、只在最後讀出、只用 CE）。

```bash
python scripts/run_experiment.py c100_vcnet_blt_orig2 c100_vcnet_blt_orig4 --seeds 0 1 2
```

在沒有其他程式的 GPU 上重量延遲（先用 `nvidia-smi` 確認沒有其他行程）：

```bash
python scripts/benchmark_adaptive.py c100_vcnet_orig c100_vcnet_blt_orig2 c100_vcnet_blt_orig4 c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.003 0.01 0.03 0.1 0.3
```

比較方式：

| 問題 | 比較 |
|---|---|
| 每步讀出 + 停止策略本身的價值 | `adapt_blt` + 停止策略 vs `blt_orig2`（約 0.28 GFLOPs）、`blt_orig4`（約 0.53 GFLOPs） |
| 跨步狀態本身的價值 | `blt_orig2` vs `orig`（都是固定 2 步、單一讀出） |
| 每步讀出是否拖累最終準確率 | `adapt_blt` 固定 4 步 vs `blt_orig4` |

## 訓練加速（階段 0，0.15.0）

第 1–2 輪的訓練幾乎沒用到 H100 的算力：`adapt` 每個 epoch 16.6 秒（約每秒 2,700 張），換算約 8 TFLOPs，不到 bf16 峰值的 1%。
模型太小，時間花在 CPU 發送數千個小 kernel，以及 CPU 上的資料增強。0.15.0 加入四項改動，都是**開關**，預設仍是原本的做法：

| 開關 | 做法 | 程式 |
|---|---|---|
| `--gpu-aug` | 訓練集以 uint8 放在 GPU 上，每批在 GPU 上逐張做同樣的增強（RandomCrop + Flip + RandAugment(2, 9) + Normalize + RandomErasing） | `fsv/gpu_aug.py` |
| `--compile-mode default` | `torch.compile` 整個「前向（所有步）+ Stage A 損失」，反向也一起編譯 | `fsv/train_step.py` |
| `--compile-mode reduce-overhead` | 再把前向與反向錄成 CUDA graph：每步 CPU 只發送一個 graph | 同上 |
| `--opt-impl fused` | AdamW 用單一 kernel 的 CUDA 實作 | 同上 |
| `run_experiment.py --parallel N` | N 個（實驗、種子）同時跑在同一張 GPU 上，CPU 上限切成 N 組互不重疊的核心 | `scripts/run_experiment.py`、`fsv/cpu.py` |

另外，VCNet 的 `forward`（訓練路徑）改成步與步之間直接傳各腦區的特徵圖，不再每步打包、拆包；與逐步介面的輸出完全相同（測試確認）。

**GPU 增強的正確性**：14 種 RandAugment 操作逐一與 torchvision 的 uint8 實作比對（`tests/test_train_speed.py`），
幾何操作、亮度、對比、色調分離、曝光、自動對比、直方圖等化完全相同，Color 與 Sharpness 因捨入最多差 1 個灰階；
RandomErasing 的觸發比例與平均面積也與 torchvision 相符。但亂數序列不同，也不是逐位元相同，所以要先驗證準確率。

**編譯的正確性**：同樣初始化、同樣資料，編譯後與 eager 的損失與梯度一致（CPU 上測試）。CUDA graph 只能在 GPU 上驗證。

### 步驟 1：量訓練速度

```bash
python scripts/bench_train.py c100_vcnet_orig c100_vcnet_adapt_blt
```

```bash
python scripts/bench_train.py c100_vcnet_adapt_blt --configs cpu-eager gpu-compile gpu-cg --procs 3
```

- 第一個指令：每個實驗量 8 種組合（CPU 增強 eager = 第 1–3 輪的做法、GPU 增強、fused AdamW、compile、CUDA graph、
  CPU 增強 + CUDA graph、batch 512 / 1024），以及資料管線本身的上限。每種組合先跑 15 次暖機（含編譯），再量 3 × 50 次取中位數。
- 第二個指令：每種組合同時跑 3 份（= 3 個種子並行），每秒張數是 3 份的總和。
- 表中 `min/run` 是 200 個 epoch 的估計訓練時間（不含每 10 個 epoch 一次的 held-out 評估與最後的 evaluate.py），
  `3 seeds min` 是跑完 3 個種子的牆鐘時間。
- batch 512 / 1024 只量速度；真的要用，學習率要重新調，準確率要重新驗證。

### 步驟 2：驗證快速管線不影響準確率

```bash
python scripts/run_experiment.py c100_vcnet_adapt_blt_fast c100_vcnet_orig_fast --seeds 0 1 2 --parallel 3
```

- `c100_vcnet_adapt_blt_fast` 對照第 2 輪 `c100_vcnet_adapt_blt`（固定 2 步 67.27±0.57%、4 步 68.68±0.58%），
  `c100_vcnet_orig_fast` 對照第 1 輪 `c100_vcnet_orig`（64.55±0.96%）。
- 差距在種子標準差以內，之後的新實驗就改用 `_FAST`（`fsv/experiments.py`）；舊實驗的名稱與設定不變，結果仍可重現。
- 若 `reduce-overhead` 在 GPU 上出錯，改用 `--extra "--compile-mode default"` 或回報錯誤訊息。

第 3 輪（`c100_vcnet_blt_orig2`、`c100_vcnet_blt_orig4`）仍用原本的管線，以便和 `orig`、`adapt_blt` 公平比較；
但可以並行跑，只改變牆鐘時間：

```bash
python scripts/run_experiment.py c100_vcnet_blt_orig2 c100_vcnet_blt_orig4 --seeds 0 1 2 --parallel 3
```

並行時每個行程只分到 8 / 3 ≈ 2 個 CPU 核心，CPU 增強可能成為瓶頸；`bench_train.py --procs 3` 的 `cpu-eager` 列會顯示這一點。
