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

### 階段 0 結果（H100，[`results/phase0_train_speed_h100`](../results/phase0_train_speed_h100/summary.md)）

**速度**（每秒張數，完整訓練步，batch 256）：

| 設定 | orig | adapt_blt |
|---|---|---|
| 原本（CPU 增強、eager） | 9,034 | 1,483 |
| GPU 增強 + 編譯 + CUDA graph | **15,494（1.7 倍）** | 6,105（4.1 倍） |
| CPU 增強 + 編譯 + CUDA graph | 11,895（受限於 CPU 增強上限 11,465） | **6,867（4.6 倍）** |

- CUDA graph 是主要的加速來源；只編譯不錄 graph 對 orig 沒有幫助。
- blt 上 GPU 增強反而比 CPU 增強慢 4.65 ms/步：GPU 增強每批把幾個常數從 CPU 複製到 GPU，每次複製都讓 CPU 等 GPU
  做完上一步，兩者無法重疊。0.15.1 已修正（常數留在 GPU 上；下一批在另一個 CUDA stream 上與訓練步同時增強）。
- 3 份並行：一旦每份都用 CUDA graph，總吞吐量不增加（5,776 vs 單份 6,105），因為沒有 MPS 時 GPU 在行程之間分時切換。
- batch 512 / 1024 的吞吐量仍明顯上升（blt 到 9,919，orig 到 33,891），但需要改學習率並重新驗證。

**驗證**：快速管線（`_FAST`）重現第 1–2 輪。`adapt_blt_fast` 每一步比第 2 輪高 0.1–0.35 點（種子標準差 0.3–0.6），
校準相同；`orig_fast` 高 0.9 點但種子差異較大（1.8），不顯著。3 個種子的牆鐘時間：blt 82.6 分鐘（原本約 300 分鐘），
orig 30.5 分鐘（原本約 50 分鐘）。**之後的新實驗改用 `_FAST`**；第 1–3 輪的實驗名稱與設定不變。

0.15.1 另外：`--parallel` 時不量 evaluate.py 的延遲（和其他執行共用 GPU，數字無效）；`--channels-last`；
`bench_train.py --profile N` 列出最耗 GPU 時間的 kernel。

### 階段 0 第 2 次量測（0.15.1）

```bash
python scripts/bench_train.py c100_vcnet_orig c100_vcnet_adapt_blt
```

```bash
python scripts/bench_train.py c100_vcnet_adapt_blt --configs gpu-compile --profile 5 --iters 20 --repeats 1 --no-data-only
```

- 第一個指令的預設組合改為：`cpu-eager`、`cpu-cg`、`gpu-cg-noprefetch`（修正常數複製、但不用另一個 stream）、
  `gpu-cg`（再加上另一個 stream）、`cpu-cg-cl` / `gpu-cg-cl`（channels_last）、batch 512 / 1024。
- 第二個指令列出 blt 訓練步中最耗 GPU 時間的 kernel（CUDA graph 內看不到個別 kernel，所以用沒有 graph 的 `gpu-compile`）。

### 階段 0 第 2 次量測結果與第 3 步（0.15.2）

[`results/phase0_train_speed_h100`](../results/phase0_train_speed_h100/summary.md)：

- 修正後 orig 用 GPU 增強 + CUDA graph 達每秒 24,387 張（原本的 2.9 倍，每個執行約 6 分鐘）。
- blt 的 GPU 增強與 CPU 增強只差 4%（6,577 vs 6,874）：blt 受 GPU 計算時間限制，不再受 CPU 限制。
- channels_last 兩者都慢 35–41%，不採用。
- blt 訓練步的 GPU 時間：每次迭代約 360 個小卷積前向、360 個反向（20–50 µs），cuDNN 在前後轉換 NCHW ↔ NHWC（約 20%）。
  單一最貴的 kernel 是 AIT → V1 雙線性上採樣的反向（Inductor 用帶原子加法的 index_put，每次 526 µs，約 9%）。
  0.15.2 在訓練時把上採樣改成兩個小矩陣乘法（數值相同；推論、FLOPs 與 checkpoint 都不變）。
- 開了 CUDA graph 後，並行多份不會增加總吞吐量（GPU 在行程間分時），**快速管線的實驗請依序跑，不要加 `--parallel`**。
- 更大的 batch 仍是最大的槓桿：驗證實驗 `c100_vcnet_adapt_blt_fast_bs512`（lr 1.4e-3）、`_bs1024`（lr 2e-3），
  學習率按 √(batch / 256) 放大，epoch 數與暖身 epoch 數不變。

```bash
python scripts/bench_train.py c100_vcnet_adapt_blt --configs cpu-cg gpu-cg gpu-cg-bs1024 --no-data-only
```

```bash
python scripts/run_experiment.py c100_vcnet_adapt_blt_fast_bs1024 c100_vcnet_adapt_blt_fast_bs512 --seeds 0 1 2
```

通過標準：與 `c100_vcnet_adapt_blt_fast`（batch 256）每一步的差距在種子標準差以內。

### 第 3 輪結果（[`results/vcnet_round3_h100`](../results/vcnet_round3_h100/summary.md)）

| 模型 | 訓練方式 | 步數 | GFLOPs | acc % |
|---|---|---|---|---|
| orig | 單一讀出、CE | 2 | 0.275 | 64.55±0.96 |
| **blt_orig2** | 單一讀出、CE | 2 | 0.284 | **67.61±0.74** |
| adapt_blt 固定 2 步（第 2 輪 / fast） | 每步讀出、Stage A | 2 | 0.284 | 67.27±0.57 / 67.62±0.44 |
| **blt_orig4** | 單一讀出、CE | 4 | 0.528 | **68.90±0.62** |
| adapt_blt 固定 4 步（第 2 輪 / fast） | 每步讀出、Stage A | 4 | 0.528 | 68.68±0.58 / 68.85±0.49 |
| adapt_blt + 停止策略 λ=0.1 / 0.03 / 0.01 | 每步讀出、Stage A | 1.73 / 2.09 / 2.41 | ≈0.251 / 0.295 / 0.334 | 67.77 / 68.20 / 68.34 |

- **增益來自架構**：同樣計算量，blt_orig2 比原始 VCNet 高 3.1 點。
- **每步讀出不拖累最終準確率**：adapt_blt 固定 2 / 4 步與 blt_orig2 / blt_orig4 相同（都在一個標準差內）。
- **自適應在 CIFAR 上的增益不大**：一個 adapt_blt 模型就能在每種深度都達到該深度專門訓練模型的準確率（原始方式的中間讀出沒有用，
  blt_orig4 第 1–3 步只有 19 / 45 / 49%）；停止策略 λ=0.1 以 88% 的計算量追平 blt_orig2，但沒有任何 λ 追上 blt_orig4。
- 延遲不必重量：blt_orig2 / 4 的計算與 adapt_blt 固定 2 / 4 步相同（batch 1 CUDA graph 1.949 / 3.882 ms）。
- 溫度縮放後，原始訓練方式的校準略好（ECE 1.5–1.6% vs 2.3–2.6%）。

### 階段 0 結論（0.15.2）

- 上採樣改成矩陣乘法後，blt 每秒 6,577 → 6,858 張（+4%）；GPU 增強與 CPU 增強差距縮到 2%。
- **更大的 batch 不採用**：學習率按 √ 放大、epoch 數不變時，batch 512 從第 2 步起每步低約 1.0 點（約 2 個標準差），
  batch 1024 低 3.4 點。訓練損失較高，原因是更新次數少（訓練不足），不是不穩定；要補回需要更高的學習率或更多 epoch，
  會吃掉大部分的速度優勢（1.2 / 1.6 倍）。
- **新實驗的預設訓練設定**：`_FAST`、batch 256、依序執行。與第 1–3 輪的設定相比（同一張閒置 GPU）：orig 快 2.9 倍
  （每個執行約 6 分鐘），adapt_blt 快 4.6 倍（每個執行約 22 分鐘，3 個種子約 65 分鐘，原本約 300 分鐘）；準確率在種子差異內重現第 1–2 輪。

## 推論加速（階段 2，0.16.0）

閒置 GPU 量測的結論是：batch 1 時自適應有效，batch 256 時停止策略讓吞吐量降到固定步數的 27–42%。
原因是壓縮式 runner（`AdaptiveRunner`）每一步都要移除已停的影像，後段 batch 越來越小，而且每步都要同步一次 CPU 與 GPU。
`main` 上為 bViT 做的**固定大小槽位池**（`fsv/halting/slots.py`）正好解決這兩點；0.16.0 讓它支援 VCNet，
並補上第 10 輪指出還沒量的**公平對照組**。

| 改動 | 做法 |
|---|---|
| VCNet 每列各自的步數 | `forward_step(state, t, x0)` 的 `t` 可為每列一個步數的 tensor：回饋、側向、由上而下各項對所有列都算，只在 t > 0 的列保留（`torch.where`，第 0 步的列忽略舊狀態）；`readout` 依每列步數取步數嵌入。與逐批計算逐位元相同（測試） |
| 槽位池支援 VCNet | VCNet 的狀態（AIT，或打包的 7 區）不是嵌入（V1），所以分成 `x0` 與狀態兩份緩衝區；補位上限預設為全部 S 個槽位（VCNet 平均 1.4–2.4 步就停，每個 tick 空出的槽位比 bViT 多） |
| 預先計算 V1 | 每張圖的 V1 在跑 tick 之前以 S 張一批算好；補位只做 gather，不必每個 tick 對全部補位候選算 V1（大多數候選不會被補上） |
| 編譯 | `SlotPoolRunner(compile=True)`：模型一步 + 讀出先以 Inductor 編譯，再錄進 tick 的 CUDA graph |
| 公平對照組 | `FixedDepthGraph`：固定 t 步時，batch 256 的整個前向（embed + 第 0..t 步 + 讀出）是一張 CUDA graph；`compile=True` 時先編譯（每個步數一個編譯圖，各深度共用） |

CPU 上槽位池與壓縮式 runner 的預測與停止步數 100% 相同（測試允許 5% 以內的差異，因為批次形狀不同時浮點捨入可能不同）。

### benchmark 新模式

`benchmark_adaptive.py` 新增兩個模式，各自帶自己的固定步數列，所以表中的加速比是同類相比：

- `graph-bs256`：固定步數 = 每批一張 CUDA graph（eager kernel）；自適應 = 槽位池 + 融合 kernel（eager kernel）。
- `graph-compiled-bs256`：兩者的模型一步都先編譯再錄成 graph。這是最強的固定步數對照組。

```bash
python scripts/benchmark_adaptive.py c100_vcnet_orig c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.003 0.01 0.03 0.1 0.3 --no-cudagraph --o3-policies --bs1-images 100
```

```bash
python scripts/summarize.py c100_vcnet_orig c100_vcnet_adapt_blt --sections bench
```

- 用第 2 輪的 `c100_vcnet_adapt_blt`（已有停止策略）；`c100_vcnet_orig` 沒有停止策略，只量固定步數，作為原始模型的對照。
- `--no-cudagraph --o3-policies --bs1-images 100`：batch 1 已在閒置 GPU 上量過，這次只看 batch 256，省時間。
- 請在沒有其他程式的 GPU 上跑（先用 `nvidia-smi` 確認）。
- 要回答的問題：在 `graph-compiled-bs256` 中，adapt_blt + 槽位池（λ=0.1，平均 1.73 步，67.77%）能否快過同準確率的固定步數（約 2 步）？
  理論上限約為固定 2 步的 1.1–1.2 倍（V1 只算一次、平均 1.73 步）；表中最後一欄 `occupancy` 是槽位使用率。

### 階段 2 第 1 次結果（[`results/vcnet_phase2_h100`](../results/vcnet_phase2_h100/summary.md)）

最強對照組 `graph-compiled-bs256`（兩邊都先編譯再錄成 CUDA graph）：

| 設定 | acc % | 平均步數 | 每秒張數 | 對同準確率固定步數 | 對固定步數混合 |
|---|---|---|---|---|---|
| 固定 2 步 | 67.33 | 2 | 105,289 | | |
| 固定 3 步 | 68.30 | 3 | 76,032 | | |
| 槽位池 λ=0.3 | 66.70 | 1.40 | 79,684 | 0.76× | 0.70× |
| 槽位池 λ=0.1 | 67.83 | 1.74 | 70,475 | 0.93× | 0.80× |
| 槽位池 λ=0.03 | 68.24 | 2.09 | 62,075 | 0.87× | 0.80× |

- 「固定步數混合」：把每張圖隨機分到相鄰的兩個固定步數，使平均準確率相同；這是停止策略要贏過的前緣。
- batch 256 時自適應仍輸（前緣的 0.70–0.80 倍），但差距大幅縮小：壓縮式 runner 只有固定步數的 27–42%，編譯後的槽位池比編譯後的壓縮式快 5.3 倍。
- 理想值（V1 一次 + 1.84 個槽位步 × 約 3.8 µs）約每張 8.8 µs、每秒 11.3 萬張，是前緣的 1.28 倍；實測每張 14.2 µs，約 5.3 µs 是額外開銷。
  明顯的嫌疑：V1 的預先計算在編譯模式下仍是 eager、不在 graph 內。0.16.1 把它編譯並錄成 graph，並輸出 V1 時間、每個 tick 的時間與 tick 數。
- 固定步數對照組本身也快很多：blt 2 步 eager 12,572 → 編譯 + graph 105,289 張/秒（8.4 倍），是原始 VCNet（115,122）的 91%，準確率高 2.8 點。

第 2 次量測（0.16.1）只量 `graph-compiled-bs256`（`--graph-only`）：

```bash
python scripts/benchmark_adaptive.py c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.01 0.03 0.1 0.3 --graph-only
```

```bash
python scripts/summarize.py c100_vcnet_adapt_blt --sections bench
```

### 階段 2 結論（0.16.1，[`results/vcnet_phase2_h100`](../results/vcnet_phase2_h100/summary.md)）

V1 預先計算改成編譯 + CUDA graph 後，槽位池快了 10–16%：

| 設定 | acc % | 每秒張數 | 對同準確率固定步數 | 對固定步數混合 |
|---|---|---|---|---|
| λ=0.3 | 66.75 | 92,653 | 0.88× | 0.82× |
| λ=0.1 | 67.78 | 79,210 | **1.05×** | 0.89× |
| λ=0.03 | 68.21 | 68,636 | 1.00× | 0.89× |
| λ=0.01 | 68.37 | 62,068 | 0.98× | 0.87× |

- 每張圖的時間（λ=0.1）：V1 2.90 µs + tick 9.75 µs（其中模型一步 7.29、tick 額外開銷 2.46 µs；每個 tick 1.336 ms，比固定步數的一步多 0.34 ms）。
- 上限：就算 tick 沒有額外開銷，λ=0.1 也只有前緣的約 1.10 倍。原因是平均只有 1.74 步、V1（每張圖的 1/4 時間）每張都要算、
  槽位中第 0 步的列也付一次帶回饋的步數成本。
- **結論**：CIFAR-100 上，batch 256 用「固定步數 + 編譯 + CUDA graph」最快；自適應槽位池落後前緣 11–18%，
  即使沒有任何額外開銷也最多快約 10%。自適應的好處在 batch 1。影像更大、每張圖步數更多時（階段 4）空間較大，屆時再量。

## 對照組（階段 1，0.17.0）：ResNet-56 與同等大小的 VCNet

驗證清單第 1 項：相同參數量與計算量下，VCNet 要勝過一般 CNN。第一個對照組是 ResNet-56（He et al. 2016 的 CIFAR 版）。

**ResNet**（`fsv/models/resnet.py`）：深度 6n + 2；3×3 stem 16 通道，三段各 n 個 basic block（16 / 32 / 64 通道，32 / 16 / 8 像素），
全域平均池化 + 一層線性。shortcut 用論文的 option A（恆等；形狀改變時取樣 + 補零通道，不含參數），
BatchNorm（`--resnet-norm gn` 可改 GroupNorm）。用同一套步驟介面、只有 1 步，所以訓練、評估、FLOPs 計算都不用改；停止曲線只有一個點。
參數量與論文相符（10 類：ResNet-20 0.27M、ResNet-56 0.85M、ResNet-110 1.7M，測試確認）。

**同等大小的 VCNet**：現有的 `blt`（全寬）是 1.15M 參數，2 步 0.284、4 步 0.528 GFLOPs，比 ResNet-56 大。
依寬度縮小，做兩個對齊版本（都是 `--lateral --topdown`）：

| 模型 | 參數 | 1 步 | 2 步 | 3 步 | 4 步 GFLOPs | 對齊什麼 |
|---|---|---|---|---|---|---|
| ResNet-56 | 0.862M | 0.252 | | | | |
| `blt`（全寬，第 2 輪） | 1.150M | 0.161 | 0.284 | 0.406 | 0.528 | |
| `vcnet_r56p` | 0.864M | 0.137 | 0.232 | 0.327 | 0.422 | 參數量（ResNet-56 的計算量落在 2 與 3 步之間） |
| `vcnet_r56f4` | 0.545M | 0.080 | 0.136 | 0.193 | 0.249 | 4 步（`blt` 最佳固定步數）的計算量 |

V1 在全解析度（32×32），每張圖佔一大塊計算量，所以寬度只要小一點，FLOPs 就降很多。

**訓練方式**：ResNet 跑兩種，VCNet 預設用 VCNet 的方式；兩者分開，才知道差距來自架構還是訓練方式。

| 實驗 | 模型 | 訓練方式 |
|---|---|---|
| `c100_resnet56_fast` | ResNet-56 | 與 `c100_vcnet_orig_fast` 完全相同，只換模型與步數：AdamW 1e-3、RandAugment、erasing、Mixup / CutMix、CE、快速管線 |
| `c100_resnet56_sgd` | ResNet-56 | 一般 CIFAR ResNet 的做法：SGD + Nesterov，batch 256 lr 0.2（= 每 128 張 0.1），wd 5e-4，cosine，只做 crop + flip，不裁梯度；用來對照文獻（CIFAR-100 約 71–72%） |
| `c100_vcnet_r56p_blt_fast` | `vcnet_r56p` | `c100_vcnet_adapt_blt_fast` 的做法（8 步、每步讀出、Stage A） |
| `c100_vcnet_r56f4_blt_fast` | `vcnet_r56f4` | 同上 |
| `c100_resnet56_gn_fast`（可選） | ResNet-56 GroupNorm | 同 `c100_resnet56_fast`；VCNet 用 GroupNorm，這個區分正規化的影響 |
| `c100_vcnet_r56p_blt_sgd`（視結果） | `vcnet_r56p` | SGD 做法；只有 `resnet56_sgd` 勝過 `resnet56_fast` 時才需要，讓兩邊都用各自最好的訓練方式 |

注意：這裡都只用 45,000 張訓練（5,000 張留做 held-out 校準），文獻用 50,000 張，ResNet-56 預期低約 0.5–1 點。

### 步驟

先量速度（ResNet 的 BatchNorm 在 CUDA graph 中會更新 running stats，要確認沒有退回非 graph 執行）：

```bash
python scripts/bench_train.py c100_resnet56_fast c100_resnet56_sgd c100_vcnet_r56p_blt_fast c100_vcnet_r56f4_blt_fast --configs gpu-cg --no-data-only
```

再依序跑 3 個種子：

```bash
python scripts/run_experiment.py c100_resnet56_sgd c100_resnet56_fast c100_vcnet_r56p_blt_fast c100_vcnet_r56f4_blt_fast --seeds 0 1 2
```

```bash
python scripts/summarize.py c100_resnet56_sgd c100_resnet56_fast c100_vcnet_r56p_blt_fast c100_vcnet_r56f4_blt_fast
```

- 訓練 log 若出現 `skipping cudagraphs`，請連同訊息一起貼上（結果仍正確，只是較慢）。
- 判讀：
  1. `resnet56_sgd` 應約 70–71%。低於 69% 表示實作或訓練方式有問題，先查這個。
  2. 參數量相同：`vcnet_r56p` 第 2 步（0.232 GFLOPs）與第 3 步（0.327）對 ResNet-56（0.252），取兩種訓練方式中 ResNet 較好的那個。
  3. 計算量相同：`vcnet_r56f4` 第 4 步對 ResNet-56（參數只有 0.63 倍）。
  4. 若 `resnet56_sgd` 較好，再跑 `c100_vcnet_r56p_blt_sgd`；若 ResNet 用 AdamW 方式明顯變差，再跑 `c100_resnet56_gn_fast` 看是不是 BatchNorm 的關係。
