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
