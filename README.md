# fast-slow-vision

方向 1「校準式快慢視覺」的實作。完整規格見 [`TODO.md`](TODO.md)。

目前進度：**bViT 式單區塊遞迴 ViT + 每步讀出**（Stage A）、**每步溫度縮放**（Stage B）、
免訓練停止基準（max-prob / 熵門檻、固定深度、oracle），以及 FLOPs / 延遲量測與每步特徵快取（Stage C 的前置）。

## 開發流程

實驗在使用者自己的機器上跑（單張 H100）。使用者的環境不能 push 或上傳檔案，所以結果以**貼在對話裡的文字摘要**回傳：

1. Claude 在分支 `claude/charming-ritchie-qycdtj` 上寫程式，並在 `fsv/experiments.py` 登記要跑的實驗。
2. 使用者更新程式，用 `scripts/run_experiment.py` 跑實驗。
3. 使用者執行 `python scripts/summarize.py`，把輸出（`BEGIN`…`END` 之間）整段貼到對話裡。
4. Claude 分析並把結果存進 `results/`（repo 內的存檔），決定下一輪實驗。

---

## 使用者操作步驟

### 0. 需求

- Linux，NVIDIA driver 支援 CUDA 12.x（`nvidia-smi` 右上角的 CUDA Version ≥ 12.6；若為 13.x 也可改用 `cu130` 的 wheel）
- Python 3.10 以上
- 能連到 `github.com` 與 `huggingface.co`（第一次執行會下載 CIFAR-100，約 140 MB）

### 1. 取得或更新程式

每一輪開始前都要更新到分支的最新版本。report 裡的 `code` 欄位（例如 `0.2.0+4533fb7e05f9`）會記錄實際使用的程式版本，不需要 git 也能比對。

**方式 A：有 git**（第一次 clone，之後只要 pull）

```bash
git clone -b claude/charming-ritchie-qycdtj https://github.com/kthfan/fast-slow-vision.git
```

```bash
cd fast-slow-vision && git pull origin claude/charming-ritchie-qycdtj
```

**方式 B：沒有 git**（下載 zip 後覆蓋到原目錄；`data/`、`runs/`、`reports/`、`.venv/` 會保留）

在 `fast-slow-vision` 的**上一層目錄**執行：

```bash
curl -L -o fsv.zip https://github.com/kthfan/fast-slow-vision/archive/refs/heads/claude/charming-ritchie-qycdtj.zip
```

```bash
unzip -q -o fsv.zip && cp -r fast-slow-vision-claude-charming-ritchie-qycdtj/. fast-slow-vision/ && rm -rf fsv.zip fast-slow-vision-claude-charming-ritchie-qycdtj
```

### 2. 建立環境（第一次）

```bash
cd fast-slow-vision
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

先裝 CUDA 版 PyTorch（H100 用 CUDA 12.x 的 wheel），再裝本專案（editable 安裝，之後更新程式不必重裝）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -e ".[test]"
```

### 3. 設定 CPU 上限

所有程式都會把自己（含 DataLoader worker）綁在最多 N 個 CPU 核心上，預設 N = 8。
要改的話，在每次開新 shell 後設定環境變數（或在每個指令加 `--max-cpus N`）：

```bash
export FSV_MAX_CPUS=8
```

限制方式：Linux CPU affinity（子行程繼承，硬性上限）+ PyTorch / OpenMP 執行緒數 + DataLoader worker 數 = N − 1。

### 4. 檢查環境（每次更新程式後）

```bash
pytest -q
```

```bash
python scripts/check_env.py
```

`pytest` 約 1 分鐘、應顯示全部通過。`check_env.py` 最後一行應為 `OK: environment ready`，並顯示 `GPU NVIDIA H100 ...`、`AMP bf16`。

### 5. 跑實驗

列出所有登記的實驗：

```bash
python scripts/run_experiment.py --list
```

一個指令可以跑多個實驗、多個種子（依序執行）。種子 0 的結果放在 `reports/<實驗名>/`，其他種子放在 `reports/<實驗名>_s<種子>/`：

```bash
python scripts/run_experiment.py c100_bvit_s --seeds 0 1 2
```

要在登出後繼續執行，用 `nohup` 並把輸出存到 `run_all.log`：

```bash
nohup python scripts/run_experiment.py c100_bvit_s c100_vit_s_8blk --seeds 1 2 > run_all.log 2>&1 &
```

```bash
tail -f run_all.log
```

- 中斷後重跑同一個指令，會從 `runs/<名稱>/last.pt` 接續訓練；要從頭來加 `--fresh`。
- 指定 GPU：`--gpu 1`（預設 0）。
- 同時跑多個（實驗、種子）：`--parallel 3`（0.15.0 起）。小模型只用到 GPU 的一小部分，3 個種子並行約等於跑 1 個的時間；
  CPU 上限會切成互不重疊的幾組核心，輸出每行前面標上執行名稱。
- `runs/` 存放 checkpoint（大檔）、`reports/` 存放每次執行的報告；兩者都只在本機，不在 repo 裡，更新程式不會覆蓋。

### 6. 回報結果

```bash
python scripts/summarize.py
```

把輸出從 `===== BEGIN FSV SUMMARY =====` 到 `===== END FSV SUMMARY =====` 整段貼到對話裡即可。
同一個實驗的多個種子會自動合併成「平均±標準差」。只想回報部分實驗時，把實驗名當參數：
`python scripts/summarize.py c100_bvit_b`。

若 Claude 需要更細的資料，會指定要貼 `reports/<名稱>/` 底下的哪個檔案。

### Report 內容（`reports/<名稱>/`）

| 檔案 | 內容 |
|---|---|
| `report.md` | 單次執行的完整摘要：環境、訓練曲線、每步測試指標、停止曲線 AUC、oracle、延遲 |
| `results.json` | 完整評估結果（每步指標、溫度、FLOPs、停止曲線各點、延遲） |
| `history.jsonl` | 每個 epoch 的訓練損失、速度與 held-out 每步準確率 |
| `config.json` | 訓練參數、模型設定、參數量、AMP |
| `env.json` | 程式版本、GPU、torch、CPU 上限、種子、執行時間、完整指令 |
| `train.log`, `eval.log` | 完整輸出 |

---

## 程式結構

| 路徑 | 內容 |
|---|---|
| `fsv/models/bvit.py` | `RecurrentViT`：共用 Transformer 區塊遞迴 `t_max` 次；共用讀出頭 `head(LN(CLS_t) + step_emb(t))`；`n_blocks=t_max` 時為不共用權重的 ViT 對照組 |
| `fsv/models/vcnet.py` | `VCNet`：仿視覺皮層 CNN（Hill et al., 2025），把 AIT→V1 預測編碼迴路展開成步數、每步讀出；設計與假設見 [`docs/vcnet.md`](docs/vcnet.md) |
| `fsv/models/resnet.py` | `ResNet`：CIFAR ResNet-20 / 56 / 110（He et al., 2016，option A shortcut），階段 1 的 CNN 對照組；同一步驟介面、只有 1 步 |
| `fsv/losses.py` | Stage A 損失：每步 CE + β·Brier + γ·Gatekeeper 信心損失 + δ·最後一步自我蒸餾 |
| `fsv/calibration.py` | 每步溫度縮放（held-out 上以 LBFGS 擬合） |
| `fsv/metrics.py` | acc、ECE、Brier、NLL、AURC |
| `fsv/flops.py` | FLOPs（含每步讀出頭）：ViT 用解析式，VCNet 用 `FlopCounterMode` 實測 |
| `fsv/halting/thresholds.py` | max-prob / 熵門檻曲線、固定深度早退、oracle 停止步數 t*、曲線下面積 |
| `fsv/engine.py` | 收集每步 logits / 特徵；逐步延遲量測；AMP 自動選擇（H100 用 bf16） |
| `fsv/cpu.py` | CPU 核心數硬性上限；`FSV_CPU_SET` 指定核心（並行執行時各用各的） |
| `fsv/gpu_aug.py` | 訓練資料放在 GPU 上、逐張在 GPU 上做同樣的增強（`--gpu-aug`） |
| `fsv/train_step.py` | 一次訓練步（前向 + 損失可 `torch.compile` / CUDA graph，fused AdamW） |
| `fsv/data.py` | CIFAR-10/100（Hugging Face parquet，快取為 npz）；從訓練集切出固定 5,000 張 held-out |
| `fsv/experiments.py` | 實驗登記表（每個名稱對應固定的訓練設定） |
| `fsv/report.py` | 產生 `report.md` |
| `fsv/version.py` | 程式版本（版本號 + 原始碼雜湊，不需要 git） |
| `scripts/run_experiment.py` | 一鍵：訓練 → 評估 → report（可多實驗、多種子） |
| `scripts/summarize.py` | 輸出可貼到對話的精簡摘要（多種子取平均±標準差） |
| `scripts/train.py`, `scripts/evaluate.py` | 底層訓練 / 評估程式（也可單獨使用，見 `--help`） |
| `scripts/check_env.py` | 環境檢查 |
| `scripts/bench_train.py` | 訓練速度量測（各種資料管線 / 編譯 / batch 組合，`--procs N` 並行） |
| `scripts/halting_study.py` | TODO-1 停止策略研究（在快取上離線訓練與評估；`run_experiment.py --halting` 會自動呼叫） |
| `fsv/halting/features.py`, `fsv/halting/policy.py` | 停止策略的輸入特徵（S0 / S2 / S3 / S5）、MLP、訓練目標、λ 掃描曲線 |
| `fsv/halting/online.py`, `scripts/benchmark_adaptive.py` | 線上自適應推論（逐步停止 + batch compaction）與實際延遲 / 吞吐量量測（TODO-4） |
| `scripts/make_report.py` | 由 JSON 重新產生 `report.md` |
| `results/` | 已回報結果的存檔（`reports/` 則是使用者本機輸出，不進 repo） |

## 與 bViT 論文（Byra et al., 2026）的差異

論文沒有公開程式碼，依論文描述自行實作：pre-norm 區塊、GELU、無 drop path、可選 bViT-TE（每步輸入時間嵌入）。
為了 CIFAR（32×32）與本專案需求做了以下調整：

- patch size 4（64 個 patch + CLS），`t_max` 預設 8（論文 ImageNet 為 12）。
- 預設 LayerNorm（TODO.md 2.1 定案；論文用 RMSNorm，可用 `--norm rmsnorm`）。
- 每一步都讀出並計算損失（論文只在最後一步讀出）。

## 實驗紀錄

CIFAR-100，T_max = 8，200 epochs，測試集，H100 bf16（另註明者除外）。
AUC = 在 [第 0 步, 第 7 步] GFLOPs 區間的平均 top-1（溫度縮放後機率）。多種子為平均±標準差。

| 實驗 | 種子 | 參數 | 最佳步 top-1 | 最後一步 | ECE TS（最後一步） | AUC max-prob / 熵 / 固定深度 | oracle any |
|---|---|---|---|---|---|---|---|
| [`kaggle_c100_bvit_s`](results/kaggle_c100_bvit_s/report.md)（2×T4, **fp16**） | 0 | 1.86M | 61.34（t=4） | 60.66 | 2.48 | 56.49 / 56.10 / 58.38 | 64.44 @ 0.37 G |
| `c100_bvit_s` | 0,1,2 | 1.86M | 59.34±0.07（t=4） | 58.85±0.31 | 3.18±0.20 | 54.43 / 54.05 / 56.58 | 62.38 @ 0.36 G |
| [`c100_bvit_s_lightaug`](results/round1_h100/c100_bvit_s_lightaug.md) | 0 | 1.86M | 56.95（t=4） | 56.49 | 1.63 | 53.50 / 52.98 / 54.93 | 61.75 @ 0.35 G |
| `c100_bvit_b`（dim 768，4× FLOPs） | 0 | 7.26M | 58.03（t=3） | 57.42 | 3.24 | 54.51 / 54.17 / 56.38 | 61.78 @ 1.30 G |
| `c100_vit_s_8blk`（不共用權重） | 0,1,2 | 14.28M | 62.63±0.69（t=7） | 62.62±0.71 | 3.26±0.32 | 57.67 / 57.11 / 58.30 | 66.40 @ 0.42 G |
| `c100_bvit_s_fp16`（fp16 訓練） | 0 | 1.86M | 59.47（t=4） | 58.49 | 3.74 | 54.42 / 54.03 / 56.71 | 62.23 @ 0.36 G |
| `c100_bvit_s_te`（時間嵌入） | 0 | 1.86M | 59.61（t=5） | 59.19 | 3.69 | 54.85 / 54.44 / 56.77 | 62.80 @ 0.36 G |
| `c100_bvit_s_inject`（輸入重新注入） | 0 | 1.86M | 59.75（t=4） | 59.24 | 3.05 | 54.94 / 54.50 / 56.92 | 62.78 @ 0.36 G |
| `c100_bvit_s_nosd`（無自我蒸餾） | 0 | 1.86M | 59.49（t=4） | 58.97 | 2.99 | 54.50 / 54.10 / 56.39 | 62.37 @ 0.37 G |
| `c100_bvit_s_linw`（線性步權重） | 0,1,2 | 1.86M | 60.97±1.58（t=5） | 60.55±1.60 | 2.87±0.30 | 55.15 / 54.68 / 57.07 | 63.61 @ 0.39 G |
| `c100_bvit_s_quadw`（平方步權重） | 0,1,2 | 1.86M | 61.46±1.10（t=5） | 61.07±0.98 | 3.19±0.24 | 55.03 / 54.49 / 56.87 | 63.93 @ 0.40 G |
| `c100_bvit_s_linw_lr5e4`（線性步權重，lr 5e-4） | 0,1,2 | 1.86M | 61.25±0.36（t=5） | 60.90±0.34 | 3.06±0.05 | 55.50 / 55.05 / 57.35 | 64.04 @ 0.39 G |
| `c100_bvit_s_lr5e4`（均等權重，lr 5e-4；對照） | 0,1,2 | 1.86M | 60.09±0.84（t=4） | 59.64±1.02 | 3.20±0.42 | 55.26 / 54.86 / 57.17 | 63.17 @ 0.36 G |
| **`c100_bvit_s_quadw_lr5e4`（平方步權重，lr 5e-4）** | 0,1,2 | 1.86M | **62.14±0.31**（t=5） | **61.93±0.24** | 3.13±0.18 | 55.83 / 55.33 / 57.54 | 64.76 @ 0.41 G |
| `c100_bvit_s_linw_te_inject` | 0 | 1.86M | 61.71（t=5） | 61.29 | 2.89 | 55.90 / 55.36 / 57.79 | 64.22 @ 0.39 G |
| `c100_bvit_s_mix2`（2 組 mixup） | 0 | 1.86M | 59.67（t=4） | 59.03 | 3.97 | 54.97 / 54.59 / 56.92 | 62.68 @ 0.36 G |
| `c100_vit_s_8blk_linw` | 0 | 14.28M | 62.78（t=7） | 62.78 | 2.47 | 56.58 / 55.89 / 57.28 | 66.11 @ 0.45 G |

原始資料：第 1 輪 [`results/round1_h100/`](results/round1_h100/)（使用者貼上的 `report.md`），
第 2 輪 [`results/round2_h100/summary.md`](results/round2_h100/summary.md)，
第 3 輪 [`results/round3_h100/summary.md`](results/round3_h100/summary.md)，
第 4 輪 [`results/round4_h100/summary.md`](results/round4_h100/summary.md)，
第 5 輪 [`results/round5_h100/summary.md`](results/round5_h100/summary.md)，
第 6 輪 [`results/round6_h100/summary.md`](results/round6_h100/summary.md)，
第 7 輪 [`results/round7_h100/summary.md`](results/round7_h100/summary.md)，
第 8 輪 [`results/round8_h100/summary.md`](results/round8_h100/summary.md)，
第 9 輪 [`results/round9_h100/summary.md`](results/round9_h100/summary.md)。

觀察（第 1–6 輪）：

1. **種子變異很小**（bViT-S 最佳步 ±0.07），所以 Kaggle（61.3）與 H100（59.3）的 2 個百分點差距是系統性的。
   第 3 輪排除了精度（fp16 59.47 vs bf16 59.34），第 4 輪排除了 mixup 分組（59.67）。
   在 CPU 上重新評估 Kaggle 的 checkpoint 得到相同結果，所以差異來自訓練；剩下的差異是 torch 2.11 vs 2.6（含 torchvision），暫不追查。
2. **減輕增強反而更差**：過擬合，「增強太強導致欠擬合」不成立；但它的溫度約 1.1，表示基準線的「信心不足」主要來自 mixup / CutMix。
3. **共用權重的模型在第 3–4 步就飽和**，加寬（bViT-B）也一樣，而且 4 倍計算量反而少 1.3 個百分點；
   不共用的 8 層 ViT 一路進步到第 7 步（+3.3）。
   **第 3 輪找到原因：每步損失權重。** 改成線性權重（w_t ∝ t+1）後最佳步 62.14（+2.8），
   用 1/7.7 的參數追上不共用 ViT；代價是第 0 步從 33 降到 27，但準確率–計算量 AUC 仍全面提升。
   時間嵌入（+0.3）、輸入注入（+0.4）、拿掉自我蒸餾（+0.15）的效果都小，且只有一個種子。
   **第 4 輪：線性權重的效果不穩定**——三個種子 60.97±1.58（推算約 62.1 / 61.6 / 59.2），其中一個種子幾乎沒有進步；
   平方權重 62.08（1 個種子）；疊加時間嵌入與輸入注入沒有額外好處（61.71）；
   不共用的 8 層 ViT 用線性權重沒有差別（62.78 vs 62.62±0.71），所以偏重後段只對共用權重有效。
   **第 5 輪：學習率降到 5e-4 讓線性權重穩定下來**（61.25±0.36，標準差從 1.58 降到 0.36）；
   平方權重平均略高（61.46）但在 1e-3 下同樣不穩定（±1.10）。
   **第 6 輪：目前最佳的共用模型是平方步權重 + lr 5e-4**（62.14±0.31，最後一步 61.93±0.24），
   距不共用的 8 層 ViT（62.63±0.69）只差 0.5，參數 1/7.7；NLL（1.453 vs 1.526）與 AURC（14.87 vs 15.10）反而更好。
   對照組拆解（皆 lr 5e-4）：學習率本身讓均等權重 +0.75（59.34 → 60.09），平方權重再 +2.05（→ 62.14）。
4. **免訓練門檻規則在所有模型上都輸給固定深度**；oracle 顯示逐樣本停止約 1.5 步即可達到最後一步的準確率，學習式停止（TODO-1）空間很大。
5. 延遲量測在 bs=256 的標準差很大（GPU 可能同時有其他工作），自 0.3.0 起改為取多次量測的中位數，並記錄最慢/最快比值。

### 學習式停止策略（TODO-1，第一版）

`scripts/halting_study.py`：骨幹凍結，在快取的每步輸出上離線訓練停止策略（2 層 MLP，hidden 64）。
策略在 held-out 5k 上訓練（與溫度縮放同一份），在測試集評估。網格：

- 特徵：S0（機率統計量：top-5、熵、差距、top-1 是否改變、KL(p_t‖p_{t-1})、top-1 連續不變的步數、步數 one-hot）；
  S3（S0 + CLS 與 pooled 特徵的收斂訊號：與前一步的 cos、相對變化量、與第 0 步的 cos）。
- 目標：`reward`（λ 條件化，最小化 E_π[Brier + λ(t+1)/T]，因為每步輸出都已快取，期望值可精確計算，不需要 RL 取樣）；
  `reward_acc`（同上，以 0/1 錯誤取代 Brier）；`oracle_full` / `oracle_any`（以 t* 為標籤的監督式 BCE）。
- 推論：第一個 sigmoid(logit) > 0.5 的步停止；`reward*` 掃描 λ ∈ [1e-4, 10]，監督式掃描門檻。
- AUC 以「全部在第 0 步停」為左端錨點：任何方法都能以隨機混合達到錨點與其最便宜點之間的連線；
  修正前把最便宜點往左平延，會高估無法到達低計算量的方法（只影響新的停止策略，門檻與固定深度曲線本來就含第 0 步）。

Kaggle checkpoint（[`results/kaggle_c100_bvit_s/halting.json`](results/kaggle_c100_bvit_s/halting.json)，3 個策略種子）：

| 方法 | AUC % | acc @ 第 1 步計算量 | acc @ 第 2 步 | acc @ 第 3 步 |
|---|---|---|---|---|
| 固定深度 | 58.38 | 55.92 | 60.16 | 60.88 |
| max-prob 門檻 | 56.50 | 49.89 | 55.75 | 58.60 |
| 熵門檻 | 56.11 | 48.72 | 54.86 | 57.89 |
| **S0-reward_acc** | **58.80±0.04** | **56.89** | **60.76** | 60.88 |
| S0-reward（Brier） | 58.53±0.04 | 56.50 | 60.59 | 60.88 |
| S0-oracle_any | 58.79±0.06 | 56.67 | 60.74 | 60.95 |
| S0-oracle_full | 56.63±0.06 | 50.27 | 56.06 | 58.80 |
| S3-reward_acc | 58.73±0.05 | 56.66 | 60.72 | 60.90 |
| oracle any（用到標籤） | 64.46% @ 0.365 GFLOPs | | | |

- 學習式策略比門檻規則好約 2.3 AUC、比固定深度好約 0.4；同計算量下最多 +1.0 個百分點。
- λ 掃描完全單調（0/40 次違反），符合 TODO-3 的要求；策略成本為單步遞迴的 0.006%（預算 2%）。
- S3 的收斂特徵沒有幫助；`oracle_full` 標籤把「最後一步也錯」的樣本送到最後一步，浪費計算量，明顯較差。
- 與 oracle 的差距仍大：oracle 的優勢主要來自讓「反正會錯」的樣本在第 0 步就停，現有特徵很難辨認這些樣本。

第 5 輪，所有 H100 骨幹（各 3 個骨幹種子 × 3 個策略種子；AUC %）：

| 骨幹 | 固定深度 | max-prob | S0-reward_acc | S0-oracle_any | 策略 − 固定深度 | oracle any |
|---|---|---|---|---|---|---|
| `c100_bvit_s` | 56.58±0.06 | 54.43 | 57.12±0.06 | 56.84 | +0.54 | 62.38 @ 0.36 G |
| `c100_bvit_s_linw` | 57.07±1.21 | 55.15 | 57.71±1.31 | 57.71 | +0.64 | 63.61 @ 0.39 G |
| `c100_bvit_s_linw_lr5e4` | 57.35±0.30 | 55.50 | **58.02±0.32** | 57.95 | +0.67 | 64.04 @ 0.39 G |
| `c100_bvit_s_quadw` | 56.87±0.75 | 55.03 | 57.62±0.85 | 57.47 | +0.75 | 63.93 @ 0.40 G |
| `c100_vit_s_8blk` | 58.30±0.39 | 57.67 | 59.39±0.47 | **59.44** | +1.09 | 66.40 @ 0.42 G |

- 五個骨幹上學習式策略都勝過固定深度（+0.5～+1.1）與門檻規則（+2～+3）；λ 掃描全部單調。
- `reward_acc` 一致優於 Brier 版 `reward`；S3 收斂特徵一致沒有幫助。不共用的 ViT 從停止策略得到的好處最大。

策略的訓練資料（Kaggle checkpoint，1 個策略種子，[`results/kaggle_c100_bvit_s/halting_policy_train.json`](results/kaggle_c100_bvit_s/halting_policy_train.json)；AUC %）：

| 特徵 | 在 held-out 4k 上訓練 | 在訓練集 20k 張的輸出上訓練（held-out 做模型選擇） |
|---|---|---|
| S0 | 58.78 | 59.07 |
| S2（CLS 原始特徵 384 維，MLP 第一層即學習式降維 = S6） | 54.80 | 58.59 |
| S5（固定隨機投影 384→64） | 58.52 | **59.11** |

- S2 在 held-out 上變差是過擬合（4k 張、384 維輸入）；改用 5 倍資料後回到 58.6。
- 用訓練集圖片的輸出訓練策略比只用 held-out 好（S0 +0.3），而且能用到高計算量那一端；分布差異沒有造成問題。
- 排除過擬合後，表徵特徵相對 S0 只有很小的好處（S5 +0.04）。

第 6 輪在四個 H100 骨幹上檢驗（3 個骨幹種子；AUC %）：

| 骨幹 | 固定深度 | S0-reward_acc（held-out） | S0-reward_acc@train | S5-reward_acc@train |
|---|---|---|---|---|
| `c100_bvit_s_lr5e4` | 57.17 | 57.80 | 57.89 | 57.79 |
| `c100_bvit_s_linw_lr5e4` | 57.35 | 58.02 | 58.03 | 57.86 |
| `c100_bvit_s_quadw_lr5e4` | 57.54 | 58.26 | **58.34** | 58.10 |
| `c100_vit_s_8blk` | 58.30 | **59.39** | 59.11 | 59.05 |

- 用訓練集圖片的輸出訓練策略，在 H100 骨幹上沒有一致的好處（−0.28～+0.09）；Kaggle checkpoint 上的 +0.3 沒有重現。
- S5 在四個骨幹上都略差於 S0。結論：在 CIFAR 規模下，S0（只用機率統計量）在 held-out 上訓練就夠了；表徵特徵沒有帶來好處。

### 實際延遲與吞吐量（TODO-4，0.7.0 起）

- `fsv/halting/online.py`：`AdaptiveRunner` 逐步執行模型、增量計算 S0 特徵（與離線版逐位元相同，有測試），
  每一步把已停止的影像移出 batch（compaction），後段步數只處理剩下的影像。
- `scripts/benchmark_adaptive.py`：在**同一個行程**內連續量測多個模型（共用 GPU 時，不同時段的延遲不可比），
  測試集預先載入 GPU；量測固定深度 t = 0…7 與多個 λ 的 batch 256 吞吐量、batch 1 延遲、準確率與平均步數。
- `halting_study.py` 會把每種策略（種子 0）存成 `reports/<名稱>/policies/<策略>.pt`，並新增
  `cost@full` / `cost@best`：達到全深度 / 最佳固定深度準確率所需的計算量（全深度計算量的 %）。

第 7 輪（3 個骨幹種子）：

**計算量（FLOPs）**：達到全深度準確率所需的計算量（全深度的 %）

| 骨幹 | 固定深度 | max-prob 門檻 | S0-reward_acc |
|---|---|---|---|
| `c100_bvit_s_lr5e4` | 48.6±6.4 | 83.2±3.9 | 41.3±11.6 |
| `c100_bvit_s_quadw_lr5e4` | 62.0±8.4 | 83.4±6.9 | **43.2±0.2** |
| `c100_vit_s_8blk` | 95.3±8.1 | 72.4±2.4 | **53.8±2.5** |

**實際時間：第一版線上推論器在 GPU 上比固定深度慢 2–4 倍**（batch 256 與 batch 1 皆然）。
例：`c100_bvit_s_quadw_lr5e4`、λ = 0.03：準確率 61.95%、平均 3.70 步，batch 256 吞吐量 37,770 img/s，
同準確率的固定深度 t=4（61.97%，5 步）為 102,911 img/s；batch 1 延遲 4.35 ms vs 1.36 ms。
原因：每一步約 9 次 GPU→CPU 同步（每個布林遮罩索引都會同步），讓 kernel 啟動延遲完全暴露；
batch 256 時每步額外開銷約 0.7–1.2 ms，大於一個遞迴步本身（約 0.49 ms）。
0.7.1 的 `AdaptiveRunner.run` 改為每步只同步一次（一次 `nonzero`，其餘用 `index_select` / `index_copy_`），
並提供 `torch.compile` 版本（每步模型呼叫與特徵 + 策略計算都編譯；固定深度用同一個編譯過的模型步，比較公平）；
舊版保留為 `run_naive`，benchmark 用 `--naive` 一併量測。

第 8 輪（`c100_bvit_s_quadw_lr5e4`，λ = 0.03：61.95%、3.70 步；同準確率的固定深度 t=4 為 61.97%、5 步）：

| 版本 | batch 256 吞吐量 | batch 1 延遲 |
|---|---|---|
| 固定深度 t=4（eager） | 98,931 img/s | 1.71 ms |
| 自適應，第一版（eager-naive） | 37,647 | 4.37 ms |
| 自適應，每步一次同步（eager） | 45,112 | 3.59 ms |
| 自適應，+ torch.compile | 52,315 | 3.02 ms |

- 改善了 20–40%，但仍比同準確率的固定深度慢（0.5–0.65 倍）。8 層 ViT compile 後 batch 1 為 0.90 倍。
- 根本原因：只要每步同步一次，GPU 佇列就清空，下一步約 55 個小 kernel 的 CPU 發送延遲（每個約 10 µs）全部暴露；
  這個模型每步本身很短（batch 256 約 0.5 ms），發送延遲與計算同一量級。

**O3（TODO-1d，0.8.0）**：在第 k 步一次預測每張圖的停止步數（`StepMLP`，輸入第 0…k 步的 S0 特徵與 λ，
以精確期望獎勵訓練）。整個 batch 只同步一次；之後依預測步數由深到淺排序，第 t 步只算 batch 的前 n_t 張（n_t 已知，不需同步）。
`halting_study.py --o3-ks 0 1` 離線評估並存檔，`O3Runner` 線上執行（與離線決策逐位元相同，有測試），
benchmark 以 `o3k0-eager`、`o3k1-compiled` 等模式量測。

Kaggle checkpoint 的離線結果（AUC %，2 個策略種子）：固定深度 58.38、O1 S0-reward_acc 58.80、
O3 k=0 58.37、k=1 58.19、k=2 56.83；S5 特徵的 O3 更差（過擬合）。提早決定會失去 O1 大部分的 FLOPs 優勢；
O3 的價值要看它能否把剩下的優勢轉成實際時間。

**batch 1 + CUDA Graphs（0.8.1）**：`CudaGraphBS1` 把每一步的計算錄成 CUDA graph，去除 kernel 發送開銷。
固定深度 = 整個前向一張圖；O1 = 每步一張圖 + 每步讀一次停止旗標；O3 = 兩張圖 + 讀一次深度。
benchmark 預設量測（`cudagraph-bs1`、`o3k*-cudagraph-bs1`；`--no-cudagraph` 關閉），只報延遲，
準確率沿用 eager 模式（決策相同，並記錄 `agree_with_eager` 檢查）。錄製期間 autocast 關閉權重快取（`cache_enabled=False`）。

### TODO-4 在 CIFAR 上的結論（第 7–9 輪）

同準確率下對固定深度的實際速度比（> 1 才是真的加速；3 個骨幹種子，c100_bvit_b 為 1 個；λ = 0.03 附近的最佳設定）：

| 骨幹（每步計算量） | batch 256，O1 | batch 256，O3 | batch 1，CUDA graphs，O1 | batch 1，CUDA graphs，O3 |
|---|---|---|---|---|
| `c100_bvit_s_quadw_lr5e4`（0.24 GFLOPs） | 0.48× | 0.83×（k=0） | 0.69× | 0.91×（k=0） |
| `c100_vit_s_8blk`（0.24 GFLOPs） | 0.48× | 0.74×（k=0） | 0.88× | 0.89×（k=1） |
| `c100_bvit_b`（0.93 GFLOPs） | 0.89× | 0.97～1.08×（k=0/1） | 0.74× | 0.85×（k=1） |

- CUDA graphs 讓 batch 1 的所有方法快 3–4 倍（固定深度跑滿 8 步 2.8 → 0.71 ms），但自適應仍比同準確率的固定深度慢。
- O1 在 FLOPs 上達到全深度準確率只需 43–54% 的計算量，但每步一次的同步 + 策略 kernel 抵銷了節省；
  O3 只同步一次，實際時間較好，但它的 FLOPs 優勢幾乎為零（AUC ≈ 固定深度），所以最多打平。
- 每步計算量越大，額外開銷的佔比越小：bViT-S → bViT-B 時 batch 256 的 O1 從 0.48× 升到 0.89×、O3 首次略超過 1×。
- 結論：在 32×32 輸入、單步 < 1 GFLOPs 的設定下，H100 上無法把 FLOPs 節省轉成實際加速；
  實際延遲的主張需要每步計算量更大的設定（224×224 輸入、更寬的模型）或受計算量限制的硬體（CPU / 邊緣裝置）。

### CPU 與 GPU 迴圈並行（0.9.0）

> T4 量測（`results/t4_*`）只保留圖；原始 JSON 與 log 已移除以縮小 repo，數字以本 README 的表格為準。H100 的原始輸出在 `results/round10_h100/`。


> 版本說明：本節到「槽位池補位改進」的 0.9.0–0.13.1 是 `claude/admiring-franklin-6zlsn0` 分支上的編號，與 VCNet 的 0.9.0–0.10.1 平行發展；rebase 到 main 後合併為 0.14.0。

> **0.12.0 起已移除這部分程式碼**（量測結果保留於此與 `results/`）：`run_pipelined`、benchmark 的 `*-pipelined` 模式與 `compile_decide`（`cudagraph-bs1-fused`）；後者已由 Triton 融合 kernel 取代。`bench_sync.py` 的不同步下限（`run_known_depth`）保留。


想法：GPU 一路往下跑、不等 CPU；CPU 只在旁邊輪詢結果，決定停止後才停掉 GPU 迴圈。實作：

- `AdaptiveRunner.run_pipelined(x, lam, lookahead)`（batch）與 `CudaGraphBS1.run_pipelined`（batch 1，CUDA graphs）：
  GPU 每步把「仍在跑的影像數 / 停止旗標」非同步複製到 pinned 記憶體並記錄 event；CPU 用不阻塞的 `event.query()` 讀已到達的結果，
  只有在超前最後一個已知結果 `lookahead` 步時才等待。停止決策在 GPU 上鎖存（latch），所以多跑的步不會改變答案；
  batch 版的壓縮（compaction）延後 `lookahead` 步。`lookahead=0` 等同原本每步同步一次。決策與同步版逐位元相同（有測試，含 GPU）。
- benchmark 新模式：`eager-pipelined`、`compiled-pipelined`、`cudagraph-bs1-pipelined`（`--lookahead`，預設 1；`--no-pipelined` 關閉），
  以及 `cudagraph-bs1-fused`（每步的特徵 + 策略 kernel 用 `torch.compile` 融合後錄進同一張 graph）。
- `scripts/bench_sync.py`：不需 checkpoint 的合成測試（隨機權重、隨機策略，平均步數調到約 3.5），並量測
  「**已知停止步、完全不讀回**」的下限：同步版減掉它，就是同步本身的成本；任何把策略留在 GPU 上的作法
  （pipelining、CUDA graph 條件節點的 GPU 端 while 迴圈）都不可能比這個下限更快。

Kaggle T4、torch 2.6.0+cu124（batch 1 CUDA graphs，ms/張）：

| 骨幹 | 固定 t=3（4 步） | O1 每步同步 | O1 pipelined L=1 | L=2 | 不同步下限 | 融合 + 同步 | 融合下限 |
|---|---|---|---|---|---|---|---|
| bViT-S（平均 3.52 步） | 0.622 | 0.988 | 1.126 | 1.296 | 0.917 | **0.822** | 0.752 |
| bViT-B（平均 3.34 步） | 1.522 | 1.720 | 2.122 | 2.572 | 1.616 | **1.545** | 1.457 |

batch 256（eager）：bViT-S 11,989 → 10,439 img/s（L=1），bViT-B 3,720 → 3,120 img/s。

- **同步本身只佔一小部分**：同步版比不同步下限只慢 0.07–0.10 ms（約 7%）。O1 比固定深度慢，主因是每步約 40 個小的特徵 + 策略 kernel
  （batch 1 每步約 0.1 ms，與一個遞迴步同量級），不是 CPU–GPU 往返。
- **pipelining 反而更慢**：CUDA stream 無法取消已排入的 kernel，CPU 看到「停」時，已超前排入的 `lookahead` 步仍會跑完，
  而且排在下一張圖前面；多算一步（0.15–0.4 ms）遠大於省下的同步（每步約 0.02 ms）。batch 256 時壓縮延後也讓已停的影像多算。
- 要真正「GPU 自己迴圈、停了就不再算」需要 CUDA graph 條件節點（while / if node）；torch 2.6 沒有這個 API，
  而且它的極限就是上表的「不同步下限」，在 T4 上只比現在快約 7%。
- **有效的方向是把策略變便宜**：`compile_decide=True` 融合特徵 + 策略後，bViT-S 0.988 → 0.822 ms、bViT-B 1.720 → 1.545 ms，
  效果比去掉同步還大。H100 的單步更快、同步佔比可能較高，數字需在 H100 上用 `bench_sync.py` / benchmark 的新模式確認。

### O1 與固定深度的逐部分耗時（0.9.0，`scripts/profile_breakdown.py`）

把兩種方法拆成相同的部分（embed、model_step、readout、features、policy、record、sync、compaction），
每部分量 CPU 時間（不開 profiler）、GPU kernel 時間（torch.profiler）、kernel 數；CUDA graph 模式把每部分各錄成一張 graph，
以 event 計時。Kaggle T4、torch 2.6.0+cu124、隨機權重（平均約 3.5 步）。
與「固定 4 步」比較，O1 多出的時間（µs/張；batch 256 為 µs/batch）：

| 設定 | O1 牆鐘 | 固定 4 步 | 差 | 少算的遞迴步 | features + policy + record | 同步 / GPU 閒置 |
|---|---|---|---|---|---|---|
| bViT-S，batch 1，CUDA graph | 1,085 | 625 | +460 | −43（GPU） | **+296（GPU）** | +187 |
| bViT-B，batch 1，CUDA graph | 1,759 | 1,527 | +232 | −244（GPU） | **+296（GPU）** | +200 |
| bViT-S，batch 1，eager | 7,746 | 4,461 | +3,285 | −420（CPU） | **+3,259（CPU 發送）** | +326（sync） |
| bViT-S，batch 256，eager | 20,654 | 22,279 | −1,625 | −7,560（GPU，壓縮） | +950（GPU） | **+3,900 GPU 閒置** |

- **CUDA graph，batch 1**：每步 graph 245 µs = 模型步 152 + 特徵 63 + 策略 28；特徵 + 策略的 GPU 時間（約 90 µs/步）是最大的額外成本，
  同步造成的 GPU 閒置約 50 µs/步（每張 190–200 µs）。
- **eager，batch 1**：完全受 CPU 發送限制（GPU 閒置 70–78%）。O1 每張多發 ~125 個小 kernel（features 78、policy 23、record 24），
  每個約 10–20 µs 的 CPU 時間；sync 本身只有 326 µs（4%）。
- **eager，batch 256**：O1 靠壓縮少算 7.6 ms 的模型步，比固定 4 步快 7%；但比固定 3 步（16,801）慢 3.9 ms，差距幾乎全是 GPU 閒置：
  每步同步後 GPU 要等 CPU 重新發送，而後段步驟只剩少數影像，GPU 工作量小於 CPU 發送時間。bViT-B 的閒置只佔 4%。
  （bViT-B 固定深度的 profiler GPU 時間比牆鐘大 9–15%，是 T4 降頻造成的量測誤差。）

### 特徵 + 策略融合成單一 kernel（0.10.0，`fsv/halting/fused.py`）

O1 每步的特徵與策略原本約 40 個小 kernel（softmax、log、top-5、熵、KL、cat、正規化、兩層 Linear、GELU、比較、狀態複製）。
`FusedDecide` 用一個 Triton kernel、每張影像一個 program 完成：校準機率 → S0 特徵 → 策略 MLP → 寫回 stop / pred / logp / run。
輸出可以與輸入共用記憶體，所以 CUDA graph 的狀態緩衝區直接原地更新。用法：`AdaptiveRunner(..., fused=True)`、`CudaGraphBS1(..., fused=True)`；
benchmark 模式 `eager-triton`、`cudagraph-bs1-triton`。測試：在 Triton CPU 直譯器（`TRITON_INTERPRET=1`）上與 PyTorch 逐項相同；
T4 上所有計時影像的決策 100% 一致。

Kaggle T4、torch 2.6.0+cu124、隨機權重（圖在 `results/t4_fused/plots/`）：

| | 每步特徵 + 策略的 GPU 時間 | O1 每步 graph | bViT-S bs1 graph | bViT-B bs1 graph | bViT-S bs1 eager | bViT-B bs1 eager |
|---|---|---|---|---|---|---|
| 原本 | 63 + 28 = 91 µs（約 40 個 kernel） | 246 µs | 1,072 µs | 1,752 µs | 7,537 µs | 7,262 µs |
| 融合 | **22 µs**（1 個 kernel） | **160 µs** | **784 µs** | **1,456 µs** | **5,478 µs** | **5,193 µs** |
| 固定 4 步 | — | 155 µs（只有模型步） | 620 µs | 1,527 µs | 4,133 µs | 4,152 µs |

（O1 平均 bViT-S 3.69 步、bViT-B 3.31 步；三次 T4 執行的數字相差 < 3%。）

- 融合後每步 graph 只比純模型步多約 5 µs；batch 1 CUDA graph 下 O1 快 17–27%，bViT-B 已比固定 4 步快約 5%。
- eager batch 1 快 27–28%：每張圖少發約 75 個 kernel。
- batch 256 不變（±4%，在 T4 降頻的量測誤差內）：那裡的瓶頸是同步後的 GPU 閒置，不是這些 kernel。
- 時間圖：`python scripts/profile_breakdown.py --out X.json` 後 `python scripts/plot_timeline.py X.json`，
  畫出單次呼叫的 CPU / GPU 時間線（每個 kernel 依發出它的部分著色）與每部分耗時長條圖。

### 兩個 batch 交錯執行（0.11.0，`AdaptiveRunner.run_interleaved`）

> **0.12.0 起已移除這部分程式碼**（量測結果保留於此與 `results/`）：`run_interleaved`、`*-interleaved*` 模式與 `profile_breakdown.py` 的兩個 batch 時間線。下一步改為固定槽位池。


k 個 batch 同時在途、輪流推進：CPU 等 A 的第 t 步結果時，GPU 執行已排入的 B 的 kernel；有 batch 完成就補下一個。
停止數量改用非同步讀回（pinned 記憶體 + event），不用 `nonzero`（它會等佇列裡所有工作，包含另一個 batch 的）。
沒有推測性的多算，結果與 `run` 逐位元相同（有測試）。benchmark 模式 `eager-interleaved2`、`triton-interleaved2`（`--inflight`）。

Kaggle T4、batch 256、隨機權重（每 batch，不開 profiler 的 CPU 時間；圖在 `results/t4_interleaved/plots/`）：

| 骨幹 / 方法 | 實際耗時 | CPU 發送 kernel | CPU 等同步 | GPU 閒置 | kernel 數 |
|---|---|---|---|---|---|
| bViT-S 固定 4 步 | 22.25 ms | 9.87 | 0 | 0% | 118 |
| bViT-S O1 | 21.32 ms | 12.51 | 7.94 | 20% | 536 |
| bViT-S O1，2 個在途 | 21.80 ms | **17.99** | **2.51** | 19% | 571 |
| bViT-S O1 融合，2 個在途 | **20.77 ms** | 13.06 | 6.51 | 7% | 359 |
| bViT-B O1 | 55.86 ms | 13.31 | 41.67 | 5% | 446 |
| bViT-B O1，2 個在途 | 54.22 ms | 14.85 | 38.65 | 1% | 476 |
| bViT-B O1 融合，2 個在途 | **53.70 ms** | 10.77 | 42.13 | 0% | 295 |

- 交錯本身有效：bViT-S 的同步等待從 7.9 降到 2.5 ms，bViT-B 的 GPU 閒置從 5% 降到 1%；3 個在途與 2 個相同。
- 但實際耗時幾乎不變（bViT-B −3%、bViT-S ±2%，都在 T4 約 ±5% 的量測誤差內）。
- 原因（bViT-S）：瓶頸是 CPU 發送 kernel，不是同步。CPU 不再等待後，省下的時間全花在變慢的發送上
  （GPU 忙碌時每次發送更慢：12.5 → 18.0 ms），CPU 總時間（發送 + 等待）兩種做法都約 20.5 ms，GPU 仍在等 CPU。
  時間圖 `timeline_bvit_s_eager-bs256x2.png` 可見 CPU 線全程忙碌。兩個 batch 同步前進，後段步驟同時只剩少數影像，交錯無法用前段的大工作量填補。
- 要再進一步，必須減少發送次數：融合 kernel（536 → 359 個）已把閒置降到 7%；下一步是 CUDA graph（依大小分級）或固定槽位池，讓每步只發送一次。

### 固定大小的槽位池（0.13.0，`fsv/halting/slots.py`）

`SlotPoolRunner`：S 個槽位同時在途，每個槽位各自在自己的步數；每個 tick 依序做補位（空槽位從 GPU 上的佇列取下一張圖，
裝置端指標，不經 CPU）→ 所有槽位走一步 → 停止決策（第 T−1 步強制停止）→ 記錄已停止者並釋放槽位。形狀固定，整個 tick 是一張
CUDA graph；CPU 只重播 graph，並晚一個 tick 非同步讀「已完成張數」。模型改動：`forward_step` / `readout` 接受每列的步數 tensor
（只支援 `n_blocks == 1`，權重跨步共用）；融合 kernel 新增每列模式。測試：與 `AdaptiveRunner.run` 的預測與停止步數逐位元相同（CPU、
Triton 直譯器），GPU 上 100% 一致。benchmark 模式 `slots`、`slots-triton`（`--slots`）。

Kaggle T4、隨機權重、10,240 張圖（吞吐量 img/s）：

| | bViT-S | bViT-B |
|---|---|---|
| 固定 4 步（batch 256） | 11,519 | 3,116 |
| O1（壓縮，每步同步） | **11,715** | **3,778** |
| 槽位池 S=256（補位上限 R=256） | 8,391 | 3,016 |
| 槽位池 S=256，R=128 | 9,218 | 3,285 |
| 槽位池 S=512，R=128，融合 | 7,711 | 2,934 |

- 目標本身達成：GPU 閒置約 0%，槽位使用率 96%（剩下 4% 是佇列用完後的收尾），決策 100% 一致。
- 但吞吐量比現在的 O1 低 20–28%：一個 tick（256 槽位走一步）在 bViT-S 要 8.4 ms，而 batch 256 的一個模型步約 5.3 ms，
  每個 tick 多約 3 ms 的 GPU 工作。R=128 少約 0.8 ms，表示補位（每 tick embed R 張圖、搬移 token 緩衝區）是主要來源之一；
  S=512 更慢（tick 時間超過線性成長）。
- 逐 kernel 剖析（bViT-S，S=256）：一個 tick 的 GPU 時間中，模型步本身約 4.3 ms，
  補位與記錄約 2.6 ms（R=256）/ 1.5 ms（R=128），其他約 0.3 ms。補位的成本來自：tok 與 x0 兩份 [256, 65, 384] fp32 緩衝區
  的 gather（indexSelect，0.53 ms）→ `where`（約 0.6 ms）→ scatter 回去（index_copy，0.46 ms）、每 tick 整份複製 tok（DtoD，0.23 ms）、
  以及對 R 張圖做 embed（約 0.7 ms，R=128 時減半）。停止決策與記錄本身很小。
- 可行的改進：補位上限依平均停止率設定（約 1.5 × S / 平均步數）、不使用 input injection 時不保存 `x0`、
  token 緩衝區雙緩衝以省去每 tick 的整份複製。

#### 槽位池補位改進（0.13.1）

依剖析結果做了三項改動（決策不變，CPU 上仍與 `AdaptiveRunner.run` 逐位元相同）：補位只對真正補位的列做一次 scatter
（token 緩衝區多一列當作未補位候選的落點，取代 gather → where → scatter）；沒有 input injection 時不保存 `x0`；
`Block` / `forward_step` 新增 `out=`，模型步把結果原地寫回 token 緩衝區，省去每 tick 一次整份複製。補位上限預設改為 S/2。

一個 tick 中模型步以外的 GPU 時間（bViT-S，S=256）：補位上限 256 時 2.89 → 1.53 ms，128 時 1.80 → 0.99 ms
。吞吐量（img/s，10,240 張；同一次執行內比較）：

| | bViT-S | bViT-B |
|---|---|---|
| 固定 4 步 | 12,216 | 3,342 |
| O1（壓縮，每步同步） | 12,181 | 3,930 |
| 槽位池 S=256（0.13.0，R=128） | 9,218 | 3,285 |
| **槽位池 S=256（0.13.1，預設 R=128）** | **11,717**（融合 11,787） | **3,664** |
| 槽位池 S=512 | 10,442 | 3,533 |

- 三項改進讓槽位池快了 27%（bViT-S）與 11%（bViT-B），GPU 閒置維持約 0%、使用率 95.5%，決策 100% 一致。
- 但在 T4 上仍比壓縮式 O1 慢 3–7%：O1 的損失是 GPU 閒置（bViT-S 約 19%），槽位池的損失是每 tick 約 1 ms 的額外工作
  （約 17%）加上收尾 4.5% 的空槽位，兩者幾乎抵銷。S=512 或補位上限太小（S=512、R=96，使用率 61%）都更差。
- 剩下的 tick 額外成本：補位候選的 embed（R=128 時約 0.4 ms）與每 tick 把權重轉成 fp16（約 0.37 ms，因 graph 內
  autocast 不能快取；壓縮式 runner 每步也付這筆）。
- 結論：T4 上壓縮式 O1 維持預設；槽位池保留為選項。CPU 相對 GPU 越慢（如 H100），O1 的閒置比例越高，槽位池越可能勝出，需在 H100 上量測。

#### H100 驗證（第 10 輪，0.14.0，[`results/round10_h100`](results/round10_h100/summary.md)）

`c100_bvit_s_quadw_lr5e4`（種子 0），對同準確率中最便宜的固定深度的速度比：

| λ | 準確率 | 平均步數 | batch 256：O1 | batch 256：槽位池 + 融合 | batch 1 graph：O1 | batch 1 graph：+ 融合 |
|---|---|---|---|---|---|---|
| 0.001 | 61.93 | 4.62 | 0.40× | 0.80× | 0.52× | 0.77× |
| 0.01 | 62.04 | 4.11 | 0.54× | 1.06×（0.89×） | 0.65× | 1.01×（0.91×） |
| 0.03 | 62.04 | 3.71 | 0.53× | 1.18×（0.99×） | 0.72× | 1.09×（0.98×） |
| 0.1 | 61.53 | 3.26 | 0.44× | **1.12×** | 0.73× | **1.12×** |
| 0.3 | 60.05 | 2.74 | 0.37× | **1.07×** | 0.71× | **1.07×** |

括號：λ=0.01 / 0.03 的準確率只比固定 5 步高 4 張圖，所以配對到 6 步；括號內是對 5 步的比值。

- **與 T4 相反，H100 上槽位池大勝壓縮式 O1**：batch 256 快 2.0–2.9 倍（44,147 → 98,454 img/s @ λ=0.03）。
  H100 的 GPU 快、CPU 相對慢，壓縮式 O1 幾乎完全卡在 CPU 發送 kernel（合成測試：O1 35,190 vs 固定 4 步 121,824 img/s），
  槽位池每 tick 只發送一次 graph，GPU 閒置約 1%。
- **第一次在 batch 256 勝過同準確率的固定深度**：λ = 0.1 / 0.3 快 12% / 7%（配對明確）；λ = 0.01 / 0.03 對 6 步為 1.06× / 1.18×，
  對準確率幾乎相同的 5 步為 0.89× / 0.99×。λ = 0.001 仍較慢（0.80×）。
- **融合 kernel 在 H100 上是必要的**：槽位池 +17%（84,189 → 98,454），batch 1 graph 延遲 −34%（0.726 → 0.477 ms），
  eager +43%。batch 1 graph + 融合在 λ = 0.1 / 0.3 也勝過固定深度（1.12× / 1.07×；第 9 輪同骨幹 O1 為 0.69×）。
- O3 在 batch 256 吞吐量雖高（約 84,000 img/s），但準確率在同步數下較低，配對後不如槽位池。
- 單一種子、單次執行；建議以 3 個種子確認後再寫入論文。
- 決定：**保留槽位池**（達到 H100 上比 O1 快 5% 以上的保留門檻），GPU 上 batch 推論建議使用 `SlotPoolRunner(..., fused=True)`。
- **注意**：上表的固定深度是 eager 執行；三個種子與 compiled 固定深度的比較見下一節，結論需修正。

#### H100 三個種子（第 10 輪續，[`results/round10_h100/seeds3.md`](results/round10_h100/seeds3.md)）

以三種子平均準確率配對（準確率不低於自適應的最便宜固定深度），槽位池 + 融合 kernel 的 batch 256 吞吐量：

| λ | 準確率 | 步數 | 配對固定深度 | 槽位池 + 融合 img/s | 對 O1 | 對 eager 固定深度 | 對 compiled 固定深度 |
|---|---|---|---|---|---|---|---|
| 0.001 | 61.94 | 4.42 | 5 步 | 82,940 | 2.58× | 0.91× | 0.73× |
| 0.01 | 61.98 | 4.04 | 6 步 | 90,375 | 2.07× | 1.18× | 0.93× |
| 0.03 | 61.95 | 3.70 | 5 步 | 98,427 | 2.29× | 1.08× | 0.87× |
| 0.1 | 61.48 | 3.24 | 5 步 | 111,647 | 2.60× | 1.22× | 0.98× |
| 0.3 | 60.35 | 2.79 | 4 步 | 128,942 | 2.99× | 1.22× | 0.96× |

- **確認**：槽位池 + 融合 kernel 比壓縮式 O1 快 2.1–3.0 倍，三個種子間吞吐量差異約 ±1.5%。
- **更正第 10 輪單種子的結論**：「勝過固定深度」是拿 CUDA graph 的槽位池對 **eager** 固定深度比較。H100 上 eager 固定深度
  本身也受 kernel 發送限制：torch.compile 讓固定深度在 4–6 步快 1.24–1.27 倍，三個種子的計時也穩定得多（±1–2%，eager 為 ±14%）。
  對 compiled 固定深度，槽位池在 λ = 0.1 / 0.3 約打平（0.98× / 0.96×），λ = 0.01 / 0.03 為 0.93× / 0.87×。
  最公平的對照（batch 256 的固定深度也錄成 CUDA graph）還沒量測。
- batch 1 CUDA graph（兩者都錄成 graph，比較公平）+ 融合：λ = 0.001 / 0.01 / 0.03 為 0.81× / 0.94× / 1.03×；
  λ = 0.1 / 0.3 有一個種子受干擾（0.581±0.296、0.502±0.238 ms），這兩格不能用。
- O3 仍落後槽位池，也低於同準確率固定深度。
- 第 10 輪單種子小節「第一次在 batch 256 勝過同準確率的固定深度」的說法以本節為準。

## VCNet：原始 vs 自適應步數（`dev/vcnet` 分支，0.9.0 起）

使用者決定改用 VCNet，目標是比較**原始 VCNet**與 **VCNet + 自適應步數**。架構、論文沒寫而由我們補上的假設、
實驗計畫見 [`docs/vcnet.md`](docs/vcnet.md)；技術說明與實驗報告見 [`docs/vcnet_report.html`](docs/vcnet_report.html)。取得程式時，把第 1 步指令裡的分支名稱 `claude/charming-ritchie-qycdtj` 換成 `dev/vcnet`。登記的實驗：`vcnet_smoke`、`c100_vcnet_orig`、`c100_vcnet_orig_ff`、
`c100_vcnet_adapt`、`c100_vcnet_adapt_carry`。

**第 1 輪（3 個種子，[`results/vcnet_round1_h100`](results/vcnet_round1_h100/)）：原始 vs 自適應**

| 模型 | 平均步數 | GFLOPs | test acc % | ECE TS % |
|---|---|---|---|---|
| `orig_ff`（原始，無回饋） | 1 | 0.161 | 63.72±1.22 | 1.89 |
| **`orig`（原始 VCNet）** | 2 | 0.275 | **64.55±0.96** | 1.79 |
| `adapt` 固定 2 步 | 2 | 0.275 | 64.91±0.94 | 2.50 |
| `adapt_carry` 固定 2 步 | 2 | 0.275 | 65.30±0.41 | 2.41 |
| `adapt_carry` 固定 3 步 | 3 | 0.388 | 65.44±0.53 | 2.41 |
| `adapt_carry` + 停止策略，λ=0.1 | 1.23 | 0.187 | 64.71±0.69 | – |
| `adapt_carry` + 停止策略，λ=0.03 | 1.36 | 0.202 | 64.84±0.85 | – |

- **原始 vs 自適應**：`adapt_carry` 加停止策略，以原始模型 68% 的計算量（0.187 vs 0.275 GFLOPs）達到同等以上的準確率
  （64.71 vs 64.55）；batch 1 + CUDA graph 延遲 1.29 ms，同架構固定 2 步為 1.82 ms。固定同樣 2 步時 carry 高 0.75 點。
  差距都在種子標準差（約 0.5–1.0）以內，需要逐種子比較才能確定。
- 回饋本身有用：原始 2 步比 1 步高 0.8 點；`adapt` 的第 0 步（63.75）與 `orig_ff`（63.72）相同，每步讀出不拖累第一步。
- 預測編碼迴圈修正 1 次後就飽和：第 3–8 步沒有增益；`adapt` 的停止策略在 benchmark 的所有 λ 下都在第 0 步停；
  batch 256 時自適應不比固定深度快。
- 溫度縮放後原始模型的 ECE 較低（1.8% vs 2.4–2.5%）：Stage A 的 Brier + Gatekeeper 沒有改善縮放後的校準。

**第 2 輪（0.10.0，3 個種子，[`results/vcnet_round2_h100`](results/vcnet_round2_h100/summary.md)）：跨步狀態**

`--lateral`：每個腦區保留跨步狀態；`--topdown`：再加 BLT 式由上而下連線（[`docs/vcnet.md`](docs/vcnet.md)「跨步狀態」）。

| 模型 | 平均步數 | GFLOPs | test acc % |
|---|---|---|---|
| **原始 VCNet（`orig`）** | 2 | 0.275 | **64.55±0.96** |
| `adapt_lat` 固定 1 步 | 1 | 0.161 | 64.75±0.49 |
| `adapt_lat` 固定 2 步 | 2 | 0.281 | 67.39±0.48 |
| `adapt_blt` 固定 2 步 | 2 | 0.284 | 67.27±0.57 |
| `adapt_blt` 固定 4 步（最佳固定深度） | 4 | 0.528 | 68.68±0.58 |
| `adapt_blt` + 停止策略，λ=0.1 | 1.74 | 0.252 | 67.77±0.42 |
| `adapt_blt` + 停止策略，λ=0.03 | 2.09 | 0.294 | 68.20±0.58 |
| `adapt_blt` + 停止策略，λ=0.01 | 2.41 | 0.333 | 68.34±0.55 |

- 跨步狀態解除了第 2 步飽和：`blt` 一路進步到第 4 步（63.7 → 67.3 → 68.3 → 68.7）。
- 同樣約 0.28 GFLOPs，`lat` / `blt` 比原始 VCNet 高 2.7–2.8 點；`blt` + 停止策略以原始模型 91% 的計算量高 3.2 點。
- 停止策略第一次在同一個模型內勝過固定深度：`blt` 在 2 步的計算量下 68.18 vs 67.27，3 步時 68.59 vs 68.30。
- 但大部分增益來自架構，不是自適應；原始 VCNet 只有一個讀出、只用 CE。第 3 輪用同樣的 `blt` 架構、照原始方式訓練，把兩者分開。
- 計時受其他程式干擾（多列標準差達平均值的 30–100%）；`blt` 的 batch 1 + CUDA graph 較一致：
  λ=0.03 為 2.22 ms（68.20%），固定 3 步 3.12 ms（68.29%）。batch 256 時自適應仍比固定深度慢。

**延遲與吞吐量（閒置 GPU 重量，[`results/vcnet_bench_idle_h100`](results/vcnet_bench_idle_h100/summary.md)）**：

| 模型、設定 | acc % | batch 256 compiled 張/秒 | batch 1 CUDA graph ms |
|---|---|---|---|
| 原始 VCNet（2 步） | 64.56 | 41,499 | 1.908 |
| `adapt_blt` 固定 2 步 | 67.27 | 39,804 | 1.949 |
| `adapt_blt` 固定 3 步 | 68.29 | 27,081 | 2.915 |
| `adapt_blt` + 停止策略 λ=0.1 | 67.77 | 13,567 | 1.825 |
| `adapt_blt` + 停止策略 λ=0.03 | 68.20 | 10,644 | 2.125 |

- batch 1：自適應有效，`blt` + 停止策略比原始 VCNet 略快且高 3.2 點；比 `blt` 固定 3 步快約 1.4 倍。
- batch 256：自適應無效，停止策略讓吞吐量降到固定步數的 27–42%；`blt` 固定 2 步的吞吐量約為原始的 96%、高 2.7 點。
- `blt` 的 eager 實作每步很慢，部署需用 torch.compile。

**第 3 輪（[`results/vcnet_round3_h100`](results/vcnet_round3_h100/summary.md)）：架構 vs 自適應**：`blt` 架構照原始方式訓練
（固定步數、只在最後讀出、只用 CE）。

| 模型 | 步數 | GFLOPs | test acc % |
|---|---|---|---|
| 原始 VCNet（`orig`） | 2 | 0.275 | 64.55±0.96 |
| `blt_orig2` | 2 | 0.284 | 67.61±0.74 |
| `adapt_blt` 固定 2 步 | 2 | 0.284 | 67.27±0.57 |
| `blt_orig4` | 4 | 0.528 | 68.90±0.62 |
| `adapt_blt` 固定 4 步 | 4 | 0.528 | 68.68±0.58 |
| `adapt_blt` + 停止策略 λ=0.1 | 1.73 | ≈0.251 | 67.77±0.42 |

- 增益幾乎全部來自架構（同計算量 +3.1 點）；每步讀出不拖累最終準確率；自適應的價值是「一個模型適用所有計算預算」，
  加上停止策略在 batch 1 小幅省計算（λ=0.1 以 88% 計算量追平 `blt_orig2`），但追不上 4 步的模型。
- 報告（[`docs/vcnet_report.html`](docs/vcnet_report.html)）已加入第 3 輪（5.4 節）。

**訓練加速（階段 0，0.15.0）**：GPU 上的資料增強、編譯 + CUDA graph 的訓練步、fused AdamW、`--parallel` 並行種子。
量測用 `scripts/bench_train.py`，驗證用 `c100_vcnet_adapt_blt_fast`、`c100_vcnet_orig_fast`；步驟見
[`docs/vcnet.md`](docs/vcnet.md)「訓練加速」。
H100 結果（[`results/phase0_train_speed_h100`](results/phase0_train_speed_h100/summary.md)）：CUDA graph 讓訓練快 1.7 倍（orig）
到 4.6 倍（blt）；快速管線在 3 個種子上重現第 1–2 輪的準確率（blt 每步 +0.1–0.35 點，在種子差異內），3 個種子的 blt 從約 300 分鐘
降到 83 分鐘。之後的新實驗改用快速管線。
0.15.1 修正 GPU 增強中的同步後，orig 快 2.9 倍；0.15.2 把訓練時的雙線性上採樣改成矩陣乘法（其反向原本佔 blt 訓練步約 9%）。
快速管線的實驗請依序跑（開了 CUDA graph 後並行多份不會更快）。
batch 512 / 1024（學習率按 √ 放大）各低 1.0 / 3.4 點，不採用；新實驗預設為快速管線 + batch 256（blt 3 個種子約 65 分鐘，原本約 300 分鐘）。

**推論加速（階段 2，0.16.0）**：槽位池支援 VCNet（每列各自的步數、V1 預先計算、可編譯），並加入公平對照組：固定步數在 batch 256
整個錄成一張 CUDA graph（`FixedDepthGraph`，benchmark 模式 `graph-bs256` / `graph-compiled-bs256`）。步驟見
[`docs/vcnet.md`](docs/vcnet.md)「推論加速」。
H100 結果（[`results/vcnet_phase2_h100`](results/vcnet_phase2_h100/summary.md)）：編譯後的槽位池比壓縮式 runner 快約 6 倍，
λ=0.1 時對同準確率固定步數為 1.05 倍，但仍低於固定步數的前緣（0.82–0.89 倍）；就算沒有任何額外開銷也最多快約 10%。
CIFAR 上 batch 推論建議用「固定步數 + 編譯 + CUDA graph」（blt 2 步每秒 105,000 張，是原始 VCNet 的 91%、高 2.8 點）。

**對照組（階段 1，0.17.0）**：ResNet-56（0.862M 參數、0.252 GFLOPs）與同等大小的 VCNet：`vcnet_r56p`（參數量相同，
0.864M）、`vcnet_r56f4`（4 步的計算量相同，0.249 GFLOPs）。ResNet-56 跑兩種訓練方式（與 VCNet 相同的 AdamW + 強增強、
一般的 SGD + crop/flip）。實驗與步驟見 [`docs/vcnet.md`](docs/vcnet.md)「對照組」。

**階段 1 結果與決定**（[`results/phase1_resnet56_h100`](results/phase1_resnet56_h100/summary.md)）：ResNet-56 BN 70.44–71.25%，
同參數的 `vcnet_r56p` 2 / 3 步 65.49 / 66.55%；差距大多來自正規化（ResNet-56 GN 62.84%），但同樣用 GN + weight standardization
時 ResNet-56 仍領先 1–2 點（68.64 vs 66.84 / 67.82%）。使用者決定方向 B：維持 GN + WS、改進仿皮質層架構（暫不實作，見
[`docs/vcnet.md`](docs/vcnet.md)「階段 1 方向決定」）。

## 下一階段方向（討論中）

使用者決定：之後尋找**適合此架構的任務**，並考慮加入**仿皮層遞迴網路**（TODO.md 第 4 節）。
CIFAR-100 上的結論是：遞迴後段步數的幫助有限（約第 4–5 步飽和）、逐樣本難度差異中等、每步計算量太小以致實際時間被發送開銷主導。
適合的任務應該讓「多想幾步」真的必要、難度可控且差異大、並能與人類反應時間 / 皮層動態對照。候選：

1. **需要逐步傳播的視覺任務**：路徑追蹤（Pathfinder、cABC）、迷宮、數數。難度（路徑長度、迷宮大小）可控，遞迴是必要的；
   可測試「難題自動多想」與超出訓練步數的零樣本外推（AdRNN 2023 即用這類任務）；人類曲線追蹤的反應時間隨路徑長度增加，
   猴子 V1 有對應的逐步傳播（Roelfsema）；仿皮層的水平連接遞迴網路（hGRU、BLT）是自然的對照與骨幹。
2. **遮擋 / 降質影像辨識**：遮擋物體、低對比、模糊。遞迴處理在神經科學上與遮擋物體辨識有關（Kar 2019 的「挑戰影像」、IT 解答時間），
   直接對應 H2 與 H3。
3. **由粗到細的分類**：CIFAR-100 的 20 個超類別 / 100 個細類別，早期步回答粗類別、後期步回答細類別。
