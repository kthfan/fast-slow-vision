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
| `fsv/losses.py` | Stage A 損失：每步 CE + β·Brier + γ·Gatekeeper 信心損失 + δ·最後一步自我蒸餾 |
| `fsv/calibration.py` | 每步溫度縮放（held-out 上以 LBFGS 擬合） |
| `fsv/metrics.py` | acc、ECE、Brier、NLL、AURC |
| `fsv/flops.py` | 解析式 FLOPs（含每步讀出頭） |
| `fsv/halting/thresholds.py` | max-prob / 熵門檻曲線、固定深度早退、oracle 停止步數 t*、曲線下面積 |
| `fsv/engine.py` | 收集每步 logits / 特徵；逐步延遲量測；AMP 自動選擇（H100 用 bf16） |
| `fsv/cpu.py` | CPU 核心數硬性上限 |
| `fsv/data.py` | CIFAR-10/100（Hugging Face parquet，快取為 npz）；從訓練集切出固定 5,000 張 held-out |
| `fsv/experiments.py` | 實驗登記表（每個名稱對應固定的訓練設定） |
| `fsv/report.py` | 產生 `report.md` |
| `fsv/version.py` | 程式版本（版本號 + 原始碼雜湊，不需要 git） |
| `scripts/run_experiment.py` | 一鍵：訓練 → 評估 → report（可多實驗、多種子） |
| `scripts/summarize.py` | 輸出可貼到對話的精簡摘要（多種子取平均±標準差） |
| `scripts/train.py`, `scripts/evaluate.py` | 底層訓練 / 評估程式（也可單獨使用，見 `--help`） |
| `scripts/check_env.py` | 環境檢查 |
| `scripts/halting_study.py` | TODO-1 停止策略研究（在快取上離線訓練與評估；`run_experiment.py --halting` 會自動呼叫） |
| `fsv/halting/features.py`, `fsv/halting/policy.py` | 停止策略的輸入特徵（S0 / S3）、MLP、訓練目標、λ 掃描曲線 |
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
| **`c100_bvit_s_linw_lr5e4`（線性步權重，lr 5e-4）** | 0,1,2 | 1.86M | **61.25±0.36**（t=5） | 60.90±0.34 | 3.06±0.05 | 55.50 / 55.05 / 57.35 | 64.04 @ 0.39 G |
| `c100_bvit_s_linw_te_inject` | 0 | 1.86M | 61.71（t=5） | 61.29 | 2.89 | 55.90 / 55.36 / 57.79 | 64.22 @ 0.39 G |
| `c100_bvit_s_mix2`（2 組 mixup） | 0 | 1.86M | 59.67（t=4） | 59.03 | 3.97 | 54.97 / 54.59 / 56.92 | 62.68 @ 0.36 G |
| `c100_vit_s_8blk_linw` | 0 | 14.28M | 62.78（t=7） | 62.78 | 2.47 | 56.58 / 55.89 / 57.28 | 66.11 @ 0.45 G |

原始資料：第 1 輪 [`results/round1_h100/`](results/round1_h100/)（使用者貼上的 `report.md`），
第 2 輪 [`results/round2_h100/summary.md`](results/round2_h100/summary.md)，
第 3 輪 [`results/round3_h100/summary.md`](results/round3_h100/summary.md)，
第 4 輪 [`results/round4_h100/summary.md`](results/round4_h100/summary.md)，
第 5 輪 [`results/round5_h100/summary.md`](results/round5_h100/summary.md)。

觀察（第 1–5 輪）：

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
   平方權重平均略高（61.46）但在 1e-3 下同樣不穩定（±1.10）。尚缺「均等權重 + lr 5e-4」的對照（第 6 輪）。
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
