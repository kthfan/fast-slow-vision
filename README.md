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

原始資料：第 1 輪 [`results/round1_h100/`](results/round1_h100/)（使用者貼上的 `report.md`），
第 2 輪 [`results/round2_h100/summary.md`](results/round2_h100/summary.md)。

觀察（第 1–2 輪）：

1. **種子變異很小**（bViT-S 最佳步 ±0.07），所以 Kaggle（61.3）與 H100（59.3）的 2 個百分點差距是系統性的。
   主要嫌疑是精度（fp16 vs bf16），其次是雙卡 DDP 的 mixup 分組；第 3 輪用 fp16 在 H100 重跑來確認。
2. **減輕增強反而更差**：過擬合，「增強太強導致欠擬合」不成立；但它的溫度約 1.1，表示基準線的「信心不足」主要來自 mixup / CutMix。
3. **共用權重的模型在第 3–4 步就飽和**，加寬（bViT-B）也一樣，而且 4 倍計算量反而少 1.3 個百分點；
   不共用的 8 層 ViT 一路進步到第 7 步（+3.3）。讓遞迴的後段步數有用，是方向 1 的核心問題；第 3 輪測試時間嵌入、輸入重新注入、偏重後段的損失權重、拿掉自我蒸餾。
4. **免訓練門檻規則在所有模型上都輸給固定深度**；oracle 顯示逐樣本停止約 1.5 步即可達到最後一步的準確率，學習式停止（TODO-1）空間很大。
5. 延遲量測在 bs=256 的標準差很大（GPU 可能同時有其他工作），自 0.3.0 起改為取多次量測的中位數，並記錄最慢/最快比值。
