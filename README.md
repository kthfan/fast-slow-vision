# fast-slow-vision

方向 1「校準式快慢視覺」的實作。完整規格見 [`TODO.md`](TODO.md)。

目前進度：**bViT 式單區塊遞迴 ViT + 每步讀出**（Stage A）、**每步溫度縮放**（Stage B）、
免訓練停止基準（max-prob / 熵門檻、固定深度、oracle），以及 FLOPs / 延遲量測與每步特徵快取（Stage C 的前置）。

## 開發流程

實驗在使用者自己的機器上跑（單張 H100），結果以 report 形式上傳回 repo：

1. Claude 在分支 `claude/charming-ritchie-qycdtj` 上寫程式，並在 `fsv/experiments.py` 登記要跑的實驗。
2. 使用者拉下最新程式，用 `scripts/run_experiment.py <實驗名>` 跑實驗。
3. 腳本自動產生 `reports/<實驗名>/`，使用者把它 commit 並 push。
4. Claude 讀 report、分析，決定下一輪實驗。

---

## 使用者操作步驟

### 0. 需求

- Linux，NVIDIA driver 支援 CUDA 12.x（`nvidia-smi` 右上角的 CUDA Version ≥ 12.6；若為 13.x 也可改用 `cu130` 的 wheel）
- Python 3.10 以上、git
- 能連到 `huggingface.co`（第一次執行會下載 CIFAR-100，約 140 MB）

### 1. 取得程式（第一次）

```bash
git clone https://github.com/kthfan/fast-slow-vision.git
```

```bash
cd fast-slow-vision
```

```bash
git checkout claude/charming-ritchie-qycdtj
```

之後每一輪開始前更新程式：

```bash
git pull origin claude/charming-ritchie-qycdtj
```

### 2. 建立環境（第一次）

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

先裝 CUDA 版 PyTorch（H100 用 CUDA 12.x 的 wheel），再裝本專案：

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

### 4. 檢查環境

```bash
pytest -q
```

```bash
python scripts/check_env.py
```

`pytest` 約 1 分鐘、應顯示全部通過。`check_env.py` 最後一行應為 `OK: environment ready`，並顯示 `GPU NVIDIA H100 ...`、`AMP bf16`。

### 5. 冒煙測試（約 1–2 分鐘）

```bash
python scripts/run_experiment.py smoke
```

跑完會印出 report 路徑，表示整條流程（下載資料 → 訓練 → 溫度縮放 → 評估 → report）正常。smoke 的 report 不需要上傳。

### 6. 跑實驗

列出所有登記的實驗：

```bash
python scripts/run_experiment.py --list
```

跑一個實驗（例如基準線）：

```bash
python scripts/run_experiment.py c100_bvit_s
```

一次依序跑多個，並在登出後繼續執行（輸出另存到 `run_all.log`）：

```bash
nohup bash -c 'for e in c100_bvit_s c100_bvit_s_lightaug c100_vit_s_8blk; do python scripts/run_experiment.py $e; done' > run_all.log 2>&1 &
```

```bash
tail -f run_all.log
```

- 中斷後重跑同一個指令，會從 `runs/<實驗名>/last.pt` 接續訓練；要從頭來加 `--fresh`。
- 指定 GPU：`--gpu 1`（預設 0）。
- `runs/` 存放 checkpoint（大檔，留在本機，不要上傳）；`reports/` 是要上傳的小檔。

### 7. 上傳 report

```bash
git add reports/c100_bvit_s
```

```bash
git commit -m "Report: c100_bvit_s"
```

```bash
git pull --rebase origin claude/charming-ritchie-qycdtj
```

```bash
git push origin claude/charming-ritchie-qycdtj
```

一次跑了多個實驗，就把每個 `reports/<實驗名>` 都 `git add`。
不方便 push 的話，也可以打包後直接在對話中附上：

```bash
tar czf reports.tgz reports/
```

上傳後在對話裡告訴 Claude 跑了哪些實驗即可。

### Report 內容

| 檔案 | 內容 |
|---|---|
| `report.md` | 摘要：環境、訓練曲線、每步測試指標、停止曲線 AUC、oracle、延遲 |
| `results.json` | 完整評估結果（每步指標、溫度、FLOPs、停止曲線各點、延遲） |
| `history.jsonl` | 每個 epoch 的訓練損失、速度與 held-out 每步準確率 |
| `config.json` | 訓練參數、模型設定、參數量、AMP |
| `env.json` | git commit（含是否有未提交修改）、GPU、torch、CPU 上限、執行時間、完整指令 |
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
| `scripts/run_experiment.py` | 一鍵：訓練 → 評估 → report |
| `scripts/train.py`, `scripts/evaluate.py` | 底層訓練 / 評估程式（也可單獨使用，見 `--help`） |
| `scripts/check_env.py` | 環境檢查 |
| `scripts/make_report.py` | 由 JSON 重新產生 `report.md` |

## 與 bViT 論文（Byra et al., 2026）的差異

論文沒有公開程式碼，依論文描述自行實作：pre-norm 區塊、GELU、無 drop path、可選 bViT-TE（每步輸入時間嵌入）。
為了 CIFAR（32×32）與本專案需求做了以下調整：

- patch size 4（64 個 patch + CLS），`t_max` 預設 8（論文 ImageNet 為 12）。
- 預設 LayerNorm（TODO.md 2.1 定案；論文用 RMSNorm，可用 `--norm rmsnorm`）。
- 每一步都讀出並計算損失（論文只在最後一步讀出）。

## 實驗紀錄

| report | 摘要 |
|---|---|
| [`reports/kaggle_c100_bvit_s`](reports/kaggle_c100_bvit_s/report.md) | Kaggle 2×T4 上的第一次 bViT-S 訓練：最佳步 top-1 61.3%（第 4 步）；溫度縮放後 ECE 1–2.6%；免訓練門檻規則的 AUC 輸給固定深度 |
