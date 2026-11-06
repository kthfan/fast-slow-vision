# fast-slow-vision

方向 1「校準式快慢視覺」的實作。完整規格見 [`TODO.md`](TODO.md)。

目前進度：**bViT 式單區塊遞迴 ViT + 每步讀出**（Stage A）、**每步溫度縮放**（Stage B）、
免訓練停止基準（max-prob / 熵門檻、固定深度、oracle），以及 FLOPs / 延遲量測與每步特徵快取（Stage C 的前置）。

## 結構

| 路徑 | 內容 |
|---|---|
| `fsv/models/bvit.py` | `RecurrentViT`：單一共用 Transformer 區塊遞迴 `t_max` 次；共用讀出頭 `head(LN(CLS_t) + step_emb(t))`；介面 `embed / forward_step / readout / features` |
| `fsv/losses.py` | Stage A 損失：每步 CE + β·Brier + γ·Gatekeeper 信心損失 + δ·最後一步自我蒸餾 |
| `fsv/calibration.py` | 每步溫度縮放（held-out 上以 LBFGS 擬合） |
| `fsv/metrics.py` | acc、ECE、Brier、NLL、AURC |
| `fsv/flops.py` | 解析式 FLOPs（含每步讀出頭；與 `torch.utils.flop_counter` 比對誤差 < 5%） |
| `fsv/halting/thresholds.py` | max-prob / 熵門檻曲線、固定深度早退、oracle 停止步數 t*、曲線下面積 |
| `fsv/engine.py` | 收集每步 logits / 特徵；逐步延遲量測 |
| `fsv/data.py` | CIFAR-10/100；從訓練集切出固定的 5,000 張 held-out |
| `scripts/train.py` | 訓練（單卡 / CPU / `torchrun` 多卡 DDP，fp16 AMP） |
| `scripts/evaluate.py` | 溫度縮放、每步指標、停止基準曲線、延遲、特徵快取 |
| `scripts/run_cifar.sh` | 一鍵訓練 + 評估（自動用所有 GPU） |

## 與 bViT 論文（Byra et al., 2026）的差異

論文沒有公開程式碼，依論文描述自行實作：pre-norm 區塊、GELU、無 drop path、可選 bViT-TE（每步輸入時間嵌入）。
為了 CIFAR（32×32）與本專案需求做了以下調整：

- patch size 4（64 個 patch + CLS），`t_max` 預設 8（論文 ImageNet 為 12）。
- 預設 LayerNorm（TODO.md 2.1 定案；論文用 RMSNorm，可用 `--norm rmsnorm`）。
- 每一步都讀出並計算損失（論文只在最後一步讀出）。

## 使用

```bash
pip install -e .[test]
pytest -q
```

```bash
# 單機
python scripts/train.py --out runs/debug --epochs 1 --max-steps 20
python scripts/evaluate.py --ckpt runs/debug/model.pt --latency --cache
```

```bash
# 多 GPU（Kaggle 2x T4）
bash scripts/run_cifar.sh c100_bvit_s
```

`evaluate.py` 會輸出 `results.json`（每步指標、溫度、FLOPs、停止曲線、oracle、延遲），
加 `--cache` 時另存 `cache_{holdout,test,train}.pt`（每步 logits、CLS 與 pooled 特徵），供 TODO-1 的停止策略離線實驗使用。

## 實驗紀錄

### `experiments/c100_bvit_s`：CIFAR-100、bViT-S、T_max=8、200 epochs（Kaggle 2× T4，訓練 65.6 分鐘）

1.86M 參數（一個共用區塊）。測試集結果；溫度縮放（TS）使用 held-out 5k 擬合：

| 步 t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| top-1 % | 34.0 | 55.9 | 60.2 | 60.9 | **61.3** | 61.3 | 61.1 | 60.7 |
| ECE 原始 % | 10.0 | 9.0 | 7.0 | 6.6 | 7.0 | 7.1 | 7.4 | 7.6 |
| ECE TS % | 1.2 | 1.5 | 1.9 | 1.8 | 2.4 | 2.4 | 2.6 | 2.5 |
| 溫度 T_t | 0.77 | 0.82 | 0.85 | 0.85 | 0.84 | 0.84 | 0.83 | 0.83 |
| 累積 GFLOPs | 0.24 | 0.48 | 0.71 | 0.95 | 1.19 | 1.42 | 1.66 | 1.90 |
| 延遲 bs=1 (ms) | 1.0 | 1.4 | 2.0 | 2.4 | 2.9 | 3.3 | 3.9 | 4.3 |

準確率–計算量曲線下面積（[0.24, 1.90] GFLOPs 區間的平均準確率）：
固定深度 58.4、max-prob 門檻 56.5、熵門檻 56.1。
Oracle（t*，最後一步錯的跑滿）：60.7% @ 1.02 GFLOPs（平均 4.3 步）。

觀察：
- 準確率在第 3–4 步飽和，之後微降；訓練結束時 held-out 準確率仍在上升（欠擬合）。
- 溫度皆 < 1：模型**信心不足**（mixup/cutmix + Brier + Gatekeeper 都把分布推平），溫度縮放後 ECE 降到 1–2.6%。
- 免訓練門檻規則**輸給固定深度**：第 0 步只有 34%，門檻在第 0 步就停下太多「第 1 步就會答對」的樣本。
