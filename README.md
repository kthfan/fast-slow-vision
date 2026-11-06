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
