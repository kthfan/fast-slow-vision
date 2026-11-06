# 方向 1：校準式快慢視覺（Calibrated Fast–Slow Vision）

> 實作規格文件，給實作 agent 參考。
> 狀態：方向 1 進行中；方向 2（粗到細雙流路由）暫緩，本文件不涵蓋。
> 標記說明：**[定案]** 已決定，照做；**[預設]** 先照此實作，可依實驗調整；**[TODO]** 尚待設計或評估，見第 12 節。

---

## 0. 一句話摘要

一個權重共用的遞迴 ViT，每算一步就輸出一次**校準過的**類別機率；一個小型停止策略依據目前狀態與**計算價格 λ** 決定停或繼續。簡單的圖早停、難的圖多想，單一模型即可涵蓋整條「準確率–計算量」曲線。

## 1. 目標與研究假說

### 期望成果

1. 加速推論：平均計算量與**實際延遲**都要下降。
2. 推進 vision 模型：把「校準 + 成本可調的停止策略」做進遞迴 ViT。
3. 引用最新方法：以 2025–2026 的方法為主要比較對象。
4. 提升效能：相同平均計算量下，準確率、穩健度、校準度更好。

### 假說

- **H1（效率）**：校準且以 λ 為條件的停止策略，在準確率–計算量曲線上優於熵門檻、最大機率門檻、ACT 與固定早退。
- **H2（穩健）**：相同平均計算量下，把遞迴集中給難圖，能提升 ImageNet-C/-R/-A 的表現與校準。
- **H3（像人類）**：模型停止步數與人類最短觀看時間（MVT）正相關；與 IT 物體解答時間吻合。

### 與最新方法的區隔（論文主張的重點）

| 對手 | 它做了什麼 | 我們多做什麼 |
|---|---|---|
| AdaPerceiver（2025） | 早退與 token 數的自適應，τ=0.95 時約省 24% FLOPs | 停止決策的信心經過校準；λ 條件化 |
| reViT（2026） | 單一 checkpoint 支援彈性深度 | 深度由每張圖自動決定，而非全域設定 |
| bViT（2026） | 單區塊遞迴、參數大減 | 自適應步數 + 校準 |
| AdRNN（2023）、BL（2020） | ACT / 熵門檻停止 | 校準損失、λ 條件化、RL 停止策略、人類與 IT 驗證 |

---

## 2. 系統架構

```
影像 ──► Patch 嵌入 ──► ┌──────────────────────┐
                        │ 共用遞迴區塊（第 t 步）│◄──────────────┐
                        └──────────┬───────────┘               │
                                   │ h_t（CLS + patch tokens）  │
                                   ▼                            │
                        ┌──────────────────────┐               │
                        │ 共用讀出頭 + 步數嵌入  │──► p_t（校準） │
                        └──────────┬───────────┘               │
                                   ▼                            │
                        ┌──────────────────────┐   繼續         │
               λ ──────►│ 停止策略（小型 MLP）  │───────────────┘
                        └──────────┬───────────┘
                                   │ 停
                                   ▼
                               輸出 p_t
```

步數記號：t = 0 表示第一次通過遞迴區塊後的狀態；最多 T_max 步。

### 2.1 骨幹 [預設]

- 主線：**reViT 式遞迴 ViT**（一個 Transformer 區塊反覆套用；每個深度的 FFN 由一小組共用專家混合）。若程式碼不可得，先用 **bViT 式**單區塊遞迴（更簡單）。
- 規模：先做 ViT-S 等級（embed 384）原型，再做 ViT-B 等級（embed 768）。
- T_max：預設 8；消融 4 / 8 / 12。
- 正規化：**一律 LayerNorm（ViT）/ GroupNorm（CNN 副線）**。禁止依時間步分別統計的 BatchNorm，避免評估時的統計假象（見 Leutenegger 2026 的更正）。[定案]
- 教師蒸餾：DINOv3（不可得時用 DINOv2）輸出特徵蒸餾 + 分類交叉熵。[預設]

### 2.2 讀出頭 [預設]

- 一個**共用**線性分類頭，輸入 `LN(CLS_t) + step_emb(t)`；避免每步各一個頭造成大量參數。
- 訓練後對**每一步**各擬合一個溫度 T_t（temperature scaling，在 held-out 驗證集上）。

### 2.3 停止策略 [預設 + TODO]

- 預設基準版本：小型 MLP（2 層，hidden 64），輸入為**機率統計量**：
  - top-5 機率、熵、top-1 與 top-2 差距
  - 與上一步的變化：top-1 是否改變、KL(p_t ‖ p_{t-1})
  - 步數 t（one-hot 或嵌入）
  - λ（以 log λ 輸入）
- 輸出：停止機率 `π(stop | ·)`。
- **輸入要不要包含特徵（state）、特徵從哪裡來、輸出長什麼樣子，見第 12 節 TODO-1，是本專案的主要待研究項目。**

### 2.4 計算價格 λ [定案]

- 訓練時每個 batch（或每個樣本）從 log-uniform 分布抽 λ ∈ [λ_min, λ_max]。
- λ 作為停止策略的輸入；推論時調整 λ 即得到不同的速度–準確率取捨點。
- 概念來源：Butkus et al. 2026 以能量價格作為條件輸入。

---

## 3. 訓練流程

### Stage A：骨幹 + 每步讀出 [預設]

每步損失加權總和：

```
L = Σ_t w_t · [ CE(p_t, y) + β · Brier(p_t, y) ]
    + γ · L_conf
    + δ · Σ_{t<T} KL( sg(p_T) ‖ p_t )      # 最後一步自我蒸餾給前面各步
    + η · L_distill(teacher)
```

- `w_t`：預設均等；若最終步準確率下降，改為偏重後段。
- `L_conf`：Gatekeeper（2025）的信心調整損失，對答錯的樣本把分布推向均勻、答對的樣本維持交叉熵（細節依原論文確認）。
- `sg`：stop-gradient。
- 預設超參：β=1、γ=0.5、δ=0.5（需掃描）。

### Stage B：每步溫度縮放 [定案]

骨幹凍結，在 held-out 集上擬合 T_t。這是**必須擊敗的基準**：Hao et al. 2026 發現 ImageNet 上溫度縮放即可把 ECE 從約 0.26 降到 0.03。

### Stage C：停止策略（骨幹凍結）[預設]

依序實作並比較：

1. 門檻規則：max-prob ≥ τ、熵 ≤ τ（無需訓練）。
2. ACT / PonderNet 式可微分停止。
3. 監督式：以 oracle 標籤訓練（見 TODO-1 的 oracle 定義）。
4. RL（PPO 或 REINFORCE），獎勵：

```
R = − Brier(p_τ, y) − λ · (τ + 1) / T_max     # τ = 停止的步數
```

用 Brier 作為 RL 獎勵的設計參考 RLCR（ICLR 2026）。

**重要實作建議**：骨幹凍結後，先對訓練子集與驗證集**快取每一步的 logits 與特徵**（h_t 的 CLS、pooled patch tokens）。之後所有停止策略都在快取上離線訓練，數分鐘即可完成一組實驗，方便大量比較 TODO-1 的設計。

### Stage D（選做）：聯合微調

骨幹與停止策略一起以小學習率微調，觀察是否進一步改善曲線。

---

## 4. 副線：仿皮層遞迴網路（科學驗證用）

- 骨幹：CORnet-RT 或 BLT 類卷積遞迴網路，ImageNet 或 ecoset 訓練；接上同一套讀出頭與停止策略。
- 一律 GroupNorm。
- 選做：Bidirectional PC（ICLR 2026）作慢速修正模組；能量下降量可作為另一種停止訊號（納入 TODO-1 的候選輸入）。
- 用途：Brain-Score、IT 物體解答時間（Kar et al. 2019）、MVT 相關性。

---

## 5. 資料集

| 用途 | 資料集 |
|---|---|
| 快速原型 | CIFAR-100、Tiny-ImageNet |
| 主要評估 | ImageNet-1K、ImageNetV2 |
| 穩健度 / 分布外 | ImageNet-C、ImageNet-R、ImageNet-A、ImageNet-Sketch |
| 人類難度 | Minimum Viewing Time（Mayo et al., NeurIPS 2023 Datasets & Benchmarks） |
| 大腦 | Brain-Score V4 / IT；Kar et al. 2019 物體解答時間 |

## 6. 比較對象

| 類別 | 方法 |
|---|---|
| 靜態模型 | DeiT-S/B、ViT-B、ConvNeXt-T；DINOv3 線性探測（上限參考） |
| 動態深度 / 早退 | MSDNet、A-ViT、AdaPerceiver、reViT 彈性深度、bViT |
| 停止規則 | max-prob 門檻、熵門檻（Spoerer 2020）、ACT（AdRNN 2023）、不確定性 RT 指標（Goetschalckx 2023）、BCE 停止頭（Adaptive Observation Control, ACCV 2026） |
| 校準 | 溫度縮放、Gatekeeper |
| 兩模型串接 | 小模型 + 大模型，以校準信心決定是否升級（Hao et al. 2026） |
| 仿皮層 | CORnet-S/RT、BL/BLT |

與兩模型串接的比較為**必做**：回答「一個遞迴模型的彈性深度」是否優於「一小一大兩個模型」。

## 7. 評估指標與規範

- **效率**
  - 平均 GFLOPs，**包含停止策略與讀出頭本身**。
  - 實際延遲與吞吐量：batch=1 與實際部署 batch 都要測；GPU（必做）與邊緣裝置如 Jetson（選做）。
  - 原因：Adaptive Observation Control（2026）發現批次推論下自適應執行沒有一致的吞吐量優勢。需實作依停止步數動態分組的批次推論。
- **準確率**：top-1；相同計算量下準確率；準確率–計算量曲線下面積。
- **校準**：每一步與停止時的 ECE、Brier、NLL；AURC（風險–覆蓋曲線）。
- **穩健度**：ImageNet-C mCE；ImageNet-R/-A 準確率；分布偏移下停止步數的變化（是否會在難資料上自動多想）。
- **人類與大腦**：停止步數 vs MVT 的 Spearman 相關；Brain-Score；IT 解答時間吻合度。
- **規範**：至少 3 個隨機種子；所有方法同一套評估程式；固定硬體與軟體版本記錄。

## 8. 消融實驗

1. 拿掉校準相關損失（Brier、L_conf）。
2. 拿掉溫度縮放。
3. 拿掉 λ 輸入（改為每個預算各訓練一個策略）。
4. 停止策略：門檻 vs ACT vs 監督式 vs RL。
5. T_max：4 / 8 / 12。
6. 自我蒸餾有無。
7. 停止策略的輸入與特徵來源（TODO-1 的所有變體）。

## 9. 風險與對策

| 風險 | 對策 |
|---|---|
| FLOPs 省了但實際時間沒省 | 同時報告 batch=1 延遲；實作依步數分組的動態批次；考慮 TODO-1 的「第 0 步一次預測總步數」版本 |
| 中間讀出拖累最終準確率 | 自我蒸餾、調整 w_t |
| 校準優勢在 ImageNet 上不明顯 | 主要在分布偏移資料上展示校準的好處 |
| 新穎性被質疑 | 主張聚焦「校準 + λ 條件化 + 人類/IT 驗證」 |
| 停止策略本身太貴 | 預算上限：停止策略成本 < 單步遞迴成本的 2%（見 TODO-1） |

## 10. 里程碑

| 週次 | 工作 | 產出 |
|---|---|---|
| 1–2 | 共用評估框架；FLOPs 與延遲量測工具 | eval 程式、量測腳本 |
| 3–4 | CIFAR-100 / Tiny-ImageNet 上的遞迴 ViT + 每步讀出 | 每步準確率與校準曲線 |
| 5–6 | 門檻基準、溫度縮放、快取每步輸出 | 基準曲線、特徵快取 |
| 7–9 | **TODO-1：停止 MLP 設計研究**（離線於快取上） | 設計比較表，選出 1–2 個版本 |
| 10–13 | ImageNet-1K 規模訓練；RL 停止策略；比較對象重現 | 主結果 |
| 14–16 | 穩健度、MVT、Brain-Score；副線 | 分析結果 |
| 17–18 | 消融、延遲實測、撰寫 | 論文初稿 |

---

## 11. 程式介面建議

```python
class RecurrentViT(nn.Module):
    def forward_step(self, tokens, t):        # -> tokens_next
        ...
    def readout(self, tokens, t):              # -> logits_t (未除溫度)
        ...

class HaltPolicy(nn.Module):
    def forward(self, question, state=None):   # -> halt_logit（或其他輸出，見 TODO-1）
        ...

def infer(x, model, policy, lam, T_max=8, temps=None):
    tokens = model.embed(x)
    feats0 = None
    for t in range(T_max):
        tokens = model.forward_step(tokens, t)
        p_t = softmax(model.readout(tokens, t) / temps[t])
        q = build_question(p_t, p_prev, t, lam)      # 機率統計量、步數、λ
        s = build_state(tokens, feats0, t)            # 依 TODO-1 的設計，可為 None
        if t == T_max - 1 or sigmoid(policy(q, s)) > 0.5:
            return p_t, t
        p_prev = p_t
```

---

## 12. TODO

### TODO-1：停止策略小型 MLP 的輸入輸出設計 ★ 主要待研究項目

**目標**：決定停止策略的輸入（question / state）、特徵來源、輸出形式，在「停止品質」與「額外成本」之間找最佳取捨。

#### 1a. Question / State 架構（沿用 Jev 的 question–state 概念）

- **question**：描述「現在要回答的問題」：目前步數 t、λ、機率統計量（top-k、熵、差距、與上一步的變化）。
- **state**：描述「模型目前看到什麼」：特徵向量。
- 候選做法：
  - 只用 question（= 2.3 節的預設基準）。
  - question + state 串接後送 MLP。
  - **特徵比對**：以 question 為 query 比對 state，或比對不同步的特徵（例：cos(h_t, h_{t-1})、‖h_t − h_{t-1}‖、cos(h_t, h_0)），以「表徵是否收斂」判斷是否停止。
    - 注意：若比對牽涉全部 patch tokens（如 cross-attention），**速度可能慢**；需量測成本，優先只比對 CLS 或 pooled 特徵。

#### 1b. 特徵（state）從何而來

| 代號 | 特徵來源 | 維度（ViT-B） | 額外成本 | 預期優缺點 |
|---|---|---|---|---|
| S0 | 不使用 state | 0 | 最低 | 基準；可能已足夠 |
| S1 | 只用第 0 步特徵 h_0 | 768 | 只算一次 | 可在第 0 步**一次預測總步數**，利於依步數分組批次推論（解決吞吐量問題）；但無法隨後續步驟修正 |
| S2 | 第 t 步特徵 h_t | 768 | 每步一次 MLP | 資訊最即時；成本隨步數累加 |
| S3 | h_t 與 h_{t-1} / h_0 的差或相似度 | 1–768 | 低 | 直接量測收斂；與 1a 的特徵比對對應 |
| S4 | 額外 feature extractor（例：小 CNN 吃低解析度影像） | 視設計 | 高 | 可能抵銷加速效果；僅作上限參考 |
| S5 | Random projection 降維（固定高斯矩陣，768 → 32/64/128） | 32–128 | 極低 | 不需訓練投影；與 S6 比較 |
| S6 | 學習式線性降維（768 → 64） | 64 | 極低 | S5 的對照 |
| S7 | 其他技術（見 1c） | — | — | 待評估 |

特徵本身的選擇：CLS token / patch tokens 平均 / 兩者串接，需一併比較。

#### 1c. 其他可評估的訊號

- 注意力圖的熵（各 head 平均）。
- 不同步之間預測的一致性（top-1 連續 k 步不變）。
- Conformal prediction 的預測集合大小。
- 特徵空間的 Mahalanobis 距離（OOD 分數）。
- Bidirectional PC 的能量下降量（副線）。
- MC dropout 的不確定性（成本高，僅作參考上限）。

#### 1d. 輸出形式

| 代號 | 輸出 | 用法 |
|---|---|---|
| O1 | 停止機率 π(stop) | 每步呼叫；最簡單 |
| O2 | 「再多算一步的預期收益」：P(答對 \| 繼續) − P(答對 \| 停) | 與 λ 比較決定停止；可解釋性高 |
| O3 | 總步數的分布（在第 0 步一次輸出） | 搭配 S1；支援依步數分組批次推論 |
| O4 | 剩餘步數的分布（每步更新） | O1 與 O3 的折衷 |

#### 1e. 訓練目標的選項

- **Oracle 標籤（監督式）**：對每個樣本找出「最早一個之後都答對的步數」t*；t ≥ t* 標為停止。若每一步都錯，標記為「跑滿 T_max」或「任何步都可停」（兩種處理需比較）。
- **RL**：見 Stage C 的獎勵。
- **ACT / PonderNet**：可微分版本。

#### 1f. 評估方式

1. 在**快取的每步輸出**上離線訓練所有組合（S × O × 訓練目標），骨幹凍結。
2. 每個組合報告：
   - 準確率–計算量曲線與曲線下面積
   - 與 **oracle 停止**的差距（oracle：每個樣本在 t* 停止）
   - 停止策略本身的 FLOPs 與延遲，占單步遞迴成本的百分比（預算上限 2%）
   - ImageNet-C / -R / -A 上的曲線（分布偏移時是否仍會正確地多想）
   - 停止步數 vs MVT 的相關
3. 先做 CIFAR-100 / Tiny-ImageNet，選出 1–2 個版本再上 ImageNet-1K。

#### 1g. 優先順序建議

1. S0 + O1（基準） → 2. S3 + O1（收斂訊號，便宜） → 3. S2 / S5 / S6 + O1 → 4. S1 + O3（吞吐量導向） → 5. O2 → 6. 其他（1c、S4、cross-attention 比對）。

### TODO-2：校準損失細節

- 確認 Gatekeeper 損失的原始定義與超參。
- 比較：Brier、Focal loss、label smoothing 對每步校準的影響。

### TODO-3：λ 的設計

- λ 的範圍與分布（log-uniform 的上下界）。
- λ 的輸入編碼：直接 log λ / Fourier features / 嵌入。
- 確認推論時掃描 λ 得到的曲線是否單調、平滑。

### TODO-4：延遲實測

- 實作依停止步數動態分組的批次推論。
- 比較 S1 + O3（事先預測步數）與逐步決策在吞吐量上的差異。

### TODO-5：確認資源

- reViT、bViT 是否有公開程式碼與權重；不可得時向使用者提問是否自行實作。
- AdaPerceiver、A-ViT、MSDNet 的官方實作。
- MVT 資料集取得方式與授權。

---

## 13. 參考文獻

### 遞迴 / 彈性深度骨幹
- Bulat, Ouali & Tzimiropoulos (2026). *One Block, Multiple Depths*（reViT）. <https://awesomepapers.io/computer-vision/papers/2610.12448>
- Byra et al. (2026). *Investigating Single-Block Recurrence in Vision Transformers*（bViT）. <https://arxiv.org/html/2605.10661>
- ViTRM (2026). <https://arxiv.org/html/2603.19503v2>（尚未細讀）

### 自適應計算
- Jajal et al. (2025). *AdaPerceiver*. <https://arxiv.org/html/2511.18105v1>
- Li, Wang & Wang (2026). *Adaptive Observation Control for Active Visual Recognition*（ACCV 2026）. <https://arxiv.org/html/2610.19624>
- Veerabadran et al. (2023). *Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels*（AdRNN, NeurIPS）. <https://pages.ucsd.edu/~desa/extrapolation_rnns_neurips_2023.pdf>
- Goetschalckx et al. (2023). *Computing a human-like reaction time metric from stable recurrent vision models*（NeurIPS）。
- Huang et al. (2018). MSDNet（ICLR）；Yin et al. (2022). A-ViT（CVPR）；Graves (2016). ACT；Banino et al. (2021). PonderNet。

### 校準與串接
- Rabanser et al. (2025). *GATEKEEPER*. <https://arxiv.org/html/2502.19335v2>
- Hao et al. (2026). *When Models Know When They Do Not Know*. <https://arxiv.org/html/2601.07965>
- Damani et al. (2026). *Beyond Binary Rewards*（RLCR, ICLR 2026）. <https://arxiv.org/abs/2507.16806>
- Guo et al. (2017). *On Calibration of Modern Neural Networks*（溫度縮放）。

### 仿皮層與神經科學
- Butkus, Ying & Kriegeskorte (2026). *How attention saves energy in vision*. <https://www.biorxiv.org/content/10.64898/2026.03.18.710397v1>
- Oliviers, Tang & Bogacz (2026). *Bidirectional Predictive Coding*（ICLR 2026）. <https://proceedings.iclr.cc/paper_files/paper/2026/file/6164b6e5352c139e9ddc1a98c09e4e4a-Paper-Conference.pdf>
- Spoerer et al. (2020). *Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision*（PLOS Comput Biol）. <https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008215>
- Kubilius et al. (2019). CORnet（NeurIPS）；Kar et al. (2019). *Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior*（Nat Neurosci）。
- Höfling et al. (2026). *Only Brains Align with Brains*（ICLR 2026）. <https://proceedings.iclr.cc/paper_files/paper/2026/file/b7868dedad7192f83c9efb042da43317-Paper-Conference.pdf>
- Leutenegger (2026)（含更正，正規化相關）. <https://arxiv.org/html/2605.30556>

### 資料集
- Mayo et al. (2023). *How hard are computer vision datasets? Calibrating dataset difficulty to viewing time*（NeurIPS D&B）. <https://groups.csail.mit.edu/infolab/publications/Image_difficulty_NeurIPS_2023_Dataset_Track.pdf>
- Hendrycks & Dietterich (2019). ImageNet-C；Hendrycks et al. (2021). ImageNet-R、ImageNet-A；Recht et al. (2019). ImageNetV2。
