# Round 5 (H100, pasted by the user)

Output of `python scripts/summarize.py c100_bvit_s c100_bvit_s_linw c100_bvit_s_quadw c100_bvit_s_linw_lr5e4 c100_vit_s_8blk`.
Halting studies on c100_bvit_s / c100_bvit_s_linw / c100_vit_s_8blk / c100_bvit_s_quadw seed 0 were run with
`--skip-train --halting` on existing checkpoints (code 0.5.0); c100_bvit_s_quadw seeds 1-2 and
c100_bvit_s_linw_lr5e4 were trained in this round. The header's `code ?` comes from a round-1 env.json
(fixed in 0.6.0: the header now shows the newest version).

Latency note: values were re-measured in this round and are about 2x the round-1/2 values for the same
models (e.g. bs256 full depth 8.4 ms vs 3.9 ms), with small spread within the round. The GPU is likely
shared with other jobs; latencies from different sessions are not comparable.

```
===== BEGIN FSV SUMMARY =====
code ? | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48
NOTE: runs use different code versions: ['0.2.0+4533fb7e05f9', '0.3.0+938e443f1e34', '0.4.0+1a80fc5c9de0', '0.5.0+ff191a1e54fb', 'None']

[c100_bvit_s] seeds 0,1,2 | params 1.86M | train 16.9±3.9 min | epoch 4.9±1.2s
  test acc%  32.91±0.43 54.23±0.27 58.48±0.08 59.22±0.08 59.34±0.07 59.22±0.18 59.10±0.20 58.85±0.31
  ECE raw%   9.67±0.30 9.04±0.33 7.22±0.33 6.87±0.25 6.89±0.40 7.01±0.63 7.33±0.76 7.68±0.76
  ECE TS%    1.65±0.28 2.26±0.46 2.59±0.23 2.79±0.34 2.90±0.31 3.15±0.09 3.16±0.17 3.18±0.20
  NLL TS     2.718±0.010 1.755±0.014 1.591±0.010 1.561±0.006 1.561±0.004 1.571±0.003 1.584±0.003 1.598±0.003
  AURC TS%   45.41±0.35 20.85±0.12 17.49±0.14 16.85±0.13 16.80±0.13 16.94±0.15 17.14±0.16 17.38±0.19
  temp       0.771±0.005 0.803±0.003 0.830±0.004 0.830±0.002 0.828±0.004 0.826±0.006 0.822±0.008 0.817±0.010
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.50±0.00 0.71±0.00 0.99±0.07 1.24±0.00 1.45±0.00 1.69±0.07 1.90±0.07 2.15±0.07
  lat bs256 ms 1.50±0.00 2.45±0.00 3.40±0.00 4.09±0.54 5.42±0.01 6.37±0.00 7.36±0.08 8.39±0.00
  holdout%   31.4±0.4 53.1±0.2 56.7±0.8 57.7±0.8 57.8±0.6 57.7±0.6 57.6±0.7 57.2±0.7
  final train loss 3.772±0.029 | AMP bf16
  best-step acc 59.34±0.07 | last-step acc 58.85±0.31
  AUC maxprob 54.43±0.20 entropy 54.05±0.20 fixed_depth 56.58±0.06
  oracle full 58.85±0.31% @ 1.039±0.003 GFLOPs | any 62.38±0.15% @ 0.358±0.003 GFLOPs

[c100_bvit_s_linw] seeds 0,1,2 | params 1.86M | train 10.2±0.0 min | epoch 2.9±0.0s
  test acc%  27.39±0.67 53.10±0.93 58.85±1.03 60.50±1.31 60.90±1.52 60.96±1.60 60.83±1.61 60.55±1.60
  ECE raw%   10.33±0.14 10.86±0.69 7.52±0.30 7.21±0.45 7.11±0.41 7.26±0.28 7.54±0.20 7.87±0.15
  ECE TS%    1.84±0.30 1.74±0.02 2.36±0.12 2.66±0.18 2.62±0.37 2.76±0.32 2.88±0.15 2.87±0.30
  NLL TS     2.975±0.032 1.823±0.036 1.574±0.049 1.515±0.058 1.503±0.060 1.507±0.061 1.518±0.060 1.532±0.058
  AURC TS%   53.87±0.91 22.22±0.80 17.22±1.17 16.08±1.31 15.83±1.34 15.86±1.31 16.03±1.25 16.27±1.22
  temp       0.727±0.004 0.785±0.009 0.825±0.002 0.830±0.003 0.830±0.004 0.829±0.004 0.826±0.004 0.821±0.004
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.51±0.00 0.72±0.00 0.97±0.07 1.17±0.07 1.42±0.07 1.63±0.07 1.87±0.01 2.09±0.00
  lat bs256 ms 1.51±0.00 2.46±0.00 3.41±0.01 4.48±0.01 5.43±0.01 6.38±0.01 7.46±0.02 8.41±0.02
  holdout%   26.6±1.3 50.8±1.1 57.2±1.6 58.8±1.6 58.9±1.8 59.1±1.8 58.9±1.7 58.4±1.8
  final train loss 3.628±0.023 | AMP bf16
  best-step acc 60.97±1.58 | last-step acc 60.55±1.60
  AUC maxprob 55.15±1.43 entropy 54.68±1.42 fixed_depth 57.07±1.21
  oracle full 60.55±1.60% @ 1.041±0.017 GFLOPs | any 63.61±1.44% @ 0.388±0.010 GFLOPs

[c100_bvit_s_linw_lr5e4] seeds 0,1,2 | params 1.86M | train 21.8±0.0 min | epoch 6.4±0.0s
  test acc%  27.49±0.23 53.34±0.34 59.35±0.48 60.79±0.38 61.14±0.44 61.23±0.40 61.17±0.28 60.90±0.34
  ECE raw%   10.27±0.30 11.38±0.25 8.57±0.37 7.51±0.30 6.98±0.39 6.89±0.39 7.05±0.39 7.23±0.34
  ECE TS%    1.88±0.53 2.36±0.17 2.38±0.33 2.47±0.41 2.72±0.46 2.79±0.41 2.89±0.28 3.06±0.05
  NLL TS     2.959±0.003 1.798±0.004 1.554±0.012 1.498±0.010 1.488±0.010 1.492±0.010 1.503±0.010 1.518±0.010
  AURC TS%   53.53±0.64 21.89±0.14 16.71±0.30 15.60±0.24 15.33±0.22 15.36±0.22 15.51±0.20 15.75±0.22
  temp       0.727±0.003 0.768±0.002 0.808±0.006 0.823±0.005 0.829±0.003 0.832±0.002 0.831±0.001 0.828±0.001
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.46±0.07 0.66±0.08 0.99±0.07 1.22±0.04 1.37±0.14 1.64±0.03 1.82±0.07 2.07±0.00
  lat bs256 ms 1.51±0.00 2.45±0.00 3.40±0.01 4.43±0.08 5.42±0.01 6.37±0.01 7.45±0.00 8.40±0.01
  holdout%   26.7±0.2 52.1±0.4 57.9±0.4 59.5±0.7 60.0±0.8 60.0±0.6 59.8±0.5 59.6±0.4
  final train loss 3.581±0.041 | AMP bf16
  best-step acc 61.25±0.36 | last-step acc 60.90±0.34
  AUC maxprob 55.50±0.24 entropy 55.05±0.22 fixed_depth 57.35±0.30
  oracle full 60.90±0.34% @ 1.036±0.005 GFLOPs | any 64.04±0.26% @ 0.388±0.001 GFLOPs

[c100_bvit_s_quadw] seeds 0,1,2 | params 1.86M | train 11.6±2.4 min | epoch 3.3±0.7s
  test acc%  25.01±0.36 51.11±0.18 58.73±0.66 60.70±0.81 61.37±1.03 61.40±1.14 61.30±1.05 61.07±0.98
  ECE raw%   10.90±0.40 11.90±0.43 7.91±0.16 6.89±0.06 6.72±0.14 6.69±0.34 6.93±0.42 7.33±0.52
  ECE TS%    2.33±0.33 1.77±0.12 2.28±0.48 2.60±0.23 2.60±0.11 2.85±0.22 3.08±0.19 3.19±0.24
  NLL TS     3.102±0.020 1.907±0.006 1.590±0.025 1.509±0.032 1.488±0.035 1.488±0.036 1.496±0.037 1.509±0.036
  AURC TS%   58.06±0.29 24.19±0.17 17.45±0.55 15.98±0.58 15.55±0.64 15.51±0.65 15.59±0.61 15.77±0.61
  temp       0.694±0.009 0.765±0.004 0.820±0.006 0.830±0.006 0.832±0.006 0.831±0.005 0.828±0.004 0.822±0.003
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.45±0.10 0.67±0.08 0.87±0.09 1.13±0.12 1.35±0.18 1.56±0.18 1.73±0.24 2.01±0.22
  lat bs256 ms 1.21±0.51 2.01±0.78 2.80±1.06 3.67±1.41 4.48±1.64 5.26±1.96 6.15±2.24 6.91±2.58
  holdout%   24.1±0.4 49.5±0.5 57.3±0.5 59.5±0.6 60.3±0.8 60.3±0.9 60.2±0.9 60.1±1.0
  final train loss 3.572±0.009 | AMP bf16
  best-step acc 61.46±1.10 | last-step acc 61.07±0.98
  AUC maxprob 55.03±0.78 entropy 54.49±0.82 fixed_depth 56.87±0.75
  oracle full 61.07±0.98% @ 1.048±0.011 GFLOPs | any 63.93±0.86% @ 0.404±0.005 GFLOPs

[c100_vit_s_8blk] seeds 0,1,2 | params 14.28M | train 21.2±1.4 min | epoch 5.2±0.6s
  test acc%  33.05±0.19 53.17±0.45 58.98±0.49 61.28±0.11 62.01±0.54 62.30±0.64 62.45±0.53 62.62±0.71
  ECE raw%   8.42±0.33 7.93±0.18 5.47±0.31 5.25±0.25 5.07±0.59 4.58±0.78 3.98±0.55 3.57±0.48
  ECE TS%    1.34±0.18 1.26±0.27 1.60±0.35 2.18±0.26 2.43±0.10 2.68±0.13 3.03±0.04 3.26±0.32
  NLL TS     2.709±0.007 1.811±0.013 1.593±0.020 1.531±0.013 1.526±0.021 1.533±0.017 1.544±0.018 1.552±0.016
  AURC TS%   44.93±0.32 21.68±0.14 16.91±0.37 15.49±0.14 15.10±0.26 15.12±0.30 15.21±0.34 15.33±0.35
  temp       0.808±0.007 0.833±0.001 0.867±0.005 0.869±0.005 0.875±0.008 0.882±0.010 0.890±0.007 0.898±0.006
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.51±0.00 0.78±0.07 1.01±0.06 1.33±0.01 1.56±0.01 1.88±0.08 2.15±0.02 2.39±0.02
  lat bs256 ms 1.51±0.00 2.48±0.00 3.58±0.00 4.57±0.01 5.66±0.01 6.67±0.09 7.73±0.01 8.83±0.02
  holdout%   31.2±0.7 51.9±0.4 57.3±0.6 59.4±0.2 60.5±0.6 60.9±0.3 61.1±0.4 61.1±0.4
  final train loss 3.533±0.030 | AMP bf16
  best-step acc 62.63±0.69 | last-step acc 62.62±0.71
  AUC maxprob 57.67±0.45 entropy 57.11±0.48 fixed_depth 58.30±0.39
  oracle full 62.62±0.71% @ 1.041±0.008 GFLOPs | any 66.40±0.44% @ 0.422±0.004 GFLOPs

[halting c100_bvit_s] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3
  fixed_depth           56.58±0.06  54.23±0.27  58.48±0.08  59.22±0.08
  maxprob               54.43±0.20  47.22±0.14  53.17±0.23  56.59±0.30
  entropy               54.05±0.20  46.39±0.10  52.43±0.15  55.89±0.38
  S0-reward             56.92±0.03  54.68±0.35  58.90±0.07  59.29±0.06
  S0-reward_acc         57.12±0.06  55.07±0.13  59.08±0.11  59.31±0.07
  S0-oracle_full        54.59±0.31  47.48±0.58  53.94±0.54  56.88±0.56
  S0-oracle_any         56.84±0.20  54.88±0.14  58.82±0.25  58.97±0.25
  S3-reward             56.79±0.26  53.95±1.53  58.83±0.04  59.26±0.06
  S3-reward_acc         57.10±0.06  55.14±0.12  59.05±0.09  59.29±0.07
  S3-oracle_full        54.60±0.30  47.60±0.54  53.82±0.54  56.92±0.53
  S3-oracle_any         56.86±0.20  54.93±0.17  58.82±0.22  59.00±0.27
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0057% of one recurrent step; oracle any 62.38±0.15% @ 0.358±0.003 GFLOPs

[halting c100_bvit_s_linw] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3
  fixed_depth           57.07±1.21  53.10±0.93  58.85±1.03  60.50±1.31
  maxprob               55.15±1.43  46.51±1.13  54.00±1.53  58.01±1.70
  entropy               54.68±1.42  45.40±1.25  52.82±1.43  57.24±1.68
  S0-reward             57.47±1.29  53.36±0.90  59.64±1.52  60.74±1.53
  S0-reward_acc         57.71±1.31  53.73±0.85  60.13±1.21  60.85±1.59
  S0-oracle_full        55.14±1.29  45.93±0.49  54.41±1.36  58.28±1.72
  S0-oracle_any         57.71±1.28  53.74±1.10  60.04±1.28  60.85±1.53
  S3-reward             57.34±1.55  52.48±2.54  59.58±1.62  60.72±1.51
  S3-reward_acc         57.70±1.31  53.69±1.01  60.15±1.21  60.84±1.57
  S3-oracle_full        55.11±1.28  45.86±0.42  54.33±1.33  58.26±1.67
  S3-oracle_any         57.71±1.28  53.77±0.98  60.03±1.29  60.86±1.54
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0057% of one recurrent step; oracle any 63.61±1.44% @ 0.388±0.010 GFLOPs

[halting c100_bvit_s_linw_lr5e4] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3
  fixed_depth           57.35±0.30  53.34±0.34  59.35±0.48  60.79±0.38
  maxprob               55.50±0.24  46.89±0.28  54.54±0.38  58.28±0.33
  entropy               55.05±0.22  45.91±0.29  53.30±0.34  57.54±0.48
  S0-reward             57.70±0.32  53.74±0.38  60.12±0.62  60.99±0.34
  S0-reward_acc         58.02±0.32  53.95±0.43  60.55±0.42  61.16±0.40
  S0-oracle_full        55.62±0.21  47.10±0.06  54.84±0.56  58.52±0.34
  S0-oracle_any         57.95±0.22  53.87±0.46  60.39±0.36  61.07±0.33
  S3-reward             57.71±0.34  53.64±0.32  60.08±0.62  61.00±0.36
  S3-reward_acc         57.98±0.40  53.89±0.28  60.44±0.63  61.12±0.51
  S3-oracle_full        55.59±0.20  47.01±0.08  54.75±0.50  58.46±0.26
  S3-oracle_any         57.94±0.26  53.81±0.35  60.36±0.43  61.07±0.35
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0057% of one recurrent step; oracle any 64.04±0.26% @ 0.388±0.001 GFLOPs

[halting c100_bvit_s_quadw] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3
  fixed_depth           56.87±0.75  51.11±0.18  58.73±0.66  60.70±0.81
  maxprob               55.03±0.78  45.27±0.52  53.74±0.90  58.29±0.87
  entropy               54.49±0.82  44.08±0.59  52.25±0.82  57.30±0.84
  S0-reward             57.30±0.81  51.36±0.14  59.70±0.78  61.21±1.14
  S0-reward_acc         57.62±0.85  51.76±0.33  60.31±0.94  61.35±1.03
  S0-oracle_full        55.17±0.80  45.05±0.42  54.25±0.96  58.85±0.84
  S0-oracle_any         57.47±0.80  51.74±0.37  60.07±0.79  61.19±1.04
  S3-reward             57.28±0.77  51.11±0.44  59.72±0.75  61.22±1.07
  S3-reward_acc         57.59±0.85  51.62±0.44  60.26±0.92  61.36±1.04
  S3-oracle_full        55.16±0.79  44.93±0.33  54.31±0.80  58.78±0.97
  S3-oracle_any         57.48±0.80  51.83±0.36  60.03±0.75  61.18±1.04
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0057% of one recurrent step; oracle any 63.93±0.86% @ 0.404±0.005 GFLOPs

[halting c100_vit_s_8blk] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3
  fixed_depth           58.30±0.39  53.17±0.45  58.98±0.49  61.28±0.11
  maxprob               57.67±0.45  49.15±0.39  56.55±0.29  60.47±0.57
  entropy               57.11±0.48  47.98±0.41  55.22±0.49  59.56±0.56
  S0-reward             58.83±0.29  53.55±0.47  59.87±0.31  61.80±0.11
  S0-reward_acc         59.39±0.47  54.30±0.53  60.95±0.51  62.27±0.50
  S0-oracle_full        57.69±0.44  49.08±0.61  56.69±0.42  60.62±0.42
  S0-oracle_any         59.44±0.50  54.35±0.42  61.00±0.27  62.42±0.67
  S3-reward             58.87±0.24  53.54±0.39  59.94±0.37  61.78±0.02
  S3-reward_acc         59.35±0.43  54.28±0.53  60.80±0.44  62.32±0.42
  S3-oracle_full        57.67±0.45  48.86±0.65  56.68±0.41  60.65±0.39
  S3-oracle_any         59.41±0.49  54.23±0.49  60.92±0.28  62.38±0.67
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0057% of one recurrent step; oracle any 66.40±0.44% @ 0.422±0.004 GFLOPs
===== END FSV SUMMARY =====
```
