# contamcheck

**Benchmark contamination auditor.** Given a model and a benchmark,
measure how much of the benchmark the model has already memorized. Three
probes, each with a pre-registered flagging rule: n-gram overlap against a
training-set proxy, echo-canary completion of held-out suffixes, and
loss-delta between the original benchmark example and a semantic
paraphrase. Reports a contamination-adjusted score table alongside the
published numbers.

![status: initial scaffold](https://img.shields.io/badge/status-initial%20scaffold-lightgrey)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why

Published contamination studies (Oren et al. 2023, Riddell et al. 2024,
Dong et al. 2024) each pick one method and one benchmark. There is no
reusable tool. Every new frontier model release triggers accusations with
no shared instrument to settle them, and the leaderboards keep publishing
the un-adjusted number.

## What is in this scaffold

- `ngram_overlap(reference, proxy, n)` — fraction of distinct n-grams of
  the reference that appear in the training-set proxy.
- `echo_canary_score(model_output, held_out_suffix)` — longest common
  prefix divided by suffix length.
- `loss_delta(original_loss, paraphrase_loss)` — positive when the model
  is specifically fluent on the original wording.
- `flagged(mean, ci_lower, threshold)` — the per-probe flagging rule.
- `bootstrap_mean_ci(values, iters, seed)` — percentile bootstrap over
  per-example signals; NaN rows (e.g. echo-canary "n/a" on models that
  refuse to continue) are dropped before resampling.
- `aggregate(results, iters, seed, thresholds) -> BenchmarkReport` —
  applies the pre-reg flagging rule to all three probes and emits a
  per-benchmark verdict (any probe flag → contaminated).
- `Thresholds::default()` returns the pre-registered table:
  n-gram 0.05, echo-canary 0.3, loss-delta 0.2.
- `contamcheck probe …` runs the three per-example probes on one
  Example; `contamcheck aggregate <probes.jsonl>` reads the probe output
  of a benchmark and prints the BenchmarkReport.
- `docs/PREREGISTRATION.md` locks the thresholds.

Not in this commit (openly tracked):

- `contamcheck audit --model <endpoint> --benchmark <path>` end-to-end
  driver.
- Benchmark adapters: MMLU, GSM8K, MATH, HumanEval, HellaSwag.
- A score-table rendering helper that produces the "published vs
  contamination-adjusted" comparison.

## Usage (scaffold)

```sh
cargo install --path .
# per-example probe:
contamcheck probe example.json proxy.txt model_output.txt suffix.txt \
  --original-loss 0.5 --paraphrase-loss 2.0 --n 13 > probes.jsonl
# per-benchmark aggregate with the pre-registered flagging rule:
contamcheck aggregate probes.jsonl
```

## License

MIT. See `LICENSE`.

## Local-model study, live

Pre-registered in `docs/STUDY-local-models-2026-10.md` (four Q8_0 models, GSM8K sample of 300 and all of HumanEval; echo canary and loss delta; n-gram not applicable). Per-item echo scores and the loss-delta result files are in `results/study-2026-10/`; model continuations are never stored. One deviation, stated here and in every loss-delta file: the serving stack does not return prompt log-probabilities, so the loss-delta interval is a bootstrap over `llama-perplexity`'s fixed-size scored chunks of the concatenated texts, not over items.

<!-- CONTAM:START -->
| Model | Benchmark | Echo canary mean [95% CI] (n, n/a) | Echo flag (>0.3, CI lo >0) | Loss delta (nats/token) [95% CI over chunks] | Loss flag (>0.2, CI lo >0) |
|---|---|---|---|---|---|
| llama-3.1-8b-instruct | gsm8k | 0.040 [0.026, 0.057] (300, 0) | not flagged | not run | — |
| qwen3-8b-stock | gsm8k | 0.033 [0.020, 0.047] (300, 0) | not flagged | not run | — |
| qwen3-8b-stock | humaneval | 0.034 [0.013, 0.060] (164, 0) | not flagged | not run | — |

| Pre-registered hypothesis | Verdict |
|---|---|
| H1 some model flagged on GSM8K by loss delta | not evaluated: no loss-delta cells yet |
| H2 some model flagged on HumanEval by echo canary | not supported: no model flagged (1 cells) |
| H3 ordering | pending: needs the published GSM8K scores with sources recorded in this file before it is read |
<!-- CONTAM:END -->
