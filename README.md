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
| gemma-2-9b-it | gsm8k | 0.042 [0.028, 0.059] (300, 0) | not flagged | +0.3621 [+0.2715, +0.4519] (179 items, 83 chunks), paraphraser: excluded from headline | flagged |
| gemma-2-9b-it | humaneval | 0.086 [0.050, 0.128] (157, 7) | not flagged | +0.0237 [-0.1172, +0.1771] (142 items, 34 chunks), paraphraser: excluded from headline | not flagged |
| llama-3.1-8b-instruct | gsm8k | 0.040 [0.026, 0.057] (300, 0) | not flagged | +0.4479 [+0.3551, +0.5414] (179 items, 70 chunks) | flagged |
| llama-3.1-8b-instruct | humaneval | 0.035 [0.014, 0.062] (164, 0) | not flagged | +0.1626 [-0.0042, +0.3287] (142 items, 29 chunks) | not flagged |
| mistral-7b-instruct-v0.3 | gsm8k | 0.028 [0.018, 0.040] (300, 0) | not flagged | +0.4057 [+0.3073, +0.4994] (179 items, 86 chunks) | flagged |
| mistral-7b-instruct-v0.3 | humaneval | 0.035 [0.014, 0.062] (164, 0) | not flagged | +0.1710 [+0.0135, +0.3189] (142 items, 37 chunks) | not flagged |
| qwen3-8b-stock | gsm8k | 0.033 [0.020, 0.047] (300, 0) | not flagged | +0.3863 [+0.2882, +0.4838] (179 items, 82 chunks) | flagged |
| qwen3-8b-stock | humaneval | 0.034 [0.013, 0.060] (164, 0) | not flagged | +0.1810 [+0.0004, +0.3541] (142 items, 29 chunks) | not flagged |

| Pre-registered hypothesis | Verdict |
|---|---|
| H1 some model flagged on GSM8K by loss delta | supported: flagged llama-3.1-8b-instruct, mistral-7b-instruct-v0.3, qwen3-8b-stock |
| H2 some model flagged on HumanEval by echo canary | not supported: no model flagged (4 cells) |
| H3 ordering | no verdict: the top published model is not separable (qwen3-8b vs llama-3.1-8b-instruct: 82.9 (5-shot, base) vs 84.5 (8-shot CoT, instruct) are not comparable setups and the unverified 89.84 would reverse them); loss-delta ranking: llama-3.1-8b-instruct +0.4479 > mistral-7b-instruct-v0.3 +0.4057 > qwen3-8b-stock +0.3863 > gemma-2-9b-it +0.3621 (paraphraser) |
<!-- CONTAM:END -->

**Reading the table (written after the run; thresholds and hypotheses were fixed before).** Four things a reader needs. (1) *Loss-delta intervals are over chunks, not items*: this serving stack returns no prompt log-probabilities, so each cell concatenates the items and scores `llama-perplexity`'s fixed 128-token chunks; the interval is a bootstrap over those chunks. (2) *Sign*: the crate's pre-registration defines loss delta as loss(paraphrase) minus loss(original), positive when the original wording is the easier one; the study text listed the two terms in the other order, and the crate's definition was applied. The first committed Qwen3-8B GSM8K cell file carried the inverted sign in its summary field; the per-chunk arrays it was computed from were unaffected, and the summary recomputes every cell from those arrays. (3) *Paraphrases*: gemma-2-9b-it accepted 179 of 300 GSM8K references and 142 of 164 HumanEval solutions under the pre-registered length and 13-gram rules (rejects are committed with reasons); gemma's own loss-delta cells are shown but excluded from the H1 headline. (4) *What the GSM8K loss deltas most plausibly mean*: every model, including Mistral-7B, which has the lowest published GSM8K score of the four, shows a delta between +0.36 and +0.45 nats per token on GSM8K, and the intervals overlap. A signal that is the same size for every model is not evidence that any one of them memorised the test set; it is evidence that GSM8K's reference answers, with their `<<16-3-4=9>>` calculator annotations, are more predictable than a natural-language paraphrase of them for any model. H1 is therefore reported as supported by the pre-registered rule and read as a format effect, not contamination; the rule has no between-model control, and a follow-up should add one (for example, paraphrasing only the prose and keeping the annotations). HumanEval echo canaries are all below 0.09 (gemma returned nothing on 7 items, recorded as n/a), so H2 is not supported. H3 has no verdict because the published GSM8K figures do not separate Qwen3-8B from Llama-3.1-8B-Instruct across comparable setups (`results/study-2026-10/published-gsm8k.json`); the observed GSM8K loss-delta ranking is Llama-3.1 > Mistral > Qwen3-8B > gemma.

