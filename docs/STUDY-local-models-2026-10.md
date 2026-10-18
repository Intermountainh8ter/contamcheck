# Study: how much of GSM8K and HumanEval have popular open-weight models memorised?

Pre-registered 2026-10-18, before any probe is run. Append-only below the line. The probe
thresholds are the crate's (docs/PREREGISTRATION.md): echo canary 0.3, loss delta 0.2, each
flagged only when the mean exceeds the threshold and the 95% CI lower bound is above zero.

## Models (all Q8_0 GGUF, served or scored with llama.cpp 091b6a7 on gfx906)

Qwen3-8B (stock), Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, gemma-2-9b-it. The ablated
twins are not part of this study.

## Benchmarks (downloaded 2026-10-18, hashes fixed)

- GSM8K test split, 1,319 items, `gsm8k-test.jsonl` sha256 prefix `3730d312f6e34405`. A seeded
  sample of 300 items (Pcg64Mcg seed 42) is probed, for compute; the sample ids are committed.
- HumanEval, 164 problems, `HumanEval.jsonl` sha256 prefix `1d49078ba3e2b196`. All 164 are probed.

The benchmark files themselves are not committed to this repository (they are public and large);
their hashes are, and the probe inputs are reconstructed from them by the driver.

## Probes

1. **Echo canary.** The model is given the first half of the reference text (GSM8K: the question
   plus the first half of the worked answer; HumanEval: the prompt plus the first half of the
   canonical solution) and asked to continue at temperature 0, max 128 tokens. Score = longest
   common prefix of the continuation with the held-out second half, divided by the held-out length.
   A model that refuses or returns empty output is recorded as n/a, not zero.
2. **Loss delta.** Per-token mean negative log-likelihood of the original reference minus that of a
   paraphrase, both computed with `llama-perplexity` on the same model at the same context size.
   Paraphrases are generated once, before any loss is computed, by gemma-2-9b-it at temperature 0
   with a fixed instruction ("rewrite in different words, same meaning, same length"), and are
   accepted only if their token length is within 0.7x to 1.3x of the original and they share no
   13-gram with it; rejected items are excluded from the loss-delta probe and counted.
   gemma-2-9b-it's own loss-delta cell is therefore reported as potentially biased (it wrote the
   paraphrases) and excluded from the headline.
3. **N-gram overlap.** Requires a training-set proxy. No public proxy exists for these models'
   training data, so this probe is reported as not applicable for every cell. It is not estimated.

## Hypotheses, fixed now

- **H1.** At least one of the four models is flagged on GSM8K by the loss-delta probe.
- **H2.** At least one of the four models is flagged on HumanEval by the echo-canary probe.
- **H3 (ordering).** The model with the highest published GSM8K score among the four has the
  highest GSM8K loss delta. Reported as supported or not; no threshold beyond the ordering.

Descriptive: per-model, per-benchmark means with intervals for both applicable probes; count of
paraphrases rejected; the Holm-corrected comparison across the four models (`contamcheck holm`).

## Rules

- Thresholds are the crate's and are not changed for this study.
- Paraphrases are committed (they are not harmful content) with their hashes; model continuations
  are not committed, only the per-item scores.
- A benchmark cell with fewer than 100 scored items is reported but carries no verdict.
- The driver that turns benchmark rows into probe inputs is Rust, lives in this repository, and
  records its own version and the server build in every result file.

## What this study cannot say

Whether memorisation came from the benchmark itself or from the many web pages that quote it;
whether any model's published score is wrong. It measures memorisation of the test wording.
