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
- `flagged(mean, ci_lower, threshold)` — the per-benchmark flagging rule.
- `contamcheck probe …` CLI reads one example and prints the three
  per-example scores.
- `docs/PREREGISTRATION.md` locks the thresholds.

Not in this commit (openly tracked):

- `contamcheck audit --model <endpoint> --benchmark <path>` end-to-end
  driver.
- Bootstrap 95% CI aggregator over N examples.
- Benchmark adapters: MMLU, GSM8K, MATH, HumanEval, HellaSwag.
- A score-table rendering helper that produces the "published vs
  contamination-adjusted" comparison.

## Usage (scaffold)

```sh
cargo install --path .
contamcheck probe example.json proxy.txt model_output.txt suffix.txt \
  --original-loss 0.5 --paraphrase-loss 2.0 --n 13
```

## License

MIT. See `LICENSE`.
