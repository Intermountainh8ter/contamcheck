# contamcheck pre-registration

Locked before the first live-benchmark audit result is committed.
Append-only.

## 1. Scope

The population under audit is **publicly listed language models** with
either a HuggingFace repo id or an OpenAI-compatible endpoint. For each
model and each benchmark we run three contamination probes and report
the per-benchmark flagging verdict with a 95% bootstrap CI over
examples. The TEST split of the benchmark is read once per model.

## 2. The three probes

**N-gram overlap.** For each benchmark example, compute the fraction of
distinct 13-grams of the reference that appear anywhere in the proxy
training set. The proxy is a document fixed at `fixtures/proxy_<name>.txt`
with a `fixtures/proxy_<name>.md` describing its provenance (what public
training-set subset or scrape it approximates). Smaller n is sometimes
useful; the pre-registered probe is n=13 and the deviation is reported
as a new row, not a replacement.

**Echo canary.** The benchmark example's reference is split at a fixed
cut point (default: 20 tokens before end). The model is prompted with
the example minus the held-out suffix and asked to continue. The signal
is the longest-common-prefix of the model's output and the held-out
suffix, divided by the suffix length.

**Loss delta.** The benchmark example's reference is replaced by a
semantic paraphrase committed in `fixtures/paraphrases_<benchmark>.jsonl`
(generated once, hash-committed, never regenerated). The signal is
`loss(paraphrase) - loss(original)`. A model that specifically memorized
the original wording has a positive loss delta.

## 3. Flagging rule

A benchmark is **flagged contaminated by a probe** when the per-example
signal across the benchmark has:

- mean > `threshold`, AND
- 95% bootstrap CI lower bound > 0.

Default thresholds, locked:

| probe | threshold |
|-------|-----------|
| n-gram overlap (n=13) | 0.05 |
| echo canary | 0.3 |
| loss delta | 0.2 |

A model is **contaminated on a benchmark** when any probe flags it. The
report shows all three probes side by side; one flag is the finding.
We do not require unanimity across probes to publish a flag, but we do
require all three probes to run.

## 4. Multiple comparisons

Across benchmarks for the same model, p-values are *not* pooled. Each
benchmark is its own row. Across models for the same benchmark, we use
Holm-Bonferroni within a probe family.

## 5. What this repo refuses to do

- No measurement without a committed paraphrase file. Generating
  paraphrases on the fly during an audit is tuning on the test.
- No measurement with a vendor-provided proxy. The proxy document must
  be a public corpus (The Pile subset, Common Crawl snapshot) committed
  in `fixtures/`.
- No contamination-adjusted score claims. We report the probe flags;
  what the model's "real" score would be without the flagged examples
  is a separate question with its own methodology.

## 6. Threats to validity

- **Proxy choice.** The training-set proxy is an approximation.
  Over-approximating produces false positives; under-approximating,
  false negatives. We publish each result with the proxy's hash so
  swapping proxies is a different row.
- **Paraphrase quality.** A bad paraphrase makes the loss-delta probe
  meaningless. We generate paraphrases once with a strong model under
  pre-registered prompts, commit them, and never regenerate.
- **Models that refuse completion.** The echo-canary probe is only
  meaningful on models that continue the given prompt. Chat-tuned
  models that refuse are reported with `echo_canary = n/a`, not zero.
