## 2026-11-27T06:23:30Z nullcal proposal:5828ca9db121e14c1ab92cb7586730b8cd236f409b34f5723ac50853526f1061

Hypothesis: on independent seeded Zipf text (vocabulary 5000, 30 tokens per example, proxy of 20000 tokens), the share of 100 null benchmarks of 50 examples that the pre-registered n-gram rule (threshold 0.05, bootstrap CI lower bound above threshold) flags is exactly 0 at n=13 and at least 0.95 at n=2, so the false-positive floor depends on n rather than on the benchmark.

Metric: /null_flag_rate/by_n/13 (also by_n/2). Expected: flag rate at n=13 equals 0.0; flag rate at n=2 is at least 0.95.

Falsifier: a flag rate above 0 at n=13, or below 0.95 at n=2, on the seeded null set.

Run: cargo run --locked -q --release -- nullcal --seed 7 --benchmarks 100 --examples 50 --out results/635-nullcal.json (bootstrap 200 iterations per benchmark, default n sweep 1,2,3,4,5,8,13).

## 2026-11-28T01:51:23Z evasion proposal:db82a50f729ee48243d25534729b8b57efae5132dad7559c38618ae9729e02a8

Hypothesis: on seeded Zipf text (vocabulary 5000, 30 tokens per example, 50 examples per benchmark, 100 benchmarks, 50% of items leaked into the proxy), the pre-registered n-gram rule at n=13 flags 100% of benchmarks when leaks are verbatim but at most 5% when each token of a leaked item is independently rewritten at rate 0.5 (the paraphrase class), while at n=2 the same paraphrase class is flagged in at least 95% of benchmarks only because the n=2 rule already false-flags at 1.0 under the null.

Metric: /recall_by_edit_rate/paraphrase/recall (n=13, edit rate 0.5). Expected: at most 0.05, with verbatim recall at n=13 equal to 1.0.

Falsifier: paraphrase-class recall at n=13 above 0.05, or verbatim recall at n=13 below 1.0, on the seeded set.

Run: cargo run --locked -q --release -- evasion --seed 7 --benchmarks 100 --examples 50 --leak-share 0.5 --out results/775-evasion.json (classes: verbatim edit rate 0, light 0.1, moderate 0.25, paraphrase 0.5; n sweep 2,3,4,5,8,13; bootstrap 200).

## 2026-11-28T02:22:38Z leakcurve proposal:4dab30be9f0094582059d5223964264ddb607f6e877f41a97209d4f941e81f3e

Hypothesis: on seeded Zipf text (vocabulary 5000, 30 tokens per example, 50 examples per benchmark, 100 benchmarks, bootstrap 200), verbatim leakage of 10% of a benchmark is flagged by the n=13 pre-registered rule in at most 50% of benchmarks, because the bootstrap CI lower bound of a 0/1-heavy per-item overlap falls below the 0.05 threshold, while verbatim leakage of 20% is flagged in at least 95%; so the detection limit sits well above the 0.05 threshold itself.

Metric: /recall_by_leak_share/verbatim/by_n/13/0.10/recall. Expected: at most 0.50; recall at 0.20 (same path with 0.20) at least 0.95.

Falsifier: verbatim recall at n=13 above 0.50 at leak share 0.10, or below 0.95 at leak share 0.20, on the seeded set.

Run: cargo run --locked -q --release -- leakcurve --seed 7 --benchmarks 100 --examples 50 --out results/782-leakcurve.json (shares 0.02,0.05,0.1,0.2,0.3,0.5; classes verbatim and light at rewrite rate 0.1; n sweep 5,8,13).
