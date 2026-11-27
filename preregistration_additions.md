## 2026-11-27T06:23:30Z nullcal proposal:5828ca9db121e14c1ab92cb7586730b8cd236f409b34f5723ac50853526f1061

Hypothesis: on independent seeded Zipf text (vocabulary 5000, 30 tokens per example, proxy of 20000 tokens), the share of 100 null benchmarks of 50 examples that the pre-registered n-gram rule (threshold 0.05, bootstrap CI lower bound above threshold) flags is exactly 0 at n=13 and at least 0.95 at n=2, so the false-positive floor depends on n rather than on the benchmark.

Metric: /null_flag_rate/by_n/13 (also by_n/2). Expected: flag rate at n=13 equals 0.0; flag rate at n=2 is at least 0.95.

Falsifier: a flag rate above 0 at n=13, or below 0.95 at n=2, on the seeded null set.

Run: cargo run --locked -q --release -- nullcal --seed 7 --benchmarks 100 --examples 50 --out results/635-nullcal.json (bootstrap 200 iterations per benchmark, default n sweep 1,2,3,4,5,8,13).
