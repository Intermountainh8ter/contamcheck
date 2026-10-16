//! Core primitives for contamination auditing.
//!
//! This crate ships three probe primitives: n-gram overlap against a
//! training-set proxy, echo-canary completion scoring, and loss-delta
//! comparison between the original benchmark example and a semantic
//! paraphrase. The per-benchmark aggregator reports mean + 95% percentile
//! bootstrap CI per probe and applies the pre-registered flagging rule
//! (mean > threshold AND CI lower bound > 0); see
//! `docs/PREREGISTRATION.md`.

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A single benchmark example — the thing we audit for contamination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    pub prompt: String,
    pub reference: String,
    /// A paraphrase of the reference; used by the loss-delta probe.
    #[serde(default)]
    pub paraphrase: Option<String>,
}

/// N-gram overlap between a benchmark reference and a training-set proxy.
/// Returns the fraction of distinct n-grams of the reference that appear
/// in the proxy. 0.0 means "disjoint", 1.0 means "every n-gram of the
/// reference is in the proxy."
pub fn ngram_overlap(reference: &str, proxy: &str, n: usize) -> f64 {
    assert!(n >= 1, "n must be >= 1");
    let ref_grams = ngrams(reference, n);
    if ref_grams.is_empty() {
        return 0.0;
    }
    let proxy_grams = ngrams(proxy, n);
    let hit = ref_grams
        .iter()
        .filter(|g| proxy_grams.contains(*g))
        .count();
    hit as f64 / ref_grams.len() as f64
}

fn ngrams(text: &str, n: usize) -> HashSet<Vec<String>> {
    let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
    if tokens.len() < n {
        return HashSet::new();
    }
    let mut out = HashSet::new();
    for w in tokens.windows(n) {
        out.insert(w.to_vec());
    }
    out
}

/// Echo-canary signal: given the model's response to a prompt that stops
/// *before* a known suffix of the reference, score how much of that
/// suffix the model echoed back verbatim. Returns the longest common
/// prefix length divided by the suffix length.
pub fn echo_canary_score(model_output: &str, held_out_suffix: &str) -> f64 {
    if held_out_suffix.is_empty() {
        return 0.0;
    }
    let a = model_output.as_bytes();
    let b = held_out_suffix.as_bytes();
    let mut i = 0usize;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i as f64 / b.len() as f64
}

/// Loss-delta signal: `loss(original) - loss(paraphrase)`. A substantially
/// *lower* loss on the original than on the paraphrase is a contamination
/// signal (the model has specifically memorized the original wording).
pub fn loss_delta(original_loss: f64, paraphrase_loss: f64) -> f64 {
    paraphrase_loss - original_loss
}

/// Per-example probe result. Downstream aggregation reports mean and 95%
/// CI across the benchmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeResult {
    pub id: String,
    pub ngram_overlap: f64,
    pub echo_canary: f64,
    pub loss_delta: f64,
}

/// A benchmark is **flagged** contaminated by a probe when its mean
/// signal is above the pre-registered threshold and the 95% CI lower
/// bound is above zero.
pub fn flagged(mean: f64, ci_lower: f64, threshold: f64) -> bool {
    mean > threshold && ci_lower > 0.0
}

/// Pre-registered thresholds for the three probes. Mirrors the table
/// in `docs/PREREGISTRATION.md` §3; override at the CLI level only
/// (and the override lands as a new report row, never a replacement).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Thresholds {
    pub ngram: f64,
    pub echo: f64,
    pub loss_delta: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            ngram: 0.05,
            echo: 0.3,
            loss_delta: 0.2,
        }
    }
}

/// Per-probe aggregate: mean, 95% percentile-bootstrap CI, pre-reg flag.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeAggregate {
    pub n: usize,
    pub mean: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub threshold: f64,
    pub flagged: bool,
}

/// One benchmark's full aggregate across all three probes and a model
/// verdict (any probe flag -> contaminated).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub ngram: ProbeAggregate,
    pub echo: ProbeAggregate,
    pub loss_delta: ProbeAggregate,
    /// A model is **contaminated on this benchmark** when any of the
    /// three probes flags it (pre-reg §3, "one flag is the finding").
    pub contaminated: bool,
}

/// Percentile bootstrap of the mean of a vector of per-example signals.
///
/// NaN values are treated as "n/a" and dropped before resampling (used
/// by the echo-canary probe for models that refuse to continue, per
/// pre-reg §6). Returns (n, mean, lo, hi) where n is the kept count.
pub fn bootstrap_mean_ci(values: &[f64], iters: usize, seed: u64) -> (usize, f64, f64, f64) {
    let kept: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = kept.len();
    if n == 0 {
        return (0, f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = kept.iter().sum::<f64>() / n as f64;
    if n == 1 || iters < 2 {
        return (n, mean, mean, mean);
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut sample = Vec::with_capacity(n);
    let mut means = Vec::with_capacity(iters);
    for _ in 0..iters {
        sample.clear();
        for _ in 0..n {
            let k = (rand::Rng::gen::<u32>(&mut rng) as usize) % n;
            sample.push(kept[k]);
        }
        means.push(sample.iter().sum::<f64>() / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[(iters as f64 * 0.025).floor() as usize];
    let hi = means[((iters as f64 * 0.975).ceil() as usize).min(iters - 1)];
    (n, mean, lo, hi)
}

/// Aggregate per-example probe results into a per-benchmark report with
/// the pre-registered flagging verdict applied.
pub fn aggregate(
    results: &[ProbeResult],
    iters: usize,
    seed: u64,
    t: Thresholds,
) -> BenchmarkReport {
    let probe =
        |values: &[f64], threshold: f64, probe_seed: u64| -> ProbeAggregate {
            let (n, mean, ci_lo, ci_hi) = bootstrap_mean_ci(values, iters, probe_seed);
            ProbeAggregate {
                n,
                mean,
                ci_lo,
                ci_hi,
                threshold,
                flagged: !mean.is_nan() && flagged(mean, ci_lo, threshold),
            }
        };
    let ngs: Vec<f64> = results.iter().map(|r| r.ngram_overlap).collect();
    let ecs: Vec<f64> = results.iter().map(|r| r.echo_canary).collect();
    let lds: Vec<f64> = results.iter().map(|r| r.loss_delta).collect();
    // Deterministic per-probe sub-seeds so swapping one probe's input
    // does not reshuffle the others.
    let ngram = probe(&ngs, t.ngram, seed);
    let echo = probe(&ecs, t.echo, seed.wrapping_add(1));
    let loss_delta = probe(&lds, t.loss_delta, seed.wrapping_add(2));
    let contaminated = ngram.flagged || echo.flagged || loss_delta.flagged;
    BenchmarkReport {
        ngram,
        echo,
        loss_delta,
        contaminated,
    }
}

/// One (model, benchmark) pair's bootstrap p-value for a single probe,
/// before any multiple-comparison correction. Pre-reg §4: "Across
/// models for the same benchmark, we use Holm-Bonferroni within a
/// probe family." The p is the one-sided bootstrap tail probability
/// that the probe's resampled mean sits at or below zero (i.e. that
/// the probe is NOT in the contaminated direction).
///
/// We approximate the pre-reg's "CI lower bound > 0" rule by its
/// bootstrap-p equivalent: the fraction of bootstrap means that is
/// ≤ 0. A probe whose mean is well above zero with a tight CI gets a
/// very small p; a probe whose CI brackets zero gets p near 0.5.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelProbePValue {
    pub model: String,
    pub p_value: f64,
}

/// Compute the one-sided bootstrap tail p-value for a probe's per-example
/// signals: `P(resampled_mean <= 0)` under the empirical distribution.
pub fn bootstrap_tail_p(values: &[f64], iters: usize, seed: u64) -> f64 {
    use rand::Rng;
    let kept: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = kept.len();
    if n == 0 || iters < 2 {
        return f64::NAN;
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut below_or_zero = 0usize;
    for _ in 0..iters {
        let mut acc = 0.0;
        for _ in 0..n {
            let k = (rng.gen::<u32>() as usize) % n;
            acc += kept[k];
        }
        let m = acc / n as f64;
        if m <= 0.0 {
            below_or_zero += 1;
        }
    }
    below_or_zero as f64 / iters as f64
}

/// Holm-Bonferroni adjusted decisions across a set of models on one
/// probe. Each input pair carries a raw p-value; the output carries the
/// adjusted p and the family-wise reject flag at `alpha`. Pre-reg §4.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HolmDecision {
    pub model: String,
    pub raw_p: f64,
    pub adjusted_p: f64,
    pub rejected: bool,
}

pub fn holm_bonferroni(input: &[ModelProbePValue], alpha: f64) -> Vec<HolmDecision> {
    let m = input.len();
    if m == 0 {
        return Vec::new();
    }
    // Sort by raw p ascending, remembering original indices.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|a, b| {
        input[*a]
            .p_value
            .partial_cmp(&input[*b].p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adjusted = vec![0.0; m];
    let mut rejected = vec![false; m];
    // Holm's step-down: adjusted p_(k) = max over i<=k of (m - i + 1) * p_(i),
    // clamped to [0, 1]; reject while adjusted p_(k) < alpha and no
    // earlier non-rejection has occurred.
    let mut running_max = 0.0f64;
    let mut rejected_so_far = true;
    for (rank, &sorted_i) in idx.iter().enumerate() {
        let scale = (m - rank) as f64;
        let adj = (scale * input[sorted_i].p_value).min(1.0).max(running_max);
        running_max = adj;
        adjusted[sorted_i] = adj;
        if rejected_so_far && adj < alpha {
            rejected[sorted_i] = true;
        } else {
            rejected_so_far = false;
            rejected[sorted_i] = false;
        }
    }
    input
        .iter()
        .enumerate()
        .map(|(i, p)| HolmDecision {
            model: p.model.clone(),
            raw_p: p.p_value,
            adjusted_p: adjusted[i],
            rejected: rejected[i],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_overlap_identical_strings_is_one() {
        let s = "the quick brown fox jumps over the lazy dog";
        assert!((ngram_overlap(s, s, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ngram_overlap_disjoint_strings_is_zero() {
        let a = "the quick brown fox";
        let b = "nothing in common here either";
        assert_eq!(ngram_overlap(a, b, 3), 0.0);
    }

    #[test]
    fn echo_canary_matches_longest_common_prefix() {
        assert!((echo_canary_score("Hello, world!", "Hello, world!") - 1.0).abs() < 1e-12);
        assert!((echo_canary_score("Hello", "Hello, world!") - 5.0 / 13.0).abs() < 1e-12);
        assert_eq!(echo_canary_score("Zelda", "Hello"), 0.0);
    }

    #[test]
    fn loss_delta_rewards_lower_original_loss() {
        // Model is more confident on the original -> positive loss_delta.
        let d = loss_delta(0.5, 2.0);
        assert!(d > 0.0);
    }

    #[test]
    fn flagged_requires_mean_above_threshold_and_ci_lower_bound_positive() {
        assert!(flagged(0.4, 0.1, 0.3));
        assert!(!flagged(0.4, -0.01, 0.3)); // CI crosses zero
        assert!(!flagged(0.2, 0.1, 0.3)); // mean below threshold
    }

    fn probe(id: &str, ng: f64, ec: f64, ld: f64) -> ProbeResult {
        ProbeResult {
            id: id.into(),
            ngram_overlap: ng,
            echo_canary: ec,
            loss_delta: ld,
        }
    }

    #[test]
    fn bootstrap_mean_ci_point_estimate_inside_interval() {
        let v: Vec<f64> = (1..=40).map(|i| i as f64 / 40.0).collect();
        let (n, mean, lo, hi) = bootstrap_mean_ci(&v, 2000, 1234);
        assert_eq!(n, 40);
        assert!(lo <= mean && mean <= hi, "lo={lo} mean={mean} hi={hi}");
        assert!(hi - lo < 0.3);
    }

    #[test]
    fn bootstrap_mean_ci_drops_nan() {
        let v = vec![1.0, f64::NAN, 1.0, 1.0, f64::NAN];
        let (n, mean, lo, hi) = bootstrap_mean_ci(&v, 500, 42);
        assert_eq!(n, 3);
        assert_eq!(mean, 1.0);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn bootstrap_mean_ci_all_nan_returns_nan() {
        let v = vec![f64::NAN, f64::NAN];
        let (n, mean, _, _) = bootstrap_mean_ci(&v, 100, 1);
        assert_eq!(n, 0);
        assert!(mean.is_nan());
    }

    #[test]
    fn bootstrap_mean_ci_deterministic_under_seed() {
        let v = vec![0.1, 0.9, 0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6];
        let a = bootstrap_mean_ci(&v, 500, 7);
        let b = bootstrap_mean_ci(&v, 500, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn aggregate_flags_contamination_when_any_probe_flags() {
        // Build a benchmark where only loss_delta is above threshold.
        let rs: Vec<ProbeResult> = (0..40)
            .map(|i| probe(&i.to_string(), 0.01, 0.05, 0.8))
            .collect();
        let r = aggregate(&rs, 500, 1234, Thresholds::default());
        assert!(!r.ngram.flagged);
        assert!(!r.echo.flagged);
        assert!(r.loss_delta.flagged, "loss_delta mean 0.8 > 0.2 and CI tight");
        assert!(r.contaminated);
    }

    #[test]
    fn aggregate_does_not_flag_when_ci_crosses_zero() {
        // Mean right at threshold but with a wide spread so the CI
        // lower bound is below zero -> no flag.
        let rs: Vec<ProbeResult> = (0..20)
            .map(|i| probe(&i.to_string(), if i < 10 { 1.0 } else { -0.9 }, 0.0, 0.0))
            .collect();
        let r = aggregate(&rs, 2000, 1234, Thresholds::default());
        assert!(!r.ngram.flagged, "wide bounds: {:?}", r.ngram);
        assert!(!r.contaminated);
    }

    #[test]
    fn aggregate_echo_canary_na_rows_are_dropped() {
        let rs: Vec<ProbeResult> = (0..30)
            .map(|i| {
                let ec = if i < 10 { f64::NAN } else { 0.9 };
                probe(&i.to_string(), 0.0, ec, 0.0)
            })
            .collect();
        let r = aggregate(&rs, 500, 1234, Thresholds::default());
        assert_eq!(r.echo.n, 20);
        assert!((r.echo.mean - 0.9).abs() < 1e-12);
        assert!(r.echo.flagged);
    }

    #[test]
    fn aggregate_is_deterministic_under_seed() {
        let rs: Vec<ProbeResult> = (0..25)
            .map(|i| {
                probe(
                    &i.to_string(),
                    (i as f64) / 25.0,
                    (i as f64) / 50.0,
                    (i as f64) / 100.0,
                )
            })
            .collect();
        let a = aggregate(&rs, 500, 42, Thresholds::default());
        let b = aggregate(&rs, 500, 42, Thresholds::default());
        assert_eq!(a.ngram.ci_lo, b.ngram.ci_lo);
        assert_eq!(a.echo.ci_hi, b.echo.ci_hi);
        assert_eq!(a.loss_delta.mean, b.loss_delta.mean);
    }

    #[test]
    fn bootstrap_tail_p_small_for_strong_signal() {
        let v = vec![0.5f64; 50];
        let p = bootstrap_tail_p(&v, 500, 1);
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn bootstrap_tail_p_near_half_when_signals_center_on_zero() {
        let v: Vec<f64> = (0..100).map(|i| (i as f64 - 49.5) / 50.0).collect();
        let p = bootstrap_tail_p(&v, 500, 1);
        assert!(0.3 < p && p < 0.7, "p={p}");
    }

    #[test]
    fn holm_rejects_strongly_contaminated_models() {
        let input = vec![
            ModelProbePValue { model: "A".into(), p_value: 0.001 },
            ModelProbePValue { model: "B".into(), p_value: 0.01 },
            ModelProbePValue { model: "C".into(), p_value: 0.6 },
        ];
        let out = holm_bonferroni(&input, 0.05);
        // A: 0.001 * 3 = 0.003 < 0.05 -> reject
        // B: 0.01 * 2 = 0.02 < 0.05 -> reject
        // C: 0.6 * 1 = 0.6 > 0.05 -> keep
        assert!(out.iter().find(|d| d.model == "A").unwrap().rejected);
        assert!(out.iter().find(|d| d.model == "B").unwrap().rejected);
        assert!(!out.iter().find(|d| d.model == "C").unwrap().rejected);
    }

    #[test]
    fn holm_stops_rejecting_after_first_non_reject() {
        // Rank 1 fails to reject -> every later rank must also not reject,
        // even if its adjusted p is < alpha.
        let input = vec![
            ModelProbePValue { model: "A".into(), p_value: 0.04 }, // *3 = 0.12 > 0.05
            ModelProbePValue { model: "B".into(), p_value: 0.045 }, // *2 = 0.09 > 0.05
            ModelProbePValue { model: "C".into(), p_value: 0.046 }, // *1 = 0.046 < 0.05
        ];
        let out = holm_bonferroni(&input, 0.05);
        assert!(!out.iter().find(|d| d.model == "A").unwrap().rejected);
        assert!(!out.iter().find(|d| d.model == "B").unwrap().rejected);
        assert!(!out.iter().find(|d| d.model == "C").unwrap().rejected,
            "Holm's step-down halts after first non-reject");
    }

    #[test]
    fn holm_empty_input_returns_empty() {
        assert_eq!(holm_bonferroni(&[], 0.05).len(), 0);
    }
}
