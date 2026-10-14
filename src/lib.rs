//! Core primitives for contamination auditing.
//!
//! This crate ships three probe primitives: n-gram overlap against a
//! training-set proxy, echo-canary completion scoring, and loss-delta
//! comparison between the original benchmark example and a semantic
//! paraphrase. Each probe reports a per-example score and a per-benchmark
//! aggregate with a 95% bootstrap CI (bootstrap arrives in the follow-up
//! release; see `docs/PREREGISTRATION.md`).

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
}
