//! Clean-split false-positive rate of a Min-K% and a zlib-ratio score whose
//! threshold is fixed on a calibration split at a target false positive rate,
//! when the clean split's per-token log-probabilities are drifted.
//! Simulated log-probabilities; no model. Every flag on a clean example is a
//! false positive.

use crate::nullcal::wilson;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Params {
    pub seed: u64,
    pub trials: usize,
    pub examples: usize,
    pub tokens: usize,
    pub k: f64,
    pub target_fpr: f64,
}

/// Per-example random effect sd and per-token noise sd of the simulated
/// log-probabilities; the per-token marginal sd is `token_sd()`.
pub const EXAMPLE_SD: f64 = 0.5;
pub const TOKEN_SD: f64 = 1.0;
pub const MEAN_LOGPROB: f64 = -3.0;

pub fn token_sd() -> f64 {
    (EXAMPLE_SD * EXAMPLE_SD + TOKEN_SD * TOKEN_SD).sqrt()
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub flagged: usize,
    pub n: usize,
    pub rate: f64,
    pub wilson_lo: f64,
    pub wilson_hi: f64,
}

fn normal(rng: &mut Pcg64Mcg) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Mean of the lowest `ceil(k * len)` values (Min-K%: higher means more member-like).
pub fn mink(lp: &[f64], k: f64) -> f64 {
    let mut v = lp.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = ((k * v.len() as f64).ceil() as usize).clamp(1, v.len());
    v[..m].iter().sum::<f64>() / m as f64
}

/// Mean log-probability divided by a complexity factor (stand-in for zlib length).
pub fn zlib_ratio(lp: &[f64], complexity: f64) -> f64 {
    lp.iter().sum::<f64>() / lp.len() as f64 / complexity
}

/// Smallest score threshold such that at most `target` of `scores` exceed it.
pub fn threshold(scores: &[f64], target: f64) -> f64 {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let allowed = (target * v.len() as f64).floor() as usize;
    v[v.len() - 1 - allowed]
}

fn example(rng: &mut Pcg64Mcg, tokens: usize, shift: f64) -> (Vec<f64>, f64) {
    let e = EXAMPLE_SD * normal(rng);
    let lp: Vec<f64> = (0..tokens).map(|_| MEAN_LOGPROB + shift + e + TOKEN_SD * normal(rng)).collect();
    let c = 0.5 + rng.gen::<f64>();
    (lp, c)
}

/// score -> drift (printed with `{}`) -> Cell, pooled over trials.
pub fn sweep(p: &Params, drifts: &[f64]) -> BTreeMap<String, BTreeMap<String, Cell>> {
    let mut acc: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in 0..p.trials {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(t as u64));
        let cal: Vec<(Vec<f64>, f64)> = (0..p.examples).map(|_| example(&mut rng, p.tokens, 0.0)).collect();
        let cm: Vec<f64> = cal.iter().map(|(l, _)| mink(l, p.k)).collect();
        let cz: Vec<f64> = cal.iter().map(|(l, c)| zlib_ratio(l, *c)).collect();
        let (tm, tz) = (threshold(&cm, p.target_fpr), threshold(&cz, p.target_fpr));
        for &d in drifts {
            let mut r = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(7_919).wrapping_add(t as u64));
            let shift = d * token_sd();
            let (mut fm, mut fz) = (0usize, 0usize);
            for _ in 0..p.examples {
                let (l, c) = example(&mut r, p.tokens, shift);
                fm += (mink(&l, p.k) > tm) as usize;
                fz += (zlib_ratio(&l, c) > tz) as usize;
            }
            *acc.entry(("mink".into(), format!("{d}"))).or_insert(0) += fm;
            *acc.entry(("zlib".into(), format!("{d}"))).or_insert(0) += fz;
            *acc.entry(("always_flag".into(), format!("{d}"))).or_insert(0) += p.examples;
        }
    }
    let n = p.trials * p.examples;
    let mut out: BTreeMap<String, BTreeMap<String, Cell>> = BTreeMap::new();
    for ((s, d), k) in acc {
        let (lo, hi) = wilson(k, n);
        out.entry(s).or_default().insert(d, Cell { flagged: k, n, rate: k as f64 / n as f64, wilson_lo: lo, wilson_hi: hi });
    }
    out
}
