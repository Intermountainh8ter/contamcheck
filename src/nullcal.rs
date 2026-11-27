//! Null calibration of the n-gram flagging rule: benchmark and proxy are
//! drawn independently from the same seeded Zipf token distribution, so any
//! flag is a false positive by construction.

use crate::{aggregate, ngram_overlap, ProbeResult, Thresholds};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

pub struct Zipf {
    cum: Vec<f64>,
}

impl Zipf {
    pub fn new(vocab: usize) -> Self {
        let mut cum = Vec::with_capacity(vocab);
        let mut s = 0.0;
        for k in 0..vocab {
            s += 1.0 / (k as f64 + 1.0);
            cum.push(s);
        }
        Zipf { cum }
    }
    pub fn sample(&self, rng: &mut Pcg64Mcg) -> u32 {
        let total = *self.cum.last().unwrap();
        let u = rng.gen::<f64>() * total;
        self.cum.partition_point(|&c| c < u).min(self.cum.len() - 1) as u32
    }
    pub fn text(&self, rng: &mut Pcg64Mcg, tokens: usize) -> Vec<u32> {
        (0..tokens).map(|_| self.sample(rng)).collect()
    }
}

pub fn render(tokens: &[u32]) -> String {
    tokens.iter().map(|t| format!("w{t}")).collect::<Vec<_>>().join(" ")
}

/// Same quantity as `ngram_overlap`, on token ids, with the proxy n-gram set
/// built once by the caller.
pub fn overlap_ids(reference: &[u32], proxy: &HashSet<Vec<u32>>, n: usize) -> f64 {
    if reference.len() < n {
        return 0.0;
    }
    let grams: HashSet<&[u32]> = reference.windows(n).collect();
    let hit = grams.iter().filter(|g| proxy.contains(**g)).count();
    hit as f64 / grams.len() as f64
}

#[derive(Serialize, Clone, Debug)]
pub struct NCell {
    pub flagged: usize,
    pub benchmarks: usize,
    pub rate: f64,
    pub wilson_lo: f64,
    pub wilson_hi: f64,
    pub mean_overlap: f64,
}

pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    let z = 1.96f64;
    let nf = n as f64;
    let p = k as f64 / nf;
    let d = 1.0 + z * z / nf;
    let c = (p + z * z / (2.0 * nf)) / d;
    let h = z * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt() / d;
    ((c - h).max(0.0), (c + h).min(1.0))
}

pub struct Params {
    pub seed: u64,
    pub benchmarks: usize,
    pub examples: usize,
    pub tokens: usize,
    pub proxy_tokens: usize,
    pub vocab: usize,
    pub boot_iters: usize,
    pub ns: Vec<usize>,
}

pub fn run(p: &Params) -> BTreeMap<usize, NCell> {
    let zipf = Zipf::new(p.vocab);
    let mut flagged: BTreeMap<usize, usize> = p.ns.iter().map(|&n| (n, 0)).collect();
    let mut overlap_sum: BTreeMap<usize, f64> = p.ns.iter().map(|&n| (n, 0.0)).collect();
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        for &n in &p.ns {
            let set: HashSet<Vec<u32>> = if proxy.len() >= n {
                proxy.windows(n).map(|w| w.to_vec()).collect()
            } else {
                HashSet::new()
            };
            let rows: Vec<ProbeResult> = bench
                .iter()
                .enumerate()
                .map(|(i, e)| ProbeResult {
                    id: format!("b{b}e{i}"),
                    ngram_overlap: overlap_ids(e, &set, n),
                    echo_canary: 0.0,
                    loss_delta: 0.0,
                })
                .collect();
            let m = rows.iter().map(|r| r.ngram_overlap).sum::<f64>() / rows.len() as f64;
            *overlap_sum.get_mut(&n).unwrap() += m;
            let rep = aggregate(&rows, p.boot_iters, p.seed.wrapping_add(b as u64), Thresholds::default());
            if rep.ngram.flagged {
                *flagged.get_mut(&n).unwrap() += 1;
            }
        }
    }
    p.ns
        .iter()
        .map(|&n| {
            let k = flagged[&n];
            let (lo, hi) = wilson(k, p.benchmarks);
            (n, NCell {
                flagged: k,
                benchmarks: p.benchmarks,
                rate: k as f64 / p.benchmarks as f64,
                wilson_lo: lo,
                wilson_hi: hi,
                mean_overlap: overlap_sum[&n] / p.benchmarks as f64,
            })
        })
        .collect()
}

/// Keep `ngram_overlap` imported for the equivalence test.
pub fn string_overlap(reference: &[u32], proxy: &[u32], n: usize) -> f64 {
    ngram_overlap(&render(reference), &render(proxy), n)
}
