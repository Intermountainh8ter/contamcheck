//! Clean-set false-positive rate of the n-gram flagging rule when a shared
//! boilerplate prefix (also present once in the proxy) is carried by only a
//! swept share of the benchmark items. No item is copied, so every flag is a
//! false positive.

use crate::evasion::{flagged_for, Cell, Params};
use crate::nullcal::{wilson, Zipf};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

pub fn default_shares() -> Vec<f64> {
    vec![0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
}

/// Returns n -> share (as printed with `{}`) -> Cell (`recall` is the
/// false-flag rate). The first `round(share * examples)` items of each
/// benchmark get the first `len` tokens (capped at the item length) replaced
/// by the shared boilerplate.
pub fn sweep(p: &Params, len: usize, shares: &[f64]) -> BTreeMap<usize, BTreeMap<String, Cell>> {
    let zipf = Zipf::new(p.vocab);
    let mut acc: BTreeMap<(usize, String), usize> = BTreeMap::new();
    let q = Params { leak_share: 0.0, ..p.clone() };
    let l = len.min(p.tokens);
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let base_proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        let mut br = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(7_919).wrapping_add(b as u64));
        let boiler = zipf.text(&mut br, l);
        let mut proxy = base_proxy.clone();
        proxy.extend_from_slice(&boiler);
        for &share in shares {
            let k = ((share * p.examples as f64).round() as usize).min(p.examples);
            let items: Vec<Vec<u32>> = bench
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    if i < k {
                        let mut v = boiler.clone();
                        v.extend_from_slice(&e[l..]);
                        v
                    } else {
                        e.clone()
                    }
                })
                .collect();
            for &n in &p.ns {
                let (f, _) = flagged_for(&items, &proxy, n, &q, b);
                *acc.entry((n, format!("{share}"))).or_insert(0) += f as usize;
            }
        }
    }
    let mut out: BTreeMap<usize, BTreeMap<String, Cell>> = BTreeMap::new();
    for ((n, share), k) in acc {
        let (lo, hi) = wilson(k, p.benchmarks);
        out.entry(n).or_default().insert(share, Cell {
            flagged: k,
            benchmarks: p.benchmarks,
            recall: k as f64 / p.benchmarks as f64,
            wilson_lo: lo,
            wilson_hi: hi,
            mean_overlap_leaked: 0.0,
        });
    }
    out
}
