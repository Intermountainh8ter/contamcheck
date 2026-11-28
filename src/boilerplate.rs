//! Clean-set false-positive rate of the n-gram flagging rule when every
//! benchmark item starts with a shared boilerplate prefix that also appears
//! once in the proxy. No item is copied, so every flag is a false positive.

use crate::evasion::{flagged_for, Cell, Params};
use crate::nullcal::{wilson, Zipf};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

pub fn default_lens() -> Vec<usize> {
    vec![0, 6, 12, 13, 20]
}

/// Returns n -> boilerplate length -> Cell (`recall` is the false-flag rate).
/// `p.leak_share` is ignored; the prefix replaces the first `len` tokens of
/// each item (capped at the item length).
pub fn sweep(p: &Params, lens: &[usize]) -> BTreeMap<usize, BTreeMap<usize, Cell>> {
    let zipf = Zipf::new(p.vocab);
    let mut acc: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let q = Params { leak_share: 0.0, ..p.clone() };
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let base_proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        let mut br = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(7_919).wrapping_add(b as u64));
        let maxlen = lens.iter().copied().max().unwrap_or(0);
        let boiler = zipf.text(&mut br, maxlen);
        for &len in lens {
            let l = len.min(p.tokens);
            let items: Vec<Vec<u32>> = bench
                .iter()
                .map(|e| {
                    let mut v = boiler[..l].to_vec();
                    v.extend_from_slice(&e[l..]);
                    v
                })
                .collect();
            let mut proxy = base_proxy.clone();
            proxy.extend_from_slice(&boiler[..l]);
            for &n in &p.ns {
                let (f, _) = flagged_for(&items, &proxy, n, &q, b);
                *acc.entry((n, len)).or_insert(0) += f as usize;
            }
        }
    }
    let mut out: BTreeMap<usize, BTreeMap<usize, Cell>> = BTreeMap::new();
    for ((n, len), k) in acc {
        let (lo, hi) = wilson(k, p.benchmarks);
        out.entry(n).or_default().insert(len, Cell {
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
