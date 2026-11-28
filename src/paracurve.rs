//! Paraphrase-share curve of the n-gram flagging rule: a fixed share of each
//! synthetic benchmark is leaked into the proxy, and a swept share of the
//! leaked items is rewritten token-wise at a set rate (the rest verbatim).

use crate::evasion::{flagged_for, rewrite, Cell, Params};
use crate::nullcal::{wilson, Zipf};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

pub fn default_shares() -> Vec<f64> {
    vec![0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
}

pub fn share_key(s: f64) -> String {
    format!("{s}")
}

/// Number of leaked items rewritten at paraphrase share `s`.
pub fn paraphrased_count(leaked: usize, s: f64) -> usize {
    ((s * leaked as f64).round() as usize).min(leaked)
}

/// Returns n -> share key -> Cell; `p.leak_share` is the fixed leaked share,
/// `rate` the token rewrite rate of the paraphrased items. The first
/// `paraphrased_count` leaked items are rewritten, the rest are verbatim.
pub fn sweep(p: &Params, shares: &[f64], rate: f64) -> BTreeMap<usize, BTreeMap<String, Cell>> {
    let zipf = Zipf::new(p.vocab);
    let leaked_n = ((p.leak_share * p.examples as f64).round() as usize).min(p.examples);
    let mut acc: BTreeMap<(usize, String), (usize, f64)> = BTreeMap::new();
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let base_proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        for (si, &s) in shares.iter().enumerate() {
            let k = paraphrased_count(leaked_n, s);
            let mut lr = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(7_919).wrapping_add((b * 131 + si) as u64));
            let mut proxy = base_proxy.clone();
            for (i, item) in bench.iter().take(leaked_n).enumerate() {
                if i < k {
                    proxy.extend(rewrite(&zipf, &mut lr, item, rate));
                } else {
                    proxy.extend_from_slice(item);
                }
            }
            for &n in &p.ns {
                let (f, ml) = flagged_for(&bench, &proxy, n, p, b);
                let e = acc.entry((n, share_key(s))).or_insert((0, 0.0));
                e.0 += f as usize;
                e.1 += ml;
            }
        }
    }
    let mut out: BTreeMap<usize, BTreeMap<String, Cell>> = BTreeMap::new();
    for ((n, key), (k, ml)) in acc {
        let (lo, hi) = wilson(k, p.benchmarks);
        out.entry(n).or_default().insert(key, Cell {
            flagged: k,
            benchmarks: p.benchmarks,
            recall: k as f64 / p.benchmarks as f64,
            wilson_lo: lo,
            wilson_hi: hi,
            mean_overlap_leaked: ml / p.benchmarks as f64,
        });
    }
    out
}
