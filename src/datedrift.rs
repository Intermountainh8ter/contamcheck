//! Clean-set false-positive rate of the n-gram flagging rule when the
//! known-out proxy comes from the same Zipf distribution with a swept share
//! of vocabulary ranks permuted among themselves (a stand-in for date shift).
//! No item is copied, so every flag is a false positive.

use crate::evasion::{flagged_for, Cell, Params};
use crate::nullcal::{wilson, Zipf};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeMap;

pub fn default_shifts() -> Vec<f64> {
    vec![0.0, 0.1, 0.25, 0.5, 1.0]
}

/// Map each token id to itself except for a seeded choice of
/// `round(shift * vocab)` ids, which are shuffled among themselves.
pub fn permutation(vocab: usize, shift: f64, rng: &mut Pcg64Mcg) -> Vec<u32> {
    let k = ((shift * vocab as f64).round() as usize).min(vocab);
    let mut ids: Vec<u32> = (0..vocab as u32).collect();
    ids.shuffle(rng);
    let chosen: Vec<u32> = ids[..k].to_vec();
    let mut shuffled = chosen.clone();
    shuffled.shuffle(rng);
    let mut map: Vec<u32> = (0..vocab as u32).collect();
    for (a, b) in chosen.iter().zip(shuffled.iter()) {
        map[*a as usize] = *b;
    }
    map
}

/// Returns n -> shift (as printed with `{}`) -> Cell (`recall` is the
/// false-flag rate). Benchmark and base proxy are drawn exactly as in
/// `nullcal`, so shift 0 reproduces the null calibration.
pub fn sweep(p: &Params, shifts: &[f64]) -> BTreeMap<usize, BTreeMap<String, Cell>> {
    let zipf = Zipf::new(p.vocab);
    let mut acc: BTreeMap<(usize, String), usize> = BTreeMap::new();
    let q = Params { leak_share: 0.0, ..p.clone() };
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let base_proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        for &shift in shifts {
            let mut pr = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(7_919).wrapping_add(b as u64));
            let map = permutation(p.vocab, shift, &mut pr);
            let proxy: Vec<u32> = base_proxy.iter().map(|&t| map[t as usize]).collect();
            for &n in &p.ns {
                let (f, _) = flagged_for(&bench, &proxy, n, &q, b);
                *acc.entry((n, format!("{shift}"))).or_insert(0) += f as usize;
            }
        }
    }
    let mut out: BTreeMap<usize, BTreeMap<String, Cell>> = BTreeMap::new();
    for ((n, shift), k) in acc {
        let (lo, hi) = wilson(k, p.benchmarks);
        out.entry(n).or_default().insert(shift, Cell {
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
