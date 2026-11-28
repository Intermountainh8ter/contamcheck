//! Evasion curve of the n-gram flagging rule: a share of each synthetic
//! benchmark is leaked into the proxy verbatim or with each token
//! independently rewritten at a set rate. Recall is the share of benchmarks
//! the pre-registered rule flags. A never-flag responder has recall 0.

use crate::nullcal::{overlap_ids, wilson, Zipf};
use crate::{aggregate, ProbeResult, Thresholds};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::collections::{BTreeMap, HashSet};

pub struct Params {
    pub seed: u64,
    pub benchmarks: usize,
    pub examples: usize,
    pub tokens: usize,
    pub proxy_tokens: usize,
    pub vocab: usize,
    pub boot_iters: usize,
    pub leak_share: f64,
    pub ns: Vec<usize>,
    pub classes: Vec<(String, f64)>,
}

pub fn default_classes() -> Vec<(String, f64)> {
    vec![
        ("verbatim".into(), 0.0),
        ("light".into(), 0.1),
        ("moderate".into(), 0.25),
        ("paraphrase".into(), 0.5),
    ]
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub flagged: usize,
    pub benchmarks: usize,
    pub recall: f64,
    pub wilson_lo: f64,
    pub wilson_hi: f64,
    pub mean_overlap_leaked: f64,
}

/// Rewrite each token independently with probability `rate` by a fresh Zipf draw.
pub fn rewrite(zipf: &Zipf, rng: &mut Pcg64Mcg, item: &[u32], rate: f64) -> Vec<u32> {
    item.iter()
        .map(|&t| if rng.gen::<f64>() < rate { zipf.sample(rng) } else { t })
        .collect()
}

pub fn flagged_for(bench: &[Vec<u32>], proxy: &[u32], n: usize, p: &Params, b: usize) -> (bool, f64) {
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
    let leaked = ((p.leak_share * p.examples as f64).round() as usize).min(rows.len());
    let ml = if leaked == 0 { 0.0 } else { rows[..leaked].iter().map(|r| r.ngram_overlap).sum::<f64>() / leaked as f64 };
    let rep = aggregate(&rows, p.boot_iters, p.seed.wrapping_add(b as u64), Thresholds::default());
    (rep.ngram.flagged, ml)
}

/// Returns class -> n -> Cell. The first `round(leak_share * examples)` items
/// of each benchmark are the leaked ones; the null (no leak) is class "null".
pub fn run(p: &Params) -> BTreeMap<String, BTreeMap<usize, Cell>> {
    let zipf = Zipf::new(p.vocab);
    let mut names: Vec<(String, Option<f64>)> = vec![("null".into(), None)];
    names.extend(p.classes.iter().map(|(k, r)| (k.clone(), Some(*r))));
    let mut acc: BTreeMap<(String, usize), (usize, f64)> = BTreeMap::new();
    let leaked_n = (p.leak_share * p.examples as f64).round() as usize;
    for b in 0..p.benchmarks {
        let mut rng = Pcg64Mcg::seed_from_u64(p.seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
        let base_proxy = zipf.text(&mut rng, p.proxy_tokens);
        let bench: Vec<Vec<u32>> = (0..p.examples).map(|_| zipf.text(&mut rng, p.tokens)).collect();
        for (ci, (name, rate)) in names.iter().enumerate() {
            let mut proxy = base_proxy.clone();
            if let Some(r) = rate {
                let mut lr = Pcg64Mcg::seed_from_u64(
                    p.seed.wrapping_mul(7_919).wrapping_add((b * 131 + ci) as u64),
                );
                for item in bench.iter().take(leaked_n.min(bench.len())) {
                    proxy.extend(rewrite(&zipf, &mut lr, item, *r));
                    // separator-free append; boundary windows mix unrelated tokens
                }
            }
            for &n in &p.ns {
                let (f, ml) = flagged_for(&bench, &proxy, n, p, b);
                let e = acc.entry((name.clone(), n)).or_insert((0, 0.0));
                e.0 += f as usize;
                e.1 += ml;
            }
        }
    }
    let mut out: BTreeMap<String, BTreeMap<usize, Cell>> = BTreeMap::new();
    for ((name, n), (k, ml)) in acc {
        let (lo, hi) = wilson(k, p.benchmarks);
        out.entry(name).or_default().insert(n, Cell {
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
