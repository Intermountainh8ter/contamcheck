//! Leak-share curve of the n-gram flagging rule: sweeps the share of each
//! synthetic benchmark leaked into the proxy and reports recall per share.

use crate::evasion::{run, Cell, Params};
use std::collections::BTreeMap;

pub fn default_shares() -> Vec<f64> {
    vec![0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
}

pub fn share_key(s: f64) -> String {
    format!("{s:.2}")
}

/// Returns class -> n -> share key -> Cell, plus the null flag rate per n
/// (averaged identically across shares; the null proxy does not depend on the share).
pub fn sweep(
    base: &Params,
    shares: &[f64],
    classes: &[(String, f64)],
) -> BTreeMap<String, BTreeMap<usize, BTreeMap<String, Cell>>> {
    let mut out: BTreeMap<String, BTreeMap<usize, BTreeMap<String, Cell>>> = BTreeMap::new();
    for &s in shares {
        let p = Params {
            seed: base.seed,
            benchmarks: base.benchmarks,
            examples: base.examples,
            tokens: base.tokens,
            proxy_tokens: base.proxy_tokens,
            vocab: base.vocab,
            boot_iters: base.boot_iters,
            leak_share: s,
            ns: base.ns.clone(),
            classes: classes.to_vec(),
        };
        for (class, cells) in run(&p) {
            for (n, c) in cells {
                out.entry(class.clone()).or_default().entry(n).or_default().insert(share_key(s), c);
            }
        }
    }
    out
}
