use contamcheck::evasion::Params;
use contamcheck::paracurve::{paraphrased_count, share_key, sweep};

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.5, ns: vec![13], classes: vec![] }
}

#[test]
fn paraphrased_count_rounds_and_caps() {
    assert_eq!(paraphrased_count(25, 0.0), 0);
    assert_eq!(paraphrased_count(25, 0.9), 23);
    assert_eq!(paraphrased_count(25, 1.0), 25);
    assert_eq!(paraphrased_count(10, 2.0), 10);
}

#[test]
fn all_verbatim_flagged_all_paraphrased_not_and_deterministic() {
    let a = sweep(&params(), &[0.0, 1.0], 0.5);
    let b = sweep(&params(), &[0.0, 1.0], 0.5);
    assert_eq!(a[&13][&share_key(0.0)].recall, 1.0);
    assert_eq!(a[&13][&share_key(1.0)].recall, 0.0);
    assert_eq!(a[&13][&share_key(1.0)].flagged, b[&13][&share_key(1.0)].flagged);
}
