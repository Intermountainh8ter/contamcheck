use contamcheck::evasion::Params;
use contamcheck::leakcurve::{share_key, sweep};

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.5, ns: vec![13], classes: vec![] }
}

#[test]
fn share_key_is_two_decimals() {
    assert_eq!(share_key(0.1), "0.10");
    assert_eq!(share_key(0.02), "0.02");
}

#[test]
fn full_verbatim_leak_flagged_and_deterministic() {
    let cl = vec![("verbatim".to_string(), 0.0)];
    let a = sweep(&params(), &[0.5, 1.0], &cl);
    let b = sweep(&params(), &[0.5, 1.0], &cl);
    assert_eq!(a["verbatim"][&13]["1.00"].recall, 1.0);
    assert_eq!(a["verbatim"][&13]["0.50"].flagged, b["verbatim"][&13]["0.50"].flagged);
    assert_eq!(a["null"][&13]["0.50"].recall, 0.0);
}
