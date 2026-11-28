use contamcheck::boilerplate_share::sweep;
use contamcheck::evasion::Params;

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.0, ns: vec![13], classes: vec![] }
}

#[test]
fn zero_share_never_flags_and_full_share_always_flags() {
    let r = sweep(&params(), 20, &[0.0, 1.0]);
    assert_eq!(r[&13]["0"].recall, 0.0);
    assert_eq!(r[&13]["1"].recall, 1.0);
}

#[test]
fn deterministic_and_keyed_by_share() {
    let a = sweep(&params(), 20, &[0.05, 0.5]);
    let b = sweep(&params(), 20, &[0.05, 0.5]);
    assert_eq!(a[&13]["0.5"].flagged, b[&13]["0.5"].flagged);
    assert!(a[&13].contains_key("0.05"));
}
