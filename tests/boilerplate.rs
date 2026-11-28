use contamcheck::boilerplate::sweep;
use contamcheck::evasion::Params;

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.0, ns: vec![13], classes: vec![] }
}

#[test]
fn no_boilerplate_never_flags_and_full_length_always_flags() {
    let r = sweep(&params(), &[0, 30]);
    assert_eq!(r[&13][&0].recall, 0.0);
    assert_eq!(r[&13][&30].recall, 1.0);
}

#[test]
fn short_prefix_below_n_does_not_flag_and_is_deterministic() {
    let a = sweep(&params(), &[12]);
    let b = sweep(&params(), &[12]);
    assert_eq!(a[&13][&12].recall, 0.0);
    assert_eq!(a[&13][&12].flagged, b[&13][&12].flagged);
}
