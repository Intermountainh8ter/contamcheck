use contamcheck::datedrift::{permutation, sweep};
use contamcheck::evasion::Params;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.0, ns: vec![3, 13], classes: vec![] }
}

#[test]
fn permutation_is_a_bijection_and_zero_shift_is_identity() {
    let mut r = Pcg64Mcg::seed_from_u64(1);
    let id = permutation(100, 0.0, &mut r);
    assert!(id.iter().enumerate().all(|(i, &v)| v as usize == i));
    let m = permutation(100, 0.5, &mut r);
    let mut s = m.clone();
    s.sort();
    assert_eq!(s, (0..100u32).collect::<Vec<_>>());
}

#[test]
fn deterministic_and_keyed_by_shift() {
    let a = sweep(&params(), &[0.0, 0.5]);
    let b = sweep(&params(), &[0.0, 0.5]);
    assert_eq!(a[&3]["0.5"].flagged, b[&3]["0.5"].flagged);
    assert!(a[&13].contains_key("0"));
    assert_eq!(a[&13]["0.5"].recall, 0.0);
}
