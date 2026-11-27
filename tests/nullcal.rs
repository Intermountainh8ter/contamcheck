use contamcheck::nullcal::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::collections::HashSet;

#[test]
fn id_overlap_matches_string_overlap() {
    let z = Zipf::new(50);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let a = z.text(&mut rng, 30);
    let b = z.text(&mut rng, 400);
    for n in [1usize, 2, 3, 5] {
        let set: HashSet<Vec<u32>> = b.windows(n).map(|w| w.to_vec()).collect();
        assert!((overlap_ids(&a, &set, n) - string_overlap(&a, &b, n)).abs() < 1e-12);
    }
}

#[test]
fn run_is_deterministic_and_n_monotone() {
    let p = Params { seed: 3, benchmarks: 6, examples: 10, tokens: 30, proxy_tokens: 2000, vocab: 200, boot_iters: 50, ns: vec![1, 13] };
    let a = run(&p);
    let b = run(&p);
    assert_eq!(a[&1].flagged, b[&1].flagged);
    assert!(a[&1].mean_overlap > a[&13].mean_overlap);
    assert_eq!(a[&13].flagged, 0);
}

#[test]
fn wilson_bounds() {
    let (lo, hi) = wilson(0, 100);
    assert_eq!(lo, 0.0);
    assert!(hi > 0.0 && hi < 0.05);
}
