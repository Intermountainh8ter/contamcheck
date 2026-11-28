use contamcheck::evasion::{default_classes, rewrite, run, Params};
use contamcheck::nullcal::Zipf;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn params() -> Params {
    Params { seed: 7, benchmarks: 6, examples: 20, tokens: 30, proxy_tokens: 5000, vocab: 2000, boot_iters: 50, leak_share: 0.5, ns: vec![13], classes: default_classes() }
}

#[test]
fn rewrite_rate_zero_is_identity_and_one_changes_most() {
    let z = Zipf::new(5000);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let item = z.text(&mut rng, 200);
    assert_eq!(rewrite(&z, &mut rng, &item, 0.0), item);
    let r = rewrite(&z, &mut rng, &item, 1.0);
    let same = item.iter().zip(&r).filter(|(a, b)| a == b).count();
    assert!(same < 100);
}

#[test]
fn verbatim_flagged_paraphrase_not_and_deterministic() {
    let a = run(&params());
    let b = run(&params());
    assert_eq!(a["verbatim"][&13].recall, 1.0);
    assert_eq!(a["paraphrase"][&13].recall, 0.0);
    assert_eq!(a["null"][&13].recall, 0.0);
    assert_eq!(a["verbatim"][&13].flagged, b["verbatim"][&13].flagged);
}
