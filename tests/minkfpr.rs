use contamcheck::minkfpr::{mink, sweep, threshold, Params};

fn params() -> Params {
    Params { seed: 7, trials: 5, examples: 100, tokens: 32, k: 0.2, target_fpr: 0.05 }
}

#[test]
fn mink_takes_lowest_share() {
    let lp = [-5.0, -4.0, -1.0, -1.0, -1.0];
    assert_eq!(mink(&lp, 0.4), -4.5);
}

#[test]
fn threshold_allows_target_share() {
    let s: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let t = threshold(&s, 0.05);
    assert_eq!(s.iter().filter(|&&x| x > t).count(), 5);
}

#[test]
fn deterministic_and_always_flag_is_one() {
    let a = sweep(&params(), &[0.0, 0.5]);
    let b = sweep(&params(), &[0.0, 0.5]);
    assert_eq!(a["mink"]["0.5"].flagged, b["mink"]["0.5"].flagged);
    assert_eq!(a["always_flag"]["0"].rate, 1.0);
}

#[test]
fn drift_raises_false_positives() {
    let r = sweep(&params(), &[0.0, 1.0]);
    assert!(r["mink"]["1"].rate > r["mink"]["0"].rate);
    assert!(r["zlib"]["1"].rate > r["zlib"]["0"].rate);
}
