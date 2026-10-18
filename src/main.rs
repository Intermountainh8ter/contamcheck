//! contamcheck CLI — scaffold.
//!
//! `probe` runs the three per-example probes on a single Example.
//! `aggregate` reads a JSON Lines file of `ProbeResult` rows (as `probe`
//! prints each one) and emits a `BenchmarkReport` with pre-registered
//! bootstrap-CI flagging. The `audit` subcommand that drives a model
//! endpoint across a benchmark end-to-end is TODO.

use clap::{Parser, Subcommand};
use contamcheck::{
    aggregate, echo_canary_score, holm_bonferroni, loss_delta, ngram_overlap, Example,
    ModelProbePValue, ProbeResult, Thresholds,
};

#[derive(Parser)]
#[command(
    name = "contamcheck",
    version,
    about = "Benchmark contamination auditor"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the three per-example probes against supplied inputs.
    Probe {
        /// Path to a JSON `Example` object.
        example: std::path::PathBuf,
        /// Path to a text file: the training-set proxy for n-gram overlap.
        proxy: std::path::PathBuf,
        /// Path to a text file: the model's completion of the held-out suffix.
        model_output: std::path::PathBuf,
        /// Path to a text file: the held-out suffix of the reference.
        held_out_suffix: std::path::PathBuf,
        /// Model loss on the original reference.
        #[arg(long)]
        original_loss: f64,
        /// Model loss on the paraphrase.
        #[arg(long)]
        paraphrase_loss: f64,
        /// N-gram size for overlap probe.
        #[arg(long, default_value_t = 13)]
        n: usize,
    },
    /// Aggregate a JSON-Lines file of `ProbeResult` rows into a
    /// `BenchmarkReport` with per-probe mean + 95% bootstrap CI and the
    /// pre-registered flagging verdict.
    Aggregate {
        /// Path to a JSONL file of ProbeResult rows (one per line).
        probes: std::path::PathBuf,
        #[arg(long, default_value_t = 2000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
        /// Override the pre-registered n-gram threshold (default 0.05).
        #[arg(long)]
        ngram_threshold: Option<f64>,
        /// Override the pre-registered echo-canary threshold (default 0.3).
        #[arg(long)]
        echo_threshold: Option<f64>,
        /// Override the pre-registered loss-delta threshold (default 0.2).
        #[arg(long)]
        loss_delta_threshold: Option<f64>,
    },
    /// Apply Holm-Bonferroni across models for one probe (pre-reg §4).
    /// Input: a JSON array of {model, p_value}. Prints the decisions
    /// JSON. Exit code 0 if any model is rejected, 1 otherwise.
    Holm {
        input: std::path::PathBuf,
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
    },
    /// Batch n-gram probe over a benchmark JSONL and a proxy text file.
    /// Emits a ProbeResult JSONL (one row per example) suitable to pipe
    /// into `aggregate`. The echo_canary and loss_delta fields are left
    /// at 0.0 (no live model required) — this subcommand is the pure-
    /// compute probe that does not need an endpoint. Combine with `probe`
    /// per-example results for the full picture.
    NgramRun {
        /// Benchmark JSONL: one `Example` per line.
        benchmark: std::path::PathBuf,
        /// Training-set proxy text file.
        proxy: std::path::PathBuf,
        /// Output path for the ProbeResult JSONL.
        out: std::path::PathBuf,
        #[arg(long, default_value_t = 13)]
        n: usize,
    },
    /// Turn the GSM8K test split and HumanEval into probe inputs for the
    /// pre-registered study (docs/STUDY-local-models-2026-10.md): Example
    /// JSONL per benchmark, echo-canary prefix/held-out halves, paraphrase
    /// requests, and a prepare.json with input hashes and the seeded sample.
    Prepare {
        /// GSM8K test split as JSONL rows with `question` and `answer`.
        #[arg(long)]
        gsm8k: std::path::PathBuf,
        /// HumanEval JSONL with `task_id`, `prompt`, `canonical_solution`.
        #[arg(long)]
        humaneval: std::path::PathBuf,
        /// Output directory (created).
        #[arg(long)]
        out: std::path::PathBuf,
        /// GSM8K items to sample (HumanEval is used whole).
        #[arg(long, default_value_t = 300)]
        sample: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Echo-canary runner: send each `prefix` from a `*.echo.jsonl` file to an
    /// OpenAI-compatible `/v1/completions` endpoint at temperature 0 (raw
    /// prompt, no chat template) and score the continuation against
    /// `held_out`. Writes one JSON row per item: id, echo_canary (or null when
    /// the model returned nothing), continuation length. Continuations are not
    /// written anywhere.
    /// Paraphrase runner (study §Probes 2): send each `text` from a `*.paraphrase-requests.jsonl`
    /// to a chat-completions endpoint with the fixed instruction at temperature 0; accept a paraphrase
    /// only if its token count is within 0.7x-1.3x of the original (counted by the server's
    /// /tokenize) and it shares no 13-gram of words with the original. Writes accepted paraphrases
    /// with hashes, and a rejects file with reasons. Paraphrases are not harmful content and may be
    /// committed.
    ParaphraseRun {
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        inputs: std::path::PathBuf,
        #[arg(long)]
        out: std::path::PathBuf,
        #[arg(long)]
        rejects_out: std::path::PathBuf,
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        #[arg(long)]
        limit: Option<usize>,
    },
    EchoRun {
        /// Base URL, e.g. http://127.0.0.1:8099 (no trailing /v1).
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        model: String,
        /// `*.echo.jsonl` from `prepare`.
        #[arg(long)]
        inputs: std::path::PathBuf,
        #[arg(long)]
        out: std::path::PathBuf,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Stop after this many items (smoke tests).
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Print the version banner and exit.
    Version,
}

fn echo_run(endpoint: &str, model: &str, inputs: &std::path::Path, out: &std::path::Path, max_tokens: u32, limit: Option<usize>) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(inputs)?;
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(180)).build();
    let url = format!("{}/v1/completions", endpoint.trim_end_matches('/'));
    let mut rows = String::new();
    let (mut n, mut n_na) = (0usize, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if limit.is_some_and(|l| n >= l) {
            break;
        }
        let item: serde_json::Value = serde_json::from_str(line)?;
        let id = item["id"].as_str().unwrap_or("?").to_string();
        let prefix = item["prefix"].as_str().unwrap_or("");
        let held = item["held_out"].as_str().unwrap_or("");
        let body = serde_json::json!({
            "model": model, "prompt": prefix, "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "seed": 42
        });
        let resp: serde_json::Value = agent.post(&url).set("Content-Type", "application/json").send_string(&body.to_string())?.into_json()?;
        let cont = resp["choices"][0]["text"].as_str().unwrap_or("");
        let score = if cont.trim().is_empty() || held.is_empty() { None } else { Some(echo_canary_score(cont, held)) };
        if score.is_none() {
            n_na += 1;
        }
        rows.push_str(&serde_json::to_string(&serde_json::json!({
            "id": id, "echo_canary": score, "continuation_chars": cont.len(), "held_out_chars": held.len()
        }))?);
        rows.push('\n');
        n += 1;
        if n % 25 == 0 {
            eprintln!("{n} items scored");
        }
    }
    std::fs::write(out, rows)?;
    eprintln!("echo-run: {n} items, {n_na} n/a, written to {}", out.display());
    Ok(())
}


const PARAPHRASE_INSTRUCTION: &str = "rewrite in different words, same meaning, same length";

fn word_13grams(s: &str) -> std::collections::HashSet<Vec<String>> {
    let w: Vec<String> = s.split_whitespace().map(|x| x.to_lowercase()).collect();
    let mut set = std::collections::HashSet::new();
    if w.len() >= 13 { for i in 0..=w.len() - 13 { set.insert(w[i..i + 13].to_vec()); } }
    set
}

fn count_tokens(agent: &ureq::Agent, endpoint: &str, text: &str) -> anyhow::Result<usize> {
    let resp: serde_json::Value = agent.post(&format!("{}/tokenize", endpoint.trim_end_matches('/'))).set("Content-Type", "application/json").send_string(&serde_json::json!({"content": text}).to_string())?.into_json()?;
    Ok(resp["tokens"].as_array().map(|a| a.len()).unwrap_or(0))
}

fn paraphrase_run(endpoint: &str, model: &str, inputs: &std::path::Path, out: &std::path::Path, rejects_out: &std::path::Path, max_tokens: u32, limit: Option<usize>) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(inputs)?;
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(300)).build();
    let url = format!("{}/v1/chat/completions", endpoint.trim_end_matches('/'));
    let (mut accepted, mut rejects) = (String::new(), String::new());
    let (mut n, mut n_acc) = (0usize, 0usize);
    let mut reasons: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if limit.is_some_and(|l| n >= l) { break; }
        let item: serde_json::Value = serde_json::from_str(line)?;
        let id = item["id"].as_str().unwrap_or("?").to_string();
        let original = item["text"].as_str().unwrap_or("").to_string();
        let body = serde_json::json!({
            "model": model, "temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": format!("{PARAPHRASE_INSTRUCTION}. Reply with the rewritten text only.\n\n{original}")}]
        });
        let resp: serde_json::Value = agent.post(&url).set("Content-Type", "application/json").send_string(&body.to_string())?.into_json()?;
        let para = resp["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
        let n_orig = count_tokens(&agent, endpoint, &original)?;
        let n_para = count_tokens(&agent, endpoint, &para)?;
        let ratio = if n_orig == 0 { 0.0 } else { n_para as f64 / n_orig as f64 };
        let shared = !word_13grams(&original).is_disjoint(&word_13grams(&para));
        let reason = if para.is_empty() { Some("empty") } else if !(0.7..=1.3).contains(&ratio) { Some("length outside 0.7x-1.3x") } else if shared { Some("shares a 13-gram with the original") } else { None };
        match reason {
            None => {
                n_acc += 1;
                accepted.push_str(&serde_json::to_string(&serde_json::json!({"id": id, "paraphrase": para, "paraphrase_sha256": sha256_hex(para.as_bytes()), "original_sha256": sha256_hex(original.as_bytes()), "tokens_original": n_orig, "tokens_paraphrase": n_para, "paraphraser": model, "instruction": PARAPHRASE_INSTRUCTION}))?);
                accepted.push('\n');
            }
            Some(r) => {
                *reasons.entry(r.to_string()).or_insert(0) += 1;
                rejects.push_str(&serde_json::to_string(&serde_json::json!({"id": id, "reason": r, "tokens_original": n_orig, "tokens_paraphrase": n_para}))?);
                rejects.push('\n');
            }
        }
        n += 1;
        if n % 25 == 0 { eprintln!("{n} items, {n_acc} accepted"); }
    }
    std::fs::write(out, accepted)?;
    std::fs::write(rejects_out, rejects)?;
    eprintln!("paraphrase-run: {n} items, {n_acc} accepted, rejected {reasons:?}");
    Ok(())
}

/// Split text at the word boundary nearest its midpoint: (prefix, held_out).
fn split_half(s: &str) -> (String, String) {
    let mut mid = s.len() / 2;
    while mid > 0 && !s.is_char_boundary(mid) {
        mid -= 1;
    }
    let cut = s[..mid]
        .rfind(char::is_whitespace)
        .map(|i| i + s[i..].chars().next().map_or(1, |c| c.len_utf8()))
        .unwrap_or(mid);
    let cut = if cut == 0 { mid } else { cut };
    (s[..cut].to_string(), s[cut..].to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(bytes))
}

fn prepare(
    gsm8k: &std::path::Path,
    humaneval: &std::path::Path,
    out: &std::path::Path,
    sample: usize,
    seed: u64,
) -> anyhow::Result<()> {
    use rand::{Rng, SeedableRng};
    std::fs::create_dir_all(out)?;
    let g_bytes = std::fs::read(gsm8k)?;
    let h_bytes = std::fs::read(humaneval)?;
    let g_rows: Vec<serde_json::Value> = String::from_utf8_lossy(&g_bytes)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<Result<_, _>>()?;
    let h_rows: Vec<serde_json::Value> = String::from_utf8_lossy(&h_bytes)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(sample <= g_rows.len(), "sample {} > {} GSM8K rows", sample, g_rows.len());

    // Seeded sample without replacement (partial Fisher-Yates), ids = row index.
    let mut idx: Vec<usize> = (0..g_rows.len()).collect();
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    for i in 0..sample {
        let j = i + rng.gen_range(0..idx.len() - i);
        idx.swap(i, j);
    }
    let mut sample_ids: Vec<usize> = idx[..sample].to_vec();
    sample_ids.sort_unstable();

    let write_bench = |name: &str, items: Vec<(String, String, String)>| -> anyhow::Result<usize> {
        let mut ex = String::new();
        let mut echo = String::new();
        let mut para = String::new();
        for (id, prompt, reference) in &items {
            let e = Example { id: id.clone(), prompt: prompt.clone(), reference: reference.clone(), paraphrase: None };
            ex.push_str(&serde_json::to_string(&e)?);
            ex.push('\n');
            let (first, held) = split_half(reference);
            echo.push_str(&serde_json::to_string(&serde_json::json!({
                "id": id, "prefix": format!("{prompt}{first}"), "held_out": held
            }))?);
            echo.push('\n');
            para.push_str(&serde_json::to_string(&serde_json::json!({ "id": id, "text": reference }))?);
            para.push('\n');
        }
        std::fs::write(out.join(format!("{name}.examples.jsonl")), ex)?;
        std::fs::write(out.join(format!("{name}.echo.jsonl")), echo)?;
        std::fs::write(out.join(format!("{name}.paraphrase-requests.jsonl")), para)?;
        Ok(items.len())
    };

    let g_items: Vec<(String, String, String)> = sample_ids
        .iter()
        .map(|&i| {
            let r = &g_rows[i];
            (
                format!("gsm8k-test-{i}"),
                r["question"].as_str().unwrap_or("").to_string() + "\n",
                r["answer"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();
    let h_items: Vec<(String, String, String)> = h_rows
        .iter()
        .map(|r| {
            (
                r["task_id"].as_str().unwrap_or("?").to_string(),
                r["prompt"].as_str().unwrap_or("").to_string(),
                r["canonical_solution"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();
    let n_g = write_bench("gsm8k", g_items)?;
    let n_h = write_bench("humaneval", h_items)?;

    let meta = serde_json::json!({
        "tool": format!("contamcheck {}", env!("CARGO_PKG_VERSION")),
        "study": "docs/STUDY-local-models-2026-10.md",
        "inputs": {
            "gsm8k": { "path": gsm8k.to_string_lossy(), "sha256": sha256_hex(&g_bytes), "rows": g_rows.len() },
            "humaneval": { "path": humaneval.to_string_lossy(), "sha256": sha256_hex(&h_bytes), "rows": h_rows.len() }
        },
        "seed": seed,
        "gsm8k_sample_size": n_g,
        "gsm8k_sample_row_indices": sample_ids,
        "humaneval_items": n_h,
        "split_rule": "reference split at the word boundary nearest its midpoint; prefix = prompt + first half; held_out = second half"
    });
    std::fs::write(out.join("prepare.json"), serde_json::to_string_pretty(&meta)?)?;
    eprintln!("prepared {n_g} GSM8K and {n_h} HumanEval items in {}", out.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Probe {
            example,
            proxy,
            model_output,
            held_out_suffix,
            original_loss,
            paraphrase_loss,
            n,
        } => {
            let e: Example = serde_json::from_str(&std::fs::read_to_string(&example)?)?;
            let proxy_text = std::fs::read_to_string(&proxy)?;
            let out_text = std::fs::read_to_string(&model_output)?;
            let suffix = std::fs::read_to_string(&held_out_suffix)?;
            let r = ProbeResult {
                id: e.id.clone(),
                ngram_overlap: ngram_overlap(&e.reference, &proxy_text, n),
                echo_canary: echo_canary_score(&out_text, &suffix),
                loss_delta: loss_delta(original_loss, paraphrase_loss),
            };
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Cmd::Aggregate {
            probes,
            bootstrap,
            seed,
            ngram_threshold,
            echo_threshold,
            loss_delta_threshold,
        } => {
            let text = std::fs::read_to_string(&probes)?;
            let mut rows: Vec<ProbeResult> = Vec::new();
            for (line_no, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let row: ProbeResult = serde_json::from_str(trimmed)
                    .map_err(|e| anyhow::anyhow!("probes line {}: {e}", line_no + 1))?;
                rows.push(row);
            }
            let defaults = Thresholds::default();
            let t = Thresholds {
                ngram: ngram_threshold.unwrap_or(defaults.ngram),
                echo: echo_threshold.unwrap_or(defaults.echo),
                loss_delta: loss_delta_threshold.unwrap_or(defaults.loss_delta),
            };
            let report = aggregate(&rows, bootstrap, seed, t);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Holm { input, alpha } => {
            let rows: Vec<ModelProbePValue> =
                serde_json::from_str(&std::fs::read_to_string(&input)?)?;
            let decisions = holm_bonferroni(&rows, alpha);
            let any_rejected = decisions.iter().any(|d| d.rejected);
            println!("{}", serde_json::to_string_pretty(&decisions)?);
            if !any_rejected {
                std::process::exit(1);
            }
        }
        Cmd::NgramRun {
            benchmark,
            proxy,
            out,
            n,
        } => {
            let proxy_text = std::fs::read_to_string(&proxy)?;
            let bench_text = std::fs::read_to_string(&benchmark)?;
            let mut lines_written = 0usize;
            let mut out_text = String::new();
            for (line_no, line) in bench_text.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let ex: Example = serde_json::from_str(trimmed).map_err(|e| {
                    anyhow::anyhow!("benchmark line {}: {e}", line_no + 1)
                })?;
                let r = ProbeResult {
                    id: ex.id.clone(),
                    ngram_overlap: ngram_overlap(&ex.reference, &proxy_text, n),
                    echo_canary: 0.0,
                    loss_delta: 0.0,
                };
                out_text.push_str(&serde_json::to_string(&r)?);
                out_text.push('\n');
                lines_written += 1;
            }
            std::fs::write(&out, out_text)?;
            eprintln!(
                "wrote {} probe rows to {} (n={})",
                lines_written,
                out.display(),
                n
            );
        }
        Cmd::Prepare { gsm8k, humaneval, out, sample, seed } => prepare(&gsm8k, &humaneval, &out, sample, seed)?,
        Cmd::EchoRun { endpoint, model, inputs, out, max_tokens, limit } => echo_run(&endpoint, &model, &inputs, &out, max_tokens, limit)?,
        Cmd::ParaphraseRun { endpoint, model, inputs, out, rejects_out, max_tokens, limit } => paraphrase_run(&endpoint, &model, &inputs, &out, &rejects_out, max_tokens, limit)?,
        Cmd::Version => println!("contamcheck {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
