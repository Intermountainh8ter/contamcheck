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
    /// Print the version banner and exit.
    Version,
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
        Cmd::Version => println!("contamcheck {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
