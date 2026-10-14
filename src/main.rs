//! contamcheck CLI — scaffold.
//!
//! `probe` reads one `Example` from JSON plus the parameters of each
//! probe and prints the per-example `ProbeResult`. The `audit`
//! subcommand that drives a model endpoint across a benchmark and
//! aggregates results is TODO.

use clap::{Parser, Subcommand};
use contamcheck::{echo_canary_score, loss_delta, ngram_overlap, Example, ProbeResult};

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
        Cmd::Version => println!("contamcheck {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
