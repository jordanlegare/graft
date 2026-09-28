use anyhow::{Context, Result};
use clap::Parser;
use graft::{
    data::Dataset,
    energy::HardwareProfile,
    model::{Activation, DenseLayer, Network},
    search::{mse, search, SearchConfig},
};
use serde::Deserialize;
use std::fs;

#[derive(Parser, Debug)]
#[command(
    name = "neuro-search",
    about = "Search dense neural-network topologies for lower estimated energy"
)]
struct Args {
    #[arg(long)]
    manifest: String,

    #[arg(long)]
    dataset: String,

    #[arg(long)]
    hardware: Option<String>,

    #[arg(long, default_value_t = 100)]
    candidates: usize,

    #[arg(long, default_value_t = 25)]
    epochs: usize,

    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,

    #[arg(long, default_value_t = 0.01)]
    accuracy_tolerance: f32,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 4)]
    min_width: usize,

    #[arg(long, default_value_t = 64)]
    max_width: usize,

    #[arg(long, default_value_t = 1)]
    min_depth: usize,

    #[arg(long, default_value_t = 4)]
    max_depth: usize,

    #[arg(long, default_value = "relu,tanh")]
    activation_candidates: String,

    #[arg(long, default_value = "results.json")]
    output: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    input_size: usize,
    output_size: usize,
    layers: Vec<ManifestLayer>,
}

#[derive(Debug, Deserialize)]
struct ManifestLayer {
    input: usize,
    output: usize,
    weights: String,
    bias: String,
    activation: String,
}

fn read_f32_file(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading {path}"))?;

    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "{path}: byte length {} is not divisible by 4",
        bytes.len()
    );

    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn load_network(manifest: &Manifest) -> Result<Network> {
    anyhow::ensure!(
        !manifest.layers.is_empty(),
        "manifest has no layers"
    );

    let mut layers = Vec::with_capacity(manifest.layers.len());

    for (index, spec) in manifest.layers.iter().enumerate() {
        anyhow::ensure!(
            spec.input > 0 && spec.output > 0,
            "layer {index} has invalid dimensions"
        );

        if index == 0 {
            anyhow::ensure!(
                spec.input == manifest.input_size,
                "first layer input does not match manifest input_size"
            );
        } else {
            anyhow::ensure!(
                spec.input == manifest.layers[index - 1].output,
                "layer {index} input does not match the previous output"
            );
        }

        if index == manifest.layers.len() - 1 {
            anyhow::ensure!(
                spec.output == manifest.output_size,
                "last layer output does not match manifest output_size"
            );
        }

        let weights = read_f32_file(&spec.weights)?;
        let bias = read_f32_file(&spec.bias)?;

        anyhow::ensure!(
            weights.len() == spec.input * spec.output,
            "{} contains {} floats, expected {}",
            spec.weights,
            weights.len(),
            spec.input * spec.output
        );

        anyhow::ensure!(
            bias.len() == spec.output,
            "{} contains {} floats, expected {}",
            spec.bias,
            bias.len(),
            spec.output
        );

        layers.push(DenseLayer {
            input: spec.input,
            output: spec.output,
            weights,
            bias,
            activation: Activation::from_name(&spec.activation),
        });
    }

    Ok(Network { layers })
}

fn main() -> Result<()> {
    let args = Args::parse();

    anyhow::ensure!(args.candidates > 0, "candidates must be greater than zero");
    anyhow::ensure!(args.batch_size > 0, "batch_size must be greater than zero");
    anyhow::ensure!(args.min_width <= args.max_width, "min_width > max_width");
    anyhow::ensure!(args.min_depth <= args.max_depth, "min_depth > max_depth");

    let manifest: Manifest =
        serde_json::from_str(&fs::read_to_string(&args.manifest)?)?;

    let dataset = Dataset::load(&args.dataset)?;
    let baseline = load_network(&manifest)?;
    let hardware = HardwareProfile::load(args.hardware.as_deref())?;

    anyhow::ensure!(
        dataset.inputs[0].len() == manifest.input_size,
        "dataset input width does not match manifest"
    );

    anyhow::ensure!(
        dataset.targets[0].len() == manifest.output_size,
        "dataset target width does not match manifest"
    );

    let activation_candidates = args
        .activation_candidates
        .split(',')
        .map(Activation::from_name)
        .collect::<Vec<_>>();

    anyhow::ensure!(
        !activation_candidates.is_empty(),
        "no activation candidates supplied"
    );

    let baseline_mse = mse(&baseline, &dataset);
    let baseline_energy =
        graft::energy::estimate_energy(
            &baseline,
            &hardware,
            args.batch_size,
        );

    println!("Baseline");
    println!("  topology: {:?}", baseline.topology());
    println!("  parameters: {}", baseline.parameter_count());
    println!("  MACs: {}", baseline.mac_count());
    println!("  MSE: {:.8}", baseline_mse);
    println!("  estimated energy: {:.3} pJ", baseline_energy);

    let config = SearchConfig {
        candidates: args.candidates,
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        accuracy_tolerance: args.accuracy_tolerance,
        batch_size: args.batch_size,
        min_width: args.min_width,
        max_width: args.max_width,
        min_depth: args.min_depth,
        max_depth: args.max_depth,
        hidden_activations: activation_candidates,
    };

    let results =
        search(
            &dataset,
            &config,
            &hardware,
            &baseline,
        );

    println!();
    println!("Top candidates:");

    for result in results.iter().take(20) {
        println!(
            "#{:03} topology={:?} activation={} params={} MACs={} mse={:.6} energy={:.3} pJ accepted={}",
            result.id,
            result.topology,
            result.hidden_activation,
            result.parameters,
            result.macs,
            result.mse,
            result.energy_pj,
            result.accuracy_accepted
        );
    }

    fs::write(
        &args.output,
        serde_json::to_string_pretty(&results)?,
    )?;

    match results.iter().find(|candidate| candidate.accuracy_accepted) {
        Some(best) => {
            println!();
            println!("Best accepted candidate by estimated energy:");
            println!("  topology: {:?}", best.topology);
            println!("  activation: {}", best.hidden_activation);
            println!("  MSE: {:.8}", best.mse);
            println!("  energy: {:.3} pJ", best.energy_pj);
        }
        None => {
            println!();
            println!("No candidate satisfied the accuracy tolerance.");
        }
    }

    println!("Results written to {}", args.output);
    Ok(())
}
