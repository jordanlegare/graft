use anyhow::{Context, Result};
use clap::Parser;
use graft::{
    data::Dataset,
    graph::{
        graph_mse, search_graph, GraphNetwork, GraphSearchConfig,
    },
};
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use std::fs;

#[derive(Parser, Debug)]
#[command(
    name = "graph-search",
    about = "Search convolutional, attention, recurrent, residual, and arbitrary DAG graphs"
)]
struct Args {
    #[arg(long)]
    graph: String,

    #[arg(long)]
    dataset: String,

    #[arg(long, default_value_t = 32)]
    candidates: usize,

    #[arg(long, default_value_t = 4)]
    mutations: usize,

    #[arg(long, default_value_t = 5)]
    epochs: usize,

    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,

    #[arg(long, default_value_t = 0.01)]
    accuracy_tolerance: f32,

    #[arg(long, default_value_t = 0.15)]
    validation_fraction: f32,

    #[arg(long, default_value_t = 0.15)]
    holdout_fraction: f32,

    #[arg(long, default_value_t = 42)]
    split_seed: u64,

    #[arg(long, default_value_t = 42)]
    search_seed: u64,

    #[arg(long)]
    export_best: Option<String>,

    #[arg(long, default_value = "graph-results.json")]
    output: String,
}

#[derive(Debug, Serialize)]
struct ResultRow {
    id: usize,
    validation_mse: f32,
    holdout_mse: f32,
    accuracy_accepted: bool,
    parameters: usize,
    macs: u64,
    mutations: Vec<String>,
}

fn dataset_mse(
    graph: &GraphNetwork,
    dataset: &Dataset,
) -> Result<f32> {
    graph_mse(
        graph,
        &dataset.inputs,
        &dataset.targets,
    )
    .context("evaluating graph dataset")
}

fn main() -> Result<()> {
    let args = Args::parse();

    anyhow::ensure!(args.candidates > 0, "candidates must be > 0");
    anyhow::ensure!(args.mutations > 0, "mutations must be > 0");
    anyhow::ensure!(args.epochs > 0, "epochs must be > 0");
    anyhow::ensure!(
        args.learning_rate > 0.0 && args.learning_rate.is_finite(),
        "learning_rate must be finite and > 0"
    );

    let graph: GraphNetwork =
        serde_json::from_str(
            &fs::read_to_string(&args.graph)
                .with_context(|| format!("reading {}", args.graph))?,
        )?;
    graph.validate()?;

    let dataset = Dataset::load(&args.dataset)?;
    let (train, validation, holdout) = dataset.split_three_way(
        args.validation_fraction,
        args.holdout_fraction,
        args.split_seed,
    )?;

    let input_shape = graph.input_shape()?;
    anyhow::ensure!(
        train.inputs[0].len() == input_shape.size(),
        "dataset input width {} != graph input width {}",
        train.inputs[0].len(),
        input_shape.size()
    );

    let output_shape = graph.output_shape()?;
    anyhow::ensure!(
        train.targets[0].len() == output_shape.size(),
        "dataset target width {} != graph output width {}",
        train.targets[0].len(),
        output_shape.size()
    );

    let baseline_validation_mse =
        dataset_mse(&graph, &validation)?;
    let mut rng = StdRng::seed_from_u64(args.search_seed);

    let results = search_graph(
        &graph,
        &train.inputs,
        &train.targets,
        &validation.inputs,
        &validation.targets,
        &GraphSearchConfig {
            candidates: args.candidates,
            mutations: args.mutations,
            epochs: args.epochs,
            learning_rate: args.learning_rate,
            accuracy_tolerance: args.accuracy_tolerance,
        },
        &mut rng,
    )?;

    println!("Graph baseline");
    println!(
        "  input shape: {}x{}",
        input_shape.channels,
        input_shape.length
    );
    println!(
        "  output size: {}",
        output_shape.size()
    );
    println!(
        "  train/validation/holdout samples: {}/{}/{}",
        train.inputs.len(),
        validation.inputs.len(),
        holdout.inputs.len()
    );
    println!(
        "  validation MSE: {:.8}",
        baseline_validation_mse
    );
    println!("  nodes: {}", graph.nodes.len());
    println!("  parameters: {}", graph.parameter_count());
    println!("  MACs: {}", graph.mac_count());

    let mut rows = Vec::with_capacity(results.len());

    for (candidate, network) in &results {
        let holdout_mse =
            dataset_mse(network, &holdout)?;

        rows.push(ResultRow {
            id: candidate.id,
            validation_mse: candidate.mse,
            holdout_mse,
            accuracy_accepted: candidate.accuracy_accepted,
            parameters: candidate.parameters,
            macs: candidate.macs,
            mutations: candidate
                .mutations
                .iter()
                .map(|mutation| format!("{mutation:?}"))
                .collect(),
        });
    }

    rows.sort_by(|a, b| {
        b.accuracy_accepted
            .cmp(&a.accuracy_accepted)
            .then_with(|| a.macs.cmp(&b.macs))
            .then_with(|| {
                a.validation_mse
                    .partial_cmp(&b.validation_mse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    fs::write(
        &args.output,
        serde_json::to_string_pretty(&rows)?,
    )?;

    if let Some(path) = args.export_best.as_deref()
        && let Some((_, graph)) = results.first()
    {
        fs::write(
            path,
            serde_json::to_string_pretty(graph)?,
        )?;
        println!("Exported graph to {path}");
    }

    println!("Results written to {}", args.output);
    Ok(())
}
