use anyhow::{Context, Result};
use clap::Parser;
use graft::{
    data::Dataset,
    energy::HardwareProfile,
    graft::GuidedMutationConfig,
    model::{Activation, DenseLayer, Network},
    search::{mse, search, SearchCandidate, SearchConfig},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "neuro-search",
    about = "Search dense neural-network topologies using weight-guided grafting"
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

    #[arg(long, default_value_t = 0.75)]
    guided_fraction: f32,

    #[arg(long, default_value_t = 3)]
    guided_mutations: usize,

    #[arg(long, default_value_t = 0.85)]
    similarity_threshold: f32,

    #[arg(long)]
    export_best_dir: Option<String>,

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
    active_mask: Option<String>,
}

#[derive(Debug, Serialize)]
struct ExportManifest {
    input_size: usize,
    output_size: usize,
    layers: Vec<ExportLayer>,
}

#[derive(Debug, Serialize)]
struct ExportLayer {
    input: usize,
    output: usize,
    weights: String,
    bias: String,
    activation: String,
    active_mask: String,
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;

    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "{}: byte length {} is not divisible by 4",
        path.display(),
        bytes.len()
    );

    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_mask_file(path: &Path, expected: usize) -> Result<Vec<bool>> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;

    anyhow::ensure!(
        bytes.len() == expected,
        "{} contains {} mask bytes, expected {}",
        path.display(),
        bytes.len(),
        expected
    );

    Ok(bytes.iter().map(|x| *x != 0).collect())
}

fn resolve_parameter_path(
    manifest_path: &Path,
    declared: &str,
) -> PathBuf {
    let direct = PathBuf::from(declared);

    if direct.exists() {
        return direct;
    }

    manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(declared)
}

fn load_network(
    manifest_path: &Path,
    manifest: &Manifest,
) -> Result<Network> {
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

        let weights_path =
            resolve_parameter_path(manifest_path, &spec.weights);
        let bias_path =
            resolve_parameter_path(manifest_path, &spec.bias);

        let weights = read_f32_file(&weights_path)?;
        let bias = read_f32_file(&bias_path)?;

        anyhow::ensure!(
            weights.len() == spec.input * spec.output,
            "{} contains {} floats, expected {}",
            weights_path.display(),
            weights.len(),
            spec.input * spec.output
        );

        anyhow::ensure!(
            bias.len() == spec.output,
            "{} contains {} floats, expected {}",
            bias_path.display(),
            bias.len(),
            spec.output
        );

        let active_mask =
            if let Some(mask_path) = spec.active_mask.as_deref() {
                read_mask_file(
                    &resolve_parameter_path(
                        manifest_path,
                        mask_path,
                    ),
                    spec.input * spec.output,
                )?
            } else {
                vec![true; spec.input * spec.output]
            };

        layers.push(DenseLayer {
            input: spec.input,
            output: spec.output,
            weights,
            bias,
            activation: Activation::from_name(&spec.activation),
            active: active_mask,
        });
    }

    Ok(Network { layers })
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let bytes = values
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();

    fs::write(path, bytes)?;
    Ok(())
}

fn write_mask(path: &Path, mask: &[bool]) -> Result<()> {
    fs::write(
        path,
        mask.iter()
            .map(|x| if *x { 1u8 } else { 0u8 })
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

fn export_candidate(
    candidate: &SearchCandidate,
    output_dir: &Path,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;

    let topology = candidate.network.topology();

    let mut layers = Vec::with_capacity(candidate.network.layers.len());

    for (index, layer) in candidate.network.layers.iter().enumerate() {
        let weights_name = format!("W{index}.bin");
        let bias_name = format!("b{index}.bin");
        let mask_name = format!("mask{index}.bin");

        write_f32(&output_dir.join(&weights_name), &layer.weights)?;
        write_f32(&output_dir.join(&bias_name), &layer.bias)?;
        write_mask(&output_dir.join(&mask_name), &layer.active)?;

        layers.push(ExportLayer {
            input: layer.input,
            output: layer.output,
            weights: weights_name,
            bias: bias_name,
            activation: format!("{:?}", layer.activation).to_ascii_lowercase(),
            active_mask: mask_name,
        });
    }

    let manifest = ExportManifest {
        input_size: topology[0],
        output_size: *topology.last().expect("non-empty topology"),
        layers,
    };

    fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    #[derive(Serialize)]
    struct TopologyExport {
        topology: Vec<usize>,
        active_edges_per_layer: Vec<usize>,
        mutations: Vec<String>,
        mse: f32,
        energy_pj: f64,
    }

    let topology_export = TopologyExport {
        topology,
        active_edges_per_layer:
            candidate.network.active_edges_per_layer(),
        mutations: candidate.result.mutations.clone(),
        mse: candidate.result.mse,
        energy_pj: candidate.result.energy_pj,
    };

    fs::write(
        output_dir.join("topology.json"),
        serde_json::to_string_pretty(&topology_export)?,
    )?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    anyhow::ensure!(args.candidates > 0, "candidates must be greater than zero");
    anyhow::ensure!(args.batch_size > 0, "batch_size must be greater than zero");
    anyhow::ensure!(args.min_width <= args.max_width, "min_width > max_width");
    anyhow::ensure!(args.min_depth <= args.max_depth, "min_depth > max_depth");
    anyhow::ensure!(
        (0.0..=1.0).contains(&args.guided_fraction),
        "guided_fraction must be between 0 and 1"
    );
    anyhow::ensure!(
        args.guided_mutations <= 100,
        "guided_mutations must be <= 100"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&args.similarity_threshold),
        "similarity_threshold must be between 0 and 1"
    );
    anyhow::ensure!(
        args.max_width >= 1,
        "max_width must be at least 1"
    );

    let manifest_path = PathBuf::from(&args.manifest);

    let manifest: Manifest =
        serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;

    let dataset = Dataset::load(&args.dataset)?;
    let baseline = load_network(&manifest_path, &manifest)?;
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
    println!("  dense parameters: {}", baseline.dense_parameter_count());
    println!(
        "  active parameters: {}",
        baseline.parameter_count()
    );
    println!(
        "  active connections: {}",
        baseline.active_connection_count()
    );
    println!("  MSE: {:.8}", baseline_mse);
    println!(
        "  estimated energy: {:.3} pJ",
        baseline_energy
    );

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
        guided_fraction: args.guided_fraction,
        guided_mutations: args.guided_mutations,
        guided_config: GuidedMutationConfig {
            min_width: args.min_width,
            max_width: args.max_width,
            similarity_threshold: args.similarity_threshold,
        },
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

    for candidate in results.iter().take(20) {
        let result = &candidate.result;

        println!(
            "#{:03} origin={} topology={:?} active={} params={} mse={:.6} energy={:.3} pJ accepted={} mutations={:?}",
            result.id,
            result.origin,
            result.topology,
            result.active_connections,
            result.parameters,
            result.mse,
            result.energy_pj,
            result.accuracy_accepted,
            result.mutations
        );
    }

    let serializable_results =
        results.iter().map(|x| &x.result).collect::<Vec<_>>();

    fs::write(
        &args.output,
        serde_json::to_string_pretty(
            &serializable_results,
        )?,
    )?;

    if let Some(best) =
        results.iter().find(|x| x.result.accuracy_accepted)
    {
        println!();
        println!("Best accepted candidate:");
        println!("  origin: {}", best.result.origin);
        println!(
            "  topology: {:?}",
            best.result.topology
        );
        println!(
            "  active connections: {}",
            best.result.active_connections
        );
        println!("  MSE: {:.8}", best.result.mse);
        println!(
            "  estimated energy: {:.3} pJ",
            best.result.energy_pj
        );
        println!(
            "  mutations: {:?}",
            best.result.mutations
        );
    } else {
        println!();
        println!("No candidate satisfied the accuracy tolerance.");
    }

    if let Some(output_dir) = args.export_best_dir.as_deref() {
        let candidate = results
            .iter()
            .find(|x| x.result.accuracy_accepted)
            .or_else(|| results.first());

        if let Some(candidate) = candidate {
            let output_dir = PathBuf::from(output_dir);
            export_candidate(candidate, &output_dir)?;
            println!(
                "Exported selected candidate to {}",
                output_dir.display()
            );
        }
    }

    println!("Results written to {}", args.output);
    Ok(())
}
