use crate::{
    data::Dataset,
    energy::{estimate_energy, HardwareProfile},
    graft::{
        apply_guided_mutations, mutation_strings, GuidedMutationConfig,
    },
    model::{Activation, Network},
};
use rand::Rng;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub candidates: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub accuracy_tolerance: f32,
    pub batch_size: usize,
    pub min_width: usize,
    pub max_width: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub hidden_activations: Vec<Activation>,

    // Fraction of candidates seeded from the supplied trained network.
    pub guided_fraction: f32,
    // Number of parameter-guided mutations applied to a guided candidate.
    pub guided_mutations: usize,
    pub guided_config: GuidedMutationConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateResult {
    pub id: usize,
    pub origin: String,
    pub topology: Vec<usize>,
    pub hidden_activation: String,
    pub mutations: Vec<String>,
    pub parameters: usize,
    pub dense_parameters: usize,
    pub active_connections: usize,
    pub macs: u64,
    pub mse: f32,
    pub energy_pj: f64,
    pub accuracy_accepted: bool,
}

#[derive(Debug, Clone)]
pub struct SearchCandidate {
    pub result: CandidateResult,
    pub network: Network,
}

pub fn mse(network: &Network, dataset: &Dataset) -> f32 {
    let mut error = 0.0f64;
    let mut count = 0usize;

    for (input, target) in dataset.inputs.iter().zip(&dataset.targets) {
        let Ok(output) = network.forward(input) else {
            return f32::INFINITY;
        };

        if output.len() != target.len() {
            return f32::INFINITY;
        }

        for (prediction, expected) in output.iter().zip(target) {
            let delta = *prediction as f64 - *expected as f64;
            error += delta * delta;
            count += 1;
        }
    }

    if count == 0 {
        f32::INFINITY
    } else {
        (error / count as f64) as f32
    }
}

pub fn generate_topology(
    input: usize,
    output: usize,
    cfg: &SearchConfig,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let depth = rng.random_range(cfg.min_depth..=cfg.max_depth);
    let mut topology = Vec::with_capacity(depth + 2);
    topology.push(input);

    for _ in 0..depth {
        topology.push(rng.random_range(cfg.min_width..=cfg.max_width));
    }

    topology.push(output);
    topology
}

pub fn train(
    network: &mut Network,
    dataset: &Dataset,
    epochs: usize,
    learning_rate: f32,
) {
    for _ in 0..epochs {
        for (input, target) in dataset.inputs.iter().zip(&dataset.targets) {
            train_sample(network, input, target, learning_rate);
        }
    }
}

fn train_sample(
    network: &mut Network,
    input: &[f32],
    target: &[f32],
    learning_rate: f32,
) {
    if network.layers.is_empty() {
        return;
    }

    let mut activations = Vec::with_capacity(network.layers.len() + 1);
    let mut preactivations = Vec::with_capacity(network.layers.len());

    activations.push(input.to_vec());
    let mut current = input.to_vec();

    for layer in &network.layers {
        let mut z = vec![0.0; layer.output];
        let mut y = vec![0.0; layer.output];

        for o in 0..layer.output {
            let mut sum = layer.bias[o];

            for i in 0..layer.input {
                let idx = layer.index(o, i);

                if layer.active[idx] {
                    sum += layer.weights[idx] * current[i];
                }
            }

            z[o] = sum;
            y[o] = layer.activation.apply(sum);
        }

        preactivations.push(z);
        activations.push(y.clone());
        current = y;
    }

    let last = network.layers.len() - 1;

    if activations[last + 1].len() != target.len() {
        return;
    }

    let mut delta = vec![0.0; network.layers[last].output];

    for j in 0..delta.len() {
        let output = activations[last + 1][j];
        let error = output - target[j];

        delta[j] =
            error
                * network.layers[last]
                    .activation
                    .derivative(preactivations[last][j]);
    }

    for layer_idx in (0..network.layers.len()).rev() {
        let input_size = network.layers[layer_idx].input;
        let output_size = network.layers[layer_idx].output;
        let weights_before = network.layers[layer_idx].weights.clone();
        let active_before = network.layers[layer_idx].active.clone();
        let previous_activation = activations[layer_idx].clone();

        let previous_delta = if layer_idx > 0 {
            let mut result = vec![0.0; input_size];

            for i in 0..input_size {
                let mut sum = 0.0;

                for o in 0..output_size {
                    let idx = o * input_size + i;

                    if active_before[idx] {
                        sum += weights_before[idx] * delta[o];
                    }
                }

                let previous_z = preactivations[layer_idx - 1][i];
                let previous_activation_fn =
                    network.layers[layer_idx - 1].activation;

                result[i] =
                    sum * previous_activation_fn.derivative(previous_z);
            }

            Some(result)
        } else {
            None
        };

        {
            let layer = &mut network.layers[layer_idx];

            for o in 0..output_size {
                layer.bias[o] -= learning_rate * delta[o];

                for i in 0..input_size {
                    let idx = layer.index(o, i);

                    if layer.active[idx] {
                        layer.weights[idx] -=
                            learning_rate * delta[o] * previous_activation[i];
                    }
                }
            }
        }

        delta = previous_delta.unwrap_or_default();
    }
}

pub fn search(
    dataset: &Dataset,
    cfg: &SearchConfig,
    hardware: &HardwareProfile,
    baseline: &Network,
) -> Vec<SearchCandidate> {
    let baseline_error = mse(baseline, dataset);
    let mut rng = rand::rng();
    let mut results = Vec::with_capacity(cfg.candidates);

    for id in 0..cfg.candidates {
        let use_guided =
            rng.random::<f32>() < cfg.guided_fraction;

        let (mut candidate, origin, activation_name, mutation_history) =
            if use_guided {
                let mut candidate = baseline.clone();

                let mutation_count = if cfg.guided_mutations == 0 {
                    0
                } else {
                    rng.random_range(1..=cfg.guided_mutations)
                };

                let history = apply_guided_mutations(
                    &mut candidate,
                    &cfg.guided_config,
                    mutation_count,
                    &mut rng,
                );

                (
                    candidate,
                    "guided",
                    "baseline-activation".to_owned(),
                    mutation_strings(&history),
                )
            } else {
                let topology = generate_topology(
                    baseline.layers[0].input,
                    baseline.layers.last()
                        .expect("non-empty baseline")
                        .output,
                    cfg,
                    &mut rng,
                );

                let activation =
                    cfg.hidden_activations[
                        rng.random_range(0..cfg.hidden_activations.len())
                    ];

                let candidate =
                    crate::model::random_network(
                        &topology,
                        activation,
                        &mut rng,
                    )
                    .expect("generated topology must be valid");

                (
                    candidate,
                    "random",
                    format!("{activation:?}"),
                    Vec::new(),
                )
            };

        train(
            &mut candidate,
            dataset,
            cfg.epochs,
            cfg.learning_rate,
        );

        let candidate_mse = mse(&candidate, dataset);
        let energy =
            estimate_energy(&candidate, hardware, cfg.batch_size);

        let result = CandidateResult {
            id,
            origin: origin.to_owned(),
            topology: candidate.topology(),
            hidden_activation: activation_name,
            mutations: mutation_history,
            parameters: candidate.parameter_count(),
            dense_parameters: candidate.dense_parameter_count(),
            active_connections: candidate.active_connection_count(),
            macs: candidate.mac_count(),
            mse: candidate_mse,
            energy_pj: energy,
            accuracy_accepted:
                candidate_mse <= baseline_error + cfg.accuracy_tolerance,
        };

        results.push(SearchCandidate {
            result,
            network: candidate,
        });
    }

    results.sort_by(|a, b| {
        match (
            a.result.accuracy_accepted,
            b.result.accuracy_accepted,
        ) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a
                .result
                .energy_pj
                .partial_cmp(&b.result.energy_pj)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    results
}
