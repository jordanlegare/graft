use crate::{
    data::Dataset,
    energy::{estimate_energy, HardwareProfile},
    graft::{
        apply_guided_mutations, mutation_strings, GuidedMutationConfig,
    },
    model::{Activation, Network},
    statistics,
};
use rand::{rngs::StdRng, Rng, SeedableRng, seq::SliceRandom};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub candidates: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub accuracy_tolerance: f32,
    pub relative_accuracy_tolerance: f32,
    pub bootstrap_samples: usize,
    pub batch_size: usize,
    pub training_macs_budget: Option<u64>,
    pub min_width: usize,
    pub max_width: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub hidden_activations: Vec<Activation>,
    pub seed: u64,
    pub guided_fraction: f32,
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
    pub holdout_mse: Option<f32>,
    pub validation_delta_mean: f32,
    pub validation_delta_ucb: f32,
    pub acceptance_tolerance: f32,
    pub energy_pj: f64,
    pub training_epochs: usize,
    pub training_macs: u64,
    pub pareto_optimal: bool,
    pub accuracy_accepted: bool,
}

#[derive(Debug, Clone)]
pub struct SearchCandidate {
    pub result: CandidateResult,
    pub network: Network,
}

pub fn sample_mse_losses(network: &Network, dataset: &Dataset) -> Vec<f32> {
    dataset
        .inputs
        .iter()
        .zip(&dataset.targets)
        .map(|(input, target)| {
            let Ok(output) = network.forward(input) else {
                return f32::INFINITY;
            };

            if output.len() != target.len() || output.is_empty() {
                return f32::INFINITY;
            }

            let total = output
                .iter()
                .zip(target)
                .map(|(prediction, expected)| {
                    let delta = *prediction as f64 - *expected as f64;
                    delta * delta
                })
                .sum::<f64>();

            (total / output.len() as f64) as f32
        })
        .collect()
}

pub fn mse(network: &Network, dataset: &Dataset) -> f32 {
    statistics::mean(&sample_mse_losses(network, dataset))
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
    if dataset.inputs.is_empty() {
        return;
    }

    for _ in 0..epochs {
        for (input, target) in dataset.inputs.iter().zip(&dataset.targets) {
            train_sample(network, input, target, learning_rate);
        }
    }
}

pub fn train_shuffled(
    network: &mut Network,
    dataset: &Dataset,
    epochs: usize,
    learning_rate: f32,
    seed: u64,
) {
    if dataset.inputs.is_empty() {
        return;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut order = (0..dataset.inputs.len()).collect::<Vec<_>>();

    for _ in 0..epochs {
        order.shuffle(&mut rng);

        for index in &order {
            train_sample(
                network,
                &dataset.inputs[*index],
                &dataset.targets[*index],
                learning_rate,
            );
        }
    }
}

fn train_epochs_for_budget(
    sample_count: usize,
    macs: u64,
    configured_epochs: usize,
    training_macs_budget: Option<u64>,
) -> usize {
    match training_macs_budget {
        Some(budget) if budget > 0 => {
            let per_epoch = (macs.max(1) as u128)
                .saturating_mul(sample_count.max(1) as u128);
            let epochs = (budget as u128) / per_epoch;
            epochs.clamp(1, usize::MAX as u128) as usize
        }
        _ => configured_epochs,
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

        for (o, (z_value, y_value)) in
            z.iter_mut().zip(y.iter_mut()).enumerate().take(layer.output)
        {
            let mut sum = layer.bias[o];

            for (i, current_value) in
                current.iter().enumerate().take(layer.input)
            {
                let idx = layer.index(o, i);

                if layer.active[idx] {
                    sum += layer.weights[idx] * *current_value;
                }
            }

            *z_value = sum;
            *y_value = layer.activation.apply(sum);
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

        delta[j] = error
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

            for (i, result_value) in
                result.iter_mut().enumerate().take(input_size)
            {
                let mut sum = 0.0;

                for (o, delta_value) in
                    delta.iter().enumerate().take(output_size)
                {
                    let idx = o * input_size + i;

                    if active_before[idx] {
                        sum += weights_before[idx] * *delta_value;
                    }
                }

                let previous_z = preactivations[layer_idx - 1][i];
                let previous_activation_fn =
                    network.layers[layer_idx - 1].activation;

                *result_value =
                    sum * previous_activation_fn.derivative(previous_z);
            }

            Some(result)
        } else {
            None
        };

        {
            let layer = &mut network.layers[layer_idx];

            for (o, delta_value) in
                delta.iter().enumerate().take(output_size)
            {
                layer.bias[o] -= learning_rate * *delta_value;

                for (i, previous_value) in
                    previous_activation.iter().enumerate().take(input_size)
                {
                    let idx = layer.index(o, i);

                    if layer.active[idx] {
                        layer.weights[idx] -=
                            learning_rate * *delta_value * *previous_value;
                    }
                }
            }
        }

        delta = previous_delta.unwrap_or_default();
    }
}

fn pareto_flags(results: &mut [SearchCandidate]) {
    for i in 0..results.len() {
        let mut pareto = true;

        if results[i].result.accuracy_accepted {
            for j in 0..results.len() {
                if i == j || !results[j].result.accuracy_accepted {
                    continue;
                }

                let i_energy = results[i].result.energy_pj;
                let j_energy = results[j].result.energy_pj;
                let i_mse = results[i].result.mse;
                let j_mse = results[j].result.mse;

                if j_energy <= i_energy
                    && j_mse <= i_mse
                    && (j_energy < i_energy || j_mse < i_mse)
                {
                    pareto = false;
                    break;
                }
            }
        } else {
            pareto = false;
        }

        results[i].result.pareto_optimal = pareto;
    }
}

pub fn search(
    train_dataset: &Dataset,
    probe_dataset: &Dataset,
    validation_dataset: &Dataset,
    cfg: &SearchConfig,
    hardware: &HardwareProfile,
    baseline: &Network,
) -> Vec<SearchCandidate> {
    let baseline_losses = sample_mse_losses(baseline, validation_dataset);
    let baseline_error = statistics::mean(&baseline_losses);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut results = Vec::with_capacity(cfg.candidates);

    for id in 0..cfg.candidates {
        let use_guided = rng.random::<f32>() < cfg.guided_fraction;

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
                    probe_dataset,
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
                    baseline.layers.last().expect("non-empty baseline").output,
                    cfg,
                    &mut rng,
                );
                let activation =
                    cfg.hidden_activations[rng.random_range(0..cfg.hidden_activations.len())];
                let candidate = crate::model::random_network(
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

        let candidate_macs = candidate.mac_count();
        let training_epochs = train_epochs_for_budget(
            train_dataset.inputs.len(),
            candidate_macs,
            cfg.epochs,
            cfg.training_macs_budget,
        );
        train_shuffled(
            &mut candidate,
            train_dataset,
            training_epochs,
            cfg.learning_rate,
            cfg.seed ^ ((id as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        );

        let candidate_losses =
            sample_mse_losses(&candidate, validation_dataset);
        let candidate_mse = statistics::mean(&candidate_losses);
        let differences = candidate_losses
            .iter()
            .zip(&baseline_losses)
            .map(|(candidate_loss, baseline_loss)| {
                *candidate_loss - *baseline_loss
            })
            .collect::<Vec<_>>();

        let paired = statistics::paired_test(
            &differences,
            baseline_error,
            cfg.accuracy_tolerance,
            cfg.relative_accuracy_tolerance,
            cfg.bootstrap_samples,
            cfg.seed ^ ((id as u64).wrapping_mul(0xD1B54A32D192ED03)),
        );

        let energy = estimate_energy(
            &candidate,
            hardware,
            cfg.batch_size,
        );
        let training_macs =
            candidate_macs.saturating_mul(
                training_epochs as u64,
            ).saturating_mul(train_dataset.inputs.len() as u64);

        results.push(SearchCandidate {
            result: CandidateResult {
                id,
                origin: origin.to_owned(),
                topology: candidate.topology(),
                hidden_activation: activation_name,
                mutations: mutation_history,
                parameters: candidate.parameter_count(),
                dense_parameters: candidate.dense_parameter_count(),
                active_connections: candidate.active_connection_count(),
                macs: candidate_macs,
                mse: candidate_mse,
                holdout_mse: None,
                validation_delta_mean: paired.mean_difference,
                validation_delta_ucb: paired.upper_confidence_bound,
                acceptance_tolerance: paired.tolerance,
                energy_pj: energy,
                training_epochs,
                training_macs,
                pareto_optimal: false,
                accuracy_accepted: paired.accepted,
            },
            network: candidate,
        });
    }

    pareto_flags(&mut results);

    results.sort_by(|a, b| {
        match (
            a.result.accuracy_accepted,
            b.result.accuracy_accepted,
        ) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a
                .result
                .pareto_optimal
                .cmp(&b.result.pareto_optimal)
                .reverse()
                .then_with(|| {
                    a.result.energy_pj
                        .partial_cmp(&b.result.energy_pj)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    a.result.mse
                        .partial_cmp(&b.result.mse)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.result.id.cmp(&b.result.id)),
        }
    });

    results
}

pub fn evaluate_holdout(
    results: &mut [SearchCandidate],
    holdout_dataset: &Dataset,
) {
    for (index, candidate) in results.iter_mut().enumerate() {
        candidate.result.holdout_mse = if index == 0 {
            Some(mse(&candidate.network, holdout_dataset))
        } else {
            None
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Activation;

    #[test]
    fn sample_mse_losses_returns_one_loss_per_sample() {
        let mut rng = StdRng::seed_from_u64(11);
        let network =
            crate::model::random_network(&[1, 1], Activation::Linear, &mut rng)
                .expect("network");
        let dataset = Dataset {
            inputs: vec![vec![1.0], vec![2.0]],
            targets: vec![vec![0.0], vec![0.0]],
        };

        assert_eq!(sample_mse_losses(&network, &dataset).len(), 2);
    }

    #[test]
    fn mlp_gradient_matches_finite_difference() {
        let mut network = Network {
            layers: vec![crate::model::DenseLayer {
                input: 1,
                output: 1,
                weights: vec![0.2],
                bias: vec![0.1],
                activation: Activation::Linear,
                active: vec![true],
            }],
        };

        let input = [0.7f32];
        let target = [0.4f32];
        let epsilon = 1e-3f32;
        let learning_rate = 1e-4f32;

        let loss_with_weight = |weight: f32| {
            let mut candidate = network.clone();
            candidate.layers[0].weights[0] = weight;
            let prediction =
                candidate.forward(&input).expect("forward")[0];
            let delta = prediction - target[0];
            0.5 * delta * delta
        };

        let plus = loss_with_weight(network.layers[0].weights[0] + epsilon);
        let minus = loss_with_weight(network.layers[0].weights[0] - epsilon);
        let numerical = (plus - minus) / (2.0 * epsilon);
        let before = network.layers[0].weights[0];

        train(&mut network, &Dataset {
            inputs: vec![input.to_vec()],
            targets: vec![target.to_vec()],
        }, 1, learning_rate);

        let analytical =
            (before - network.layers[0].weights[0]) / learning_rate;
        let relative_error = (analytical - numerical).abs()
            / (1.0 + analytical.abs() + numerical.abs());

        assert!(
            relative_error < 1e-4,
            "analytical={analytical}, numerical={numerical}, relative_error={relative_error}"
        );
    }

    #[test]
    fn budget_training_produces_at_least_one_epoch() {
        assert_eq!(train_epochs_for_budget(10, 100, 5, Some(1)), 1);
        assert_eq!(train_epochs_for_budget(10, 100, 5, Some(2_000)), 2);
        assert_eq!(train_epochs_for_budget(10, 100, 5, None), 5);
    }
}
