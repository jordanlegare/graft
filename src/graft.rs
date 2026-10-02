use crate::{
    data::Dataset,
    model::{DenseLayer, Network},
};
use rand::Rng;
use std::fmt;

#[derive(Debug, Clone)]
pub enum MutationKind {
    PruneNeuron { layer: usize, neuron: usize },
    MergeNeurons { layer: usize, first: usize, second: usize },
    SplitNeuron { layer: usize, neuron: usize },
    PruneConnection { layer: usize, output: usize, input: usize },
    RewireConnection {
        layer: usize,
        from_output: usize,
        from_input: usize,
        to_output: usize,
        to_input: usize,
    },
}

impl fmt::Display for MutationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PruneNeuron { layer, neuron } => {
                write!(f, "prune-neuron:L{layer}:N{neuron}")
            }
            Self::MergeNeurons {
                layer,
                first,
                second,
            } => write!(f, "merge-neurons:L{layer}:N{first}+N{second}"),
            Self::SplitNeuron { layer, neuron } => {
                write!(f, "split-neuron:L{layer}:N{neuron}")
            }
            Self::PruneConnection {
                layer,
                output,
                input,
            } => write!(f, "prune-edge:L{layer}:{output}->{input}"),
            Self::RewireConnection {
                layer,
                from_output,
                from_input,
                to_output,
                to_input,
            } => write!(
                f,
                "rewire-edge:L{layer}:{from_output}->{from_input}=>{to_output}->{to_input}"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuidedMutationConfig {
    pub min_width: usize,
    pub max_width: usize,
    pub similarity_threshold: f32,
    pub behavior_samples: usize,
    pub behavior_candidates: usize,
    pub behavior_correlation_threshold: f32,
    pub behavior_merge_tolerance: f32,
}

impl Default for GuidedMutationConfig {
    fn default() -> Self {
        Self {
            min_width: 2,
            max_width: 128,
            similarity_threshold: 0.85,
            behavior_samples: 256,
            behavior_candidates: 16,
            behavior_correlation_threshold: 0.90,
            behavior_merge_tolerance: 0.01,
        }
    }
}

fn hidden_layer_indices(network: &Network) -> impl Iterator<Item = usize> {
    0..network.layers.len().saturating_sub(1)
}

fn neuron_utility(
    network: &Network,
    layer_idx: usize,
    neuron: usize,
) -> f32 {
    let layer = &network.layers[layer_idx];

    let incoming = (0..layer.input)
        .filter_map(|input| {
            let idx = neuron * layer.input + input;
            layer.active[idx].then_some(layer.weights[idx].abs())
        })
        .sum::<f32>();

    let bias = layer.bias[neuron].abs();

    let outgoing = if layer_idx + 1 < network.layers.len() {
        let next = &network.layers[layer_idx + 1];

        (0..next.output)
            .filter_map(|output| {
                let idx = output * next.input + neuron;
                next.active[idx].then_some(next.weights[idx].abs())
            })
            .sum::<f32>()
    } else {
        0.0
    };

    (incoming + bias + 1e-6) * (outgoing + 1e-6)
}

pub fn neuron_utilities(network: &Network) -> Vec<Vec<f32>> {
    hidden_layer_indices(network)
        .map(|layer_idx| {
            (0..network.layers[layer_idx].output)
                .map(|neuron| neuron_utility(network, layer_idx, neuron))
                .collect()
        })
        .collect()
}

fn incoming_signature(network: &Network, layer_idx: usize, neuron: usize) -> Vec<f32> {
    let layer = &network.layers[layer_idx];
    let mut signature = Vec::with_capacity(layer.input + 1);

    for input in 0..layer.input {
        let idx = neuron * layer.input + input;
        signature.push(if layer.active[idx] {
            layer.weights[idx]
        } else {
            0.0
        });
    }

    signature.push(layer.bias[neuron]);
    signature
}

fn outgoing_signature(network: &Network, layer_idx: usize, neuron: usize) -> Vec<f32> {
    if layer_idx + 1 >= network.layers.len() {
        return Vec::new();
    }

    let next = &network.layers[layer_idx + 1];
    (0..next.output)
        .map(|output| {
            let idx = output * next.input + neuron;
            if next.active[idx] {
                next.weights[idx]
            } else {
                0.0
            }
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut a_norm = 0.0f64;
    let mut b_norm = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        (dot / (a_norm.sqrt() * b_norm.sqrt())) as f32
    }
}

fn neuron_similarity(network: &Network, layer_idx: usize, a: usize, b: usize) -> f32 {
    let mut left = incoming_signature(network, layer_idx, a);
    let mut right = incoming_signature(network, layer_idx, b);
    left.extend(outgoing_signature(network, layer_idx, a));
    right.extend(outgoing_signature(network, layer_idx, b));
    cosine_similarity(&left, &right)
}

fn eligible_edge_prune(layer: &DenseLayer, output: usize, input: usize) -> bool {
    let idx = layer.index(output, input);

    if !layer.active[idx] {
        return false;
    }

    let row_degree = (0..layer.input)
        .filter(|other| layer.is_active(output, *other))
        .count();

    let column_degree = (0..layer.output)
        .filter(|other| layer.is_active(*other, input))
        .count();

    row_degree > 1 && column_degree > 1
}

fn remove_row(layer: &mut DenseLayer, row: usize) {
    let mut weights = Vec::with_capacity((layer.output - 1) * layer.input);
    let mut active = Vec::with_capacity((layer.output - 1) * layer.input);

    for current in 0..layer.output {
        if current == row {
            continue;
        }

        let start = current * layer.input;
        let end = start + layer.input;
        weights.extend_from_slice(&layer.weights[start..end]);
        active.extend_from_slice(&layer.active[start..end]);
    }

    layer.output -= 1;
    layer.weights = weights;
    layer.active = active;
    layer.bias.remove(row);
}

fn remove_column(layer: &mut DenseLayer, column: usize) {
    let mut weights = Vec::with_capacity(layer.output * (layer.input - 1));
    let mut active = Vec::with_capacity(layer.output * (layer.input - 1));

    for output in 0..layer.output {
        for input in 0..layer.input {
            if input == column {
                continue;
            }

            let idx = layer.index(output, input);
            weights.push(layer.weights[idx]);
            active.push(layer.active[idx]);
        }
    }

    layer.input -= 1;
    layer.weights = weights;
    layer.active = active;
}

fn insert_row(
    layer: &mut DenseLayer,
    row: usize,
    weights_to_insert: &[f32],
    active_to_insert: &[bool],
    bias: f32,
) {
    debug_assert_eq!(weights_to_insert.len(), layer.input);
    debug_assert_eq!(active_to_insert.len(), layer.input);

    let mut weights = Vec::with_capacity((layer.output + 1) * layer.input);
    let mut active = Vec::with_capacity((layer.output + 1) * layer.input);

    for current in 0..layer.output {
        let start = current * layer.input;
        let end = start + layer.input;

        weights.extend_from_slice(&layer.weights[start..end]);
        active.extend_from_slice(&layer.active[start..end]);

        if current == row {
            weights.extend_from_slice(weights_to_insert);
            active.extend_from_slice(active_to_insert);
        }
    }

    layer.output += 1;
    layer.weights = weights;
    layer.active = active;
    layer.bias.insert(row + 1, bias);
}

fn insert_column(
    layer: &mut DenseLayer,
    column: usize,
    weights_to_insert: &[f32],
    active_to_insert: &[bool],
) {
    debug_assert_eq!(weights_to_insert.len(), layer.output);
    debug_assert_eq!(active_to_insert.len(), layer.output);

    let mut weights = Vec::with_capacity(layer.output * (layer.input + 1));
    let mut active = Vec::with_capacity(layer.output * (layer.input + 1));

    for output in 0..layer.output {
        let idx = layer.index(output, column);
        let split_weight = layer.weights[idx] * 0.5;
        let split_active = layer.active[idx];

        for input in 0..layer.input {
            let source = layer.index(output, input);

            if input == column {
                weights.push(split_weight);
                active.push(split_active);
                weights.push(weights_to_insert[output]);
                active.push(active_to_insert[output]);
            } else {
                weights.push(layer.weights[source]);
                active.push(layer.active[source]);
            }
        }
    }

    layer.input += 1;
    layer.weights = weights;
    layer.active = active;
}

fn prune_neuron(network: &mut Network, layer_idx: usize, neuron: usize) {
    remove_row(&mut network.layers[layer_idx], neuron);
    remove_column(&mut network.layers[layer_idx + 1], neuron);
}

fn merge_neurons(
    network: &mut Network,
    layer_idx: usize,
    first: usize,
    second: usize,
) {
    let layer = &network.layers[layer_idx];
    let input_width = layer.input;

    let mut merged_weights = Vec::with_capacity(input_width);
    let mut merged_active = Vec::with_capacity(input_width);

    for input in 0..input_width {
        let first_idx = layer.index(first, input);
        let second_idx = layer.index(second, input);
        let first_active = layer.active[first_idx];
        let second_active = layer.active[second_idx];

        let value = match (first_active, second_active) {
            (true, true) => (layer.weights[first_idx] + layer.weights[second_idx]) * 0.5,
            (true, false) => layer.weights[first_idx],
            (false, true) => layer.weights[second_idx],
            (false, false) => 0.0,
        };

        merged_weights.push(value);
        merged_active.push(first_active || second_active);
    }

    let merged_bias = (layer.bias[first] + layer.bias[second]) * 0.5;

    {
        let layer = &mut network.layers[layer_idx];
        let start = first * layer.input;
        let end = start + layer.input;
        layer.weights[start..end].copy_from_slice(&merged_weights);
        layer.active[start..end].copy_from_slice(&merged_active);
        layer.bias[first] = merged_bias;
    }

    remove_row(&mut network.layers[layer_idx], second);

    let next_input_width = network.layers[layer_idx].output + 1;
    let next = &mut network.layers[layer_idx + 1];
    debug_assert_eq!(next.input, next_input_width);

    for output in 0..next.output {
        let first_idx = next.index(output, first);
        let second_idx = next.index(output, second);

        let first_weight = if next.active[first_idx] {
            next.weights[first_idx]
        } else {
            0.0
        };

        let second_weight = if next.active[second_idx] {
            next.weights[second_idx]
        } else {
            0.0
        };

        let active = next.active[first_idx] || next.active[second_idx];

        next.weights[first_idx] = first_weight + second_weight;
        next.active[first_idx] = active;
    }

    remove_column(next, second);
}

fn merge_neurons_with_trace(
    network: &mut Network,
    layer_idx: usize,
    first: usize,
    second: usize,
    trace: &BehaviorTrace,
    _inputs: &[Vec<f32>],
) {
    let layer = &network.layers[layer_idx];
    let input_width = layer.input;

    let mut merged_weights = Vec::with_capacity(input_width);
    let mut merged_active = Vec::with_capacity(input_width);

    for input in 0..input_width {
        let first_idx = layer.index(first, input);
        let second_idx = layer.index(second, input);
        let first_active = layer.active[first_idx];
        let second_active = layer.active[second_idx];

        let value = match (first_active, second_active) {
            (true, true) => {
                (layer.weights[first_idx] + layer.weights[second_idx]) * 0.5
            }
            (true, false) => layer.weights[first_idx],
            (false, true) => layer.weights[second_idx],
            (false, false) => 0.0,
        };

        merged_weights.push(value);
        merged_active.push(first_active || second_active);
    }

    let merged_bias = (layer.bias[first] + layer.bias[second]) * 0.5;

    {
        let layer = &mut network.layers[layer_idx];
        let start = first * layer.input;
        let end = start + layer.input;
        layer.weights[start..end].copy_from_slice(&merged_weights);
        layer.active[start..end].copy_from_slice(&merged_active);
        layer.bias[first] = merged_bias;
    }

    // Fit each outgoing coefficient by least squares on the probe activation
    // trace: c = argmin_c ||h_m c - (h_a w_a + h_b w_b)||^2.
    // This is a local optimality step for the linear next-layer readout.
    if layer_idx + 1 < network.layers.len() && !trace.samples.is_empty() {
        let next = &mut network.layers[layer_idx + 1];

        for output in 0..next.output {
            let first_idx = next.index(output, first);
            let second_idx = next.index(output, second);

            let first_weight = if next.active[first_idx] {
                next.weights[first_idx]
            } else {
                0.0
            };
            let second_weight = if next.active[second_idx] {
                next.weights[second_idx]
            } else {
                0.0
            };

            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;

            for sample in &trace.samples {
                let merged_activation =
                    0.5 * (sample[first] + sample[second]);
                let target =
                    first_weight * sample[first]
                        + second_weight * sample[second];

                numerator +=
                    merged_activation as f64 * target as f64;
                denominator +=
                    merged_activation as f64 * merged_activation as f64;
            }

            let coefficient = if denominator > 1e-12 {
                (numerator / denominator) as f32
            } else {
                0.0
            };

            next.weights[first_idx] = coefficient;
            next.active[first_idx] = coefficient.abs() > 1e-12;
        }
    }

    remove_row(&mut network.layers[layer_idx], second);
    remove_column(&mut network.layers[layer_idx + 1], second);
}

fn split_neuron<R: Rng>(
    network: &mut Network,
    layer_idx: usize,
    neuron: usize,
    rng: &mut R,
) {
    let layer = &network.layers[layer_idx];
    let input_width = layer.input;

    let mut duplicate_weights = Vec::with_capacity(input_width);
    let mut duplicate_active = Vec::with_capacity(input_width);

    let row_start = neuron * input_width;

    let row_norm = layer.weights[row_start..row_start + input_width]
        .iter()
        .map(|x| x.abs())
        .sum::<f32>()
        .max(1.0);

    let noise_scale = row_norm * 1e-3;

    for input in 0..input_width {
        let idx = row_start + input;

        duplicate_weights.push(
            if layer.active[idx] {
                layer.weights[idx] + rng.random_range(-noise_scale..noise_scale)
            } else {
                0.0
            },
        );
        duplicate_active.push(layer.active[idx]);
    }

    let duplicate_bias =
        layer.bias[neuron] + rng.random_range(-noise_scale..noise_scale);

    insert_row(
        &mut network.layers[layer_idx],
        neuron,
        &duplicate_weights,
        &duplicate_active,
        duplicate_bias,
    );

    let next = &mut network.layers[layer_idx + 1];
    let mut duplicate_outgoing_weights = Vec::with_capacity(next.output);
    let mut duplicate_outgoing_active = Vec::with_capacity(next.output);

    for output in 0..next.output {
        let original_idx = next.index(output, neuron);

        duplicate_outgoing_weights.push(
            if next.active[original_idx] {
                next.weights[original_idx] * 0.5
            } else {
                0.0
            },
        );
        duplicate_outgoing_active.push(next.active[original_idx]);
    }

    insert_column(
        next,
        neuron,
        &duplicate_outgoing_weights,
        &duplicate_outgoing_active,
    );
}

fn prune_connection(
    network: &mut Network,
    layer_idx: usize,
    output: usize,
    input: usize,
) {
    network.layers[layer_idx].set_active(output, input, false);
}

fn rewire_connection(
    network: &mut Network,
    source: (usize, usize, usize),
    target: (usize, usize, usize),
) -> MutationKind {
    let (source_layer, source_output, source_input) = source;
    let (target_layer, target_output, target_input) = target;

    if source_layer != target_layer {
        return MutationKind::PruneConnection {
            layer: source_layer,
            output: source_output,
            input: source_input,
        };
    }

    let layer = &mut network.layers[source_layer];
    let source_idx = layer.index(source_output, source_input);
    let target_idx = layer.index(target_output, target_input);

    let value = layer.weights[source_idx];

    layer.weights[target_idx] = value;
    layer.active[target_idx] = true;
    layer.set_active(source_output, source_input, false);

    MutationKind::RewireConnection {
        layer: source_layer,
        from_output: source_output,
        from_input: source_input,
        to_output: target_output,
        to_input: target_input,
    }
}

fn apply_mutation<R: Rng>(
    network: &mut Network,
    mutation: &MutationKind,
    rng: &mut R,
) {
    match mutation {
        MutationKind::PruneNeuron { layer, neuron } => {
            prune_neuron(network, *layer, *neuron);
        }
        MutationKind::MergeNeurons {
            layer,
            first,
            second,
        } => {
            merge_neurons(network, *layer, *first, *second);
        }
        MutationKind::SplitNeuron { layer, neuron } => {
            split_neuron(network, *layer, *neuron, rng);
        }
        MutationKind::PruneConnection {
            layer,
            output,
            input,
        } => {
            prune_connection(network, *layer, *output, *input);
        }
        MutationKind::RewireConnection {
            layer,
            from_output,
            from_input,
            to_output,
            to_input,
        } => {
            let _ = rewire_connection(
                network,
                (*layer, *from_output, *from_input),
                (*layer, *to_output, *to_input),
            );
        }
    }
}

#[derive(Debug, Clone)]
struct BehaviorTrace {
    samples: Vec<Vec<f32>>,
    mean_abs: Vec<f32>,
    variance: Vec<f32>,
    zero_fraction: Vec<f32>,
    weight_saliency: Vec<f32>,
    bias_saliency: Vec<f32>,
}

#[derive(Debug, Clone)]
struct BehaviorProfile {
    inputs: Vec<Vec<f32>>,
    indices: Vec<usize>,
    layers: Vec<BehaviorTrace>,
    baseline_mse: f32,
}

fn behavior_mse(
    network: &Network,
    dataset: &Dataset,
    sample_indices: &[usize],
) -> f32 {
    if sample_indices.is_empty() {
        return f32::INFINITY;
    }

    let mut error = 0.0f64;
    let mut count = 0usize;

    for index in sample_indices.iter().copied() {
        let Ok(output) = network.forward(&dataset.inputs[index]) else {
            return f32::INFINITY;
        };

        if output.len() != dataset.targets[index].len() {
            return f32::INFINITY;
        }

        for (prediction, target) in
            output.iter().zip(dataset.targets[index].iter())
        {
            let delta = *prediction as f64 - *target as f64;
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

fn behavior_profile(
    network: &Network,
    dataset: &Dataset,
    sample_indices: &[usize],
) -> Option<BehaviorProfile> {
    let sample_count = sample_indices.len();

    if sample_count == 0 || network.layers.is_empty() {
        return None;
    }

    let mut layers = network
        .layers
        .iter()
        .map(|layer| BehaviorTrace {
            samples: Vec::with_capacity(sample_count),
            mean_abs: vec![0.0; layer.output],
            variance: vec![0.0; layer.output],
            zero_fraction: vec![0.0; layer.output],
            weight_saliency: vec![0.0; layer.weights.len()],
            bias_saliency: vec![0.0; layer.bias.len()],
        })
        .collect::<Vec<_>>();

    for sample_index in sample_indices.iter().copied() {
        let activations =
            network.forward_activations(&dataset.inputs[sample_index]).ok()?;

        for (layer_index, activation) in
            activations.iter().enumerate()
        {
            layers[layer_index].samples.push(activation.clone());

            for (neuron, value) in activation.iter().enumerate() {
                layers[layer_index].mean_abs[neuron] += value.abs();

                if value.abs() <= 1e-7 {
                    layers[layer_index].zero_fraction[neuron] += 1.0;
                }
            }
        }
    }

    let samples = sample_count as f32;

    for trace in &mut layers {
        for neuron in 0..trace.mean_abs.len() {
            trace.mean_abs[neuron] /= samples;
            trace.zero_fraction[neuron] /= samples;

            let mean = trace.samples
                .iter()
                .map(|sample| sample[neuron])
                .sum::<f32>()
                / samples;

            trace.variance[neuron] = trace.samples
                .iter()
                .map(|sample| {
                    let delta = sample[neuron] - mean;
                    delta * delta
                })
                .sum::<f32>()
                / samples;
        }
    }

    let (weight_saliency, bias_saliency) =
        parameter_saliency(network, dataset, sample_indices)?;

    for (layer, (weights, biases)) in
        layers.iter_mut()
            .zip(weight_saliency.into_iter().zip(bias_saliency))
    {
        layer.weight_saliency = weights;
        layer.bias_saliency = biases;
    }

    Some(BehaviorProfile {
        inputs: sample_indices
            .iter()
            .map(|index| dataset.inputs[*index].clone())
            .collect(),
        indices: sample_indices.to_vec(),
        layers,
        baseline_mse: behavior_mse(network, dataset, sample_indices),
    })
}

fn parameter_saliency(
    network: &Network,
    dataset: &Dataset,
    sample_indices: &[usize],
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    if sample_indices.is_empty() || network.layers.is_empty() {
        return None;
    }

    let mut weight_saliency = network
        .layers
        .iter()
        .map(|layer| vec![0.0; layer.weights.len()])
        .collect::<Vec<_>>();
    let mut bias_saliency = network
        .layers
        .iter()
        .map(|layer| vec![0.0; layer.bias.len()])
        .collect::<Vec<_>>();

    for sample_index in sample_indices.iter().copied() {
        let input = &dataset.inputs[sample_index];
        let target = &dataset.targets[sample_index];
        let mut activations = Vec::with_capacity(network.layers.len() + 1);
        let mut preactivations =
            Vec::with_capacity(network.layers.len());
        activations.push(input.clone());

        let mut current = input.clone();
        for layer in &network.layers {
            let mut z = vec![0.0; layer.output];
            let mut y = vec![0.0; layer.output];

            for output in 0..layer.output {
                let mut sum = layer.bias[output];

                for input_index in 0..layer.input {
                    let index = layer.index(output, input_index);
                    if layer.active[index] {
                        sum += layer.weights[index] * current[input_index];
                    }
                }

                z[output] = sum;
                y[output] = layer.activation.apply(sum);
            }

            preactivations.push(z);
            activations.push(y.clone());
            current = y;
        }

        let last = network.layers.len() - 1;
        if activations[last + 1].len() != target.len() {
            return None;
        }

        let output_width = target.len().max(1) as f32;
        let mut delta = vec![0.0; network.layers[last].output];

        for output in 0..delta.len() {
            let error = activations[last + 1][output] - target[output];
            delta[output] = 2.0 * error / output_width
                * network.layers[last]
                    .activation
                    .derivative(preactivations[last][output]);
        }

        for layer_index in (0..network.layers.len()).rev() {
            let layer = &network.layers[layer_index];
            let previous_activation = &activations[layer_index];

            for output in 0..layer.output {
                let gradient = delta[output];
                bias_saliency[layer_index][output] +=
                    (layer.bias[output] * gradient).abs();

                for input_index in 0..layer.input {
                    let index = layer.index(output, input_index);
                    if layer.active[index] {
                        let gradient_weight =
                            gradient * previous_activation[input_index];
                        weight_saliency[layer_index][index] +=
                            (layer.weights[index] * gradient_weight).abs();
                    }
                }
            }

            if layer_index > 0 {
                let mut previous_delta =
                    vec![0.0; network.layers[layer_index - 1].output];

                for input_index in 0..layer.input {
                    let mut sum = 0.0;

                    for output in 0..layer.output {
                        let index = layer.index(output, input_index);
                        if layer.active[index] {
                            sum += layer.weights[index] * delta[output];
                        }
                    }

                    previous_delta[input_index] = sum
                        * network.layers[layer_index - 1]
                            .activation
                            .derivative(
                                preactivations[layer_index - 1][input_index],
                            );
                }

                delta = previous_delta;
            }
        }
    }

    let scale = sample_indices.len() as f32;

    for weights in &mut weight_saliency {
        for value in weights {
            *value /= scale;
        }
    }

    for biases in &mut bias_saliency {
        for value in biases {
            *value /= scale;
        }
    }

    Some((weight_saliency, bias_saliency))
}

fn activation_correlation(
    trace: &BehaviorTrace,
    first: usize,
    second: usize,
) -> f32 {
    let mean_first = trace.samples
        .iter()
        .map(|sample| sample[first])
        .sum::<f32>()
        / trace.samples.len() as f32;

    let mean_second = trace.samples
        .iter()
        .map(|sample| sample[second])
        .sum::<f32>()
        / trace.samples.len() as f32;

    let mut covariance = 0.0f64;
    let mut first_var = 0.0f64;
    let mut second_var = 0.0f64;

    for sample in &trace.samples {
        let first_delta = (sample[first] - mean_first) as f64;
        let second_delta = (sample[second] - mean_second) as f64;

        covariance += first_delta * second_delta;
        first_var += first_delta * first_delta;
        second_var += second_delta * second_delta;
    }

    if first_var == 0.0 || second_var == 0.0 {
        0.0
    } else {
        (covariance / (first_var.sqrt() * second_var.sqrt())) as f32
    }
}

fn neuron_ablation_delta(
    network: &Network,
    dataset: &Dataset,
    baseline_mse: f32,
    layer: usize,
    neuron: usize,
    sample_indices: &[usize],
) -> f32 {
    let mut candidate = network.clone();
    prune_neuron(&mut candidate, layer, neuron);

    behavior_mse(&candidate, dataset, sample_indices) - baseline_mse
}

fn edge_ablation_delta(
    network: &Network,
    dataset: &Dataset,
    baseline_mse: f32,
    layer: usize,
    output: usize,
    input: usize,
    sample_indices: &[usize],
) -> f32 {
    let mut candidate = network.clone();
    prune_connection(&mut candidate, layer, output, input);

    behavior_mse(&candidate, dataset, sample_indices) - baseline_mse
}

fn behavior_prunable_neuron(
    network: &Network,
    profile: &BehaviorProfile,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
) -> Option<(usize, usize, f32)> {
    let mut scored = Vec::new();

    for layer in hidden_layer_indices(network) {
        if network.layers[layer].output <= config.min_width {
            continue;
        }

        for neuron in 0..network.layers[layer].output {
            let trace = &profile.layers[layer];

            let incoming_saliency = trace
                .weight_saliency
                .iter()
                .skip(neuron * network.layers[layer].input)
                .take(network.layers[layer].input)
                .copied()
                .sum::<f32>();
            let outgoing_saliency = if layer + 1 < network.layers.len() {
                let next = &profile.layers[layer + 1];
                (0..network.layers[layer + 1].output)
                    .map(|output| {
                        next.weight_saliency[
                            network.layers[layer + 1].index(output, neuron)
                        ]
                    })
                    .sum::<f32>()
            } else {
                0.0
            };
            let proxy = trace.bias_saliency[neuron]
                + incoming_saliency
                + outgoing_saliency;

            scored.push((layer, neuron, proxy));
        }
    }

    scored.sort_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let limit = config.behavior_candidates.max(1).min(scored.len());

    let mut best = None;

    for (layer, neuron, _) in scored.into_iter().take(limit) {
        let delta = neuron_ablation_delta(
            network,
            dataset,
            profile.baseline_mse,
            layer,
            neuron,
            &profile.indices,
        );

        let score = (
            delta,
            profile.layers[layer].mean_abs[neuron],
        );

        if best.is_none_or(|candidate: (usize, usize, f32, f32)| {
            score.0 < candidate.2
                || (score.0 == candidate.2 && score.1 < candidate.3)
        }) {
            best = Some((layer, neuron, score.0, score.1));
        }
    }

    best.map(|(layer, neuron, delta, _)| (layer, neuron, delta))
}

fn behavior_split_neuron(
    network: &Network,
    profile: &BehaviorProfile,
    config: &GuidedMutationConfig,
) -> Option<(usize, usize, f32)> {
    let mut best = None;

    for layer in hidden_layer_indices(network) {
        if network.layers[layer].output >= config.max_width {
            continue;
        }

        let trace = &profile.layers[layer];

        for neuron in 0..network.layers[layer].output {
            let saliency = trace
                .weight_saliency
                .iter()
                .skip(neuron * network.layers[layer].input)
                .take(network.layers[layer].input)
                .copied()
                .sum::<f32>()
                + trace.bias_saliency[neuron];

            let score = trace.variance[neuron]
                * (1.0 + trace.mean_abs[neuron])
                * (1.0 + saliency);

            if best.is_none_or(|candidate: (usize, usize, f32)| {
                score > candidate.2
            }) {
                best = Some((layer, neuron, score));
            }
        }
    }

    best
}

fn behavior_merge_pair(
    network: &Network,
    profile: &BehaviorProfile,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
) -> Option<(usize, usize, usize, f32)> {
    let mut pairs = Vec::new();

    for layer in hidden_layer_indices(network) {
        let width = network.layers[layer].output;

        if width <= config.min_width {
            continue;
        }

        let trace = &profile.layers[layer];

        for first in 0..width {
            for second in (first + 1)..width {
                let correlation =
                    activation_correlation(trace, first, second);

                let parameter_similarity =
                    neuron_similarity(network, layer, first, second);

                if correlation >= config.behavior_correlation_threshold
                    && parameter_similarity >= config.similarity_threshold
                {
                    pairs.push((
                        layer,
                        first,
                        second,
                        correlation,
                        parameter_similarity,
                    ));
                }
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.3.partial_cmp(&a.3)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let limit =
        config.behavior_candidates.max(1).min(pairs.len());

    let mut best = None;

    for (layer, first, second, correlation, parameter_similarity) in
        pairs.into_iter().take(limit)
    {
        let mut candidate = network.clone();
        merge_neurons_with_trace(
            &mut candidate,
            layer,
            first,
            second,
            &profile.layers[layer],
            &profile.inputs,
        );

        let delta = behavior_mse(
            &candidate,
            dataset,
            &profile.indices,
        ) - profile.baseline_mse;

        if delta <= config.behavior_merge_tolerance
            && best.is_none_or(
                |current: (usize, usize, usize, f32, f32)| {
                    correlation > current.3
                        || (
                            correlation == current.3
                                && parameter_similarity > current.4
                        )
                },
            )
        {
            best = Some((
                layer,
                first,
                second,
                correlation,
                parameter_similarity,
            ));
        }
    }

    best.map(|(layer, first, second, correlation, _)| {
        (layer, first, second, correlation)
    })
}

fn behavior_prunable_connection(
    network: &Network,
    profile: &BehaviorProfile,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
) -> Option<(usize, usize, usize, f32)> {
    let mut edges = Vec::new();

    for layer in 0..network.layers.len() {
        let current = &network.layers[layer];

        for output in 0..current.output {
            for input in 0..current.input {
                if !eligible_edge_prune(current, output, input) {
                    continue;
                }

                let index = current.index(output, input);
                let saliency =
                    profile.layers[layer].weight_saliency[index];

                let input_activity = if layer == 0 {
                    profile
                        .inputs
                        .iter()
                        .map(|sample| sample[input].abs())
                        .sum::<f32>()
                        / profile.inputs.len().max(1) as f32
                } else {
                    profile.layers[layer - 1].mean_abs[input]
                };

                edges.push((
                    layer,
                    output,
                    input,
                    saliency + 1e-6 * input_activity,
                ));
            }
        }
    }

    edges.sort_by(|a, b| {
        a.3.partial_cmp(&b.3)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let limit = config.behavior_candidates.max(1).min(edges.len());
    let mut best = None;

    for (layer, output, input, _) in
        edges.into_iter().take(limit)
    {
        let delta = edge_ablation_delta(
            network,
            dataset,
            profile.baseline_mse,
            layer,
            output,
            input,
            &profile.indices,
        );

        if best.is_none_or(|current: (usize, usize, usize, f32)| {
            delta < current.3
        }) {
            best = Some((layer, output, input, delta));
        }
    }

    best
}

fn behavior_rewire(
    network: &Network,
    profile: &BehaviorProfile,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
) -> Option<MutationKind> {
    let source =
        behavior_prunable_connection(
            network,
            profile,
            config,
            dataset,
        )?;

    let layer = source.0;
    let current = &network.layers[layer];

    let mut best_target = None;

    for output in 0..current.output {
        for input in 0..current.input {
            if current.is_active(output, input) {
                continue;
            }

            let mut candidate = network.clone();
            let _ = rewire_connection(
                &mut candidate,
                (layer, source.1, source.2),
                (layer, output, input),
            );

            let delta = behavior_mse(
                &candidate,
                dataset,
                &profile.indices,
            ) - profile.baseline_mse;

            if best_target
                .is_none_or(|current: (usize, usize, f32)| delta < current.2)
            {
                best_target = Some((output, input, delta));
            }
        }
    }

    let (target_output, target_input, _) = best_target?;
    Some(MutationKind::RewireConnection {
        layer,
        from_output: source.1,
        from_input: source.2,
        to_output: target_output,
        to_input: target_input,
    })
}

pub fn apply_guided_mutation<R: Rng>(
    network: &mut Network,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
    rng: &mut R,
) -> Option<MutationKind> {
    let mut sample_indices =
        (0..dataset.inputs.len()).collect::<Vec<_>>();
    sample_indices.shuffle(rng);
    sample_indices.truncate(
        config.behavior_samples.min(sample_indices.len()),
    );

    let profile = behavior_profile(
        network,
        dataset,
        &sample_indices,
    )?;

    let prune_neuron =
        behavior_prunable_neuron(
            network,
            &profile,
            config,
            dataset,
        );

    let split_neuron =
        behavior_split_neuron(network, &profile, config);

    let merge_pair =
        behavior_merge_pair(
            network,
            &profile,
            config,
            dataset,
        );

    let prune_edge =
        behavior_prunable_connection(
            network,
            &profile,
            config,
            dataset,
        );

    let rewire =
        behavior_rewire(
            network,
            &profile,
            config,
            dataset,
        );

    let mut choices = Vec::new();

    if prune_neuron.is_some() {
        choices.push(0usize);
    }

    if split_neuron.is_some() {
        choices.push(1usize);
    }

    if merge_pair.is_some() {
        choices.push(2usize);
    }

    if prune_edge.is_some() {
        choices.push(3usize);
    }

    if rewire.is_some() {
        choices.push(4usize);
    }

    if choices.is_empty() {
        return None;
    }

    let choice = choices[rng.random_range(0..choices.len())];

    let mutation = match choice {
        0 => {
            let (layer, neuron, _) = prune_neuron?;
            MutationKind::PruneNeuron { layer, neuron }
        }
        1 => {
            let (layer, neuron, _) = split_neuron?;
            MutationKind::SplitNeuron { layer, neuron }
        }
        2 => {
            let (layer, first, second, _) = merge_pair?;
            MutationKind::MergeNeurons {
                layer,
                first,
                second,
            }
        }
        3 => {
            let (layer, output, input, _) = prune_edge?;
            MutationKind::PruneConnection {
                layer,
                output,
                input,
            }
        }
        4 => rewire?,
        _ => return None,
    };

    match &mutation {
        MutationKind::MergeNeurons { layer, first, second } => {
            merge_neurons_with_trace(
                network,
                *layer,
                *first,
                *second,
                &profile.layers[*layer],
                &profile.inputs,
            );
        }
        _ => apply_mutation(network, &mutation, rng),
    }

    Some(mutation)
}

pub fn apply_guided_mutations<R: Rng>(
    network: &mut Network,
    config: &GuidedMutationConfig,
    dataset: &Dataset,
    count: usize,
    rng: &mut R,
) -> Vec<MutationKind> {
    let mut history = Vec::with_capacity(count);

    for _ in 0..count {
        let Some(mutation) =
            apply_guided_mutation(network, config, dataset, rng)
        else {
            break;
        };

        history.push(mutation);
    }

    history
}

pub fn mutation_strings(history: &[MutationKind]) -> Vec<String> {
    history.iter().map(ToString::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::random_network;

    #[test]
    fn prune_neuron_preserves_connected_shapes() {
        let mut rng = rand::rng();
        let mut network = random_network(
            &[4, 6, 3],
            crate::model::Activation::Relu,
            &mut rng,
        )
        .expect("network");

        let before_error = network
            .forward(&[0.2, -0.4, 0.7, 0.1])
            .expect("forward");

        prune_neuron(&mut network, 0, 2);

        assert_eq!(network.topology(), vec![4, 5, 3]);
        assert_eq!(network.layers[0].weights.len(), 20);
        assert_eq!(network.layers[1].weights.len(), 15);

        let after = network
            .forward(&[0.2, -0.4, 0.7, 0.1])
            .expect("forward");

        assert_eq!(after.len(), before_error.len());
    }

    #[test]
    fn split_neuron_preserves_interface() {
        let mut rng = rand::rng();
        let mut network = random_network(
            &[4, 6, 3],
            crate::model::Activation::Relu,
            &mut rng,
        )
        .expect("network");

        split_neuron(&mut network, 0, 1, &mut rng);

        assert_eq!(network.topology(), vec![4, 7, 3]);
        assert_eq!(network.layers[0].weights.len(), 28);
        assert_eq!(network.layers[1].weights.len(), 21);
    }

    #[test]
    fn connection_pruning_creates_sparse_graph() {
        let mut rng = rand::rng();
        let mut network = random_network(
            &[4, 6, 3],
            crate::model::Activation::Relu,
            &mut rng,
        )
        .expect("network");

        prune_connection(&mut network, 0, 0, 0);

        assert_eq!(network.layers[0].active_count(), 23);
        assert!(!network.layers[0].active[0]);
        assert_eq!(network.mac_count(), 41);
    }

    #[test]
    fn rewire_uses_existing_sparse_slot() {
        let mut rng = rand::rng();
        let mut network = random_network(
            &[4, 6, 3],
            crate::model::Activation::Relu,
            &mut rng,
        )
        .expect("network");

        prune_connection(&mut network, 0, 0, 0);

        let source = (0, 0, 1);
        let target = (0, 0, 0);
        let kind = rewire_connection(&mut network, source, target);

        assert!(matches!(kind, MutationKind::RewireConnection { .. }));
        assert!(network.layers[0].is_active(0, 0));
        assert!(!network.layers[0].is_active(0, 1));
    }
}
