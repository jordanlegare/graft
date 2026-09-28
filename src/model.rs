use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub input: usize,
    pub output: usize,
    pub activation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpec {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<LayerSpec>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Relu,
    Tanh,
}

impl Activation {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "relu" => Self::Relu,
            "tanh" => Self::Tanh,
            _ => Self::Linear,
        }
    }

    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::Relu => x.max(0.0),
            Self::Tanh => x.tanh(),
        }
    }

    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Self::Linear => 1.0,
            Self::Relu => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: Activation,
}

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<DenseLayer>,
}

impl Network {
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum()
    }

    pub fn mac_count(&self) -> u64 {
        self.layers.iter().map(|l| (l.input * l.output) as u64).sum()
    }

    pub fn topology(&self) -> Vec<usize> {
        if self.layers.is_empty() {
            return Vec::new();
        }

        let mut shape = Vec::with_capacity(self.layers.len() + 1);
        shape.push(self.layers[0].input);
        shape.extend(self.layers.iter().map(|l| l.output));
        shape
    }

    pub fn forward(&self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(!self.layers.is_empty(), "network contains no layers");
        anyhow::ensure!(
            input.len() == self.layers[0].input,
            "input size {} != expected {}",
            input.len(),
            self.layers[0].input
        );

        let mut x = input.to_vec();

        for layer in &self.layers {
            let mut y = vec![0.0; layer.output];

            for o in 0..layer.output {
                let mut sum = layer.bias[o];

                for i in 0..layer.input {
                    sum += layer.weights[o * layer.input + i] * x[i];
                }

                y[o] = layer.activation.apply(sum);
            }

            x = y;
        }

        Ok(x)
    }
}

pub fn random_network(
    topology: &[usize],
    hidden_activation: Activation,
    rng: &mut impl Rng,
) -> anyhow::Result<Network> {
    anyhow::ensure!(topology.len() >= 2, "topology needs input and output");

    let mut layers = Vec::with_capacity(topology.len() - 1);

    for layer_idx in 0..topology.len() - 1 {
        let input = topology[layer_idx];
        let output = topology[layer_idx + 1];

        let activation = if layer_idx + 1 == topology.len() - 1 {
            Activation::Linear
        } else {
            hidden_activation
        };

        let scale = (2.0 / input as f32).sqrt();

        let weights = (0..input * output)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        layers.push(DenseLayer {
            input,
            output,
            weights,
            bias: vec![0.0; output],
            activation,
        });
    }

    Ok(Network { layers })
}
