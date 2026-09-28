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
            Self::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
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

    // Explicit connection support. A false entry represents a pruned edge.
    // Inactive weights are set to zero and exported together with the mask.
    pub active: Vec<bool>,
}

impl DenseLayer {
    pub fn new(
        input: usize,
        output: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
        activation: Activation,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weights.len() == input * output,
            "weight count {} != {}x{}",
            weights.len(),
            input,
            output
        );
        anyhow::ensure!(
            bias.len() == output,
            "bias count {} != output {}",
            bias.len(),
            output
        );

        Ok(Self {
            input,
            output,
            active: vec![true; input * output],
            weights,
            bias,
            activation,
        })
    }

    #[inline]
    pub fn index(&self, output: usize, input: usize) -> usize {
        output * self.input + input
    }

    #[inline]
    pub fn is_active(&self, output: usize, input: usize) -> bool {
        self.active[self.index(output, input)]
    }

    #[inline]
    pub fn set_active(
        &mut self,
        output: usize,
        input: usize,
        active: bool,
    ) {
        let idx = self.index(output, input);
        self.active[idx] = active;

        if !active {
            self.weights[idx] = 0.0;
        }
    }

    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|x| **x).count()
    }
}

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<DenseLayer>,
}

impl Network {
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.active_count() + l.bias.len())
            .sum()
    }

    pub fn dense_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum()
    }

    pub fn active_connection_count(&self) -> usize {
        self.layers.iter().map(DenseLayer::active_count).sum()
    }

    pub fn mac_count(&self) -> u64 {
        self.active_connection_count() as u64
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

    pub fn active_edges_per_layer(&self) -> Vec<usize> {
        self.layers.iter().map(DenseLayer::active_count).collect()
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

            for (o, y_value) in y.iter_mut().enumerate().take(layer.output) {
                let mut sum = layer.bias[o];

                for (i, x_value) in x.iter().enumerate().take(layer.input) {
                    let idx = o * layer.input + i;

                    if layer.active[idx] {
                        sum += layer.weights[idx] * *x_value;
                    }
                }

                *y_value = layer.activation.apply(sum);
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
            active: vec![true; input * output],
        });
    }

    Ok(Network { layers })
}
