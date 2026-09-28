use crate::model::Network;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub mac_energy_pj: f64,
    pub memory_read_energy_pj: f64,
    pub memory_write_energy_pj: f64,
    pub activation_energy_pj: f64,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            mac_energy_pj: 1.0,
            memory_read_energy_pj: 2.0,
            memory_write_energy_pj: 2.5,
            activation_energy_pj: 0.2,
        }
    }
}

impl HardwareProfile {
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        match path {
            Some(path) => Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?),
            None => Ok(Self::default()),
        }
    }
}

pub fn estimate_energy(
    network: &Network,
    hardware: &HardwareProfile,
    batch_size: usize,
) -> f64 {
    let mut energy = 0.0;

    for layer in &network.layers {
        let batch = batch_size as f64;
        let macs = batch * layer.input as f64 * layer.output as f64;
        let reads = batch * layer.weights.len() as f64;
        let writes = batch * layer.output as f64;
        let activations = batch * layer.output as f64;
        let bias_reads = batch * layer.bias.len() as f64;

        energy += macs * hardware.mac_energy_pj;
        energy += reads * hardware.memory_read_energy_pj;
        energy += writes * hardware.memory_write_energy_pj;
        energy += activations * hardware.activation_energy_pj;
        energy += bias_reads * hardware.memory_read_energy_pj;
    }

    energy
}
