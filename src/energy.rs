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
    pub fn validate(&self) -> anyhow::Result<()> {
        let values = [
            ("mac_energy_pj", self.mac_energy_pj),
            ("memory_read_energy_pj", self.memory_read_energy_pj),
            ("memory_write_energy_pj", self.memory_write_energy_pj),
            ("activation_energy_pj", self.activation_energy_pj),
        ];

        for (name, value) in values {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and non-negative"
            );
        }

        Ok(())
    }

    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let profile = match path {
            Some(path) => {
                serde_json::from_str(&std::fs::read_to_string(path)?)?
            }
            None => Self::default(),
        };

        profile.validate()?;
        Ok(profile)
    }
}

/// Estimates the energy of one inference batch using a simple analytical model.
///
/// The estimate intentionally uses the active connection mask for MACs and
/// weight reads. It is therefore a useful sparse-topology proxy, but it is
/// not a physical power measurement: a runtime that executes sparse weights
/// with dense kernels may realize substantially less savings.
pub fn estimate_energy(
    network: &Network,
    hardware: &HardwareProfile,
    batch_size: usize,
) -> f64 {
    let batch = batch_size as f64;
    let mut energy = 0.0;

    for layer in &network.layers {
        let active = layer.active_count() as f64;
        let outputs = layer.output as f64;
        let bias_reads = layer.bias.len() as f64;

        let macs = batch * active;
        let weight_reads = batch * active;
        let bias_reads = batch * bias_reads;
        let writes = batch * outputs;
        let activations = batch * outputs;

        energy += macs * hardware.mac_energy_pj;
        energy += weight_reads * hardware.memory_read_energy_pj;
        energy += bias_reads * hardware.memory_read_energy_pj;
        energy += writes * hardware.memory_write_energy_pj;
        energy += activations * hardware.activation_energy_pj;
    }

    energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{random_network, Activation};

    #[test]
    fn pruning_reduces_analytical_mac_energy() {
        let mut rng = rand::rng();
        let mut network =
            random_network(&[4, 3, 2], Activation::Relu, &mut rng)
                .expect("network");

        let hardware = HardwareProfile::default();
        let dense_energy = estimate_energy(&network, &hardware, 1);

        network.layers[0].set_active(0, 0, false);
        let sparse_energy = estimate_energy(&network, &hardware, 1);

        assert!(sparse_energy < dense_energy);
        assert_eq!(network.mac_count(), 17);
    }

    #[test]
    fn zero_active_connections_remove_mac_and_weight_read_cost() {
        let mut rng = rand::rng();
        let mut network =
            random_network(&[2, 2], Activation::Linear, &mut rng)
                .expect("network");

        let hardware = HardwareProfile::default();
        let dense_energy = estimate_energy(&network, &hardware, 1);

        for output in 0..2 {
            for input in 0..2 {
                network.layers[0].set_active(output, input, false);
            }
        }

        let sparse_energy = estimate_energy(&network, &hardware, 1);
        let non_edge_cost =
            2.0 * hardware.memory_read_energy_pj
                + 2.0 * hardware.memory_write_energy_pj
                + 2.0 * hardware.activation_energy_pj;

        assert!((dense_energy - sparse_energy
            - 4.0 * (hardware.mac_energy_pj + hardware.memory_read_energy_pj))
            .abs()
            < 1e-9);
        assert!((sparse_energy - non_edge_cost).abs() < 1e-9);
    }

    #[test]
    fn invalid_hardware_profile_is_rejected() {
        let profile = HardwareProfile {
            mac_energy_pj: -1.0,
            ..HardwareProfile::default()
        };

        assert!(profile.validate().is_err());
    }
}
