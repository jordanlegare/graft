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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct WorkloadCost {
    pub macs: u64,
    pub memory_reads: u64,
    pub memory_writes: u64,
    pub activation_ops: u64,
}

impl WorkloadCost {
    pub fn scale(self, factor: usize) -> Self {
        let factor = factor as u64;
        Self {
            macs: self.macs.saturating_mul(factor),
            memory_reads: self.memory_reads.saturating_mul(factor),
            memory_writes: self.memory_writes.saturating_mul(factor),
            activation_ops: self.activation_ops.saturating_mul(factor),
        }
    }

    pub fn add_assign(&mut self, other: Self) {
        self.macs = self.macs.saturating_add(other.macs);
        self.memory_reads =
            self.memory_reads.saturating_add(other.memory_reads);
        self.memory_writes =
            self.memory_writes.saturating_add(other.memory_writes);
        self.activation_ops =
            self.activation_ops.saturating_add(other.activation_ops);
    }
}

pub fn estimate_energy_from_cost(
    cost: WorkloadCost,
    hardware: &HardwareProfile,
) -> f64 {
    cost.macs as f64 * hardware.mac_energy_pj
        + cost.memory_reads as f64
            * hardware.memory_read_energy_pj
        + cost.memory_writes as f64
            * hardware.memory_write_energy_pj
        + cost.activation_ops as f64
            * hardware.activation_energy_pj
}

pub fn network_workload_cost(
    network: &Network,
    batch_size: usize,
) -> WorkloadCost {
    let batch = batch_size as u64;
    let mut cost = WorkloadCost::default();

    for layer in &network.layers {
        let active = layer.active_count() as u64;
        let outputs = layer.output as u64;

        cost.macs = cost.macs.saturating_add(
            batch.saturating_mul(active),
        );
        cost.memory_reads = cost.memory_reads.saturating_add(
            batch.saturating_mul(active + layer.bias.len() as u64),
        );
        cost.memory_writes = cost.memory_writes.saturating_add(
            batch.saturating_mul(outputs),
        );
        cost.activation_ops = cost.activation_ops.saturating_add(
            batch.saturating_mul(outputs),
        );
    }

    cost
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
    estimate_energy_from_cost(
        network_workload_cost(network, batch_size),
        hardware,
    )
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
    fn workload_cost_matches_sparse_network_accounting() {
        let mut rng = rand::rng();
        let network =
            random_network(&[2, 3], Activation::Relu, &mut rng)
                .expect("network");

        let cost = network_workload_cost(&network, 2);
        assert_eq!(cost.macs, 12);
        assert_eq!(cost.memory_reads, 18);
        assert_eq!(cost.memory_writes, 6);
        assert_eq!(cost.activation_ops, 6);
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
