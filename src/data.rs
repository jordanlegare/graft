use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub inputs: Vec<Vec<f32>>,
    pub targets: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.inputs.is_empty(), "dataset contains no samples");
        anyhow::ensure!(
            self.inputs.len() == self.targets.len(),
            "input/target sample counts differ"
        );

        let input_width = self.inputs[0].len();
        let target_width = self.targets[0].len();

        anyhow::ensure!(input_width > 0, "input width must be > 0");
        anyhow::ensure!(target_width > 0, "target width must be > 0");

        anyhow::ensure!(
            self.inputs.iter().all(|x| x.len() == input_width),
            "input rows have inconsistent widths"
        );

        anyhow::ensure!(
            self.targets.iter().all(|x| x.len() == target_width),
            "target rows have inconsistent widths"
        );

        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let dataset: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.validate()?;
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn split_three_way(
        &self,
        validation_fraction: f32,
        holdout_fraction: f32,
        seed: u64,
    ) -> anyhow::Result<(Self, Self, Self)> {
        self.validate()?;

        anyhow::ensure!(
            validation_fraction > 0.0 && validation_fraction < 1.0,
            "validation_fraction must be between 0 and 1"
        );
        anyhow::ensure!(
            holdout_fraction > 0.0 && holdout_fraction < 1.0,
            "holdout_fraction must be between 0 and 1"
        );
        anyhow::ensure!(
            validation_fraction + holdout_fraction < 1.0,
            "validation_fraction + holdout_fraction must be less than 1"
        );
        anyhow::ensure!(
            self.inputs.len() >= 3,
            "dataset must contain at least 3 samples for a three-way split"
        );

        let sample_count = self.inputs.len();
        let validation_count =
            ((sample_count as f64) * validation_fraction as f64)
                .round()
                .max(1.0) as usize;
        let holdout_count =
            ((sample_count as f64) * holdout_fraction as f64)
                .round()
                .max(1.0) as usize;

        anyhow::ensure!(
            validation_count + holdout_count < sample_count,
            "split fractions leave no training samples"
        );

        let mut indices = (0..sample_count).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let validation_start = sample_count - validation_count - holdout_count;
        let holdout_start = sample_count - holdout_count;

        let train = self.subset(&indices[..validation_start]);
        let validation = self.subset(
            &indices[validation_start..holdout_start],
        );
        let holdout = self.subset(&indices[holdout_start..]);

        Ok((train, validation, holdout))
    }

    pub fn split_four_way(
        &self,
        probe_fraction: f32,
        validation_fraction: f32,
        holdout_fraction: f32,
        seed: u64,
    ) -> anyhow::Result<(Self, Self, Self, Self)> {
        self.validate()?;

        for (name, fraction) in [
            ("probe_fraction", probe_fraction),
            ("validation_fraction", validation_fraction),
            ("holdout_fraction", holdout_fraction),
        ] {
            anyhow::ensure!(
                fraction > 0.0 && fraction < 1.0,
                "{name} must be between 0 and 1"
            );
        }

        anyhow::ensure!(
            probe_fraction + validation_fraction + holdout_fraction < 1.0,
            "probe_fraction + validation_fraction + holdout_fraction must be less than 1"
        );
        anyhow::ensure!(
            self.inputs.len() >= 4,
            "dataset must contain at least 4 samples for a four-way split"
        );

        let sample_count = self.inputs.len();
        let probe_count = ((sample_count as f64) * probe_fraction as f64)
            .round()
            .max(1.0) as usize;
        let validation_count =
            ((sample_count as f64) * validation_fraction as f64)
                .round()
                .max(1.0) as usize;
        let holdout_count =
            ((sample_count as f64) * holdout_fraction as f64)
                .round()
                .max(1.0) as usize;

        anyhow::ensure!(
            probe_count + validation_count + holdout_count < sample_count,
            "split fractions leave no training samples"
        );

        let mut indices = (0..sample_count).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let train_end =
            sample_count - probe_count - validation_count - holdout_count;
        let probe_end = train_end + probe_count;
        let validation_end = probe_end + validation_count;

        Ok((
            self.subset(&indices[..train_end]),
            self.subset(&indices[train_end..probe_end]),
            self.subset(&indices[probe_end..validation_end]),
            self.subset(&indices[validation_end..]),
        ))
    }

    fn subset(&self, indices: &[usize]) -> Self {
        Self {
            inputs: indices
                .iter()
                .map(|index| self.inputs[*index].clone())
                .collect(),
            targets: indices
                .iter()
                .map(|index| self.targets[*index].clone())
                .collect(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn four_way_split_is_reproducible_and_disjoint() {
        let dataset = Dataset {
            inputs: (0..40)
                .map(|value| vec![value as f32])
                .collect(),
            targets: (0..40)
                .map(|value| vec![value as f32])
                .collect(),
        };

        let first =
            dataset.split_four_way(0.1, 0.15, 0.15, 42).expect("split");
        let second =
            dataset.split_four_way(0.1, 0.15, 0.15, 42).expect("split");

        assert_eq!(first.0.inputs, second.0.inputs);
        assert_eq!(first.1.inputs, second.1.inputs);
        assert_eq!(first.2.inputs, second.2.inputs);
        assert_eq!(first.3.inputs, second.3.inputs);

        assert_eq!(first.0.inputs.len(), 24);
        assert_eq!(first.1.inputs.len(), 4);
        assert_eq!(first.2.inputs.len(), 6);
        assert_eq!(first.3.inputs.len(), 6);

        let values = first
            .0
            .inputs
            .iter()
            .chain(&first.1.inputs)
            .chain(&first.2.inputs)
            .chain(&first.3.inputs)
            .map(|row| row[0] as usize)
            .collect::<Vec<_>>();

        let unique = values.iter().copied().collect::<HashSet<_>>();
        assert_eq!(values.len(), 40);
        assert_eq!(unique.len(), 40);
    }

    #[test]
    fn three_way_split_is_reproducible_and_disjoint() {
        let dataset = Dataset {
            inputs: (0..20)
                .map(|value| vec![value as f32])
                .collect(),
            targets: (0..20)
                .map(|value| vec![value as f32])
                .collect(),
        };

        let first =
            dataset.split_three_way(0.2, 0.2, 42).expect("split");
        let second =
            dataset.split_three_way(0.2, 0.2, 42).expect("split");

        assert_eq!(first.0.inputs, second.0.inputs);
        assert_eq!(first.1.inputs, second.1.inputs);
        assert_eq!(first.2.inputs, second.2.inputs);

        assert_eq!(first.0.inputs.len(), 12);
        assert_eq!(first.1.inputs.len(), 4);
        assert_eq!(first.2.inputs.len(), 4);

        let values = first
            .0
            .inputs
            .iter()
            .chain(&first.1.inputs)
            .chain(&first.2.inputs)
            .map(|row| row[0] as usize)
            .collect::<Vec<_>>();

        let unique = values.iter().copied().collect::<HashSet<_>>();
        assert_eq!(values.len(), 20);
        assert_eq!(unique.len(), 20);
    }
}
