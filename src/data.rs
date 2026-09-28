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
