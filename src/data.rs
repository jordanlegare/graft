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
}
