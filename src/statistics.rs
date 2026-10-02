use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
pub struct PairedTest {
    pub mean_difference: f32,
    pub upper_confidence_bound: f32,
    pub tolerance: f32,
    pub accepted: bool,
}

pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }

    let total = values.iter().map(|value| *value as f64).sum::<f64>();
    (total / values.len() as f64) as f32
}

pub fn paired_bootstrap_upper_bound(
    differences: &[f32],
    resamples: usize,
    seed: u64,
) -> f32 {
    if differences.is_empty() {
        return f32::INFINITY;
    }

    if resamples == 0 || differences.len() == 1 {
        return mean(differences);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let n = differences.len();
    let mut bootstrap_means = Vec::with_capacity(resamples);

    for _ in 0..resamples {
        let mut total = 0.0f64;

        for _ in 0..n {
            let index = rng.random_range(0..n);
            total += differences[index] as f64;
        }

        bootstrap_means.push((total / n as f64) as f32);
    }

    bootstrap_means.sort_by(|a, b| a.total_cmp(b));

    let percentile_index =
        ((bootstrap_means.len() as f64) * 0.95)
            .ceil()
            .max(1.0) as usize
            - 1;

    bootstrap_means[
        percentile_index.min(bootstrap_means.len() - 1)
    ]
}

pub fn paired_test(
    differences: &[f32],
    baseline_mean: f32,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    resamples: usize,
    seed: u64,
) -> PairedTest {
    let mean_difference = mean(differences);
    let upper_confidence_bound =
        paired_bootstrap_upper_bound(differences, resamples, seed);
    let tolerance = absolute_tolerance.max(
        baseline_mean.abs() * relative_tolerance,
    );

    PairedTest {
        mean_difference,
        upper_confidence_bound,
        tolerance,
        accepted: upper_confidence_bound <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_bootstrap_is_reproducible_and_conservative() {
        let differences = vec![-0.10, -0.05, -0.02, -0.08, -0.04];

        let first =
            paired_bootstrap_upper_bound(&differences, 500, 42);
        let second =
            paired_bootstrap_upper_bound(&differences, 500, 42);

        assert_eq!(first, second);
        assert!(first < 0.0);
    }

    #[test]
    fn paired_test_combines_absolute_and_relative_tolerance() {
        let result = paired_test(
            &[-0.01, -0.02, 0.00],
            1.0,
            0.005,
            0.01,
            200,
            7,
        );

        assert_eq!(result.tolerance, 0.01);
        assert!(result.accepted);
    }
}
