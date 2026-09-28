use anyhow::Result;
use graft::{
    data::Dataset,
    graph::{
        GraphNetwork, GraphNode, GraphOp, TensorShape,
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs;

const SAMPLES: usize = 200;

fn main() -> Result<()> {
    fs::create_dir_all("seed")?;

    let mut rng = StdRng::seed_from_u64(1234);
    let input_shape = TensorShape::sequence(2, 4)?;

    let input = GraphNode {
        id: 0,
        inputs: Vec::new(),
        op: GraphOp::Input {
            shape: input_shape,
        },
        weights: Vec::new(),
        bias: Vec::new(),
    };

    let conv = GraphNode::with_random_parameters(
        1,
        vec![0],
        GraphOp::Conv1d {
            input_channels: 2,
            output_channels: 2,
            kernel: 1,
            stride: 1,
        },
        Some(input_shape),
        &mut rng,
    )?;

    let attention = GraphNode::with_random_parameters(
        2,
        vec![1],
        GraphOp::SelfAttention {
            channels: 2,
            heads: 1,
        },
        Some(input_shape),
        &mut rng,
    )?;

    let recurrent = GraphNode::with_random_parameters(
        3,
        vec![2],
        GraphOp::Recurrent {
            input_size: 2,
            hidden_size: 2,
        },
        Some(input_shape),
        &mut rng,
    )?;

    let residual = GraphNode::with_random_parameters(
        4,
        vec![2, 3],
        GraphOp::Add,
        None,
        &mut rng,
    )?;

    let activated = GraphNode::with_random_parameters(
        5,
        vec![4],
        GraphOp::Activation {
            activation: graft::model::Activation::Tanh,
        },
        Some(input_shape),
        &mut rng,
    )?;

    let output = GraphNode::with_random_parameters(
        6,
        vec![5],
        GraphOp::Dense {
            input: input_shape.size(),
            output: 4,
        },
        Some(input_shape),
        &mut rng,
    )?;

    let graph = GraphNetwork {
        nodes: vec![
            input,
            conv,
            attention,
            recurrent,
            residual,
            activated,
            output,
        ],
        output: 6,
    };

    graph.validate()?;

    let mut inputs = Vec::with_capacity(SAMPLES);
    let mut targets = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let input = (0..input_shape.size())
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<_>>();
        let target = graph.forward(&input)?;

        inputs.push(input);
        targets.push(target);
    }

    Dataset { inputs, targets }.save("seed/graph-dataset.json")?;
    fs::write(
        "seed/graph.json",
        serde_json::to_string_pretty(&graph)?,
    )?;

    println!("Generated seed/graph.json");
    println!("Generated seed/graph-dataset.json");
    Ok(())
}
