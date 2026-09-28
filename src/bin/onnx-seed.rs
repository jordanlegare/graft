use anyhow::Result;
use graft::{
    data::Dataset,
    onnx::{
        float_tensor,
        value_info,
        write_model,
        GraphProto,
        ModelProto,
        NodeProto,
        OperatorSetIdProto,
    },
};
use rand::Rng;
use serde::Serialize;
use std::{
    fs,
    fs::File,
    io::Write,
};

const INPUT: usize = 8;
const HIDDEN: usize = 16;
const OUTPUT: usize = 4;
const SAMPLES: usize = 500;

#[derive(Debug, Serialize)]
struct Manifest {
    input_size: usize,
    output_size: usize,
    layers: Vec<LayerManifest>,
}

#[derive(Debug, Serialize)]
struct LayerManifest {
    input: usize,
    output: usize,
    weights: String,
    bias: String,
    activation: String,
}

fn write_f32(path: &str, values: &[f32]) -> Result<()> {
    let mut file = File::create(path)?;

    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }

    Ok(())
}

fn main() -> Result<()> {
    fs::create_dir_all("seed")?;

    let mut rng = rand::rng();

    let w0: Vec<f32> = (0..INPUT * HIDDEN)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();

    let b0 = vec![0.0; HIDDEN];

    let w1: Vec<f32> = (0..HIDDEN * OUTPUT)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();

    let b1 = vec![0.0; OUTPUT];

    write_f32("seed/W0.bin", &w0)?;
    write_f32("seed/b0.bin", &b0)?;
    write_f32("seed/W1.bin", &w1)?;
    write_f32("seed/b1.bin", &b1)?;

    let onnx_w0: Vec<f32> = (0..INPUT)
        .flat_map(|i| (0..HIDDEN).map(move |h| w0[h * INPUT + i]))
        .collect();

    let onnx_w1: Vec<f32> = (0..HIDDEN)
        .flat_map(|h| (0..OUTPUT).map(move |o| w1[o * HIDDEN + h]))
        .collect();

    let graph = GraphProto {
        node: vec![
            NodeProto {
                input: vec!["X".into(), "W0".into()],
                output: vec!["MM0".into()],
                name: Some("MatMul0".into()),
                op_type: Some("MatMul".into()),
            },
            NodeProto {
                input: vec!["MM0".into(), "b0".into()],
                output: vec!["A0".into()],
                name: Some("Add0".into()),
                op_type: Some("Add".into()),
            },
            NodeProto {
                input: vec!["A0".into()],
                output: vec!["H0".into()],
                name: Some("Relu0".into()),
                op_type: Some("Relu".into()),
            },
            NodeProto {
                input: vec!["H0".into(), "W1".into()],
                output: vec!["MM1".into()],
                name: Some("MatMul1".into()),
                op_type: Some("MatMul".into()),
            },
            NodeProto {
                input: vec!["MM1".into(), "b1".into()],
                output: vec!["Y".into()],
                name: Some("Add1".into()),
                op_type: Some("Add".into()),
            },
        ],
        name: Some("GraftSeed".into()),
        initializer: vec![
            float_tensor("W0", &[INPUT, HIDDEN], &onnx_w0),
            float_tensor("b0", &[HIDDEN], &b0),
            float_tensor("W1", &[HIDDEN, OUTPUT], &onnx_w1),
            float_tensor("b1", &[OUTPUT], &b1),
        ],
        input: vec![value_info("X", &[1, INPUT])],
        output: vec![value_info("Y", &[1, OUTPUT])],
    };

    let model = ModelProto {
        ir_version: Some(9),
        producer_name: Some("graft".into()),
        producer_version: Some(env!("CARGO_PKG_VERSION").into()),
        domain: None,
        graph: Some(graph),
        opset_import: vec![
            OperatorSetIdProto {
                domain: Some("".into()),
                version: Some(18),
            },
        ],
    };

    write_model("seed/model.onnx", model)?;

    let mut inputs = Vec::with_capacity(SAMPLES);
    let mut targets = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let x: Vec<f32> = (0..INPUT)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let mut hidden = vec![0.0; HIDDEN];

        for h in 0..HIDDEN {
            let mut sum = b0[h];

            for i in 0..INPUT {
                sum += w0[h * INPUT + i] * x[i];
            }

            hidden[h] = sum.max(0.0);
        }

        let mut y = vec![0.0; OUTPUT];

        for o in 0..OUTPUT {
            let mut sum = b1[o];

            for h in 0..HIDDEN {
                sum += w1[o * HIDDEN + h] * hidden[h];
            }

            y[o] = sum;
        }

        inputs.push(x);
        targets.push(y);
    }

    Dataset { inputs, targets }.save("seed/dataset.json")?;

    let manifest = Manifest {
        input_size: INPUT,
        output_size: OUTPUT,
        layers: vec![
            LayerManifest {
                input: INPUT,
                output: HIDDEN,
                weights: "seed/W0.bin".into(),
                bias: "seed/b0.bin".into(),
                activation: "relu".into(),
            },
            LayerManifest {
                input: HIDDEN,
                output: OUTPUT,
                weights: "seed/W1.bin".into(),
                bias: "seed/b1.bin".into(),
                activation: "linear".into(),
            },
        ],
    };

    fs::write(
        "seed/manifest.json",
        serde_json::to_string_pretty(&manifest)?,
    )?;

    println!("Generated seed/model.onnx");
    println!("Generated seed/manifest.json");
    println!("Generated seed/dataset.json");
    println!("Generated seed/W0.bin and seed/b0.bin");
    println!("Generated seed/W1.bin and seed/b1.bin");

    Ok(())
}
