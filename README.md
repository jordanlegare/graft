# Graft

Graft is a Rust prototype for discovering lower-compute dense neural-network
topologies from an existing model's weights, biases, tensor shapes, activation
metadata, and representative input/target samples.

## Current engine

The prototype:

1. Loads raw little-endian f32 weight and bias files.
2. Reconstructs the supplied baseline MLP from a JSON manifest.
3. Generates alternative dense topologies.
4. Searches hidden activation choices.
5. Fine-tunes each candidate on representative samples.
6. Computes validation MSE.
7. Estimates energy from MACs, memory traffic, and activation work.
8. Ranks candidates that satisfy the supplied MSE tolerance.

The energy value is an analytical estimate, not measured electrical energy.
Hardware benchmarking should replace the analytical profile before making a
physical energy-efficiency claim.

## Build

```text
cargo build --release
cargo test
```

## Generate the ONNX test model

```text
cargo run --release --bin onnx-seed
```

This generates an 8 -> 16 -> 4 MLP:

```text
seed/
  model.onnx
  manifest.json
  dataset.json
  W0.bin
  b0.bin
  W1.bin
  b1.bin
```

## Search

```text
cargo run --release --bin neuro-search -- \
  --manifest seed/manifest.json \
  --dataset seed/dataset.json \
  --candidates 100 \
  --epochs 25 \
  --learning-rate 0.01 \
  --accuracy-tolerance 0.01 \
  --min-width 4 \
  --max-width 64 \
  --min-depth 1 \
  --max-depth 4 \
  --activation-candidates relu,tanh \
  --output results.json
```

## Optional hardware profile

```json
{
  "mac_energy_pj": 1.0,
  "memory_read_energy_pj": 2.0,
  "memory_write_energy_pj": 2.5,
  "activation_energy_pj": 0.2
}
```

Run with:

```text
--hardware hardware.json
```

## Manifest

The manifest supplies tensor shapes, paths, and activation metadata:

```json
{
  "input_size": 8,
  "output_size": 4,
  "layers": [
    {
      "input": 8,
      "output": 16,
      "weights": "seed/W0.bin",
      "bias": "seed/b0.bin",
      "activation": "relu"
    },
    {
      "input": 16,
      "output": 4,
      "weights": "seed/W1.bin",
      "bias": "seed/b1.bin",
      "activation": "linear"
    }
  ]
}
```

Raw weight and bias files are little-endian f32. For the search engine, weight
matrices are stored row-major as [output, input].

## Scope

This first implementation searches feed-forward dense MLPs. It does not yet
infer convolutional, attention, recurrent, residual, or arbitrary DAG graphs.

The next graft stage should use the trained parameters themselves to drive
topology mutation: neuron pruning, neuron merging, neuron splitting, connection
rewiring, and parameter transplantation instead of random reinitialization.

CI verification for the repository build and seed/search smoke test is defined
in .github/workflows/ci.yml.
