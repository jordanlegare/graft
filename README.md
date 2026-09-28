# Graft

Graft is a Rust prototype for discovering lower-compute dense neural-network
topologies from an existing model's weights, biases, tensor shapes, activation
metadata, and representative input/target samples.

## Search modes

Graft now has two complementary candidate-generation modes.

### Dataset separation

The search uses a reproducible three-way dataset split:

- training samples are used only for candidate fine-tuning
- validation samples drive behavior-aware mutation probes and accuracy acceptance
- holdout samples are evaluated only after the search, and never influence mutation selection or ranking

The split is controlled with `--validation-fraction`, `--holdout-fraction`,
and `--split-seed`. The default is 70% training, 15% validation, and 15%
holdout.

### Weight-guided grafting

Guided candidates start from the supplied trained network instead of random
initialization. The mutation engine uses the existing parameters plus representative
validation behavior to identify structurally useful and redundant neurons,
then applies parameter-preserving changes followed by fine-tuning:

- low-utility neuron pruning
- high-utility neuron splitting
- similar-neuron merging
- low-utility connection pruning
- sparse connection rewiring

Weight utility is combined with activation magnitude, activation variance,
activation sparsity, activation correlation, and validation-ablation loss.
For pruning, Graft temporarily removes a neuron or edge and measures the
change in validation MSE. For merging, it requires highly correlated
activation traces and checks the merged network's validation error before
accepting the candidate. Splits favor behaviorally active neurons with
substantial activation variance. Rewiring chooses a low-sensitivity source
edge and tests candidate inactive destinations against the validation set.

The expensive behavior probes are capped by configurable sample and candidate
limits so the search remains usable on larger MLPs.

Connections have an explicit active mask. This lets the search engine
represent sparse topology instead of treating every zero weight as a dense
connection.

### Random architecture search

A configurable fraction of candidates can still be generated from scratch over
random depth, width, and activation choices. This gives the optimizer a way to
explore architectures outside the local neighborhood of the supplied network.

## Build

~~~
cargo build --release
cargo test
~~~

## Generate the ONNX test model

~~~
cargo run --release --bin onnx-seed
~~~

This generates an 8 -> 16 -> 4 MLP:

~~~
seed/
  model.onnx
  manifest.json
  dataset.json
  W0.bin
  b0.bin
  W1.bin
  b1.bin
~~~

## Weight-guided search

~~~
cargo run --release --bin neuro-search --   --manifest seed/manifest.json   --dataset seed/dataset.json   --candidates 100   --epochs 25   --learning-rate 0.01   --accuracy-tolerance 0.01   --min-width 4   --max-width 64   --min-depth 1   --max-depth 4   --guided-fraction 0.75   --guided-mutations 3   --similarity-threshold 0.85   --activation-candidates relu,tanh   --validation-fraction 0.15   --holdout-fraction 0.15   --split-seed 42   --export-best-dir grafted   --output results.json
~~~

The default guided-fraction=0.75 means roughly three quarters of the
candidates start from the supplied trained parameters. Set it to 1.0 for
fully guided local topology search, or 0.0 for the original random-search
behavior.

## Exported graft

When --export-best-dir is supplied, the selected candidate produces:

~~~
grafted/
  manifest.json
  topology.json
  W0.bin
  b0.bin
  mask0.bin
  W1.bin
  b1.bin
  mask1.bin
  ...
~~~

maskN.bin contains one byte per stored weight: 1 means the connection is
active and 0 means it is pruned. This makes the discovered sparse topology
portable to a later sparse compiler/runtime stage.

## Optional hardware profile

~~~json
{
  "mac_energy_pj": 1.0,
  "memory_read_energy_pj": 2.0,
  "memory_write_energy_pj": 2.5,
  "activation_energy_pj": 0.2
}
~~~

Run with:

~~~
--hardware hardware.json
~~~

The model uses the number of active connections when estimating MAC and weight
read energy. It is still an analytical estimate; electrical measurements on
the target device are required for a physical energy claim.

Search result `mse` is validation MSE. Each result also records
`holdout_mse`; the holdout value is computed only after candidate generation
and does not affect candidate selection.

## Manifest

The manifest supplies tensor shapes, paths, activation metadata, and optional
sparse masks:

~~~json
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
~~~

Exported manifests add active_mask per layer.

Raw weight and bias files are little-endian f32. Weight matrices are stored
row-major as [output, input].

## Scope

This stage searches feed-forward dense MLPs. Convolutional, attention,
recurrent, residual, and arbitrary DAG graph transformations are not yet
implemented.

The intended next stage is hardware-aware compilation of the exported masks
and sparse weights, followed by real-device energy measurement.

CI formats the workspace before compile and test.
