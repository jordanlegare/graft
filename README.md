# Graft

Graft is a Rust prototype for discovering **lower-compute neural-network topologies and graph structures** while preserving the behavior required by the workload.

The practical idea is simple:

> **Start with a trained model, remove or reorganize computation that the model does not need, fine-tune the result, validate it, and export the smaller structure for a later hardware/runtime implementation.**

For large language models (LLMs), this matters because electricity is ultimately spent on the computation, memory movement, networking, cooling, and other infrastructure required to serve the model. Graft focuses on the **model-architecture part** of that problem.

It can therefore be used as an optimization stage between a trained neural network and a future low-power inference implementation.

---

## What Graft can do

Graft currently has two complementary search paths.

### 1. Weight-guided sparse MLP grafting

For feed-forward networks, Graft can start from an existing trained network and search for a smaller topology instead of starting over from random initialization.

It can:

- prune low-utility neurons
- split high-utility neurons
- merge highly similar neurons
- prune low-utility connections
- rewire sparse connections
- preserve active/inactive connection masks
- fine-tune the resulting candidate with gradient descent
- compare candidates on a separate validation set
- evaluate the selected candidate on a never-seen holdout set
- export the sparse topology and parameters

The mutation engine combines weight information with observed behavior such as activation magnitude, variance, sparsity, correlation, and validation-ablation loss.

### 2. General neural-graph search

Graft also works with architectures that cannot be represented as a simple sequential MLP:

- **1-D convolution**
- **multi-head self-attention**
- **recurrent sequence nodes**
- **residual/additive branches**
- **arbitrary directed acyclic graphs (DAGs)**

The graph engine performs:

- shape inference
- cycle detection
- topological execution
- input rewiring
- operator replacement
- node insertion
- node removal
- residual insertion
- reverse-mode gradient training through the resulting DAG

The current graph trainer includes:

- dense backpropagation
- convolution gradients
- attention gradients through Q/K/V and softmax attention
- recurrent backpropagation through time (BPTT)
- residual/additive gradient propagation
- activation gradients

This means a mutated graph is not merely scored structurally: it can be **fine-tuned end-to-end before it is selected**.

---

# Why this matters for LLM electricity use

Modern AI data centers consume very large amounts of electricity. The International Energy Agency estimated global data-center electricity consumption at about **415 TWh in 2024**, and its 2025 base case projected about **945 TWh by 2030**. An updated IEA outlook published in 2026 puts the trajectory at roughly **485 TWh in 2025 and 950 TWh in 2030**, with AI-focused data-center electricity consumption growing much faster than overall data-center consumption.

Sources:

- [IEA — Energy and AI](https://www.iea.org/reports/energy-and-ai)
- [IEA — Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary)
- [U.S. DOE / LBNL — 2025 U.S. Data Center Energy Usage Update](https://www.energy.gov/documents/united-states-data-center-energy-usage-report-2025-update)

For an LLM, the electricity needed to answer a prompt is not determined by parameter count alone. It depends on the model graph, token count, batching, hardware utilization, memory traffic, communication, cooling, and the data-center power usage effectiveness (PUE).

That is why Graft reports **MAC count, parameter count, active connections, and graph structure** rather than claiming that a particular percentage of grid electricity has already been eliminated.

## The key energy mechanism

A dense model performs many operations whether or not every connection contributes equally to the final result.

Graft searches for architectures in which some of that computation can be removed or reorganized while maintaining the required behavior.

A simplified first-order model is:

~~~
baseline compute
        |
        v
trained model
        |
        |  Graft mutations
        v
smaller / sparser graph
        |
        |  fine-tune
        v
validated grafted model
        |
        v
lower operation + memory workload
~~~

At the architecture level, a useful proxy is:

~~~
compute reduction ≈ 1 - (grafted MACs / baseline MACs)
~~~

This is a **compute proxy, not a measured electricity reduction**.

Real electrical savings depend on whether the target hardware and runtime actually skip the removed work. A sparse graph that still executes dense kernels can have little or no wall-power benefit. Conversely, an implementation that maps the new topology efficiently can capture savings in arithmetic, memory movement, and sometimes cooling.

---

# Graft's energy accounting: before vs. after

The end-to-end energy question should be expressed at the data-center electrical input, not only at the neural-network layer.

Let:

- E_IT = electricity consumed by servers, accelerators, memory, storage, and networking attributable to the workload
- **PUE** = Power Usage Effectiveness
- E_DC = total data-center electrical input attributable to the workload

Then:

~~~
E_DC = E_IT × PUE
~~~

For example, a workload using 100 MWh of IT electricity at a PUE of 1.30 requires:

~~~
100 MWh × 1.30 = 130 MWh
~~~

of facility electricity.

## What Graft changes

Let:

- E_before = baseline model electricity
- E_after = electricity after the grafted architecture is deployed
- r = measured fraction of workload electricity eliminated

Then:

~~~
E_after  = E_before × (1 - r)
Savings  = E_before × r
~~~

The important distinction is that Graft can directly discover a lower-compute graph, but **r must ultimately be measured on the target inference stack**.

Until that hardware measurement exists, the honest planning model is to treat MAC reduction as a scenario input:

~~~
scenario electrical reduction ≈ scenario compute reduction
~~~

and label the result as an **engineering estimate**.

---

# What current LLM measurements tell us

Published measurements show that LLM inference energy varies enormously with model size, hardware, prompt length, utilization, and test-time reasoning.

A 2025 bottom-up analysis estimated a median of about **0.34 Wh per query** for frontier-scale models above 200 billion parameters under its H100-based, realistic-workload assumptions. The same study estimated about **4.32 Wh per query** when test-time scaling increased the typical token workload by roughly 15×.

In a separate 2025 benchmark of 30 state-of-the-art models, a short GPT-4o query was estimated at about **0.43 Wh**, while long-prompt/reasoning cases could be much higher.

The 2026 Stanford AI Index summarizes another benchmark for approximately 1,000 input + 1,000 output tokens: some lower-energy models were around **5–6 Wh per query**, while reported values for more compute-intensive models reached about **21.9 Wh for GPT-5 (high)** and **23 Wh for DeepSeek V3.2**.

These numbers come from different methodologies and should **not** be treated as an apples-to-apples league table. They are useful here as a scale reference showing how much inference energy can vary across deployed LLM workloads.

Sources:

- [Oviedo et al. — Energy Use of AI Inference](https://arxiv.org/abs/2509.20241)
- [Jegham et al. — How Hungry is AI?](https://arxiv.org/abs/2505.09598)
- [Stanford AI Index 2026 — Energy and Environmental Impact](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf)

---

# Quantifying a possible Graft effect

The following table is **not a claim that Graft has already achieved these reductions**. It shows what would happen if a deployed LLM workload measured a given reduction in attributable inference electricity after grafting.

For a simple scale example, assume a baseline of **0.34 Wh/query**.

| Measured deployment reduction | Energy after grafting | Savings per query |
|---:|---:|---:|
| 0% | 0.340 Wh | 0.000 Wh |
| 10% | 0.306 Wh | 0.034 Wh |
| 25% | 0.255 Wh | 0.085 Wh |
| 50% | 0.170 Wh | 0.170 Wh |
| 75% | 0.085 Wh | 0.255 Wh |

At **1 billion queries/day**, the corresponding annual electricity use would be approximately:

| Reduction | Before | After | Annual saving |
|---:|---:|---:|---:|
| 0% | 124.1 GWh | 124.1 GWh | 0 GWh |
| 10% | 124.1 GWh | 111.7 GWh | 12.4 GWh |
| 25% | 124.1 GWh | 93.1 GWh | 31.0 GWh |
| 50% | 124.1 GWh | 62.1 GWh | 62.1 GWh |
| 75% | 124.1 GWh | 31.0 GWh | 93.1 GWh |

The same percentages scale linearly with workload. For example, at the same query volume, a baseline of 5 Wh/query would consume about **1.825 TWh/year** before grafting; a measured 50% reduction would lower that to about **0.913 TWh/year**, saving about **0.913 TWh/year**.

These are arithmetic illustrations. They are not measurements of any particular commercial LLM deployment.

---

# From model reduction to total data-center input

For production planning, use the following sequence.

### Before grafting

~~~
LLM workload
  ×
baseline energy/query
  ×
queries/day
  ×
365
  =
annual IT or workload electricity
~~~

Then account for facility overhead:

~~~
annual data-center input
  =
annual IT electricity × PUE
~~~

### After grafting

Measure the same workload on the same target hardware:

~~~
grafted LLM workload
  ×
measured energy/query
  ×
queries/day
  ×
365
  =
annual post-graft IT electricity
~~~

Then:

~~~
annual data-center input after grafting
  =
annual post-graft IT electricity × PUE
~~~

Finally:

~~~
annual electricity saved
  =
before - after
~~~

This is the number that matters to a facility operator, utility planner, or AI infrastructure team.

---

# A worked data-center example

Consider a hypothetical inference fleet with:

- 1 billion queries/day
- 0.34 Wh/query baseline workload energy
- PUE = 1.30
- a production graft that is later measured to reduce total attributable workload electricity by 50%

Baseline workload:

~~~
0.34 Wh × 1,000,000,000 × 365
≈ 124.1 GWh/year
~~~

At PUE 1.30:

~~~
124.1 GWh × 1.30
≈ 161.3 GWh/year of facility electricity
~~~

After a measured 50% reduction:

~~~
62.1 GWh/year IT
62.1 × 1.30
≈ 80.7 GWh/year facility input
~~~

Illustrative facility saving:

~~~
161.3 - 80.7
≈ 80.7 GWh/year
~~~

Again, the **50% is an example measurement target, not a demonstrated Graft result**.

For a larger 5 Wh/query workload, the same 50% reduction at 1 billion queries/day would imply a facility-input reduction of roughly **1.186 TWh/year at PUE 1.30**, assuming the quoted 5 Wh figure represents IT-side energy.

---

# Why the reduction is not simply "parameters × electricity"

A smaller parameter count is helpful, but it is not sufficient to predict wall power.

A practical energy stack looks like:

~~~
Model topology
      ↓
MACs / FLOPs
      ↓
Memory reads + writes
      ↓
Accelerator utilization
      ↓
Interconnect / communication
      ↓
Cooling and facility overhead
      ↓
Total data-center electrical input
~~~

Graft directly operates on the first two layers and can reduce active connections and graph computation. The later layers require a compatible compiler, runtime, hardware mapping, and physical measurement.

This is why a realistic deployment evaluation should record at least:

- parameter count before/after
- active connection count before/after
- MAC count before/after
- tokens/second
- batch size
- accelerator type
- accelerator utilization
- memory bandwidth
- measured watts at the server or rack
- PUE
- end-to-end energy/query
- validation quality and latency

---

# What Graft can mean for an LLM pipeline

Graft is best understood as an **architecture optimization stage**, not as a complete LLM serving system.

A production workflow could look like:

~~~
1. Train or obtain a baseline model
              ↓
2. Collect representative prompts / targets
              ↓
3. Measure baseline accuracy, latency, and energy
              ↓
4. Import a supported graph representation
              ↓
5. Run Graft architecture search
              ↓
6. Fine-tune every viable candidate
              ↓
7. Select using validation quality + compute cost
              ↓
8. Evaluate exactly once on holdout data
              ↓
9. Export the grafted graph
              ↓
10. Compile/map it to sparse or specialized kernels
              ↓
11. Measure real hardware power
              ↓
12. Compare before vs. after at the data-center meter
~~~

The important engineering loop is therefore:

**discover → fine-tune → validate → compile → measure → repeat**

---

# What Graft does not yet claim

Graft should not currently be described as having already reduced the electricity consumption of production LLMs.

The repository does **not** currently provide:

- a direct GPT-4/GPT-5/Claude/Gemini/DeepSeek production integration
- tokenizer or KV-cache optimization
- distributed tensor/pipeline parallel scheduling
- GPU-specific kernel generation
- accelerator-specific sparse execution guarantees
- quantization as a complete LLM deployment path
- live rack-level or facility-level electrical measurement
- a demonstrated percentage reduction in the total power bill of a commercial data center

Those are separate engineering layers.

What Graft already provides is the architecture-search foundation needed to test one important hypothesis:

> **Can a trained neural computation graph be made materially smaller or sparser while retaining the required behavior?**

That hypothesis is now testable in code.

---

# Dataset separation and experimental integrity

Graft uses a reproducible three-way split:

- **training samples** are used for candidate fine-tuning
- **validation samples** drive mutation probes, acceptance, and candidate ranking
- **holdout samples** are evaluated only after the search

The default split is 70% training, 15% validation, and 15% holdout.

The split is controlled with:

~~~
--validation-fraction
--holdout-fraction
--split-seed
~~~

This separation matters for energy research because a topology that saves compute but quietly loses task quality is not a valid optimization.

---

# General graph search

Generate the multi-operator graph seed and run the graph search with:

~~~bash
cargo run --release --bin graph-seed

cargo run --release --bin graph-search -- \
  --graph seed/graph.json \
  --dataset seed/graph-dataset.json \
  --candidates 32 \
  --mutations 4 \
  --epochs 5 \
  --learning-rate 0.01 \
  --accuracy-tolerance 0.01 \
  --validation-fraction 0.15 \
  --holdout-fraction 0.15 \
  --split-seed 42 \
  --export-best seed/grafted-graph.json
~~~

The seed graph intentionally contains convolution, attention, recurrence, residual, and dense nodes so that the general graph path exercises all of them.

The graph JSON file is the serialized GraphNetwork intermediate representation (IR), so graph architectures can be saved, mutated, evaluated, and exported without flattening everything into dense MLP layers.

---

# Weight-guided MLP search

~~~bash
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
  --guided-fraction 0.75 \
  --guided-mutations 3 \
  --similarity-threshold 0.85 \
  --activation-candidates relu,tanh \
  --validation-fraction 0.15 \
  --holdout-fraction 0.15 \
  --split-seed 42 \
  --export-best-dir grafted \
  --output results.json
~~~

The guided fraction of 0.75 means roughly three quarters of candidates start from the supplied trained parameters. Set it to 1.0 for fully guided local topology search, or 0.0 for the original random-search behavior.

---

# Exported graft

When the export-best-dir option is supplied, the selected MLP candidate produces:

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

maskN.bin contains one byte per stored weight:

- **1** = active connection
- **0** = pruned connection

This makes the discovered sparse topology portable to a later sparse compiler/runtime stage.

---

# Optional hardware energy model

For the MLP path, an optional hardware profile can provide analytical energy coefficients:

~~~json
{
  "mac_energy_pj": 1.0,
  "memory_read_energy_pj": 2.0,
  "memory_write_energy_pj": 2.5,
  "activation_energy_pj": 0.2
}
~~~

Run with:

~~~bash
--hardware hardware.json
~~~

The model uses active connections when estimating MAC and weight-read energy.

This is an **analytical estimate**, not a physical measurement. The final production claim must come from measurements on the actual accelerator, runtime, server, and data-center facility.

---

# Interpreting search results

Search result mse is validation MSE.

Each result also records holdout_mse; the holdout value is computed only after candidate generation and does not affect candidate selection.

A useful energy-oriented experiment should compare at least:

~~~
quality:
  validation MSE
  holdout MSE

complexity:
  parameter count
  active connections
  MAC count

deployment:
  latency
  throughput
  energy/query
  watts
  PUE
  total facility electricity
~~~

A candidate is only an energy win in production when its quality and service requirements remain acceptable **and** the target implementation actually consumes less electricity.

---

# Build

~~~bash
cargo build --release
cargo test
~~~

## Generate the ONNX test model

~~~bash
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

---

# Manifest

The manifest supplies tensor shapes, paths, activation metadata, and optional sparse masks:

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

Raw weight and bias files are little-endian f32. Weight matrices are stored row-major as [output, input].

---

# Current status and next hardware stage

Graft now supports two structural search paths:

1. a behavior-aware sparse MLP path
2. a general graph path covering convolution, attention, recurrence, residual branches, and arbitrary DAG rewrites

The graph path uses an explicit graph IR, reverse-mode training, and validation/holdout separation.

The **next major step for proving electricity savings** is target-specific compilation and measurement:

1. map the exported graph to real accelerator kernels
2. preserve the discovered sparsity/structure during execution
3. benchmark identical workloads before and after grafting
4. measure server/rack power
5. include PUE to obtain total data-center electrical input
6. report energy per token/query and annual fleet electricity

That is the point at which a model-topology reduction becomes a defensible data-center energy result.

---

## Energy reference points

The README's energy examples are anchored to the following public sources and should be rechecked as new measurements become available:

- **International Energy Agency (2025/2026):** global data-center electricity demand and AI-driven growth
  https://www.iea.org/reports/energy-and-ai
  https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary

- **U.S. Department of Energy / Lawrence Berkeley National Laboratory:** U.S. data-center electricity estimates and projections
  https://www.energy.gov/documents/united-states-data-center-energy-usage-report-2025-update

- **Oviedo et al. (2025):** bottom-up LLM inference energy estimates, including a 0.34 Wh median for frontier-scale models in the study's baseline scenario
  https://arxiv.org/abs/2509.20241

- **Jegham et al. (2025):** infrastructure-aware LLM inference energy benchmarking
  https://arxiv.org/abs/2505.09598

- **Stanford AI Index 2026:** current compiled discussion of AI energy use and model-level inference benchmarks
  https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf

All per-query figures in this README are **reference measurements or scenario calculations**, not measurements produced by Graft itself.


---

# Search validity model

The search pipeline now separates **training**, **probe**, **validation**, and **holdout** data on the MLP path.

- **Training** data is used only for candidate fine-tuning.
- **Probe** data drives behavior profiling, ablation tests, Taylor saliency, and guided mutation decisions.
- **Validation** data is reserved for candidate acceptance and Pareto analysis.
- **Holdout** data is evaluated only for the final selected candidate.

Candidate acceptance uses paired per-sample MSE differences rather than comparing two aggregate MSE values:

[
d_i = operatorname{MSE}_i(mathrm{candidate}) -
      operatorname{MSE}_i(mathrm{baseline})
]

A deterministic bootstrap estimates the 95% upper confidence bound of the mean paired difference. A candidate is accepted when that upper bound is no larger than:

[
max(
epsilon_{mathrm{absolute}},
epsilon_{mathrm{relative}},
mathrm{MSE}_{mathrm{baseline}}
)
]

This keeps the quality criterion scale-aware while accounting for paired sampling uncertainty.

## Guided mutation mathematics

Neuron and connection pruning use a first-order Taylor saliency proxy:

[
S_j approx |	heta_j 
abla_{	heta_j} L|
]

Saliency is aggregated over the parameter group affected by a mutation. This is combined with activation statistics and, where applicable, direct ablation measurements.

Neuron merging no longer simply averages incoming parameters and sums outgoing weights. The incoming representation is initialized from the two neurons, then each outgoing coefficient is locally refit by least squares on the probe activation trace:

[
c^* =
rac{sum_i h_i,y_i}
     {sum_i h_i^2}
]

where (h_i) is the merged-neuron activation and (y_i) is the contribution previously produced by the two outgoing edges.

## Cost vector

Both search engines expose the same analytical workload dimensions:

- MACs
- memory reads
- memory writes
- activation / softmax operations

The hardware profile maps those dimensions to an analytical energy estimate. This is still a deployment proxy: real savings depend on sparse-kernel support, accelerator utilization, memory behavior, and physical power measurement.

## Training budget

Search can optionally cap candidate fine-tuning by a common training-MAC budget with:

```text
--training-macs-budget <MACs>
```

With this option, candidates receive enough epochs to consume approximately the requested optimizer-work budget, with at least one epoch.

Without it, the configured epoch count is used.

