use crate::model::Activation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorShape {
    pub channels: usize,
    pub length: usize,
}

impl TensorShape {
    pub fn new(channels: usize, length: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(channels > 0, "tensor channels must be > 0");
        anyhow::ensure!(length > 0, "tensor length must be > 0");
        Ok(Self { channels, length })
    }

    pub fn vector(size: usize) -> anyhow::Result<Self> {
        Self::new(size, 1)
    }

    pub fn sequence(channels: usize, length: usize) -> anyhow::Result<Self> {
        Self::new(channels, length)
    }

    pub fn size(self) -> usize {
        self.channels * self.length
    }

    pub fn is_vector(self) -> bool {
        self.length == 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphOp {
    Input { shape: TensorShape },
    Dense { input: usize, output: usize },
    Conv1d {
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    },
    SelfAttention {
        channels: usize,
        heads: usize,
    },
    Recurrent {
        input_size: usize,
        hidden_size: usize,
    },
    Add,
    Activation {
        activation: Activation,
    },
}

impl GraphOp {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Input { .. } => "input",
            Self::Dense { .. } => "dense",
            Self::Conv1d { .. } => "conv1d",
            Self::SelfAttention { .. } => "self-attention",
            Self::Recurrent { .. } => "recurrent",
            Self::Add => "add",
            Self::Activation { .. } => "activation",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: usize,
    pub inputs: Vec<usize>,
    pub op: GraphOp,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl GraphNode {
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    pub fn with_random_parameters<R: Rng>(
        id: usize,
        inputs: Vec<usize>,
        op: GraphOp,
        input_shape: Option<TensorShape>,
        rng: &mut R,
    ) -> anyhow::Result<Self> {
        let (weight_count, bias_count, scale) =
            parameter_layout(&op, input_shape)?;

        let weights = (0..weight_count)
            .map(|_| rng.random_range(-scale..scale))
            .collect::<Vec<_>>();

        Ok(Self {
            id,
            inputs,
            op,
            weights,
            bias: vec![0.0; bias_count],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNetwork {
    pub nodes: Vec<GraphNode>,
    pub output: usize,
}

impl GraphNetwork {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.nodes.is_empty(), "graph contains no nodes");

        let mut ids = self
            .nodes
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        anyhow::ensure!(
            ids.len() == self.nodes.len(),
            "graph node ids must be unique"
        );

        anyhow::ensure!(
            self.nodes.iter().any(|node| {
                matches!(&node.op, GraphOp::Input { .. })
            }),
            "graph has no input node"
        );

        let _ = self.topological_order()?;

        let shapes = self.infer_shapes()?;
        anyhow::ensure!(
            shapes.contains_key(&self.output),
            "graph output node {} is missing",
            self.output
        );

        anyhow::ensure!(
            shapes[&self.output].is_vector(),
            "graph output must be a flat vector"
        );

        Ok(())
    }

    pub fn topological_order(&self) -> anyhow::Result<Vec<usize>> {
        let by_id = self.node_map();
        let mut indegree = by_id
            .keys()
            .map(|id| (*id, 0usize))
            .collect::<std::collections::HashMap<_, _>>();
        let mut consumers =
            std::collections::HashMap::<usize, Vec<usize>>::new();

        for node in &self.nodes {
            for input in &node.inputs {
                anyhow::ensure!(
                    by_id.contains_key(input),
                    "node {} references missing input {}",
                    node.id,
                    input
                );

                *indegree
                    .get_mut(&node.id)
                    .expect("node id from graph") += 1;
                consumers.entry(*input).or_default().push(node.id);
            }
        }

        let mut ready = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect::<Vec<_>>();
        ready.sort_unstable();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = ready.pop() {
            order.push(id);

            if let Some(list) = consumers.get(&id) {
                for consumer in list {
                    let degree = indegree
                        .get_mut(consumer)
                        .expect("consumer from graph");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push(*consumer);
                    }
                }
                ready.sort_unstable_by(|a, b| b.cmp(a));
            }
        }

        anyhow::ensure!(
            order.len() == self.nodes.len(),
            "graph contains a cycle"
        );

        Ok(order)
    }

    pub fn infer_shapes(
        &self,
    ) -> anyhow::Result<std::collections::HashMap<usize, TensorShape>> {
        let by_id = self.node_map();
        let order = self.topological_order()?;
        let mut shapes = std::collections::HashMap::new();

        for id in order {
            let node = by_id.get(&id).expect("node from order");
            let input_shapes = node
                .inputs
                .iter()
                .map(|input| {
                    shapes
                        .get(input)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "node {} input {} has no inferred shape",
                                node.id,
                                input
                            )
                        })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let shape = infer_node_shape(&node.op, &input_shapes)?;
            validate_parameters(node, shape, &input_shapes)?;
            shapes.insert(node.id, shape);
        }

        Ok(shapes)
    }

    pub fn input_shape(&self) -> anyhow::Result<TensorShape> {
        let inputs = self
            .nodes
            .iter()
            .filter_map(|node| {
                if let GraphOp::Input { shape } = &node.op {
                    Some((node.id, *shape))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        anyhow::ensure!(
            inputs.len() == 1,
            "graph must contain exactly one input node"
        );

        Ok(inputs[0].1)
    }

    pub fn output_shape(&self) -> anyhow::Result<TensorShape> {
        Ok(self.infer_shapes()?[&self.output])
    }

    pub fn parameter_count(&self) -> usize {
        self.nodes
            .iter()
            .map(GraphNode::parameter_count)
            .sum()
    }

    pub fn mac_count(&self) -> u64 {
        let shapes = match self.infer_shapes() {
            Ok(shapes) => shapes,
            Err(_) => return 0,
        };
        let by_id = self.node_map();

        self.nodes
            .iter()
            .map(|node| match &node.op {
                GraphOp::Dense { .. } => node.weights.len() as u64,
                GraphOp::Conv1d {
                    input_channels,
                    output_channels,
                    kernel,
                    stride,
                } => {
                    let Some(input) = node.inputs.first() else {
                        return 0;
                    };
                    let Some(shape) = shapes.get(input) else {
                        return 0;
                    };
                    let output_length =
                        (shape.length.saturating_sub(*kernel) + *stride)
                            .div_ceil(*stride);
                    (output_length
                        * output_channels
                        * input_channels
                        * kernel) as u64
                }
                GraphOp::SelfAttention { channels, .. } => {
                    let Some(input) = node.inputs.first() else {
                        return 0;
                    };
                    let Some(shape) = shapes.get(input) else {
                        return 0;
                    };
                    let sequence = shape.length as u64;
                    let c = *channels as u64;
                    4 * sequence * c * c
                        + sequence * sequence * c
                        + sequence * sequence * c
                }
                GraphOp::Recurrent {
                    input_size,
                    hidden_size,
                } => {
                    let Some(input) = node.inputs.first() else {
                        return 0;
                    };
                    let Some(shape) = shapes.get(input) else {
                        return 0;
                    };
                    (shape.length
                        * hidden_size
                        * (input_size + hidden_size)) as u64
                }
                GraphOp::Add => 0,
                GraphOp::Activation { .. } => by_id
                    .get(&node.id)
                    .and_then(|_| shapes.get(&node.id))
                    .map_or(0, |shape| shape.size() as u64),
                GraphOp::Input { .. } => 0,
            })
            .sum()
    }

    pub fn forward(&self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.validate()?;
        let expected = self.input_shape()?;
        anyhow::ensure!(
            input.len() == expected.size(),
            "input size {} != expected {}",
            input.len(),
            expected.size()
        );

        let by_id = self.node_map();
        let order = self.topological_order()?;
        let mut values =
            std::collections::HashMap::<usize, Vec<f32>>::new();
        let shapes = self.infer_shapes()?;

        for id in order {
            let node = by_id.get(&id).expect("node from order");
            let node_shape = shapes[&id];

            let output = match &node.op {
                GraphOp::Input { .. } => input.to_vec(),
                GraphOp::Dense { input: width, output } => {
                    let x = single_input(&values, node)?.to_vec();
                    let mut y = vec![0.0; *output];

                    for (o, y_value) in
                        y.iter_mut().enumerate().take(*output)
                    {
                        let mut sum = node.bias[o];

                        for (i, x_value) in
                            x.iter().enumerate().take(*width)
                        {
                            sum +=
                                node.weights[o * *width + i] * *x_value;
                        }

                        *y_value = sum;
                    }

                    y
                }
                GraphOp::Conv1d {
                    input_channels,
                    output_channels,
                    kernel,
                    stride,
                } => {
                    let x = single_input(&values, node)?.to_vec();
                    conv1d(
                        &x,
                        *input_channels,
                        *output_channels,
                        *kernel,
                        *stride,
                        &node.weights,
                        &node.bias,
                    )?
                }
                GraphOp::SelfAttention { channels, heads } => {
                    let x = single_input(&values, node)?.to_vec();
                    self_attention(
                        &x,
                        *channels,
                        *heads,
                        &node.weights,
                        &node.bias,
                    )?
                }
                GraphOp::Recurrent {
                    input_size,
                    hidden_size,
                } => {
                    let x = single_input(&values, node)?.to_vec();
                    recurrent(
                        &x,
                        *input_size,
                        *hidden_size,
                        &node.weights,
                        &node.bias,
                    )?
                }
                GraphOp::Add => {
                    anyhow::ensure!(
                        !node.inputs.is_empty(),
                        "add node {} needs inputs",
                        node.id
                    );
                    let mut y =
                        vec![0.0; node_shape.size()];

                    for source in &node.inputs {
                        let value =
                            values.get(source).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "missing value for node {} input {}",
                                    node.id,
                                    source
                                )
                            })?;

                        anyhow::ensure!(
                            value.len() == y.len(),
                            "add node {} input shape mismatch",
                            node.id
                        );

                        for (dst, src) in
                            y.iter_mut().zip(value.iter())
                        {
                            *dst += *src;
                        }
                    }

                    y
                }
                GraphOp::Activation { activation } => {
                    let x = single_input(&values, node)?.to_vec();
                    x.iter()
                        .map(|value| activation.apply(*value))
                        .collect()
                }
            };

            values.insert(id, output);
        }

        values
            .remove(&self.output)
            .ok_or_else(|| anyhow::anyhow!("graph produced no output"))
    }


    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        epochs: usize,
        learning_rate: f32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            inputs.len() == targets.len(),
            "training input/target counts differ"
        );
        anyhow::ensure!(!inputs.is_empty(), "training dataset is empty");
        anyhow::ensure!(epochs > 0, "training epochs must be > 0");
        anyhow::ensure!(
            learning_rate > 0.0 && learning_rate.is_finite(),
            "learning_rate must be finite and > 0"
        );

        self.validate()?;

        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets) {
                self.train_sample(input, target, learning_rate)?;
            }
        }

        Ok(())
    }

    fn train_sample(
        &mut self,
        input: &[f32],
        target: &[f32],
        learning_rate: f32,
    ) -> anyhow::Result<()> {
        let (order, values, caches) =
            self.forward_training(input)?;
        let output = values
            .get(&self.output)
            .ok_or_else(|| anyhow::anyhow!("training graph produced no output"))?;

        anyhow::ensure!(
            output.len() == target.len(),
            "graph output width {} != target width {}",
            output.len(),
            target.len()
        );

        let mut gradients = values
            .iter()
            .map(|(id, value)| (*id, vec![0.0; value.len()]))
            .collect::<std::collections::HashMap<_, _>>();

        let output_gradient = gradients
            .get_mut(&self.output)
            .expect("output gradient initialized");

        for ((gradient, prediction), expected) in
            output_gradient.iter_mut().zip(output).zip(target)
        {
            *gradient = *prediction - *expected;
        }

        let index_by_id = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect::<std::collections::HashMap<_, _>>();

        for id in order.into_iter().rev() {
            let gradient_output = gradients
                .remove(&id)
                .expect("gradient initialized for node");
            let node_index =
                *index_by_id.get(&id).expect("node id indexed");

            let node = &self.nodes[node_index];
            let cache =
                caches.get(&id).expect("cache initialized for node");

            let input_gradients =
                backward_node(node, cache, &gradient_output)?;

            for (input_id, gradient_input) in input_gradients {
                let entry = gradients
                    .get_mut(&input_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "missing gradient buffer for node {}",
                            input_id
                        )
                    })?;

                anyhow::ensure!(
                    entry.len() == gradient_input.len(),
                    "gradient shape mismatch for node {}",
                    input_id
                );

                for (dst, src) in
                    entry.iter_mut().zip(gradient_input.iter())
                {
                    *dst += *src;
                }
            }

            let parameter_gradient =
                parameter_gradients(node, cache, &gradient_output)?;

            let node_mut =
                &mut self.nodes[node_index];

            anyhow::ensure!(
                node_mut.weights.len()
                    == parameter_gradient.0.len(),
                "weight gradient shape mismatch for node {}",
                id
            );
            anyhow::ensure!(
                node_mut.bias.len()
                    == parameter_gradient.1.len(),
                "bias gradient shape mismatch for node {}",
                id
            );

            for (weight, gradient) in
                node_mut.weights.iter_mut().zip(parameter_gradient.0)
            {
                *weight -= learning_rate * gradient;
            }

            for (bias, gradient) in
                node_mut.bias.iter_mut().zip(parameter_gradient.1)
            {
                *bias -= learning_rate * gradient;
            }
        }

        Ok(())
    }

    fn forward_training(
        &self,
        input: &[f32],
    ) -> anyhow::Result<TrainingForward> {
        self.validate()?;
        let expected = self.input_shape()?;

        anyhow::ensure!(
            input.len() == expected.size(),
            "input size {} != expected {}",
            input.len(),
            expected.size()
        );

        let by_id = self.node_map();
        let order = self.topological_order()?;
        let shapes = self.infer_shapes()?;
        let mut values = ValueMap::new();
        let mut caches = CacheMap::new();

        for id in &order {
            let node = by_id.get(id).expect("node from order");
            let node_shape = shapes[id];

            match &node.op {
                GraphOp::Input { .. } => {
                    values.insert(*id, input.to_vec());
                    caches.insert(*id, TrainingCache::Input);
                }
                GraphOp::Dense { input: width, output } => {
                    let x = single_input(&values, node)?.to_vec();
                    let mut y = vec![0.0; *output];

                    for (o, y_value) in
                        y.iter_mut().enumerate().take(*output)
                    {
                        let mut sum = node.bias[o];

                        for (i, x_value) in
                            x.iter().enumerate().take(*width)
                        {
                            sum +=
                                node.weights[o * *width + i] * *x_value;
                        }

                        *y_value = sum;
                    }

                    values.insert(*id, y);
                    caches.insert(
                        *id,
                        TrainingCache::Dense {
                            input: x.to_vec(),
                        },
                    );
                }
                GraphOp::Conv1d {
                    input_channels,
                    output_channels,
                    kernel,
                    stride,
                } => {
                    let x = single_input(&values, node)?.to_vec();
                    let y = conv1d(
                        &x,
                        *input_channels,
                        *output_channels,
                        *kernel,
                        *stride,
                        &node.weights,
                        &node.bias,
                    )?;

                    values.insert(*id, y);
                    caches.insert(
                        *id,
                        TrainingCache::Conv1d {
                            input: x.to_vec(),
                        },
                    );
                }
                GraphOp::SelfAttention { channels, heads } => {
                    let x = single_input(&values, node)?.to_vec();
                    let attention =
                        self_attention_training_forward(
                            &x,
                            *channels,
                            *heads,
                            &node.weights,
                            &node.bias,
                        )?;

                    values.insert(*id, attention.output);
                    caches.insert(
                        *id,
                        TrainingCache::SelfAttention {
                            input: x,
                            q: attention.q,
                            k: attention.k,
                            v: attention.v,
                            probabilities: attention.probabilities,
                        },
                    );
                }
                GraphOp::Recurrent {
                    input_size,
                    hidden_size,
                } => {
                    let x = single_input(&values, node)?.to_vec();
                    let (y, states) =
                        recurrent_training_forward(
                            &x,
                            *input_size,
                            *hidden_size,
                            &node.weights,
                            &node.bias,
                        )?;

                    values.insert(*id, y);
                    caches.insert(
                        *id,
                        TrainingCache::Recurrent {
                            input: x.to_vec(),
                            states,
                        },
                    );
                }
                GraphOp::Add => {
                    let mut y =
                        vec![0.0; node_shape.size()];

                    for source in &node.inputs {
                        let value =
                            values.get(source).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "missing value for node {} input {}",
                                    node.id,
                                    source
                                )
                            })?;

                        anyhow::ensure!(
                            value.len() == y.len(),
                            "add node {} input shape mismatch",
                            node.id
                        );

                        for (dst, src) in
                            y.iter_mut().zip(value.iter())
                        {
                            *dst += *src;
                        }
                    }

                    values.insert(*id, y);
                    caches.insert(*id, TrainingCache::Add);
                }
                GraphOp::Activation { activation } => {
                    let x = single_input(&values, node)?.to_vec();
                    let y = x
                        .iter()
                        .map(|value| activation.apply(*value))
                        .collect::<Vec<_>>();

                    values.insert(*id, y);
                    caches.insert(
                        *id,
                        TrainingCache::Activation {
                            input: x.to_vec(),
                        },
                    );
                }
            }
        }

        Ok((order, values, caches))
    }

    pub fn replace_op(
        &mut self,
        node_id: usize,
        op: GraphOp,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        let node_index = self.index_of(node_id)?;
        let old_shape = self.infer_shapes()?[&node_id];
        let input_shape = {
            let node = &self.nodes[node_index];
            if node.inputs.len() == 1 {
                Some(self.infer_shapes()?[&node.inputs[0]])
            } else {
                None
            }
        };

        let inputs = self.nodes[node_index].inputs.clone();
        let replacement =
            GraphNode::with_random_parameters(
                node_id,
                inputs,
                op,
                input_shape,
                rng,
            )?;

        let mut candidate = self.clone();
        candidate.nodes[node_index] = replacement;
        candidate.validate()?;
        anyhow::ensure!(
            candidate.infer_shapes()?[&node_id] == old_shape,
            "replacement changes node {} shape",
            node_id
        );
        *self = candidate;
        Ok(())
    }

    pub fn rewire_input(
        &mut self,
        node_id: usize,
        slot: usize,
        new_input: usize,
    ) -> anyhow::Result<()> {
        let node_index = self.index_of(node_id)?;
        let old_output_shape = self.output_shape()?;
        anyhow::ensure!(
            new_input != node_id,
            "node cannot consume itself"
        );
        anyhow::ensure!(
            slot < self.nodes[node_index].inputs.len(),
            "input slot {} is out of range",
            slot
        );

        let mut candidate = self.clone();
        candidate.nodes[node_index].inputs[slot] = new_input;
        candidate.validate()?;
        anyhow::ensure!(
            candidate.output_shape()? == old_output_shape,
            "rewire changes graph output shape"
        );
        *self = candidate;
        Ok(())
    }

    pub fn add_node(
        &mut self,
        op: GraphOp,
        inputs: Vec<usize>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<usize> {
        let id = self
            .nodes
            .iter()
            .map(|node| node.id)
            .max()
            .map_or(0, |max_id| max_id + 1);

        let input_shape = if inputs.len() == 1 {
            Some(self.infer_shapes()?[&inputs[0]])
        } else {
            None
        };

        let node = GraphNode::with_random_parameters(
            id,
            inputs,
            op,
            input_shape,
            rng,
        )?;

        let mut candidate = self.clone();
        candidate.nodes.push(node);
        candidate.validate()?;
        *self = candidate;

        Ok(id)
    }

    pub fn remove_node(
        &mut self,
        node_id: usize,
        replacement: usize,
    ) -> anyhow::Result<()> {
        let old_output_shape = self.output_shape()?;
        anyhow::ensure!(node_id != replacement, "replacement matches removed node");
        anyhow::ensure!(
            self.output != node_id || replacement != node_id,
            "removed node cannot be the replacement"
        );

        let by_id = self.node_map();
        let removed = by_id
            .get(&node_id)
            .ok_or_else(|| anyhow::anyhow!("unknown node {}", node_id))?;

        anyhow::ensure!(
            !matches!(&removed.op, GraphOp::Input { .. }),
            "input node cannot be removed"
        );
        anyhow::ensure!(
            by_id.contains_key(&replacement),
            "replacement node {} is missing",
            replacement
        );

        let mut candidate = self.clone();

        candidate.nodes.retain(|node| node.id != node_id);

        for node in &mut candidate.nodes {
            for input in &mut node.inputs {
                if *input == node_id {
                    *input = replacement;
                }
            }
        }

        if candidate.output == node_id {
            candidate.output = replacement;
        }

        candidate.validate()?;
        anyhow::ensure!(
            candidate.output_shape()? == old_output_shape,
            "removal changes graph output shape"
        );
        *self = candidate;
        Ok(())
    }

    pub fn mutate<R: Rng>(
        &mut self,
        rng: &mut R,
        mutation_count: usize,
    ) -> Vec<GraphMutation> {
        let mut history = Vec::with_capacity(mutation_count);

        for _ in 0..mutation_count {
            let Some(mutation) = self.random_mutation(rng) else {
                break;
            };

            if self.apply_mutation(&mutation, rng).is_ok() {
                history.push(mutation);
            }
        }

        history
    }

    fn random_mutation<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Option<GraphMutation> {
        let shapes = self.infer_shapes().ok()?;
        let selectable = self
            .nodes
            .iter()
            .filter(|node| !matches!(&node.op, GraphOp::Input { .. }))
            .collect::<Vec<_>>();

        if selectable.is_empty() {
            return None;
        }

        match rng.random_range(0..5) {
            0 => {
                let node = selectable[rng.random_range(0..selectable.len())];
                let candidates = self
                    .nodes
                    .iter()
                    .filter(|candidate| {
                        candidate.id != node.id
                            && shapes
                                .get(&candidate.id)
                                .copied()
                                .is_some_and(|shape| {
                                    node.inputs
                                        .first()
                                        .and_then(|input| shapes.get(input))
                                        .is_some_and(|input_shape| {
                                            *input_shape == shape
                                        })
                                })
                    })
                    .map(|candidate| candidate.id)
                    .collect::<Vec<_>>();

                if candidates.is_empty() {
                    None
                } else {
                    candidates
                        .get(rng.random_range(0..candidates.len()))
                        .copied()
                        .map(|new_input| GraphMutation::RewireInput {
                            node: node.id,
                            slot: 0,
                            new_input,
                        })
                }
            }
            1 => {
                let node = selectable[rng.random_range(0..selectable.len())];
                let input = node.inputs.first()?;
                let shape = shapes[input];

                let op = match rng.random_range(0..4) {
                    0 if shape.is_vector() => {
                        GraphOp::Activation {
                            activation: random_activation(rng),
                        }
                    }
                    1 if shape.is_vector() => GraphOp::Dense {
                        input: shape.size(),
                        output: shape.size(),
                    },
                    2 if !shape.is_vector() => {
                        GraphOp::Conv1d {
                            input_channels: shape.channels,
                            output_channels: shape.channels,
                            kernel: 1,
                            stride: 1,
                        }
                    }
                    _ if !shape.is_vector() => {
                        let heads =
                            (1..=4)
                                .rev()
                                .find(|heads| shape.channels % heads == 0)
                                .unwrap_or(1);

                        if rng.random::<bool>() {
                            GraphOp::SelfAttention {
                                channels: shape.channels,
                                heads,
                            }
                        } else {
                            GraphOp::Recurrent {
                                input_size: shape.channels,
                                hidden_size: shape.channels,
                            }
                        }
                    }
                    _ => {
                        GraphOp::Activation {
                            activation: random_activation(rng),
                        }
                    }
                };

                Some(GraphMutation::ReplaceOp {
                    node: node.id,
                    op,
                })
            }
            2 => {
                let shape = {
                    let node = selectable[rng.random_range(0..selectable.len())];
                    shapes[&node.id]
                };

                if shapes.get(&self.output).copied()? != shape {
                    return None;
                }

                let peers = selectable
                    .iter()
                    .filter(|node| shapes[&node.id] == shape)
                    .map(|node| node.id)
                    .collect::<Vec<_>>();

                if peers.len() < 2 {
                    return None;
                }

                let first_index =
                    rng.random_range(0..peers.len());
                let mut second_index =
                    rng.random_range(0..peers.len());
                while second_index == first_index {
                    second_index =
                        rng.random_range(0..peers.len());
                }

                Some(GraphMutation::AddResidual {
                    left: peers[first_index],
                    right: peers[second_index],
                })
            }
            3 => {
                if selectable.len() <= 1 {
                    return None;
                }

                let node = selectable[rng.random_range(0..selectable.len())];
                let replacement = node.inputs.first().copied()?;

                Some(GraphMutation::RemoveNode {
                    node: node.id,
                    replacement,
                })
            }
            _ => {
                let node = selectable[rng.random_range(0..selectable.len())];
                if !node.inputs.is_empty() {
                    let slot =
                        rng.random_range(0..node.inputs.len());
                    let shape = shapes[&node.inputs[slot]];
                    let candidates = self
                        .nodes
                        .iter()
                        .filter(|candidate| {
                            candidate.id != node.id
                                && shapes[&candidate.id] == shape
                        })
                        .map(|candidate| candidate.id)
                        .collect::<Vec<_>>();

                    if candidates.is_empty() {
                        None
                    } else {
                        candidates
                            .get(rng.random_range(0..candidates.len()))
                            .copied()
                            .map(|new_input| GraphMutation::RewireInput {
                                node: node.id,
                                slot,
                                new_input,
                            })
                    }
                } else {
                    None
                }
            }
        }
    }

    fn apply_mutation<R: Rng>(
        &mut self,
        mutation: &GraphMutation,
        rng: &mut R,
    ) -> anyhow::Result<()> {
        match mutation {
            GraphMutation::RewireInput {
                node,
                slot,
                new_input,
            } => self.rewire_input(*node, *slot, *new_input),
            GraphMutation::ReplaceOp { node, op } => {
                self.replace_op(*node, op.clone(), rng)
            }
            GraphMutation::AddResidual { left, right } => {
                let id = self.add_node(
                    GraphOp::Add,
                    vec![*left, *right],
                    rng,
                )?;
                self.output = id;
                self.validate()
            }
            GraphMutation::RemoveNode {
                node,
                replacement,
            } => self.remove_node(*node, *replacement),
        }
    }

    fn node_map(&self) -> std::collections::HashMap<usize, &GraphNode> {
        self.nodes
            .iter()
            .map(|node| (node.id, node))
            .collect()
    }

    fn index_of(&self, id: usize) -> anyhow::Result<usize> {
        self.nodes
            .iter()
            .position(|node| node.id == id)
            .ok_or_else(|| anyhow::anyhow!("unknown node {}", id))
    }
}


type ValueMap = std::collections::HashMap<usize, Vec<f32>>;
type CacheMap = std::collections::HashMap<usize, TrainingCache>;
type TrainingForward = (Vec<usize>, ValueMap, CacheMap);

#[derive(Debug, Clone)]
struct AttentionForward {
    output: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probabilities: Vec<f32>,
}

#[derive(Debug, Clone)]
enum TrainingCache {
    Input,
    Dense {
        input: Vec<f32>,
    },
    Conv1d {
        input: Vec<f32>,
    },
    SelfAttention {
        input: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        probabilities: Vec<f32>,
    },
    Recurrent {
        input: Vec<f32>,
        states: Vec<f32>,
    },
    Add,
    Activation {
        input: Vec<f32>,
    },
}

fn backward_node(
    node: &GraphNode,
    cache: &TrainingCache,
    gradient_output: &[f32],
) -> anyhow::Result<Vec<(usize, Vec<f32>)>> {
    match cache {
        TrainingCache::Input => Ok(Vec::new()),
        TrainingCache::Dense { input: _ } => {
            let GraphOp::Dense { input: width, output } = &node.op else {
                anyhow::bail!("dense cache does not match node {}", node.id);
            };

            anyhow::ensure!(
                gradient_output.len() == *output,
                "dense output gradient mismatch"
            );

            let mut gradient_input = vec![0.0; *width];

            for (o, gradient_value) in
                gradient_output.iter().enumerate()
            {
                for (i, input_gradient) in
                    gradient_input.iter_mut().enumerate().take(*width)
                {
                    *input_gradient +=
                        node.weights[o * *width + i]
                            * *gradient_value;
                }
            }

            Ok(vec![(node.inputs[0], gradient_input)])
        }
        TrainingCache::Conv1d { input } => {
            let GraphOp::Conv1d {
                input_channels,
                output_channels,
                kernel,
                stride,
            } = &node.op else {
                anyhow::bail!("conv cache does not match node {}", node.id);
            };

            let length = input.len() / *input_channels;
            let output_length =
                (length - *kernel + *stride).div_ceil(*stride);
            let mut gradient_input =
                vec![0.0; input.len()];

            for oc in 0..*output_channels {
                for pos in 0..output_length {
                    let gradient_value =
                        gradient_output[oc * output_length + pos];

                    for ic in 0..*input_channels {
                        for k in 0..*kernel {
                            let source =
                                pos * *stride + k;

                            if source < length {
                                let weight =
                                    node.weights[
                                        (oc * *input_channels + ic)
                                            * *kernel
                                            + k
                                    ];

                                gradient_input[
                                    ic * length + source
                                ] += weight * gradient_value;
                            }
                        }
                    }
                }
            }

            Ok(vec![(node.inputs[0], gradient_input)])
        }
        TrainingCache::SelfAttention {
            input,
            q,
            k,
            v,
            probabilities,
        } => {
            let GraphOp::SelfAttention {
                channels,
                heads,
            } = &node.op else {
                anyhow::bail!(
                    "attention cache does not match node {}",
                    node.id
                );
            };

            let length = input.len() / *channels;
            let head_dim = *channels / *heads;
            let mut gradient_q =
                vec![0.0; q.len()];
            let mut gradient_k =
                vec![0.0; k.len()];
            let mut gradient_v =
                vec![0.0; v.len()];
            let mut gradient_context =
                vec![0.0; input.len()];
            let output_base =
                3 * *channels * *channels;

            for seq in 0..length {
                for row in 0..*channels {
                    let gradient_value =
                        gradient_output[
                            row * length + seq
                        ];

                    for column in 0..*channels {
                        gradient_context[
                            column * length + seq
                        ] += node.weights[
                            output_base
                                + row * *channels
                                + column
                        ] * gradient_value;
                    }
                }
            }

            for head in 0..*heads {
                let start = head * head_dim;
                let end = start + head_dim;

                for query in 0..length {
                    let mut gradient_probability =
                        vec![0.0; length];

                    for key in 0..length {
                        let mut value = 0.0;

                        for channel in start..end {
                            value +=
                                gradient_context[
                                    channel * length + query
                                ] * v[
                                    channel * length + key
                                ];

                            gradient_v[
                                channel * length + key
                            ] += probabilities[
                                (head * length + query)
                                    * length
                                    + key
                            ] * gradient_context[
                                channel * length + query
                            ];
                        }

                        gradient_probability[key] = value;
                    }

                    let probability_base =
                        (head * length + query)
                            * length;
                    let mut weighted_probability = 0.0;

                    for key in 0..length {
                        weighted_probability +=
                            gradient_probability[key]
                                * probabilities[
                                    probability_base + key
                                ];
                    }

                    for key in 0..length {
                        let probability =
                            probabilities[
                                probability_base + key
                            ];
                        let gradient_score =
                            probability
                                * (gradient_probability[key]
                                    - weighted_probability);

                        for channel in start..end {
                            let q_value =
                                q[channel * length + query];
                            let k_value =
                                k[channel * length + key];
                            let scale =
                                (head_dim as f32).sqrt();

                            gradient_q[
                                channel * length + query
                            ] += gradient_score
                                * k_value
                                / scale;

                            gradient_k[
                                channel * length + key
                            ] += gradient_score
                                * q_value
                                / scale;
                        }
                    }
                }
            }

            let mut gradient_input =
                vec![0.0; input.len()];
            let q_base = 0;
            let k_base =
                *channels * *channels;
            let v_base =
                2 * *channels * *channels;

            for seq in 0..length {
                for row in 0..*channels {
                    let q_grad =
                        gradient_q[row * length + seq];
                    let k_grad =
                        gradient_k[row * length + seq];
                    let v_grad =
                        gradient_v[row * length + seq];

                    for column in 0..*channels {
                        gradient_input[
                            column * length + seq
                        ] +=
                            node.weights[
                                q_base
                                    + row * *channels
                                    + column
                            ] * q_grad
                            + node.weights[
                                k_base
                                    + row * *channels
                                    + column
                            ] * k_grad
                            + node.weights[
                                v_base
                                    + row * *channels
                                    + column
                            ] * v_grad;
                    }
                }
            }

            Ok(vec![(node.inputs[0], gradient_input)])
        }
        TrainingCache::Recurrent {
            input,
            states,
        } => {
            let GraphOp::Recurrent {
                input_size,
                hidden_size,
            } = &node.op else {
                anyhow::bail!(
                    "recurrent cache does not match node {}",
                    node.id
                );
            };

            let length = input.len() / *input_size;
            let input_weight_count =
                *input_size * *hidden_size;
            let mut gradient_input =
                vec![0.0; input.len()];
            let mut gradient_state =
                vec![0.0; *hidden_size];

            for time in (0..length).rev() {
                let state_base =
                    (time + 1) * *hidden_size;
                let mut gradient_pre =
                    vec![0.0; *hidden_size];

                for hidden in 0..*hidden_size {
                    let state_value =
                        states[state_base + hidden];
                    gradient_pre[hidden] =
                        (gradient_output[
                            hidden * length + time
                        ] + gradient_state[hidden])
                            * (1.0 - state_value * state_value);
                }

                for (hidden, gradient_value) in
                    gradient_pre.iter().copied().enumerate()
                {
                    for input_index in 0..*input_size {
                        gradient_input[
                            input_index * length
                                + time
                        ] += node.weights[
                            hidden * *input_size
                                + input_index
                        ] * gradient_value;
                    }

                    let recurrent_base =
                        input_weight_count
                            + hidden * *hidden_size;

                    for (previous, state_gradient) in
                        gradient_state.iter_mut().enumerate()
                    {
                        *state_gradient +=
                            node.weights[
                                recurrent_base + previous
                            ] * gradient_value;
                    }
                }

            }

            Ok(vec![(node.inputs[0], gradient_input)])
        }
        TrainingCache::Add => {
            Ok(node
                .inputs
                .iter()
                .map(|input| {
                    (*input, gradient_output.to_vec())
                })
                .collect())
        }
        TrainingCache::Activation { input } => {
            let GraphOp::Activation { activation } = &node.op else {
                anyhow::bail!(
                    "activation cache does not match node {}",
                    node.id
                );
            };

            let gradient_input = input
                .iter()
                .zip(gradient_output.iter())
                .map(|(value, gradient)| {
                    *gradient * activation.derivative(*value)
                })
                .collect::<Vec<_>>();

            Ok(vec![(node.inputs[0], gradient_input)])
        }
    }
}

fn parameter_gradients(
    node: &GraphNode,
    cache: &TrainingCache,
    gradient_output: &[f32],
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    match cache {
        TrainingCache::Input
        | TrainingCache::Add
        => Ok((
            vec![0.0; node.weights.len()],
            vec![0.0; node.bias.len()],
        )),
        TrainingCache::Activation { .. } => Ok((
            Vec::new(),
            Vec::new(),
        )),
        TrainingCache::Dense { input } => {
            let GraphOp::Dense { input: width, output } = &node.op else {
                anyhow::bail!("dense cache does not match node {}", node.id);
            };
            let mut gradient_weights =
                vec![0.0; node.weights.len()];
            let mut gradient_bias =
                vec![0.0; node.bias.len()];

            for o in 0..*output {
                gradient_bias[o] = gradient_output[o];

                for i in 0..*width {
                    gradient_weights[
                        o * *width + i
                    ] = gradient_output[o] * input[i];
                }
            }

            Ok((gradient_weights, gradient_bias))
        }
        TrainingCache::Conv1d { input } => {
            let GraphOp::Conv1d {
                input_channels,
                output_channels,
                kernel,
                stride,
            } = &node.op else {
                anyhow::bail!("conv cache does not match node {}", node.id);
            };

            let length = input.len() / *input_channels;
            let output_length =
                (length - *kernel + *stride).div_ceil(*stride);
            let mut gradient_weights =
                vec![0.0; node.weights.len()];
            let mut gradient_bias =
                vec![0.0; node.bias.len()];

            for oc in 0..*output_channels {
                for pos in 0..output_length {
                    let gradient_value =
                        gradient_output[
                            oc * output_length + pos
                        ];
                    gradient_bias[oc] +=
                        gradient_value;

                    for ic in 0..*input_channels {
                        for k in 0..*kernel {
                            let source =
                                pos * *stride + k;

                            if source < length {
                                gradient_weights[
                                    (oc * *input_channels
                                        + ic) * *kernel
                                        + k
                                ] += gradient_value
                                    * input[
                                        ic * length
                                            + source
                                    ];
                            }
                        }
                    }
                }
            }

            Ok((gradient_weights, gradient_bias))
        }
        TrainingCache::SelfAttention {
            input,
            q,
            k,
            v,
            probabilities,
        } => {
            let GraphOp::SelfAttention {
                channels,
                heads,
            } = &node.op else {
                anyhow::bail!(
                    "attention cache does not match node {}",
                    node.id
                );
            };

            let length = input.len() / *channels;
            let head_dim = *channels / *heads;
            let output_base = 3 * *channels * *channels;
            let mut gradient_weights =
                vec![0.0; node.weights.len()];
            let mut gradient_bias =
                vec![0.0; node.bias.len()];
            let mut gradient_context =
                vec![0.0; input.len()];
            let mut gradient_q =
                vec![0.0; q.len()];
            let mut gradient_k =
                vec![0.0; k.len()];
            let mut gradient_v =
                vec![0.0; v.len()];

            for head in 0..*heads {
                let start_channel = head * head_dim;
                let end_channel =
                    start_channel + head_dim;

                for query in 0..length {
                    let base =
                        (head * length + query)
                            * length;

                    for channel in
                        start_channel..end_channel
                    {
                        let gradient_value =
                            gradient_output[
                                channel * length
                                    + query
                            ];

                        for key in 0..length {
                            gradient_v[
                                channel * length
                                    + key
                            ] += probabilities[
                                base + key
                            ] * gradient_value;
                        }
                    }
                }
            }

            for seq in 0..length {
                for row in 0..*channels {
                    let gradient_value =
                        gradient_output[
                            row * length + seq
                        ];
                    gradient_bias[row] +=
                        gradient_value;

                    for column in 0..*channels {
                        let context_value = attention_context_value(
                            channels,
                            heads,
                            length,
                            probabilities,
                            v,
                            column,
                            seq,
                        );

                        gradient_weights[
                            output_base
                                + row * *channels
                                + column
                        ] += gradient_value
                            * context_value;

                        gradient_context[
                            column * length + seq
                        ] += node.weights[
                            output_base
                                + row * *channels
                                + column
                        ] * gradient_value;
                    }
                }
            }

            for head in 0..*heads {
                let start_channel = head * head_dim;
                let end_channel =
                    start_channel + head_dim;

                for query in 0..length {
                    let base =
                        (head * length + query)
                            * length;
                    let mut gradient_probability =
                        vec![0.0; length];

                    for key in 0..length {
                        for channel in
                            start_channel..end_channel
                        {
                            gradient_probability[key] +=
                                gradient_context[
                                    channel * length
                                        + query
                                ] * v[
                                    channel * length
                                        + key
                                ];
                        }
                    }

                    let weighted_probability =
                        gradient_probability
                            .iter()
                            .enumerate()
                            .map(|(key, value)| {
                                *value
                                    * probabilities[
                                        base + key
                                    ]
                            })
                            .sum::<f32>();

                    for key in 0..length {
                        let probability =
                            probabilities[base + key];
                        let gradient_score =
                            probability
                                * (gradient_probability[key]
                                    - weighted_probability);
                        let scale =
                            (head_dim as f32).sqrt();

                        for channel in
                            start_channel..end_channel
                        {
                            gradient_q[
                                channel * length
                                    + query
                            ] += gradient_score
                                * k[
                                    channel * length
                                        + key
                                ] / scale;

                            gradient_k[
                                channel * length + key
                            ] += gradient_score
                                * q[
                                    channel * length
                                        + query
                                ] / scale;
                        }
                    }
                }
            }

            for seq in 0..length {
                for row in 0..*channels {
                    for column in 0..*channels {
                        gradient_weights[
                            row * *channels
                                + column
                        ] += gradient_q[
                            row * length + seq
                        ] * input[
                            column * length + seq
                        ];

                        gradient_weights[
                            *channels * *channels
                                + row * *channels
                                + column
                        ] += gradient_k[
                            row * length + seq
                        ] * input[
                            column * length + seq
                        ];

                        gradient_weights[
                            2 * *channels * *channels
                                + row * *channels
                                + column
                        ] += gradient_v[
                            row * length + seq
                        ] * input[
                            column * length + seq
                        ];
                    }
                }
            }

            Ok((gradient_weights, gradient_bias))
        }
        TrainingCache::Recurrent {
            input,
            states,
        } => {
            let GraphOp::Recurrent {
                input_size,
                hidden_size,
            } = &node.op else {
                anyhow::bail!(
                    "recurrent cache does not match node {}",
                    node.id
                );
            };

            let length = input.len() / *input_size;
            let input_weight_count =
                *input_size * *hidden_size;
            let mut gradient_weights =
                vec![0.0; node.weights.len()];
            let mut gradient_bias =
                vec![0.0; node.bias.len()];
            let mut gradient_state =
                vec![0.0; *hidden_size];

            for time in (0..length).rev() {
                let state_base =
                    (time + 1) * *hidden_size;
                let mut gradient_pre =
                    vec![0.0; *hidden_size];

                for hidden in 0..*hidden_size {
                    let state_value =
                        states[state_base + hidden];
                    gradient_pre[hidden] =
                        (gradient_output[
                            hidden * length + time
                        ] + gradient_state[hidden])
                            * (1.0 - state_value * state_value);
                    gradient_bias[hidden] +=
                        gradient_pre[hidden];

                    for input_index in 0..*input_size {
                        gradient_weights[
                            hidden * *input_size
                                + input_index
                        ] += gradient_pre[hidden]
                            * input[
                                input_index * length
                                    + time
                            ];
                    }

                    let recurrent_base =
                        input_weight_count
                            + hidden * *hidden_size;

                    for previous in 0..*hidden_size {
                        gradient_weights[
                            recurrent_base + previous
                        ] += gradient_pre[hidden]
                            * states[
                                time * *hidden_size
                                    + previous
                            ];

                        gradient_state[previous] +=
                            node.weights[
                                recurrent_base
                                    + previous
                            ] * gradient_pre[hidden];
                    }
                }
            }

            Ok((gradient_weights, gradient_bias))
        }
    }
}

fn attention_context_value(
    channels: &usize,
    heads: &usize,
    length: usize,
    probabilities: &[f32],
    values: &[f32],
    channel: usize,
    query: usize,
) -> f32 {
    let head_dim = *channels / *heads;
    let head = channel / head_dim;
    let base =
        (head * length + query) * length;

    (0..length)
        .map(|key| {
            probabilities[base + key]
                * values[channel * length + key]
        })
        .sum()
}

fn self_attention_training_forward(
    x: &[f32],
    channels: usize,
    heads: usize,
    weights: &[f32],
    bias: &[f32],
) -> anyhow::Result<AttentionForward> {
    let length = x.len() / channels;
    let head_dim = channels / heads;
    let mut q = vec![0.0; channels * length];
    let mut k = vec![0.0; channels * length];
    let mut v = vec![0.0; channels * length];

    for seq in 0..length {
        for row in 0..channels {
            let mut q_value = 0.0;
            let mut k_value = 0.0;
            let mut v_value = 0.0;

            for column in 0..channels {
                let x_value =
                    x[column * length + seq];

                q_value +=
                    weights[
                        row * channels + column
                    ] * x_value;

                k_value +=
                    weights[
                        channels * channels
                            + row * channels
                            + column
                    ] * x_value;

                v_value +=
                    weights[
                        2 * channels * channels
                            + row * channels
                            + column
                    ] * x_value;
            }

            q[row * length + seq] = q_value;
            k[row * length + seq] = k_value;
            v[row * length + seq] = v_value;
        }
    }

    let mut probabilities =
        vec![0.0; heads * length * length];

    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;

        for query in 0..length {
            let mut scores = vec![0.0; length];

            for key in 0..length {
                let mut dot = 0.0;

                for channel in start..end {
                    dot +=
                        q[channel * length + query]
                            * k[channel * length + key];
                }

                scores[key] =
                    dot / (head_dim as f32).sqrt();
            }

            softmax_in_place(&mut scores);

            let base =
                (head * length + query)
                    * length;

            probabilities[
                base..base + length
            ]
                .copy_from_slice(&scores);
        }
    }

    let mut context =
        vec![0.0; channels * length];

    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;

        for query in 0..length {
            let base =
                (head * length + query)
                    * length;

            for channel in start..end {
                context[
                    channel * length + query
                ] = (0..length)
                    .map(|key| {
                        probabilities[
                            base + key
                        ] * v[
                            channel * length + key
                        ]
                    })
                    .sum();
            }
        }
    }

    let output_base =
        3 * channels * channels;
    let mut y =
        vec![0.0; channels * length];

    for seq in 0..length {
        for row in 0..channels {
            let mut sum =
                bias[row];

            for column in 0..channels {
                sum +=
                    weights[
                        output_base
                            + row * channels
                            + column
                    ] * context[
                        column * length + seq
                    ];
            }

            y[row * length + seq] =
                sum;
        }
    }

    Ok(AttentionForward {
        output: y,
        q,
        k,
        v,
        probabilities,
    })
}

fn recurrent_training_forward(
    x: &[f32],
    input_size: usize,
    hidden_size: usize,
    weights: &[f32],
    bias: &[f32],
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let length = x.len() / input_size;
    let input_weight_count =
        input_size * hidden_size;
    let mut states =
        vec![0.0; (length + 1) * hidden_size];

    for time in 0..length {
        for hidden in 0..hidden_size {
            let mut sum = bias[hidden];

            for input_index in 0..input_size {
                sum += weights[
                    hidden * input_size
                        + input_index
                ] * x[
                    input_index * length
                        + time
                ];
            }

            let recurrent_base =
                input_weight_count
                    + hidden * hidden_size;

            for previous in 0..hidden_size {
                sum += weights[
                    recurrent_base + previous
                ] * states[
                    time * hidden_size + previous
                ];
            }

            states[
                (time + 1) * hidden_size + hidden
            ] = sum.tanh();
        }
    }

    let mut channel_major =
        vec![0.0; hidden_size * length];

    for time in 0..length {
        for hidden in 0..hidden_size {
            channel_major[
                hidden * length + time
            ] = states[
                (time + 1) * hidden_size + hidden
            ];
        }
    }

    Ok((channel_major, states))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphMutation {
    RewireInput {
        node: usize,
        slot: usize,
        new_input: usize,
    },
    ReplaceOp {
        node: usize,
        op: GraphOp,
    },
    AddResidual {
        left: usize,
        right: usize,
    },
    RemoveNode {
        node: usize,
        replacement: usize,
    },
}

pub fn random_graph<R: Rng>(
    input_shape: TensorShape,
    output_size: usize,
    rng: &mut R,
) -> anyhow::Result<GraphNetwork> {
    anyhow::ensure!(output_size > 0, "output_size must be > 0");

    let input = GraphNode {
        id: 0,
        inputs: Vec::new(),
        op: GraphOp::Input { shape: input_shape },
        weights: Vec::new(),
        bias: Vec::new(),
    };

    let feature_node_id = 1;
    let feature_op = if input_shape.is_vector() {
        GraphOp::Dense {
            input: input_shape.size(),
            output: input_shape.size(),
        }
    } else {
        GraphOp::SelfAttention {
            channels: input_shape.channels,
            heads: (1..=4)
                .rev()
                .find(|heads| input_shape.channels.is_multiple_of(*heads))
                .unwrap_or(1),
        }
    };

    let feature = GraphNode::with_random_parameters(
        feature_node_id,
        vec![0],
        feature_op,
        Some(input_shape),
        rng,
    )?;

    let activation = GraphNode::with_random_parameters(
        2,
        vec![feature_node_id],
        GraphOp::Activation {
            activation: Activation::Relu,
        },
        Some(if input_shape.is_vector() {
            input_shape
        } else {
            TensorShape {
                channels: input_shape.channels,
                length: input_shape.length,
            }
        }),
        rng,
    )?;

    let output_input_shape =
        infer_node_shape(&feature.op, &[input_shape])?;

    let flatten_size = output_input_shape.size();
    let output = GraphNode::with_random_parameters(
        3,
        vec![2],
        GraphOp::Dense {
            input: flatten_size,
            output: output_size,
        },
        Some(output_input_shape),
        rng,
    )?;

    let graph = GraphNetwork {
        nodes: vec![input, feature, activation, output],
        output: 3,
    };

    graph.validate()?;
    Ok(graph)
}

fn parameter_layout(
    op: &GraphOp,
    _input_shape: Option<TensorShape>,
) -> anyhow::Result<(usize, usize, f32)> {
    match op {
        GraphOp::Input { .. }
        | GraphOp::Add
        | GraphOp::Activation { .. } => Ok((0, 0, 1.0)),
        GraphOp::Dense { input, output } => {
            anyhow::ensure!(*input > 0 && *output > 0, "dense dimensions must be > 0");
            Ok((
                input * output,
                *output,
                (2.0 / *input as f32).sqrt(),
            ))
        }
        GraphOp::Conv1d {
            input_channels,
            output_channels,
            kernel,
            stride,
        } => {
            anyhow::ensure!(
                *input_channels > 0
                    && *output_channels > 0
                    && *kernel > 0
                    && *stride > 0,
                "conv dimensions must be > 0"
            );
            let fan_in = input_channels * kernel;
            Ok((
                output_channels * fan_in,
                *output_channels,
                (2.0 / fan_in as f32).sqrt(),
            ))
        }
        GraphOp::SelfAttention { channels, heads } => {
            anyhow::ensure!(*channels > 0, "attention channels must be > 0");
            anyhow::ensure!(
                *heads > 0 && channels % heads == 0,
                "attention heads must divide channels"
            );
            Ok((
                4 * channels * channels,
                *channels,
                (2.0 / *channels as f32).sqrt(),
            ))
        }
        GraphOp::Recurrent {
            input_size,
            hidden_size,
        } => {
            anyhow::ensure!(
                *input_size > 0 && *hidden_size > 0,
                "recurrent dimensions must be > 0"
            );
            let fan_in = input_size + hidden_size;
            Ok((
                hidden_size * fan_in,
                *hidden_size,
                (2.0 / fan_in as f32).sqrt(),
            ))
        }
    }
}

fn validate_parameters(
    node: &GraphNode,
    output_shape: TensorShape,
    input_shapes: &[TensorShape],
) -> anyhow::Result<()> {
    let (weight_count, bias_count, _) =
        parameter_layout(&node.op, input_shapes.first().copied())?;

    anyhow::ensure!(
        node.weights.len() == weight_count,
        "node {} ({}) has {} weights, expected {}",
        node.id,
        node.op.kind_name(),
        node.weights.len(),
        weight_count
    );
    anyhow::ensure!(
        node.bias.len() == bias_count,
        "node {} ({}) has {} biases, expected {}",
        node.id,
        node.op.kind_name(),
        node.bias.len(),
        bias_count
    );

    match &node.op {
        GraphOp::Dense { input, output } => {
            anyhow::ensure!(
                input_shapes.len() == 1
                    && input_shapes[0].size() == *input,
                "dense node {} input shape mismatch",
                node.id
            );
            anyhow::ensure!(
                output_shape.size() == *output,
                "dense node {} output shape mismatch",
                node.id
            );
        }
        GraphOp::Conv1d {
            input_channels,
            output_channels,
            kernel,
            stride,
        } => {
            anyhow::ensure!(input_shapes.len() == 1, "conv node {} needs one input", node.id);
            let input = input_shapes[0];
            anyhow::ensure!(
                input.channels == *input_channels,
                "conv node {} channel mismatch",
                node.id
            );
            anyhow::ensure!(
                input.length >= *kernel,
                "conv node {} kernel exceeds sequence length",
                node.id
            );
            anyhow::ensure!(*stride > 0, "conv stride must be > 0");
            anyhow::ensure!(
                output_shape.channels == *output_channels,
                "conv node {} output channel mismatch",
                node.id
            );
        }
        GraphOp::SelfAttention { channels, heads } => {
            anyhow::ensure!(
                input_shapes.len() == 1,
                "attention node {} needs one input",
                node.id
            );
            anyhow::ensure!(
                input_shapes[0].channels == *channels
                    && channels % heads == 0,
                "attention node {} shape/head mismatch",
                node.id
            );
        }
        GraphOp::Recurrent {
            input_size,
            hidden_size,
        } => {
            anyhow::ensure!(
                input_shapes.len() == 1
                    && input_shapes[0].channels == *input_size,
                "recurrent node {} input mismatch",
                node.id
            );
            anyhow::ensure!(
                output_shape.channels == *hidden_size,
                "recurrent node {} output mismatch",
                node.id
            );
        }
        GraphOp::Add => {
            anyhow::ensure!(
                !input_shapes.is_empty(),
                "add node {} needs at least one input",
                node.id
            );
            anyhow::ensure!(
                input_shapes
                    .iter()
                    .all(|shape| *shape == input_shapes[0]),
                "add node {} inputs must have identical shapes",
                node.id
            );
        }
        GraphOp::Activation { .. } => {
            anyhow::ensure!(
                input_shapes.len() == 1,
                "activation node {} needs one input",
                node.id
            );
        }
        GraphOp::Input { shape } => {
            anyhow::ensure!(
                input_shapes.is_empty(),
                "input node {} cannot have inputs",
                node.id
            );
            anyhow::ensure!(
                *shape == output_shape,
                "input node {} shape mismatch",
                node.id
            );
        }
    }

    Ok(())
}

fn infer_node_shape(
    op: &GraphOp,
    input_shapes: &[TensorShape],
) -> anyhow::Result<TensorShape> {
    match op {
        GraphOp::Input { shape } => Ok(*shape),
        GraphOp::Dense { output, .. } => TensorShape::vector(*output),
        GraphOp::Conv1d {
            output_channels,
            kernel,
            stride,
            ..
        } => {
            anyhow::ensure!(input_shapes.len() == 1, "conv needs one input");
            let input = input_shapes[0];
            anyhow::ensure!(
                input.length >= *kernel,
                "conv kernel exceeds sequence length"
            );
            let length =
                (input.length - *kernel + *stride).div_ceil(*stride);
            TensorShape::sequence(*output_channels, length)
        }
        GraphOp::SelfAttention { channels, .. } => {
            anyhow::ensure!(input_shapes.len() == 1, "attention needs one input");
            TensorShape::sequence(*channels, input_shapes[0].length)
        }
        GraphOp::Recurrent { hidden_size, .. } => {
            anyhow::ensure!(input_shapes.len() == 1, "recurrent needs one input");
            TensorShape::sequence(*hidden_size, input_shapes[0].length)
        }
        GraphOp::Add => {
            anyhow::ensure!(!input_shapes.is_empty(), "add needs inputs");
            Ok(input_shapes[0])
        }
        GraphOp::Activation { .. } => {
            anyhow::ensure!(input_shapes.len() == 1, "activation needs one input");
            Ok(input_shapes[0])
        }
    }
}

fn single_input<'a>(
    values: &'a std::collections::HashMap<usize, Vec<f32>>,
    node: &GraphNode,
) -> anyhow::Result<&'a [f32]> {
    anyhow::ensure!(
        node.inputs.len() == 1,
        "node {} needs exactly one input",
        node.id
    );

    values
        .get(&node.inputs[0])
        .map(|value| value.as_slice())
        .ok_or_else(|| anyhow::anyhow!("missing value for node {}", node.inputs[0]))
}

fn conv1d(
    x: &[f32],
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    weights: &[f32],
    bias: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let length = x.len() / input_channels;
    let output_length =
        (length - kernel + stride).div_ceil(stride);

    anyhow::ensure!(
        weights.len() == output_channels * input_channels * kernel,
        "conv weight count mismatch"
    );
    anyhow::ensure!(
        bias.len() == output_channels,
        "conv bias count mismatch"
    );

    let mut y = vec![0.0; output_channels * output_length];

    for oc in 0..output_channels {
        for pos in 0..output_length {
            let mut sum = bias[oc];

            for ic in 0..input_channels {
                for k in 0..kernel {
                    let source = pos * stride + k;
                    if source < length {
                        let weight =
                            weights[(oc * input_channels + ic) * kernel + k];
                        sum += weight
                            * x[ic * length + source];
                    }
                }
            }

            y[oc * output_length + pos] = sum;
        }
    }

    Ok(y)
}

fn self_attention(
    x: &[f32],
    channels: usize,
    heads: usize,
    weights: &[f32],
    bias: &[f32],
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(weights.len() == 4 * channels * channels);
    anyhow::ensure!(bias.len() == channels);

    let length = x.len() / channels;
    let head_dim = channels / heads;

    let matrix = |base: usize, row: usize, column: usize| {
        weights[base + row * channels + column]
    };

    let project = |base: usize, seq: usize, row: usize| {
        (0..channels)
            .map(|column| matrix(base, row, column) * x[column * length + seq])
            .sum::<f32>()
    };

    let mut q = vec![0.0; channels * length];
    let mut k = vec![0.0; channels * length];
    let mut v = vec![0.0; channels * length];

    for seq in 0..length {
        for row in 0..channels {
            q[row * length + seq] =
                project(0, seq, row);
            k[row * length + seq] =
                project(channels * channels, seq, row);
            v[row * length + seq] =
                project(2 * channels * channels, seq, row);
        }
    }

    let mut context = vec![0.0; channels * length];

    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;

        for query in 0..length {
            let mut scores = vec![0.0; length];

            for key in 0..length {
                let mut dot = 0.0;
                for channel in start..end {
                    dot +=
                        q[channel * length + query]
                            * k[channel * length + key];
                }
                scores[key] =
                    dot / (head_dim as f32).sqrt();
            }

            softmax_in_place(&mut scores);

            for channel in start..end {
                context[channel * length + query] =
                    scores
                        .iter()
                        .enumerate()
                        .map(|(key, score)| {
                            score * v[channel * length + key]
                        })
                        .sum();
            }
        }
    }

    let output_base = 3 * channels * channels;
    let mut y = vec![0.0; channels * length];

    for seq in 0..length {
        for row in 0..channels {
            let projected = (0..channels)
                .map(|column| {
                    weights[
                        output_base + row * channels + column
                    ] * context[column * length + seq]
                })
                .sum::<f32>();

            y[row * length + seq] =
                projected + bias[row];
        }
    }

    Ok(y)
}

fn recurrent(
    x: &[f32],
    input_size: usize,
    hidden_size: usize,
    weights: &[f32],
    bias: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let length = x.len() / input_size;
    let input_weights =
        input_size * hidden_size;
    let recurrent_weights =
        hidden_size * hidden_size;

    anyhow::ensure!(
        weights.len() == input_weights + recurrent_weights,
        "recurrent weight count mismatch"
    );
    anyhow::ensure!(
        bias.len() == hidden_size,
        "recurrent bias count mismatch"
    );

    let mut state = vec![0.0; hidden_size];
    let mut output = vec![0.0; hidden_size * length];

    for time in 0..length {
        let mut next = vec![0.0; hidden_size];

        for h in 0..hidden_size {
            let mut sum = bias[h];

            for i in 0..input_size {
                sum += weights[h * input_size + i]
                    * x[i * length + time];
            }

            let recurrent_base =
                input_weights + h * hidden_size;

            for prev in 0..hidden_size {
                sum += weights[recurrent_base + prev]
                    * state[prev];
            }

            next[h] = sum.tanh();
            output[h * length + time] = next[h];
        }

        state = next;
    }

    Ok(output)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn random_activation<R: Rng>(rng: &mut R) -> Activation {
    match rng.random_range(0..3) {
        0 => Activation::Linear,
        1 => Activation::Relu,
        _ => Activation::Tanh,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchConfig {
    pub candidates: usize,
    pub mutations: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub accuracy_tolerance: f32,
}

impl Default for GraphSearchConfig {
    fn default() -> Self {
        Self {
            candidates: 32,
            mutations: 4,
            epochs: 5,
            learning_rate: 0.01,
            accuracy_tolerance: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCandidate {
    pub id: usize,
    pub mse: f32,
    pub parameters: usize,
    pub macs: u64,
    pub mutations: Vec<GraphMutation>,
}

pub fn search_graph<R: Rng>(
    baseline: &GraphNetwork,
    train_inputs: &[Vec<f32>],
    train_targets: &[Vec<f32>],
    validation_inputs: &[Vec<f32>],
    validation_targets: &[Vec<f32>],
    config: &GraphSearchConfig,
    rng: &mut R,
) -> anyhow::Result<Vec<(GraphCandidate, GraphNetwork)>> {
    anyhow::ensure!(
        train_inputs.len() == train_targets.len(),
        "training input/target counts differ"
    );
    anyhow::ensure!(
        validation_inputs.len() == validation_targets.len(),
        "validation input/target counts differ"
    );
    anyhow::ensure!(config.candidates > 0, "graph candidate count must be > 0");
    anyhow::ensure!(config.epochs > 0, "graph training epochs must be > 0");
    anyhow::ensure!(
        config.learning_rate > 0.0
            && config.learning_rate.is_finite(),
        "graph learning_rate must be finite and > 0"
    );

    let baseline_mse =
        graph_mse(baseline, validation_inputs, validation_targets)?;

    let mut results = Vec::with_capacity(config.candidates);

    for id in 0..config.candidates {
        let mut candidate = baseline.clone();
        let mutations =
            candidate.mutate(rng, config.mutations);

        candidate.train(
            train_inputs,
            train_targets,
            config.epochs,
            config.learning_rate,
        )?;

        let mse =
            graph_mse(&candidate, validation_inputs, validation_targets)?;

        results.push((
            GraphCandidate {
                id,
                mse,
                parameters: candidate.parameter_count(),
                macs: candidate.mac_count(),
                mutations,
            },
            candidate,
        ));
    }

    results.sort_by(|a, b| {
        let a_accepted =
            a.0.mse <= baseline_mse + config.accuracy_tolerance;
        let b_accepted =
            b.0.mse <= baseline_mse + config.accuracy_tolerance;

        match (a_accepted, b_accepted) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => a
                .0
                .macs
                .cmp(&b.0.macs)
                .then_with(|| {
                    a.0.mse
                        .partial_cmp(&b.0.mse)
                        .unwrap_or(Ordering::Equal)
                }),
        }
    });

    Ok(results)
}

pub fn graph_mse(
    network: &GraphNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> anyhow::Result<f32> {
    let mut error = 0.0f64;
    let mut count = 0usize;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;

        anyhow::ensure!(
            output.len() == target.len(),
            "graph output width {} != target width {}",
            output.len(),
            target.len()
        );

        for (prediction, expected) in
            output.iter().zip(target)
        {
            let delta =
                *prediction as f64 - *expected as f64;
            error += delta * delta;
            count += 1;
        }
    }

    anyhow::ensure!(count > 0, "validation dataset is empty");

    Ok((error / count as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn convolution_forward_has_expected_shape() {
        let input_shape =
            TensorShape::sequence(2, 5).expect("shape");
        let mut rng = StdRng::seed_from_u64(7);

        let input = GraphNode {
            id: 0,
            inputs: Vec::new(),
            op: GraphOp::Input {
                shape: input_shape,
            },
            weights: Vec::new(),
            bias: Vec::new(),
        };

        let conv =
            GraphNode::with_random_parameters(
                1,
                vec![0],
                GraphOp::Conv1d {
                    input_channels: 2,
                    output_channels: 3,
                    kernel: 3,
                    stride: 1,
                },
                Some(input_shape),
                &mut rng,
            )
            .expect("conv");

        let out =
            GraphNode::with_random_parameters(
                2,
                vec![1],
                GraphOp::Dense {
                    input: 3 * 3,
                    output: 2,
                },
                Some(TensorShape::sequence(3, 3).expect("shape")),
                &mut rng,
            )
            .expect("dense");

        let graph = GraphNetwork {
            nodes: vec![input, conv, out],
            output: 2,
        };

        graph.validate().expect("valid graph");
        assert_eq!(
            graph.forward(&[0.1; 10]).expect("forward").len(),
            2
        );
    }

    #[test]
    fn attention_forward_has_expected_shape() {
        let shape =
            TensorShape::sequence(4, 3).expect("shape");
        let mut rng = StdRng::seed_from_u64(11);

        let input = GraphNode {
            id: 0,
            inputs: Vec::new(),
            op: GraphOp::Input { shape },
            weights: Vec::new(),
            bias: Vec::new(),
        };

        let attention =
            GraphNode::with_random_parameters(
                1,
                vec![0],
                GraphOp::SelfAttention {
                    channels: 4,
                    heads: 2,
                },
                Some(shape),
                &mut rng,
            )
            .expect("attention");

        let flatten =
            GraphNode::with_random_parameters(
                2,
                vec![1],
                GraphOp::Dense {
                    input: 12,
                    output: 2,
                },
                Some(shape),
                &mut rng,
            )
            .expect("dense");

        let graph = GraphNetwork {
            nodes: vec![input, attention, flatten],
            output: 2,
        };

        graph.validate().expect("valid graph");
        assert_eq!(
            graph.forward(&[0.2; 12]).expect("forward").len(),
            2
        );
    }

    #[test]
    fn recurrent_forward_has_expected_shape() {
        let shape =
            TensorShape::sequence(3, 4).expect("shape");
        let mut rng = StdRng::seed_from_u64(13);

        let input = GraphNode {
            id: 0,
            inputs: Vec::new(),
            op: GraphOp::Input { shape },
            weights: Vec::new(),
            bias: Vec::new(),
        };

        let recurrent =
            GraphNode::with_random_parameters(
                1,
                vec![0],
                GraphOp::Recurrent {
                    input_size: 3,
                    hidden_size: 5,
                },
                Some(shape),
                &mut rng,
            )
            .expect("recurrent");

        let dense =
            GraphNode::with_random_parameters(
                2,
                vec![1],
                GraphOp::Dense {
                    input: 20,
                    output: 2,
                },
                Some(TensorShape::sequence(5, 4).expect("shape")),
                &mut rng,
            )
            .expect("dense");

        let graph = GraphNetwork {
            nodes: vec![input, recurrent, dense],
            output: 2,
        };

        graph.validate().expect("valid graph");
        assert_eq!(
            graph.forward(&[0.3; 12]).expect("forward").len(),
            2
        );
    }

    #[test]
    fn residual_dag_forward_and_rewire_are_valid() {
        let shape =
            TensorShape::vector(4).expect("shape");
        let mut rng = StdRng::seed_from_u64(17);
        let graph = random_graph(shape, 2, &mut rng)
            .expect("graph");

        let feature = 1;
        let residual = {
            let mut candidate = graph.clone();
            candidate
                .add_node(GraphOp::Add, vec![feature, 2], &mut rng)
                .expect("residual")
        };

        assert_eq!(residual, 4);
        assert_eq!(graph.forward(&[0.1; 4]).expect("forward").len(), 2);
    }

    #[test]
    fn cycle_detection_rejects_bad_rewire() {
        let shape =
            TensorShape::vector(2).expect("shape");
        let mut rng = StdRng::seed_from_u64(19);
        let mut graph =
            random_graph(shape, 2, &mut rng).expect("graph");

        let result = graph.rewire_input(1, 0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn dense_graph_training_reduces_error() {
        let shape = TensorShape::vector(1).expect("shape");
        let input = GraphNode {
            id: 0,
            inputs: Vec::new(),
            op: GraphOp::Input { shape },
            weights: Vec::new(),
            bias: Vec::new(),
        };
        let dense = GraphNode {
            id: 1,
            inputs: vec![0],
            op: GraphOp::Dense {
                input: 1,
                output: 1,
            },
            weights: vec![0.0],
            bias: vec![0.0],
        };

        let mut graph = GraphNetwork {
            nodes: vec![input, dense],
            output: 1,
        };
        graph.validate().expect("valid graph");

        let inputs = vec![
            vec![-2.0],
            vec![-1.0],
            vec![1.0],
            vec![2.0],
        ];
        let targets = inputs
            .iter()
            .map(|input| vec![3.0 * input[0] + 1.0])
            .collect::<Vec<_>>();

        let before =
            graph_mse(&graph, &inputs, &targets)
                .expect("before MSE");

        graph
            .train(&inputs, &targets, 25, 0.05)
            .expect("graph training");

        let after =
            graph_mse(&graph, &inputs, &targets)
                .expect("after MSE");

        assert!(
            after < before,
            "training did not reduce MSE: before={before}, after={after}"
        );
    }

    #[test]
    fn random_graph_search_produces_candidates() {
        let shape =
            TensorShape::sequence(2, 3).expect("shape");
        let mut rng = StdRng::seed_from_u64(23);
        let graph =
            random_graph(shape, 2, &mut rng).expect("graph");
        let inputs = vec![vec![0.1; 6], vec![0.2; 6]];
        let targets = vec![vec![0.0; 2], vec![0.0; 2]];

        let results = search_graph(
            &graph,
            &inputs,
            &targets,
            &inputs,
            &targets,
            &GraphSearchConfig {
                candidates: 4,
                mutations: 2,
                epochs: 1,
                learning_rate: 0.001,
                accuracy_tolerance: 0.1,
            },
            &mut rng,
        )
        .expect("search");

        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|item| item.0.mse.is_finite()));
    }
}
