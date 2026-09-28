use prost::Message;
use std::fs::File;
use std::io::Write;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, optional, tag = "1")]
    pub ir_version: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub producer_name: Option<String>,
    #[prost(string, optional, tag = "3")]
    pub producer_version: Option<String>,
    #[prost(string, optional, tag = "4")]
    pub domain: Option<String>,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, optional, tag = "1")]
    pub domain: Option<String>,
    #[prost(int64, optional, tag = "2")]
    pub version: Option<i64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, optional, tag = "3")]
    pub name: Option<String>,
    #[prost(string, optional, tag = "4")]
    pub op_type: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, optional, tag = "2")]
    pub data_type: Option<i32>,
    #[prost(string, optional, tag = "8")]
    pub name: Option<String>,
    #[prost(bytes, optional, tag = "9")]
    pub raw_data: Option<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, optional, tag = "1")]
    pub name: Option<String>,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorType>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorType {
    #[prost(int32, optional, tag = "1")]
    pub elem_type: Option<i32>,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeDimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeDimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
}

pub fn float_tensor(
    name: &str,
    shape: &[usize],
    data: &[f32],
) -> TensorProto {
    let mut raw = Vec::with_capacity(data.len() * 4);

    for value in data {
        raw.extend_from_slice(&value.to_le_bytes());
    }

    TensorProto {
        dims: shape.iter().map(|x| *x as i64).collect(),
        data_type: Some(1),
        name: Some(name.to_owned()),
        raw_data: Some(raw),
    }
}

pub fn value_info(
    name: &str,
    shape: &[usize],
) -> ValueInfoProto {
    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorType {
                elem_type: Some(1),
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|x| TensorShapeDimension {
                            dim_value: Some(*x as i64),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

pub fn write_model(
    path: &str,
    model: ModelProto,
) -> anyhow::Result<()> {
    let mut bytes = Vec::new();
    model.encode(&mut bytes)?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}
