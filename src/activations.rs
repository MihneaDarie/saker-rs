#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    Relu,
    LeakyRelu(f32),
    Silu,
    #[default]
    None,
}
