#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    Silu,
    #[default]
    None,
}
