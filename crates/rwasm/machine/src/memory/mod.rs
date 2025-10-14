mod consistency;
mod global;
mod instructions;
mod local;
mod program;
mod address;
pub use consistency::*;
pub use global::*;
pub use instructions::*;
pub use local::*;
pub use program::*;
pub use address::*;

/// The type of global/local memory chip that is being initialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryChipType {
    Initialize,
    Finalize,
}
