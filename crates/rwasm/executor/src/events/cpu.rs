use rwasm::mem_index::TypedAddress;
use serde::{Deserialize, Serialize};

use super::memory::MemoryRecordEnum;

/// CPU Event.
///
/// This object encapsulates the information needed to prove a CPU operation. This includes its
/// shard, opcode, operands, and other relevant information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuEvent {
    /// The clock cycle.
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The stack pointer,
    pub sp: u32,
    /// The next stack pointer,
    pub next_sp: u32,
    /// The first operand memory record.
    pub res_record: Option<MemoryRecordEnum>,
    /// The second operand memory record.
    pub arg1_record: Option<MemoryRecordEnum>,
    /// The third operand memory record.
    pub arg2_record: Option<MemoryRecordEnum>,
    pub exit_code: u32,
}
