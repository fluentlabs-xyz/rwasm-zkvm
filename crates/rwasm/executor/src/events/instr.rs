use rwasm::{
    mem::{MemoryReadRecord, MemoryWriteRecord},
    Opcode,
};
use serde::{Deserialize, Serialize};

use super::MemoryRecordEnum;

/// Alu Opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm ALU operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct AluEvent {
    /// The program counter.
    pub pc: u32,
    /// Stack pointer.
    pub sp: u32,
    /// rwasm opcode
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// u32 representation of Opcode
    pub code: u32,
}

impl AluEvent {
    /// Create a new [`AluEvent`].
    #[must_use]
    pub fn new(pc: u32, sp: u32, opcode: Opcode, a: u32, b: u32, c: u32, code: u32) -> Self {
        Self { pc, sp, opcode, a, b, c, code }
    }
}

pub type ExtendEvent = AluEvent;

/// Memory Opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm memory operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct MemInstrEvent {
    /// The shard.
    pub shard: u32,
    /// The clk.
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    /// Stack pointer.
    pub sp: u32,
    /// The Opcode
    pub opcode: Opcode,

    pub res: u32,

    /// The first operand value.
    pub arg1: u32,

    pub arg2: u32,

    /// The memory access record for memory operations.
    pub mem_access: MemoryRecordEnum,

    /// The memory access record for memory operations.
    pub mem_access_hi: Option<MemoryRecordEnum>,
}

impl MemInstrEvent {
    /// Create a new [`MemInstrEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shard: u32,
        clk: u32,
        pc: u32,
        sp: u32,
        opcode: Opcode,
        arg1: u32,
        arg2: u32,
        res: u32,

        mem_access: MemoryRecordEnum,
        mem_access_hi: Option<MemoryRecordEnum>,
    ) -> Self {
        Self { shard, clk, pc, sp, opcode, arg1, arg2, res, mem_access, mem_access_hi }
    }
}

/// Branch Opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct BranchEvent {
    /// The program counter.
    pub pc: u32,

    pub sp: u32,

    /// The next program counter.
    pub next_pc: u32,
    /// The Opcode
    pub opcode: Opcode,

    /// The first operand value.
    pub res: u32,
    /// The second operand value.
    pub arg1: u32,
    /// The third operand value.
    pub arg2: u32,
}

impl BranchEvent {
    /// Create a new [`BranchEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(pc: u32, next_pc: u32, sp: u32, opcode: Opcode, a: u32, b: u32, c: u32) -> Self {
        Self { pc, next_pc, sp, opcode, res: a, arg1: b, arg2: c }
    }
}

/// Const Opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct ConstEvent {
    /// The program counter.
    pub pc: u32,
    pub sp: u32,
    /// The Opcode
    pub opcode: Opcode,
}

/// Const Opcode Event.
///
/// This object encapsulated the information needed to prove a RISC-V branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct ParamsCheckEvent {
    pub shard: u32,
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    pub sp: u32,
    /// The Opcode
    pub opcode: Opcode,
    pub params_read_record: MemoryReadRecord,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct FuelEvent {
    pub pc: u32,
    pub sp: u32,
    /// The Opcode
    pub opcode: Opcode,
    pub arg1: u32,
    pub fuel_consumed: u64,
}

impl ConstEvent {
    /// Create a new [`ConstEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(pc: u32, sp: u32, opcode: Opcode, aux_value: u32) -> Self {
        Self { pc, sp, opcode }
    }
}

///TODO: this event is for changing the state of rwasm engine. not finished yet.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct SysStateEvent {
    /// The fuel before op
    pub fuel: u32,
    /// The fuel after op
    pub next_fuel: u32,
    /// The Opcode
    pub opcode: Opcode,
    /// maximum memory before op
    pub max_memory: u32,
    /// maximum memory after op
    pub next_max_memory: u32,
}

impl SysStateEvent {
    ///create a new system state event
    #[must_use]
    pub fn new(
        opcode: Opcode,
        fuel: u32,
        next_fuel: u32,
        max_memory: u32,
        next_max_memory: u32,
    ) -> Self {
        SysStateEvent { fuel, next_fuel, opcode, max_memory, next_max_memory }
    }
}
/// Call opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct CallEvent {
    pub shard: u32,
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The Opcode
    pub opcode: Opcode,

    /// The first operand value.
    pub sp: u32,

    pub table_idx: u32,
    pub func_index: Option<u32>,
    pub call_stack_address: u32,
    pub call_stack_access: Option<MemoryRecordEnum>,
    pub table_access: Option<MemoryReadRecord>,
    pub signature_write_record: Option<MemoryWriteRecord>,
}

impl CallEvent {
    /// Create a new [`CallEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shard: u32,
        clk: u32,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        sp: u32,
        table_idx: u32,
        func_index: Option<u32>,
        call_stack_address: u32,
        call_stack_access: Option<MemoryRecordEnum>,
        table_access: Option<MemoryReadRecord>,
        signature_write_record: Option<MemoryWriteRecord>,
    ) -> Self {
        Self {
            shard,
            clk,
            pc,
            next_pc,
            opcode,
            sp,
            table_idx,
            func_index,
            call_stack_address,
            call_stack_access,
            table_access,
            signature_write_record,
        }
    }
}

/// Alu Opcode Event.
///
/// This object encapsulated the information needed to prove a Rwasm ALU operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct I64AluEvent {
    /// The program counter.
    pub pc: u32,
    pub sp: u32,

    pub shard: u32,

    pub clk: u32,

    /// Rwasm opcode
    pub opcode: Opcode,
    /// The result value
    pub res_hi: u32,
    /// The result value's hi bits
    pub res_lo_write_record: MemoryWriteRecord,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// u32 representation of Opcode
    pub code: u32,
}

impl I64AluEvent {
    /// Create a new [`I64AluEvent`].
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        pc: u32,
        sp: u32,
        shard: u32,
        clk: u32,
        opcode: Opcode,
        res_hi: u32,
        res_lo_write_record: MemoryWriteRecord,
        b: u32,
        c: u32,
        code: u32,
    ) -> Self {
        Self { pc, sp, shard, clk, opcode, res_hi, res_lo_write_record, b, c, code }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct LocalEvent {
    pub pc: u32,
    pub sp: u32,
    pub clk: u32,
    pub shard: u32,
    pub opcode: Opcode,
    pub arg1: u32,
    pub depth_access: MemoryRecordEnum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct TableInitEvent {
    pub clk: u32,
    pub shard: u32,
    pub sp: u32,
    pub pc: u32,
    pub s: u32,
    pub n: u32,
    pub table_idx: u32,
    pub opcode: Opcode,
    pub dst_index_record: MemoryReadRecord,
    pub memory_read_records: Vec<MemoryReadRecord>,
    pub memory_write_records: Vec<MemoryWriteRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct TableGrowEvent {
    pub pc: u32,
    pub clk: u32,
    pub shard: u32,
    pub sp: u32,
    pub res: u32,
    pub delta: u32,
    pub init: u32,
    pub opcode: Opcode,
    pub dst_write_records: Vec<MemoryWriteRecord>,
    pub table_size_read_record: MemoryReadRecord,
    pub table_size_write_record: Option<MemoryWriteRecord>,
}
