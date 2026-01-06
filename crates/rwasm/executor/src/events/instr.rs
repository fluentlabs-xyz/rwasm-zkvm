use rwasm::Opcode;
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
    /// riscv opcode
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
    pub fn new(pc: u32, opcode: Opcode, a: u32, b: u32, c: u32, code: u32) -> Self {
        Self { pc, opcode, a, b, c, code }
    }
}

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
    /// The Opcode
    pub opcode: Opcode,

    /// The first operand value.
    pub raw_addr: u32,
    /// The second operand value.
    pub offset: u32,
    /// The third operand value.
    pub res: u32,
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
        opcode: Opcode,
        raw_addr: u32,
        offset: u32,
        res: u32,

        mem_access: MemoryRecordEnum,
        mem_access_hi: Option<MemoryRecordEnum>,
    ) -> Self {
        Self { shard, clk, pc, opcode, raw_addr, offset, res, mem_access, mem_access_hi }
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
    pub fn new(pc: u32, next_pc: u32, opcode: Opcode, a: u32, b: u32, c: u32) -> Self {
        Self { pc, next_pc, opcode, res: a, arg1: b, arg2: c }
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
    /// The Opcode
    pub opcode: Opcode,
    /// The value
    pub value: u32,
}

impl ConstEvent {
    /// Create a new [`ConstEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(pc: u32, opcode: Opcode, value: u32) -> Self {
        Self { pc, opcode, value }
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
    pub call_sp: u32,
    /// The second operand value.
    pub next_call_sp: u32,
    /// The third operand value.
    pub signature_id: u32,
    pub func_ref: u32,

    pub table_id: u32,
    pub table_idx: u32,
    pub call_stack_access: Option<MemoryRecordEnum>,
    pub table_access: Option<MemoryRecordEnum>,
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
        call_sp: u32,
        next_call_sp: u32,
        signature_id: u32,
        func_ref: u32,
        table_id: u32,
        table_idx: u32,
        call_stack_access: Option<MemoryRecordEnum>,
        table_access: Option<MemoryRecordEnum>,
    ) -> Self {
        Self {
            shard,
            clk,
            pc,
            next_pc,
            opcode,
            call_sp,
            next_call_sp,
            signature_id,
            func_ref,
            table_id,
            table_idx,
            call_stack_access,
            table_access,
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
    /// riscv opcode
    pub opcode: Opcode,
    /// The result value
    pub a_lo: u32,
    /// The result value's hi bits
    pub a_hi: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// u32 representation of Opcode
    pub code: u32,

    pub res_hi_addr: u32,

    pub res_hi_access: Option<MemoryRecordEnum>,
}

impl I64AluEvent {
    /// Create a new [`I64AluEvent`].
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        pc: u32,
        opcode: Opcode,
        a: u32,
        a_hi: u32,
        b: u32,
        c: u32,
        code: u32,
        res_hi_addr: u32,
        res_hi_access: Option<MemoryRecordEnum>,
    ) -> Self {
        Self { pc, opcode, a_lo: a, a_hi, b, c, code, res_hi_addr, res_hi_access }
    }
}

///Fuel Opcode Event.
///
/// This object encapsulated the information needed to prove fuel consumption instructions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct FuelEvent {
    /// The shard.
    pub shard: u32,
    /// The clk.
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// rwasm opcode
    pub opcode: Opcode,
    /// The fuel before op
    pub fuel: u64,
    /// The fuel after op
    pub next_fuel: u64,
    /// The amount of fuel to consume
    pub to_consume_fuel: u32,

    pub fuel_consumed_low_record: MemoryRecordEnum,
    pub fuel_consumed_high_record: MemoryRecordEnum,
    pub next_consumed_fuel_low_record: MemoryRecordEnum,
    pub next_consumed_fuel_high_record: MemoryRecordEnum,
}
