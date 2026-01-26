mod instruction;

pub use instruction::*;
mod alu;
pub use alu::*;
mod calldata;
pub use calldata::*;
use p3_util::indices_arr;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::{size_of, transmute};

use crate::memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols, StackAddressCols};

pub const NUM_CPU_COLS: usize = size_of::<CpuCols<u8>>();

pub const CPU_COL_MAP: CpuCols<usize> = make_col_map();

/// The column layout for the CPU.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuCols<T: Copy> {
    /// The current shard.
    pub shard: T,

    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The most significant 8 bit limb of clk.
    pub clk_8bit_limb: T,

    /// The shard to send to the opcode specific tables.  This should be 0 for all instructions
    /// other than the ecall and memory instructions.
    pub shard_to_send: T,
    /// The clk to send to the opcode specific tables.  This should be 0 for all instructions other
    /// than the ecall and memory instructions.
    pub clk_to_send: T,

    /// The program counter value.
    pub pc: T,

    /// The expected next program counter value.
    pub next_pc: T,

    pub sp: StackAddressCols<T>,
    pub next_sp: StackAddressCols<T>,

    pub op_arg2_sp: StackAddressCols<T>,

    /// Columns related to the instruction.
    pub instruction: InstructionCols<T>,

    // /// Columns related to the call data
    // pub call_data: CallDataCols<T>,
    ///Alu cols:
    pub alu_cols: AluCols<T>,

    /// Whether this is a memory instruction.
    pub is_memory: T,

    /// Whether this is a syscall instruction.
    pub is_syscall: T,

    pub is_complex_opcode: T,

    /// Whether this is a halt instruction.
    pub is_halt: T,

    /// The number of extra cycles to add to the clk for a syscall instruction.
    pub num_extra_cycles: T,

    /// Operand values, either from registers or immediate values.
    pub op_res_access: MemoryReadWriteCols<T>,
    pub op_arg1_access: MemoryReadCols<T>,
    pub op_arg2_access: MemoryReadCols<T>,
}

impl<T: Copy> CpuCols<T> {
    /// Gets the value of the first operand.
    pub fn op_res_val(&self) -> Word<T> {
        *self.op_res_access.value()
    }

    /// Gets the value of the second operand.
    pub fn op_arg1_val(&self) -> Word<T> {
        *self.op_arg1_access.value()
    }

    /// Gets the value of the third operand.
    pub fn op_arg2_val(&self) -> Word<T> {
        *self.op_arg2_access.value()
    }
}

/// Creates the column map for the CPU.
const fn make_col_map() -> CpuCols<usize> {
    let indices_arr = indices_arr::<NUM_CPU_COLS>();
    unsafe { transmute::<[usize; NUM_CPU_COLS], CpuCols<usize>>(indices_arr) }
}
