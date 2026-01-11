use rwasm::{N_MAX_TABLES, N_MAX_TABLE_SIZE};
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::size_of;

use crate::{
    memory::{CallStackAddressCols, MemoryReadCols, MemoryReadWriteCols},
    operations::{BabyBearWordRangeChecker, Range16bCols, Range8bCols},
};

pub const NUM_CALL_COLS: usize = size_of::<CallColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CallColumns<T> {
    // --- System State ---
    /// The shard identifier (execution chunk ID) for the current instruction.
    pub shard: T,
    /// The clock cycle counter.
    pub clk: T,

    // --- Program Counter (PC) Logic ---
    /// The current program counter (address of the instruction being executed).
    pub pc: Word<T>,
    /// Range checker to verify that `pc` fits within valid bounds (BabyBear field limits).
    pub pc_range_checker: BabyBearWordRangeChecker<T>,

    /// The next program counter (target address for jump or sequential execution).
    pub next_pc: Word<T>,
    /// Range checker for `next_pc`.
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,

    // --- Stack & Execution State ---
    /// The current stack pointer (value stack, not call stack).
    pub sp: T,

    /// Auxiliary value, often used for immediate arguments or temporary storage.
    pub aux_value: Word<T>,

    // --- Call Stack Management ---
    /// Columns related to the address of the call stack pointer (where we push/pop return
    /// addresses).
    pub call_stack_address: CallStackAddressCols<T>,

    /// Columns for reading/writing the return address to/from the call stack memory.
    pub call_stack_access: MemoryReadWriteCols<T>,

    // --- Indirect Call Support (Table lookups) ---
    /// Columns for accessing the function table (used in `call_indirect`).
    pub table_access: MemoryReadCols<T>,

    // TODO: add table size check
    /// Columns for storing/checking the Table Index (which table is being accessed).
    pub table_idx: TableIdxCols<T>,

    /// Columns for storing/checking the Function Index (element index within the table).
    pub func_index: FuncIndex<T>,

    // --- Control Flow Flags (Selectors) ---
    // These flags are mutually exclusive (except for `not_real_return` which modifies
    // `is_return`).
    /// Selector: The instruction is a direct `call`.
    pub is_call: T,
    /// Selector: The instruction is an internal call (used for optimized standard library
    /// functions).
    pub is_call_internal: T,
    /// Selector: The instruction is an indirect call (`call_indirect`).
    pub is_call_indirect: T,
    /// Selector: The instruction is a `return`.
    pub is_return: T,

    /// Flag indicating a "fake" return at the end of execution (exit from main).
    /// Used to disable stack reads when the stack is empty.
    pub is_main_return: T,
}

pub type TableIdxCols<T> = Range8bCols<T, 0, N_MAX_TABLES>;

pub type FuncIndex<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;
