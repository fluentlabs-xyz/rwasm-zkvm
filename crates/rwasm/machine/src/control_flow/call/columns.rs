use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::size_of;

use crate::operations::BabyBearWordRangeChecker;

pub const NUM_CALL_COLS: usize = size_of::<CallColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CallColumns<T> {
    /// The current program counter.
    pub pc: Word<T>,
    pub pc_range_checker: BabyBearWordRangeChecker<T>,

    /// The next program counter.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,

    pub call_sp:T,
    pub next_call_sp:T,
    pub func_ref:Word<T>,
    pub signature_id:T,
    pub table_id:T,
    pub table_idx:T,
    pub opcode:T,

     /// Call Instructions.
    pub is_call: T,
    pub is_call_internal: T,
    pub is_call_indirect: T,
    pub is_return: T,

}
