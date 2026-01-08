use crate::operations::IsZeroWordOperation;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::size_of;

use crate::operations::BabyBearWordRangeChecker;

pub const NUM_BRANCH_COLS: usize = size_of::<BranchColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BranchColumns<T> {
    /// The current program counter.
    pub pc: Word<T>,
    pub pc_range_checker: BabyBearWordRangeChecker<T>,

    /// The next program counter.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,

    pub sp: T,

    pub aux_value: Word<T>,

    /// The value of the first operand.
    pub offset_value: Word<T>,
    /// The value of the second operand.
    pub op_arg1_value: Word<T>,

    pub target: Word<T>,
    /// Branch Instructions.
    pub is_br: T,
    pub is_brifnez: T,
    pub is_brifeqz: T,
    pub is_brtable: T,

    /// The is_branching column is equal to:
    ///
    /// > is_beq & a_eq_b ||
    /// > is_bne & (a_lt_b | a_gt_b) ||
    /// > (is_blt | is_bltu) & a_lt_b ||
    /// > (is_bge | is_bgeu) & (a_eq_b | a_gt_b)
    pub is_branching: T,

    // pub is_branching_non_table: T,
    /// The not branching column is equal to:
    ///
    /// > is_beq & !a_eq_b ||
    /// > is_bne & !(a_lt_b | a_gt_b) ||
    /// > (is_blt | is_bltu) & !a_lt_b ||
    /// > (is_bge | is_bgeu) & !(a_eq_b | a_gt_b)
    ///
    /// Note that we probably can do away with this column and just use !is_branching.
    /// However, the branching related constraints were auditted twice when they were part of the
    /// CPU table, so I'm preserving those columns/constraints for now.
    pub not_branching: T,

    /// Whether a equals zero.
    pub arg1_eq_zero: IsZeroWordOperation<T>,

    /// Whether a greater than target.
    pub a_lt_target: T,
}
