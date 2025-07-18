use p3_util::indices_arr;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::{size_of, transmute};

use crate::memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols};

pub const NUM_ALU_COLS: usize = size_of::<AluCols<u8>>();

pub const CPU_ALU_MAP: AluCols<usize> = make_col_map();


#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AluCols<T> {
     /// Whether a equals b.
     pub arg1_eq_arg2: T,

     /// Whether a is greater than b.
     pub arg1_gt_arg2: T,
 
     /// Whether a is less than b.
     pub arg1_lt_arg2: T, 
     
     /// The comparision result. gurantee to be bool
     pub res_bool:T,
}

/// Creates the column map for the CPU.
const fn make_col_map() -> AluCols<usize> {
    let indices_arr = indices_arr::<NUM_ALU_COLS>();
    unsafe { transmute::<[usize; NUM_ALU_COLS], AluCols<usize>>(indices_arr) }
}
