use p3_util::indices_arr;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::{size_of, transmute};

use crate::memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols};

pub const NUM_CALL_DATA_COLS: usize = size_of::<CallDataCols<u8>>();

pub const CALL_MAP: CallDataCols<usize> = make_col_map();

/// The column layout for the CPU.

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CallDataCols<T> {
    pub call_sp:T,
    pub next_call_sp:T,
    pub signature_id:T,
    pub func_ref:T,
    pub table_id:T,
    pub table_idx:T,
}

/// Creates the column map for the CPU.
const fn make_col_map() -> CallDataCols<usize> {
    let indices_arr = indices_arr::<NUM_CALL_DATA_COLS>();
    unsafe { transmute::<[usize; NUM_CALL_DATA_COLS], CallDataCols<usize>>(indices_arr) }
}
