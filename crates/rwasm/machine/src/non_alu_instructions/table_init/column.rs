use crate::{
    memory::{ElementAddressCols, TableAddressCols},
    operations::{Range16bCols, Range8bCols},
};
use rwasm::{
    mem_index::{SP_END, UNIT},
    N_MAX_TABLES, N_MAX_TABLE_SIZE,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

use crate::memory::{MemoryReadCols, MemoryWriteCols};

pub const NUM_TABLE_INIT_SIZE: usize = num_table_cols();
pub const fn num_table_cols() -> usize {
    size_of::<TableInitCols<u8>>()
}

// TODO(Aliaksei): try to place several src → dst pairs in one row
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableInitCols<T> {
    pub pc: T,
    pub sp: T,
    pub shard: T,
    pub clk: T,
    pub table_idx: TableIdxCols<T>,
    pub dst_access: MemoryReadCols<T>,
    pub src: Word<T>,
    pub length: LengthCols<T>,
    pub src_read_access: MemoryReadCols<T>,
    pub dst_write_access: MemoryWriteCols<T>,
    pub is_first: T,
    pub is_last: T,
    pub is_non_zero_length: T,
    pub src_address: ElementAddressCols<T>,
    pub dst_address: TableAddressCols<T>,
    pub is_real: T,
}

pub type TableIdxCols<T> = Range8bCols<T, 0, N_MAX_TABLES>;

pub type LengthCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;

const SP_START: u32 = rwasm::mem_index::SP_START - 2 * UNIT;
pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;
