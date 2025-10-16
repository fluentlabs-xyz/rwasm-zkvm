use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

use crate::memory::{MemoryReadCols, MemoryWriteCols};

pub const NUM_TABLE_INIT_SIZE: usize = num_table_cols();
pub const fn num_table_cols() -> usize {
    size_of::<TableCols<u8>>()
}

// TODO(Aliaksei): try to place several src → dst pairs in one row
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableCols<T> {
    pub sp: T,
    pub shard: T,
    pub clk: T,
    pub table_idx: T,
    pub src_access: MemoryReadCols<T>,
    pub dst_access: MemoryReadCols<T>,
    pub length_access: MemoryReadCols<T>,
    pub src_read_access: MemoryReadCols<T>,
    pub dst_write_access: MemoryWriteCols<T>,
    pub is_first: T,
    pub is_last: T,
    pub is_non_zero_length: T,
    pub src_offset: Word<T>,
    pub dst_offset: Word<T>,
    pub is_real: T,
}
