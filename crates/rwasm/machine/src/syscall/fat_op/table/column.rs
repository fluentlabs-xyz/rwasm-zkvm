use sp1_derive::AlignedBorrow;

use crate::memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols};
use rwasm::{
    mem_index::{AddressType, TABLE_ELEM_SIZE, UNIT},
    N_MAX_TABLE_SIZE,
};

pub const NUM_TABLE_INIT_SIZE: usize = num_table_cols();
pub const fn num_table_cols() -> usize {
    size_of::<TableCols<u8>>()
}

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableCols<T> {
    pub sp: T,
    pub is_real: T,
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
    pub idx: T,
    pub is_non_zero_length: T,
}
