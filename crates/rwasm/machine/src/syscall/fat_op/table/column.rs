use crate::operations::{DynamicLE16bCols, Range16bCols, Range8bCols};
use rwasm::{
    mem_index::{SP_END, UNIT},
    N_MAX_TABLES,
};
use sp1_derive::AlignedBorrow;

use crate::memory::{MemoryReadCols, MemoryWriteCols};

pub const NUM_TABLE_INIT_SIZE: usize = num_table_cols();
pub const fn num_table_cols() -> usize {
    size_of::<TableInitCols<u8>>()
}

// TODO(Aliaksei): try to place several src → dst pairs in one row
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableInitCols<T> {
    pub sp: StackAddressCols<T>,
    pub shard: T,
    pub clk: T,
    pub src_table_idx: TableIdxCols<T>,
    pub dst_table_idx: TableIdxCols<T>,
    pub src_access: MemoryReadCols<T>,
    pub dst_access: MemoryReadCols<T>,
    pub length_access: MemoryReadCols<T>,
    pub src_read_access: MemoryReadCols<T>,
    pub dst_write_access: MemoryWriteCols<T>,
    pub is_first: T,
    pub is_last: T,
    pub is_non_zero_length: T,
    pub src_address: DynamicSrcAddressCols<T>,
    pub src_end: [T; 2],
    pub dst_address: DynamicDstAddressCols<T>,
    pub table_src_size_read_access: MemoryReadCols<T>,
    pub table_dst_size_read_access: MemoryReadCols<T>,
    pub is_table_init: T,
    pub is_table_fill: T,
    pub is_table_copy: T,
    pub aux_value: T,
    pub should_read_elements: T,
    pub should_read_src_table: T,
    pub should_read_src_table_size: T,
}

pub type TableIdxCols<T> = Range8bCols<T, 0, N_MAX_TABLES>;

const SP_START: u32 = rwasm::mem_index::SP_START - 2 * UNIT;
pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;

pub type DynamicDstAddressCols<T> = DynamicLE16bCols<T>;

pub type DynamicSrcAddressCols<T> = DynamicLE16bCols<T>;
