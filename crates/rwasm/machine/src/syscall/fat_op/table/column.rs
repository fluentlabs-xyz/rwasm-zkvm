use crate::{
    memory::{ElementAddressCols, TableAddressCols},
    operations::Range16bCols,
};
use rwasm::N_MAX_TABLE_SIZE;
use sp1_derive::AlignedBorrow;

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
    pub table_idx: TableIdxCols<T>,
    pub src_access: MemoryReadCols<T>,
    pub dst_access: MemoryReadCols<T>,
    pub length_access: MemoryReadCols<T>,
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

pub type TableIdxCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;

pub type LengthCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;
