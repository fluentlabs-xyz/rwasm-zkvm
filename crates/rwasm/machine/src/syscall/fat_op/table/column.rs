
use sp1_derive::AlignedBorrow;


use crate::{
    memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols},

};
use rwasm::{
    mem_index::{AddressType, TABLE_ELEM_SIZE, UNIT},
    N_MAX_TABLE_SIZE,
};

pub const NUM_TABLE_INIT_SIZE:usize= num_table_cols();
pub const fn num_table_cols() -> usize {
    size_of::<TableCols<u8>>()
}


/// A set of columns for the FpAdd operation.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableCols<T> {

    pub(crate) inner:[TableSubCols<T>; N_MAX_TABLE_SIZE as usize]
}
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableSubCols<T>{
     pub sp:T,
    pub next_sp:T,
    pub is_real: T,
    pub is_first:T,
    pub is_last:T,
    pub idx:T,
    pub shard: T,
    pub clk: T,
    pub is_table_init: T,
    pub is_table_grow: T,
    pub table_idx:T,
    pub src_idx:T,
    pub dst_idx:T,
    pub length:T,
    pub src_read_access:MemoryReadCols<T>,
    pub dst_write_access: MemoryWriteCols<T>,
    pub src_access:MemoryReadCols<T>,
    pub dst_access:MemoryReadCols<T>,
    pub length_access:MemoryReadCols<T>,   
}