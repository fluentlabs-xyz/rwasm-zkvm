use crate::operations::{Range16bCols, Range32bCols};
use rwasm::{
    mem_index::{SP_END, TABLE_SEG_END, TABLE_SEG_START},
    N_MAX_ELEM_SEGMENTS_BITS, N_MAX_TABLE_SIZE,
};
use rwasm_executor::SP_START;

pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;

pub type TableAddressCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;

const N_MAX_ELEM_SEGMENTS_BITS_U32: u32 = N_MAX_ELEM_SEGMENTS_BITS as u32;
pub type ElementAddressCols<T> = Range16bCols<T, 0, N_MAX_ELEM_SEGMENTS_BITS_U32>;

const TABLE_ADDR_START_HI16: u32 = TABLE_SEG_START >> 16;
const TABLE_ADDR_START_LOW16: u32 = TABLE_SEG_START as u16 as u32;
const TABLE_ADDR_END_HI16: u32 = TABLE_SEG_END >> 16;
const TABLE_ADDR_END_LOW16: u32 = TABLE_SEG_END as u16 as u32;

pub type TableAccessCol<T> = Range32bCols<
    T,
    TABLE_ADDR_START_HI16,
    TABLE_ADDR_START_LOW16,
    TABLE_ADDR_END_HI16,
    TABLE_ADDR_END_LOW16,
>;
