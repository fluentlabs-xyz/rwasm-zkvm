use crate::operations::Range16bCols;
use rwasm::{mem_index::SP_END, N_MAX_ELEM_SEGMENTS_BITS, N_MAX_TABLE_SIZE};
use rwasm_executor::SP_START;

pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;

pub type TableAddressCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;

const N_MAX_ELEM_SEGMENTS_BITS_U32: u32 = N_MAX_ELEM_SEGMENTS_BITS as u32;
pub type ElementAddressCols<T> = Range16bCols<T, 0, N_MAX_ELEM_SEGMENTS_BITS_U32>;
