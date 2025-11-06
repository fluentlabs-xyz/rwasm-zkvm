use super::super::TableIdxCols;
use crate::{
    memory::{MemoryReadCols, MemoryWriteCols, TableAddressCols},
    operations::Range16bCols,
};
use rwasm::{
    mem_index::{SP_END, UNIT},
    N_MAX_TABLE_SIZE,
};
use sp1_derive::AlignedBorrow;

/// Total size of a TableGrow trace row in field elements.
pub const NUM_TABLE_GROW_SIZE: usize = num_table_cols();

/// Computes the size of TableGrowCols structure in field elements.
pub const fn num_table_cols() -> usize {
    size_of::<TableGrowCols<u8>>()
}

/// Column structure for WASM table.grow instruction execution trace.
///
/// Each row represents either:
/// - A complete failed or zero-delta operation (single row)
/// - One table entry initialization during successful growth (delta rows total)
///
/// The structure captures all state needed to verify correct table.grow execution,
/// including memory accesses, range checks, and control flow flags.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableGrowCols<T> {
    /// Stack pointer at the time of execution
    pub sp: StackAddressCols<T>,
    /// Shard identifier for this execution
    pub shard: T,
    /// Clock cycle when the operation executes
    pub clk: T,

    /// Index of the table being grown
    pub table_idx: TableIdxCols<T>,
    /// Destination address in the table for the current write (old_size + offset)
    pub dst_address: TableAddressCols<T>,

    /// Flag indicating the first row of a table.grow event
    pub is_first: T,
    /// Flag indicating the last row of a table.grow event
    pub is_last: T,
    /// Flag indicating non-zero delta (actual growth occurs)
    pub is_non_zero_length: T,
    /// Flag indicating a real row (not padding)
    pub is_real: T,

    // TODO(Aliaksei): find way to avoid this flags
    /// Flag to trigger table size memory write (set on last row of successful growth)
    pub should_update_table_size: T,
    /// Flag to trigger result memory write (set on first row)
    pub should_update_result: T,
    /// Flag indicating operation failure (result == u32::MAX)
    pub not_successful_result: T,

    /// Memory read of current table size
    pub table_size_read_access: MemoryReadCols<T>,
    /// Memory write of updated table size (old_size + delta)
    pub table_size_write_access: MemoryWriteCols<T>,

    /// Memory read of initialization value from stack (used to fill new entries)
    pub init_access: MemoryReadCols<T>,
    /// Memory read of delta from stack (number of entries to grow)
    pub delta_access: MemoryReadCols<T>,
    /// Range check decomposition of delta value (16-bit limbs)
    pub delta: DeltaCols<T>,

    /// Memory write of result to stack (old_size on success, u32::MAX on failure)
    pub result_write_access: MemoryWriteCols<T>,
    /// Memory write of initialization value to table at dst_address
    pub dst_write_access: MemoryWriteCols<T>,
}

/// Type alias for delta range check columns (16-bit decomposition).
pub type DeltaCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;

const SP_START: u32 = rwasm::mem_index::SP_START - UNIT;
pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;
