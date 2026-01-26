use crate::{
    control_flow::TableIdxCols,
    memory::{MemoryReadCols, MemoryWriteCols, TableAddressCols},
    operations::Range16bCols,
};
use rwasm::N_MAX_TABLE_SIZE;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

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
    pub pc: T,
    /// Stack pointer at the time of execution
    pub sp: T,
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

    /// Range check decomposition of delta value (16-bit limbs)
    pub delta: DeltaCols<T>,

    pub res: Word<T>,

    pub init: Word<T>,

    /// Memory write of initialization value to table at dst_address
    pub dst_write_access: MemoryWriteCols<T>,
}

/// Type alias for delta range check columns (16-bit decomposition).
pub type DeltaCols<T> = Range16bCols<T, 0, N_MAX_TABLE_SIZE>;
