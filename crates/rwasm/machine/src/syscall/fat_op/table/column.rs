use crate::{
    memory::{MemoryReadCols, MemoryWriteCols},
    operations::{DynamicLE16bCols, Range16bCols, Range8bCols},
};
use rwasm::{
    mem_index::{SP_END, UNIT},
    N_MAX_TABLES,
};
use sp1_derive::AlignedBorrow;

/// Total number of columns in the trace table for table operations
pub const NUM_TABLE_COPY_SIZE: usize = num_table_cols();

/// Calculates the size of TableInitCols structure in bytes
pub const fn num_table_cols() -> usize {
    size_of::<TableInitCols<u8>>()
}

// TODO(Aliaksei): Optimize by placing multiple src/dst pairs in a single row

/// Column layout for WebAssembly table operations (table.init, table.fill, table.copy)
/// Each row represents one element operation in the trace execution
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct TableInitCols<T> {
    // Stack and execution context
    /// Stack pointer at time of operation
    pub sp: StackAddressCols<T>,
    /// Shard identifier for parallel execution
    pub shard: T,
    /// Clock cycle when operation executes
    pub clk: T,

    // Table identifiers
    /// Source table index (used by table.init and table.copy)
    pub src_table_idx: TableIdxCols<T>,
    /// Destination table index (used by all three operations)
    pub dst_table_idx: TableIdxCols<T>,

    // Stack memory accesses for operation parameters
    /// Read access for destination offset parameter (from stack)
    pub dst_access: MemoryReadCols<T>,
    /// Read access for source offset parameter (from stack)
    pub src_access: MemoryReadCols<T>,
    /// Read access for length/count parameter (from stack)
    pub length_access: MemoryReadCols<T>,

    // Element memory accesses
    /// Read access for source element value
    pub src_read_access: MemoryReadCols<T>,
    /// Write access for destination element value
    pub dst_write_access: MemoryWriteCols<T>,

    // Event boundary markers
    /// Flag indicating first row of an operation event (1 = first, 0 = not first)
    pub is_first: T,
    /// Flag indicating last row of an operation event (1 = last, 0 = not last)
    pub is_last: T,

    // Operation characteristics
    /// Flag indicating non-zero length operation (1 = length > 0, 0 = length == 0)
    pub is_non_zero_length: T,

    // Address tracking with dynamic range checking
    /// Source address being read from (increments during multi-element operations)
    pub src_address: DynamicSrcAddressCols<T>,
    /// Upper bound for source address range checking (stored as 2-byte value)
    pub src_end: [T; 2],
    /// Destination address being written to (increments during multi-element operations)
    pub dst_address: DynamicDstAddressCols<T>,

    // Table size accesses
    /// Read access for source table size (used by table.copy)
    pub table_src_size_read_access: MemoryReadCols<T>,
    /// Read access for destination table size (used by all operations)
    pub table_dst_size_read_access: MemoryReadCols<T>,

    // Operation type flags (exactly one must be 1)
    /// Flag for table.init operation (copies from element segment to table)
    pub is_table_init: T,
    /// Flag for table.fill operation (fills table range with single value)
    pub is_table_fill: T,
    /// Flag for table.copy operation (copies between two table regions)
    pub is_table_copy: T,

    // Auxiliary and control flags
    /// Auxiliary value encoding operation-specific data:
    /// - table.init: 0 (all elements in segment 0)
    /// - table.fill: dst_table_idx
    /// - table.copy: (dst_table_idx << 16) | src_table_idx
    pub aux_value: T,

    /// Flag to read from element segment (table.init with length > 0)
    pub should_read_elements: T,
    /// Flag to read from source table (table.copy with length > 0)
    pub should_read_src_table: T,
    /// Flag to read source table size (table.copy only)
    pub should_read_src_table_size: T,
}

/// Type alias for table index columns with range [0, NMAXTABLES)
pub type TableIdxCols<T> = Range8bCols<T, 0, N_MAX_TABLES>;

/// Stack pointer valid range: [SPEND, SPSTART]
const SP_START: u32 = rwasm::mem_index::SP_START - 2 * UNIT;
pub type StackAddressCols<T> = Range16bCols<T, SP_END, SP_START>;

/// Type alias for destination address with dynamic range checking
pub type DynamicDstAddressCols<T> = DynamicLE16bCols<T>;

/// Type alias for source address with dynamic range checking
pub type DynamicSrcAddressCols<T> = DynamicLE16bCols<T>;
