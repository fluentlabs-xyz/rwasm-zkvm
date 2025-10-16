/// The number of local memory entries per row of the memory local chip.
pub const NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC: usize = 4;
pub type MemoryRecord = rwasm::mem::MemoryRecord;
pub type MemoryReadRecord = rwasm::mem::MemoryReadRecord;
pub type MemoryWriteRecord = rwasm::mem::MemoryWriteRecord;

pub type MemoryRecordEnum = rwasm::mem::MemoryRecordEnum;
pub type MemoryInitializeFinalizeEvent = rwasm::mem::MemoryInitializeFinalizeEvent;

pub type MemoryLocalEvent = rwasm::mem::MemoryLocalEvent;
