use std::borrow::BorrowMut;

use crate::{
    syscall::fat_op::table_grow::column::{TableGrowCols, NUM_TABLE_GROW_SIZE},
    utils::pad_rows_fixed,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm::event::TableGrowEvent;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use super::TableGrowChip;

impl<F: PrimeField32> MachineAir<F> for TableGrowChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "TableGrow".to_string()
    }

    /// Generates the execution trace for WASM table.grow operations.
    ///
    /// Processes each TableGrowEvent from the execution record and converts it into
    /// trace rows. The trace is then padded to match the required circuit size.
    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let rows = Vec::new();

        let mut wrapped_rows = Some(rows);
        // Extract and process all TABLE_GROW events from the execution record
        for (_, event) in input.get_precompile_events(SyscallCode::TABLE_GROW) {
            let event =
                if let PrecompileEvent::TableGrow(event) = event { event } else { unreachable!() };
            // Convert the event into trace rows
            self.event_to_rows(event, &mut wrapped_rows, &mut Vec::new());
        }
        let mut rows = wrapped_rows.unwrap();

        let num_real_rows = rows.len();

        // Pad the trace with zero rows to match the required size
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_TABLE_GROW_SIZE],
            // TODO (Aliaksei): find a way to provide `size_log2` from the shape
            None,
        );

        // Convert the flat vector into a row-major matrix
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_TABLE_GROW_SIZE)
    }

    /// Generates byte lookup dependencies for range checks and memory operations.
    ///
    /// Processes TABLE_GROW events in parallel to collect all byte lookup events
    /// required for constraint verification without generating actual trace rows.
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        println!("generate deps Table:");
        let events = input.get_precompile_events(SyscallCode::TABLE_GROW);
        println!("table events:{:?}", events);
        let chunk_size = 1usize;

        // Process events in parallel chunks to collect byte lookup events efficiently
        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::TableGrow(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    // Collect byte lookups without generating trace rows
                    self.event_to_rows::<F>(event, &mut None, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        // Merge all byte lookup events into the output record
        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    /// Determines whether this chip is included in the given shard.
    ///
    /// Returns true if the shard contains any TABLE_GROW events that need processing.
    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::TABLE_GROW).is_empty()
        }
    }
}

impl TableGrowChip {
    /// Creates a base trace row with common fields populated for all execution cases.
    ///
    /// Initializes stack accesses, table index, table size read, and execution metadata
    /// (shard, clock, stack pointer). This base row is then specialized for different
    /// execution scenarios (failure, zero-delta, or successful growth).
    ///
    /// # Arguments
    /// * `populate_range_check` - Whether to perform byte lookups for range checks
    /// * `blu` - Byte lookup record for collecting range check events
    /// * `event` - The TableGrowEvent containing execution data
    fn create_base_row<F: PrimeField32>(
        populate_range_check: bool,
        blu: &mut impl ByteRecord,
        event: &TableGrowEvent,
    ) -> [F; NUM_TABLE_GROW_SIZE] {
        let mut row = [F::zero(); NUM_TABLE_GROW_SIZE];
        let local: &mut TableGrowCols<F> = row.as_mut_slice().borrow_mut();

        // Populate stack reads for initialization value and delta
        local.init_access.populate(event.stack_access[0], blu);
        local.delta_access.populate(event.stack_access[1], blu);

        // Populate table index with range check

        local.table_idx.populate(event.table_idx, blu, populate_range_check);
        local.sp.populate(event.sp, blu, populate_range_check);

        // Populate the current table size read
        local.table_size_read_access.populate(event.table_size_read_acess, blu);

        // Set execution context metadata
        local.is_real = F::one();
        local.shard = F::from_canonical_u32(event.shard);
        local.clk = F::from_canonical_u32(event.clk);

        row
    }

    /// Converts a TableGrowEvent into execution trace rows based on the operation outcome.
    ///
    /// Handles three distinct execution paths:
    /// 1. **Failure case** (result == u32::MAX): Operation failed due to insufficient table
    ///    capacity. Generates a single row marking the failure.
    /// 2. **Zero-delta case** (delta == 0): Request to grow by zero elements. Generates a single
    ///    row with no actual table modifications.
    /// 3. **Success case** (delta > 0): Table successfully grows by delta elements. Generates delta
    ///    rows, one for each new table entry being initialized.
    ///
    /// # Arguments
    /// * `event` - The TableGrowEvent to process
    /// * `rows` - Optional vector to collect generated rows (None when only collecting
    ///   dependencies)
    /// * `blu` - Byte lookup record for range checks and memory operations
    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &TableGrowEvent,
        rows: &mut Option<Vec<[F; NUM_TABLE_GROW_SIZE]>>,
        blu: &mut impl ByteRecord,
    ) {
        // Helper closure to conditionally push rows when trace generation is enabled
        let mut push_row = |row: [F; NUM_TABLE_GROW_SIZE]| {
            if rows.as_ref().is_some() {
                rows.as_mut().unwrap().push(row);
            }
        };

        // Case 1: Failure - table cannot grow (exceeds maximum size or other constraints)
        // Returns u32::MAX to indicate failure per WASM specification
        if event.result_write_access.value == u32::MAX {
            let mut row = Self::create_base_row(true, blu, event);
            let local: &mut TableGrowCols<F> = row.as_mut_slice().borrow_mut();

            local.is_first = F::one();
            local.is_last = F::one();
            local.should_update_result = F::one();
            local.result_write_access.populate(event.result_write_access, blu);
            local.not_successful_result = F::one();
            local.delta.populate(event.delta, blu, true);

            push_row(row);
            return;
        }

        // Case 2: Zero-delta - request to grow by zero elements (essentially a no-op)
        // Still needs to write the old table size as the result
        if event.delta == 0 {
            let mut row = Self::create_base_row(true, blu, event);
            let local: &mut TableGrowCols<F> = row.as_mut_slice().borrow_mut();

            local.is_first = F::one();
            local.is_last = F::one();

            push_row(row);
            return;
        }

        // Case 3: Successful growth - table grows by delta elements
        // Each iteration handles one new table entry initialization
        for idx in 0..event.delta as usize {
            // Create base row, performing byte lookups only on the first iteration
            let mut row = if idx == 0 {
                Self::create_base_row(true, blu, event)
            } else {
                Self::create_base_row(false, &mut Vec::new(), event)
            };

            let local: &mut TableGrowCols<F> = row.as_mut_slice().borrow_mut();

            // First row: mark as first and write the result (old table size) to stack
            if idx == 0 {
                local.is_first = F::one();
                local.should_update_result = F::one();
                local.result_write_access.populate(event.result_write_access, blu);
                local.delta.populate(event.delta, blu, true);
            } else {
                // Subsequent rows: reuse values without byte lookups to avoid duplicates
                local.result_write_access.populate(event.result_write_access, &mut Vec::new());
                local.table_idx.populate(event.table_idx, blu, false);
            }

            // Mark that the operation has non-zero length
            local.is_non_zero_length = F::one();

            // Calculate destination address in table (old_size + idx)
            // Perform range checks only on first and last addresses to optimize
            if idx == 0 || idx == event.delta as usize - 1 {
                local.dst_address.populate(
                    event.table_size_read_acess.value + idx as u32,
                    blu,
                    true,
                );
            } else {
                local.dst_address.populate(
                    event.table_size_read_acess.value + idx as u32,
                    blu,
                    false,
                );
            }

            // Write initialization value to this table entry
            if let Some(memory_write_access) = event.memory_write_acess.get(idx) {
                local.dst_write_access.populate(*memory_write_access, blu);
            }

            // Last row: update table size in memory to reflect the growth
            if idx == event.delta as usize - 1 {
                local.is_last = F::one();
                local.should_update_table_size = F::one();
                local.table_size_write_access.populate(event.table_size_write_acess, blu);
            }

            push_row(row);
        }
    }
}
