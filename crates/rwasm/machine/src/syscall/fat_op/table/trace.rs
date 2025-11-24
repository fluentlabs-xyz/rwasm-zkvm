use std::borrow::BorrowMut;

use crate::{
    syscall::fat_op::table::{TableInitCols, NUM_TABLE_COPY_SIZE},
    utils::pad_rows_fixed,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm::{event::TableCopyEvent, N_MAX_ELEM_SEGMENTS_BITS};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use super::TableCopyChip;

impl<F: PrimeField32> MachineAir<F> for TableCopyChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "TableCopy".to_string()
    }

    /// Generates the execution trace matrix for WebAssembly table operations.
    ///
    /// This function processes all table.init, table.fill, and table.copy events
    /// from the execution record and converts them into trace rows that will be
    /// verified by the AIR constraints.
    ///
    /// Each event may generate multiple rows (one per element operation), and
    /// the final trace is padded to a fixed size for the proving system.
    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let rows = Vec::new();
        let mut wrapped_rows = Some(rows);

        // Process all three table operation types in sequence
        for code in [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY] {
            for (_, event) in input.get_precompile_events(code) {
                // Extract the TableCopyEvent from the PrecompileEvent enum
                let event = if let PrecompileEvent::TableCopy(event) = event {
                    event
                } else {
                    unreachable!();
                };

                // Convert each event into one or more trace rows
                self.event_to_rows(event, &mut wrapped_rows, &mut Vec::new(), code);
            }
        }

        let mut rows = wrapped_rows.unwrap();
        let num_real_rows = rows.len();

        // Pad the trace to the next power of 2 for the FFT-based proving system
        pad_rows_fixed(&mut rows, || [F::zero(); NUM_TABLE_COPY_SIZE], None);

        // Convert the 2D array of rows into a flat row-major matrix
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_TABLE_COPY_SIZE)
    }

    /// Generates byte lookup dependencies for range checking.
    ///
    /// This function runs in parallel over events to collect all byte lookup
    /// operations required for range checking addresses and indices. The byte
    /// lookups are later verified against a global byte lookup table.
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = 1usize;

        for code in [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY] {
            let events = input.get_precompile_events(code);

            // Process events in parallel to collect byte lookups
            let blu_batches = events
                .par_chunks(chunk_size)
                .map(|events| {
                    let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                    events.iter().for_each(|(_, event)| {
                        let event = match event {
                            PrecompileEvent::TableCopy(event) => event,
                            _ => unreachable!(),
                        };
                        // Collect byte lookups without generating full rows
                        self.event_to_rows::<F>(event, &mut None, &mut blu, code);
                    });
                    blu
                })
                .collect::<Vec<_>>();

            // Merge all byte lookup events into the output record
            output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        }
    }

    /// Determines whether this AIR chip should be included in the proof.
    ///
    /// The chip is included if there are any table operation events in the shard,
    /// or if the execution shape explicitly requires it.
    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            // Include if any table operation events exist
            [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY]
                .iter()
                .any(|code| !shard.get_precompile_events(*code).is_empty())
        }
    }
}

impl TableCopyChip {
    /// Converts a single TableCopyEvent into one or more trace rows.
    ///
    /// Each event represents a complete table operation (init/fill/copy) that may
    /// operate on multiple elements. This function generates one row per element,
    /// with special handling for the first and last rows.
    ///
    /// # Arguments
    /// * `event` - The table operation event to convert
    /// * `rows` - Optional output buffer for trace rows (None when only collecting byte lookups)
    /// * `blu` - Byte lookup accumulator for range checking
    /// * `syscall_code` - The operation type (TABLE_INIT, TABLE_FILL, or TABLE_COPY)
    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &TableCopyEvent,
        rows: &mut Option<Vec<[F; NUM_TABLE_COPY_SIZE]>>,
        blu: &mut impl ByteRecord,
        syscall_code: SyscallCode,
    ) {
        // Determine operation type
        let is_table_init = syscall_code == SyscallCode::TABLE_INIT;
        let is_table_fill = syscall_code == SyscallCode::TABLE_FILL;
        let is_table_copy = syscall_code == SyscallCode::TABLE_COPY;

        // Generate at least one row even for zero-length operations
        let n_rows = (event.n as usize).max(1);
        let has_elements = event.n != 0;

        for idx in 0..n_rows {
            let mut row = [F::zero(); NUM_TABLE_COPY_SIZE];
            let local: &mut TableInitCols<F> = row.as_mut_slice().borrow_mut();

            // Helper flags for clarity
            let is_first_row = idx == 0;
            let is_last_row = idx == n_rows - 1;
            let is_boundary = is_first_row || is_last_row;

            // === Encode operation metadata in aux_value ===
            if is_table_fill {
                local.aux_value = F::from_canonical_u32(event.dst_table_idx);
            } else if is_table_copy {
                local.aux_value =
                    F::from_canonical_u32((event.dst_table_idx << 16) + event.src_table_idx);
            }

            // === Common fields for all operations ===
            local.shard = F::from_canonical_u32(event.shard);
            local.clk = F::from_canonical_u32(event.clk);
            local.is_table_fill = F::from_bool(is_table_fill);
            local.is_table_init = F::from_bool(is_table_init);
            local.is_table_copy = F::from_bool(is_table_copy);

            // === Set length and source reading flags ===
            if has_elements {
                local.is_non_zero_length = F::one();
                if is_table_init {
                    local.should_read_elements = F::one();
                } else if is_table_copy {
                    local.should_read_src_table = F::one();
                }
            }

            // === Mark first and last rows ===
            if is_first_row {
                local.is_first = F::one();
            }
            if is_last_row {
                local.is_last = F::one();
            }

            // === Populate stack parameters ===
            if is_first_row {
                // First row: use blu for byte lookups
                local.dst_access.populate(event.stack_access[0], blu);
                local.src_access.populate(event.stack_access[1], blu);
                local.length_access.populate(event.stack_access[2], blu);

                // Populate table sizes with byte lookups
                if is_table_copy {
                    local.table_src_size_read_access.populate(event.src_table_size_read_acess, blu);
                    local.should_read_src_table_size = F::one();
                }
                local.table_dst_size_read_access.populate(event.dst_table_size_read_acess, blu);

                // Range-check indices and stack pointer
                local.src_table_idx.populate(event.src_table_idx, blu, is_table_copy);
                local.dst_table_idx.populate(event.dst_table_idx, blu, true);
                local.sp.populate(event.sp, blu, true);
            } else {
                // Non-first rows: skip byte lookups
                local.dst_access.populate(event.stack_access[0], &mut Vec::new());
                local.src_access.populate(event.stack_access[1], &mut Vec::new());
                local.length_access.populate(event.stack_access[2], &mut Vec::new());

                // Populate table sizes without byte lookups
                if is_table_copy {
                    local
                        .table_src_size_read_access
                        .populate(event.src_table_size_read_acess, &mut Vec::new());
                }
                local
                    .table_dst_size_read_access
                    .populate(event.dst_table_size_read_acess, &mut Vec::new());

                // Populate indices without range checking
                local.src_table_idx.populate(event.src_table_idx, blu, false);
                local.dst_table_idx.populate(event.dst_table_idx, blu, false);
                local.sp.populate(event.sp, blu, false);
            }

            // === Populate and range-check addresses for current element ===
            if is_table_init {
                // table.init: source from element segment with fixed max size
                local.src_end[0] = F::from_canonical_u8(N_MAX_ELEM_SEGMENTS_BITS as u8);
                local.src_end[1] = F::from_canonical_u8((N_MAX_ELEM_SEGMENTS_BITS >> 8) as u8);
                local.src_address.populate(
                    event.s + idx as u32,
                    blu,
                    N_MAX_ELEM_SEGMENTS_BITS as u32,
                    is_boundary,
                );
            } else if is_table_copy {
                // table.copy: source from source table with dynamic size
                let src_size = event.src_table_size_read_acess.value;
                local.src_end[0] = F::from_canonical_u8(src_size as u8);
                local.src_end[1] = F::from_canonical_u8((src_size >> 8) as u8);
                local.src_address.populate(event.s + idx as u32, blu, src_size, is_boundary);
            }
            // table.fill: no source address (fills with immediate value)

            // Destination address (all operations)
            local.dst_address.populate(
                event.d + idx as u32,
                blu,
                event.dst_table_size_read_acess.value,
                is_boundary,
            );

            // === Memory read/write accesses for this element ===
            if let Some(memory_read_access) = event.memory_read_access.get(idx) {
                local.src_read_access.populate(*memory_read_access, blu);
            }

            if let Some(memory_write_access) = event.memory_write_acess.get(idx) {
                local.dst_write_access.populate(*memory_write_access, blu);
            }

            // === Add row to trace ===
            if let Some(ref mut r) = rows {
                r.push(row);
            }
        }
    }
}
