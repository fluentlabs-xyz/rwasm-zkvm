use std::borrow::BorrowMut;

use crate::{
    syscall::fat_op::table::{TableInitCols, NUM_TABLE_INIT_SIZE},
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

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let rows = Vec::new();

        let mut wrapped_rows = Some(rows);
        for code in [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY] {
            for (_, event) in input.get_precompile_events(code) {
                let event = if let PrecompileEvent::TableCopy(event) = event {
                    event
                } else {
                    unreachable!()
                };
                self.event_to_rows(event, &mut wrapped_rows, &mut Vec::new(), code);
            }
        }

        let mut rows = wrapped_rows.unwrap();

        let num_real_rows = rows.len();

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_TABLE_INIT_SIZE],
            // TODO (Aliaksei): find a way to provide `size_log2` from the shape
            None,
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_TABLE_INIT_SIZE)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = 1usize;

        for code in [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY] {
            let events = input.get_precompile_events(code);

            let blu_batches = events
                .par_chunks(chunk_size)
                .map(|events| {
                    let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                    events.iter().for_each(|(_, event)| {
                        let event = match event {
                            PrecompileEvent::TableCopy(event) => event,
                            _ => unreachable!(),
                        };
                        self.event_to_rows::<F>(event, &mut None, &mut blu, code);
                    });
                    blu
                })
                .collect::<Vec<_>>();

            output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            [SyscallCode::TABLE_INIT, SyscallCode::TABLE_FILL, SyscallCode::TABLE_COPY]
                .iter()
                .any(|code| !shard.get_precompile_events(*code).is_empty())
        }
    }
}

impl TableCopyChip {
    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &TableCopyEvent,
        rows: &mut Option<Vec<[F; NUM_TABLE_INIT_SIZE]>>,
        blu: &mut impl ByteRecord,
        syscall_code: SyscallCode,
    ) {
        let is_table_init = syscall_code == SyscallCode::TABLE_INIT;
        let is_table_fill = syscall_code == SyscallCode::TABLE_FILL;
        let is_table_copy = syscall_code == SyscallCode::TABLE_COPY;

        for idx in 0..=event.n as usize {
            let mut row = [F::zero(); NUM_TABLE_INIT_SIZE];
            let local: &mut TableInitCols<F> = row.as_mut_slice().borrow_mut();

            if idx == event.n as usize && event.n > 0 {
                break;
            }

            if is_table_fill {
                local.aux_value = F::from_canonical_u32(event.dst_table_idx);
            } else if is_table_copy {
                local.aux_value =
                    F::from_canonical_u32((event.dst_table_idx << 16) + event.src_table_idx)
            }

            // Basic fields (moved up to avoid duplication)
            local.shard = F::from_canonical_u32(event.shard);
            local.clk = F::from_canonical_u32(event.clk);
            local.is_table_fill = F::from_bool(is_table_fill);
            local.is_table_init = F::from_bool(is_table_init);
            local.is_table_copy = F::from_bool(is_table_copy);

            // Length flags
            if event.n != 0 {
                local.is_non_zero_length = F::one();
                if is_table_init {
                    local.should_read_elements = F::one();
                } else if is_table_copy {
                    local.should_read_src_table = F::one();
                }
            }

            // Stack access population
            if idx == 0 {
                local.is_first = F::one();
                local.dst_access.populate(event.stack_access[0], blu);
                local.src_access.populate(event.stack_access[1], blu);
                local.length_access.populate(event.stack_access[2], blu);
                local.src_table_idx.populate(event.src_table_idx, blu, is_table_copy);
                local.dst_table_idx.populate(event.dst_table_idx, blu, true);
                local.sp.populate(event.sp, blu, true);
                if is_table_copy {
                    local.table_src_size_read_access.populate(event.src_table_size_read_acess, blu);
                    local.should_read_src_table_size = F::one();
                }
                local.table_dst_size_read_access.populate(event.dst_table_size_read_acess, blu);
            } else {
                local.dst_access.populate(event.stack_access[0], &mut Vec::new());
                local.src_access.populate(event.stack_access[1], &mut Vec::new());
                local.length_access.populate(event.stack_access[2], &mut Vec::new());
                if is_table_copy {
                    local
                        .table_src_size_read_access
                        .populate(event.src_table_size_read_acess, &mut Vec::new());
                }
                local
                    .table_dst_size_read_access
                    .populate(event.dst_table_size_read_acess, &mut Vec::new());
                local.src_table_idx.populate(event.src_table_idx, blu, false);
                local.dst_table_idx.populate(event.dst_table_idx, blu, false);
                local.sp.populate(event.sp, blu, false);
            }

            // Address population (simplified duplication)
            let is_boundary = idx == 0 || idx == event.n as usize - 1;

            if is_table_init {
                local.src_end[0] = F::from_canonical_u8(N_MAX_ELEM_SEGMENTS_BITS as u8);
                local.src_end[1] = F::from_canonical_u8((N_MAX_ELEM_SEGMENTS_BITS >> 8) as u8);

                local.src_address.populate(
                    event.s + idx as u32,
                    blu,
                    N_MAX_ELEM_SEGMENTS_BITS as u32,
                    is_boundary,
                );
            } else if is_table_copy {
                local.src_end[0] =
                    F::from_canonical_u8(event.src_table_size_read_acess.value as u8);
                local.src_end[1] =
                    F::from_canonical_u8((event.src_table_size_read_acess.value >> 8) as u8);

                local.src_address.populate(
                    event.s + idx as u32,
                    blu,
                    event.src_table_size_read_acess.value,
                    is_boundary,
                );
            }

            // dst_address: always increments
            local.dst_address.populate(
                event.d + idx as u32,
                blu,
                event.dst_table_size_read_acess.value,
                is_boundary,
            );

            // Memory accesses
            if let Some(memory_read_access) = event.memory_read_access.get(idx) {
                local.src_read_access.populate(*memory_read_access, blu);
            }

            if let Some(memory_write_access) = event.memory_write_acess.get(idx) {
                local.dst_write_access.populate(*memory_write_access, blu);
            }

            if idx == event.n as usize - 1 || event.n == 0 {
                local.is_last = F::one();
            }

            if let Some(ref mut r) = rows {
                r.push(row);
            }
        }
    }
}
