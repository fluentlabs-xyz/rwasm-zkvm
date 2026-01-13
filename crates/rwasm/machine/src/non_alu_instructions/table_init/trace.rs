use std::borrow::BorrowMut;

use crate::{
    non_alu_instructions::{TableInitCols, NUM_TABLE_INIT_SIZE},
    utils::pad_rows_fixed,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, TableInitEvent},
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use super::TableInitChip;
impl<F: PrimeField32> MachineAir<F> for TableInitChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "TableInit".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let rows = Vec::new();

        let mut wrapped_rows = Some(rows);
        for event in &input.table_init_events {
            self.event_to_rows(event, &mut wrapped_rows, &mut Vec::new());
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

        let blu_batches = input
            .table_init_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    self.event_to_rows::<F>(event, &mut None, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.table_init_events.is_empty()
        }
    }
}

impl TableInitChip {
    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &TableInitEvent,
        rows: &mut Option<Vec<[F; NUM_TABLE_INIT_SIZE]>>,
        blu: &mut impl ByteRecord,
    ) {
        for idx in 0..=event.n as usize {
            let mut row = [F::zero(); NUM_TABLE_INIT_SIZE];
            let local: &mut TableInitCols<F> = row.as_mut_slice().borrow_mut();

            if idx == event.n as usize && event.n > 0 {
                break;
            }

            let is_first = idx == 0;

            local.is_first = F::from_bool(is_first);

            local.src = event.s.into();

            local.sp = F::from_canonical_u32(event.sp);
            local.pc = F::from_canonical_u32(event.pc);

            local.length.populate(event.n, blu, is_first);

            local.table_idx.populate(event.table_idx, blu, is_first);

            // populate SP access
            if idx == 0 {
                local.dst_access.populate(event.dst_index_record, blu);
            } else {
                local.dst_access.populate(event.dst_index_record, &mut Vec::new());
            }

            let d = event.dst_index_record.value;

            // populate address
            if idx == 0 || idx == event.n as usize - 1 {
                local.src_address.populate(event.s + idx as u32, blu, true);
                local.dst_address.populate(d + idx as u32, blu, true);
            } else {
                local.src_address.populate(event.s + idx as u32, blu, false);
                local.dst_address.populate(d + idx as u32, blu, false);
            }

            if event.n != 0 {
                local.is_non_zero_length = F::one();
            }

            local.is_real = F::one();
            local.shard = F::from_canonical_u32(event.shard);
            local.clk = F::from_canonical_u32(event.clk);

            if let (Some(memory_read_access), Some(memory_write_access)) =
                (event.memory_read_records.get(idx), event.memory_write_records.get(idx))
            {
                local.src_read_access.populate(*memory_read_access, blu);
                local.dst_write_access.populate(*memory_write_access, blu);
            }

            if idx == event.n as usize - 1 || event.n == 0 {
                local.is_last = F::one();
            }

            if rows.as_ref().is_some() {
                rows.as_mut().unwrap().push(row);
            }
        }
    }
}
