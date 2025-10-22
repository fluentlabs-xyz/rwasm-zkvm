use std::borrow::BorrowMut;

use crate::{
    syscall::fat_op::table::{TableCols, NUM_TABLE_INIT_SIZE},
    utils::pad_rows_fixed,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm::event::TableInitEvent;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use super::TableChip;
impl<F: PrimeField32> MachineAir<F> for TableChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Table".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        println!("generate trace Table:");
        let rows = Vec::new();

        let mut wrapped_rows = Some(rows);
        for (_, event) in input.get_precompile_events(SyscallCode::TABLE_INIT) {
            let event =
                if let PrecompileEvent::TableInit(event) = event { event } else { unreachable!() };
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
        println!("generate deps Table:");
        let events = input.get_precompile_events(SyscallCode::TABLE_INIT);
        println!("table events:{:?}", events);
        let chunk_size = 1usize;

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::TableInit(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
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
            !shard.get_precompile_events(SyscallCode::TABLE_INIT).is_empty()
        }
    }
}

impl TableChip {
    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &TableInitEvent,
        rows: &mut Option<Vec<[F; NUM_TABLE_INIT_SIZE]>>,
        blu: &mut impl ByteRecord,
    ) {
        println!("table_init_+event:{:?}", event);

        let idx = 0;

        for idx in 0..=event.n as usize {
            let mut row = [F::zero(); NUM_TABLE_INIT_SIZE];
            let local: &mut TableCols<F> = row.as_mut_slice().borrow_mut();

            if idx == event.n as usize && event.n > 0 {
                break;
            }

            // populate SP access
            if idx == 0 {
                local.is_first = F::one();
                local.dst_access.populate(event.stack_access[0], blu);
                local.src_access.populate(event.stack_access[1], blu);
                local.length_access.populate(event.stack_access[2], blu);

                local.table_idx.populate(event.table_idx, blu);
                local.length.populate(event.n, blu);
            } else {
                local.dst_access.populate(event.stack_access[0], &mut Vec::new());
                local.src_access.populate(event.stack_access[1], &mut Vec::new());
                local.length_access.populate(event.stack_access[2], &mut Vec::new());

                local.table_idx.populate_value(event.table_idx);
            }

            // populate address
            if idx == 0 || idx == event.n as usize - 1 {
                local.src_address.populate(event.s + idx as u32, blu);
                local.dst_address.populate(event.d + idx as u32, blu);
            } else {
                local.src_address.populate_value(event.s + idx as u32);
                local.dst_address.populate_value(event.d + idx as u32);
            }

            if event.n != 0 {
                local.is_non_zero_length = F::one();
            }

            local.sp = F::from_canonical_u32(event.sp);
            local.is_real = F::one();
            local.shard = F::from_canonical_u32(event.shard);
            local.clk = F::from_canonical_u32(event.clk);

            if let (Some(memory_read_access), Some(memory_write_access)) =
                (event.memory_read_access.get(idx), event.memory_write_acess.get(idx))
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
