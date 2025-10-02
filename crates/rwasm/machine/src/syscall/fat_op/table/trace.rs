use std::borrow::BorrowMut;

use crate::syscall::fat_op::table::MemoryReadCols;
use crate::syscall::fat_op::table::MemoryWriteCols;
use crate::syscall::fat_op::table::TableCols;
use crate::{syscall::fat_op::table::NUM_TABLE_INIT_SIZE, utils::pad_rows_fixed};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm::event::TableInitEvent;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent, ShaCompressEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

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
            input.fixed_log2_rows::<F, _>(self),
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

    fn local_only(&self) -> bool {
        true
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
        let mut row = [F::zero(); NUM_TABLE_INIT_SIZE];
        let main_cols: &mut TableCols<F> = row.as_mut_slice().borrow_mut();

        main_cols.clk = F::from_canonical_u32(event.clk);
        main_cols.shard = F::from_canonical_u32(event.shard);
        main_cols.table_idx = F::from_canonical_u32(event.table_idx);
        main_cols.sp = F::from_canonical_u32(event.sp);

        main_cols.is_real = F::one();

        main_cols.dst_access.populate(event.stack_access[0], blu);
        main_cols.src_access.populate(event.stack_access[1], blu);
        main_cols.length_access.populate(event.stack_access[2], blu);

        for idx in 0..event.n as usize {
            let cols = &mut main_cols.inner[idx];
            cols.is_real = F::one();
            cols.src_read_access.populate(event.memory_read_access[idx], blu);
            cols.dst_write_access.populate(event.memory_write_acess[idx], blu);
        }
        if rows.as_ref().is_some() {
            rows.as_mut().unwrap().push(row);
        }
    }
}
