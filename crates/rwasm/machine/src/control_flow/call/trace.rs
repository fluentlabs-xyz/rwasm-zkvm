use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};


use rwasm_executor::{
    events::{CallEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::{next_power_of_two, zeroed_f_vec};

use super::{CallChip, CallColumns, NUM_CALL_COLS};


impl<F: PrimeField32> MachineAir<F> for CallChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Call".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.call_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.call_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_CALL_COLS);
        println!("!!! make call evnet");
        let blu_events = values
            .chunks_mut(chunk_size * NUM_CALL_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_CALL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut CallColumns<F> = row.borrow_mut();

                    if idx < input.call_events.len() {
                        let event = &input.call_events[idx];
                        self.event_to_row(event, cols, &mut blu);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_CALL_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.call_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl CallChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CallEvent,
        cols: &mut CallColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        cols.pc = event.pc.into();
        cols.next_pc =event.next_pc.into();
        cols.pc_range_checker.populate(cols.pc, blu);
        cols.next_pc_range_checker.populate(cols.next_pc, blu);
        cols.call_sp = F::from_canonical_u32(event.call_sp);
        cols.next_call_sp=F::from_canonical_u32(event.next_call_sp);
        cols.signature_id=F::from_canonical_u32(event.signature_id);
        cols.func_ref=event.func_ref.into();
        cols.table_id = F::from_canonical_u32(event.table_id);
        cols.table_idx=F::from_canonical_u32(event.table_idx);
        cols.opcode = F::from_canonical_u32(event.opcode.code());

        match event.opcode {
            Opcode::Call(_)=>{cols.is_call=F::from_bool(true);},
            Opcode::CallIndirect(_)=>{cols.is_call_indirect=F::from_bool(true);},
            Opcode::CallInternal(_)=>{cols.is_call_internal=F::from_bool(true);},
            Opcode::Return=>{cols.is_call_internal=F::from_bool(true);},
            _=>unreachable!(),
        }
        

    }
}
