use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};

use rwasm::{mem_index::TypedAddress, N_MAX_TABLE_SIZE};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, CallEvent},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;

use crate::{
    shape::Shapeable,
    utils::{next_power_of_two, zeroed_f_vec},
};

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
        println!("!!! make call event");
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
                        self.event_to_row(event, cols, input.shard(), &mut blu);
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
        shard: u32,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        println!("callevent:{:?}", event);
        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.pc = event.pc.into();
        cols.next_pc = event.next_pc.into();
        cols.pc_range_checker.populate(cols.pc, blu);
        cols.next_pc_range_checker.populate(cols.next_pc, blu);

        cols.opcode = F::from_canonical_u32(event.opcode.code());
        cols.call_sp = F::from_canonical_u32(event.call_sp);
        let call_sp_addr = TypedAddress::FuncFrame(event.call_sp);
        cols.call_sp_addr.populate(call_sp_addr.to_virtual_addr(), blu);
        cols.next_call_sp = F::from_canonical_u32(event.next_call_sp);
        let next_call_sp_addr = TypedAddress::FuncFrame(event.next_call_sp);
        cols.next_call_sp_addr.populate(next_call_sp_addr.to_virtual_addr(), blu);

        cols.signature_id = F::from_canonical_u32(event.signature_id);

        cols.func_ref = F::from_canonical_u32(event.func_ref);
        cols.table_id = F::from_canonical_u32(event.table_id);
        cols.table_idx = F::from_canonical_u32(event.table_idx);
        if let Some(record) = event.table_access {
            cols.table_access.populate(record, blu);
            let table_addr =
                TypedAddress::Table(event.table_id * N_MAX_TABLE_SIZE + event.table_idx);
            // cols.table_access_addr.populate(table_addr.to_virtual_addr(), blu);
        }
        println!("opcode  for call: {}", event.opcode.code());
        cols.opcode_aux_val = event.opcode.aux_value().into();
        println!("col.opcode:{:?}", cols.opcode);
        match event.opcode {
            Opcode::Call(_) => {
                cols.is_call = F::from_bool(true);
            }
            Opcode::CallIndirect(_) => {
                cols.is_call_indirect = F::from_bool(true);
            }
            Opcode::CallInternal(_) => {
                cols.is_call_internal = F::from_bool(true);
            }
            Opcode::Return => {
                cols.is_return = F::from_bool(true);
            }
            _ => unreachable!(),
        }
        if !(event.opcode == Opcode::Return && event.call_sp == 0) {
            assert!(event.call_stack_access.is_some());
            cols.call_stack_access.populate(event.call_stack_access.unwrap(), blu);
        } else {
            cols.not_real_return = F::from_bool(true);
        }
    }
}
