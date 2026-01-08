use super::{CallChip, CallColumns, NUM_CALL_COLS};
use crate::utils::{next_power_of_two, zeroed_f_vec};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use rwasm::mem_index::TypedAddress;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, CallEvent},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;
use std::borrow::BorrowMut;

// --- Trait Implementation for MachineAir ---
// This block connects the CallChip to the SP1 machine, defining how it generates its execution
// trace.

impl<F: PrimeField32> MachineAir<F> for CallChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Call".to_string()
    }

    /// Generates the execution trace for the CallChip from a given execution record.
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

    /// Determines if this chip should be included in the proof, based on the record's shape or
    /// event presence.
    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.call_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        false
    }
}

// --- Helper Methods for CallChip ---

impl CallChip {
    /// Converts a single `CallEvent` into a row in the trace matrix.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CallEvent,
        cols: &mut CallColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        println!("^^^^^^^^^^^ {:?}", event);

        // --- 1. Populate Basic Columns ---
        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.pc = event.pc.into();
        cols.next_pc = event.next_pc.into();
        cols.sp = F::from_canonical_u32(event.sp);
        cols.aux_value = event.opcode.aux_value().into();

        // --- 2. Populate Range-Check and Memory-Related Columns ---

        // Populate range checkers for pc and next_pc to ensure they are valid addresses.
        cols.pc_range_checker.populate(cols.pc, blu);
        cols.next_pc_range_checker.populate(cols.next_pc, blu);

        println!("!!!call_stack_address: {}", event.call_stack_address);

        // Populate the call stack address. This is the pointer to the call stack frame.
        cols.call_stack_address.populate(event.call_stack_address, blu, true);

        // --- 3. Populate Columns for Indirect Calls ---

        // If this is a CallIndirect, we need to populate table access information.
        if let Some(record) = event.table_access {
            cols.table_access.populate(record, blu);
            cols.table_idx.populate(event.table_idx, blu, true);
            cols.func_index.populate(event.func_index.unwrap_or(0), blu, true);
        }

        // --- 4. Decode Opcode and Set Flags ---
        match event.opcode {
            Opcode::Call(_) => cols.is_call = F::one(),
            Opcode::CallIndirect(_) => cols.is_call_indirect = F::one(),
            Opcode::CallInternal(_) => cols.is_call_internal = F::one(),
            Opcode::Return => cols.is_return = F::one(),
            _ => unreachable!("Invalid opcode for CallChip"),
        }

        // --- 5. Handle Call Stack Access and "Fake" Return ---

        // A "real" return is any return that is not the final exit from the main function (where
        // sp=0). Real returns MUST have an associated stack access to read the return
        // address.
        cols.call_stack_access.populate(event.call_stack_access, blu);
    }
}
