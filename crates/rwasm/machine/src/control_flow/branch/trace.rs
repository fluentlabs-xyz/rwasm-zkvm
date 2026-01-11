use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};

use rwasm_executor::{
    events::{BranchEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::{next_power_of_two, zeroed_f_vec};

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

impl<F: PrimeField32> MachineAir<F> for BranchChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Branch".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.branch_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.branch_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_BRANCH_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_BRANCH_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_BRANCH_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut BranchColumns<F> = row.borrow_mut();

                    if idx < input.branch_events.len() {
                        let event = &input.branch_events[idx];
                        self.event_to_row(event, cols, &mut blu);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_BRANCH_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.branch_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BranchChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &BranchEvent,
        cols: &mut BranchColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        println!("^^^^^^^^^^^ {:?}", event);

        if matches!(event.opcode, Opcode::BrTable(_)) {
            // We limited it for optimization
            assert!(event.arg1 <= u16::MAX as u32);
        }

        cols.target = match event.opcode {
            Opcode::BrTable(_) => (event.opcode.aux_value() - 1).into(),
            _ => 0.into(),
        };

        cols.offset_value = match event.opcode {
            Opcode::BrTable(_) => {
                let max_index = event.opcode.aux_value() - 1;
                let index = event.arg1;
                let normalized_index = std::cmp::min(index, max_index);
                (2 * normalized_index + 1).into()
            }
            _ => event.opcode.aux_value().into(),
        };
        let arg1_eq_zero = event.arg1 == 0;

        cols.arg1_eq_zero.populate(event.arg1);

        cols.a_lt_target = F::from_bool(event.arg1 < event.opcode.aux_value() - 1);

        let branching = match event.opcode {
            Opcode::Br(_) => true,
            Opcode::BrTable(_) => true,
            Opcode::BrIfEqz(_) => arg1_eq_zero,
            Opcode::BrIfNez(_) => !arg1_eq_zero,
            _ => unreachable!(),
        };
        match event.opcode {
            Opcode::Br(_) => cols.is_br = F::from_bool(true),
            Opcode::BrTable(_) => cols.is_brtable = F::from_bool(true),
            Opcode::BrIfEqz(_) => cols.is_brifeqz = F::from_bool(true),
            Opcode::BrIfNez(_) => cols.is_brifnez = F::from_bool(true),
            _ => unreachable!(),
        }

        cols.pc = event.pc.into();
        cols.next_pc = event.next_pc.into();

        cols.sp = F::from_canonical_u32(event.sp);

        cols.op_arg1_value = event.arg1.into();

        cols.pc_range_checker.populate(cols.pc, blu);
        cols.next_pc_range_checker.populate(cols.next_pc, blu);

        cols.aux_value = event.opcode.aux_value().into();

        if branching {
            cols.is_branching = F::one();
        } else {
            cols.not_branching = F::one();
        }
    }
}
