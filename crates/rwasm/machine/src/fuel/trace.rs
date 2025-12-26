use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};

use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, EmptyByteRecord, FuelEvent},
    ExecutionRecord, Opcode, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::{next_power_of_two, zeroed_f_vec};

use super::{FuelChip, FuelColumns, NUM_FUEL_COLS};

impl<F: PrimeField32> MachineAir<F> for FuelChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Fuel".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Calculate total rows needed for both event types
        let nb_rows = input.fuel_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_FUEL_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        // Parallel generation of trace rows
        values.chunks_mut(chunk_size * NUM_FUEL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_FUEL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut FuelColumns<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = EmptyByteRecord;
                        let event = &input.fuel_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_FUEL_COLS)
    }
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max((input.branch_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.branch_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_FUEL_COLS);
        println!("!!! make fuel event");
        let blu_events = values
            .chunks_mut(chunk_size * NUM_FUEL_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_FUEL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut FuelColumns<F> = row.borrow_mut();

                    if idx < input.fuel_events.len() {
                        let event = &input.fuel_events[idx];
                        self.event_to_row(event, cols, &mut blu);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());
    }
    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.fuel_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl FuelChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &FuelEvent,
        cols: &mut FuelColumns<F>,
        blu: &mut impl ByteRecord,
    ) {
        // Populate the clk and pc columns.
        cols.clk = F::from_canonical_u32(event.clk);
        cols.shard = F::from_canonical_u32(event.shard);
        cols.pc = event.pc.into();
        cols.next_pc = event.next_pc.into();

        // Populate the fuel consumed columns.
        cols.fuel_consumed_low_record.populate(event.fuel_consumed_low_record, blu);
        cols.fuel_consumed_high_record.populate(event.fuel_consumed_high_record, blu);
        cols.next_consumed_fuel_low_record.populate(event.next_consumed_fuel_low_record, blu);
        cols.next_consumed_fuel_high_record.populate(event.next_consumed_fuel_high_record, blu);

        cols.to_consume_fuel = event.to_consume_fuel.into();
        let is_carryed = {
            let fuel_low = event.fuel as u32;
            fuel_low.checked_add(event.to_consume_fuel).is_none()
        };
        cols.is_carryed = F::from_bool(is_carryed);
        // Set the selector columns.
        match event.opcode {
            Opcode::ConsumeFuel(_) => {
                cols.is_consume_fuel = F::one();
            }
            Opcode::ConsumeFuelStack => {
                cols.is_consume_fuel_stack = F::one();
            }
            _ => panic!("Invalid opcode for FuelEvent: {:?}", event.opcode),
        };
    }
}
