use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_POPCNT_COLS: usize = size_of::<PopcntCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct PopcntCols<T> {
    pub pc: T,
    pub b_low_weight: T,
    pub b_high_weight: T,
    pub b: Word<T>,
    pub is_real: T,
}

#[derive(Default)]
pub struct PopcntChip;

impl PopcntChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut PopcntCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.is_real = F::one();

        cols.b = event.b.into();

        let b_low = event.b as u16;
        let b_high = event.b >> 16;

        let b_low_weight = b_low.count_ones();
        let b_high_weight = b_high.count_ones();

        cols.b_low_weight = F::from_canonical_u32(b_low_weight);
        cols.b_high_weight = F::from_canonical_u32(b_high_weight);

        let b = event.b.to_le_bytes();

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U16POPCNT,
            a1: b_low_weight as u16,
            a2: 0,
            b: b[1],
            c: b[0],
        });

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U16POPCNT,
            a1: b_high_weight as u16,
            a2: 0,
            b: b[3],
            c: b[2],
        });
    }
}

impl<AB> Air<AB> for PopcntChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PopcntCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U16POPCNT as u32),
            local.b_low_weight,
            local.b[1],
            local.b[0],
            local.is_real,
        );

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U16POPCNT as u32),
            local.b_high_weight,
            local.b[3],
            local.b[2],
            local.is_real,
        );

        let a = local.b_low_weight + local.b_high_weight;

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Popcnt.code()),
            Word::extend_expr::<AB>(a),
            local.b,
            Word::<AB::Expr>::default(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
    }
}

impl<F> BaseAir<F> for PopcntChip {
    fn width(&self) -> usize {
        NUM_POPCNT_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for PopcntChip {
    type Record = ExecutionRecord;
    type Program = Program;
    fn name(&self) -> String {
        "Popcnt".to_string()
    }
    // Only operates on individual rows without cross-row interactions
    fn local_only(&self) -> bool {
        true
    }
    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows: Vec<[F; NUM_POPCNT_COLS]> = vec![];
        // Convert each popcnt event into a trace row
        for event in input.popcnt_events.iter() {
            let mut row = [F::zero(); NUM_POPCNT_COLS];
            let cols: &mut PopcntCols<F> = row.as_mut_slice().borrow_mut();
            // Pass output to event_to_row so it can add byte lookups
            self.event_to_row(event, cols, output);
            rows.push(row);
        }
        // Pad the trace to the nearest power of 2
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_POPCNT_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_POPCNT_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.popcnt_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .popcnt_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_POPCNT_COLS];
                    let cols: &mut PopcntCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        // Only include chip if there are popcnt events to process
        !shard.popcnt_events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::{PopcntChip, NUM_POPCNT_COLS};
    use crate::{
        alu::PopcntCols,
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use core::borrow::BorrowMut;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};
    use rwasm_executor::{
        events::{AluEvent, MemoryRecordEnum},
        ExecutionRecord, Opcode, Program,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let b: u32 = 0x137_137;
        let a = b.count_ones();
        shard.popcnt_events =
            vec![AluEvent::new(0, Opcode::I32Popcnt, a, b, 0, Opcode::I32Popcnt.code())];
        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        assert_eq!(trace.width(), NUM_POPCNT_COLS);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut events: Vec<AluEvent> = Vec::new();
        // Test with various bit patterns
        let samples = vec![
            0b00000000_00000000_00000000_00000000u32,
            0b11111111_11111111_11111111_11111111u32,
            0b00000000_00000000_00000000_00000001u32,
            0b10000000_00000000_00000000_00000000u32,
            0b01111111_11111111_11111111_11111111u32,
            0b01010101_01010101_01010101_01010101u32,
            0b10101010_10101010_10101010_10101010u32,
            0b00001111_00001111_00001111_00001111u32,
            0b11110000_11110000_11110000_11110000u32,
            0b00000000_11111111_00000000_11111111u32,
        ];
        // Create events for each sample value
        for b in samples.into_iter() {
            let a = b.count_ones();
            events.push(AluEvent::new(0, Opcode::I32Popcnt, a, b, 0, Opcode::I32Popcnt.code()));
        }
        let mut shard = ExecutionRecord::default();
        shard.popcnt_events = events;
        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, PopcntChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_popcnt() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;
        const NUM_TESTS: usize = 1;

        let mut rng = thread_rng();
        let opcode = Opcode::I32Popcnt;
        for _ in 0..NUM_TESTS {
            let op_b: u32 = rng.gen();
            let correct: u32 = op_b.count_ones();

            let op_a = correct.wrapping_add(1); // wrong value
            assert_ne!(op_a, correct);

            let program = Program::from_instrs(vec![
                Opcode::I32Const(524u32.into()),
                Opcode::I32Const(3u32.into()),
                Opcode::I32Const(22u32.into()),
                Opcode::I32Const(op_b.into()),
                opcode,
            ]);
            let stdin = SP1Stdin::new();

            let malicious = move |prover: &P, record: &mut ExecutionRecord| {
                let mut malicious_record = record.clone();

                // also forge the PopcntChip chip’s `a` column to match the bad value
                let chip = chip_name!(PopcntChip, BabyBear);
                let mut traces = prover.generate_traces(&malicious_record);
                if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                    let row = trace.row_mut(0);
                    let row: &mut PopcntCols<BabyBear> = row.borrow_mut();

                    row.b_low_weight = BabyBear::from_canonical_u32((op_a as u16).count_ones());
                    row.b_high_weight = BabyBear::from_canonical_u32((op_a >> 16).count_ones());
                }

                traces
            };

            let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
            assert!(result.is_err());
        }
    }
}
