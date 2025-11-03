use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    utils::pad_rows_fixed,
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm_executor::{
    events::{AluEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_POPCNT_COLS: usize = size_of::<PopcntCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct PopcntCols<T> {
    pub pc: T,
    pub a: Word<T>,
    pub b: [T; 32],
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
        _: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.is_real = F::one();
        for i in 0..32 {
            cols.b[i] = F::from_canonical_u32((event.b >> i) & 1);
        }
        cols.a = event.a.into();
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
        let mut b = Word::<AB::Expr>::default();
        // Process each byte consisting of 8 bits
        for byte_idx in 0..4 {
            for bit_idx in 0..8 {
                let global_bit_idx = byte_idx * 8 + bit_idx;
                let bit = local.b[global_bit_idx];
                // Assert that each bit is binary when the row is real
                builder.when(local.is_real).assert_bool(bit);
                // Accumulate bits to reconstruct the byte value
                b[byte_idx] += bit * AB::Expr::from_canonical_u32(1 << bit_idx);
            }
        }
        // Sum all bits to compute the popcount result
        let a = Word::<AB::Expr>::extend_expr::<AB>(
            local.b.iter().copied().fold(AB::Expr::zero(), |acc, var| acc + var),
        );
        builder.when(local.is_real).assert_word_eq(a, local.a);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Popcnt.code()),
            local.a,
            b,
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
        // Pad to ~1000 rows
        /*  events.resize_with(1000, || {
            AluEvent::new(0, Opcode::I32Popcnt, 0, 0, 0, Opcode::I32Popcnt.code())
        });*/
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

                // forge CPU result cell + memory write
                if malicious_record.cpu_events.len() > 4 {
                    malicious_record.cpu_events[4].res = op_a as u32;
                    if let Some(MemoryRecordEnum::Write(mut write_record)) =
                        malicious_record.cpu_events[4].res_record
                    {
                        write_record.value = op_a as u32;
                    }
                }

                // also forge the PopcntChip chip’s `a` column to match the bad value
                let chip = chip_name!(PopcntChip, BabyBear);
                let mut traces = prover.generate_traces(&malicious_record);
                if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                    let row = trace.row_mut(0);
                    let row: &mut PopcntCols<BabyBear> = row.borrow_mut();
                    row.a = op_a.into();
                }

                traces
            };

            let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
            let chip = chip_name!(PopcntChip, BabyBear);
            assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
        }
    }
    #[test]
    fn test_malicious_popcnt_event() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        // Create a malicious event: popcnt(13) is 3, but we claim it's 5.
        let b = 0b1101;
        let malicious_a = 5;
        let event =
            AluEvent::new(0, Opcode::I32Popcnt, malicious_a, b, 0, Opcode::I32Popcnt.code());

        let mut shard = ExecutionRecord::default();
        shard.popcnt_events.push(event);

        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // The trace itself is generated correctly based on `b`, but the `a` in the event is a lie.
        // The prover will prove a statement about a trace derived from `b`, but the verifier
        // should check this against the public inputs (the event), which contains the incorrect
        // `a`. This inconsistency should cause verification to fail.
        let proof = prove::<BabyBearPoseidon2, PopcntChip>(&config, &chip, &mut challenger, trace);
        let mut verifier_challenger = config.challenger();
        let result = verify(&config, &chip, &mut verifier_challenger, &proof);
        assert!(result.is_err(), "verification should fail for malicious popcnt result");
    }
}
