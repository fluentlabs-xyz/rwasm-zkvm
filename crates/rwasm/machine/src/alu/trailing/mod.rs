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
use rwasm::Opcode::{I32Clz, I32Ctz};
use rwasm_executor::{
    events::{AluEvent, ByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_TRAILING_COLS: usize = size_of::<TrailingCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct TrailingCols<T> {
    pub pc: T,
    pub a: Word<T>,
    pub b: [T; 32],
    pub is_ctz: T,
    pub is_clz: T,
    // Degree-2 prefix products to avoid 32-fold products in AIR
    pub p_ctz: [T; 33],
    pub p_clz: [T; 33],
}

impl<T> Default for TrailingCols<T>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self {
            pc: T::default(),
            a: Word::<T>::default(),
            b: [T::default(); 32],
            is_ctz: T::default(),
            is_clz: T::default(),
            p_ctz: [T::default(); 33],
            p_clz: [T::default(); 33],
        }
    }
}

#[derive(Default)]
pub struct TrailingChip;

impl TrailingChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut TrailingCols<F>,
        _: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.is_clz = F::from_bool(event.opcode == I32Clz);
        cols.is_ctz = F::from_bool(event.opcode == I32Ctz);
        for i in 0..32 {
            cols.b[i] = F::from_canonical_u32((event.b >> i) & 1);
        }
        // Initialize CTZ prefixes: p_ctz[0] = 1; p_ctz[i+1] = p_ctz[i] * (1 - b[i])
        cols.p_ctz[0] = F::one();
        for i in 0..32 {
            let inv = F::one() - cols.b[i];
            cols.p_ctz[i + 1] = cols.p_ctz[i] * inv;
        }
        // Initialize CLZ prefixes from MSB: p_clz[0] = 1; p_clz[i+1] = p_clz[i] * (1 - b[31 - i])
        cols.p_clz[0] = F::one();
        for i in 0..32 {
            let inv = F::one() - cols.b[31 - i];
            cols.p_clz[i + 1] = cols.p_clz[i] * inv;
        }
        cols.a = event.a.into();
    }
}

impl<AB> Air<AB> for TrailingChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &TrailingCols<AB::Var> = (*local).borrow();
        let mut b = Word::<AB::Expr>::default();

        builder.assert_bool(local.is_ctz);
        builder.assert_bool(local.is_clz);
        let is_real = local.is_ctz + local.is_clz;
        builder.assert_bool(is_real.clone());
        // Ensure exactly one of is_ctz or is_clz is set when the row is real.
        builder.when(is_real.clone()).assert_zero(local.is_ctz * local.is_clz);
        // Process each byte consisting of 8 bits
        for byte_idx in 0..4 {
            for bit_idx in 0..8 {
                let global_bit_idx = byte_idx * 8 + bit_idx;
                let bit = local.b[global_bit_idx];
                // Assert that each bit is binary when the row is real
                builder.when(is_real.clone()).assert_bool(bit);
                // Accumulate bits to reconstruct the byte value
                b[byte_idx] += bit * AB::Expr::from_canonical_u32(1 << bit_idx);
            }
        }

        // Degree-2 prefix recurrences for CTZ and CLZ
        builder.when(is_real.clone()).assert_eq(local.p_ctz[0], AB::Expr::one());
        for i in 0..32 {
            let bit = local.b[i];
            builder
                .when(is_real.clone())
                .assert_eq(local.p_ctz[i + 1], local.p_ctz[i] * (AB::Expr::one() - bit));
        }
        builder.when(is_real.clone()).assert_eq(local.p_clz[0], AB::Expr::one());
        for i in 0..32 {
            let bit = local.b[31 - i];
            builder
                .when(is_real.clone())
                .assert_eq(local.p_clz[i + 1], local.p_clz[i] * (AB::Expr::one() - bit));
        }

        // Compute CTZ and CLZ from prefixes
        let mut ctz_sum = AB::Expr::zero();
        for i in 0..32 {
            ctz_sum += local.p_ctz[i] * (AB::Expr::one() - local.b[i]);
        }

        let mut clz_sum = AB::Expr::zero();
        for i in 0..32 {
            clz_sum += local.p_clz[i] * (AB::Expr::one() - local.b[31 - i]);
        }

        // Constrain results separately per opcode to avoid multiplying by selector bits.
        // This keeps constraint degree low and reduces the required FRI domain size.
        let mut ctz_word = Word::<AB::Expr>::default();
        ctz_word[0] = ctz_sum.clone();
        builder.when(local.is_ctz).assert_word_eq(ctz_word, local.a);

        let mut clz_word = Word::<AB::Expr>::default();
        clz_word[0] = clz_sum.clone();
        builder.when(local.is_clz).assert_word_eq(clz_word, local.a);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            local.is_ctz * AB::Expr::from_canonical_u32(I32Ctz.code()) +
                local.is_clz * AB::Expr::from_canonical_u32(I32Clz.code()),
            local.a,
            b,
            Word::<AB::Expr>::default(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );
    }
}

impl<F> BaseAir<F> for TrailingChip {
    fn width(&self) -> usize {
        NUM_TRAILING_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for TrailingChip {
    type Record = ExecutionRecord;
    type Program = Program;
    fn name(&self) -> String {
        "Trailing".to_string()
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
        let mut rows: Vec<[F; NUM_TRAILING_COLS]> = vec![];
        // Convert each ctz event into a trace row
        for event in input.trailing_events.iter() {
            let mut row = [F::zero(); NUM_TRAILING_COLS];
            let cols: &mut TrailingCols<F> = row.as_mut_slice().borrow_mut();
            // Pass output to event_to_row so it can add byte lookups
            self.event_to_row(event, cols, output);
            rows.push(row);
        }
        let mut log_rows = input.fixed_log2_rows::<F, _>(self);
        // Ensure enough rows to satisfy FRI domain for degree-2 constraints.
        // Empirically, a floor of 13 (8192 rows) avoids lde<domain panics across configs.
        if log_rows < Some(13) {
            log_rows = Some(13);
        }
        pad_rows_fixed(&mut rows, || [F::zero(); NUM_TRAILING_COLS], log_rows);
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_TRAILING_COLS)
    }
    fn included(&self, shard: &Self::Record) -> bool {
        // Only include chip if there are ctz events to process
        !shard.trailing_events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::{TrailingChip, NUM_TRAILING_COLS};
    use crate::{
        alu::TrailingCols,
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
        shard.trailing_events = vec![
            AluEvent::new(0, Opcode::I32Ctz, b.trailing_zeros(), b, 0, Opcode::I32Ctz.code()),
            AluEvent::new(0, Opcode::I32Clz, b.leading_zeros(), b, 0, Opcode::I32Clz.code()),
        ];
        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        assert_eq!(trace.width(), NUM_TRAILING_COLS);
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
        for b in samples.clone().into_iter() {
            let a = b.trailing_zeros();
            events.push(AluEvent::new(0, Opcode::I32Ctz, a, b, 0, Opcode::I32Ctz.code()));
        }
        for b in samples.into_iter() {
            let a = b.leading_zeros();
            events.push(AluEvent::new(0, Opcode::I32Clz, a, b, 0, Opcode::I32Clz.code()));
        }
        // No external padding needed; TrailingChip pads internally to a safe minimum (2^13 rows).

        let mut shard = ExecutionRecord::default();
        shard.trailing_events = events;
        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof =
            prove::<BabyBearPoseidon2, TrailingChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_trailing() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;
        const NUM_TESTS: usize = 1;

        let mut rng = thread_rng();
        let opcodes = [Opcode::I32Ctz, Opcode::I32Clz];
        for opcode in opcodes {
            for _ in 0..NUM_TESTS {
                let op_b: u32 = rng.gen();
                let correct: u32 = if opcode == Opcode::I32Ctz {
                    op_b.trailing_zeros()
                } else {
                    op_b.leading_zeros()
                };

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

                    // also forge the chip’s `a` column to match the bad value
                    let chip = chip_name!(TrailingChip, BabyBear);
                    let mut traces = prover.generate_traces(&malicious_record);
                    if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                        let row = trace.row_mut(0);
                        let row: &mut TrailingCols<BabyBear> = row.borrow_mut();
                        row.a = op_a.into();
                    }

                    traces
                };

                let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
                let chip = chip_name!(TrailingChip, BabyBear);
                assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
            }
        }
    }
    #[test]
    fn test_malicious_ctz_event() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        // Create a malicious event: ctz(13) is 0, but we claim it's 5.
        let b = 0b1101;
        let malicious_a = 5;
        let event = AluEvent::new(0, Opcode::I32Ctz, malicious_a, b, 0, Opcode::I32Ctz.code());

        let mut shard = ExecutionRecord::default();
        shard.trailing_events.push(event);

        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let proof =
            prove::<BabyBearPoseidon2, TrailingChip>(&config, &chip, &mut challenger, trace);
        let mut verifier_challenger = config.challenger();
        let result = verify(&config, &chip, &mut verifier_challenger, &proof);
        assert!(result.is_err(), "verification should fail for malicious ctz result");
    }
}
