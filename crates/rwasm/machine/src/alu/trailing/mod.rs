use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    utils::pad_rows_fixed,
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::Opcode::{I32Clz, I32Ctz};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_TRAILING_COLS: usize = size_of::<TrailingCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct TrailingCols<T> {
    pub pc: T,
    pub a: Word<T>,
    pub b_bytes: [T; 4],
    // Unified per-op halves:
    //  - CTZ rows:  [ctz_low16,  ctz_high16]
    //  - CLZ rows:  [clz_high16, clz_low16]
    pub half_word_z: [T; 2],
    pub is_ctz: T,
    pub is_clz: T,
    // Minimal boolean flags (no inverse witnesses)
    pub ctz_low_is_16: T,  // used only on CTZ rows
    pub clz_high_is_16: T, // used only on CLZ rows
}

#[derive(Default)]
pub struct TrailingChip;

impl TrailingChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &rwasm_executor::events::AluEvent,
        cols: &mut TrailingCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.a = event.a.into();
        cols.is_ctz = F::from_bool(event.opcode == I32Ctz);
        cols.is_clz = F::from_bool(event.opcode == I32Clz);

        let b_val = event.b;
        let b_bytes = b_val.to_le_bytes();
        for i in 0..4 {
            cols.b_bytes[i] = F::from_canonical_u8(b_bytes[i]);
        }

        // halves
        let low_half: u16 = (b_val & 0xFFFF) as u16;
        let high_half: u16 = ((b_val >> 16) & 0xFFFF) as u16;
        let ctz_low16 = low_half.trailing_zeros();
        let ctz_high16 = high_half.trailing_zeros();
        let clz_low16 = low_half.leading_zeros();
        let clz_high16 = high_half.leading_zeros();

        if cols.is_ctz == F::one() {
            // CTZ rows
            cols.half_word_z =
                [F::from_canonical_u32(ctz_low16), F::from_canonical_u32(ctz_high16)];
            cols.ctz_low_is_16 = F::from_bool(ctz_low16 == 16);
            cols.clz_high_is_16 = F::zero();
            // byte lookups: low then high
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16CTZ,
                ctz_low16 as u16,
                0,
                b_bytes[1],
                b_bytes[0],
            ));
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16CTZ,
                ctz_high16 as u16,
                0,
                b_bytes[3],
                b_bytes[2],
            ));
        } else {
            // CLZ rows
            cols.half_word_z =
                [F::from_canonical_u32(clz_high16), F::from_canonical_u32(clz_low16)];
            cols.clz_high_is_16 = F::from_bool(clz_high16 == 16);
            cols.ctz_low_is_16 = F::zero();
            // byte lookups: high then low
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16CLZ,
                clz_high16 as u16,
                0,
                b_bytes[3],
                b_bytes[2],
            ));
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16CLZ,
                clz_low16 as u16,
                0,
                b_bytes[1],
                b_bytes[0],
            ));
        }
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

        builder.assert_bool(local.is_ctz);
        builder.assert_bool(local.is_clz);
        // At most one of CTZ or CLZ can be active in a row.
        builder.assert_zero(local.is_ctz * local.is_clz);
        let is_real = local.is_ctz + local.is_clz;

        // Send gated byte lookups via unified halves.
        // CTZ (low then high)
        builder.send_byte(
            ByteOpcode::U16CTZ.as_field::<AB::F>(),
            local.half_word_z[0],
            local.b_bytes[1],
            local.b_bytes[0],
            local.is_ctz,
        );
        builder.send_byte(
            ByteOpcode::U16CTZ.as_field::<AB::F>(),
            local.half_word_z[1],
            local.b_bytes[3],
            local.b_bytes[2],
            local.is_ctz,
        );
        // CLZ (high then low)
        builder.send_byte(
            ByteOpcode::U16CLZ.as_field::<AB::F>(),
            local.half_word_z[0],
            local.b_bytes[3],
            local.b_bytes[2],
            local.is_clz,
        );
        builder.send_byte(
            ByteOpcode::U16CLZ.as_field::<AB::F>(),
            local.half_word_z[1],
            local.b_bytes[1],
            local.b_bytes[0],
            local.is_clz,
        );

        // Reconstruct the 32-bit operand `b`.
        let reconstructed_b = Word([
            local.b_bytes[0].into(),
            local.b_bytes[1].into(),
            local.b_bytes[2].into(),
            local.b_bytes[3].into(),
        ]);

        let sixteen = AB::Expr::from_canonical_u32(16);

        // Minimal boolean constraints (no inverse witnesses)
        builder.assert_bool(local.ctz_low_is_16);
        builder.assert_bool(local.clz_high_is_16);

        // If a flag is 1, the corresponding half must equal 16
        builder
            .when(local.is_ctz)
            .assert_zero((local.half_word_z[0] - sixteen.clone()) * local.ctz_low_is_16);
        builder
            .when(local.is_clz)
            .assert_zero((local.half_word_z[0] - sixteen.clone()) * local.clz_high_is_16);

        // Unified result formulas: z0 + flag*(16 + z1 - z0)
        let a_ctz = local.half_word_z[0] +
            local.ctz_low_is_16 * (sixteen.clone() + local.half_word_z[1] - local.half_word_z[0]);
        let a_clz = local.half_word_z[0] +
            local.clz_high_is_16 * (sixteen + local.half_word_z[1] - local.half_word_z[0]);

        builder.when(local.is_ctz).assert_word_eq(local.a, Word::extend_expr::<AB>(a_ctz));
        builder.when(local.is_clz).assert_word_eq(local.a, Word::extend_expr::<AB>(a_clz));

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            local.is_ctz * AB::Expr::from_canonical_u32(I32Ctz.code()) +
                local.is_clz * AB::Expr::from_canonical_u32(I32Clz.code()),
            local.a,
            reconstructed_b,
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

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.trailing_events.iter() {
            let mut row = [F::zero(); NUM_TRAILING_COLS];
            let cols: &mut TrailingCols<F> = row.as_mut_slice().borrow_mut();
            self.event_to_row(event, cols, output);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_TRAILING_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_TRAILING_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use rayon::{iter::ParallelIterator, slice::ParallelSlice};
        let chunk_size = std::cmp::max(input.trailing_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .trailing_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_TRAILING_COLS];
                    let cols: &mut TrailingCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }
    fn included(&self, shard: &Self::Record) -> bool {
        !shard.trailing_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
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
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rwasm_executor::{events::AluEvent, ExecutionRecord, Opcode, Program};
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();
        let b: u32 = 0x137_137;
        shard.trailing_events = vec![
            AluEvent::new(0, Opcode::I32Ctz, b.trailing_zeros(), b, 0, Opcode::I32Ctz.code()),
            AluEvent::new(0, Opcode::I32Clz, b.leading_zeros(), b, 0, Opcode::I32Clz.code()),
        ];
        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.width(), NUM_TRAILING_COLS);
        assert_eq!(output.byte_lookups.len(), 4);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        let samples = vec![
            0b00000000_00000000_00000000_00000000u32,
            0b11111111_11111111_11111111_11111111u32,
            0b00000000_00000000_00000000_00000001u32,
            0b10000000_00000000_00000000_00000000u32,
            0b00000000_11111111_00000000_11111111u32,
        ];

        for b in samples.into_iter() {
            shard.trailing_events.push(AluEvent::new(
                0,
                Opcode::I32Ctz,
                b.trailing_zeros(),
                b,
                0,
                Opcode::I32Ctz.code(),
            ));
            shard.trailing_events.push(AluEvent::new(
                0,
                Opcode::I32Clz,
                b.leading_zeros(),
                b,
                0,
                Opcode::I32Clz.code(),
            ));
        }

        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof =
            prove::<BabyBearPoseidon2, TrailingChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
    #[test]
    fn test_malicious_popcnt_ctz() {
        use core::borrow::BorrowMut;
        let config = BabyBearPoseidon2::new();
        let challenger = config.challenger();
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Create a valid event.
        let op_b = 0b01001101u32;
        let op_c = 0b00001101u32;
        let opcode = Opcode::I32Ctz;

        let program = Program::from_instrs(vec![
            Opcode::I32Const(op_b.into()),
            Opcode::I32Const(op_c.into()),
            opcode,
        ]);

        let stdin = SP1Stdin::new();
        // Define a malicious action that forges the trace *after* it's generated.
        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            // First, generate the honest traces.
            let mut traces = prover.generate_traces(record);
            let chip = chip_name!(TrailingChip, BabyBear);

            // Find the TrailingChip's trace and forge it.
            if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                if trace.height() > 0 {
                    // TrailingChip's internal sum constraint will pass.
                    let row = trace.row_mut(0);
                    let cols: &mut TrailingCols<BabyBear> = row.borrow_mut();

                    // Keep byte-lookup inputs intact so cross-table lookups remain consistent.
                    // Instead, forge the local witness to trigger a chip-level constraint failure.
                    cols.ctz_low_is_16 = BabyBear::one();
                }
            }

            traces
        };
        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip = chip_name!(TrailingChip, BabyBear);
        println!("result.is_err =  {}", result.is_err());
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
    }
}
