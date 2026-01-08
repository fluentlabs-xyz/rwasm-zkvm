use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::Opcode::{I32Clz, I32Ctz, I32Popcnt};
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
    pub sp: T,
    pub a: T,
    pub b: Word<T>,
    // Unified per-op halves:
    //  - CTZ rows:  [ctz_low16,  ctz_high16]
    //  - CLZ rows:  [clz_high16, clz_low16]
    pub half_word_z: [T; 2],
    pub is_ctz: T,
    pub is_clz: T,
    pub is_popcnt: T,
    // Minimal boolean flag
    pub z0_is_16: T,        // 1 iff half_word_z[0] == 16 (CTZ: low16, CLZ: high16)
    pub z0_minus_16_inv: T, // inverse witness for isZero(half_word_z[0] - 16) on CTZ/CLZ rows
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
        cols.sp = F::from_canonical_u32(event.sp);
        cols.a = F::from_canonical_u32(event.a);
        cols.is_ctz = F::from_bool(event.opcode == I32Ctz);
        cols.is_clz = F::from_bool(event.opcode == I32Clz);
        cols.is_popcnt = F::from_bool(event.opcode == I32Popcnt);

        let b_val = event.b;
        let b_bytes = b_val.to_le_bytes();
        cols.b = event.b.into();

        // halves
        let low_half: u16 = b_val as u16;
        let high_half: u16 = (b_val >> 16) as u16;
        let ctz_low16 = low_half.trailing_zeros();
        let ctz_high16 = high_half.trailing_zeros();
        let clz_low16 = low_half.leading_zeros();
        let clz_high16 = high_half.leading_zeros();
        if event.opcode == I32Ctz {
            // CTZ rows
            cols.half_word_z =
                [F::from_canonical_u32(ctz_low16), F::from_canonical_u32(ctz_high16)];
            cols.z0_is_16 = F::from_bool(ctz_low16 == 16);
            if ctz_low16 == 16 {
                cols.z0_minus_16_inv = F::zero();
            } else {
                let diff = F::from_canonical_u32(ctz_low16) - F::from_canonical_u32(16);
                cols.z0_minus_16_inv = diff.inverse();
            }
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
        } else if event.opcode == I32Clz {
            // CLZ rows
            cols.half_word_z =
                [F::from_canonical_u32(clz_high16), F::from_canonical_u32(clz_low16)];
            cols.z0_is_16 = F::from_bool(clz_high16 == 16);
            if clz_high16 == 16 {
                cols.z0_minus_16_inv = F::zero();
            } else {
                let diff = F::from_canonical_u32(clz_high16) - F::from_canonical_u32(16);
                cols.z0_minus_16_inv = diff.inverse();
            }
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
        } else {
            // POPCNT rows
            let pop_low16 = low_half.count_ones();
            let pop_high16 = high_half.count_ones();
            cols.half_word_z =
                [F::from_canonical_u32(pop_low16), F::from_canonical_u32(pop_high16)];
            cols.z0_is_16 = F::zero(); // unused for popcnt
            cols.z0_minus_16_inv = F::zero();

            // byte lookups: low then high
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16POPCNT,
                pop_low16 as u16,
                0,
                b_bytes[1],
                b_bytes[0],
            ));
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::U16POPCNT,
                pop_high16 as u16,
                0,
                b_bytes[3],
                b_bytes[2],
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
        builder.assert_bool(local.is_popcnt);
        // CTZ or CLZ can be active in a row.
        let is_real = local.is_ctz + local.is_clz + local.is_popcnt;
        builder.assert_bool(is_real.clone());

        // Send gated byte lookups via unified halves.
        // CTZ (low then high)
        builder.send_byte(
            ByteOpcode::U16CTZ.as_field::<AB::F>(),
            local.half_word_z[0],
            local.b.0[1],
            local.b.0[0],
            local.is_ctz,
        );
        builder.send_byte(
            ByteOpcode::U16CTZ.as_field::<AB::F>(),
            local.half_word_z[1],
            local.b.0[3],
            local.b.0[2],
            local.is_ctz,
        );
        // CLZ (high then low)
        builder.send_byte(
            ByteOpcode::U16CLZ.as_field::<AB::F>(),
            local.half_word_z[0],
            local.b.0[3],
            local.b.0[2],
            local.is_clz,
        );
        builder.send_byte(
            ByteOpcode::U16CLZ.as_field::<AB::F>(),
            local.half_word_z[1],
            local.b.0[1],
            local.b.0[0],
            local.is_clz,
        );
        // POPCNT (low then high)
        builder.send_byte(
            ByteOpcode::U16POPCNT.as_field::<AB::F>(),
            local.half_word_z[0],
            local.b.0[1],
            local.b.0[0],
            local.is_popcnt,
        );
        builder.send_byte(
            ByteOpcode::U16POPCNT.as_field::<AB::F>(),
            local.half_word_z[1],
            local.b.0[3],
            local.b.0[2],
            local.is_popcnt,
        );

        let sixteen = AB::Expr::from_canonical_u32(16);
        let one = AB::Expr::one();
        let is_ctz_or_clz = local.is_ctz + local.is_clz;

        // Flag is boolean.
        builder.assert_bool(local.z0_is_16);

        // isZero gadget for "z0_is_16 <=> (half_word_z[0] == 16)", gated off on POPCNT rows:
        // 1) (z - 16) * z0_is_16 = 0            (if flag is 1, z must be 16)
        builder
            .when(is_ctz_or_clz.clone())
            .assert_zero((local.half_word_z[0] - sixteen.clone()) * local.z0_is_16);
        // 2) (z - 16) * inv - (1 - z0_is_16) = 0  (if z != 16, inv enforces flag=0; if z == 16,
        //    forces flag=1)
        builder.when(is_ctz_or_clz.clone()).assert_zero(
            (local.half_word_z[0] - sixteen.clone()) * local.z0_minus_16_inv -
                (one.clone() - local.z0_is_16),
        );

        // Ensure the flag is never set on POPCNT rows (keeps the constraint inactive for POPCNT).
        // Quadratic: z0_is_16 * is_popcnt = 0
        builder.assert_zero(local.z0_is_16 * local.is_popcnt);

        // Unified result formula: z0 + flag*(16 + z1 - z0)
        let a_any = local.half_word_z[0] +
            local.z0_is_16 * (sixteen.clone() + local.half_word_z[1] - local.half_word_z[0]);

        let popcnt_result = local.half_word_z[0] + local.half_word_z[1];
        let result = a_any.clone() + local.is_popcnt * (popcnt_result - a_any);
        builder.assert_zero(local.a - result);

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            local.is_ctz * AB::Expr::from_canonical_u32(I32Ctz.code()) +
                local.is_clz * AB::Expr::from_canonical_u32(I32Clz.code()) +
                local.is_popcnt * AB::Expr::from_canonical_u32(I32Popcnt.code()),
            Word::extend_expr::<AB>(local.a.into()),
            local.b,
            Word::<AB::Expr>::default(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
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
    use num::PrimInt;
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
            AluEvent::new(0, 0, Opcode::I32Popcnt, b.count_ones(), b, 0, Opcode::I32Popcnt.code()),
            AluEvent::new(0, 0, Opcode::I32Ctz, b.trailing_zeros(), b, 0, Opcode::I32Ctz.code()),
            AluEvent::new(0, 0, Opcode::I32Clz, b.leading_zeros(), b, 0, Opcode::I32Clz.code()),
        ];
        let chip = TrailingChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.width(), NUM_TRAILING_COLS);
        assert_eq!(output.byte_lookups.len(), 6);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        let samples = vec![
            0b00000000_00000000_00000000_00000000u32, // all zeros
            0b11111111_11111111_11111111_11111111u32, // all ones
            0b00000000_00000000_00000000_00000001u32, // LSB set
            0b10000000_00000000_00000000_00000000u32, // MSB set
            0b00000000_11111111_00000000_11111111u32, // byte-wise pattern
            // --- Added 20 more edge-case samples ---
            0x0000_FFFFu32, // low half all ones, high half zero
            0xFFFF_0000u32, // high half all ones, low half zero
            0x0001_0000u32, // bit 16 set (low half zero -> CTZ low16 = 16)
            0x0000_8000u32, // low half highest bit set
            0x8000_8000u32, // highest bit set in both halves
            0x0000_0002u32, // CTZ(low) = 1
            0x0000_0100u32, // CTZ(low) = 8
            0x0100_0000u32, // high half non-zero, low half zero
            0x00FF_FF00u32, // middle two bytes 0xFF, edges 0x00
            0xFF00_00FFu32, // inverse of above across halves
            0xF0F0_F0F0u32, // alternating nibbles (11110000 pattern)
            0x0F0F_0F0Fu32, // alternating nibbles (00001111 pattern)
            0x3333_CCCCu32, // mixed half-word densities
            0xCCCC_3333u32, // swapped halves
            0x0000_00FFu32, // low byte ones
            0xFF00_FF00u32, // byte stripes across halves
            0x7FFF_0000u32, // high half just below 0x8000
            0x0000_7FFFu32, // low half just below 0x8000
            0x8000_0001u32, // MSB and LSB set
            0x0001_8000u32, // split bits across halves
        ];

        for b in samples.into_iter() {
            shard.trailing_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Popcnt,
                b.count_ones(),
                b,
                0,
                Opcode::I32Popcnt.code(),
            ));
            shard.trailing_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Ctz,
                b.trailing_zeros(),
                b,
                0,
                Opcode::I32Ctz.code(),
            ));
            shard.trailing_events.push(AluEvent::new(
                0,
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
    fn test_malicious_trailing() {
        use core::borrow::BorrowMut;
        let config = BabyBearPoseidon2::new();
        let challenger = config.challenger();
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Create a valid event.
        let op_b = 0b01001101u32;
        let op_c = 0b01101101u32;
        let opcodes = [Opcode::I32Popcnt, Opcode::I32Clz, Opcode::I32Ctz];
        for opcode in opcodes {
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
                        // Instead, forge the local witness to trigger a chip-level constraint
                        // failure.
                        cols.half_word_z[0] = BabyBear::from_canonical_u32(8);
                        cols.half_word_z[1] = BabyBear::from_canonical_u32(12);
                    }
                }

                traces
            };
            let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
            let chip = chip_name!(TrailingChip, BabyBear);
            println!(
                "test_malicious_trailing of opcode {:?} result.is_err =  {}",
                opcode,
                result.is_err()
            );
            assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
        }
    }
}
