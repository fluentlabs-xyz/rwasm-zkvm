use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::Opcode::{I32Extend16S, I32Extend8S};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};
pub const NUM_EXTEND_COLS: usize = size_of::<ExtendCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ExtendCols<T> {
    pub pc: T,
    pub sp: T,
    pub a: Word<T>, // result bytes
    pub b: Word<T>, // input bytes
    pub is_extend8s: T,
    pub is_extend16s: T,
    // Unified sign bit (certified via byte lookup)
    pub msb: T, // bit7(b[0]) for extend8s, bit7(b[1]) for extend16s
}

#[derive(Default)]
pub struct ExtendChip;

impl ExtendChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut ExtendCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);
        cols.a = event.a.into();
        cols.b = event.b.into();
        cols.is_extend8s = F::from_bool(event.opcode == Opcode::I32Extend8S);
        cols.is_extend16s = F::from_bool(event.opcode == Opcode::I32Extend16S);

        let b = event.b;
        let b_bytes = b.to_le_bytes();

        // Emit one lookup per active opcode and set the unified `msb` column.
        if event.opcode == Opcode::I32Extend8S {
            let msb = b_bytes[0] >> 7;
            cols.msb = F::from_canonical_u32(msb as u32);
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::MSB,
                msb as u16,
                0,
                b_bytes[0],
                0,
            ));
        } else if event.opcode == Opcode::I32Extend16S {
            let msb = b_bytes[1] >> 7;
            cols.msb = F::from_canonical_u32(msb as u32);
            blu.add_byte_lookup_event(ByteLookupEvent::new(
                ByteOpcode::MSB,
                msb as u16,
                0,
                b_bytes[1],
                0,
            ));
        }
    }
}

impl<AB> Air<AB> for ExtendChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ExtendCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_extend8s);
        builder.assert_bool(local.is_extend16s);
        builder.assert_bool(local.msb);

        let is_real = local.is_extend8s + local.is_extend16s;
        builder.assert_bool(is_real.clone());

        let ff = AB::Expr::from_canonical_u32(0xFF);
        // Constrain the unified `msb` column depending on the active opcode.
        // The correct MSB is certified by the corresponding lookup.
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.msb,         // Certified result
            local.b[0],        // op1: byte being checked
            AB::Expr::zero(),  // op2
            local.is_extend8s, // Multiplicity
        );
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.msb,          // Certified result
            local.b[1],         // op1: byte being checked
            AB::Expr::zero(),   // op2
            local.is_extend16s, // Multiplicity
        );

        // i32.extend8_s result bytes
        builder.when(local.is_extend8s).assert_eq(local.a[0], local.b[0]);
        builder.when(local.is_extend8s).assert_eq(local.a[1], local.msb * ff.clone());
        builder.when(local.is_extend8s).assert_eq(local.a[2], local.msb * ff.clone());
        builder.when(local.is_extend8s).assert_eq(local.a[3], local.msb * ff.clone());

        // i32.extend16_s result bytes
        builder.when(local.is_extend16s).assert_eq(local.a[0], local.b[0]);
        builder.when(local.is_extend16s).assert_eq(local.a[1], local.b[1]);
        builder.when(local.is_extend16s).assert_eq(local.a[2], local.msb * ff.clone());
        builder.when(local.is_extend16s).assert_eq(local.a[3], local.msb * ff.clone());

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            local.is_extend8s * AB::Expr::from_canonical_u32(I32Extend8S.code()) +
                local.is_extend16s * AB::Expr::from_canonical_u32(I32Extend16S.code()),
            local.a,
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

impl<F> BaseAir<F> for ExtendChip {
    fn width(&self) -> usize {
        NUM_EXTEND_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ExtendChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Extend".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.extend_events.iter() {
            let mut row = [F::zero(); NUM_EXTEND_COLS];
            let cols: &mut ExtendCols<F> = row.as_mut_slice().borrow_mut();
            self.event_to_row(event, cols, output);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_EXTEND_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_EXTEND_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use rayon::{iter::ParallelIterator, slice::ParallelSlice};
        let chunk_size = std::cmp::max(input.extend_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .extend_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_EXTEND_COLS];
                    let cols: &mut ExtendCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();
        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.extend_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::{ExtendChip, ExtendCols, NUM_EXTEND_COLS};
    use crate::{
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

        // A few canonical extend cases
        let cases = [
            (Opcode::I32Extend8S, 0x0000_007Fu32, 0x0000_007Fu32), // +127
            (Opcode::I32Extend8S, 0x0000_0080u32, 0xFFFF_FF80u32), // -128
            (Opcode::I32Extend16S, 0x0000_7FFFu32, 0x0000_7FFFu32), // +32767
            (Opcode::I32Extend16S, 0x0000_8000u32, 0xFFFF_8000u32), // -32768
        ];
        for (op, b, a) in cases {
            shard.extend_events.push(AluEvent::new(0, 0, op, a, b, 0, op.code()));
        }

        let chip = ExtendChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.width(), NUM_EXTEND_COLS);
        assert_eq!(output.byte_lookups.len(), 2);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        let samples: [u32; 10] = [
            0x0000_0000,
            0x0000_007F, // +127
            0x0000_0080, // -128
            0x0000_00FF, // -1 (8-bit)
            0x0000_7FFF, // +32767
            0x0000_8000, // -32768
            0x0000_FFFF, // -1 (16-bit)
            0xDEAD_8001,
            0xBEEF_1234,
            0xFEED_BEEF,
        ];

        for &b in &samples {
            let a8 = (((b as i32) as i8) as i32) as u32;
            let a16 = (((b as i32) as i16) as i32) as u32;
            shard.extend_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Extend8S,
                a8,
                b,
                0,
                Opcode::I32Extend8S.code(),
            ));
            shard.extend_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Extend16S,
                a16,
                b,
                0,
                Opcode::I32Extend16S.code(),
            ));
        }

        let chip = ExtendChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, ExtendChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_extend() {
        use core::borrow::BorrowMut;
        let config = BabyBearPoseidon2::new();
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Program that triggers both extend ops
        let program = Program::from_instrs(vec![
            Opcode::I32Const(0x0000_0080u32.into()),
            Opcode::I32Extend8S,
            Opcode::I32Const(0x0000_8000u32.into()),
            Opcode::I32Extend16S,
        ]);

        let stdin = SP1Stdin::new();
        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            // Generate honest traces first
            let mut traces = prover.generate_traces(record);
            let chip = chip_name!(ExtendChip, BabyBear);

            if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                if trace.height() > 0 {
                    // Corrupt a boolean bit to violate the (b*(1-b)=0) constraint
                    let row0 = trace.row_mut(0);
                    let cols: &mut ExtendCols<BabyBear> = row0.borrow_mut();
                    cols.msb = BabyBear::from_canonical_u32(2);
                }
            }

            traces
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip = chip_name!(ExtendChip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
    }
}
