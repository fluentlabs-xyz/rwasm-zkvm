use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm_executor::{
    events::ByteLookupEvent, ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_POPCNT_COLS: usize = size_of::<PopcntCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct PopcntCols<T> {
    pub pc: T,
    pub b_bytes: [T; 4],
    pub byte_popcnts: [T; 4],
    pub is_real: T,
}

#[derive(Default)]
pub struct PopcntChip;

impl PopcntChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &rwasm_executor::events::AluEvent,
        cols: &mut PopcntCols<F>,
        output: &mut ExecutionRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.is_real = F::one();

        let b_val = event.b;
        for i in 0..4 {
            let byte = (b_val >> (i * 8)) & 0xFF;
            let popcnt = byte.count_ones();
            cols.b_bytes[i] = F::from_canonical_u32(byte);
            cols.byte_popcnts[i] = F::from_canonical_u32(popcnt);
            *output
                .byte_lookups
                .entry(ByteLookupEvent::new(ByteOpcode::POPCNT, popcnt as u16, 0, byte as u8, 0))
                .or_insert(0) += 1;
        }
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

        // Send the byte lookups for each byte of `b`.
        for i in 0..4 {
            builder.send_byte(
                ByteOpcode::POPCNT.as_field::<AB::F>(),
                local.byte_popcnts[i],
                local.b_bytes[i],
                AB::Expr::zero(),
                local.is_real,
            );
        }
        // Prepare CPU operand/result words directly from local columns.
        let reconstructed_b = Word([
            local.b_bytes[0].into(),
            local.b_bytes[1].into(),
            local.b_bytes[2].into(),
            local.b_bytes[3].into(),
        ]);
        let total_popcnt = local.byte_popcnts.iter().fold(AB::Expr::zero(), |acc, &x| acc + x);
        let result_a = Word::extend_expr::<AB>(total_popcnt);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Popcnt.code()),
            result_a,
            reconstructed_b,
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

    fn local_only(&self) -> bool {
        true
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.popcnt_events.iter() {
            let mut row = [F::zero(); NUM_POPCNT_COLS];
            let cols: &mut PopcntCols<F> = row.as_mut_slice().borrow_mut();
            self.event_to_row(event, cols, output);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_POPCNT_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_POPCNT_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
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
        rwasm::{ByteChip, RwasmAir},
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use core::borrow::BorrowMut;
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, PrimeField};
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
        let a = b.count_ones();
        shard.popcnt_events =
            vec![AluEvent::new(0, Opcode::I32Popcnt, a, b, 0, Opcode::I32Popcnt.code())];
        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.width(), NUM_POPCNT_COLS);
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
            0b01010101_01010101_01010101_01010101u32,
            0b10101010_10101010_10101010_10101010u32,
        ];

        for b in samples.into_iter() {
            let a = b.count_ones();
            shard.popcnt_events.push(AluEvent::new(
                0,
                Opcode::I32Popcnt,
                a,
                b,
                0,
                Opcode::I32Popcnt.code(),
            ));
        }

        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, PopcntChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_popcnt() {
        let config = BabyBearPoseidon2::new();
        let challenger = config.challenger();
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Create a valid event.
        let op_b = 0b01001101u32;
        let op_c = 0b00001101u32;
        let opcode = Opcode::I32Popcnt;

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
            let chip = chip_name!(PopcntChip, BabyBear);

            // Find the PopcntChip's trace and forge it.
            if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                if trace.height() > 0 {
                    // PopcntChip's internal sum constraint will pass.
                    let row = trace.row_mut(0);
                    let cols: &mut PopcntCols<BabyBear> = row.borrow_mut();

                    cols.byte_popcnts[0] = BabyBear::from_canonical_u32(22);
                    cols.byte_popcnts[1] = BabyBear::from_canonical_u32(12);
                }
            }

            traces
        };
        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
    }
}
