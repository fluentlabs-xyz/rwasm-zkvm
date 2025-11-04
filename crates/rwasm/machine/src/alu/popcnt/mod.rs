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
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_POPCNT_COLS: usize = size_of::<PopcntCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct PopcntCols<T> {
    pub pc: T,
    pub b_bytes: [T; 4],
    pub half_word_popcnts: [T; 2],
    pub is_real: T,
}

#[derive(Default)]
pub struct PopcntChip;

impl PopcntChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &rwasm_executor::events::AluEvent,
        cols: &mut PopcntCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.is_real = F::one();

        let b_val = event.b;
        let b_bytes = b_val.to_le_bytes();
        for i in 0..4 {
            cols.b_bytes[i] = F::from_canonical_u8(b_bytes[i]);
        }

        for i in 0..2 {
            let half_word = (b_val >> (i * 16)) & 0xFFFF;
            let popcnt = half_word.count_ones();
            cols.half_word_popcnts[i] = F::from_canonical_u32(popcnt);

            let high_byte = ((half_word >> 8) & 0xFF) as u8;
            let low_byte = (half_word & 0xFF) as u8;

            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U16POPCNT,
                a1: popcnt as u16,
                a2: 0,
                b: high_byte,
                c: low_byte,
            });
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

        // Send the byte lookups for each half-word of `b`.
        // Note: The `b` operand for the lookup is the high byte, and `c` is the low byte.
        builder.send_byte(
            ByteOpcode::U16POPCNT.as_field::<AB::F>(),
            local.half_word_popcnts[0],
            local.b_bytes[1],
            local.b_bytes[0],
            local.is_real,
        );
        builder.send_byte(
            ByteOpcode::U16POPCNT.as_field::<AB::F>(),
            local.half_word_popcnts[1],
            local.b_bytes[3],
            local.b_bytes[2],
            local.is_real,
        );

        // Prepare CPU operand/result words directly from local columns.
        let reconstructed_b = Word([
            local.b_bytes[0].into(),
            local.b_bytes[1].into(),
            local.b_bytes[2].into(),
            local.b_bytes[3].into(),
        ]);
        let a = local.half_word_popcnts[0] + local.half_word_popcnts[1];

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Popcnt.code()),
            Word::extend_expr::<AB>(a),
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
        assert_eq!(output.byte_lookups.len(), 2);
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
        use core::borrow::BorrowMut;
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

                    cols.half_word_popcnts[0] = BabyBear::from_canonical_u32(22);
                    cols.half_word_popcnts[1] = BabyBear::from_canonical_u32(12);
                }
            }

            traces
        };
        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
    }
}
