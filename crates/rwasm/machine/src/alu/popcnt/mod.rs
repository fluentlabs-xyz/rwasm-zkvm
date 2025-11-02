use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
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
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Popcnt.code()),
            a,
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
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_baby_bear::BabyBear;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rwasm_executor::{events::AluEvent, ExecutionRecord, Opcode};
    use sp1_stark::{air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let b: u32 = 0x137_137;
        let a = b.count_ones();
        shard.rotate_events =
            vec![AluEvent::new(0, Opcode::I32Rotl, a, b, 0, Opcode::I32Popcnt.code())];
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
        events.resize_with(1000, || {
            AluEvent::new(0, Opcode::I32Popcnt, 0, 0, 0, Opcode::I32Popcnt.code())
        });
        let mut shard = ExecutionRecord::default();
        shard.rotate_events = events;
        let chip = PopcntChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, PopcntChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
