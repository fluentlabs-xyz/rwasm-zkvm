use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::Opcode::I32Add64;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, I64AluEvent},
    ExecutionRecord, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{BYTE_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};

pub const NUM_ADD64_COLS: usize = size_of::<Add64Cols<u8>>();

#[derive(Default)]
pub struct Add64Chip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Add64Cols<T> {
    pub pc: T,
    pub a_lo: Word<T>,
    pub a_hi: Word<T>,
    pub b: Word<T>,
    pub c: Word<T>,
    // OPTIMIZATION: We only need carries for the lower 32 bits (4 bytes).
    // The upper 32 bits are determined entirely by carry[3].
    pub carry: [T; WORD_SIZE],
    pub is_real: T,
}

impl<F> BaseAir<F> for Add64Chip {
    fn width(&self) -> usize {
        NUM_ADD64_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Add64Chip {
    type Record = ExecutionRecord;
    type Program = rwasm_executor::Program;

    fn name(&self) -> String {
        "Add64".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.add64_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD64_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_ADD64_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD64_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut Add64Cols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.add64_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_ADD64_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.add64_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .add64_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_ADD64_COLS];
                    let cols: &mut Add64Cols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.add64_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl Add64Chip {
    fn event_to_row<F: PrimeField>(
        &self,
        event: &I64AluEvent,
        cols: &mut Add64Cols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);

        // Inputs as little-endian 32-bit words.
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        // Calculate carries for lower 32 bits
        let mut carry_in: u16 = 0;
        for i in 0..WORD_SIZE {
            let bi = b_word[i] as u16;
            let ci = c_word[i] as u16;
            let s = bi + ci + carry_in;

            // No need to store sum_bytes manually, event.a_lo has the result
            let carry_out = s >> 8;

            cols.carry[i] = F::from_canonical_u32(carry_out as u32);
            carry_in = carry_out;
        }

        // Fill columns.
        cols.a_lo = event.a_lo.into();
        cols.a_hi = event.a_hi.into();
        cols.b = event.b.into();
        cols.c = event.c.into();
        cols.is_real = F::one();

        let a_lo_word = event.a_lo.to_le_bytes();
        let a_hi_word = event.a_hi.to_le_bytes();

        // Send range checks.
        // Note: We range check a_hi even though we know it's mostly zero
        // to satisfy the byte lookup interface requirements.
        blu.add_u8_range_checks(&b_word);
        blu.add_u8_range_checks(&c_word);
        blu.add_u8_range_checks(&a_lo_word);
        blu.add_u8_range_checks(&a_hi_word);
    }
}

impl<AB> Air<AB> for Add64Chip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Add64Cols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << BYTE_SIZE); // 256
        let zero = AB::Expr::zero();
        let one = AB::Expr::one();

        builder.assert_bool(local.is_real);

        // Constrain carries to be boolean (0 or 1).
        // Optimization: Since max column sum is 255+255+1 = 511,
        // carry cannot be > 1. Boolean check is cheaper than u16 lookup.
        for c in local.carry.iter() {
            builder.assert_bool(*c);
        }

        // Range checks for inputs and outputs.
        builder.slice_range_check_u8(&local.b.0, local.is_real);
        builder.slice_range_check_u8(&local.c.0, local.is_real);
        builder.slice_range_check_u8(&local.a_lo.0, local.is_real);
        builder.slice_range_check_u8(&local.a_hi.0, local.is_real);

        // 1. Logic for Lower 32-bits (Standard Addition)
        let mut prev_carry = zero.clone();

        for i in 0..WORD_SIZE {
            let lhs = local.b[i].into() + local.c[i].into() + prev_carry.clone();
            let rhs = local.a_lo[i].into() + local.carry[i].into() * base;

            builder.when(local.is_real).assert_zero(lhs - rhs);

            prev_carry = local.carry[i].into();
        }

        // 2. Logic for Upper 32-bits (Implicit Handling)
        // Since inputs are u32, the result is at most 33 bits (2^32 + 2^32 = 2^33).
        // This means a_hi[0] is 1 if there was a carry out of the 32nd bit, else 0.
        // a_hi[1..3] must always be 0.

        // Enforce: a_hi[0] == carry[3] (The last carry from the loop above)
        builder.when(local.is_real).assert_eq(local.a_hi[0], prev_carry);

        // Enforce: a_hi[1], a_hi[2], a_hi[3] == 0
        for i in 1..WORD_SIZE {
            builder.when(local.is_real).assert_zero(local.a_hi[i]);
        }

        // Wire the instruction
        let opcode = local.is_real * AB::F::from_canonical_u32(I32Add64.code());

        builder.receive_64_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            opcode,
            local.a_lo,
            local.a_hi,
            local.b,
            local.c,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use rwasm_executor::{ExecutionRecord, Opcode};
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();
        let b = 2u32;
        let c = 3u32;
        let result = (b as u64) + (c as u64);
        shard.add64_events.push(I64AluEvent {
            pc: 0,
            opcode: Opcode::I32Add64,
            a_lo: result as u32,
            a_hi: (result >> 32) as u32,
            b,
            c,
            code: Opcode::I32Add64.code(),
            res_hi_addr: 0,
            res_hi_access: None,
        });
        let chip = Add64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.height(), 16);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        // Added u32::MAX + u32::MAX to test the a_hi[0] == 1 case
        let test_cases = vec![
            (0, 0),
            (1, 0),
            (u32::MAX, 0),
            (1, 1),
            (u32::MAX, 1),
            (2, 3),
            (u32::MAX, u32::MAX),
            (1 << 20, 1 << 20),
        ];

        for (b, c) in test_cases {
            let result = (b as u64) + (c as u64);
            shard.add64_events.push(I64AluEvent {
                pc: 0,
                opcode: Opcode::I32Add64,
                a_lo: result as u32,
                a_hi: (result >> 32) as u32,
                b,
                c,
                code: Opcode::I32Add64.code(),
                res_hi_addr: 0,
                res_hi_access: None,
            });
        }

        let chip = Add64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_add64() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let b = 12345u32;
        let c = 54321u32;
        let correct_res = (b as u64) + (c as u64);
        let wrong_res = correct_res + 1;

        let program = rwasm_executor::Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Add64,
            Opcode::Drop,
            Opcode::Drop,
        ]);
        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();
            if let Some(event) =
                malicious_record.add64_events.iter_mut().find(|e| e.opcode == Opcode::I32Add64)
            {
                event.a_lo = wrong_res as u32;
                event.a_hi = (wrong_res >> 32) as u32;
            }
            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip_name = chip_name!(Add64Chip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip_name));
    }
}
