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
use sp1_primitives::consts::{BYTE_SIZE, LONG_WORD_SIZE, WORD_SIZE};
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
    pub carry: [T; LONG_WORD_SIZE],
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

        // Zero-extend b and c from 32 bits to 64 bits at the byte level.
        let mut b_bytes = [0u8; LONG_WORD_SIZE];
        let mut c_bytes = [0u8; LONG_WORD_SIZE];

        b_bytes[..WORD_SIZE].copy_from_slice(&b_word[..WORD_SIZE]);
        c_bytes[..WORD_SIZE].copy_from_slice(&c_word[..WORD_SIZE]);

        // Compute the 64-bit sum and per-byte carries in host code,
        // and use them as the witness for the STARK.
        let mut carry_in: u16 = 0;
        let mut carries: [u16; LONG_WORD_SIZE] = [0; LONG_WORD_SIZE];
        let mut sum_bytes = [0u8; LONG_WORD_SIZE];

        for i in 0..LONG_WORD_SIZE {
            let bi = b_bytes[i] as u16;
            let ci = c_bytes[i] as u16;
            let s = bi + ci + carry_in;

            let out_byte = (s & 0x00ff) as u8;
            let carry_out = s >> 8; // 0 or 1

            sum_bytes[i] = out_byte;
            carries[i] = carry_out;
            carry_in = carry_out;

            cols.carry[i] = F::from_canonical_u32(carry_out as u32);
        }

        // Fill columns.
        cols.a_lo = event.a_lo.into();
        cols.a_hi = event.a_hi.into();
        cols.b = event.b.into();
        cols.c = event.c.into();
        cols.is_real = F::one();

        let a_lo_word = event.a_lo.to_le_bytes();
        let a_hi_word = event.a_hi.to_le_bytes();

        // Send range checks for all 32-bit words (inputs and outputs).
        blu.add_u8_range_checks(&b_word);
        blu.add_u8_range_checks(&c_word);
        blu.add_u8_range_checks(&a_lo_word);
        blu.add_u8_range_checks(&a_hi_word);

        // Range-check the carry bytes as u16.
        blu.add_u16_range_checks(&carries);
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

        // is_real is used to gate lookups and instruction wiring.
        builder.assert_bool(local.is_real);

        // Zero-extend b and c from 32 bits to 64 bits at the byte level.
        let (b, c) = {
            let mut b = vec![zero.clone(); LONG_WORD_SIZE];
            let mut c = vec![zero.clone(); LONG_WORD_SIZE];
            for i in 0..WORD_SIZE {
                b[i] = local.b[i].into();
                c[i] = local.c[i].into();
            }
            (b, c)
        };

        // Range checks for inputs, outputs, and carries.
        builder.slice_range_check_u8(&local.b.0, local.is_real);
        builder.slice_range_check_u8(&local.c.0, local.is_real);
        builder.slice_range_check_u8(&local.a_lo.0, local.is_real);
        builder.slice_range_check_u8(&local.a_hi.0, local.is_real);
        builder.slice_range_check_u16(&local.carry, local.is_real);

        // Enforce byte-wise 64-bit addition:
        //
        // For i = 0:
        //   b_0 + c_0 = out_0 + 256 * carry_0
        //
        // For i > 0:
        //   b_i + c_i + carry_{i-1} = out_i + 256 * carry_i
        //
        // where out_0..3 are bytes of a_lo, and out_4..7 are bytes of a_hi.
        let mut prev_carry = zero.clone();

        for i in 0..LONG_WORD_SIZE {
            // Select the corresponding output byte from (a_lo, a_hi).
            let out_byte: AB::Expr =
                if i < WORD_SIZE { local.a_lo[i].into() } else { local.a_hi[i - WORD_SIZE].into() };

            let mut lhs = b[i].clone() + c[i].clone();
            if i > 0 {
                lhs = lhs + prev_carry.clone();
            }

            let rhs = out_byte + local.carry[i] * base;

            // Constrain addition relation byte-by-byte.
            builder.when(local.is_real).assert_zero(lhs - rhs);

            prev_carry = local.carry[i].into();
        }

        // Wire the instruction into the global CPU/instruction table.
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
        assert_eq!(trace.height(), 16); // Padded to a minimum of 16
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

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
