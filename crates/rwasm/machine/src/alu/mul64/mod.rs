use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::Opcode::I32Mul64;
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

pub const NUM_MUL64_COLS: usize = size_of::<Mul64Cols<u8>>();

#[derive(Default)]
pub struct Mul64Chip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Mul64Cols<T> {
    pub pc: T,
    pub a_lo: Word<T>,
    pub a_hi: Word<T>,
    pub b: Word<T>,
    pub c: Word<T>,
    pub carry: [T; LONG_WORD_SIZE],
    pub is_real: T,
}

impl<F> BaseAir<F> for Mul64Chip {
    fn width(&self) -> usize {
        NUM_MUL64_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Mul64Chip {
    type Record = ExecutionRecord;
    type Program = rwasm_executor::Program;

    fn name(&self) -> String {
        "Mul64".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.mul64_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL64_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL64_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL64_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut Mul64Cols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.mul64_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_MUL64_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.mul64_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .mul64_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_MUL64_COLS];
                    let cols: &mut Mul64Cols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.mul64_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl Mul64Chip {
    fn event_to_row<F: PrimeField>(
        &self,
        event: &I64AluEvent,
        cols: &mut Mul64Cols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);

        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        let mut product = [0u32; LONG_WORD_SIZE];
        for i in 0..b_word.len() {
            for j in 0..c_word.len() {
                if i + j < LONG_WORD_SIZE {
                    product[i + j] += (b_word[i] as u32) * (c_word[j] as u32);
                }
            }
        }

        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; LONG_WORD_SIZE];
        for i in 0..LONG_WORD_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < LONG_WORD_SIZE {
                product[i + 1] += carry[i];
            }
            cols.carry[i] = F::from_canonical_u32(carry[i]);
        }

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

        blu.add_u16_range_checks(&carry.map(|x| x as u16));
    }
}

impl<AB> Air<AB> for Mul64Chip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Mul64Cols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << 8);
        let zero = AB::Expr::zero();

        builder.assert_bool(local.is_real);

        // Zero-extend b and c from 32 bits to 64 bits (unsigned semantics).
        let (b, c) = {
            let mut b = vec![zero.clone(); LONG_WORD_SIZE];
            let mut c = vec![zero.clone(); LONG_WORD_SIZE];
            for i in 0..WORD_SIZE {
                b[i] = local.b[i].into();
                c[i] = local.c[i].into();
            }
            (b, c)
        };

        // Send range checks for original b and c bytes
        builder.slice_range_check_u8(&local.b.0, local.is_real);
        builder.slice_range_check_u8(&local.c.0, local.is_real);

        // Uncarried product
        let mut m: Vec<AB::Expr> = vec![zero.clone(); LONG_WORD_SIZE];
        for i in 0..LONG_WORD_SIZE {
            for j in 0..LONG_WORD_SIZE {
                if i + j < LONG_WORD_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * c[j].clone();
                }
            }
        }

        // Carry propagation, directly constraining against a_lo/a_hi bytes.
        for i in 0..LONG_WORD_SIZE {
            let mut v = m[i].clone();
            if i > 0 {
                v += local.carry[i - 1].into();
            }
            v -= local.carry[i] * base;

            // Select the corresponding output byte from (a_lo, a_hi).
            let out_byte: AB::Expr =
                if i < WORD_SIZE { local.a_lo[i].into() } else { local.a_hi[i - WORD_SIZE].into() };

            // Enforce that the carried product byte equals the output byte.
            builder.assert_eq(out_byte, v);
        }

        // Range checks
        builder.slice_range_check_u16(&local.carry, local.is_real);
        builder.slice_range_check_u8(&local.a_lo.0, local.is_real);
        builder.slice_range_check_u8(&local.a_hi.0, local.is_real);

        let opcode = local.is_real * AB::F::from_canonical_u32(I32Mul64.code());

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
        let result = (b as i64).wrapping_mul(c as i64);
        shard.mul64_events.push(I64AluEvent {
            pc: 0,
            opcode: Opcode::I32Mul64,
            a_lo: result as u32,
            a_hi: (result >> 32) as u32,
            b,
            c,
            code: Opcode::I32Mul64.code(),
            res_hi_addr: 0,
            res_hi_access: None,
        });
        let chip = Mul64Chip::default();
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
            let result = (b as i64).wrapping_mul(c as i64);
            shard.mul64_events.push(I64AluEvent {
                pc: 0,
                opcode: Opcode::I32Mul64,
                a_lo: result as u32,
                a_hi: (result >> 32) as u32,
                b,
                c,
                code: Opcode::I32Mul64.code(),
                res_hi_addr: 0,
                res_hi_access: None,
            });
        }

        let chip = Mul64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_mul64() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let b = 12345u32;
        let c = 54321u32;
        let correct_res = (b as i64).wrapping_mul(c as i64);
        let wrong_res = correct_res.wrapping_add(1);

        let program = rwasm_executor::Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Mul64,
            Opcode::Drop,
            Opcode::Drop,
        ]);
        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();
            if let Some(event) =
                malicious_record.mul64_events.iter_mut().find(|e| e.opcode == Opcode::I32Mul64)
            {
                event.a_lo = wrong_res as u32;
                event.a_hi = (wrong_res >> 32) as u32;
            }
            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip_name = chip_name!(Mul64Chip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip_name));
    }
}
