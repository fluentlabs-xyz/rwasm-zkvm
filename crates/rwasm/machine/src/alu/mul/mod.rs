use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::Opcode;
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, EmptyByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{BYTE_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};

pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

#[derive(Default)]
pub struct MulChip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulCols<T> {
    pub pc: T,
    pub a: Word<T>, // Result (Lower 32 bits)
    pub b: Word<T>, // Input 1
    pub c: Word<T>, // Input 2
    // Only need 4 carries for the lower 32 bits.
    pub carry: [T; WORD_SIZE],
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for MulChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.mul_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = EmptyByteRecord;
                        let event = &input.mul_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_MUL_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);
        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.mul_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MulChip {
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut MulCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        let a_word = event.a.to_le_bytes();
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        cols.a = event.a.into();
        cols.b = event.b.into();
        cols.c = event.c.into();
        cols.is_real = F::one();

        // Calculate partial products and carries for the lower 32 bits only
        let mut product = [0u32; WORD_SIZE];
        for i in 0..WORD_SIZE {
            for j in 0..WORD_SIZE {
                // We only care about terms that affect indices 0..3
                if i + j < WORD_SIZE {
                    product[i + j] += (b_word[i] as u32) * (c_word[j] as u32);
                }
            }
        }

        let base = (1 << BYTE_SIZE) as u32;
        // We only need to propagate carries up to the 4th byte
        let mut carry_vals = [0u16; WORD_SIZE];

        for i in 0..WORD_SIZE {
            let c_val = product[i] / base;
            // Propagate carry to next limb (if within 32-bit window)
            if i + 1 < WORD_SIZE {
                product[i + 1] += c_val;
            }
            // Even for the last byte (i=3), we calculate the carry-out (overflow)
            // and store it to balance the constraint equation, though it's discarded later.
            cols.carry[i] = F::from_canonical_u32(c_val);
            carry_vals[i] = c_val as u16;
        }

        // Range checks
        if !blu.is_dummy() {
            blu.add_u16_range_checks(&carry_vals);
            blu.add_u8_range_checks(&a_word);
            blu.add_u8_range_checks(&b_word);
            blu.add_u8_range_checks(&c_word);
        }
    }
}

impl<F> BaseAir<F> for MulChip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<AB> Air<AB> for MulChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MulCols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << 8);
        let zero = AB::Expr::zero();

        builder.assert_bool(local.is_real);

        // 1. Compute uncarried product terms for lower 32 bits
        let mut m: Vec<AB::Expr> = vec![zero.clone(); WORD_SIZE];
        for i in 0..WORD_SIZE {
            for j in 0..WORD_SIZE {
                if i + j < WORD_SIZE {
                    m[i + j] = m[i + j].clone() + local.b[i].into() * local.c[j].into();
                }
            }
        }

        // 2. Carry Propagation Constraint:
        // m[i] + prev_carry = a[i] + carry[i] * 256
        for i in 0..WORD_SIZE {
            let prev_carry = if i == 0 { zero.clone() } else { local.carry[i - 1].into() };

            // Note: we use local.a[i] directly instead of a product column
            let lhs = m[i].clone() + prev_carry;
            let rhs = local.a[i].into() + local.carry[i] * base;

            builder.when(local.is_real).assert_eq(lhs, rhs);
        }

        // 3. Range Checks
        builder.slice_range_check_u16(&local.carry, local.is_real);
        builder.slice_range_check_u8(&local.a.0, local.is_real);
        builder.slice_range_check_u8(&local.b.0, local.is_real);
        builder.slice_range_check_u8(&local.c.0, local.is_real);

        let opcode: AB::Expr = AB::F::from_canonical_u32(Opcode::I32Mul.code()).into();
        // 4. Receive Instruction
        builder.receive_instruction_old(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            opcode,
            local.a,
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
    use super::MulChip;
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};
    use rwasm::Opcode;
    use rwasm_executor::{
        events::{AluEvent, MemoryRecordEnum},
        ExecutionRecord, Program,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig, Val,
    };

    #[test]
    fn generate_trace_mul() {
        let mut shard = ExecutionRecord::default();
        let mut mul_events: Vec<AluEvent> = Vec::new();
        for _ in 0..100 {
            let b = thread_rng().gen::<u32>();
            let c = thread_rng().gen::<u32>();
            let a = b.wrapping_mul(c);
            mul_events.push(AluEvent::new(0, Opcode::I32Mul, a, b, c, Opcode::I32Mul.code()));
        }
        shard.mul_events = mul_events;
        let chip = MulChip::default();
        let _trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut mul_events: Vec<AluEvent> = Vec::new();

        // Edge cases + Random
        let instructions: Vec<(u32, u32)> =
            vec![(0, 0), (1, 1), (u32::MAX, 1), (u32::MAX, u32::MAX), (12345, 67890)];

        for (b, c) in instructions {
            let a = b.wrapping_mul(c);
            mul_events.push(AluEvent::new(0, Opcode::I32Mul, a, b, c, Opcode::I32Mul.code()));
        }

        shard.mul_events = mul_events;
        let chip = MulChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_mul() {
        const NUM_TESTS: usize = 5;
        for _ in 0..NUM_TESTS {
            let b = thread_rng().gen::<u32>();
            let c = thread_rng().gen::<u32>();
            let a_correct = b.wrapping_mul(c);
            let a_malicious = a_correct.wrapping_add(1);

            let program = Program::from_instrs(vec![
                Opcode::I32Const(b.into()),
                Opcode::I32Const(c.into()),
                Opcode::I32Mul,
            ]);
            let stdin = SP1Stdin::new();
            type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

            let malicious_gen = move |prover: &P, record: &mut ExecutionRecord| {
                let mut mal_rec = record.clone();
                if !mal_rec.mul_events.is_empty() {
                    mal_rec.mul_events[0].a = a_malicious;
                }
                // Manipulate the CPU event result (instruction index 2)
                if mal_rec.cpu_events.len() > 2 {
                    mal_rec.cpu_events[2].res = a_malicious;
                    if let Some(MemoryRecordEnum::Write(mut write_record)) =
                        mal_rec.cpu_events[2].res_record
                    {
                        write_record.value = a_malicious;
                    }
                }
                prover.generate_traces(&mal_rec)
            };

            let result = run_malicious_test::<P>(program, stdin, Box::new(malicious_gen));
            let name = chip_name!(MulChip, BabyBear);
            assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&name));
        }
    }
}
