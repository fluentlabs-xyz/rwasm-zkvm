use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::{mem_index::UNIT, Opcode};
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

    pub sp: T,

    /// The output operand.
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

    /// Flag indicating whether `a` is not register 0.
    pub op_a_not_0: T,

    /// Trace.
    pub carry: [T; LONG_WORD_SIZE],

    /// An array storing the product of `b * c` after the carry propagation.
    pub product: [T; LONG_WORD_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The most significant bit of `c`.
    pub c_msb: T,

    /// The sign extension of `b`.
    pub b_sign_extend: T,

    /// The sign extension of `c`.
    pub c_sign_extend: T,

    /// Flag indicating whether the opcode is `MUL`  (`u32 x u32`).
    pub is_mul: T,

    /// Flag indicating whether the opcode is `MULH` (`i32 x i32`, upper half).
    pub is_mulh: T,

    /// Flag indicating whether the opcode is `MULHU` (`u32 x u32`, upper half).
    pub is_mulhu: T,

    /// Flag indicating whether the opcode is `MULHSU` (`i32 x u32`, upper half).
    pub is_mulhsu: T,

    /// Selector to know whether this row is enabled.
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
        cols.sp = F::from_canonical_u32(event.sp);

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

        // If signed extended, the MSB better be 1.
        builder.when(local.b_sign_extend).assert_eq(local.b_msb, one.clone());
        builder.when(local.c_sign_extend).assert_eq(local.c_msb, one.clone());

        // SAFETY: All selectors `is_mul`, `is_mulh`, `is_mulhu`, `is_mulhsu` are checked to be
        // boolean. Also, the multiplicity `is_real` is checked to be boolean, and `is_real
        // = 0` leads to no interactions. Each "real" row has exactly one selector turned
        // on, as constrained below. Therefore, in "real" rows, the `opcode` matches the
        // corresponding opcode.

        // Calculate the opcode.
        let opcode = {
            // Exactly one of the opcodes must be on.
            builder
                .when(local.is_real)
                .assert_one(local.is_mul + local.is_mulh + local.is_mulhu + local.is_mulhsu);

            let mul: AB::Expr = AB::F::from_canonical_u32(Opcode::I32Mul.code()).into();
            let mulh: AB::Expr = AB::F::from_canonical_u32(I32MULH_CODE).into();
            let mulhu: AB::Expr = AB::F::from_canonical_u32(I32MULHU_CODE).into();
            let mulhsu: AB::Expr = AB::F::from_canonical_u32(I32MULHSU_CODE).into();
            local.is_mul * mul +
                local.is_mulh * mulh +
                local.is_mulhu * mulhu +
                local.is_mulhsu * mulhsu
        };

        // Range check.
        {
            // Ensure that the carry is at most 2^16. This ensures that
            // product_before_carry_propagation - carry * base + last_carry never overflows or
            // underflows enough to "wrap" around to create a second solution.
            builder.slice_range_check_u16(&local.carry, local.is_real);

            builder.slice_range_check_u8(&local.product, local.is_real);
        }

        // Receive the arguments.
        // SAFETY: This checks the following.
        // - `next_pc = pc + 4`
        // - `num_extra_cycles = 0`
        // - `op_a_val` is constrained by the chip when `op_a_not_0 == 1`
        // - `op_a_not_0` is correct, due to the sent `op_a_0` being equal to `1 - op_a_not_0`
        // - `op_a_immutable = 0`
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            opcode,
            local.a,
            local.b,
            local.c,
            Word::zero::<AB>(),
            AB::Expr::zero(),
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
        for _ in 0..10i32.pow(7) {
            mul_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Mul, //MULHSU
                0x80004000,
                0x80000000,
                0xffff8000,
                Opcode::I32Mul.code(),
            ));
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

        let mul_instructions: Vec<(Opcode, u32, u32, u32)> = vec![
            (Opcode::I32Mul, 0x00001200, 0x00007e00, 0xb6db6db7),
            (Opcode::I32Mul, 0x00001240, 0x00007fc0, 0xb6db6db7),
            (Opcode::I32Mul, 0x00000000, 0x00000000, 0x00000000),
            (Opcode::I32Mul, 0x00000001, 0x00000001, 0x00000001),
            (Opcode::I32Mul, 0x00000015, 0x00000003, 0x00000007),
            (Opcode::I32Mul, 0x00000000, 0x00000000, 0xffff8000),
            (Opcode::I32Mul, 0x00000000, 0x80000000, 0x00000000),
            (Opcode::I32Mul, 0x00000000, 0x80000000, 0xffff8000),
            (Opcode::I32Mul, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d),
            (Opcode::I32Mul, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab),
            (Opcode::I32Mul, 0x00000000, 0xff000000, 0xff000000),
            (Opcode::I32Mul, 0x00000001, 0xffffffff, 0xffffffff),
            (Opcode::I32Mul, 0xffffffff, 0xffffffff, 0x00000001),
            (Opcode::I32Mul, 0xffffffff, 0x00000001, 0xffffffff),
        ];
        for t in mul_instructions.iter() {
            mul_events.push(AluEvent::new(0, 0, t.0, t.1, t.2, t.3, t.0.code()));
        }

        // Append more events until we have 1000 tests.
        for _ in 0..(1000 - mul_instructions.len()) {
            mul_events.push(AluEvent::new(0, 0, Opcode::I32Mul, 8, 2, 4, Opcode::I32Mul.code()));
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

                let op_a = thread_rng().gen_range(0..u32::MAX);
                assert_ne!(op_a, correct_op_a);

                let instructions = vec![
                    Opcode::I32Const(5u32.into()),
                    Opcode::I32Const(10u32.into()),
                    Opcode::I32Const(op_b.into()),
                    Opcode::I32Const(op_c.into()),
                    opcode,
                    Opcode::I32Mul,
                ];

                let program = Program::from_instrs(instructions);
                let stdin = SP1Stdin::new();

                type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

                let malicious_trace_pv_generator = move |prover: &P,
                                                         record: &mut ExecutionRecord|
                      -> Vec<(
                    String,
                    RowMajorMatrix<Val<BabyBearPoseidon2>>,
                )> {
                    let mut malicious_record = record.clone();
                    // The ALU op of interest is the 5th instruction (index 4)
                    if malicious_record.cpu_events.len() > 4 {
                        if let Some(MemoryRecordEnum::Write(mut write_record)) =
                            malicious_record.cpu_events[4].res_record
                        {
                            write_record.value = op_a as u32;
                        }
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
