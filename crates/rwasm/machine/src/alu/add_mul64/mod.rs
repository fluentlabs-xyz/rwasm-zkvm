use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use rwasm::{
    mem_index::UNIT,
    Opcode::{self, I32Add64, I32Mul64},
};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, I64AluEvent},
    ExecutionRecord, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{BYTE_SIZE, LONG_WORD_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    air::SP1CoreAirBuilder,
    memory::{MemoryCols, MemoryWriteCols},
    utils::{next_power_of_two, zeroed_f_vec},
};

/// The number of columns in the chip.
pub const NUM_ADDMUL64_COLS: usize = size_of::<AddMul64Cols<u8>>();

#[derive(Default)]
pub struct AddMul64Chip;

/// The column layout.
/// Shared columns: pc, a_lo, a_hi, b, c, carry.
/// Selectors: is_add, is_mul.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddMul64Cols<T> {
    pub pc: T,
    pub shard: T,
    pub clk: T,
    pub sp: T,
    // a_lo will be write in cpu chip
    pub res_hi: Word<T>,
    pub res_lo_write_record: MemoryWriteCols<T>,
    pub b: Word<T>,
    pub c: Word<T>,
    // Must be size 8 to support Mul64. Add64 uses only the first 4.
    pub carry: [T; LONG_WORD_SIZE],
    pub is_add: T,
    pub is_mul: T,
}

impl<F> BaseAir<F> for AddMul64Chip {
    fn width(&self) -> usize {
        NUM_ADDMUL64_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for AddMul64Chip {
    type Record = ExecutionRecord;
    type Program = rwasm_executor::Program;

    fn name(&self) -> String {
        "AddMul64".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Calculate total rows needed for both event types
        let nb_rows = input.i64_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADDMUL64_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_ADDMUL64_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADDMUL64_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddMul64Cols<F> = row.borrow_mut();

                    self.event_to_row(&input.i64_events[idx], cols, &mut Vec::new());
                });
            },
        );

        RowMajorMatrix::new(values, NUM_ADDMUL64_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let nb_rows = input.i64_events.len();

        if nb_rows == 0 {
            return;
        }

        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let num_chunks = nb_rows.div_ceil(chunk_size);

        let blu_batches = input
            .i64_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_ADDMUL64_COLS];
                    let cols: &mut AddMul64Cols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.i64_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddMul64Chip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &I64AluEvent,
        cols: &mut AddMul64Cols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let is_add_op = matches!(event.opcode, Opcode::I32Add64);

        cols.pc = F::from_canonical_u32(event.pc);

        cols.sp = F::from_canonical_u32(event.sp);

        cols.clk = F::from_canonical_u32(event.clk);

        cols.shard = F::from_canonical_u32(event.shard);

        cols.b = event.b.into();
        cols.c = event.c.into();
        cols.res_hi = event.res_hi.into();
        cols.res_lo_write_record.populate(event.res_lo_write_record, blu);

        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        // Range checks for inputs and outputs (always active)
        let a_lo_word = event.res_hi.to_le_bytes();
        let a_hi_word = event.res_lo_write_record.value.to_le_bytes();
        blu.add_u8_range_checks(&b_word);
        blu.add_u8_range_checks(&c_word);
        blu.add_u8_range_checks(&a_lo_word);
        blu.add_u8_range_checks(&a_hi_word);

        if is_add_op {
            cols.is_add = F::one();
            cols.is_mul = F::zero();

            // --- Add64 Witness Logic ---
            let mut carry_in: u16 = 0;
            for i in 0..WORD_SIZE {
                let s = (b_word[i] as u16) + (c_word[i] as u16) + carry_in;
                let carry_out = s >> 8;

                cols.carry[i] = F::from_canonical_u32(carry_out as u32);
                carry_in = carry_out;
            }
            // Remaining carries (indices 4..7) stay 0 (F::zero() default).
        } else {
            cols.is_add = F::zero();
            cols.is_mul = F::one();

            // --- Mul64 Witness Logic ---
            let mut product = [0u32; LONG_WORD_SIZE];

            // u32 * u32 optimization: only iterate 4x4
            for i in 0..WORD_SIZE {
                for j in 0..WORD_SIZE {
                    product[i + j] += (b_word[i] as u32) * (c_word[j] as u32);
                }
            }

            let base = (1 << BYTE_SIZE) as u32;
            let mut carry_vals = [0u16; LONG_WORD_SIZE];

            for i in 0..LONG_WORD_SIZE {
                let c = product[i] / base;
                product[i] %= base;
                if i + 1 < LONG_WORD_SIZE {
                    product[i + 1] += c;
                }
                carry_vals[i] = c as u16;
                cols.carry[i] = F::from_canonical_u32(c);
            }

            // Mul64 requires u16 range checks for carries.
            blu.add_u16_range_checks(&carry_vals);
        }
    }
}

impl<AB> Air<AB> for AddMul64Chip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddMul64Cols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << BYTE_SIZE);
        let zero = AB::Expr::zero();

        // Selector constraints
        let is_real = local.is_add + local.is_mul;
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_mul);
        builder.assert_bool(is_real.clone());

        // Shared Range Checks (Bytes)
        builder.slice_range_check_u8(&local.b.0, is_real.clone());
        builder.slice_range_check_u8(&local.c.0, is_real.clone());
        builder.slice_range_check_u8(&local.res_hi.0, is_real.clone());
        builder.slice_range_check_u8(&local.res_lo_write_record.value().0, is_real.clone());

        // --- ADD Operation Constraints ---
        {
            let mut builder_add = builder.when(local.is_add);
            // 1. Carries must be boolean (specific to Add optimization)
            for i in 0..WORD_SIZE {
                builder_add.assert_bool(local.carry[i]);
            }
            // 2. Upper carries (4..7) must be zero
            for i in WORD_SIZE..LONG_WORD_SIZE {
                builder_add.assert_zero(local.carry[i]);
            }

            // 3. Addition Logic (Lower 32 bits)
            let mut prev_carry = zero.clone();
            for i in 0..WORD_SIZE {
                let lhs = local.b[i].into() + local.c[i].into() + prev_carry.clone();
                let rhs = local.res_hi[i].into() + local.carry[i].into() * base;

                builder_add.assert_zero(lhs - rhs);
                prev_carry = local.carry[i].into();
            }

            // 4. Upper 32 bits logic (Implicit zero-extension of inputs)
            // Result a_hi[0] must equal the carry out from the lower 32 bits.
            builder_add.assert_eq(local.res_lo_write_record.value()[0], prev_carry);

            // The rest of a_hi must be zero.
            for i in 1..WORD_SIZE {
                builder_add.assert_zero(local.res_lo_write_record.value()[i]);
            }
        }

        // --- MUL Operation Constraints ---
        {
            // 1. Carries range check (u16)
            // We apply this check only when is_mul is active.
            // (Though Add carries are boolean, which satisfies u16, doing it selectively is
            // cleaner).
            builder.slice_range_check_u16(&local.carry, local.is_mul);
            let mut builder_mul = builder.when(local.is_mul);
            // 2. Multiplication Grid (4x4)
            let mut m: Vec<AB::Expr> = vec![zero.clone(); LONG_WORD_SIZE];
            for i in 0..WORD_SIZE {
                for j in 0..WORD_SIZE {
                    m[i + j] = m[i + j].clone() + local.b[i].into() * local.c[j].into();
                }
            }

            // 3. Carry Propagation
            let mut prev_carry = zero.clone();
            for i in 0..LONG_WORD_SIZE {
                // sum = partial_products + prev_carry
                let lhs = m[i].clone() + prev_carry.clone();

                // result = output_byte + 256 * new_carry
                let out_byte = if i < WORD_SIZE {
                    local.res_hi[i].into()
                } else {
                    local.res_lo_write_record.value()[i - WORD_SIZE].into()
                };

                let rhs = out_byte + local.carry[i] * base;

                builder_mul.assert_eq(lhs, rhs);
                prev_carry = local.carry[i].into();
            }
        }

        // --- Instruction Receiving ---
        let opcode = local.is_add * AB::F::from_canonical_u32(I32Add64.code()) +
            local.is_mul * AB::F::from_canonical_u32(I32Mul64.code());

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            &local.res_lo_write_record,
            is_real.clone(),
        );

        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            opcode,
            local.res_hi,
            *local.res_lo_write_record.value(),
            local.b,
            local.c,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            is_real,
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
    use rwasm_executor::{
        events::{MemoryRecordEnum, MemoryWriteRecord},
        ExecutionRecord, Opcode,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace_and_prove_addmul() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        // --- Test Cases for Add ---
        let add_cases = vec![
            (0u32, 0u32),
            (u32::MAX, 1),        // Overflow lower 32 -> a_hi[0]=1
            (u32::MAX, u32::MAX), // Large overflow
            (100, 200),
        ];

        for (b, c) in add_cases {
            let result = (b as u64) + (c as u64);

            let res_lo_write_record = MemoryWriteRecord::new(result as u32, 0, 1, 0, 0, 0);

            shard.i64_events.push(I64AluEvent {
                clk: 0,
                shard: 0,
                pc: 0,
                sp: 0,
                opcode: Opcode::I32Add64,
                res_hi: (result >> 32) as u32,
                res_lo_write_record,
                b,
                c,
                code: Opcode::I32Add64.code(),
            });
        }

        // --- Test Cases for Mul ---
        let mul_cases = vec![
            (0u32, 123u32),
            (u32::MAX, 1),
            (u32::MAX, u32::MAX), // Full 64-bit result
            (12345, 67890),
        ];

        for (b, c) in mul_cases {
            let result = (b as i64).wrapping_mul(c as i64);

            let res_lo_write_record = MemoryWriteRecord::new(result as u32, 0, 1, 0, 0, 0);

            shard.i64_events.push(I64AluEvent {
                pc: 4,
                shard: 0,
                clk: 0,
                sp: 0,
                opcode: Opcode::I32Mul64,
                res_hi: (result >> 32) as u32,
                res_lo_write_record,
                b,
                c,
                code: Opcode::I32Mul64.code(),
            });
        }

        let chip = AddMul64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);

        // Prove
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        // Verify
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_mul64() {
        // This test checks if the chip detects a malicious modification
        // We will attempt to provide a wrong result for a multiplication.
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let b = 100u32;
        let c = 200u32;
        let correct_res = (b as i64).wrapping_mul(c as i64);
        let wrong_res = correct_res + 0xffff;

        let program = rwasm_executor::Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Mul64, // This triggers the logic
            Opcode::Drop,
            Opcode::Drop,
        ]);
        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();
            // Find the mul event and break it
            if let Some(event) =
                malicious_record.i64_events.iter_mut().find(|e| e.opcode == Opcode::I32Mul64)
            {
                event.res_hi = (wrong_res >> 32) as u32;
                event.res_lo_write_record = MemoryWriteRecord::new(wrong_res as u32, 0, 1, 0, 0, 0);
            }
            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip_name = chip_name!(AddMul64Chip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip_name));
    }

    #[test]
    fn test_malicious_add64() {
        // This test checks if the chip detects a malicious modification
        // We will attempt to provide a wrong result for an addition.
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let b = 100u32;
        let c = 200u32;

        let correct_res = (b as u64) + (c as u64);
        let wrong_res = correct_res + 1;

        let program = rwasm_executor::Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Add64, // This triggers the logic
            Opcode::Drop,
            Opcode::Drop,
        ]);
        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();
            // Find the mul event and break it
            if let Some(event) =
                malicious_record.i64_events.iter_mut().find(|e| e.opcode == Opcode::I32Add64)
            {
                event.res_hi = (wrong_res >> 32) as u32;
                event.res_lo_write_record = MemoryWriteRecord::new(wrong_res as u32, 0, 1, 0, 0, 0);
            }
            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip_name = chip_name!(AddMul64Chip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip_name));
    }
}
