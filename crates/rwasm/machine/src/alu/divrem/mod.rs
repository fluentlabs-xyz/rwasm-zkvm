use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{pad_rows_fixed, word_to_expr},
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use rwasm::Opcode;
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, EmptyByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{
    air::{BaseAirBuilder, MachineAir},
    Word,
};

pub const NUM_DIV_REM_COLS: usize = size_of::<DivRemCols<u8>>();

#[derive(Default)]
pub struct DivRemChip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct DivRemCols<T> {
    pub pc: T,
    pub b: Word<T>,
    pub c: Word<T>,
    pub a: Word<T>,
    pub is_div_u: T,
    pub is_div_s: T,
    pub is_rem_u: T,
    pub is_rem_s: T,
    pub sign_xor: T,
    pub b_abs: Word<T>,
    pub c_abs: Word<T>,
    pub q_abs: Word<T>,
    pub r_abs: Word<T>,
    pub diff: Word<T>,
    pub carry: [T; WORD_SIZE],
    pub diff_borrow: [T; WORD_SIZE],
    pub b_sign: T,
    pub c_sign: T,
    pub q_sign: T,
    pub r_sign: T,
}

impl<F: PrimeField32> MachineAir<F> for DivRemChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "DivRem".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = input
            .divrem_events
            .par_iter()
            .map(|event| {
                let mut row = [F::zero(); NUM_DIV_REM_COLS];
                let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = EmptyByteRecord;
                self.event_to_row(event, cols, &mut blu);
                row
            })
            .collect::<Vec<_>>();

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_DIV_REM_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIV_REM_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.divrem_events.len() / num_cpus::get(), 1);
        let blu_batches = input
            .divrem_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_DIV_REM_COLS];
                    let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.divrem_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl DivRemChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut DivRemCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        let b_val = event.b;
        let c_val = event.c;
        let opcode_enum = event.opcode;

        cols.b = Word(event.b.to_le_bytes().map(F::from_canonical_u8));
        cols.c = Word(event.c.to_le_bytes().map(F::from_canonical_u8));

        let mut is_signed = false;
        match opcode_enum {
            Opcode::I32DivS => {
                cols.is_div_s = F::one();
                is_signed = true;
            }
            Opcode::I32DivU => {
                cols.is_div_u = F::one();
            }
            Opcode::I32RemS => {
                cols.is_rem_s = F::one();
                is_signed = true;
            }
            Opcode::I32RemU => {
                cols.is_rem_u = F::one();
            }
            _ => panic!("Invalid opcode for DivRemChip"),
        }

        let (b_abs_val, b_sign) = if is_signed && (b_val as i32) < 0 {
            (b_val.wrapping_neg(), true)
        } else {
            (b_val, false)
        };

        let (c_abs_val, c_sign) = if is_signed && (c_val as i32) < 0 {
            (c_val.wrapping_neg(), true)
        } else {
            (c_val, false)
        };

        cols.sign_xor = F::from_bool(b_sign ^ c_sign);

        let (q_abs_val, r_abs_val) = if c_abs_val == 0 {
            (0, b_abs_val)
        } else {
            (b_abs_val / c_abs_val, b_abs_val % c_abs_val)
        };

        cols.b_sign = F::from_bool(b_sign);
        cols.c_sign = F::from_bool(c_sign);
        cols.b_abs = Word(b_abs_val.to_le_bytes().map(F::from_canonical_u8));
        cols.c_abs = Word(c_abs_val.to_le_bytes().map(F::from_canonical_u8));
        cols.q_abs = Word(q_abs_val.to_le_bytes().map(F::from_canonical_u8));
        cols.r_abs = Word(r_abs_val.to_le_bytes().map(F::from_canonical_u8));

        let diff_val = c_abs_val.wrapping_sub(r_abs_val).wrapping_sub(1);
        cols.diff = Word(diff_val.to_le_bytes().map(F::from_canonical_u8));

        let mut borrow = 0u32;
        let r_bytes = r_abs_val.to_le_bytes();
        let diff_bytes = diff_val.to_le_bytes();
        for i in 0..WORD_SIZE {
            let mut sum = (r_bytes[i] as u32) + (diff_bytes[i] as u32) + borrow;
            if i == 0 {
                sum += 1;
            }
            cols.diff_borrow[i] = F::from_canonical_u32(sum / 256);
            borrow = sum / 256;
        }

        let q_bytes = q_abs_val.to_le_bytes();
        let c_bytes = c_abs_val.to_le_bytes();
        let mut carry = 0u32;
        for k in 0..WORD_SIZE {
            let mut sum = carry + (r_bytes[k] as u32);
            for i in 0..WORD_SIZE {
                for j in 0..WORD_SIZE {
                    if i + j == k {
                        sum += (q_bytes[i] as u32) * (c_bytes[j] as u32);
                    }
                }
            }
            cols.carry[k] = F::from_canonical_u32(sum / 256);
            carry = sum / 256;
        }

        // Trace Gen Logic for Sign
        let q_sign_bool = if is_signed && q_abs_val != 0 { b_sign ^ c_sign } else { false };
        let r_sign_bool = if is_signed && r_abs_val != 0 { b_sign } else { false };
        cols.q_sign = F::from_bool(q_sign_bool);
        cols.r_sign = F::from_bool(r_sign_bool);

        // IMPORTANT: We use the 'event.a' directly for the output column.
        // In honest execution, event.a matches the calculated q/r logic.
        // In malicious tests, event.a is spoofed, causing a mismatch with q/r logic -> constraint
        // failure.
        cols.a = Word(event.a.to_le_bytes().map(F::from_canonical_u8));

        if !blu.as_any().is::<EmptyByteRecord>() {
            blu.add_u8_range_checks(&cols.b_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.c_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.q_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.r_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.diff.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u16_range_checks(&cols.carry.map(|x| x.as_canonical_u32() as u16));
        }
    }
}

impl<F> BaseAir<F> for DivRemChip {
    fn width(&self) -> usize {
        NUM_DIV_REM_COLS
    }
}

impl<AB> Air<AB> for DivRemChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &DivRemCols<AB::Var> = (*local).borrow();

        let zero = AB::Expr::zero();
        let one = AB::Expr::one();
        let base = AB::F::from_canonical_u32(256);
        let two = AB::F::from_canonical_u32(2);

        let p32 = AB::Expr::from(AB::F::from_canonical_u32(268435454));

        let is_real = local.is_div_u + local.is_div_s + local.is_rem_u + local.is_rem_s;
        builder.assert_bool(is_real.clone());

        builder.when(is_real.clone()).assert_bool(local.is_div_u);
        builder.when(is_real.clone()).assert_bool(local.is_div_s);
        builder.when(is_real.clone()).assert_bool(local.is_rem_u);
        builder.when(is_real.clone()).assert_bool(local.is_rem_s);

        let is_signed = local.is_div_s + local.is_rem_s;

        builder.when(is_real.clone()).assert_bool(local.b_sign);
        builder.when(is_real.clone()).assert_bool(local.c_sign);
        builder.when(is_real.clone()).assert_bool(local.sign_xor);

        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.b_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.c_sign);

        // --- Core Math ---
        let mut m: Vec<AB::Expr> = vec![zero.clone(); WORD_SIZE];
        for i in 0..WORD_SIZE {
            for j in 0..WORD_SIZE {
                if i + j < WORD_SIZE {
                    m[i + j] = m[i + j].clone() + local.q_abs[i].into() * local.c_abs[j].into();
                }
            }
        }
        for i in 0..WORD_SIZE {
            let prev_carry = if i == 0 { zero.clone() } else { local.carry[i - 1].into() };
            let lhs = m[i].clone() + local.r_abs[i].into() + prev_carry;
            let rhs = local.b_abs[i].into() + local.carry[i].into() * base;
            builder.when(is_real.clone()).assert_eq(lhs, rhs);
        }

        // --- Inequality ---
        for i in 0..WORD_SIZE {
            let carry_in = if i == 0 { zero.clone() } else { local.diff_borrow[i - 1].into() };
            let extra = if i == 0 { one.clone() } else { zero.clone() };
            let lhs = local.r_abs[i].into() + local.diff[i].into() + extra + carry_in;
            let rhs = local.c_abs[i].into() + local.diff_borrow[i].into() * base;
            builder.when(is_real.clone()).assert_eq(lhs, rhs);
        }

        // --- Sign Logic ---
        let computed_xor =
            local.b_sign + local.c_sign - (AB::Expr::from(two) * local.b_sign * local.c_sign);
        builder.when(is_real.clone()).assert_eq(local.sign_xor, computed_xor);

        let q_abs_expr = word_to_expr::<AB>(&local.q_abs);
        let r_abs_expr = word_to_expr::<AB>(&local.r_abs);
        let expected_q_sign = is_signed.clone() * local.sign_xor;
        let expected_r_sign = is_signed.clone() * local.b_sign;

        builder.assert_eq(local.q_sign * q_abs_expr.clone(), expected_q_sign * q_abs_expr.clone());
        builder.assert_eq(local.r_sign * r_abs_expr.clone(), expected_r_sign * r_abs_expr.clone());

        // --- Range Checks ---
        builder.slice_range_check_u8(&local.b_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.c_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.q_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.r_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.diff.0, is_real.clone());
        builder.slice_range_check_u16(&local.carry, is_real.clone());
        for i in 0..WORD_SIZE {
            builder.when(is_real.clone()).assert_bool(local.diff_borrow[i]);
        }

        // --- Input Binding ---
        let b_expr = word_to_expr::<AB>(&local.b);
        let b_abs_expr = word_to_expr::<AB>(&local.b_abs);
        let c_expr = word_to_expr::<AB>(&local.c);
        let c_abs_expr = word_to_expr::<AB>(&local.c_abs);

        let term_b =
            b_abs_expr.clone() + local.b_sign * (p32.clone() - AB::Expr::from(two) * b_abs_expr);
        let term_c =
            c_abs_expr.clone() + local.c_sign * (p32.clone() - AB::Expr::from(two) * c_abs_expr);

        builder.when(is_real.clone()).assert_eq(b_expr, term_b);
        builder.when(is_real.clone()).assert_eq(c_expr, term_c);

        // --- Output Binding ---
        let a_expr = word_to_expr::<AB>(&local.a);

        let q_signed = q_abs_expr.clone() +
            local.q_sign * (p32.clone() - AB::Expr::from(two) * q_abs_expr.clone());
        let r_signed = r_abs_expr.clone() +
            local.r_sign * (p32.clone() - AB::Expr::from(two) * r_abs_expr.clone());

        let term_div = (local.is_div_s + local.is_div_u) * q_signed;
        let term_rem = (local.is_rem_s + local.is_rem_u) * r_signed;

        builder.assert_eq(a_expr, term_div + term_rem);

        // --- Instruction Binding ---
        let op_rem_u = AB::Expr::from_canonical_u32(Opcode::I32RemU.code());
        let op_div_u = AB::Expr::from_canonical_u32(Opcode::I32DivU.code());
        let op_rem_s = AB::Expr::from_canonical_u32(Opcode::I32RemS.code());
        let op_div_s = AB::Expr::from_canonical_u32(Opcode::I32DivS.code());

        let calculated_opcode = local.is_rem_u * op_rem_u +
            local.is_div_u * op_div_u +
            local.is_rem_s * op_rem_s +
            local.is_div_s * op_div_s;

        builder.receive_instruction(
            zero.clone(),
            zero.clone(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            zero.clone(),
            calculated_opcode,
            local.a,
            local.b,
            local.c,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::DivRemChip;
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

    fn compute_expected(opcode: Opcode, b: u32, c: u32) -> u32 {
        match opcode {
            Opcode::I32DivU => {
                if c == 0 {
                    0
                } else {
                    b / c
                }
            }
            Opcode::I32RemU => {
                if c == 0 {
                    b
                } else {
                    b % c
                }
            }
            Opcode::I32DivS => {
                if c == 0 {
                    0
                } else {
                    (b as i32).wrapping_div(c as i32) as u32
                }
            }
            Opcode::I32RemS => {
                if c == 0 {
                    b
                } else {
                    (b as i32).wrapping_rem(c as i32) as u32
                }
            }
            _ => 0,
        }
    }

    #[test]
    fn generate_trace_divrem() {
        let mut shard = ExecutionRecord::default();
        let mut divrem_events: Vec<AluEvent> = Vec::new();
        let opcodes = vec![Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];
        for _ in 0..50 {
            let b = thread_rng().gen::<u32>();
            let c = thread_rng().gen::<u32>();
            for op in &opcodes {
                let a = compute_expected(*op, b, c);
                divrem_events.push(AluEvent::new(0, *op, a, b, c, op.code()));
            }
        }
        shard.divrem_events = divrem_events;
        let chip = DivRemChip::default();
        let _trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
    }

    #[test]
    fn prove_babybear_divrem() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut divrem_events: Vec<AluEvent> = Vec::new();

        let instructions: Vec<(u32, u32)> = vec![
            // --- Basic Identity ---
            (0, 1),   // 0 / 1 = 0
            (1, 1),   // 1 / 1 = 1
            (50, 50), // x / x = 1
            // --- Basic Arithmetic ---
            (100, 3), // Remainder test (33, rem 1)
            (1, 2),   // Fraction (0, rem 1)
            // --- Unsigned Boundaries ---
            (u32::MAX, 1),            // Max / 1
            (u32::MAX, u32::MAX),     // Max / Max
            (u32::MAX, 2),            // Large / Small
            (u32::MAX, u32::MAX - 1), // Max / (Max-1)
            (1, u32::MAX),            // Small / Large
            // --- Signed Boundaries (Two's Complement) ---
            (i32::MIN as u32, 1),               // INT_MIN / 1
            (i32::MAX as u32, 1),               // INT_MAX / 1
            (i32::MIN as u32, i32::MIN as u32), // INT_MIN / INT_MIN
            // --- Signed Overflow Case ---
            // In WASM/x86, INT_MIN / -1 traps or overflows.
            // Your chip logic should handle it via wrapping (if standard rust wrapping is used)
            // or return the specific defined behavior.
            (i32::MIN as u32, u32::MAX), // INT_MIN / -1
            // --- Signed Negative Operands ---
            // -5 / 2   = -2 rem -1 (Rem sign follows Dividend)
            ((-5i32) as u32, 2),
            // 5 / -2   = -2 rem 1
            (5, (-2i32) as u32),
            // -5 / -2  = 2 rem -1
            ((-5i32) as u32, (-2i32) as u32),
            // --- Division by Zero ---
            (100, 0),
            (0, 0),
            (u32::MAX, 0),
        ];

        let opcodes = vec![Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];

        for (b, c) in instructions {
            for op in &opcodes {
                let a = compute_expected(*op, b, c);
                divrem_events.push(AluEvent::new(0, *op, a, b, c, op.code()));
            }
        }

        shard.divrem_events = divrem_events;
        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_divrem() {
        const NUM_TESTS: usize = 5;
        let opcodes = [Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];

        for _ in 0..NUM_TESTS {
            let b = thread_rng().gen::<u32>();
            // Avoid 0 or 1 for c to ensure wrapping_add(1) produces a truly invalid result
            let c = thread_rng().gen_range(2..u32::MAX);

            // Pick a random opcode to test coverage
            let op = opcodes[thread_rng().gen_range(0..opcodes.len())];

            let a_correct = compute_expected(op, b, c);
            let a_malicious = a_correct.wrapping_add(1);

            let program = Program::from_instrs(vec![
                Opcode::I32Const(b.into()),
                Opcode::I32Const(c.into()),
                op,
            ]);

            let stdin = SP1Stdin::new();
            type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

            let malicious_gen = move |prover: &P, record: &mut ExecutionRecord| {
                let mut mal_rec = record.clone();

                // 1. Corrupt the Chip Internal Trace
                // This forces the chip to witness "A_malicious" in its trace rows.
                // Since A_malicious != B / C, the arithmetic constraints in `eval` will fail.
                if !mal_rec.divrem_events.is_empty() {
                    mal_rec.divrem_events[0].a = a_malicious;
                }

                // 2. Corrupt the CPU Bus Event
                // We also update the CPU to expect A_malicious.
                // If we didn't do this, the test might fail earlier on a "Bus Interaction"
                // mismatch. We want to verify the Chip's *math* checks are working.
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
            let name = chip_name!(DivRemChip, BabyBear);

            // Assert that the failure comes specifically from the DivRemChip constraints
            assert!(result.is_err());
            assert!(result.unwrap_err().is_constraints_failing(&name));
        }
    }
}
