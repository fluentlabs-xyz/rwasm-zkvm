use crate::{
    air::SP1CoreAirBuilder,
    utils::{pad_rows_fixed, word_to_expr},
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32}; // Added Field for inverse()
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use rwasm::Opcode;
use rwasm_executor::{
    events::{AluEvent, ByteRecord, EmptyByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{
    air::{BaseAirBuilder, MachineAir},
    Word,
};

/// The total width of the chip trace.
pub const NUM_DIV_REM_COLS: usize = size_of::<DivRemCols<u8>>();

#[derive(Default)]
pub struct DivRemChip;

/// Layout for the Division/Remainder Chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct DivRemCols<T> {
    /// The Program Counter.
    pub pc: T,
    /// Input B (Dividend) - Raw 32-bit value.
    pub b: Word<T>,
    /// Input C (Divisor) - Raw 32-bit value.
    pub c: Word<T>,
    /// Output A (Quotient OR Remainder, depending on opcode) - Raw 32-bit value.
    pub a: Word<T>,

    // --- Selectors (One-Hot Encoded) ---
    pub is_div_u: T, // Unsigned Div
    pub is_div_s: T, // Signed Div
    pub is_rem_u: T, // Unsigned Rem
    pub is_rem_s: T, // Signed Rem

    /// Flag: 1 if operation is `INT_MIN / -1` (Signed Overflow).
    pub is_overflow: T,

    // --- Overflow Helpers ---
    pub is_b_int_min: T, // 1 if b == INT_MIN
    pub is_c_neg_one: T, // 1 if c == -1

    //1 if c == INT_MIN (Needed for magnitude check soundness)
    pub is_c_int_min: T,
    // --- Soundness Helpers for Signed Magnitude ---
    // These columns store 2 * (abs[3] - 128 * is_int_min).
    // Range checking these to u8 ensures abs[3] < 128 when not INT_MIN.
    pub b_check_msb: T,
    pub c_check_msb: T,

    // Inverse helpers to force the flags above:
    // if is_b_int_min is 0, b_diff_inv must be (b - INT_MIN)^-1
    pub b_diff_inv: T,         // (b - INT_MIN)^-1
    pub c_diff_inv: T,         // (c - -1)^-1
    pub c_int_min_diff_inv: T, // (c - INT_MIN)^-1

    // --- Sign Handling ---
    pub sign_xor: T, // Stores `b_sign ^ c_sign`
    pub b_sign: T,   // 1 if Dividend < 0
    pub c_sign: T,   // 1 if Divisor < 0
    pub q_sign: T,   // 1 if Quotient should be negative
    pub r_sign: T,   // 1 if Remainder should be negative

    // --- Absolute Values ---
    pub b_abs: Word<T>,
    pub c_abs: Word<T>,
    pub q_abs: Word<T>,
    pub r_abs: Word<T>,

    // --- Intermediate Calculation Columns ---
    pub carry: [T; WORD_SIZE],

    // --- Inequality Helper Columns (R < C) ---
    // We prove R < C by ensuring R + diff + 1 = C
    pub diff: Word<T>,
    pub diff_carry: [T; WORD_SIZE],
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

        let blu_batches: Vec<_> = input
            .divrem_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu = HashMap::new();
                let mut row = [F::zero(); NUM_DIV_REM_COLS];

                events.iter().for_each(|event| {
                    let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect();
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
        // [FIX] Use wrapped conversion for PC and all large u32s
        cols.pc = F::from_wrapped_u32(event.pc);
        let b_val = event.b;
        let c_val = event.c;

        cols.b = event.b.into();
        cols.c = event.c.into();

        // --- Overflow & Magnitude Helpers Generation ---
        let int_min = 0x8000_0000u32;
        let neg_one = 0xFFFF_FFFFu32;

        // [FIX] Helper to safely map large u32s to Field
        let to_field = |x: u32| F::from_wrapped_u32(x);

        // 1. Handle B == INT_MIN
        cols.is_b_int_min = F::from_bool(b_val == int_min);
        cols.b_diff_inv = if b_val == int_min {
            F::zero()
        } else {
            // [FIX] Use to_field here
            (to_field(b_val) - to_field(int_min)).inverse()
        };

        // 2. Handle C == -1
        cols.is_c_neg_one = F::from_bool(c_val == neg_one);
        cols.c_diff_inv = if c_val == neg_one {
            F::zero()
        } else {
            // [FIX] Use to_field here
            (to_field(c_val) - to_field(neg_one)).inverse()
        };

        // 3. Handle C == INT_MIN
        cols.is_c_int_min = F::from_bool(c_val == int_min);
        cols.c_int_min_diff_inv = if c_val == int_min {
            F::zero()
        } else {
            // [FIX] Use to_field here
            (to_field(c_val) - to_field(int_min)).inverse()
        };

        // --- Opcode Decoding ---
        let mut is_signed = false;
        match event.opcode {
            Opcode::I32DivS => {
                cols.is_div_s = F::one();
                is_signed = true;
                cols.is_overflow = F::from_bool(event.b == int_min && event.c == neg_one);
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

        // --- Handle Signed Inputs (Two's Complement) ---
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

        // --- Perform Arithmetic ---
        let (q_abs_val, r_abs_val) = if c_abs_val == 0 {
            (0, b_abs_val) // Trap behavior
        } else {
            (b_abs_val / c_abs_val, b_abs_val % c_abs_val)
        };

        cols.b_sign = F::from_bool(b_sign);
        cols.c_sign = F::from_bool(c_sign);
        cols.b_abs = b_abs_val.into();
        cols.c_abs = c_abs_val.into();
        cols.q_abs = q_abs_val.into();
        cols.r_abs = r_abs_val.into();

        // Unsigned numbers are allowed to have MSB set (e.g. 0xFFFFFFFF).
        if is_signed {
            let b_msb_byte = (b_abs_val >> 24) as u32;
            let c_msb_byte = (c_abs_val >> 24) as u32;

            let b_min_offset = if b_val == int_min { 128 } else { 0 };
            let c_min_offset = if c_val == int_min { 128 } else { 0 };

            cols.b_check_msb = F::from_wrapped_u32(2 * (b_msb_byte.wrapping_sub(b_min_offset)));
            cols.c_check_msb = F::from_wrapped_u32(2 * (c_msb_byte.wrapping_sub(c_min_offset)));
        } else {
            // For unsigned ops, we don't check MSB limits. Set to 0 to pass range check.
            cols.b_check_msb = F::zero();
            cols.c_check_msb = F::zero();
        }

        // --- Compute Carries ---
        let q_bytes = q_abs_val.to_le_bytes();
        let c_bytes = c_abs_val.to_le_bytes();
        let r_bytes = r_abs_val.to_le_bytes();
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

        // --- Compute Inequality Helper (R < C) ---
        if c_abs_val != 0 {
            let diff_val = c_abs_val.wrapping_sub(r_abs_val).wrapping_sub(1);
            cols.diff = diff_val.into();

            let d_bytes = diff_val.to_le_bytes();
            let sum_0 = (r_bytes[0] as u32) + (d_bytes[0] as u32) + 1;
            cols.diff_carry[0] = F::from_canonical_u32(sum_0 / 256);
            let mut d_carry = sum_0 / 256;

            for k in 1..WORD_SIZE {
                let sum = (r_bytes[k] as u32) + (d_bytes[k] as u32) + d_carry;
                cols.diff_carry[k] = F::from_canonical_u32(sum / 256);
                d_carry = sum / 256;
            }
        }

        // --- Result Signs & Output ---
        let q_sign_bool = if is_signed && q_abs_val != 0 { b_sign ^ c_sign } else { false };
        let r_sign_bool = if is_signed && r_abs_val != 0 { b_sign } else { false };
        cols.q_sign = F::from_bool(q_sign_bool);
        cols.r_sign = F::from_bool(r_sign_bool);

        cols.a = event.a.into();

        if !blu.as_any().is::<EmptyByteRecord>() {
            blu.add_u8_range_checks(&cols.b_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.c_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.q_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.r_abs.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u16_range_checks(&cols.carry.map(|x| x.as_canonical_u32() as u16));

            blu.add_u8_range_checks(&cols.diff.0.map(|x| x.as_canonical_u32() as u8));
            blu.add_u8_range_checks(&cols.diff_carry.map(|x| x.as_canonical_u32() as u8));

            blu.add_u8_range_checks(&[cols.b_check_msb.as_canonical_u32() as u8]);
            blu.add_u8_range_checks(&[cols.c_check_msb.as_canonical_u32() as u8]);
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
        let p32 = AB::Expr::from(AB::F::from_canonical_u32(268435454)); // 2^32 approx

        // 1. Selector Constraints
        let is_real = local.is_div_u + local.is_div_s + local.is_rem_u + local.is_rem_s;
        builder.assert_bool(is_real.clone());
        builder.when(is_real.clone()).assert_bool(local.is_div_u);
        builder.when(is_real.clone()).assert_bool(local.is_div_s);
        builder.when(is_real.clone()).assert_bool(local.is_rem_u);
        builder.when(is_real.clone()).assert_bool(local.is_rem_s);

        // 2. Sign Decomposition
        let is_signed = local.is_div_s + local.is_rem_s;
        builder.when(is_real.clone()).assert_bool(local.b_sign);
        builder.when(is_real.clone()).assert_bool(local.c_sign);
        builder.when(is_real.clone()).assert_bool(local.sign_xor);

        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.b_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.c_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.sign_xor);

        // 3. Core Arithmetic
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

        // 4. Internal Inequality (R < C)
        builder.slice_range_check_u8(&local.diff.0, is_real.clone());
        builder.slice_range_check_u8(&local.diff_carry, is_real.clone());

        let sum_0 = local.r_abs[0].into() + local.diff[0].into() + one.clone();
        let res_0 = local.c_abs[0].into() + local.diff_carry[0].into() * base;
        builder.when(is_real.clone()).assert_eq(sum_0, res_0);

        for i in 1..WORD_SIZE {
            let sum_i =
                local.r_abs[i].into() + local.diff[i].into() + local.diff_carry[i - 1].into();
            let res_i = local.c_abs[i].into() + local.diff_carry[i].into() * base;
            builder.when(is_real.clone()).assert_eq(sum_i, res_i);
        }
        builder.when(is_real.clone()).assert_zero(local.diff_carry[WORD_SIZE - 1]);

        // 5. Overflow & Helper Gadgets
        let int_min_val = AB::Expr::from(AB::F::from_wrapped_u32(0x8000_0000u32));
        let neg_one_val = p32.clone() - one.clone();
        let b_val = word_to_expr::<AB>(&local.b);
        let c_val = word_to_expr::<AB>(&local.c);

        // Gadget: is_b_int_min
        let diff_b = b_val.clone() - int_min_val.clone();
        builder.when(local.is_b_int_min).assert_zero(diff_b.clone());
        builder
            .when(is_real.clone())
            .assert_eq(diff_b * local.b_diff_inv, one.clone() - local.is_b_int_min);

        // Gadget: is_c_neg_one
        let diff_c = c_val.clone() - neg_one_val;
        builder.when(local.is_c_neg_one).assert_zero(diff_c.clone());
        builder
            .when(is_real.clone())
            .assert_eq(diff_c * local.c_diff_inv, one.clone() - local.is_c_neg_one);

        // [NEW] Gadget: is_c_int_min
        let diff_c_min = c_val.clone() - int_min_val.clone();
        builder.when(local.is_c_int_min).assert_zero(diff_c_min.clone());
        builder
            .when(is_real.clone())
            .assert_eq(diff_c_min * local.c_int_min_diff_inv, one.clone() - local.is_c_int_min);

        // Overflow Flag
        let expected_overflow = local.is_b_int_min * local.is_c_neg_one;
        builder.when(local.is_div_s).assert_eq(local.is_overflow, expected_overflow);
        builder.when(is_real.clone() - local.is_div_s).assert_zero(local.is_overflow);

        // 6. Sign Logic
        let computed_xor =
            local.b_sign + local.c_sign - (AB::Expr::from(two) * local.b_sign * local.c_sign);
        builder.when(is_signed.clone()).assert_eq(local.sign_xor, computed_xor);

        // Force Sign=1 if Magnitude is INT_MIN in Signed Mode
        // If we are in signed mode, and the magnitude is 2^31, the number MUST be negative.
        // (Positive 2^31 is not representable in 32-bit Two's Complement).
        builder
            .when(is_signed.clone())
            .when(local.is_b_int_min)
            .assert_one(local.b_sign);

        builder
            .when(is_signed.clone())
            .when(local.is_c_int_min)
            .assert_one(local.c_sign);

        let q_abs_expr = word_to_expr::<AB>(&local.q_abs);
        let expected_q_sign = is_signed.clone() * local.sign_xor;
        builder.assert_eq(local.q_sign * q_abs_expr.clone(), expected_q_sign * q_abs_expr.clone());

        let r_abs_expr = word_to_expr::<AB>(&local.r_abs);
        let expected_r_sign = is_signed.clone() * local.b_sign;
        builder.assert_eq(local.r_sign * r_abs_expr.clone(), expected_r_sign * r_abs_expr.clone());

        // 7. Range Checks
        builder.slice_range_check_u8(&local.b_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.c_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.q_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.r_abs.0, is_real.clone());
        builder.slice_range_check_u16(&local.carry, is_real.clone());

        // [NEW] Magnitude Soundness Constraints
        // Enforce: check_msb = 2 * (abs[3] - 128 * is_int_min)
        let msb_128 = AB::Expr::from_canonical_u8(128);

        let b_rhs =
            AB::Expr::from(two) * (local.b_abs[3].into() - msb_128.clone() * local.is_b_int_min);
        // Only enforce formula if Signed. If Unsigned, enforce Zero.
        builder.when(is_signed.clone()).assert_eq(local.b_check_msb, b_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.b_check_msb);

        let c_rhs =
            AB::Expr::from(two) * (local.c_abs[3].into() - msb_128.clone() * local.is_c_int_min);
        builder.when(is_signed.clone()).assert_eq(local.c_check_msb, c_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.c_check_msb);

        // Verify the check values are u8.
        // If they are u8 (0..255), then the term inside must be < 128.
        // This effectively proves abs[3] < 128 (when not int_min).
        builder.slice_range_check_u8(&[local.b_check_msb], is_real.clone());
        builder.slice_range_check_u8(&[local.c_check_msb], is_real.clone());

        // 8. Reconstruction
        let b_abs_expr = word_to_expr::<AB>(&local.b_abs);
        let c_abs_expr = word_to_expr::<AB>(&local.c_abs);
        let term_b =
            b_abs_expr.clone() + local.b_sign * (p32.clone() - AB::Expr::from(two) * b_abs_expr);
        let term_c =
            c_abs_expr.clone() + local.c_sign * (p32.clone() - AB::Expr::from(two) * c_abs_expr);

        builder.when(is_real.clone()).assert_eq(b_val, term_b);
        builder.when(is_real.clone()).assert_eq(c_val, term_c);

        // 9. Output Mux & Instruction
        let a_expr = word_to_expr::<AB>(&local.a);
        let q_signed = q_abs_expr.clone() +
            local.q_sign * (p32.clone() - AB::Expr::from(two) * q_abs_expr.clone());
        let r_signed = r_abs_expr.clone() +
            local.r_sign * (p32.clone() - AB::Expr::from(two) * r_abs_expr.clone());

        let is_div = local.is_div_u + local.is_div_s;
        let is_rem = local.is_rem_u + local.is_rem_s;
        builder.when(is_div).assert_eq(a_expr.clone(), q_signed);
        builder.when(is_rem).assert_eq(a_expr.clone(), r_signed);

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
                    0
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
                    0
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
            ((-5i32) as u32, 2),
            // 5 / -2   = -2 rem 1
            (5, (-2i32) as u32),
            // -5 / -2  = 2 rem -1
            ((-5i32) as u32, (-2i32) as u32),
            (-1i32 as u32, 0x80000000u32),
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
    #[test]
    fn test_divrem_divide_by_zero_trap_compliance() {
        // 1. Setup the inputs that MUST trap according to WASM Spec
        let b = 3232u32;
        let c = 0;
        let op = Opcode::I32DivS;

        let mut shard = ExecutionRecord::default();
        shard.divrem_events.push(AluEvent::new(0, op, 0x800000, b, c, op.code()));

        // 2. Generate Trace
        let chip = DivRemChip::default();
        let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // 3. Run the Prover
        // According to WASM spec, this operation is invalid.
        // Therefore, the STARK constraints MUST NOT be satisfiable.
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        let result = verify(&config, &chip, &mut challenger, &proof);
        assert!(result.is_err(), "Violation of WASM Spec: divide by zero did not trap!");
    }
}
