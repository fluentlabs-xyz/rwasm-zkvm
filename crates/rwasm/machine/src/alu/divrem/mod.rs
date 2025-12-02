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
use p3_field::{AbstractField, PrimeField32}; // PrimeField32 needed for inverse()
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
/// Computed from the size of `DivRemCols<u8>` so that the layout is
/// guaranteed to stay in sync with the struct.
pub const NUM_DIV_REM_COLS: usize = size_of::<DivRemCols<u8>>();

#[derive(Default)]
pub struct DivRemChip;

/// Layout for the Division/Remainder Chip.
///
/// Each row corresponds to a single ALU event for one of:
///   - I32DivU, I32DivS, I32RemU, I32RemS
///
/// The columns store:
///   - IO:   pc, b, c, a
///   - Flags: opcode selectors, special-case flags, sign bits
///   - Magnitudes: b_abs, c_abs, q_abs, r_abs
///   - Arithmetic helpers: carries for b_abs = q_abs * c_abs + r_abs, diff & diff_carry for
///     enforcing r_abs < c_abs
///   - Soundness gadgets: INT_MIN detection, MSB checks, zero gadgets.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct DivRemCols<T> {
    /// Program counter for this ALU event.
    pub pc: T,

    /// Input B (dividend), as raw 32-bit word (little-endian bytes).
    pub b: Word<T>,

    /// Input C (divisor), as raw 32-bit word (little-endian bytes).
    pub c: Word<T>,

    /// Output A (quotient or remainder, depending on opcode), raw 32-bit word.
    pub a: Word<T>,

    // -------------------------------------------------------------------------
    // Opcode Selector Columns (one-hot over the four Div/Rem variants)
    // -------------------------------------------------------------------------
    /// 1 iff opcode == I32DivU.
    pub is_div_u: T,
    /// 1 iff opcode == I32DivS.
    pub is_div_s: T,
    /// 1 iff opcode == I32RemU.
    pub is_rem_u: T,
    /// 1 iff opcode == I32RemS.
    pub is_rem_s: T,

    /// Flag: 1 iff operation is the signed overflow case `INT_MIN / -1`.
    /// This is tracked but not used for any further arithmetic decisions.
    pub is_overflow: T,

    // -------------------------------------------------------------------------
    // Overflow / Special-Case Helpers
    // -------------------------------------------------------------------------
    /// 1 iff b == INT_MIN (0x80000000). Used to detect signed overflow and
    /// also to normalize the MSB magnitude checks.
    pub is_b_int_min: T,

    /// 1 iff c == -1 (0xFFFFFFFF).
    pub is_c_neg_one: T,

    /// 1 iff c == INT_MIN (0x80000000). Needed to ensure sound sign handling
    /// and MSB checks for the divisor in signed mode.
    pub is_c_int_min: T,

    // -------------------------------------------------------------------------
    // Magnitude Soundness Helpers (MSB Constraints)
    //
    // We store:
    //    b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
    //    c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
    //
    // and range-check them as u8. This implies that the inner terms are in
    // [-128, 127]. Combined with the flags, this guarantees:
    //
    //   - For non-INT_MIN signed values, b_abs[3] < 128 and c_abs[3] < 128.
    //   - For INT_MIN, the MSB byte is allowed to be 128, and this is the only case where it can
    //     be 128.
    //
    // This closes the magnitude/sign ambiguity for 32-bit signed ints.
    // -------------------------------------------------------------------------
    pub b_check_msb: T,
    pub c_check_msb: T,

    // -------------------------------------------------------------------------
    // Inverse Helpers for the Special-Case Flags
    //
    // These are classic "is-equal" gadgets:
    //
    //   - Let diff = x - v.
    //   - If `is_flag == 1`, we enforce diff == 0.
    //   - If `is_flag == 0`, we enforce diff * diff_inv == 1.
    //
    // This forces the flag to be *exactly* the equality test `x == v`.
    // -------------------------------------------------------------------------
    /// Inverse for (b - INT_MIN) when b != INT_MIN; zero otherwise.
    pub b_diff_inv: T,

    /// Inverse for (c - (-1)) when c != -1; zero otherwise.
    pub c_diff_inv: T,

    /// Inverse for (c - INT_MIN) when c != INT_MIN; zero otherwise.
    pub c_int_min_diff_inv: T,

    // -------------------------------------------------------------------------
    // Sign Handling
    //
    // b_sign, c_sign, q_sign, r_sign are 1 iff the corresponding value is
    // considered negative in signed mode.
    //
    // sign_xor = b_sign ^ c_sign encodes the quotient sign for signed division.
    // -------------------------------------------------------------------------
    /// Stores `b_sign ^ c_sign` for convenience.
    pub sign_xor: T,

    /// 1 iff dividend is negative in signed mode.
    pub b_sign: T,

    /// 1 iff divisor is negative in signed mode.
    pub c_sign: T,

    /// 1 iff quotient should be negative in signed mode.
    pub q_sign: T,

    /// 1 iff remainder should be negative in signed mode.
    pub r_sign: T,

    // -------------------------------------------------------------------------
    // Zero-Checks for Output Magnitudes
    //
    // These are standard "is-zero" gadgets, used to force the signs of
    // q_abs / r_abs to be zero when the corresponding magnitude is zero.
    //
    // For x in {q_abs, r_abs}:
    //   - is_x_zero = 1 if x == 0, else 0
    //   - x_inv     = x^-1 if x != 0, else 0
    //   - Constraint: x * x_inv = 1 - is_x_zero
    //
    // Combined, this says:
    //   - if is_x_zero == 1, then x must be 0
    //   - if is_x_zero == 0, then x != 0 and x_inv is its inverse.
    // -------------------------------------------------------------------------
    pub is_q_zero: T,
    pub q_inv: T,
    pub is_r_zero: T,
    pub r_inv: T,

    // -------------------------------------------------------------------------
    // Absolute Values (Magnitude Representation)
    //
    // For signed operations, we represent:
    //   b = sign_extend(b_abs, b_sign)
    //   c = sign_extend(c_abs, c_sign)
    //   q = sign_extend(q_abs, q_sign)
    //   r = sign_extend(r_abs, r_sign)
    //
    // For unsigned operations, b_abs == b, c_abs == c, and signs are 0.
    // -------------------------------------------------------------------------
    pub b_abs: Word<T>,
    pub c_abs: Word<T>,
    pub q_abs: Word<T>,
    pub r_abs: Word<T>,

    // -------------------------------------------------------------------------
    // Core Arithmetic Helpers: Carries for b_abs = q_abs * c_abs + r_abs
    //
    // We work in base 256 (bytes) and use carry[i] as the carry into byte i+1.
    //
    // For each byte index i:
    //   let M[i] = sum_{k+l=i} q_abs[k] * c_abs[l]
    //   then:
    //       M[i] + r_abs[i] + carry[i-1] = b_abs[i] + base * carry[i]
    //
    // with carry[-1] = 0.
    // -------------------------------------------------------------------------
    pub carry: [T; WORD_SIZE],

    // -------------------------------------------------------------------------
    // Inequality Helper Columns (R < C)
    //
    // We prove R < C by explicitly constructing `diff` and `diff_carry` such
    // that:
    //
    //   R + diff + 1 = C   in base 256 with proper carries.
    //
    // Since diff is range-checked as a u32 (byte-wise), and all operations are
    // done with base-256 carries, this enforces 0 <= R < C.
    // -------------------------------------------------------------------------
    /// `diff` is the 32-bit word such that R + diff + 1 = C (if c_abs != 0).
    pub diff: Word<T>,

    /// Carry bytes for the equation R + diff + 1 = C.
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
        // For each div/rem event, fill a row with the corresponding witness.
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

        // Pad trace to a power-of-two length and convert to matrix form.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_DIV_REM_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIV_REM_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        // Aggregate byte-lookup usage (range checks) over batches of events.
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
        // All constraints are local; no cross-row linking required.
        true
    }
}

impl DivRemChip {
    /// Fill a single trace row (`cols`) from an `AluEvent`.
    ///
    /// This method computes all derived quantities:
    ///   - absolute values
    ///   - signs
    ///   - carry bytes
    ///   - inequality helpers
    ///   - special-case flags and performs the associated byte-range checks via `ByteRecord`.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut DivRemCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        // Helper to map u32 -> field using wrapped representation.
        let to_field = |x: u32| F::from_wrapped_u32(x);

        // Use wrapped conversion for PC and all 32-bit values to be robust
        // against BabyBear's modulus and keep semantics consistent.
        cols.pc = to_field(event.pc);
        let b_val = event.b;
        let c_val = event.c;

        cols.b = event.b.into();
        cols.c = event.c.into();

        // ---------------------------
        // Overflow & Magnitude Helpers
        // ---------------------------
        let int_min = 0x8000_0000u32; // INT_MIN as u32
        let neg_one = 0xFFFF_FFFFu32; // -1 as u32

        // 1) is_b_int_min & b_diff_inv encode (b == INT_MIN).
        cols.is_b_int_min = F::from_bool(b_val == int_min);
        cols.b_diff_inv = if b_val == int_min {
            F::zero()
        } else {
            (to_field(b_val) - to_field(int_min)).inverse()
        };

        // 2) is_c_neg_one & c_diff_inv encode (c == -1).
        cols.is_c_neg_one = F::from_bool(c_val == neg_one);
        cols.c_diff_inv = if c_val == neg_one {
            F::zero()
        } else {
            (to_field(c_val) - to_field(neg_one)).inverse()
        };

        // 3) is_c_int_min & c_int_min_diff_inv encode (c == INT_MIN).
        cols.is_c_int_min = F::from_bool(c_val == int_min);
        cols.c_int_min_diff_inv = if c_val == int_min {
            F::zero()
        } else {
            (to_field(c_val) - to_field(int_min)).inverse()
        };

        // ---------------------------
        // Opcode Decoding (Selectors)
        // ---------------------------
        let mut is_signed = false;
        match event.opcode {
            Opcode::I32DivS => {
                cols.is_div_s = F::one();
                is_signed = true;
                // Track the overflow case INT_MIN / -1 explicitly.
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

        // ---------------------------
        // Signed Magnitudes for b, c
        //
        // For signed ops:
        //   - If the raw i32 is negative, take two's-complement magnitude: abs = -raw
        //   - Otherwise, abs = raw
        //
        // For unsigned ops:
        //   - abs == raw, sign == 0.
        // ---------------------------
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

        // ---------------------------
        // Core Arithmetic: q_abs, r_abs
        //
        // For non-zero c_abs:
        //   b_abs = q_abs * c_abs + r_abs, with 0 <= r_abs < c_abs.
        //
        // For c_abs == 0:
        //   We fill some "trap" values, but this combination must not pass
        //   constraints in the WASM-compliance tests.
        // ---------------------------
        let (q_abs_val, r_abs_val) = if c_abs_val == 0 {
            (0, b_abs_val) // trap-like behavior; proof must fail for this case
        } else {
            (b_abs_val / c_abs_val, b_abs_val % c_abs_val)
        };

        // ---------------------------
        // Output Zero Checks (is_q_zero / is_r_zero)
        //
        // These are the classic "is-zero" gadgets for the magnitudes q_abs/r_abs.
        // ---------------------------
        cols.is_q_zero = F::from_bool(q_abs_val == 0);
        cols.q_inv = if q_abs_val == 0 { F::zero() } else { to_field(q_abs_val).inverse() };

        cols.is_r_zero = F::from_bool(r_abs_val == 0);
        cols.r_inv = if r_abs_val == 0 { F::zero() } else { to_field(r_abs_val).inverse() };

        // Store signs and magnitudes as words.
        cols.b_sign = F::from_bool(b_sign);
        cols.c_sign = F::from_bool(c_sign);
        cols.b_abs = b_abs_val.into();
        cols.c_abs = c_abs_val.into();
        cols.q_abs = q_abs_val.into();
        cols.r_abs = r_abs_val.into();

        // ---------------------------
        // MSB Checks (Magnitude Soundness)
        //
        // For signed operations, we want:
        //   - b_abs[3] < 128, unless b == INT_MIN
        //   - c_abs[3] < 128, unless c == INT_MIN
        //
        // We encode:
        //   b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
        //   c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
        //
        // and range-check them as u8.
        // For unsigned ops, simply set the checks to 0.
        // ---------------------------
        if is_signed {
            let b_msb_byte = (b_abs_val >> 24) as u32;
            let c_msb_byte = (c_abs_val >> 24) as u32;

            let b_min_offset = if b_val == int_min { 128 } else { 0 };
            let c_min_offset = if c_val == int_min { 128 } else { 0 };

            cols.b_check_msb = F::from_wrapped_u32(2 * (b_msb_byte.wrapping_sub(b_min_offset)));
            cols.c_check_msb = F::from_wrapped_u32(2 * (c_msb_byte.wrapping_sub(c_min_offset)));
        } else {
            cols.b_check_msb = F::zero();
            cols.c_check_msb = F::zero();
        }

        // ---------------------------
        // Carries for b_abs = q_abs * c_abs + r_abs (byte-wise)
        //
        // We compute the total sum for each byte k:
        //
        //   sum_k = r_bytes[k] + carry_{k-1} + sum_{i+j=k} q_bytes[i]*c_bytes[j]
        //
        // and store carry_k = floor(sum_k / 256).
        // ---------------------------
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

        // ---------------------------
        // Inequality Helper: R + diff + 1 = C
        //
        // If c_abs_val != 0, we set:
        //
        //   diff_val = c_abs_val - r_abs_val - 1
        //
        // and then compute its byte representation and carry chain so that
        // the constraints can enforce:
        //
        //   R + diff + 1 = C
        //
        // This forces 0 <= r_abs < c_abs.
        // ---------------------------
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

        // ---------------------------
        // Signs for q and r
        //
        // For signed ops:
        //   q_sign = b_sign ^ c_sign if q_abs != 0, else 0
        //   r_sign = b_sign          if r_abs != 0, else 0
        //
        // For unsigned ops: both are 0.
        // ---------------------------
        let q_sign_bool = if is_signed && q_abs_val != 0 { b_sign ^ c_sign } else { false };
        let r_sign_bool = if is_signed && r_abs_val != 0 { b_sign } else { false };
        cols.q_sign = F::from_bool(q_sign_bool);
        cols.r_sign = F::from_bool(r_sign_bool);

        // Output A as word.
        cols.a = event.a.into();

        // ---------------------------
        // Byte-range checks via ByteRecord (if enabled).
        // These ensure that:
        //   - all bytes of magnitudes are in [0, 255]
        //   - carries are in u16
        //   - diff and diff_carry are proper 8-bit values
        //   - check_msb values are also in u8 range
        // ---------------------------
        if !blu.is_dummy() {
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
        // Approximation of 2^32 in the field (used for two's complement).
        let p32 = AB::Expr::from(AB::F::from_canonical_u32(268435454));

        // ---------------------------------------------------------------------
        // 1. Selector Constraints
        //
        // is_real := sum of the four opcode selector bits.
        // We enforce:
        //   - is_real is boolean
        //   - each selector is boolean when is_real == 1
        // ---------------------------------------------------------------------
        let is_real = local.is_div_u + local.is_div_s + local.is_rem_u + local.is_rem_s;
        builder.assert_bool(is_real.clone());
        builder.when(is_real.clone()).assert_bool(local.is_div_u);
        builder.when(is_real.clone()).assert_bool(local.is_div_s);
        builder.when(is_real.clone()).assert_bool(local.is_rem_u);
        builder.when(is_real.clone()).assert_bool(local.is_rem_s);

        // ---------------------------------------------------------------------
        // 2. Sign Decomposition Flags
        //
        // is_signed := 1 iff this opcode is signed (DivS or RemS).
        // For real rows, b_sign, c_sign, sign_xor must be boolean.
        // For unsigned opcodes, they must be 0.
        // ---------------------------------------------------------------------
        let is_signed = local.is_div_s + local.is_rem_s;
        builder.when(is_real.clone()).assert_bool(local.b_sign);
        builder.when(is_real.clone()).assert_bool(local.c_sign);
        builder.when(is_real.clone()).assert_bool(local.sign_xor);

        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.b_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.c_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.sign_xor);

        // ---------------------------------------------------------------------
        // 3. Core Arithmetic: b_abs = q_abs * c_abs + r_abs  (byte-wise)
        //
        // In base 256:
        //   For each byte i:
        //     let M[i] = sum_{k+l=i} q_abs[k] * c_abs[l]
        //     then:
        //       M[i] + r_abs[i] + carry[i-1] = b_abs[i] + base * carry[i]
        //
        // with carry[-1] := 0.
        // ---------------------------------------------------------------------
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

        // ---------------------------------------------------------------------
        // 4. Internal Inequality (R < C)
        //
        // We have precomputed `diff` and `diff_carry` so that:
        //
        //   R + diff + 1 = C    (in base 256, with carries diff_carry)
        //
        // Concretely:
        //   byte 0:
        //     r0 + d0 + 1 = c0 + base * diff_carry[0]
        //   byte i>0:
        //     ri + di + diff_carry[i-1] = ci + base * diff_carry[i]
        //
        // and we require diff_carry[3] == 0.
        //
        // Combined with range checks on diff and diff_carry, this enforces
        // 0 <= R < C whenever c_abs != 0.
        // ---------------------------------------------------------------------
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

        // Final carry must be zero: ensures that the equality holds as a
        // 32-bit equality, not just modulo base^4.
        builder.when(is_real.clone()).assert_zero(local.diff_carry[WORD_SIZE - 1]);

        // ---------------------------------------------------------------------
        // 5. Overflow & Equality Helper Gadgets
        //
        // b_val, c_val are the reconstructed raw 32-bit words for b and c.
        // We enforce that:
        //   - is_b_int_min encodes (b == INT_MIN)
        //   - is_c_neg_one encodes (c == -1)
        //   - is_c_int_min encodes (c == INT_MIN)
        //
        // using standard "is-equal" gadgets with inverses.
        // ---------------------------------------------------------------------
        let int_min_val = AB::Expr::from(AB::F::from_wrapped_u32(0x8000_0000u32));
        let neg_one_val = p32.clone() - one.clone();
        let b_val = word_to_expr::<AB>(&local.b);
        let c_val = word_to_expr::<AB>(&local.c);

        // Gadget: is_b_int_min
        let diff_b = b_val.clone() - int_min_val.clone();
        // If flag is 1, then diff_b must be 0.
        builder.when(local.is_b_int_min).assert_zero(diff_b.clone());
        // If flag is 0, diff_b * b_diff_inv must be 1; this forces diff_b != 0.
        builder
            .when(is_real.clone())
            .assert_eq(diff_b * local.b_diff_inv, one.clone() - local.is_b_int_min);

        // Gadget: is_c_neg_one
        let diff_c = c_val.clone() - neg_one_val;
        builder.when(local.is_c_neg_one).assert_zero(diff_c.clone());
        builder
            .when(is_real.clone())
            .assert_eq(diff_c * local.c_diff_inv, one.clone() - local.is_c_neg_one);

        // Gadget: is_c_int_min
        let diff_c_min = c_val.clone() - int_min_val.clone();
        builder.when(local.is_c_int_min).assert_zero(diff_c_min.clone());
        builder
            .when(is_real.clone())
            .assert_eq(diff_c_min * local.c_int_min_diff_inv, one.clone() - local.is_c_int_min);

        // Overflow Flag: is_overflow should encode `is_b_int_min & is_c_neg_one`
        // only for the signed division instruction.
        let expected_overflow = local.is_b_int_min * local.is_c_neg_one;
        builder.when(local.is_div_s).assert_eq(local.is_overflow, expected_overflow);
        builder.when(is_real.clone() - local.is_div_s).assert_zero(local.is_overflow);

        // ---------------------------------------------------------------------
        // 6. Sign Logic
        //
        // For signed operations, we enforce:
        //
        //   sign_xor = b_sign ^ c_sign
        //
        // and use this to define q_sign. Additionally, for INT_MIN magnitudes,
        // we force the sign bits to 1 in signed mode to avoid ambiguity.
        // ---------------------------------------------------------------------
        let computed_xor =
            local.b_sign + local.c_sign - (AB::Expr::from(two) * local.b_sign * local.c_sign);
        builder.when(is_signed.clone()).assert_eq(local.sign_xor, computed_xor);

        // Force sign = 1 if magnitude is INT_MIN in signed mode.
        builder.when(is_signed.clone()).when(local.is_b_int_min).assert_one(local.b_sign);

        builder.when(is_signed.clone()).when(local.is_c_int_min).assert_one(local.c_sign);

        let q_abs_expr = word_to_expr::<AB>(&local.q_abs);
        let expected_q_sign = is_signed.clone() * local.sign_xor;
        // q_sign is only meaningful when q_abs != 0; we encode:
        //   q_sign * q_abs = expected_q_sign * q_abs
        builder.assert_eq(local.q_sign * q_abs_expr.clone(), expected_q_sign * q_abs_expr.clone());

        let r_abs_expr = word_to_expr::<AB>(&local.r_abs);
        let expected_r_sign = is_signed.clone() * local.b_sign;
        // Similarly for r_sign:
        //   r_sign * r_abs = expected_r_sign * r_abs
        builder.assert_eq(local.r_sign * r_abs_expr.clone(), expected_r_sign * r_abs_expr.clone());

        // ---------------------------------------------------------------------
        // 6.1. IsZero Gadget for q_abs
        //
        // Enforces:
        //   - if is_q_zero == 1, then q_abs == 0
        //   - if is_q_zero == 0, then q_abs != 0 and q_inv is its inverse
        // ---------------------------------------------------------------------
        let q_abs_val = word_to_expr::<AB>(&local.q_abs);
        builder.when(local.is_q_zero).assert_zero(q_abs_val.clone());
        builder
            .when(is_real.clone())
            .assert_eq(q_abs_val.clone() * local.q_inv, one.clone() - local.is_q_zero);

        // ---------------------------------------------------------------------
        // 6.2. IsZero Gadget for r_abs
        //
        // Same pattern as q_abs:
        //   - if is_r_zero == 1, then r_abs == 0
        //   - if is_r_zero == 0, then r_abs != 0 and r_inv is its inverse
        // ---------------------------------------------------------------------
        let r_abs_val = word_to_expr::<AB>(&local.r_abs);
        builder.when(local.is_r_zero).assert_zero(r_abs_val.clone());
        builder
            .when(is_real.clone())
            .assert_eq(r_abs_val.clone() * local.r_inv, one.clone() - local.is_r_zero);

        // ---------------------------------------------------------------------
        // 6.3. Signs Must Be Zero when Magnitude is Zero
        //
        // If q_abs == 0, we force q_sign == 0.
        // If r_abs == 0, we force r_sign == 0.
        // ---------------------------------------------------------------------
        builder.when(local.is_q_zero).assert_zero(local.q_sign);
        builder.when(local.is_r_zero).assert_zero(local.r_sign);

        // ---------------------------------------------------------------------
        // 7. Range Checks & Magnitude Soundness
        //
        // All magnitude bytes must be in u8, carries in u16.
        // For signed operations, we also enforce the MSB constraints:
        //
        //   b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
        //   c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
        //
        // and range-check these as u8.
        // ---------------------------------------------------------------------
        builder.slice_range_check_u8(&local.b_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.c_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.q_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.r_abs.0, is_real.clone());
        builder.slice_range_check_u16(&local.carry, is_real.clone());

        let msb_128 = AB::Expr::from_canonical_u8(128);

        let b_rhs =
            AB::Expr::from(two) * (local.b_abs[3].into() - msb_128.clone() * local.is_b_int_min);
        builder.when(is_signed.clone()).assert_eq(local.b_check_msb, b_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.b_check_msb);

        let c_rhs =
            AB::Expr::from(two) * (local.c_abs[3].into() - msb_128.clone() * local.is_c_int_min);
        builder.when(is_signed.clone()).assert_eq(local.c_check_msb, c_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.c_check_msb);

        builder.slice_range_check_u8(&[local.b_check_msb], is_real.clone());
        builder.slice_range_check_u8(&[local.c_check_msb], is_real.clone());

        // ---------------------------------------------------------------------
        // 8. Reconstruction of b and c from (abs, sign)
        //
        // For x in {b, c}:
        //
        //   x_val = x_abs + sign * (p32 - 2 * x_abs)
        //
        // which corresponds to:
        //   x_val = x_abs          if sign == 0
        //   x_val = x_abs - 2^32   if sign == 1
        //
        // up to the approximation 2^32 ~ p32 in the field. The MSB constraints
        // guarantee a unique representation.
        // ---------------------------------------------------------------------
        let b_abs_expr = word_to_expr::<AB>(&local.b_abs);
        let c_abs_expr = word_to_expr::<AB>(&local.c_abs);
        let term_b =
            b_abs_expr.clone() + local.b_sign * (p32.clone() - AB::Expr::from(two) * b_abs_expr);
        let term_c =
            c_abs_expr.clone() + local.c_sign * (p32.clone() - AB::Expr::from(two) * c_abs_expr);

        builder.when(is_real.clone()).assert_eq(b_val, term_b);
        builder.when(is_real.clone()).assert_eq(c_val, term_c);

        // ---------------------------------------------------------------------
        // 9. Output Mux & Instruction Bus
        //
        // We reconstruct q and r from (abs, sign) just like b and c, and then:
        //
        //   - if opcode is Div*, a == q_signed
        //   - if opcode is Rem*, a == r_signed
        //
        // Finally we feed the (pc, pc+DEFAULT_PC_INC, opcode, a, b, c, is_real)
        // tuple into the CPU instruction bus via `receive_instruction`.
        // ---------------------------------------------------------------------
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

        // Reconstruct the opcode from the one-hot selectors.
        let calculated_opcode = local.is_rem_u * op_rem_u +
            local.is_div_u * op_div_u +
            local.is_rem_s * op_rem_s +
            local.is_div_s * op_div_s;

        // Hook into the CPU instruction bus.
        builder.receive_instruction(
            zero.clone(),                                            // shard / context
            zero.clone(),                                            // cycle
            local.pc,                                                // pc
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC), // next_pc
            zero.clone(),                                            // gas or aux
            calculated_opcode,                                       // opcode
            local.a,                                                 // result
            local.b,                                                 // operand 1
            local.c,                                                 // operand 2
            zero.clone(),                                            // extra...
            zero.clone(),
            zero.clone(),
            is_real, // enabled flag
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

    /// Helper to compute the expected result of div/rem, matching WASM semantics
    /// with wrapping for signed operations and 0 on divide-by-zero.
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

    /// Smoke test: generate a trace from random div/rem events and ensure
    /// that trace generation doesn't panic.
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

    /// Prove & verify a batch of carefully chosen div/rem events including
    /// boundary cases: 0, 1, MAX, MIN, and some negative combinations.
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
            (100, 3), // 100 / 3 = 33 rem 1
            (1, 2),   // 1 / 2   = 0  rem 1
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
            ((-5i32) as u32, 2),                // -5 / 2
            (5, (-2i32) as u32),                // 5 / -2
            ((-5i32) as u32, (-2i32) as u32),   // -5 / -2
            (-1i32 as u32, 0x80000000u32),      // -1 / INT_MIN
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

    /// Malicious test: we corrupt the output `a` and also the CPU's expectation
    /// of it, and check that the DivRemChip's constraints fail specifically.
    #[test]
    fn test_malicious_divrem() {
        const NUM_TESTS: usize = 5;
        let opcodes = [Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];

        for _ in 0..NUM_TESTS {
            let b = thread_rng().gen::<u32>();
            // Avoid 0 or 1 for c so that wrapping_add(1) gives a truly invalid result.
            let c = thread_rng().gen_range(2..u32::MAX);

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

                // 1. Corrupt the DivRemChip's internal trace (A witness).
                if !mal_rec.divrem_events.is_empty() {
                    mal_rec.divrem_events[0].a = a_malicious;
                }

                // 2. Corrupt the CPU bus event to expect A_malicious as well. This ensures we are
                //    testing the chip's *math* constraints, not just a bus mismatch.
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

            // The failure must come specifically from DivRemChip constraints.
            assert!(result.is_err());
            assert!(result.unwrap_err().is_constraints_failing(&name));
        }
    }

    /// WASM compliance: division by zero must trap. We encode a "fake"
    /// event with c=0 and a bogus result, and check that the proof
    /// verification fails for this trace.
    #[test]
    fn test_divrem_divide_by_zero_trap_compliance() {
        // Inputs that MUST trap according to WASM spec.
        let b = 3232u32;
        let c = 0;
        let op = Opcode::I32DivS;

        let mut shard = ExecutionRecord::default();
        shard.divrem_events.push(AluEvent::new(0, op, 0x800000, b, c, op.code()));

        let chip = DivRemChip::default();
        let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // This proof MUST NOT verify.
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        let result = verify(&config, &chip, &mut challenger, &proof);
        assert!(result.is_err(), "Violation of WASM Spec: divide by zero did not trap!");
    }
}
