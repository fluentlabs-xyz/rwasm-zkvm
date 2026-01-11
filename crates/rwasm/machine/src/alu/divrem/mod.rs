//! DivRemChip: sound 32-bit division/remainder AIR over BabyBear.
//!
//! This chip verifies all four WASM 32-bit div/rem ops:
//!   - I32DivU, I32RemU   (unsigned)
//!   - I32DivS, I32RemS   (signed, two's complement)
//!
//! Core algebraic spec (on integers, not mod p):
//!   Let B, C, Q, R ∈ {0..2^32-1}, and interpret the 4-byte words
//!   `b_abs, c_abs, q_abs, r_abs` as unsigned integers:
//!     B_abs = Σ b_abs[i] * 256^i, etc.
//!
//!   For real rows with C_abs ≠ 0, constraints enforce that:
//!     1) B_abs = Q_abs * C_abs + R_abs             (base-256 carry chain)
//!     2) 0 ≤ R_abs < C_abs                         (inequality gadget)
//!
//!   Together with two's-complement sign gadgets, this implies:
//!   - For unsigned ops: a = Q_abs (div) or R_abs (rem).
//!   - For signed ops: a encodes ±Q_abs or ±R_abs in two's complement with the correct WASM sign
//!     rules.
//!
//!   For BabyBear (p < 2^31), every integer expression we use is
//!   strictly < p in absolute value, so "equality in the field" is
//!   literally "equality in Z" (no mod-p wrap can hide inconsistencies).
//!
//! Trap behavior:
//!   - Divide by zero: the inequality gadget has no solution when c_abs = 0, so any such row cannot
//!     satisfy AIR.
//!   - I32DivS(INT_MIN, -1): an explicit degree-2 trap constraint `is_divs_intmin * is_c_neg_one =
//!     0` forbids that combination, again leaving no satisfying assignment for that CPU state.

use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::{mem_index::UNIT, Opcode};
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

/// Total width of a DivRem row in the trace (in field elements).
///
/// This is the size of `DivRemCols<u8>`, not `DivRemCols<F>`, which
/// guarantees that layout is independent of the concrete field type.
pub const NUM_DIV_REM_COLS: usize = size_of::<DivRemCols<u8>>();

/// DivRemChip: verifies WASM I32DivU / I32DivS / I32RemU / I32RemS.
#[derive(Default)]
pub struct DivRemChip;

/// Column layout for DivRem AIR.
///
/// All `Word<T>` fields represent 32-bit values in little-endian bytes.
/// Range checks on bytes / carries are enforced via the ByteRecord and
/// `slice_range_check_*` calls in AIR.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct DivRemCols<T> {
    /// Program counter of the associated CPU instruction.
    pub pc: T,

    pub sp: T,

    /// The output operand.
    pub a: Word<T>,

    /// Opcode selectors for the four supported instructions.
    ///
    /// Exactly one of these is 1 for a real row; all are 0 for a dummy
    /// row. See `eval` for the booleanity and sum-to-one constraints.
    pub is_div_u: T,
    pub is_div_s: T,
    pub is_rem_u: T,
    pub is_rem_s: T,

    /// Flag and inverse for the gadget "c_abs == 1".
    ///
    /// We encode c_abs == 1 via a small linear aggregator:
    ///   agg1 = (c0 - 1) + 2*c1 + 4*c2 + 8*c3,  ci ∈ [0,255].
    ///
    /// Over BabyBear, agg1 ∈ [-1, 3824] ⊂ (-p, p),
    /// so agg1 == 0 (in the field) ⇔ agg1 == 0 (in Z) ⇔ c_abs = 1.
    ///
    /// `c_abs_one_inv` is 0 when `c_abs_is_one = 1` and `agg1^{-1}`
    /// otherwise, giving a two-way linkage:
    ///   - `c_abs_is_one = 1 => agg1 = 0`;
    ///   - `agg1 != 0 => c_abs_is_one = 0`.
    pub c_abs_is_one: T,
    pub c_abs_one_inv: T,

    /// Derived flag for signed ops: c == -1.
    ///
    /// Defined in AIR as:
    ///   is_c_neg_one = c_sign * c_abs_is_one
    ///
    /// Together with the two's-complement constraints for (c, c_abs),
    /// this exactly characterizes the integer value c = -1.
    pub is_c_neg_one: T,

    /// Flags for signed INT_MIN detection on b_abs / c_abs.
    ///
    /// INT_MIN = 0x80000000, which in little-endian bytes is:
    ///   [0, 0, 0, 128]
    ///
    /// Detection is split:
    ///   - Low three bytes are zero, via `agg_low = x0 + 2*x1 + 4*x2`.
    ///   - MSB = 128, enforced via `check_msb` and an extra u8 range check.
    ///
    /// Over BabyBear, these gadgets are two-way:
    ///   - If is_b_int_min = 1, then b_abs = INT_MIN;
    ///   - If b_abs = INT_MIN, any attempt to set is_b_int_min = 0 violates the MSB gadget (since
    ///     2*128 = 256 is not a u8).
    pub is_b_int_min: T,
    pub is_c_int_min: T,

    /// MSB magnitude gadgets for signed magnitudes.
    ///
    /// For signed rows, AIR enforces:
    ///   b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
    ///   c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
    ///
    /// with `b_check_msb, c_check_msb` range-checked as u8.
    ///
    /// - If is_x_int_min = 0: 0 ≤ 2 * x_abs[3] ≤ 255  => x_abs[3] ≤ 127. So magnitudes are
    ///   strictly < 2^31 in this case.
    ///
    /// - If is_x_int_min = 1: 0 ≤ 2 * (x_abs[3] - 128) ≤ 255  => x_abs[3] ∈ [128,255].
    ///
    /// Combined with the "low 3 bytes == 0" gadget and two's-complement
    /// relation between x and x_abs, this uniquely identifies INT_MIN.
    pub b_check_msb: T,
    pub c_check_msb: T,

    /// Sign bits for b, c, q, r, and XOR(b_sign, c_sign).
    ///
    /// All are boolean. For unsigned ops, AIR forces them to be zero.
    pub sign_xor: T,
    pub b_sign: T,
    pub c_sign: T,
    pub q_sign: T,
    pub r_sign: T,

    /// Zero/non-zero gadgets for q_abs and r_abs.
    ///
    /// We use:
    ///   agg(x) = x0 + 2*x1 + 4*x2 + 8*x3
    /// with xi ∈ [0,255], so 0 ≤ agg(x) ≤ 3825 < p.
    ///
    /// Thus:
    ///   - is_x_zero = 1  => agg(x) = 0 => all xi = 0;
    ///   - is_x_zero = 0  => agg(x) != 0 (via x_inv).
    ///
    /// This gives a field-safe 0/non-0 test without risking mod-p wrap.
    pub is_q_zero: T,
    pub q_inv: T,
    pub is_r_zero: T,
    pub r_inv: T,

    /// Flags for "signed op AND non-zero magnitude" for q and r.
    ///
    /// AIR enforces:
    ///   is_q_nz_signed = is_signed * (1 - is_q_zero)
    ///   is_r_nz_signed = is_signed * (1 - is_r_zero)
    ///
    /// These control where q_sign and r_sign constraints apply.
    pub is_q_nz_signed: T,
    pub is_r_nz_signed: T,

    /// Absolute values (magnitudes) of b, c, q, r as unsigned 32-bit words.
    ///
    /// For unsigned ops: x_abs = x.
    /// For signed ops: x_abs = x if sign=0, and x_abs = 2^32 - x if sign=1.
    /// The latter is enforced by the two's-complement carry chains.
    pub b_abs: Word<T>,
    pub c_abs: Word<T>,
    pub q_abs: Word<T>,
    pub r_abs: Word<T>,

    /// Carries for the equation:
    ///   b_abs = q_abs * c_abs + r_abs
    ///
    /// in base 256. This is the usual schoolbook multiplication plus
    /// addition with carries, implemented per-byte. All intermediate
    /// values are < 2^24, so these equations hold over Z, not mod p.
    pub carry: [T; WORD_SIZE],

    /// Difference word satisfying:
    ///   r_abs + diff + 1 = c_abs  (in base 256 with carries)
    ///
    /// This encodes 0 ≤ r_abs < c_abs when c_abs ≠ 0, again with all
    /// intermediate values < 2^16, so there is no mod-p aliasing.
    pub diff: Word<T>,

    /// Carry chain for the inequality equation r_abs + diff + 1 = c_abs.
    ///
    /// We store WORD_SIZE - 1 carries, and enforce the final carry to
    /// be 0 in AIR. This implies strict inequality r_abs < c_abs and
    /// also forbids c_abs = 0, since that would require an impossible
    /// equation r0 + 1 = 0 with r0 ∈ [0,255].
    pub diff_carry: [T; WORD_SIZE - 1],

    /// Two's-complement carry chains for (b, b_abs), (c, c_abs),
    /// and (a, q_abs / r_abs).
    ///
    /// For x ∈ {b, c}:
    ///   - If x_sign = 0:  x == x_abs.
    ///   - If x_sign = 1:  x + x_abs = 2^32, encoded via: x[0] + x_abs[0]          = carry[0] *
    ///     256 x[1] + x_abs[1] + carry[0] = carry[1] * 256 ... carry[3] = 1
    ///
    /// For (a, q_abs / r_abs) in signed ops we use the same pattern
    /// to encode a = -q_abs or a = -r_abs in two's complement.
    pub b_tc_carry: [T; WORD_SIZE],
    pub c_tc_carry: [T; WORD_SIZE],
    pub a_tc_carry: [T; WORD_SIZE],
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
        // For each ALU div/rem event, materialize one DivRem row.
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

            cols.pc = F::from_canonical_u32(event.pc);
            cols.sp = F::from_canonical_u32(event.sp);

            // Initialize cols with basic operands and flags derived from the current event.
            {
                cols.a = Word::from(event.a);
                cols.b = Word::from(event.b);
                cols.c = Word::from(event.c);
                cols.op_a_not_0 = F::one(); // <-- Added this line
                cols.is_real = F::one();
                cols.is_divu = F::from_bool(event.opcode == Opcode::I32DivU);
                cols.is_remu = F::from_bool(event.opcode == Opcode::I32RemU);
                cols.is_div = F::from_bool(event.opcode == Opcode::I32DivS);
                cols.is_rem = F::from_bool(event.opcode == Opcode::I32RemS);
                cols.is_c_0.populate(event.c);
            }

            let (quotient, remainder) = get_quotient_and_remainder(event.b, event.c, event.opcode);
            cols.quotient = Word::from(quotient);
            cols.remainder = Word::from(remainder);

            // Calculate flags for sign detection.
            {
                cols.rem_msb = F::from_canonical_u8(get_msb(remainder));
                cols.b_msb = F::from_canonical_u8(get_msb(event.b));
                cols.c_msb = F::from_canonical_u8(get_msb(event.c));
                cols.is_overflow_b.populate(event.b, i32::MIN as u32);
                cols.is_overflow_c.populate(event.c, -1i32 as u32);
                if is_signed_operation(event.opcode) {
                    cols.rem_neg = cols.rem_msb;
                    cols.b_neg = cols.b_msb;
                    cols.c_neg = cols.c_msb;
                    cols.is_overflow =
                        F::from_bool(event.b as i32 == i32::MIN && event.c as i32 == -1);
                    cols.abs_remainder = Word::from((remainder as i32).abs() as u32);
                    cols.abs_c = Word::from((event.c as i32).abs() as u32);
                    cols.max_abs_c_or_1 = Word::from(u32::max(1, (event.c as i32).abs() as u32));
                } else {
                    cols.abs_remainder = cols.remainder;
                    cols.abs_c = cols.c;
                    cols.max_abs_c_or_1 = Word::from(u32::max(1, event.c));
                }

                // Set the `alu_event` flags.
                cols.abs_c_alu_event = cols.c_neg * cols.is_real;
                cols.abs_rem_alu_event = cols.rem_neg * cols.is_real;

                // Insert the MSB lookup events.
                {
                    let words = [event.b, event.c, remainder];
                    let mut blu_events: Vec<ByteLookupEvent> = vec![];
                    for word in words.iter() {
                        let most_significant_byte = word.to_le_bytes()[WORD_SIZE - 1];
                        blu_events.push(ByteLookupEvent {
                            opcode: ByteOpcode::MSB,
                            a1: get_msb(*word) as u16,
                            a2: 0,
                            b: most_significant_byte,
                            c: 0,
                        });
                    }
                    output.add_byte_lookup_events(blu_events);
                }
            }

            // Calculate the modified multiplicity
            {
                cols.remainder_check_multiplicity = cols.is_real * (F::one() - cols.is_c_0.result);
            }

            // Calculate c * quotient + remainder.
            {
                let c_times_quotient = {
                    if is_signed_operation(event.opcode) {
                        (((quotient as i32) as i64) * ((event.c as i32) as i64)).to_le_bytes()
                    } else {
                        ((quotient as u64) * (event.c as u64)).to_le_bytes()
                    }
                };
                cols.c_times_quotient = c_times_quotient.map(F::from_canonical_u8);

                let remainder_bytes = {
                    if is_signed_operation(event.opcode) {
                        ((remainder as i32) as i64).to_le_bytes()
                    } else {
                        (remainder as u64).to_le_bytes()
                    }
                };

                // Add remainder to product.
                let mut carry = [0u32; 8];
                let base = 1 << BYTE_SIZE;
                for i in 0..LONG_WORD_SIZE {
                    let mut x = c_times_quotient[i] as u32 + remainder_bytes[i] as u32;
                    if i > 0 {
                        x += carry[i - 1];
                    }
                    carry[i] = x / base;
                    cols.carry[i] = F::from_canonical_u32(carry[i]);
                }

                // Range check.
                {
                    output.add_u8_range_checks(&quotient.to_le_bytes());
                    output.add_u8_range_checks(&remainder.to_le_bytes());
                    output.add_u8_range_checks(&c_times_quotient);
                }
            }

            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_DIV_REM_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIV_REM_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        // Compute byte-lookup tables in parallel over chunks of events.
        let chunk_size = core::cmp::max(input.divrem_events.len() / num_cpus::get(), 1);

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
        // This chip has no cross-row linking; everything is local.
        true
    }
}

impl<F> BaseAir<F> for DivRemChip {
    fn width(&self) -> usize {
        NUM_DIV_REM_COLS
    }
}

impl DivRemChip {
    /// Fill a single DivRem row from an ALU event.
    ///
    /// This function constructs a *canonical* witness consistent with
    /// the WASM semantics, including:
    ///   - signed/unsigned interpretation of b, c, a;
    ///   - Euclidean quotient/remainder (q_abs, r_abs);
    ///   - two's-complement magnitudes and sign bits;
    ///   - INT_MIN and c_abs == 1 gadgets.
    ///
    /// For malformed events (e.g., divide by zero), we still produce a
    /// syntactically valid row, but the AIR constraints will *not* be
    /// satisfiable, so any proof containing such rows will fail.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut DivRemCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_wrapped_u32(event.pc);

        let b_val = event.b;
        let c_val = event.c;
        let a_val = event.a;

        cols.b = b_val.into();
        cols.c = c_val.into();

        // ---------------------------------
        // 1. Opcode decoding
        // ---------------------------------
        let mut is_signed_bool = false;
        match event.opcode {
            Opcode::I32DivS => {
                cols.is_div_s = F::one();
                is_signed_bool = true;
            }
            Opcode::I32DivU => {
                cols.is_div_u = F::one();
            }
            Opcode::I32RemS => {
                cols.is_rem_s = F::one();
                is_signed_bool = true;
            }
            Opcode::I32RemU => {
                cols.is_rem_u = F::one();
            }
            _ => panic!("Invalid opcode for DivRemChip"),
        }

        // ---------------------------------
        // 2. Signed magnitudes for b and c
        //
        // Compute b_abs, c_abs as unsigned magnitudes:
        //   if signed & value < 0: x_abs = -x (mod 2^32), sign=1
        //   else:                  x_abs =  x,            sign=0
        // ---------------------------------
        let (b_abs_val, b_sign_bool) = if is_signed_bool && (b_val as i32) < 0 {
            (b_val.wrapping_neg(), true)
        } else {
            (b_val, false)
        };
        let (c_abs_val, c_sign_bool) = if is_signed_bool && (c_val as i32) < 0 {
            (c_val.wrapping_neg(), true)
        } else {
            (c_val, false)
        };

        cols.sign_xor = F::from_bool(b_sign_bool ^ c_sign_bool);

        // ---------------------------------
        // 3. Core arithmetic: q_abs, r_abs
        //
        // Compute Euclidean quotient and remainder on magnitudes:
        //   If c_abs != 0:
        //     q_abs = b_abs / c_abs
        //     r_abs = b_abs % c_abs
        //
        //   If c_abs == 0:
        //     we put a dummy (trap-like) witness; AIR will reject it.
        // ---------------------------------
        let (q_abs_val, r_abs_val) = if c_abs_val == 0 {
            // Dummy witness for divide-by-zero; constraints must reject.
            (0, b_abs_val)
        } else {
            (b_abs_val / c_abs_val, b_abs_val % c_abs_val)
        };

        cols.b_sign = F::from_bool(b_sign_bool);
        cols.c_sign = F::from_bool(c_sign_bool);
        cols.b_abs = b_abs_val.into();
        cols.c_abs = c_abs_val.into();
        cols.q_abs = q_abs_val.into();
        cols.r_abs = r_abs_val.into();

        // ---------------------------------
        // 4. Lightweight INT_MIN detection on b_abs / c_abs
        //
        // INT_MIN = 0x80000000 -> [0,0,0,128] in little-endian.
        // We check:
        //   - low 3 bytes are zero via a small linear aggregator;
        //   - msb is 128, enforced via b_check_msb/c_check_msb in AIR.
        // ---------------------------------
        let b_abs_bytes = b_abs_val.to_le_bytes();
        let c_abs_bytes = c_abs_val.to_le_bytes();

        let b_low_agg_val = (b_abs_bytes[0] as u32) +
            ((b_abs_bytes[1] as u32) << 1) +
            ((b_abs_bytes[2] as u32) << 2);

        let c_low_agg_val = (c_abs_bytes[0] as u32) +
            ((c_abs_bytes[1] as u32) << 1) +
            ((c_abs_bytes[2] as u32) << 2);

        let is_b_int_min_bool = b_low_agg_val == 0 && b_abs_bytes[3] == 128;
        let is_c_int_min_bool = c_low_agg_val == 0 && c_abs_bytes[3] == 128;

        cols.is_b_int_min = F::from_bool(is_b_int_min_bool);
        cols.is_c_int_min = F::from_bool(is_c_int_min_bool);

        // ---------------------------------
        // 5. MSB checks (magnitude soundness)
        //
        // For signed rows:
        //   b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
        //   c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
        //
        // These are later range-checked as u8 to force:
        //   - if is_x_int_min = 0, then x_abs[3] ≤ 127;
        //   - if is_x_int_min = 1, then x_abs[3] ≥ 128.
        // ---------------------------------
        if is_signed_bool {
            let b_msb_byte = b_abs_bytes[3] as u32;
            let c_msb_byte = c_abs_bytes[3] as u32;

            let b_min_offset = if is_b_int_min_bool { 128 } else { 0 };
            let c_min_offset = if is_c_int_min_bool { 128 } else { 0 };

            cols.b_check_msb = F::from_wrapped_u32(2 * (b_msb_byte.wrapping_sub(b_min_offset)));
            cols.c_check_msb = F::from_wrapped_u32(2 * (c_msb_byte.wrapping_sub(c_min_offset)));
        } else {
            cols.b_check_msb = F::zero();
            cols.c_check_msb = F::zero();
        }

        // ---------------------------------
        // 6. Gadget: c_abs == 1 (used to encode c == -1 for signed ops)
        //
        // Let bytes be [c0,c1,c2,c3]. Define:
        //   agg1 = (c0 - 1) + 2*c1 + 4*c2 + 8*c3.
        //
        // With ci ∈ [0,255]:
        //   agg1 ∈ [-1, 3824] ⊂ (-p, p) for BabyBear.
        // Thus:
        //   agg1 == 0  <=>  c_abs == [1,0,0,0]  <=> c_abs == 1.
        // ---------------------------------
        let c0_f = F::from_canonical_u32(c_abs_bytes[0] as u32);
        let c1_f = F::from_canonical_u32(c_abs_bytes[1] as u32);
        let c2_f = F::from_canonical_u32(c_abs_bytes[2] as u32);
        let c3_f = F::from_canonical_u32(c_abs_bytes[3] as u32);

        let two_f = F::from_canonical_u32(2);
        let four_f = F::from_canonical_u32(4);
        let eight_f = F::from_canonical_u32(8);

        let mut agg1 = c0_f - F::one();
        agg1 = agg1 + c1_f * two_f;
        agg1 = agg1 + c2_f * four_f;
        agg1 = agg1 + c3_f * eight_f;

        let is_c_abs_one_bool = c_abs_val == 1;
        cols.c_abs_is_one = F::from_bool(is_c_abs_one_bool);
        cols.c_abs_one_inv = if is_c_abs_one_bool {
            // When agg1 == 0 we set inverse to 0; AIR never multiplies
            // by this in that branch, so no division-by-zero occurs.
            F::zero()
        } else {
            agg1.inverse()
        };

        // ---------------------------------
        // 7. Derived flags:
        //    - is_c_neg_one : c == -1 (signed)
        // ---------------------------------
        let is_c_neg_one_bool = is_signed_bool && (c_val as i32) == -1;
        cols.is_c_neg_one = F::from_bool(is_c_neg_one_bool);

        // ---------------------------------
        // 8. Carries for b_abs = q_abs * c_abs + r_abs (byte-wise)
        //
        // We compute:
        //   m[k] = Σ_{i+j=k} q_bytes[i] * c_bytes[j]  for k < 4
        //   then enforce:
        //     m[k] + r_bytes[k] + carry[k-1] = b_abs[k] + carry[k]*256
        //
        // Bounds:
        //   m[k] ≤ 4 * 255^2  = 260100
        //   r_bytes[k] ≤ 255
        //   carry[k] ≤ 65535 (via u16 range check)
        //   => each side < 2^24 < p, so equality is over Z.
        // ---------------------------------
        let q_bytes = q_abs_val.to_le_bytes();
        let c_bytes = c_abs_val.to_le_bytes();
        let r_bytes = r_abs_val.to_le_bytes();

        let mut carry = 0u32;
        for k in 0..WORD_SIZE {
            let mut sum = carry + r_bytes[k] as u32;
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

        // ---------------------------------
        // 9. Safe small-agg zero gadgets for q_abs / r_abs
        //
        // agg(x) = x0 + 2*x1 + 4*x2 + 8*x3, with xi ∈ [0,255] so
        // 0 ≤ agg(x) ≤ 3825 < p.
        //
        // This lets us test "x_abs == 0" ⇔ "agg(x) == 0" over the field
        // with no mod-p aliasing, since the aggregator's image is a
        // small subset of Z_p.
        // ---------------------------------
        let q_zero_agg_val = (q_bytes[0] as u32) +
            ((q_bytes[1] as u32) << 1) +
            ((q_bytes[2] as u32) << 2) +
            ((q_bytes[3] as u32) << 3);

        let r_bytes_for_agg = r_abs_val.to_le_bytes();
        let r_zero_agg_val = (r_bytes_for_agg[0] as u32) +
            ((r_bytes_for_agg[1] as u32) << 1) +
            ((r_bytes_for_agg[2] as u32) << 2) +
            ((r_bytes_for_agg[3] as u32) << 3);

        let is_q_zero_bool = q_zero_agg_val == 0;
        let is_r_zero_bool = r_zero_agg_val == 0;

        cols.is_q_zero = F::from_bool(is_q_zero_bool);
        cols.q_inv = if is_q_zero_bool {
            F::zero()
        } else {
            F::from_canonical_u32(q_zero_agg_val).inverse()
        };

        cols.is_r_zero = F::from_bool(is_r_zero_bool);
        cols.r_inv = if is_r_zero_bool {
            F::zero()
        } else {
            F::from_canonical_u32(r_zero_agg_val).inverse()
        };

        // Flags: "signed op AND non-zero magnitude"
        cols.is_q_nz_signed = F::from_bool(is_signed_bool && !is_q_zero_bool);
        cols.is_r_nz_signed = F::from_bool(is_signed_bool && !is_r_zero_bool);

        // ---------------------------------
        // 10. Inequality helper: r_abs + diff + 1 = c_abs
        //
        // For c_abs != 0 we set:
        //   diff = c_abs - r_abs - 1  (mod 2^32)
        // and build carries so that:
        //   r_abs + diff + 1 = c_abs
        //
        // Bounds:
        //   each byte sum ≤ 255 + 255 + 255 = 765 < p;
        //   carries are u8, so per-digit equations hold over Z.
        //
        // This enforces 0 ≤ r_abs < c_abs and forbids c_abs = 0 (since
        // r0 + 1 = 0 has no solution with r0 ∈ [0,255]).
        // ---------------------------------
        if c_abs_val != 0 {
            let diff_val = c_abs_val.wrapping_sub(r_abs_val).wrapping_sub(1);
            cols.diff = diff_val.into();

            let d_bytes = diff_val.to_le_bytes();
            let sum_0 = (r_bytes[0] as u32) + (d_bytes[0] as u32) + 1;
            cols.diff_carry[0] = F::from_canonical_u32(sum_0 / 256);
            let mut d_carry = sum_0 / 256;

            for k in 1..WORD_SIZE {
                let sum = (r_bytes[k] as u32) + (d_bytes[k] as u32) + d_carry;
                if k < WORD_SIZE - 1 {
                    cols.diff_carry[k] = F::from_canonical_u32(sum / 256);
                }
                d_carry = sum / 256;
            }
            // Final carry d_carry is enforced to be 0 in AIR.
        } else {
            cols.diff = 0u32.into();
            cols.diff_carry = [F::zero(); WORD_SIZE - 1];
        }

        // ---------------------------------
        // 11. Signs for q and r (witness)
        //
        // For signed semantics:
        //   - q_sign = sign(b) XOR sign(c)   if q_abs != 0
        //   - r_sign = sign(b)               if r_abs != 0
        // ---------------------------------
        let q_sign_bool =
            if is_signed_bool && q_abs_val != 0 { b_sign_bool ^ c_sign_bool } else { false };
        let r_sign_bool = if is_signed_bool && r_abs_val != 0 { b_sign_bool } else { false };

        cols.q_sign = F::from_bool(q_sign_bool);
        cols.r_sign = F::from_bool(r_sign_bool);

        // Output word (already computed by the CPU).
        cols.a = a_val.into();

        // ---------------------------------
        // 12. Two's-complement carry chains for b, c, a
        //
        // For b and c:
        //   - When sign=0: AIR enforces x == x_abs.
        //   - When sign=1: AIR enforces x + x_abs = 2^32 via carries.
        //
        // For a:
        //   - For I32DivS: a + q_abs = 2^32 if q_sign = 1.
        //   - For I32RemS: a + r_abs = 2^32 if r_sign = 1.
        //
        // All intermediate sums are < 2^24, so these equalities are
        // again integer equalities, not just mod-p relations.
        // ---------------------------------
        let b_bytes = b_val.to_le_bytes();
        let c_bytes_full = c_val.to_le_bytes();
        let a_bytes = a_val.to_le_bytes();

        // b + b_abs
        let mut tc_carry = 0u32;
        for i in 0..WORD_SIZE {
            let sum = (b_bytes[i] as u32) + (b_abs_bytes[i] as u32) + tc_carry;
            cols.b_tc_carry[i] = F::from_canonical_u32(sum / 256);
            tc_carry = sum / 256;
        }

        // c + c_abs
        tc_carry = 0u32;
        for i in 0..WORD_SIZE {
            let sum = (c_bytes_full[i] as u32) + (c_abs_bytes[i] as u32) + tc_carry;
            cols.c_tc_carry[i] = F::from_canonical_u32(sum / 256);
            tc_carry = sum / 256;
        }

        // a + q_abs (for I32DivS), or a + r_abs (for I32RemS).
        tc_carry = 0u32;
        if is_signed_bool && matches!(event.opcode, Opcode::I32DivS) {
            // I32DivS: a_tc_carry encodes a + q_abs.
            for i in 0..WORD_SIZE {
                let sum = (a_bytes[i] as u32) + (q_bytes[i] as u32) + tc_carry;
                cols.a_tc_carry[i] = F::from_canonical_u32(sum / 256);
                tc_carry = sum / 256;
            }
        } else if is_signed_bool && matches!(event.opcode, Opcode::I32RemS) {
            // I32RemS: a_tc_carry encodes a + r_abs.
            for i in 0..WORD_SIZE {
                let sum = (a_bytes[i] as u32) + (r_bytes[i] as u32) + tc_carry;
                cols.a_tc_carry[i] = F::from_canonical_u32(sum / 256);
                tc_carry = sum / 256;
            }
        } else {
            // For unsigned ops, AIR ignores a_tc_carry; we fill it
            // deterministically to avoid leaky witnesses.
            cols.a_tc_carry = [F::zero(); WORD_SIZE];
        }

        // ---------------------------------
        // 13. Byte-range checks (if enabled)
        //
        // These are pushed to the global byte-lookup machinery, which
        // provides range checks for:
        //   - b_abs, c_abs, q_abs, r_abs bytes (u8);
        //   - carry, b_tc_carry, c_tc_carry, a_tc_carry (u16/u8).
        //
        // Together with the degree-1/2 constraints in AIR, they ensure
        // all "byte" and "carry" values lie in their intended ranges.
        // ---------------------------------
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

            blu.add_u16_range_checks(&cols.b_tc_carry.map(|x| x.as_canonical_u32() as u16));
            blu.add_u16_range_checks(&cols.c_tc_carry.map(|x| x.as_canonical_u32() as u16));
            blu.add_u16_range_checks(&cols.a_tc_carry.map(|x| x.as_canonical_u32() as u16));
        }
    }
}

impl<AB> Air<AB> for DivRemChip
where
    AB: SP1CoreAirBuilder,
{
    /// Evaluate all AIR constraints for a single DivRem row.
    ///
    /// This function assumes that `event_to_row` produced a syntactically
    /// valid row. It then enforces all the algebraic invariants described
    /// in the module-level docs, in particular:
    ///
    ///   - opcode selector consistency;
    ///   - two's-complement sign/magnitude relations;
    ///   - Euclidean division: B_abs = Q_abs * C_abs + R_abs,  0 ≤ R_abs < C_abs;
    ///   - correct signed/unsigned output a for each opcode;
    ///   - INT_MIN and c_abs==1 gadgets;
    ///   - explicit trap for I32DivS(INT_MIN, -1).
    ///
    /// All intermediate expressions are bounded strictly below the
    /// BabyBear modulus, so equality in the field is equality in Z.
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &DivRemCols<AB::Var> = (*local).borrow();

        let zero = AB::Expr::zero();
        let one = AB::Expr::one();
        let base = AB::F::from_canonical_u32(256);
        let two = AB::F::from_canonical_u32(2);
        let msb_128 = AB::Expr::from_canonical_u8(128);

        // -----------------------------
        // 1. Selector constraints
        //
        // is_real = sum of the four opcode flags.
        // We enforce:
        //   - is_real ∈ {0,1}
        //   - each opcode flag is boolean when is_real=1
        //   - at most one opcode flag is 1 (enforced by the CPU bus).
        // -----------------------------
        let is_real = local.is_div_u + local.is_div_s + local.is_rem_u + local.is_rem_s;
        builder.assert_bool(is_real.clone());
        builder.when(is_real.clone()).assert_bool(local.is_div_u);
        builder.when(is_real.clone()).assert_bool(local.is_div_s);
        builder.when(is_real.clone()).assert_bool(local.is_rem_u);
        builder.when(is_real.clone()).assert_bool(local.is_rem_s);

        // -----------------------------
        // 2. Sign decomposition & flags
        //
        // is_signed = 1 iff op ∈ {I32DivS, I32RemS}.
        // All sign flags and zero flags are boolean for real rows.
        // Unsigned ops must have zero sign bits.
        // -----------------------------
        let is_signed = local.is_div_s + local.is_rem_s;

        builder.when(is_real.clone()).assert_bool(local.b_sign);
        builder.when(is_real.clone()).assert_bool(local.c_sign);
        builder.when(is_real.clone()).assert_bool(local.sign_xor);
        builder.when(is_real.clone()).assert_bool(local.q_sign);
        builder.when(is_real.clone()).assert_bool(local.r_sign);
        builder.when(is_real.clone()).assert_bool(local.is_q_zero);
        builder.when(is_real.clone()).assert_bool(local.is_r_zero);
        builder.when(is_real.clone()).assert_bool(local.is_q_nz_signed);
        builder.when(is_real.clone()).assert_bool(local.is_r_nz_signed);

        // New flags: INT_MIN detection and c_abs==1 / c==-1.
        builder.when(is_real.clone()).assert_bool(local.is_b_int_min);
        builder.when(is_real.clone()).assert_bool(local.is_c_int_min);
        builder.when(is_real.clone()).assert_bool(local.c_abs_is_one);
        builder.when(is_real.clone()).assert_bool(local.is_c_neg_one);

        // Unsigned ops: all sign bits must be zero.
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.b_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.c_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.sign_xor);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.q_sign);
        builder.when(is_real.clone()).when_not(is_signed.clone()).assert_zero(local.r_sign);

        // -----------------------------
        // 3. Core arithmetic: b_abs = q_abs * c_abs + r_abs  (base 256)
        //
        // For i = 0..3:
        //   m[i] = Σ_{j+k=i} q_abs[j] * c_abs[k]
        //   m[i] + r_abs[i] + carry[i-1] = b_abs[i] + carry[i] * 256
        //
        // With all byte/carry ranges enforced, these equations hold
        // in Z, not just mod p, and encode the usual grade-school
        // multiplication plus addition with carries.
        // -----------------------------
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

            // The lower 4 bytes of c_times_quotient must match the lower 4 bytes of (c * quotient).
            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32Mul.code()),
                Word(lower_half),
                local.quotient,
                local.c,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.is_real,
            );

        let sum_0 = local.r_abs[0].into() + local.diff[0].into() + one.clone();
        let res_0 = local.c_abs[0].into() + local.diff_carry[0].into() * base;
        builder.when(is_real.clone()).assert_eq(sum_0, res_0);

        for i in 1..WORD_SIZE {
            let carry_in = local.diff_carry[i - 1].into();
            let carry_out = if i < WORD_SIZE - 1 {
                local.diff_carry[i].into()
            } else {
                zero.clone() // last carry is enforced to be 0
            };

            let upper_half: [AB::Expr; 4] = [
                local.c_times_quotient[4].into(),
                local.c_times_quotient[5].into(),
                local.c_times_quotient[6].into(),
                local.c_times_quotient[7].into(),
            ];

            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                opcode_for_upper_half,
                Word(upper_half),
                local.quotient,
                local.c,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.is_real,
            );
        }

        // -----------------------------
        // 5. INT_MIN detection & c_abs == 1 / c == -1
        // -----------------------------
        let is_b_int_min = local.is_b_int_min.into();
        let is_c_int_min = local.is_c_int_min.into();

        // Low 3 bytes zero for INT_MIN rows:
        //   agg_low(x) = x0 + 2*x1 + 4*x2.
        //
        // Enforced only when flag = 1. The reverse direction comes
        // from the MSB gadget plus u8 byte range checks.
        // b_abs
        let mut b_low_agg = zero.clone();
        for i in 0..3 {
            let coeff = AB::F::from_canonical_u32(1u32 << i); // 1,2,4
            b_low_agg += local.b_abs[i].into() * coeff;
        }
        builder.when(local.is_b_int_min).assert_zero(b_low_agg);

        // c_abs
        let mut c_low_agg = zero.clone();
        for i in 0..3 {
            let coeff = AB::F::from_canonical_u32(1u32 << i);
            c_low_agg += local.c_abs[i].into() * coeff;
        }
        builder.when(local.is_c_int_min).assert_zero(c_low_agg);

        // c_abs == 1 gadget:
        //   agg1 = (c0 - 1) + 2*c1 + 4*c2 + 8*c3.
        let mut c_abs_eq1_agg = local.c_abs[0].into() - one.clone();
        c_abs_eq1_agg += local.c_abs[1].into() * AB::F::from_canonical_u32(2);
        c_abs_eq1_agg += local.c_abs[2].into() * AB::F::from_canonical_u32(4);
        c_abs_eq1_agg += local.c_abs[3].into() * AB::F::from_canonical_u32(8);

        // If c_abs_is_one = 1 then agg1 must be 0.
        builder.when(local.c_abs_is_one).assert_zero(c_abs_eq1_agg.clone());

        // Two-way link: agg1 == 0  <=>  c_abs_is_one = 1.
        builder.when(is_real.clone()).assert_eq(
            c_abs_eq1_agg.clone() * local.c_abs_one_inv,
            one.clone() - local.c_abs_is_one,
        );

        // is_c_neg_one = c_sign AND (c_abs == 1) for real rows.
        builder
            .when(is_real.clone())
            .assert_eq(local.is_c_neg_one, local.c_sign * local.c_abs_is_one);

        // I32DivS(INT_MIN, -1) must trap; no valid row for this combination.
        // New form: is_div_s * is_b_int_min * is_c_neg_one = 0 (degree 3).
        builder.assert_zero(local.is_div_s * local.is_b_int_min * local.is_c_neg_one);

        // -----------------------------
        // 6. Sign logic (XOR + q_sign / r_sign)
        // -----------------------------
        // sign_xor = b_sign XOR c_sign = b + c - 2*b*c.
        let computed_xor =
            local.b_sign + local.c_sign - (AB::Expr::from(two) * local.b_sign * local.c_sign);
        builder.when(is_signed.clone()).assert_eq(local.sign_xor, computed_xor);

        // Force sign bit = 1 in INT_MIN cases for signed rows.
        builder.when(is_signed.clone()).when(is_b_int_min.clone()).assert_one(local.b_sign);
        builder.when(is_signed.clone()).when(is_c_int_min.clone()).assert_one(local.c_sign);

        // Small-agg zero gadgets: q_abs and r_abs.
        let mut q_zero_agg = zero.clone();
        let mut r_zero_agg = zero.clone();
        for i in 0..WORD_SIZE {
            let coeff = AB::F::from_canonical_u32(1u32 << i); // 1,2,4,8
            q_zero_agg += local.q_abs[i].into() * coeff;
            r_zero_agg += local.r_abs[i].into() * coeff;
        }

        // q_abs == 0  <=>  is_q_zero == 1
        builder.when(local.is_q_zero).assert_zero(q_zero_agg.clone());
        builder
            .when(is_real.clone())
            .assert_eq(q_zero_agg.clone() * local.q_inv, one.clone() - local.is_q_zero);

        // r_abs == 0  <=>  is_r_zero == 1
        builder.when(local.is_r_zero).assert_zero(r_zero_agg.clone());
        builder
            .when(is_real.clone())
            .assert_eq(r_zero_agg.clone() * local.r_inv, one.clone() - local.is_r_zero);

        // Signs must be zero when magnitude is zero.
        builder.when(local.is_q_zero).assert_zero(local.q_sign);
        builder.when(local.is_r_zero).assert_zero(local.r_sign);

        // is_q_nz_signed = is_signed * (1 - is_q_zero)
        let is_q_nz_signed_rhs = is_signed.clone() * (one.clone() - local.is_q_zero);
        builder.assert_eq(local.is_q_nz_signed, is_q_nz_signed_rhs);

        // is_r_nz_signed = is_signed * (1 - is_r_zero)
        let is_r_nz_signed_rhs = is_signed.clone() * (one.clone() - local.is_r_zero);
        builder.assert_eq(local.is_r_nz_signed, is_r_nz_signed_rhs);

        // For signed & non-zero quotient: q_sign = sign_xor.
        builder.when(local.is_q_nz_signed).assert_eq(local.q_sign, local.sign_xor);

        // For signed & non-zero remainder: r_sign = b_sign.
        builder.when(local.is_r_nz_signed).assert_eq(local.r_sign, local.b_sign);

        // -----------------------------
        // 7. Range checks & magnitude soundness
        //
        // These enforce:
        //   - all bytes are u8;
        //   - all carries are u16;
        //   - MSB gadgets force the correct sign-magnitude relation.
        // -----------------------------
        builder.slice_range_check_u8(&local.b_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.c_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.q_abs.0, is_real.clone());
        builder.slice_range_check_u8(&local.r_abs.0, is_real.clone());

        builder.slice_range_check_u16(&local.carry, is_real.clone());
        builder.when(is_real.clone()).assert_zero(local.carry[WORD_SIZE - 1]);

        // MSB magnitude gadget:
        //   b_check_msb = 2 * (b_abs[3] - 128 * is_b_int_min)
        //   c_check_msb = 2 * (c_abs[3] - 128 * is_c_int_min)
        let b_rhs =
            AB::Expr::from(two) * (local.b_abs[3].into() - msb_128.clone() * is_b_int_min.clone());
        builder.when(is_signed.clone()).assert_eq(local.b_check_msb, b_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.b_check_msb);

        let c_rhs =
            AB::Expr::from(two) * (local.c_abs[3].into() - msb_128.clone() * is_c_int_min.clone());
        builder.when(is_signed.clone()).assert_eq(local.c_check_msb, c_rhs);
        builder.when(is_real.clone() - is_signed.clone()).assert_zero(local.c_check_msb);

        // Range-check MSB helpers to force:
        //   - if is_x_int_min = 0: x_abs[3] ≤ 127;
        //   - if is_x_int_min = 1: x_abs[3] ≥ 128.
        builder.slice_range_check_u8(&[local.b_check_msb], is_real.clone());
        builder.slice_range_check_u8(&[local.c_check_msb], is_real.clone());

        builder.slice_range_check_u16(&local.b_tc_carry, is_real.clone());
        builder.slice_range_check_u16(&local.c_tc_carry, is_real.clone());
        builder.slice_range_check_u16(&local.a_tc_carry, is_real.clone());

        // -----------------------------
        // 8. Byte-level reconstruction of b and c
        //
        // Unsigned:
        //   b_abs == b, c_abs == c.
        //
        // Signed:
        //   - if sign=0: x_abs == x;
        //   - if sign=1: x + x_abs = 2^32 encoded via tc_carry.
        // -----------------------------
        // Unsigned: b_abs == b, c_abs == c.
        for i in 0..WORD_SIZE {
            builder
                .when(is_real.clone())
                .when_not(is_signed.clone())
                .assert_eq(local.b[i].into(), local.b_abs[i].into());
            builder
                .when(is_real.clone())
                .when_not(is_signed.clone())
                .assert_eq(local.c[i].into(), local.c_abs[i].into());
        }

        // Signed & sign = 0: still x_abs == x.
        for i in 0..WORD_SIZE {
            builder
                .when(is_signed.clone())
                .when_not(local.b_sign)
                .assert_eq(local.b[i].into(), local.b_abs[i].into());
            builder
                .when(is_signed.clone())
                .when_not(local.c_sign)
                .assert_eq(local.c[i].into(), local.c_abs[i].into());
        }

        // Signed & sign = 1: enforce x + x_abs = 0 (mod 256^4)
        // with final carry 1, i.e. x_abs = 2^32 - x.
        // b
        let mut lhs0 = local.b[0].into() + local.b_abs[0].into();
        let mut rhs0 = local.b_tc_carry[0].into() * base;
        builder.when(is_signed.clone()).when(local.b_sign).assert_eq(lhs0, rhs0);
        for i in 1..WORD_SIZE {
            lhs0 = local.b[i].into() + local.b_abs[i].into() + local.b_tc_carry[i - 1].into();
            rhs0 = local.b_tc_carry[i].into() * base;
            builder.when(is_signed.clone()).when(local.b_sign).assert_eq(lhs0, rhs0);
        }
        builder
            .when(is_signed.clone())
            .when(local.b_sign)
            .assert_one(local.b_tc_carry[WORD_SIZE - 1]);

        // c
        lhs0 = local.c[0].into() + local.c_abs[0].into();
        rhs0 = local.c_tc_carry[0].into() * base;
        builder.when(is_signed.clone()).when(local.c_sign).assert_eq(lhs0, rhs0);
        for i in 1..WORD_SIZE {
            lhs0 = local.c[i].into() + local.c_abs[i].into() + local.c_tc_carry[i - 1].into();
            rhs0 = local.c_tc_carry[i].into() * base;
            builder.when(is_signed.clone()).when(local.c_sign).assert_eq(lhs0, rhs0);
        }
        builder
            .when(is_signed.clone())
            .when(local.c_sign)
            .assert_one(local.c_tc_carry[WORD_SIZE - 1]);

        // -----------------------------
        // 9. Output mux & instruction bus
        //
        // Unsigned:
        //   - I32DivU: a == q_abs
        //   - I32RemU: a == r_abs
        //
        // Signed:
        //   - I32DivS: if q_sign = 0: a == q_abs if q_sign = 1: a + q_abs = 2^32 via a_tc_carry
        //   - I32RemS: if r_sign = 0: a == r_abs if r_sign = 1: a + r_abs = 2^32 via a_tc_carry
        //
        // Finally, we hook this row into the CPU instruction bus.
        // -----------------------------
        // Unsigned: a = q_abs or r_abs
        for i in 0..WORD_SIZE {
            builder.when(local.is_div_u).assert_eq(local.a[i].into(), local.q_abs[i].into());
            builder.when(local.is_rem_u).assert_eq(local.a[i].into(), local.r_abs[i].into());
        }

        // Signed division: if q_sign == 0, a == q_abs.
        for i in 0..WORD_SIZE {
            builder
                .when(local.is_div_s)
                .when_not(local.q_sign)
                .assert_eq(local.a[i].into(), local.q_abs[i].into());
        }

        // Range check remainder. (i.e., |remainder| < |c| when not is_c_0)
        {
            // For each of `c` and `rem`, assert that the absolute value is equal to the original
            // value, if the original value is non-negative or the minimum i32.
            for i in 0..WORD_SIZE {
                builder.when_not(local.c_neg).assert_eq(local.c[i], local.abs_c[i]);
                builder
                    .when_not(local.rem_neg)
                    .assert_eq(local.remainder[i], local.abs_remainder[i]);
            }
            // In the case that `c` or `rem` is negative, instead check that their sum is zero by
            // sending an AddEvent.
            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::zero() + AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32Add.code()),
                Word::zero::<AB>(),
                local.c,
                local.abs_c,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.abs_c_alu_event,
            );
            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::zero() + AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32Add.code()),
                Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]),
                local.remainder,
                local.abs_remainder,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.abs_rem_alu_event,
            );

        // Signed remainder: if r_sign == 0, a == r_abs.
        for i in 0..WORD_SIZE {
            builder
                .when(local.is_rem_s)
                .when_not(local.r_sign)
                .assert_eq(local.a[i].into(), local.r_abs[i].into());
        }

        // Signed remainder: if r_sign == 1, a + r_abs = 2^32 via a_tc_carry.
        let mut lhs_ar = local.a[0].into() + local.r_abs[0].into();
        let mut rhs_ar = local.a_tc_carry[0].into() * base;
        builder.when(local.is_rem_s).when(local.r_sign).assert_eq(lhs_ar, rhs_ar);
        for i in 1..WORD_SIZE {
            lhs_ar = local.a[i].into() + local.r_abs[i].into() + local.a_tc_carry[i - 1].into();
            rhs_ar = local.a_tc_carry[i].into() * base;
            builder.when(local.is_rem_s).when(local.r_sign).assert_eq(lhs_ar, rhs_ar);
        }
        builder.when(local.is_rem_s).when(local.r_sign).assert_one(local.a_tc_carry[WORD_SIZE - 1]);

        // Instruction bus hookup:
        //   tie opcode, pc, a, b, c into the global CPU trace.
        let op_rem_u = AB::Expr::from_canonical_u32(Opcode::I32RemU.code());
        let op_div_u = AB::Expr::from_canonical_u32(Opcode::I32DivU.code());
        let op_rem_s = AB::Expr::from_canonical_u32(Opcode::I32RemS.code());
        let op_div_s = AB::Expr::from_canonical_u32(Opcode::I32DivS.code());

        let calculated_opcode = local.is_rem_u * op_rem_u +
            local.is_div_u * op_div_u +
            local.is_rem_s * op_rem_s +
            local.is_div_s * op_div_s;

                // Set the least significant byte to 1 if is_c_0 is true.
                v[0] = local.is_c_0.result * one.clone() +
                    (one.clone() - local.is_c_0.result) * local.abs_c[0];

                // Set the remaining bytes to 0 if is_c_0 is true.
                for i in 1..WORD_SIZE {
                    v[i] = (one.clone() - local.is_c_0.result) * local.abs_c[i];
                }
                Word(v.try_into().unwrap_or_else(|_| panic!("Incorrect length")))
            };
            for i in 0..WORD_SIZE {
                builder.assert_eq(local.max_abs_c_or_1[i], max_abs_c_or_1[i].clone());
            }

            // Handle cases:
            // - If is_real == 0 then remainder_check_multiplicity == 0 is forced.
            // - If is_real == 1 then is_c_0_result must be the expected one, so
            //   remainder_check_multiplicity = (1 - is_c_0_result) * is_real.
            builder.assert_eq(
                (AB::Expr::one() - local.is_c_0.result) * local.is_real,
                local.remainder_check_multiplicity,
            );

            // the cleaner idea is simply remainder_check_multiplicity == (1 - is_c_0_result) *
            // is_real

            // Check that the absolute value selector columns are computed correctly.
            // This enforces the send multiplicities are zero when `is_real == 0`.
            builder.assert_eq(local.abs_c_alu_event, local.c_neg * local.is_real);
            builder.assert_eq(local.abs_rem_alu_event, local.rem_neg * local.is_real);

            // Dispatch abs(remainder) < max(abs(c), 1), this is equivalent to abs(remainder) <
            // abs(c) if not division by 0.
            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::zero() + AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32LtU.code()),
                Word::extend_expr::<AB>(AB::Expr::one()),
                local.abs_remainder,
                local.max_abs_c_or_1,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.remainder_check_multiplicity,
            );
        }

        // Check that the MSBs are correct.
        {
            let msb_pairs = [
                (local.b_msb, local.b[WORD_SIZE - 1]),
                (local.c_msb, local.c[WORD_SIZE - 1]),
                (local.rem_msb, local.remainder[WORD_SIZE - 1]),
            ];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            for msb_pair in msb_pairs.iter() {
                let msb = msb_pair.0;
                let byte = msb_pair.1;
                builder.send_byte(opcode, msb, byte, zero.clone(), local.is_real);
            }
        }

        // Range check all the bytes.
        {
            builder.slice_range_check_u8(&local.quotient.0, local.is_real);
            builder.slice_range_check_u8(&local.remainder.0, local.is_real);

            local.carry.iter().for_each(|carry| {
                builder.assert_bool(*carry);
            });

            builder.slice_range_check_u8(&local.c_times_quotient, local.is_real);
        }

        // Check that the flags are boolean.
        {
            let bool_flags = [
                local.is_div,
                local.is_divu,
                local.is_rem,
                local.is_remu,
                local.is_overflow,
                local.b_msb,
                local.rem_msb,
                local.c_msb,
                local.b_neg,
                local.rem_neg,
                local.c_neg,
                local.is_real,
                local.abs_c_alu_event,
                local.abs_rem_alu_event,
            ];

            for flag in bool_flags.iter() {
                builder.assert_bool(*flag);
            }
        }

        // Receive the arguments.
        {
            // Exactly one of the opcode flags must be on.
            // SAFETY: All selectors `is_divu`, `is_remu`, `is_div`, `is_rem` are checked to be
            // boolean. Each row has exactly one selector turned on, as their sum is
            // checked to be one. Therefore, the `opcode` matches the corresponding
            // opcode of the instruction.
            builder.assert_eq(
                one.clone(),
                local.is_divu + local.is_remu + local.is_div + local.is_rem,
            );

            let opcode = {
                let divu: AB::Expr = AB::F::from_canonical_u32(Opcode::I32DivU.code()).into();
                let remu: AB::Expr = AB::F::from_canonical_u32(Opcode::I32RemU.code()).into();
                let div: AB::Expr = AB::F::from_canonical_u32(Opcode::I32DivS.code()).into();
                let rem: AB::Expr = AB::F::from_canonical_u32(Opcode::I32RemS.code()).into();

                local.is_divu * divu +
                    local.is_remu * remu +
                    local.is_div * div +
                    local.is_rem * rem
            };

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

            builder.when(local.op_a_not_0).assert_one(local.is_real);
        }
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
        MachineProver, StarkGenericConfig,
    };

    /// Compute an expected result of div/rem, matching WASM semantics.
    ///
    /// Behavior:
    ///   - For unsigned ops:
    ///       - divide by zero => result 0 (CPU must trap separately).
    ///   - For signed ops:
    ///       - divide by zero => result 0 (CPU must trap separately);
    ///       - I32DivS(INT_MIN, -1) => wrapping_div, which is INT_MIN.
    ///
    /// The chip enforces the *trap* behavior separately for
    ///   - divide-by-zero; and
    ///   - I32DivS(INT_MIN, -1).
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

    /// Smoke test: build a trace from random div/rem events.
    #[test]
    fn generate_trace_divrem() {
        let mut shard = ExecutionRecord::default();
        shard.divrem_events =
            vec![AluEvent::new(0, 0, Opcode::I32DivU, 2, 17, 3, Opcode::I32DivU.code())];
        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("Generated trace for divrem {}", trace.width);
    }

    /// Full prove/verify test over BabyBear + Poseidon2 for a variety
    /// of boundary and random inputs, including INT_MIN, INT_MAX, and
    /// mixed-sign cases.
    #[test]
    fn prove_babybear_divrem() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        // b=17, c=3 -> quotient=5, remainder=2
        shard.divrem_events =
            vec![AluEvent::new(0, 0, Opcode::I32DivU, 0, 17, 3, Opcode::I32DivU.code())];

        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // Read the first (and only) row back into column form
        let mut row = [BabyBear::zero(); super::NUM_DIVREM_COLS];
        row.copy_from_slice(&trace.values[..super::NUM_DIVREM_COLS]);
        let cols: &super::DivRemCols<BabyBear> = row.as_slice().borrow();

        // Correct opcode flags
        assert_eq!(cols.is_divu, BabyBear::one());
        assert_eq!(cols.is_div, BabyBear::zero());

        // Quotient and remainder bytes
        assert_eq!(cols.quotient[0], BabyBear::from_canonical_u8(5));
        assert_eq!(cols.remainder[0], BabyBear::from_canonical_u8(2));
        for i in 1..sp1_primitives::consts::WORD_SIZE {
            assert_eq!(cols.quotient[i], BabyBear::zero());
            assert_eq!(cols.remainder[i], BabyBear::zero());
        }

        // c * q = 3 * 5 = 15 -> least-significant byte is 15
        assert_eq!(cols.c_times_quotient[0], BabyBear::from_canonical_u8(15));
    }

    #[test]
    fn divs_overflow_flags_and_remainder_zero() {
        use core::borrow::Borrow;
        use p3_field::AbstractField;

        let b = i32::MIN as u32;
        let c = (-1i32) as u32;

        let mut shard = ExecutionRecord::default();
        shard.divrem_events =
            vec![AluEvent::new(0, 0, Opcode::I32DivS, 0, b, c, Opcode::I32DivS.code())];

        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut row = [BabyBear::zero(); super::NUM_DIVREM_COLS];
        row.copy_from_slice(&trace.values[..super::NUM_DIVREM_COLS]);
        let cols: &super::DivRemCols<BabyBear> = row.as_slice().borrow();

        // Signed division and overflow are flagged
        assert_eq!(cols.is_div, BabyBear::one());
        assert_eq!(cols.is_overflow, BabyBear::one());

        // In the overflow case (INT_MIN / -1), remainder must be 0
        for i in 0..sp1_primitives::consts::WORD_SIZE {
            assert_eq!(cols.remainder[i], BabyBear::zero());
        }

        // Signs: b and c are both negative
        assert_eq!(cols.b_neg, BabyBear::one());
        assert_eq!(cols.c_neg, BabyBear::one());

        // max(abs(c), 1) == 1
        assert_eq!(cols.max_abs_c_or_1[0], BabyBear::one());
        for i in 1..sp1_primitives::consts::WORD_SIZE {
            assert_eq!(cols.max_abs_c_or_1[i], BabyBear::zero());
        }
    }

    #[test]
    fn div_by_zero_sets_quotient_to_all_ones() {
        use core::borrow::Borrow;
        use p3_field::AbstractField;

        let mut shard = ExecutionRecord::default();
        // Division by zero: c = 0. For DIV(U), quotient must be 0xffffffff
        shard.divrem_events =
            vec![AluEvent::new(0, 0, Opcode::I32DivU, 0, 123_456u32, 0u32, Opcode::I32DivU.code())];

        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut row = [BabyBear::zero(); super::NUM_DIVREM_COLS];
        row.copy_from_slice(&trace.values[..super::NUM_DIVREM_COLS]);
        let cols: &super::DivRemCols<BabyBear> = row.as_slice().borrow();

        // c==0 is detected
        assert_eq!(cols.is_c_0.result, BabyBear::one());

        // Quotient is 0xffffffff (all bytes 0xff)
        for i in 0..sp1_primitives::consts::WORD_SIZE {
            assert_eq!(cols.quotient[i], BabyBear::from_canonical_u8(u8::MAX));
        }

        // Remainder range check multiplicity is disabled when c==0
        assert_eq!(cols.remainder_check_multiplicity, BabyBear::zero());
    }

    fn neg(a: u32) -> u32 {
        u32::MAX - a + 1
    }

    /// Malicious-witness test:
    ///
    /// We:
    ///   1. Run a program that performs a single div/rem op.
    ///   2. Clone the ExecutionRecord and tweak:
    ///        - mal_rec.divrem_events[0].a
    ///        - mal_rec.cpu_events[2].res (+ write record)
    ///      by adding 1 to the correct result.
    ///   3. Ask the prover to generate traces from this *corrupted* record.
    ///
    /// The test asserts that proof verification fails *and* that the
    /// failure is localized to this chip (DivRem).
    #[test]
    fn test_malicious_divrem() {
        const NUM_TESTS: usize = 5;
        let opcodes = [Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];

        for _ in 0..NUM_TESTS {
            let b = thread_rng().gen::<u32>();
            // Avoid 0 or 1 for c so that wrapping_add(1) clearly corrupts the result.
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

            let malicious = move |prover: &P, record: &mut ExecutionRecord| {
                let mut malicious_record = record.clone();
                // The ALU op is the 3rd instruction (index 2)
                if malicious_record.cpu_events.len() > 2 {
                    // keep memory write consistent
                    if let Some(MemoryRecordEnum::Write(mut write_record)) =
                        malicious_record.cpu_events[2].res_record
                    {
                        write_record.value = op_a;
                    }
                    malicious_record.divrem_events[0].a = op_a;
                }

                // Corrupt CPU bus expectation as well to keep the CPU
                // trace "internally consistent" but inconsistent with
                // the true arithmetic reality.
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
            assert!(result.is_err());
            assert!(result.unwrap_err().is_constraints_failing(&name));
        }
    }

    /// Divide-by-zero trap compliance test.
    ///
    /// We manually construct an ExecutionRecord with:
    ///   - op = I32DivS
    ///   - c = 0
    ///   - some dummy a != correct result
    ///
    /// The DivRem AIR enforces:
    ///   - inequality gadget r_abs + diff + 1 = c_abs;
    ///   - c_abs == 0 implies an impossible equation r0 + 1 = 0 with r0 ∈ [0,255].
    ///
    /// Therefore, no satisfying witness can exist, and verification
    /// must fail.
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
