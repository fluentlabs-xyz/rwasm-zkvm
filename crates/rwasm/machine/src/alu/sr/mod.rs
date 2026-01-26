//! Logical and Arithmetic Right Shift (SRL / SRA) Verification.
//!
//! This chip verifies 32-bit WebAssembly right shifts:
//!
//! - `I32ShrU` (logical right shift, SRL)
//! - `I32ShrS` (arithmetic right shift, SRA)
//!
//! For an event `(opcode, a, b, c)` we want to enforce:
//!
//!   a = b >> c    (interpreted as 32-bit Wasm semantics)
//!
//! with:
//!   - logical shift for `I32ShrU`,
//!   - arithmetic shift for `I32ShrS` (sign bit is propagated).
//!
//! ## Decomposition of the shift
//!
//! Let `c_eff = c & 31` be the effective shift amount in Wasm (since shifts are mod 32).
//! We split `c_eff` into:
//!
//!   num_bits_to_shift  = c_eff % 8 ∈ {0, ..., 7}
//!   num_bytes_to_shift = c_eff / 8 ∈ {0, ..., 3}
//!
//! Then we can write:
//!
//!   b >> c_eff
//!   = (b >> (8 * num_bytes_to_shift)) >> num_bits_to_shift
//!
//! We:
//! 1. Perform a **byte-level shift** by `num_bytes_to_shift` on a 64-bit sign-extended version of
//!    `b` (for SRA) or zero-extended version (for SRL).
//! 2. Perform a **bit-level shift** by `num_bits_to_shift` using the `ShrCarry` byte table, which
//!    captures how bits flow between neighboring bytes when shifting right.
//!
//! The 32-bit result `a` is then recovered from the least significant 4 bytes of the
//! 64-bit shifted value.
//!
//! ## Sign extension model
//!
//! To handle SRA, we conceptually embed the 32-bit `b` into a 64-bit word:
//!
//! - For SRL (`I32ShrU`): zero-extend B64 = (b as u64)            // high 32 bits = 0
//!
//! - For SRA (`I32ShrS`): sign-extend B64 = (b as i32 as i64)     // high 32 bits all 0xff if sign
//!   bit = 1
//!
//! After sign-extension, we perform byte-level shifting on the 8 bytes of `B64`.
//!
//! ## Byte tables used
//!
//! We rely on a global ByteChip table that provides the following opcodes:
//!
//! - `MSB`: Given a byte `x`, returns `msb = (x >> 7) & 1`. Used here to tie `b_msb` to the most
//!   significant byte of `b`.
//!
//! - `ShrCarry`: Given a byte `x` and a bit shift `k ∈ {0..7}`, returns: (shifted, carry) =
//!   shr_carry(x, k) so that multi-byte right shifts can be reconstructed using: new_byte_i =
//!   shifted_i + carry_{i+1} * carry_multiplier
//!
//! - `ShiftMeta` (merged metadata table): For each low 8-bit shift operand `shift_lo = c & 0xff`:
//!
//! ```text
//!       masked   = shift_lo & 31         ∈ {0..31}  (effective shift)
//!       k        = masked & 7            ∈ {0..7}   (num_bits_to_shift)
//!       raw_cm   = 1u16 << (8 - k)       (carry_multiplier)
//! ```
//!
//! The table stores rows of the form:
//!
//! ```text
//!       ByteLookupEvent {
//!           opcode = ShiftMeta,
//!           a1     = raw_cm,             // carry_multiplier (u16)
//!           a2     = masked,             // masked shift ∈ {0..31}
//!           b      = arbitrary (unused here),
//!           c      = shift_lo
//!       }
//! ```
//!
//! This chip enforces that its internal columns
//!   - `num_bits_to_shift`
//!   - `num_bytes_to_shift`
//!   - `carry_multiplier` are consistent with the `ShiftMeta` table via a single lookup:
//!
//! ```text
//!     masked = num_bits_to_shift + 8 * num_bytes_to_shift
//! ```
//!
//! and then:
//!
//! ```text
//!     send_byte_pair(
//!         ShiftMeta,
//!         carry_multiplier,                  // a1
//!         masked,                            // a2
//!         b = 0,
//!         c = shift_lo                       // low byte of c
//!     )
//! ```
//!
//! ## Bit-level combination via ShrCarry
//!
//! Consider the 8 bytes after the byte-level shift, `byte_shift_result[i]` for i=0..7
//! (little-endian, i = 0 is LSB).
//!
//! For each byte index `i` (processed from MSB to LSB), `ShrCarry` gives:
//!
//! ```text
//!     (shifted_i, carry_i) = shr_carry(byte_shift_result[i], num_bits_to_shift)
//! ```
//!
//! Intuitively, `carry_{i+1}` encodes the low `num_bits_to_shift` bits that are shifted out
//! of the next more significant byte and should enter byte `i`.
//!
//! We then reconstruct the final shifted 64-bit value one byte at a time using:
//!
//! ```text
//!     combined_i = (shifted_i + carry_{i+1} * carry_multiplier) mod 256
//! ```
//!
//! where:
//!
//! ```text
//!     carry_multiplier = 1 << (8 - num_bits_to_shift)
//! ```
//!
//! The chip enforces that the least significant 4 bytes of this combined value match `a`,
//! the CPU's result, via AIR constraints.

use crate::{
    air::SP1CoreAirBuilder,
    bytes::utils::shr_carry,
    utils::{nb_bits_to_shift, nb_bytes_to_shift, next_power_of_two, zeroed_f_vec},
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::{mem_index::UNIT, Opcode};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{air::MachineAir, Word};

/// The number of main trace columns for `ShiftRightChip`.
pub const NUM_SHIFT_RIGHT_COLS: usize = size_of::<ShiftRightCols<u8>>();

/// The number of bytes necessary to represent a 64-bit integer.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE; // 8

/// Number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise right shifts for `I32ShrU` and `I32ShrS`.
#[derive(Default)]
pub struct ShiftRightChip;

/// Column layout for the chip.
///
/// For `T = u8`, the width breakdown is:
/// - pc: 1
/// - a, b, c: 3 * 4 = 12
/// - shift_by_n_bytes: 4
/// - byte_shift_result: 8
/// - shr_carry_output_carry: 8
/// - shr_carry_output_shifted_byte: 8
/// - b_msb: 1
/// - num_bits_to_shift, num_bytes_to_shift, carry_multiplier: 3
/// - is_srl, is_sra: 2
/// - Total = 47 columns.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftRightCols<T> {
    /// Program counter.
    pub pc: T,

    pub sp: T,

    /// Output operand: `a = b >> c` (32-bit).
    pub a: Word<T>,

    /// First input operand (shifted value).
    pub b: Word<T>,

    /// Second input operand (shift amount).
    pub c: Word<T>,

    /// Boolean selector array: `shift_by_n_bytes[i] = 1` iff `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    /// Result of the **byte-level shift** on the 64-bit sign/zero-extended `b`.
    pub byte_shift_result: [T; LONG_WORD_SIZE],

    /// `shr_carry` carry output for each byte of `byte_shift_result`.
    pub shr_carry_output_carry: [T; LONG_WORD_SIZE],

    /// `shr_carry` shifted-byte output for each byte of `byte_shift_result`.
    pub shr_carry_output_shifted_byte: [T; LONG_WORD_SIZE],

    /// Most significant bit of the 32-bit `b` (sign bit).
    pub b_msb: T,

    /// `num_bits_to_shift = (c & 31) % 8  ∈ {0..7}`.
    pub num_bits_to_shift: T,

    /// `num_bytes_to_shift = (c & 31) / 8 ∈ {0..3}`.
    pub num_bytes_to_shift: T,

    /// `carry_multiplier = 1 << (8 - num_bits_to_shift) ∈ {2,4,...,256}`.
    pub carry_multiplier: T,

    /// Flag: opcode is `I32ShrU`.
    pub is_srl: T,

    /// Flag: opcode is `I32ShrS`.
    pub is_sra: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftRightChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShiftRight".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.shift_right_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SHIFT_RIGHT_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SHIFT_RIGHT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_RIGHT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ShiftRightCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.shift_right_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    } else {
                        // Padding row: make it a valid "no-op" row.
                        // Enforce a valid selector: shift_by_n_bytes[0] = 1, others 0.
                        cols.shift_by_n_bytes[0] = F::one();
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_SHIFT_RIGHT_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.shift_right_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_right_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SHIFT_RIGHT_COLS];
                    let cols: &mut ShiftRightCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.shift_right_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftRightChip {
    /// Populate a single trace row from an `AluEvent` and record associated byte lookups.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftRightCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        //
        // 1. Basic wiring and opcode flags.
        //
        {
            cols.pc = F::from_canonical_u32(event.pc);
            cols.sp = F::from_canonical_u32(event.sp);
            cols.a = Word::from(event.a);
            cols.b = Word::from(event.b);
            cols.c = Word::from(event.c);

            // Sign bit of 32-bit b: (b >> 31) & 1.
            cols.b_msb = F::from_canonical_u32((event.b >> 31) & 1);

            cols.is_srl = F::from_bool(event.code == Opcode::I32ShrU.code());
            cols.is_sra = F::from_bool(event.code == Opcode::I32ShrS.code());

            // MSB lookup tying `b_msb` to the most significant byte of b.
            let most_significant_byte = event.b.to_le_bytes()[WORD_SIZE - 1];
            blu.add_byte_lookup_events(vec![ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: ((most_significant_byte >> 7) & 1) as u16,
                a2: 0,
                b: most_significant_byte,
                c: 0,
            }]);
        }

        //
        // 2. Compute shift decomposition and link it to ShiftMeta table.
        //
        let num_bytes_to_shift = nb_bytes_to_shift(event.c); // ∈ {0..3}
        let num_bits_to_shift = nb_bits_to_shift(event.c); // ∈ {0..7}

        cols.num_bytes_to_shift = F::from_canonical_u32(num_bytes_to_shift as u32);
        cols.num_bits_to_shift = F::from_canonical_u32(num_bits_to_shift as u32);

        // carry_multiplier = 1 << (8 - num_bits_to_shift).
        let raw_cm: u16 = 1u16 << (BYTE_SIZE as u16 - num_bits_to_shift as u16);
        cols.carry_multiplier = F::from_canonical_u32(raw_cm as u32);

        // masked shift: k = num_bits + 8 * num_bytes = (c & 31).
        let k = (num_bits_to_shift as u8) + BYTE_SIZE as u8 * (num_bytes_to_shift as u8);

        // Low byte of c, used as key in the ShiftMeta table.
        let shift_lo = (event.c & 0xff) as u8;

        // Single merged lookup into ShiftMeta:
        //  a1 = carry_multiplier, a2 = masked shift amount.
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::ShiftMeta,
            a1: raw_cm,
            a2: k,
            b: 0,
            c: shift_lo,
        });

        //
        // 3. Byte-level shift on sign/zero-extended 64-bit b.
        //
        let mut byte_shift_result = [0u8; LONG_WORD_SIZE];
        {
            // Mark the selector for num_bytes_to_shift.
            for i in 0..WORD_SIZE {
                cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
            }

            // Sign- or zero-extend b to 64 bits.
            let sign_extended_b = if event.code == Opcode::I32ShrS.code() {
                // Arithmetic right shift: replicate sign bit in high 32 bits.
                ((event.b as i32) as i64).to_le_bytes()
            } else {
                // Logical right shift: high 32 bits are zero.
                (event.b as u64).to_le_bytes()
            };

            // Byte-level shift: move bytes down by num_bytes_to_shift positions.
            // byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift]
            // for valid indices; other bytes remain 0.
            for i in 0..LONG_WORD_SIZE {
                if i + num_bytes_to_shift < LONG_WORD_SIZE {
                    byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift];
                }
            }
            cols.byte_shift_result = byte_shift_result.map(F::from_canonical_u8);
        }

        //
        // 4. Bit-level shift via ShrCarry and reconstruction of final bytes.
        //
        {
            let mut last_carry = 0u32;
            let mut shr_carry_output_carry = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_shifted_byte = [0u8; LONG_WORD_SIZE];

            // Process bytes from most significant (index 7) down to least (index 0).
            for i in (0..LONG_WORD_SIZE).rev() {
                let (shift, carry) = shr_carry(byte_shift_result[i], num_bits_to_shift as u8);

                // Record ShrCarry lookup for this byte.
                let byte_event = ByteLookupEvent {
                    opcode: ByteOpcode::ShrCarry,
                    a1: shift as u16,
                    a2: carry,
                    b: byte_shift_result[i],
                    c: num_bits_to_shift as u8,
                };
                blu.add_byte_lookup_event(byte_event);

                shr_carry_output_carry[i] = carry;
                shr_carry_output_shifted_byte[i] = shift;

                // Reconstruct the combined byte (mod 256):
                //
                //   combined_i = shifted_i + last_carry * carry_multiplier (mod 256).
                //
                // This matches the flow of bits across byte boundaries for a right shift.
                let combined = ((shift as u32 + last_carry * raw_cm as u32) & 0xff) as u8;

                // Debug sanity check: the low 4 bytes of the combined result
                // should match the CPU's 32-bit output `a`.
                if i < WORD_SIZE {
                    debug_assert_eq!(combined, (event.a >> (8 * i)) as u8);
                }

                last_carry = carry as u32;
            }

            cols.shr_carry_output_carry = shr_carry_output_carry.map(F::from_canonical_u8);
            cols.shr_carry_output_shifted_byte =
                shr_carry_output_shifted_byte.map(F::from_canonical_u8);

            // Range checks: all of these arrays must contain bytes.
            blu.add_u8_range_checks(&byte_shift_result);
            blu.add_u8_range_checks(&shr_carry_output_carry);
            blu.add_u8_range_checks(&shr_carry_output_shifted_byte);
        }
    }
}

impl<F> BaseAir<F> for ShiftRightChip {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_COLS
    }
}

impl<AB> Air<AB> for ShiftRightChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftRightCols<AB::Var> = (*local).borrow();
        let zero: AB::Expr = AB::F::zero().into();
        let one: AB::Expr = AB::F::one().into();

        // Real row indicator: we are either SRL or SRA.
        let is_real = local.is_sra + local.is_srl;

        //
        // 1. Basic boolean checks for flags.
        //
        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.b_msb);
        builder.assert_bool(is_real.clone());

        //
        // 2. MSB consistency: tie b_msb to most-significant byte via MSB table.
        //
        {
            let byte = local.b[WORD_SIZE - 1];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            let msb = local.b_msb;
            builder.send_byte(opcode, msb, byte, zero.clone(), is_real.clone());
        }

        //
        // 3. ShiftMeta lookup: enforce consistency with (num_bits, num_bytes, carry_multiplier).
        //
        {
            let opcode_shift_meta = AB::F::from_canonical_u32(ByteOpcode::ShiftMeta as u32);
            let shift_lo = local.c[0]; // low byte of c

            // masked = num_bits + 8 * num_bytes.
            let eight = AB::F::from_canonical_u32(8);
            let k_expr = local.num_bits_to_shift + local.num_bytes_to_shift * eight;

            // Enforce:
            //   a1 = carry_multiplier,
            //   a2 = masked = num_bits + 8 * num_bytes,
            //   c  = shift_lo.
            builder.send_byte_pair(
                opcode_shift_meta,
                local.carry_multiplier, // a1
                k_expr,                 // a2
                zero.clone(),           // b = 0 (unused dimension)
                shift_lo,               // c = low byte of c
                is_real.clone(),
            );
        }

        //
        // 4. Byte-level shift constraints for sign-extended b.
        //
        {
            // Leading bytes for SRA (sign extension): if SRA and sign bit = 1, we expect 0xff,
            // otherwise 0x00.
            let leading_byte = local.is_sra * local.b_msb * AB::Expr::from_canonical_u8(0xff);

            // Construct the conceptual 64-bit sign-/zero-extended b (8 bytes).
            let mut sign_extended_b: Vec<AB::Expr> = vec![];
            for i in 0..WORD_SIZE {
                sign_extended_b.push(local.b[i].into());
            }
            for _ in 0..WORD_SIZE {
                sign_extended_b.push(leading_byte.clone());
            }

            // Enforce: if shift_by_n_bytes[j] = 1, then
            //   byte_shift_result[i] = sign_extended_b[i + j]
            // for all valid i.
            for num_bytes_to_shift in 0..WORD_SIZE {
                for i in 0..(LONG_WORD_SIZE - num_bytes_to_shift) {
                    builder.when(local.shift_by_n_bytes[num_bytes_to_shift]).assert_eq(
                        local.byte_shift_result[i],
                        sign_extended_b[i + num_bytes_to_shift].clone(),
                    );
                }
            }
        }

        //
        // 5. Sanity checks on shift_by_n_bytes selectors.
        //
        {
            // Exactly one `shift_by_n_bytes[i]` must be 1.
            let sum_shift_by_n_bytes =
                local.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x);
            builder.assert_eq(sum_shift_by_n_bytes, one.clone());

            // If `shift_by_n_bytes[i] = 1` then `num_bytes_to_shift = i`.
            for i in 0..WORD_SIZE {
                builder
                    .when(local.shift_by_n_bytes[i])
                    .assert_eq(local.num_bytes_to_shift, AB::F::from_canonical_usize(i));
            }

            // Each selector is boolean.
            for shift_by_n_byte in local.shift_by_n_bytes.iter() {
                builder.assert_bool(*shift_by_n_byte);
            }
        }

        //
        // 6. Bit-level shift via ShrCarry and reconstruction of final 32-bit result.
        //
        {
            let opcode_shrcarry = AB::F::from_canonical_u32(ByteOpcode::ShrCarry as u32);

            // ShrCarry lookups for each byte of byte_shift_result.
            for i in (0..LONG_WORD_SIZE).rev() {
                builder.send_byte_pair(
                    opcode_shrcarry,
                    local.shr_carry_output_shifted_byte[i],
                    local.shr_carry_output_carry[i],
                    local.byte_shift_result[i],
                    local.num_bits_to_shift,
                    is_real.clone(),
                );
            }

            // Combine ShrCarry outputs to get the least significant 4 bytes of the
            // fully shifted 64-bit value, and enforce equality with `a`.
            //
            // Algebraically for i in 0..3:
            //
            //   v_i = shifted_i + carry_{i+1} * carry_multiplier
            //
            // (carry_multiplier comes from ShiftMeta and is validated by lookup.)
            for i in 0..WORD_SIZE {
                let mut v: AB::Expr = local.shr_carry_output_shifted_byte[i].into();
                if i + 1 < LONG_WORD_SIZE {
                    v = v + local.shr_carry_output_carry[i + 1] * local.carry_multiplier;
                }
                builder.when(is_real.clone()).assert_eq(local.a[i], v);
            }
        }

        //
        // 7. Range checks for all byte arrays.
        //
        {
            let long_words = [
                local.byte_shift_result,
                local.shr_carry_output_carry,
                local.shr_carry_output_shifted_byte,
            ];

            for long_word in long_words.iter() {
                builder.slice_range_check_u8(long_word, is_real.clone());
            }
        }

        //
        // 8. CPU wiring (receive_instruction_old).
        //
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            local.is_srl * AB::F::from_canonical_u32(Opcode::I32ShrU.code() as u32) +
                local.is_sra * AB::F::from_canonical_u32(Opcode::I32ShrS.code() as u32),
            local.a,
            local.b,
            local.c,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use std::borrow::BorrowMut;

    use crate::{
        alu::ShiftRightCols,
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};
    use rwasm_executor::{
        events::{AluEvent, MemoryRecordEnum},
        ExecutionRecord, Opcode, Program,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig, Val,
    };

    use super::ShiftRightChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.shift_right_events = vec![
            AluEvent::new(0, 0, Opcode::I32ShrU, 6, 12, 1, Opcode::I32ShrU.code()),
            AluEvent::new(0, 0, Opcode::I32ShrS, 6, 12, 1, Opcode::I32ShrS.code()),
        ];
        let chip = ShiftRightChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("trace.width {:?}", trace.width)
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let shifts = vec![
            (Opcode::I32ShrU, 0xffff8000, 0xffff8000, 0),
            (Opcode::I32ShrU, 0x7fffc000, 0xffff8000, 1),
            (Opcode::I32ShrU, 0x01ffff00, 0xffff8000, 7),
            (Opcode::I32ShrU, 0x0003fffe, 0xffff8000, 14),
            (Opcode::I32ShrU, 0x0001ffff, 0xffff8001, 15),
            (Opcode::I32ShrU, 0xffffffff, 0xffffffff, 0),
            (Opcode::I32ShrU, 0x7fffffff, 0xffffffff, 1),
            (Opcode::I32ShrU, 0x01ffffff, 0xffffffff, 7),
            (Opcode::I32ShrU, 0x0003ffff, 0xffffffff, 14),
            (Opcode::I32ShrU, 0x00000001, 0xffffffff, 31),
            (Opcode::I32ShrU, 0x21212121, 0x21212121, 0),
            (Opcode::I32ShrU, 0x10909090, 0x21212121, 1),
            (Opcode::I32ShrU, 0x00424242, 0x21212121, 7),
            (Opcode::I32ShrU, 0x00008484, 0x21212121, 14),
            (Opcode::I32ShrU, 0x00000000, 0x21212121, 31),
            (Opcode::I32ShrU, 0x21212121, 0x21212121, 0xffffffe0),
            (Opcode::I32ShrU, 0x10909090, 0x21212121, 0xffffffe1),
            (Opcode::I32ShrU, 0x00424242, 0x21212121, 0xffffffe7),
            (Opcode::I32ShrU, 0x00008484, 0x21212121, 0xffffffee),
            (Opcode::I32ShrU, 0x00000000, 0x21212121, 0xffffffff),
            (Opcode::I32ShrS, 0x00000000, 0x00000000, 0),
            (Opcode::I32ShrS, 0xc0000000, 0x80000000, 1),
            (Opcode::I32ShrS, 0xff000000, 0x80000000, 7),
            (Opcode::I32ShrS, 0xfffe0000, 0x80000000, 14),
            (Opcode::I32ShrS, 0xffffffff, 0x80000001, 31),
            (Opcode::I32ShrS, 0x7fffffff, 0x7fffffff, 0),
            (Opcode::I32ShrS, 0x3fffffff, 0x7fffffff, 1),
            (Opcode::I32ShrS, 0x00ffffff, 0x7fffffff, 7),
            (Opcode::I32ShrS, 0x0001ffff, 0x7fffffff, 14),
            (Opcode::I32ShrS, 0x00000000, 0x7fffffff, 31),
            (Opcode::I32ShrS, 0x81818181, 0x81818181, 0),
            (Opcode::I32ShrS, 0xc0c0c0c0, 0x81818181, 1),
            (Opcode::I32ShrS, 0xff030303, 0x81818181, 7),
            (Opcode::I32ShrS, 0xfffe0606, 0x81818181, 14),
            (Opcode::I32ShrS, 0xffffffff, 0x81818181, 31),
        ];
        let mut shift_events: Vec<AluEvent> = Vec::new();
        for t in shifts.iter() {
            shift_events.push(AluEvent::new(0, 0, t.0, t.1, t.2, t.3, t.0.code()));
        }
        let mut shard = ExecutionRecord::default();
        shard.shift_right_events = shift_events;
        let chip = ShiftRightChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_sr() {
        const NUM_TESTS: usize = 5;

        for opcode in [Opcode::I32ShrU, Opcode::I32ShrS] {
            for _ in 0..NUM_TESTS {
                let (correct_op_a, op_b, op_c) = if opcode == Opcode::I32ShrS {
                    let op_b = thread_rng().gen_range(0..u32::MAX);
                    let op_c = thread_rng().gen_range(0..u32::MAX) & 0x1F;
                    (op_b >> op_c, op_b, op_c)
                } else if opcode == Opcode::I32ShrU {
                    let op_b = thread_rng().gen_range(0..i32::MAX);
                    let op_c = thread_rng().gen_range(0..u32::MAX) & 0x1F;
                    ((op_b >> op_c) as u32, op_b as u32, op_c)
                } else {
                    unreachable!()
                };

                let op_a = thread_rng().gen_range(0..u32::MAX);
                assert_ne!(op_a, correct_op_a);

                let instructions = vec![
                    Opcode::I32Const(5u32.into()),
                    Opcode::I32Const(10u32.into()),
                    Opcode::I32Const(op_b.into()),
                    Opcode::I32Const(op_c.into()),
                    opcode,
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
                    if malicious_record.cpu_events.len() > 4 {
                        if let Some(MemoryRecordEnum::Write(mut write_record)) =
                            malicious_record.cpu_events[4].res_record
                        {
                            write_record.value = op_a as u32;
                        }
                    }
                    let mut traces = prover.generate_traces(&malicious_record);
                    let shift_right_chip_name = chip_name!(ShiftRightChip, BabyBear);
                    for (name, trace) in traces.iter_mut() {
                        if *name == shift_right_chip_name {
                            let first_row = trace.row_mut(0);
                            let first_row: &mut ShiftRightCols<BabyBear> = first_row.borrow_mut();
                            first_row.a = op_a.into();
                        }
                    }
                    traces
                };

                let result =
                    run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
                let shift_right_chip_name = chip_name!(ShiftRightChip, BabyBear);
                assert!(
                    result.is_err() &&
                        result.unwrap_err().is_constraints_failing(&shift_right_chip_name)
                );
            }
        }
    }
}
