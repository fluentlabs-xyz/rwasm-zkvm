//! Logical Left Shift (SLL) Verification.
//!
//! This chip verifies 32-bit WebAssembly left shifts:
//!
//! - `I32Shl` (logical left shift)
//!
//! For an event `(opcode, a, b, c)` we want to enforce:
//!
//!   a = b << c    (interpreted as 32-bit Wasm semantics)
//!
//! where the shift amount is reduced modulo 32:
//!
//!   c_eff = c & 31
//!
//! ## Decomposition of the shift
//!
//! We split the effective shift amount `c_eff` into:
//!
//!   num_bits_to_shift  = c_eff % 8 ∈ {0, ..., 7}
//!   num_bytes_to_shift = c_eff / 8 ∈ {0, ..., 3}
//!
//! so that:
//!
//!   b << c_eff
//!   = (b << (8 * num_bytes_to_shift)) << num_bits_to_shift.
//!
//! We implement this as:
//!
//! 1. **Bit-level shift** (within the 32-bit word) by `num_bits_to_shift` via a base-256
//!    multiplication: `bit_shift_result = b * (1 << num_bits_to_shift)`.
//!
//! 2. **Byte-level shift** by `num_bytes_to_shift` using 2 boolean bits `(nb0, nb1)` encoding
//!    `num_bytes_to_shift = nb0 + 2 * nb1`. We then move bytes of `bit_shift_result` upward.
//!
//! ## ShiftMeta table (shared with ShiftRight)
//!
//! We reuse the same `ShiftMeta` byte table used by the right-shift chip. For each low 8-bit
//! shift operand `shift_lo = c & 0xff` it stores:
//!
//!   masked   = shift_lo & 31              ∈ {0..31}
//!   k        = masked & 7                 ∈ {0..7}   (bit-shift amount)
//!   raw_cm   = 1u16 << (8 - k)            (carry_multiplier, used by Shr)
//!
//! The table row is represented as:
//!
//!   ByteLookupEvent {
//!       opcode = ShiftMeta,
//!       a1     = raw_cm,                  // carry_multiplier (u16)
//!       a2     = masked,                  // effective shift (0..31)
//!       b      = 0,
//!       c      = shift_lo,
//!   }
//!
//! This chip enforces that its internal columns
//!   - `num_bits_to_shift`
//!   - `num_bytes_to_shift`
//!   - `carry_multiplier`
//!
//! are consistent with the `ShiftMeta` table via:
//!
//!   masked = num_bits_to_shift + 8 * num_bytes_to_shift
//!
//! and a single lookup:
//!
//!   send_byte_pair(
//!       ShiftMeta,
//!       carry_multiplier,                 // a1
//!       masked,                           // a2
//!       b = 0,
//!       c = shift_lo,
//!   )
//!
//! Note: `carry_multiplier` is unused in the SLL algebra; it is present only to reuse the same
//! table as `ShiftRightChip`.
//!
//! ## Column design
//!
//! Compared to the older SLL chip, this version:
//! - Removes `c_least_sig_byte[8]`
//! - Removes `shift_by_n_bits[8]`
//! - Removes `shift_by_n_bytes[4]`
//!
//! and replaces `num_bytes_to_shift` with two boolean bits `nb0, nb1` plus a small selector
//! gadget to implement the byte shift with max polynomial degree ≤ 3.

use crate::{
    air::SP1CoreAirBuilder,
    utils::{nb_bits_to_shift, nb_bytes_to_shift, next_power_of_two, zeroed_f_vec},
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use rwasm::mem_index::UNIT;
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, EmptyByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{air::MachineAir, Word};

/// The number of main trace columns for `ShiftLeftChip`.
pub const NUM_SHIFT_LEFT_COLS: usize = size_of::<ShiftLeftCols<u8>>();

/// Number of bits in a byte.
pub const BYTE_SIZE: usize = 8;

/// A chip that implements 32-bit logical left shifts for `I32Shl`.
#[derive(Default)]
pub struct ShiftLeft;

/// Column layout for the chip.
///
/// For `T = u8`, the width breakdown is:
/// - pc: 1
/// - a, b, c: 3 * 4 = 12
/// - num_bits_to_shift, num_bytes_to_shift: 2
/// - nb0, nb1: 2
/// - bit_shift_multiplier, carry_multiplier: 2
/// - bit_shift_result: 4
/// - bit_shift_result_carry: 4
/// - is_real: 1
/// - Total = 28 columns.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftLeftCols<T> {
    /// Program counter.
    pub pc: T,

    pub sp: T,

    /// The output operand.
    pub a: Word<T>,

    /// First input operand (shifted value).
    pub b: Word<T>,

    /// Second input operand (shift amount).
    pub c: Word<T>,

    /// `num_bits_to_shift = (c & 31) % 8  ∈ {0..7}`.
    pub num_bits_to_shift: T,

    /// `num_bytes_to_shift = (c & 31) / 8 ∈ {0..3}`.
    pub num_bytes_to_shift: T,

    /// Low bit of `num_bytes_to_shift`, boolean.
    pub nb0: T,

    /// High bit of `num_bytes_to_shift`, boolean.
    pub nb1: T,

    /// `bit_shift_multiplier = 1 << num_bits_to_shift` (1,2,4,...,128).
    pub bit_shift_multiplier: T,

    /// `carry_multiplier = 1 << (8 - num_bits_to_shift)` (shared ShiftMeta usage).
    pub carry_multiplier: T,

    /// Result of the bit-level shift: `b * bit_shift_multiplier` in base-256.
    pub bit_shift_result: [T; WORD_SIZE],

    /// Carry bytes used in the base-256 multiplication.
    pub bit_shift_result_carry: [T; WORD_SIZE],

    /// Boolean flag whether this row corresponds to a real I32Shl event.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftLeft {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShiftLeft".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.shift_left_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SHIFT_LEFT_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SHIFT_LEFT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_LEFT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ShiftLeftCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let event = &input.shift_left_events[idx];
                        let mut byte_lookup_events = EmptyByteRecord;
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    } else {
                        // Padding row: "no-op" shift (b=0, c=0, a=0), so constraints hold.
                        cols.bit_shift_multiplier = F::one();
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_SHIFT_LEFT_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.shift_left_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_left_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
                    let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.shift_left_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftLeft {
    /// Populate a single trace row from an `AluEvent` and record associated byte lookups.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftLeftCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        //
        // 1. Basic wiring.
        //
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();
        cols.a = Word(a.map(F::from_canonical_u8));
        cols.b = Word(b.map(F::from_canonical_u8));
        cols.c = Word(c.map(F::from_canonical_u8));
        cols.op_a_not_0 = F::from_bool(true);
        cols.is_real = F::one();

        //
        // 2. Compute shift decomposition (same helpers as ShiftRight).
        //
        let num_bytes_to_shift = nb_bytes_to_shift(event.c); // ∈ {0..3}
        let num_bits_to_shift = nb_bits_to_shift(event.c); // ∈ {0..7}

        cols.num_bytes_to_shift = F::from_canonical_u32(num_bytes_to_shift as u32);
        cols.num_bits_to_shift = F::from_canonical_u32(num_bits_to_shift as u32);

        // Encode num_bytes_to_shift as nb0 + 2 * nb1.
        let nb0 = num_bytes_to_shift & 1;
        let nb1 = (num_bytes_to_shift >> 1) & 1;
        cols.nb0 = F::from_canonical_u32(nb0 as u32);
        cols.nb1 = F::from_canonical_u32(nb1 as u32);

        // bit_shift_multiplier = 1 << num_bits_to_shift.
        let bit_mult = 1u32 << num_bits_to_shift;
        cols.bit_shift_multiplier = F::from_canonical_u32(bit_mult);

        // carry_multiplier = 1 << (8 - num_bits_to_shift) (only used for ShiftMeta lookup).
        let raw_cm: u16 = 1u16 << (BYTE_SIZE as u16 - num_bits_to_shift as u16);
        cols.carry_multiplier = F::from_canonical_u32(raw_cm as u32);

        // masked shift: k = num_bits + 8 * num_bytes = (c & 31).
        let masked = (num_bits_to_shift as u8) + BYTE_SIZE as u8 * (num_bytes_to_shift as u8);

        // Low byte of c, used as key in the ShiftMeta table.
        let shift_lo = (event.c & 0xff) as u8;

        //
        // 3. Bit-level shift: multiply b by bit_shift_multiplier in base 256.
        //
        let base = 1u32 << BYTE_SIZE;
        let mut carry = 0u32;
        let mut bit_shift_result = [0u8; WORD_SIZE];
        let mut bit_shift_result_carry = [0u8; WORD_SIZE];

        let b_bytes = event.b.to_le_bytes();
        for i in 0..WORD_SIZE {
            let v = b_bytes[i] as u32 * bit_mult + carry;
            carry = v / base;
            bit_shift_result[i] = (v % base) as u8;
            bit_shift_result_carry[i] = carry as u8;
        }

        cols.bit_shift_result = bit_shift_result.map(F::from_canonical_u8);
        cols.bit_shift_result_carry = bit_shift_result_carry.map(F::from_canonical_u8);

        // Range checks for the multiplication outputs.
        if !blu.is_dummy() {
            // Single lookup into ShiftMeta table: (carry_multiplier, masked, 0, shift_lo).
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::ShiftMeta,
                a1: raw_cm,
                a2: masked,
                b: 0,
                c: shift_lo,
            });
            blu.add_u8_range_checks(&bit_shift_result);
            blu.add_u8_range_checks(&bit_shift_result_carry);
        }

        //
        // 4. Debug sanity: check SLL result matches executor's a.
        //
        let nb = num_bytes_to_shift as usize;
        let a_bytes = event.a.to_le_bytes();

        for i in 0..WORD_SIZE {
            let expected = if i < nb { 0 } else { bit_shift_result[i - nb] };
            debug_assert_eq!(expected, a_bytes[i]);
        }
    }
}

impl<F> BaseAir<F> for ShiftLeft {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }
}

impl<AB> Air<AB> for ShiftLeft
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftLeftCols<AB::Var> = (*local).borrow();

        let zero: AB::Expr = AB::F::zero().into();
        let one: AB::Expr = AB::F::one().into();
        let base: AB::Expr = AB::F::from_canonical_u32(1 << BYTE_SIZE).into();
        let two: AB::Expr = AB::F::from_canonical_u32(2).into();

        let is_real = local.is_real;

        //
        // 1. Boolean checks.
        //
        builder.assert_bool(local.nb0);
        builder.assert_bool(local.nb1);
        builder.assert_bool(local.is_real);

        //
        // 2. ShiftMeta lookup consistency:
        //
        //    masked = num_bits_to_shift + 8 * num_bytes_to_shift
        //    send_byte_pair(ShiftMeta, carry_multiplier, masked, 0, shift_lo, is_real).
        //
        {
            let opcode_shift_meta = AB::F::from_canonical_u32(ByteOpcode::ShiftMeta as u32);
            let eight = AB::F::from_canonical_u32(8);
            let masked_expr = local.num_bits_to_shift + local.num_bytes_to_shift * eight;
            let shift_lo = local.c[0];

            builder.send_byte_pair(
                opcode_shift_meta,
                local.carry_multiplier,
                masked_expr,
                zero.clone(), // b = 0 (unused)
                shift_lo,
                is_real,
            );
        }

        //
        // 3. Bit-level shift: enforce bit_shift_result = b * bit_shift_multiplier (base 256).
        //
        for i in 0..WORD_SIZE {
            let mut v = local.b[i] * local.bit_shift_multiplier -
                local.bit_shift_result_carry[i] * base.clone();
            if i > 0 {
                v = v + local.bit_shift_result_carry[i - 1].into();
            }
            builder.assert_eq(local.bit_shift_result[i], v);
        }

        //
        // 4. Encode num_bytes_to_shift via nb0, nb1:
        //
        //      num_bytes_to_shift = nb0 + 2*nb1
        //
        {
            let num_bytes_expr = local.nb0 + local.nb1 * two;
            builder.assert_eq(local.num_bytes_to_shift, num_bytes_expr);
        }

        //
        // 5. Byte-level shift using 2-bit selectors.
        //
        //    num_bytes = nb0 + 2*nb1, but we never use it explicitly; instead we use
        //    the four minterm selectors:
        //
        //      sel0 = (1 - nb0)(1 - nb1)   // num_bytes = 0
        //      sel1 = nb0(1 - nb1)         // num_bytes = 1
        //      sel2 = (1 - nb0)nb1         // num_bytes = 2
        //      sel3 = nb0 nb1              // num_bytes = 3
        //
        //    For each byte i:
        //
        //      if nb = 0: a[i] = bit_shift_result[i]
        //      if nb = 1: a[i] = (i<1 ? 0 : bit_shift_result[i-1])
        //      if nb = 2: a[i] = (i<2 ? 0 : bit_shift_result[i-2])
        //      if nb = 3: a[i] = (i<3 ? 0 : bit_shift_result[i-3])
        //
        {
            let nb0 = local.nb0;
            let nb1 = local.nb1;
            let not_nb0 = one.clone() - nb0;
            let not_nb1 = one.clone() - nb1;

            let sel0 = not_nb0.clone() * not_nb1.clone();
            let sel1 = nb0 * not_nb1.clone();
            let sel2 = not_nb0 * nb1;
            let sel3 = nb0 * nb1;

            for i in 0..WORD_SIZE {
                let r0 = local.bit_shift_result[i];

                let r1 = if i < 1 { zero.clone() } else { local.bit_shift_result[i - 1].into() };

                let r2 = if i < 2 { zero.clone() } else { local.bit_shift_result[i - 2].into() };

                let r3 = if i < 3 { zero.clone() } else { local.bit_shift_result[i - 3].into() };

                let rhs =
                    sel0.clone() * r0 + sel1.clone() * r1 + sel2.clone() * r2 + sel3.clone() * r3;

                // No gating here: padding rows are constructed to satisfy this identically.
                builder.assert_eq(local.a[i], rhs);
            }
        }

        //
        // 6. Range checks for multiplication outputs (gated by is_real).
        //
        builder.slice_range_check_u8(&local.bit_shift_result, is_real);
        builder.slice_range_check_u8(&local.bit_shift_result_carry, is_real);

        for shift in local.shift_by_n_bits.iter() {
            builder.assert_bool(*shift);
        }
        builder.assert_eq(
            local.shift_by_n_bits.iter().fold(zero.clone(), |acc, &x| acc + x),
            one.clone(),
        );

        // Range check.
        {
            builder.slice_range_check_u8(&local.bit_shift_result, local.is_real);
            builder.slice_range_check_u8(&local.bit_shift_result_carry, local.is_real);
        }

        for shift in local.shift_by_n_bytes.iter() {
            builder.assert_bool(*shift);
        }

        builder.assert_eq(
            local.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
            one.clone(),
        );

        // SAFETY: `is_real` is checked to be boolean.
        // All interactions are done with multiplicity `is_real`, so padding rows lead to no
        // interactions. This chip only deals with the `SLL` opcode, so the opcode matches
        // the instruction.
        builder.assert_bool(local.is_real);

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
            AB::F::from_canonical_u32(Opcode::I32Shl.code()),
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
        alu::ShiftLeftCols,
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

    use super::ShiftLeft;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.shift_left_events =
            vec![AluEvent::new(0, 0, Opcode::I32Shl, 16, 8, 1, Opcode::I32Shl.code())];
        let chip = ShiftLeft::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("trace.width {:?}", trace.width)
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        // Some deterministic SLL test vectors: (opcode, expected_a, b, c)
        let shifts = vec![
            (Opcode::I32Shl, 0x00000000, 0x00000000, 0),
            (Opcode::I32Shl, 0x00000002, 0x00000001, 1),
            (Opcode::I32Shl, 0x00000080, 0x00000001, 7),
            (Opcode::I32Shl, 0x00008000, 0x00000001, 15),
            (Opcode::I32Shl, 0x80000000, 0x00000001, 31),
            (Opcode::I32Shl, 0x00000000, 0x80000000, 1),
            (Opcode::I32Shl, 0x21212121, 0x21212121, 0),
            (Opcode::I32Shl, 0x42424242, 0x21212121, 1),
            (Opcode::I32Shl, 0x90909080, 0x12121210, 3),
            // Shifts with large c; Wasm uses c & 31
            (Opcode::I32Shl, 0x42424242, 0x21212121, 0x00000021),
            (Opcode::I32Shl, 0x90909080, 0x12121210, 0x00000023),
        ];
        for t in shift_instructions.iter() {
            shift_events.push(AluEvent::new(0, 0, t.0, t.1, t.2, t.3, t.0.code()));
        }

        // Append more events until we have 1000 tests.
        for _ in 0..(1000 - shift_instructions.len()) {
            shift_events.push(AluEvent::new(
                0,
                0,
                Opcode::I32Shl,
                256,
                1,
                8,
                Opcode::I32Shl.code(),
            ));
        }

        let mut shard = ExecutionRecord::default();
        shard.shift_left_events = shift_events;
        let chip = ShiftLeft::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn sll_splits_bit_and_byte_shift() {
        use core::borrow::Borrow;
        use p3_baby_bear::BabyBear;
        use sp1_primitives::consts::WORD_SIZE;

        let mut shard = ExecutionRecord::default();
        // b = 1, c = 9 -> a = 1 << 9 = 0x0000_0200 (1 byte + 1 bit)
        shard.shift_left_events = vec![AluEvent::new(
            0,
            0,
            Opcode::I32Shl,
            0x0000_0200,
            0x0000_0001,
            9,
            Opcode::I32Shl.code(),
        )];

        let chip = ShiftLeft::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let row = trace.row_slice(0);
        let cols: &ShiftLeftCols<BabyBear> = (*row).borrow();

        // 9 % 8 = 1 -> multiplier 2, `shift_by_n_bits[1] = 1`; 9 / 8 = 1 -> `shift_by_n_bytes[1] =
        // 1`.
        assert_eq!(cols.bit_shift_multiplier, BabyBear::from_canonical_u32(2));
        assert_eq!(cols.shift_by_n_bits[1], BabyBear::one());
        assert_eq!(cols.shift_by_n_bytes[1], BabyBear::one());

        let expected = 0x0000_0200u32.to_le_bytes();
        for i in 0..WORD_SIZE {
            assert_eq!(cols.a[i], BabyBear::from_canonical_u8(expected[i]));
        }
    }

    #[test]
    fn sll_masks_to_low_five_bits() {
        use core::borrow::Borrow;
        use p3_baby_bear::BabyBear;
        use sp1_primitives::consts::WORD_SIZE;

        let mut shard = ExecutionRecord::default();
        // Use a value of `c` with high bits set. Low 5 bits are 16, so shift is by 16.
        let c = 0xffff_fff0u32; // 240 -> 240 & 31 = 16
        let b = 0x0000_0001u32;
        let a = b.wrapping_shl((c & 0x1f) as u32);
        debug_assert_eq!(a, 0x0001_0000);

        shard.shift_left_events =
            vec![AluEvent::new(0, 0, Opcode::I32Shl, a, b, c, Opcode::I32Shl.code())];

        let chip = ShiftLeft::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let row = trace.row_slice(0);
        let cols: &ShiftLeftCols<BabyBear> = (*row).borrow();

        // 16 % 8 = 0 -> multiplier 1 and `shift_by_n_bits[0] = 1`; 16 / 8 = 2 ->
        // `shift_by_n_bytes[2] = 1`.
        assert_eq!(cols.bit_shift_multiplier, BabyBear::from_canonical_u32(1));
        assert_eq!(cols.shift_by_n_bits[0], BabyBear::one());
        assert_eq!(cols.shift_by_n_bytes[2], BabyBear::one());

        let expected = a.to_le_bytes();
        for i in 0..WORD_SIZE {
            assert_eq!(cols.a[i], BabyBear::from_canonical_u8(expected[i]));
        }
    }

    #[test]
    fn test_malicious_sll() {
        const NUM_TESTS: usize = 5;

        for _ in 0..NUM_TESTS {
            let op_b = thread_rng().gen_range(0..u32::MAX);
            let op_c = thread_rng().gen_range(0..u32::MAX) & 0x1F;
            let correct_op_a = op_b.wrapping_shl(op_c);

            let op_a = thread_rng().gen_range(0..u32::MAX);
            assert_ne!(op_a, correct_op_a);

            let instructions = vec![
                Opcode::I32Const(5u32.into()),
                Opcode::I32Const(10u32.into()),
                Opcode::I32Const(op_b.into()),
                Opcode::I32Const(op_c.into()),
                Opcode::I32Shl,
            ];

            let program = Program::from_instrs(instructions);
            let stdin = SP1Stdin::new();

            type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

            let malicious_trace_pv_generator =
                move |prover: &P,
                      record: &mut ExecutionRecord|
                      -> Vec<(String, RowMajorMatrix<Val<BabyBearPoseidon2>>)> {
                    let mut malicious_record = record.clone();

                    // Corrupt CPU result for the I32Shl instruction.
                    if malicious_record.cpu_events.len() > 4 {
                        if let Some(MemoryRecordEnum::Write(mut write_record)) =
                            malicious_record.cpu_events[4].res_record
                        {
                            write_record.value = op_a as u32;
                        }
                    }

                    let mut traces = prover.generate_traces(&malicious_record);
                    let shift_left_chip_name = chip_name!(ShiftLeft, BabyBear);
                    for (name, trace) in traces.iter_mut() {
                        if *name == shift_left_chip_name {
                            let first_row = trace.row_mut(0);
                            let first_row: &mut ShiftLeftCols<BabyBear> = first_row.borrow_mut();
                            // Also corrupt the chip's view of a.
                            first_row.a = op_a.into();
                        }
                    }
                    traces
                };

            let result =
                run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));
            let shift_left_chip_name = chip_name!(ShiftLeft, BabyBear);

            assert!(
                result.is_err() &&
                    result.unwrap_err().is_constraints_failing(&shift_left_chip_name)
            );
        }
    }
}
