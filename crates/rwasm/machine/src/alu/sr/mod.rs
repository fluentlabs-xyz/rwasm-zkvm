//! Logical And Arithmetic Right Shift Verification.
//!
//! Implements verification for a = b >> c, decomposing the shift into bit and byte components:
//!
//! 1. num_bits_to_shift  = (c & 31) % 8  (bit-level shift, via ShrCarry).
//! 2. num_bytes_to_shift = (c & 31) / 8  (byte-level shift).
//!
//! The right shift is verified by reformulating it as:
//!     (b >> c) = (b >> (num_bytes_to_shift * 8)) >> num_bits_to_shift.
//!
//! The correct leading bits of logical and arithmetic right shifts are verified by sign extending b
//! to 64 bits before shifting.
//!
//! # Semantics used via byte tables
//!
//! Let `shift_lo` be the low byte of the Wasm shift operand `c` (i.e., c & 0xff).
//!
//! - ByteOpcode::ShiftMeta table (in `bytes::trace`):
//!       masked    = shift_lo & 31
//!       num_bits  = masked & 7
//!       num_bytes = masked >> 3
//!       ByteLookupEvent { opcode=ShiftMeta, a1=num_bits, a2=num_bytes, b=*, c=shift_lo }
//!
//! - ByteOpcode::CarryMul table (in `bytes::trace`):
//!       k      = shift_lo & 7       // num_bits
//!       raw_cm = 1u16 << (8 - k)
//!       ByteLookupEvent { opcode=CarryMul, a1=raw_cm, a2=k, b=*, c=shift_lo }
//!
//! This chip enforces consistency of its local `num_bits_to_shift`, `num_bytes_to_shift`,
//! and `carry_multiplier` columns with those tables via lookups.

mod utils;

use crate::{
    air::SP1CoreAirBuilder,
    alu::sr::utils::{nb_bits_to_shift, nb_bytes_to_shift},
    bytes::utils::shr_carry,
    utils::{next_power_of_two, zeroed_f_vec},
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
use rwasm::Opcode;
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
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise operations for the opcodes SRL and SRA (I32ShrU / I32ShrS).
#[derive(Default)]
pub struct ShiftRightChip;

/// The column layout for the chip.
///
/// Width breakdown (for T = u8):
/// - pc: 1
/// - a, b, c: 3 * 4 = 12
/// - shift_by_n_bytes: 4
/// - byte_shift_result: 8
/// - shr_carry_output_carry: 8
/// - shr_carry_output_shifted_byte: 8
/// - b_msb: 1
/// - num_bits_to_shift, num_bytes_to_shift, carry_multiplier: 3
/// - is_srl, is_sra: 2
/// Total = 47 columns.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftRightCols<T> {
    /// The program counter.
    pub pc: T,

    /// The output operand (a = b >> c).
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    /// The result of "byte-shifting" the (sign-extended) input operand `b` by `num_bytes_to_shift`.
    pub byte_shift_result: [T; LONG_WORD_SIZE],

    /// The carry output of `shr_carry` on each byte of `byte_shift_result`.
    pub shr_carry_output_carry: [T; LONG_WORD_SIZE],

    /// The shift byte output of `shr_carry` on each byte of `byte_shift_result`.
    pub shr_carry_output_shifted_byte: [T; LONG_WORD_SIZE],

    /// The most significant bit of `b` (sign bit).
    pub b_msb: T,

    /// num_bits_to_shift = (c & 31) % 8  (0..7).
    pub num_bits_to_shift: T,

    /// num_bytes_to_shift = (c & 31) / 8 (0..3).
    pub num_bytes_to_shift: T,

    /// carry_multiplier = 1 << (8 - num_bits_to_shift) (2..256).
    pub carry_multiplier: T,

    /// If the opcode is I32ShrU.
    pub is_srl: T,

    /// If the opcode is I32ShrS.
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

        values
            .chunks_mut(chunk_size * NUM_SHIFT_RIGHT_COLS)
            .enumerate()
            .par_bridge()
            .for_each(|(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_RIGHT_COLS)
                    .enumerate()
                    .for_each(|(j, row)| {
                        let idx = i * chunk_size + j;
                        let cols: &mut ShiftRightCols<F> = row.borrow_mut();

                        if idx < nb_rows {
                            let mut byte_lookup_events = Vec::new();
                            let event = &input.shift_right_events[idx];
                            self.event_to_row(event, cols, &mut byte_lookup_events);
                        } else {
                            // Padding row:
                            // shift_by_n_bytes[0] = 1, everything else zero.
                            cols.shift_by_n_bytes[0] = F::one();
                        }
                    });
            });

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
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftRightCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        // Basic fields and flags.
        {
            cols.pc = F::from_canonical_u32(event.pc);
            cols.a = Word::from(event.a);
            cols.b = Word::from(event.b);
            cols.c = Word::from(event.c);

            cols.b_msb = F::from_canonical_u32((event.b >> 31) & 1);

            cols.is_srl = F::from_bool(event.code == Opcode::I32ShrU.code());
            cols.is_sra = F::from_bool(event.code == Opcode::I32ShrS.code());

            // MSB lookup of most significant byte of b.
            let most_significant_byte = event.b.to_le_bytes()[WORD_SIZE - 1];
            blu.add_byte_lookup_events(vec![ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: ((most_significant_byte >> 7) & 1) as u16,
                a2: 0,
                b: most_significant_byte,
                c: 0,
            }]);
        }

        // Host-side computation of shift parameters (for the trace).
        let num_bytes_to_shift = nb_bytes_to_shift(event.c); // 0..3
        let num_bits_to_shift = nb_bits_to_shift(event.c);   // 0..7

        cols.num_bytes_to_shift = F::from_canonical_u32(num_bytes_to_shift as u32);
        cols.num_bits_to_shift = F::from_canonical_u32(num_bits_to_shift as u32);

        // carry_multiplier = 1 << (8 - num_bits_to_shift)
        let raw_cm: u16 = 1u16 << (8 - num_bits_to_shift as u16);
        cols.carry_multiplier = F::from_canonical_u32(raw_cm as u32);

        let shift_lo = (event.c & 0xff) as u8;
        // k = num_bits + 8 * num_bytes, which equals (shift_lo & 31) by construction
        let k = (num_bits_to_shift as u8) + 8 * (num_bytes_to_shift as u8);

        // Emit lookups that match ByteChip's unary ShiftMeta / CarryMul.
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::ShiftMeta,
            a1: k as u16,  // masked = k = c & 31
            a2: 0,
            b: 0,          // we choose the row with b = 0
            c: shift_lo,
        });

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::CarryMul,
            a1: raw_cm,    // carry_multiplier
            a2: 0,
            b: 0,          // row with b = 0
            c: shift_lo,
        });

        // Byte shifting (on sign-extended b for SRA).
        let mut byte_shift_result = [0u8; LONG_WORD_SIZE];
        {
            for i in 0..WORD_SIZE {
                cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
            }

            let sign_extended_b = if event.code == Opcode::I32ShrS.code() {
                // Sign extension for arithmetic right shift.
                ((event.b as i32) as i64).to_le_bytes()
            } else {
                (event.b as u64).to_le_bytes()
            };

            for i in 0..LONG_WORD_SIZE {
                if i + num_bytes_to_shift < LONG_WORD_SIZE {
                    byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift];
                }
            }
            cols.byte_shift_result = byte_shift_result.map(F::from_canonical_u8);
        }

        // Bit shifting and ShrCarry outputs.
        {
            let mut last_carry = 0u32;
            let mut shr_carry_output_carry = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_shifted_byte = [0u8; LONG_WORD_SIZE];

            for i in (0..LONG_WORD_SIZE).rev() {
                let (shift, carry) = shr_carry(byte_shift_result[i], num_bits_to_shift as u8);

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

                let combined =
                    ((shift as u32 + last_carry * raw_cm as u32) & 0xff) as u8;

                // Debug only: low 4 bytes should match a.
                if i < WORD_SIZE {
                    debug_assert_eq!(combined, (event.a >> (8 * i)) as u8);
                }

                last_carry = carry as u32;
            }

            cols.shr_carry_output_carry =
                shr_carry_output_carry.map(F::from_canonical_u8);
            cols.shr_carry_output_shifted_byte =
                shr_carry_output_shifted_byte.map(F::from_canonical_u8);

            // Range checks.
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

        let is_real = local.is_sra + local.is_srl;

        // Boolean flags.
        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.b_msb);
        builder.assert_bool(is_real.clone());

        // Check that the MSB of the most significant byte of b matches local.b_msb via lookup.
        {
            let byte = local.b[WORD_SIZE - 1];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            let msb = local.b_msb;
            builder.send_byte(opcode, msb, byte, zero.clone(), is_real.clone());
        }

        // Byte lookups for ShiftMeta and CarryMul (unary ops).
        {
            let opcode_shift_meta = AB::F::from_canonical_u32(ByteOpcode::ShiftMeta as u32);
            let opcode_carry_mul  = AB::F::from_canonical_u32(ByteOpcode::CarryMul as u32);
            let shift_lo          = local.c[0]; // low byte of c

            // k = num_bits + 8 * num_bytes
            let eight = AB::F::from_canonical_u32(8);
            let k_expr = local.num_bits_to_shift + local.num_bytes_to_shift * eight;

            // ShiftMeta: value = k = (c & 31)
            builder.send_byte(
                opcode_shift_meta,
                k_expr,            // a1
                zero.clone(),      // b = 0
                shift_lo,          // c = low byte of c
                is_real.clone(),
            );

            // CarryMul: value = carry_multiplier = 1 << (8 - num_bits)
            builder.send_byte(
                opcode_carry_mul,
                local.carry_multiplier, // a1
                zero.clone(),           // b = 0
                shift_lo,               // c = low byte of c
                is_real.clone(),
            );
        }

        // Byte shift the sign-extended b.
        {
            // Leading bytes are 0xff if SRA and b_msb == 1, else 0.
            let leading_byte =
                local.is_sra * local.b_msb * AB::Expr::from_canonical_u8(0xff);
            let mut sign_extended_b: Vec<AB::Expr> = vec![];
            for i in 0..WORD_SIZE {
                sign_extended_b.push(local.b[i].into());
            }
            for _ in 0..WORD_SIZE {
                sign_extended_b.push(leading_byte.clone());
            }

            // Shift sign_extended_b by num_bytes_to_shift according to selectors.
            for num_bytes_to_shift in 0..WORD_SIZE {
                for i in 0..(LONG_WORD_SIZE - num_bytes_to_shift) {
                    builder
                        .when(local.shift_by_n_bytes[num_bytes_to_shift])
                        .assert_eq(
                            local.byte_shift_result[i],
                            sign_extended_b[i + num_bytes_to_shift].clone(),
                        );
                }
            }
        }

        // Sanity checks on shift_by_n_bytes.
        {
            // Exactly one of shift_by_n_bytes must be 1.
            let sum_shift_by_n_bytes = local
                .shift_by_n_bytes
                .iter()
                .fold(zero.clone(), |acc, &x| acc + x);
            builder.assert_eq(sum_shift_by_n_bytes, one.clone());

            // If shift_by_n_bytes[i] = 1 then num_bytes_to_shift == i.
            for i in 0..WORD_SIZE {
                builder
                    .when(local.shift_by_n_bytes[i])
                    .assert_eq(
                        local.num_bytes_to_shift,
                        AB::F::from_canonical_usize(i),
                    );
            }

            for shift_by_n_byte in local.shift_by_n_bytes.iter() {
                builder.assert_bool(*shift_by_n_byte);
            }
        }

        // Bit shift via ShrCarry and carry_multiplier; constrain result to a.
        {
            let opcode_shrcarry =
                AB::F::from_canonical_u32(ByteOpcode::ShrCarry as u32);

            // ShrCarry lookups.
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

            // Combine ShrCarry outputs to get final least significant 4 bytes
            // and assert equality with a.
            for i in 0..WORD_SIZE {
                let mut v: AB::Expr = local.shr_carry_output_shifted_byte[i].into();
                if i + 1 < LONG_WORD_SIZE {
                    v = v + local.shr_carry_output_carry[i + 1] * local.carry_multiplier;
                }
                builder.when(is_real.clone()).assert_eq(local.a[i], v);
            }
        }

        // Range check byte arrays.
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

        // CPU receive_instruction wiring.
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            local.is_srl * AB::F::from_canonical_u32(Opcode::I32ShrU.code() as u32) +
                local.is_sra * AB::F::from_canonical_u32(Opcode::I32ShrS.code() as u32),
            local.a,
            local.b,
            local.c,
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
        shard.shift_right_events =
            vec![AluEvent::new(0, Opcode::I32ShrU, 6, 12, 1, Opcode::I32ShrU.code()),
                 AluEvent::new(0, Opcode::I32ShrS, 6, 12, 1, Opcode::I32ShrS.code())];
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
            shift_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3, t.0.code()));
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
                        malicious_record.cpu_events[4].res = op_a as u32;
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
