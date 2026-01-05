pub mod air;
pub mod columns;
// pub mod event;
// pub mod opcode;
pub mod trace;
pub mod utils;

use rwasm_executor::{events::ByteLookupEvent, ByteOpcode};

use core::borrow::BorrowMut;
use std::marker::PhantomData;

use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use self::{
    columns::{BytePreprocessedCols, NUM_BYTE_PREPROCESSED_COLS},
    utils::shr_carry,
};
use crate::{bytes::trace::NUM_ROWS, utils::zeroed_f_vec};

/// The number of different byte operations.
pub const NUM_BYTE_OPS: usize = 13;

/// A chip for computing byte operations.
///
/// The chip contains a preprocessed table of all possible byte operations. Other chips can then
/// use lookups into this table to compute their own operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteChip<F>(PhantomData<F>);

impl<F: Field> ByteChip<F> {
    /// Creates the preprocessed byte trace.
    ///
    /// This function returns a `trace` which is a matrix containing all possible byte operations.
    pub fn trace() -> RowMajorMatrix<F> {
        // The trace containing all values, with all multiplicities set to zero.
        let mut initial_trace = RowMajorMatrix::new(
            zeroed_f_vec(NUM_ROWS * NUM_BYTE_PREPROCESSED_COLS),
            NUM_BYTE_PREPROCESSED_COLS,
        );

        // Record all the necessary operations for each byte lookup.
        let opcodes = ByteOpcode::all();

        // Iterate over all options for pairs of bytes `a` and `b`.
        for (row_index, (b, c)) in (0..=u8::MAX).cartesian_product(0..=u8::MAX).enumerate() {
            let b = b as u8;
            let c = c as u8;
            let col: &mut BytePreprocessedCols<F> = initial_trace.row_mut(row_index).borrow_mut();

            // Set the values of `b` and `c`.
            col.b = F::from_canonical_u8(b);
            col.c = F::from_canonical_u8(c);

            // Iterate over all operations for results and updating the table map.
            for opcode in opcodes.iter() {
                match opcode {
                    ByteOpcode::AND => {
                        let and = b & c;
                        col.and = F::from_canonical_u8(and);
                        ByteLookupEvent::new(*opcode, and as u16, 0, b, c)
                    }
                    ByteOpcode::OR => {
                        let or = b | c;
                        col.or = F::from_canonical_u8(or);
                        ByteLookupEvent::new(*opcode, or as u16, 0, b, c)
                    }
                    ByteOpcode::XOR => {
                        let xor = b ^ c;
                        col.xor = F::from_canonical_u8(xor);
                        ByteLookupEvent::new(*opcode, xor as u16, 0, b, c)
                    }
                    ByteOpcode::SLL => {
                        let sll = b << (c & 7);
                        col.sll = F::from_canonical_u8(sll);
                        ByteLookupEvent::new(*opcode, sll as u16, 0, b, c)
                    }
                    ByteOpcode::U8Range => ByteLookupEvent::new(*opcode, 0, 0, b, c),
                    ByteOpcode::ShrCarry => {
                        let (res, carry) = shr_carry(b, c);
                        col.shr = F::from_canonical_u8(res);
                        col.shr_carry = F::from_canonical_u8(carry);
                        ByteLookupEvent::new(*opcode, res as u16, carry, b, c)
                    }
                    ByteOpcode::LTU => {
                        let ltu = b < c;
                        col.ltu = F::from_bool(ltu);
                        ByteLookupEvent::new(*opcode, ltu as u16, 0, b, c)
                    }
                    ByteOpcode::MSB => {
                        let msb = (b & 0b1000_0000) != 0;
                        col.msb = F::from_bool(msb);
                        ByteLookupEvent::new(*opcode, msb as u16, 0, b, 0)
                    }
                    ByteOpcode::U16Range => {
                        let v = ((b as u32) << 8) + c as u32;
                        col.value_u16 = F::from_canonical_u32(v);
                        ByteLookupEvent::new(*opcode, v as u16, 0, 0, 0)
                    }
                    ByteOpcode::U16POPCNT => {
                        let u16_val = ((b as u16) << 8) + c as u16;
                        let popcnt = u16_val.count_ones() as u8;
                        col.u16_popcnt = F::from_canonical_u8(popcnt);
                        ByteLookupEvent::new(*opcode, popcnt as u16, 0, b, c)
                    }
                    ByteOpcode::U16CTZ => {
                        let u16_val = ((b as u16) << 8) + c as u16;
                        let ctz = u16_val.trailing_zeros() as u8;
                        col.u16_ctz = F::from_canonical_u8(ctz);
                        ByteLookupEvent::new(*opcode, ctz as u16, 0, b, c)
                    }
                    ByteOpcode::U16CLZ => {
                        let u16_val = ((b as u16) << 8) + c as u16;
                        let clz = u16_val.leading_zeros() as u8;
                        col.u16_clz = F::from_canonical_u8(clz);
                        ByteLookupEvent::new(*opcode, clz as u16, 0, b, c)
                    }
                    ByteOpcode::ShiftMeta => {
                        let shift_lo = c;
                        let masked = shift_lo & 31; // 0..31
                        let k = masked & 7; // low 3 bits
                        let raw_cm: u16 = 1u16 << (8 - k as u16);

                        // You can store them in separate columns if you want:
                        col.shift_meta = F::from_canonical_u8(masked);
                        col.carry_mul = F::from_canonical_u32(raw_cm as u32);

                        // Key:
                        //   a1 = carry_multiplier (u16)
                        //   a2 = masked (0..31)
                        //   b  = arbitrary (we'll use b=0 in SR)
                        //   c  = shift_lo
                        ByteLookupEvent::new(*opcode, raw_cm, masked, b, c)
                    }
                };
            }
        }

        initial_trace
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use std::time::Instant;

    use super::*;

    #[test]
    pub fn test_trace_and_map() {
        let start = Instant::now();
        ByteChip::<BabyBear>::trace();
        println!("trace and map: {:?}", start.elapsed());
    }
}
