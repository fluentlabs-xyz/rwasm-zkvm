use rwasm_executor::events::ByteRecord;
use sp1_stark::{air::SP1AirBuilder, Word};

use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

use crate::air::WordAirBuilder;

/// A set of columns needed to compute the add of two words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddOperation<T> {
    /// The result of `a + b`.
    pub value: Word<T>,

    /// Trace.
    pub carry: [T; 3],
}

impl<F: Field> AddOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32) -> u32 {
        let expected = a_u32.wrapping_add(b_u32);
        self.value = Word::from(expected);
        let a = a_u32.to_le_bytes();
        let b = b_u32.to_le_bytes();

        // Calculate carries for the trace
        let mut current_carry = 0;
        for i in 0..3 {
            let sum = (a[i] as u16) + (b[i] as u16) + current_carry;
            current_carry = sum >> 8; // Get the overflow bit (0 or 1)
            self.carry[i] = F::from_canonical_u16(current_carry);
        }

        // Range check
        {
            record.add_u8_range_checks(&a);
            record.add_u8_range_checks(&b);
            record.add_u8_range_checks(&expected.to_le_bytes());
        }
        expected
    }

    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base = AB::F::from_canonical_u32(256);

        // Range check the result (inputs should be checked by caller, but checking here is safe)
        builder.slice_range_check_u8(&a.0, is_real.clone());
        builder.slice_range_check_u8(&b.0, is_real.clone());
        builder.slice_range_check_u8(&cols.value.0, is_real.clone());

        let mut builder_is_real = builder.when(is_real.clone());
        // Constrain Carries to be boolean (0 or 1)
        builder_is_real.assert_bool(cols.carry[0]);
        builder_is_real.assert_bool(cols.carry[1]);
        builder_is_real.assert_bool(cols.carry[2]);

        // Arithmetic Constraints
        // We enforce: a[i] + b[i] + carry_in = value[i] + 256 * carry_out
        // Rearranged: a[i] + b[i] + carry_in - value[i] - 256 * carry_out = 0

        // Limb 0: No carry in. Carry out is cols.carry[0].
        builder_is_real.assert_zero(a[0] + b[0] - cols.value[0] - cols.carry[0] * base);

        // Limb 1: Carry in is cols.carry[0]. Carry out is cols.carry[1].
        builder_is_real
            .assert_zero(a[1] + b[1] + cols.carry[0] - cols.value[1] - cols.carry[1] * base);

        // Limb 2: Carry in is cols.carry[1]. Carry out is cols.carry[2].
        builder_is_real
            .assert_zero(a[2] + b[2] + cols.carry[1] - cols.value[2] - cols.carry[2] * base);

        // Limb 3: Carry in is cols.carry[2]. No explicit carry out (wrapping add).
        // We must ensure that (a[3] + b[3] + c[2]) is congruent to value[3] mod 256.
        // Effectively: a[3] + b[3] + c[2] - value[3] must equal either 0 OR 256.
        let overflow_3 = a[3] + b[3] + cols.carry[2] - cols.value[3];
        builder_is_real.assert_zero(overflow_3.clone() * (overflow_3 - base));
    }
}
