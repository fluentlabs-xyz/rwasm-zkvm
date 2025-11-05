use p3_air::AirBuilder;
use p3_field::{AbstractField, PrimeField32};
use rwasm::mem_index::UNIT;
use rwasm_executor::events::{ByteLookupEvent, ByteRecord};
use sp1_derive::AlignedBorrow;

use rwasm_executor::ByteOpcode;
use sp1_stark::air::SP1AirBuilder;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Range16bCols<T, const START: u32, const END: u32> {
    hi_8bits: T,
    low_8bits: T,
    pub hi_is_zero: T,
    pub hi_is_not_edge: T,
    pub hi_is_eq_ub_hi: T,
}

impl<F: PrimeField32, const START: u32, const END: u32> Range16bCols<F, START, END> {
    pub fn populate_value(&mut self, value: u32) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]

        let (shifted_value, overflow) = value.overflowing_sub(START);

        assert!(!overflow);

        let shifted_value: u16 = shifted_value.try_into().unwrap();

        let hi_8bits: u8 = (shifted_value >> 8) as u8;
        let low_8bits: u8 = shifted_value as u8;
        self.hi_8bits = F::from_canonical_u8(hi_8bits);
        self.low_8bits = F::from_canonical_u8(low_8bits);
    }

    pub fn populate(&mut self, value: u32, output: &mut impl ByteRecord) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]
        let (shifted_value, overflow) = value.overflowing_sub(START);

        assert!(!overflow);

        let shifted_value: u16 = shifted_value.try_into().unwrap();

        let hi_8bits: u8 = (shifted_value >> 8) as u8;
        let low_8bits: u8 = shifted_value as u8;
        self.hi_8bits = F::from_canonical_u8(hi_8bits);
        self.low_8bits = F::from_canonical_u8(low_8bits);

        let hi_is_eq_ub_hi = hi_8bits == Self::UB_HI_8BITS_SHIFTED;
        self.hi_is_eq_ub_hi = F::from_bool(hi_is_eq_ub_hi);

        let hi_is_zero = hi_8bits == 0;
        self.hi_is_zero = F::from_bool(hi_is_zero);

        let hi_is_not_edge = !hi_is_zero && !hi_is_eq_ub_hi;
        self.hi_is_not_edge = F::from_bool(hi_is_not_edge);

        output.add_u8_range_check(hi_8bits, low_8bits);

        if hi_is_not_edge {
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: hi_8bits,
                c: Self::UB_HI_8BITS_SHIFTED,
            });
        }

        if hi_is_eq_ub_hi {
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: low_8bits,
                c: Self::UB_LOW_8BITS_SHIFTED,
            });
        }
    }
}

impl<T: Copy, const START: u32, const END: u32> Range16bCols<T, START, END> {
    const UB_LOW_8BITS_SHIFTED: u8 = (END - START + UNIT) as u8;
    const UB_HI_8BITS_SHIFTED: u8 = ((END - START + UNIT) >> 8) as u8;

    pub fn value<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        let hi = self.hi_8bits.into();
        let low = self.low_8bits.into();
        hi * AB::Expr::from_canonical_u32(1 << 8) + low + AB::Expr::from_canonical_u32(START)
    }

    pub fn is_real<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        self.hi_is_not_edge.into() + self.hi_is_zero.into() + self.hi_is_eq_ub_hi.into()
    }

    pub fn range_check<AB: SP1AirBuilder>(
        builder: &mut AB,
        cols: Range16bCols<AB::Var, START, END>,
    ) {
        let is_real = cols.hi_is_not_edge + cols.hi_is_zero + cols.hi_is_eq_ub_hi;

        builder.assert_bool(is_real.clone());

        //range check the hi and low bits of value
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U8Range as u32),
            AB::Expr::zero(),
            cols.hi_8bits,
            cols.low_8bits,
            is_real.clone(),
        );

        // check edge cases of hi_8bits
        builder.when(is_real.clone()).when(cols.hi_is_zero).assert_zero(cols.hi_8bits);

        builder
            .when(is_real)
            .when(cols.hi_is_eq_ub_hi)
            .assert_eq(cols.hi_8bits, AB::Expr::from_canonical_u8(Self::UB_HI_8BITS_SHIFTED));

        // If it's not an edge case, we check that hi_8bits is located within the space between
        // the edges
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.hi_8bits,
            AB::Expr::from_canonical_u8(Self::UB_HI_8BITS_SHIFTED),
            cols.hi_is_not_edge,
        );

        // Check low_8bits in case hi_8bits is an edge case
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.low_8bits,
            AB::Expr::from_canonical_u8(Self::UB_LOW_8BITS_SHIFTED),
            cols.hi_is_eq_ub_hi,
        );
    }
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Range8bCols<T, const START: u32, const END: u32> {
    pub byte: T,
    pub is_zero: T,
    pub is_not_zero: T,
}

impl<F: PrimeField32, const START: u32, const END: u32> Range8bCols<F, START, END> {
    pub fn populate_value(&mut self, value: u32) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]

        let (shifted_value, overflow) = value.overflowing_sub(START);

        assert!(!overflow);

        let byte: u8 = shifted_value.try_into().unwrap();

        self.byte = F::from_canonical_u8(byte);
    }

    pub fn populate(&mut self, value: u32, output: &mut impl ByteRecord) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]
        let (shifted_value, overflow) = value.overflowing_sub(START);

        assert!(!overflow);

        let byte: u8 = shifted_value.try_into().unwrap();

        self.byte = F::from_canonical_u8(byte);

        let is_zero = byte == 0;

        if is_zero {
            self.is_zero = F::one();
        } else {
            self.is_not_zero = F::one();

            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: byte,
                c: (END - START) as u8,
            });
        }
    }
}

impl<T: Copy, const START: u32, const END: u32> Range8bCols<T, START, END> {
    pub fn value<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        let byte = self.byte.into();
        byte + AB::Expr::from_canonical_u32(START)
    }

    pub fn is_real<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        self.is_zero.into() + self.is_not_zero.into()
    }

    pub fn range_check<AB: SP1AirBuilder>(
        builder: &mut AB,
        cols: Range8bCols<AB::Var, START, END>,
    ) {
        let is_real = cols.is_real::<AB>();

        builder.assert_bool(is_real.clone());

        // check edge cases of hi_8bits
        builder.when(is_real.clone()).when(cols.is_zero).assert_zero(cols.byte);

        // If it's not an edge case, we check that hi_8bits is located within the space between
        // the edges
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.byte,
            AB::Expr::from_canonical_u8((END - START) as u8),
            cols.is_not_zero,
        );
    }
}
