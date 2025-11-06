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
    hi_16bits: T,
    low_16bits: T,
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
        self.hi_16bits = F::from_canonical_u8(hi_8bits);
        self.low_16bits = F::from_canonical_u8(low_8bits);
    }

    pub fn populate(&mut self, value: u32, output: &mut impl ByteRecord) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]
        let (shifted_value, overflow) = value.overflowing_sub(START);

        assert!(!overflow);

        let shifted_value: u16 = shifted_value.try_into().unwrap();

        let hi_8bits: u8 = (shifted_value >> 8) as u8;
        let low_8bits: u8 = shifted_value as u8;
        self.hi_16bits = F::from_canonical_u8(hi_8bits);
        self.low_16bits = F::from_canonical_u8(low_8bits);

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
    const UB_LOW_8BITS_SHIFTED: u8 = (END - START + 1u32) as u8;
    const UB_HI_8BITS_SHIFTED: u8 = ((END - START + 1u32) >> 8) as u8;

    pub fn value<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        let hi = self.hi_16bits.into();
        let low = self.low_16bits.into();
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
            cols.hi_16bits,
            cols.low_16bits,
            is_real.clone(),
        );

        // check edge cases of hi_8bits
        builder.when(is_real.clone()).when(cols.hi_is_zero).assert_zero(cols.hi_16bits);

        builder
            .when(is_real)
            .when(cols.hi_is_eq_ub_hi)
            .assert_eq(cols.hi_16bits, AB::Expr::from_canonical_u8(Self::UB_HI_8BITS_SHIFTED));

        // If it's not an edge case, we check that hi_8bits is located within the space between
        // the edges
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.hi_16bits,
            AB::Expr::from_canonical_u8(Self::UB_HI_8BITS_SHIFTED),
            cols.hi_is_not_edge,
        );

        // Check low_8bits in case hi_8bits is an edge case
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.low_16bits,
            AB::Expr::from_canonical_u8(Self::UB_LOW_8BITS_SHIFTED),
            cols.hi_is_eq_ub_hi,
        );
    }

    /// The prove always have known whether a column of rangechecker is real or not.
    /// So the input do check should always  equals rangechecker.is_real.
    pub fn do_range_check<AB: SP1AirBuilder>(
        builder: &mut AB,
        cols: Range16bCols<AB::Var, START, END>,
        do_check: impl Into<AB::Expr>,
    ) {
        let is_real = cols.hi_is_not_edge + cols.hi_is_zero + cols.hi_is_eq_ub_hi;
        builder.assert_eq(do_check, is_real);
        Range16bCols::<AB::Var, START, END>::range_check(builder, cols);
    }
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Range32bCols<
    T,
    const START_HI16: u32,
    const START_LOW16: u32,
    const END_HI16: u32,
    const END_LOW16: u32,
> {
    hi_16bits: Range16bCols<T, START_HI16, END_HI16>,
    low_16bits: Range16bCols<T, START_LOW16, END_LOW16>,
    pub hi_is_zero: T,
    pub hi_is_not_edge: T,
    pub hi_is_eq_ub_hi: T,
}

impl<
        F: PrimeField32,
        const START_HI16: u32,
        const START_LOW16: u32,
        const END_HI16: u32,
        const END_LOW16: u32,
    > Range32bCols<F, START_HI16, START_LOW16, END_HI16, END_LOW16>
{
    pub fn populate_value(&mut self, value: u32) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]
        let start = START_HI16 << 16 + START_LOW16;

        let (shifted_value, overflow) = value.overflowing_sub(START_HI16);

        assert!(!overflow);

        let shifted_value: u32 = shifted_value.try_into().unwrap();

        let hi_16bits: u16 = (shifted_value >> 16) as u16;
        let low_16bits: u16 = shifted_value as u16;
        self.hi_16bits.populate_value(hi_16bits as u32);
        self.low_16bits.populate_value(low_16bits as u32);
    }

    pub fn populate(&mut self, value: u32, output: &mut impl ByteRecord) {
        // We subtract the START to work with the value that is in the range [0..END -
        // START]
        let start = START_HI16 << 16 + START_LOW16;
        let (shifted_value, overflow) = value.overflowing_sub(start);

        assert!(!overflow);
        let hi_16bits: u16 = (shifted_value >> 16) as u16;
        let low_16bits: u16 = shifted_value as u16;

        let hi_is_eq_ub_hi = hi_16bits == Self::UB_HI_16BITS_SHIFTED;
        self.hi_is_eq_ub_hi = F::from_bool(hi_is_eq_ub_hi);

        let hi_is_zero = hi_16bits == 0;
        self.hi_is_zero = F::from_bool(hi_is_zero);

        let hi_is_not_edge = !hi_is_zero && !hi_is_eq_ub_hi;
        self.hi_is_not_edge = F::from_bool(hi_is_not_edge);

        output.add_u16_range_check(hi_16bits);
        output.add_u16_range_check(low_16bits);
        if hi_is_eq_ub_hi {
            self.low_16bits.populate(low_16bits as u32, output);
        } else {
            self.low_16bits.populate_value(low_16bits as u32);
        }
        if hi_is_not_edge {
            self.hi_16bits.populate(hi_16bits as u32, output);
        } else {
            self.hi_16bits.populate_value(hi_16bits as u32);
        }
    }
}

impl<
        T: Copy,
        const START_HI16: u32,
        const START_LOW16: u32,
        const END_HI16: u32,
        const END_LOW16: u32,
    > Range32bCols<T, START_HI16, START_LOW16, END_HI16, END_LOW16>
{
    const START: u32 = (START_HI16 << 16 + START_LOW16);
    const END: u32 = (END_HI16 << 16 + END_LOW16);
    const UB_LOW_16BITS_SHIFTED: u16 =
        ((END_HI16 << 16 + END_LOW16) - (START_HI16 << 16 + START_LOW16) + 1) as u16;
    const UB_HI_16BITS_SHIFTED: u16 =
        (((END_HI16 << 16 + END_LOW16) - (START_HI16 << 16 + START_LOW16) + 1) >> 16) as u16;

    pub fn value<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        let hi = self.hi_16bits.value::<AB>();
        let low = self.low_16bits.value::<AB>();
        hi * AB::Expr::from_canonical_u32(1 << 16) + low + AB::Expr::from_canonical_u32(Self::START)
    }

    pub fn is_real<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        self.hi_is_not_edge.into() + self.hi_is_zero.into() + self.hi_is_eq_ub_hi.into()
    }

    pub fn range_check<AB: SP1AirBuilder>(
        builder: &mut AB,
        cols: Range32bCols<AB::Var, START_HI16, START_LOW16, END_HI16, END_LOW16>,
    ) {
        let is_real = cols.hi_is_not_edge + cols.hi_is_zero + cols.hi_is_eq_ub_hi;

        builder.assert_bool(is_real.clone());
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U16Range as u32),
            cols.hi_16bits.value::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U16Range as u32),
            cols.low_16bits.value::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        Range16bCols::<AB::Var, START_HI16, END_HI16>::do_range_check(
            builder,
            cols.hi_16bits,
            cols.hi_is_not_edge,
        );
        Range16bCols::<AB::Var, START_LOW16, END_LOW16>::do_range_check(
            builder,
            cols.low_16bits,
            cols.hi_is_eq_ub_hi,
        );

        // check edge cases of hi_8bits
        builder
            .when(is_real.clone())
            .when(cols.hi_is_zero)
            .assert_zero(cols.hi_16bits.value::<AB>());

        builder.when(is_real).when(cols.hi_is_eq_ub_hi).assert_eq(
            cols.hi_16bits.value::<AB>(),
            AB::Expr::from_canonical_u16(Self::UB_HI_16BITS_SHIFTED),
        );
    }
}

const fn hi_16_bits(x: u32) -> u32 {
    x >> 16
}

const fn low_16_bits(x: u32) -> u32 {
    x as u16 as u32
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
