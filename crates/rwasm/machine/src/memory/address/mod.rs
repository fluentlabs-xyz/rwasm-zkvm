use p3_air::AirBuilder;
use p3_field::{AbstractField, Field, PrimeField32};
use rwasm::mem_index::{SP_END, UNIT};
use rwasm_executor::events::{
    ByteLookupEvent, ByteRecord,
};
use sp1_derive::AlignedBorrow;

use rwasm_executor::{ByteOpcode, SP_START};
use sp1_stark::air::{BaseAirBuilder, SP1AirBuilder};

const STACK_END_LOW_8BITS: u8 = (SP_END & 0xFF) as u8;
const STACK_END_HI_8BITS: u8 = ((SP_END & 0xFF00) >> 8) as u8;
const STACK_UB_LOW_8BITS: u8 = ((SP_START + UNIT) & 0xFF) as u8;
const STACK_UB_HI_8BITS: u8 = (((SP_START + UNIT) & 0xFF00) >> 8) as u8;
/// Memory read-write access.

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StackAddressCols<T> {
    pub addr: T,
    pub addr_hi_8bits: T,
    pub addr_low_8bits: T,
    pub addr_hi_is_lt_stack_ub_hi: T,
    pub addr_hi_is_zero: T,
    pub addr_hi_is_non_zero: T,
    pub is_real: T,
}

impl<F: PrimeField32> StackAddressCols<F> {
    pub fn populate(&mut self, addr: u32, output: &mut impl ByteRecord) {
        assert_ne!(addr, 0);
        let addr: u16 = addr.try_into().unwrap();
        self.is_real = F::from_bool(true);
        self.addr = F::from_canonical_u16(addr);

        let addr_hi_8bits: u8 = ((addr & 0xFF00) >> 8).try_into().unwrap();
        let addr_low_8bits: u8 = (addr & 0xFF).try_into().unwrap();
        self.addr_hi_8bits = F::from_canonical_u8(addr_hi_8bits);
        self.addr_low_8bits = F::from_canonical_u8(addr_low_8bits);
        let addr_hi_is_lt_stack_ub_hi = addr_hi_8bits < STACK_UB_HI_8BITS;
        self.addr_hi_is_lt_stack_ub_hi = F::from_bool(addr_hi_is_lt_stack_ub_hi);
        let addr_hi_is_zero = addr_hi_8bits == 0;
        self.addr_hi_is_zero = F::from_bool(addr_hi_is_zero);
        output.add_u8_range_check(addr_hi_8bits, addr_low_8bits);
        output.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: addr_hi_is_lt_stack_ub_hi as u16,
            a2: 0,
            b: addr_hi_8bits,
            c: STACK_UB_HI_8BITS,
        });

        if addr_hi_is_lt_stack_ub_hi {
            if addr_hi_is_zero {
                output.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::LTU,
                    a1: true as u16,
                    a2: 0,
                    b: STACK_END_LOW_8BITS,
                    c: addr_low_8bits,
                });
            }
        } else {
            self.addr_hi_is_non_zero = F::from_bool(true);
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: addr_low_8bits,
                c: STACK_UB_LOW_8BITS,
            });
        };
    }
}
impl<F: Field> StackAddressCols<F> {
    pub fn range_check<AB: SP1AirBuilder>(builder: &mut AB, cols: StackAddressCols<AB::Var>) {
        builder.assert_bool(cols.is_real);
        //range check addr so that it is no more than 16 bit.
        //range check the hi and low bits of addr
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U8Range as u32),
            AB::Expr::zero(),
            cols.addr_hi_8bits,
            cols.addr_low_8bits,
            cols.is_real,
        );
        //assert that the two componets are correct;
        builder.when(cols.is_real).assert_eq(
            cols.addr,
            cols.addr_hi_8bits * AB::Expr::from_canonical_u32(1 << 8) +
                cols.addr_low_8bits,
        );

        builder.assert_bool(cols.addr_hi_is_lt_stack_ub_hi);
        //We first check if the addr_hi is less than the STACK UB hi

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            cols.addr_hi_is_lt_stack_ub_hi,
            cols.addr_hi_8bits,
            AB::Expr::from_canonical_u8(STACK_UB_HI_8BITS),
            cols.is_real,
        );
        builder.assert_bool(cols.addr_hi_is_zero);
        //addr_hi will equals STACK UB hi when less than does not hold.
        //Note this case we do not need do LB check
        builder
            .when(cols.is_real)
            .when_not(cols.addr_hi_is_lt_stack_ub_hi)
            .assert_eq(cols.addr_hi_8bits, AB::Expr::from_canonical_u8(STACK_UB_HI_8BITS));

        //LB check.
        // Note that we cannot rely on addr_hi_is_zero because when not addr_hi_is_zero might be the
        // case it is an empty cols

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            cols.addr_hi_is_non_zero,
            cols.addr_low_8bits,
            AB::Expr::from_canonical_u8(STACK_UB_LOW_8BITS),
            cols.addr_hi_is_non_zero,
        );

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            AB::Expr::from_canonical_u8(STACK_END_LOW_8BITS),
            cols.addr_low_8bits,
            cols.addr_hi_is_zero,
        );
    }
}
