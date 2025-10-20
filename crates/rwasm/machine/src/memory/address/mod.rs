use p3_air::AirBuilder;
use p3_field::{AbstractField, PrimeField32};
use rwasm::mem_index::{SP_END, UNIT};
use rwasm_executor::events::{ByteLookupEvent, ByteRecord};
use sp1_derive::AlignedBorrow;

use rwasm_executor::{ByteOpcode, SP_START};
use sp1_stark::air::SP1AirBuilder;

const STACK_END_LOW_8BITS: u8 = (SP_END & 0xFF) as u8;
const STACK_UB_LOW_8BITS: u8 = ((SP_START + UNIT) & 0xFF) as u8;
const STACK_UB_HI_8BITS: u8 = (((SP_START + UNIT) & 0xFF00) >> 8) as u8;
/// Memory read-write access.

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StackAddressCols<T> {
    pub addr_hi_8bits: T,
    pub addr_low_8bits: T,
    pub addr_hi_is_zero: T,
    pub addr_hi_is_not_edge: T,
    pub addr_hi_is_eq_stack_ub_hi: T,
}

impl<F: PrimeField32> StackAddressCols<F> {
    pub fn populate(&mut self, addr: u32, output: &mut impl ByteRecord) {
        assert_ne!(addr, 0);
        let addr: u16 = addr.try_into().unwrap();

        let addr_hi_8bits: u8 = ((addr & 0xFF00) >> 8).try_into().unwrap();
        let addr_low_8bits: u8 = (addr & 0xFF).try_into().unwrap();
        self.addr_hi_8bits = F::from_canonical_u8(addr_hi_8bits);
        self.addr_low_8bits = F::from_canonical_u8(addr_low_8bits);

        let addr_hi_is_eq_stack_ub_hi = addr_hi_8bits == STACK_UB_HI_8BITS;
        self.addr_hi_is_eq_stack_ub_hi = F::from_bool(addr_hi_is_eq_stack_ub_hi);

        let addr_hi_is_zero = addr_hi_8bits == 0;
        self.addr_hi_is_zero = F::from_bool(addr_hi_is_zero);

        let addr_hi_is_not_edge = !addr_hi_is_zero && !addr_hi_is_eq_stack_ub_hi;
        self.addr_hi_is_not_edge = F::from_bool(addr_hi_is_not_edge);

        output.add_u8_range_check(addr_hi_8bits, addr_low_8bits);

        if addr_hi_is_not_edge {
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: addr_hi_8bits,
                c: STACK_UB_HI_8BITS,
            });
        }

        if addr_hi_is_zero {
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: STACK_END_LOW_8BITS,
                c: addr_low_8bits,
            });
        }

        if addr_hi_is_eq_stack_ub_hi {
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: true as u16,
                a2: 0,
                b: addr_low_8bits,
                c: STACK_UB_LOW_8BITS,
            });
        }
    }
}
impl<T: Copy> StackAddressCols<T> {
    pub fn addr<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr
    where
        T: Into<AB::Expr>,
    {
        let hi = self.addr_hi_8bits.into();
        let low = self.addr_low_8bits.into();
        hi * AB::Expr::from_canonical_u32(1 << 8) + low
    }

    pub fn range_check<AB: SP1AirBuilder>(builder: &mut AB, cols: StackAddressCols<AB::Var>) {
        let is_real =
            cols.addr_hi_is_not_edge + cols.addr_hi_is_zero + cols.addr_hi_is_eq_stack_ub_hi;

        builder.assert_bool(is_real.clone());

        //range check the hi and low bits of addr
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U8Range as u32),
            AB::Expr::zero(),
            cols.addr_hi_8bits,
            cols.addr_low_8bits,
            is_real.clone(),
        );

        // check edge cases of addr_hi_8bits

        builder.when(is_real.clone()).when(cols.addr_hi_is_zero).assert_zero(cols.addr_hi_8bits);

        builder
            .when(is_real)
            .when(cols.addr_hi_is_eq_stack_ub_hi)
            .assert_eq(cols.addr_hi_8bits, AB::Expr::from_canonical_u8(STACK_UB_HI_8BITS));

        // If it's not an edge case, we check that addr_hi_8bits is located within the space between
        // the edges
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.addr_hi_8bits,
            AB::Expr::from_canonical_u8(STACK_UB_HI_8BITS),
            cols.addr_hi_is_not_edge,
        );

        // Check addr_low_8bits in case addr_hi_8bits is an edge case

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            AB::Expr::from_canonical_u8(STACK_END_LOW_8BITS),
            cols.addr_low_8bits,
            cols.addr_hi_is_zero,
        );

        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::from_bool(true),
            cols.addr_low_8bits,
            AB::Expr::from_canonical_u8(STACK_UB_LOW_8BITS),
            cols.addr_hi_is_eq_stack_ub_hi,
        );
    }
}
