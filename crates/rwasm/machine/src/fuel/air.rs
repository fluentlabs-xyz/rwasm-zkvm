use std::borrow::Borrow;

use p3_air::Air;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::mem_index::TypedAddress;
use rwasm_executor::{Opcode, DEFAULT_PC_INC, UNUSED_PC};

use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::MemoryAirBuilder,
    fuel::{FuelChip, FuelColumns},
    memory::MemoryCols,
};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the helper bool columns (a_eq_b, a_gt_b,
///    a_lt_b).
/// 3. It verifier the correct values of the helper bool columns based on op_a and op_b.
impl<AB> Air<AB> for FuelChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &FuelColumns<AB::Var> = (*local).borrow();
        // SAFETY: All selector columns is_consume_fuel and is_consume_fuel_stack are
        // checked to be boolean. Each "real" row has exactly one selector turned on, as
        // `is_real`, the sum of the two selectors, is boolean. Therefore, the `opcode`
        // matches the corresponding opcode.
        builder.assert_bool(local.is_consume_fuel);
        builder.assert_bool(local.is_consume_fuel_stack);

        let is_real = local.is_consume_fuel + local.is_consume_fuel_stack;

        builder.assert_bool(is_real.clone());

        let opcode = local.is_consume_fuel *
            AB::Expr::from_canonical_u32(Opcode::ConsumeFuel(0u32).code()) +
            local.is_consume_fuel_stack *
                AB::Expr::from_canonical_u32(Opcode::ConsumeFuelStack.code());

        // SAFETY: This checks the following.
        // - `num_extra_cycles = 0`
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        builder.receive_64_instruction(
            local.shard,
            local.clk,
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            AB::Expr::zero(),
            opcode,
            *local.next_consumed_fuel_low_record.value(),
            *local.next_consumed_fuel_high_record.value(),
            Word::zero::<AB>(), /* we do use conusmed fuel value frome memory because the
                                 * busline cannot send two values for op_arg1. */
            local.to_consume_fuel,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_consume_fuel + local.is_consume_fuel_stack,
        );

        // Memory related checks
        builder.eval_memory_access(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(
                TypedAddress::ReservedAddrEnum(rwasm::mem_index::ReservedAddrEnum::ConsumedFuelLow)
                    .to_virtual_addr(),
            ),
            &local.fuel_consumed_low_record,
            is_real.clone(),
        );
        builder.eval_memory_access(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(
                TypedAddress::ReservedAddrEnum(rwasm::mem_index::ReservedAddrEnum::ConsumedFuelHi)
                    .to_virtual_addr(),
            ),
            &local.fuel_consumed_high_record,
            is_real.clone(),
        );

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            AB::Expr::from_canonical_u32(
                TypedAddress::ReservedAddrEnum(rwasm::mem_index::ReservedAddrEnum::ConsumedFuelHi)
                    .to_virtual_addr(),
            ),
            &local.next_consumed_fuel_high_record,
            is_real.clone(),
        );
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            AB::Expr::from_canonical_u32(
                TypedAddress::ReservedAddrEnum(rwasm::mem_index::ReservedAddrEnum::ConsumedFuelLow)
                    .to_virtual_addr(),
            ),
            &local.next_consumed_fuel_low_record,
            is_real.clone(),
        );

        // we do not do memory check for to_consume_fuel when op is consume_fuel_stack because stack
        // check is done in cpu chip.

        // We check two equlity here:
        // next_consumed_fuel_low = fuel_consumed_low + to_consume_fuel with carry
        // next_consumed_fuel_high = fuel_consumed_high + carry
        // the second addition cannot overflow because this implies the total fuel consumed exceed
        // u64::MAX which is impossible.

        builder.send_64_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Add64.code()),
            *local.next_consumed_fuel_low_record.value(),
            Word::extend_var::<AB>(local.is_carryed),
            *local.fuel_consumed_low_record.value(),
            local.to_consume_fuel,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        builder.send_64_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Add64.code()),
            *local.next_consumed_fuel_high_record.value(),
            Word::zero::<AB>(), // no carry in for high part
            *local.fuel_consumed_high_record.value(),
            Word::extend_var::<AB>(local.is_carryed),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );
    }
}
