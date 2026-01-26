use p3_air::AirBuilder;
use p3_field::AbstractField;
use rwasm::mem_index::UNIT;
use sp1_stark::air::SP1AirBuilder;

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    cpu::{columns::CpuCols, CpuChip},
    memory::StackAddressCols,
};

impl CpuChip {
    /// Evaluates constraints for stack operations in RWASM instructions.
    ///
    /// This function verifies three key aspects:
    /// 1. Memory access constraints for operand reads and result writes
    /// 2. Proper stack pointer progression across operations
    /// 3. Range validity of stack addresses
    pub(crate) fn eval_stack<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        let is_real = local.instruction.is_with_zero_params +
            local.instruction.is_with_one_param +
            local.instruction.is_with_two_three_params;

        let next_is_real = next.instruction.is_with_zero_params +
            next.instruction.is_with_one_param +
            next.instruction.is_with_two_three_params;

        // Verify access to the first operand at the current stack pointer (sp).
        // This operand is read for both unary and binary operations.
        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp.value::<AB>() + AB::Expr::from_canonical_u32(UNIT),
            &local.op_arg1_access,
            local.instruction.is_with_one_param,
        );

        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp.value::<AB>() + AB::Expr::from_canonical_u32(2 * UNIT),
            &local.op_arg1_access,
            local.instruction.is_with_two_three_params,
        );

        // Verify access to the second operand at address sp + UNIT.
        // This operand is only read for binary operations.
        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp.value::<AB>() + AB::Expr::from_canonical_u32(UNIT),
            &local.op_arg2_access,
            local.instruction.is_with_two_three_params,
        );

        // TODO: fix
        // For binary operations, ensure that op_arg2_sp is correctly calculated
        // as sp + UNIT (the address of the second operand on the stack).
        // builder.when(local.instruction.is_binary).assert_eq(
        //     local.sp.value::<AB>(),
        //     local.op_arg2_sp.value::<AB>(),
        // );

        // Verify writing the result to the next stack position (next_sp).
        // This happens in the next cycle (clk + 1) for instructions that produce a result.
        builder.eval_memory_access(
            local.shard,
            clk.clone() + AB::Expr::one(),
            local.next_sp.value::<AB>() + AB::Expr::from_canonical_u32(UNIT),
            &local.op_res_access,
            local.instruction.has_result,
        );

        // Ensure the current stack pointer is within valid range.
        // This check is activated for both unary and binary operations.
        StackAddressCols::<AB::F>::do_range_check(builder, local.sp, is_real.clone());

        // Ensure the second operand's address is within valid range.
        // This check is only activated for binary operations.
        StackAddressCols::<AB::F>::do_range_check(
            builder,
            local.op_arg2_sp,
            local.instruction.is_with_two_three_params,
        );

        // Ensure the next stack pointer is within valid range.
        // This check is activated for instructions that write a result.
        StackAddressCols::<AB::F>::do_range_check(builder, local.next_sp, is_real.clone());

        // For unary and nullary operations, the second operand must be zero
        // since these operations don't use a second operand from the stack.
        builder
            .when(local.instruction.is_with_one_param + local.instruction.is_with_zero_params)
            .assert_word_zero(local.op_arg2_val());

        // For nullary operations (e.g., constants), the first operand must also be zero
        // since these operations don't consume any stack operands.
        builder.when(local.instruction.is_with_zero_params).assert_word_zero(local.op_arg1_val());

        builder
            .when(is_real.clone())
            .when(next_is_real)
            .assert_eq(local.next_sp.value::<AB>(), next.sp.value::<AB>());

        // Range-check each byte of the result word when the instruction produces a result.
        // This ensures that any value written to memory is a well-formed word with bytes in
        // 0..=255.
        builder.slice_range_check_u8(
            &local.op_res_access.access.value.0,
            local.instruction.has_result,
        );
    }
}
