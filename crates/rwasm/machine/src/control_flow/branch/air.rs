use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::mem_index::UNIT;
use rwasm_executor::{Opcode, DEFAULT_PC_INC, UNUSED_PC};

use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};

use crate::{
    air::WordAirBuilder,
    operations::{BabyBearWordRangeChecker, IsZeroWordOperation},
};

use super::{BranchChip, BranchColumns};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the helper bool columns (a_eq_b, a_gt_b,
///    a_lt_b).
/// 3. It verifier the correct values of the helper bool columns based on op_a and op_b.
impl<AB> Air<AB> for BranchChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BranchColumns<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_beq`, `is_bne`, `is_blt`, `is_bge`, `is_bltu`, `is_bgeu` are
        // checked to be boolean. Each "real" row has exactly one selector turned on, as
        // `is_real`, the sum of the six selectors, is boolean. Therefore, the `opcode`
        // // matches the corresponding opcode.
        builder.assert_bool(local.is_br);
        builder.assert_bool(local.is_brifeqz);
        builder.assert_bool(local.is_brifnez);
        builder.assert_bool(local.is_brtable);

        let is_real = local.is_br + local.is_brifeqz + local.is_brifnez + local.is_brtable;

        builder.assert_bool(is_real.clone());

        let opcode = local.is_br * AB::Expr::from_canonical_u32(Opcode::Br(0i32.into()).code()) +
            local.is_brifeqz * AB::Expr::from_canonical_u32(Opcode::BrIfEqz(0i32.into()).code()) +
            local.is_brifnez * AB::Expr::from_canonical_u32(Opcode::BrIfNez(0i32.into()).code()) +
            local.is_brtable * AB::Expr::from_canonical_u32(Opcode::BrTable(0u32).code());

        // SAFETY: This checks the following.
        // - `num_extra_cycles = 0`
        // - `op_a_val` will be constrained in the CpuChip as `op_a_immutable = 1`
        // - `op_a_immutable = 1`, as this is a branch instruction
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        // `next_pc` still has to be constrained, and this is done below.
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            opcode.clone(),
            Word::zero::<AB>(),
            local.op_arg1_value,
            Word::zero::<AB>(),
            local.aux_value,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_brifeqz + local.is_brtable + local.is_brifnez,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            opcode,
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.offset_value,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_br,
        );

        // Evaluate program counter constraints.
        {
            // Range check branch_cols.pc and branch_cols.next_pc.
            // SAFETY: `is_real` is already checked to be boolean.
            // The `BabyBearWordRangeChecker` assumes that the value is checked to be a valid word.
            // This is done when the word form is relevant, i.e. when `pc` and `next_pc` are sent to
            // the ADD ALU table. The ADD ALU table checks the inputs are valid words,
            // when it invokes `AddOperation`.
            BabyBearWordRangeChecker::<AB::F>::range_check(
                builder,
                local.pc,
                local.pc_range_checker,
                is_real.clone(),
            );
            BabyBearWordRangeChecker::<AB::F>::range_check(
                builder,
                local.next_pc,
                local.next_pc_range_checker,
                is_real.clone(),
            );

            // When we are branching, assert that local.next_pc <==> local.pc + c.
            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32Add.code()),
                local.next_pc,
                local.pc,
                local.offset_value,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.is_branching,
            );

            // When we are not branching, assert that local.pc + 4 <==> next.pc.
            builder.when(is_real.clone()).when(local.not_branching).assert_eq(
                local.pc.reduce::<AB>() + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
                local.next_pc.reduce::<AB>(),
            );

            // When local.not_branching is true, assert that local.is_real is true.
            builder.when(is_real.clone()).when(local.not_branching).assert_eq(
                local.pc.reduce::<AB>() + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
                local.next_pc.reduce::<AB>(),
            );

            // When local.not_branching is true, assert that local.is_real is true.
            builder.when(local.not_branching).assert_one(is_real.clone());

            // To prevent the ALU send above to be non-zero when the row is a padding row.
            builder.when_not(is_real.clone()).assert_zero(local.is_branching);

            // Assert that either we are branching or not branching when the instruction is a
            // branch.
            // The `next_pc` is constrained in both branching and not branching cases, so it is
            // fully constrained.
            builder.when(is_real.clone()).assert_one(local.is_branching + local.not_branching);
            builder.when(is_real.clone()).assert_bool(local.is_branching);
            builder.when(is_real.clone()).assert_bool(local.not_branching);
        }

        // Evaluate branching value constraints.
        {
            builder
                .when(local.is_br + local.is_brifeqz + local.is_brifnez)
                .assert_word_eq(local.aux_value, local.offset_value);

            // When the opcode is BrIfEqz and we are branching, assert that a_eq_b is true.
            builder
                .when(
                    local.is_brifeqz * local.is_branching + local.is_brifnez * local.not_branching,
                )
                .assert_one(local.arg1_eq_zero.result);

            // When the opcode is BrIfNez and we are branching, assert that either a_gt_b
            builder
                .when(
                    local.is_brifnez * local.is_branching + local.is_brifeqz * local.not_branching,
                )
                .assert_zero(local.arg1_eq_zero.result);

            IsZeroWordOperation::<AB::F>::eval(
                builder,
                local.op_arg1_value.map(|x| x.into()),
                local.arg1_eq_zero,
                is_real,
            );

            builder.when(local.is_br + local.is_brtable).assert_one(local.is_branching);

            builder.when(local.is_brtable).assert_zero(local.offset_value[3]);
            builder.when(local.is_brtable).assert_zero(local.offset_value[2]);

            builder
                .when(local.is_brtable)
                .when(local.a_lt_target)
                .assert_zero(local.op_arg1_value[3]);
            builder
                .when(local.is_brtable)
                .when(local.a_lt_target)
                .assert_zero(local.op_arg1_value[2]);

            builder
                .when(local.is_brtable)
                .when_not(local.a_lt_target)
                .assert_zero(local.aux_value[3]);
            builder
                .when(local.is_brtable)
                .when_not(local.a_lt_target)
                .assert_zero(local.aux_value[2]);

            builder.when(local.is_brtable).when(local.a_lt_target).assert_eq(
                local.offset_value.reduce::<AB>(),
                local.op_arg1_value.reduce::<AB>() * AB::Expr::from_canonical_u32(2u32) +
                    AB::Expr::one(),
            );
            builder.when(local.is_brtable).when_not(local.a_lt_target).assert_eq(
                local.offset_value.reduce::<AB>(),
                local.aux_value.reduce::<AB>() * AB::Expr::from_canonical_u32(2u32) -
                    AB::Expr::one(),
            );

            builder.when(local.is_brtable).assert_eq(
                local.aux_value.reduce::<AB>() - AB::Expr::one(),
                local.target.reduce::<AB>(),
            );

            builder.send_rwasm_instruction(
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNUSED_PC),
                AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(UNIT),
                AB::Expr::zero(),
                AB::Expr::from_canonical_u32(Opcode::I32LtU.code()),
                Word::extend_var::<AB>(local.a_lt_target),
                local.op_arg1_value,
                local.target,
                Word::zero::<AB>(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
                local.is_brtable,
            );
        }
    }
}
