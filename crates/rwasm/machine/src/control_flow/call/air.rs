use std::borrow::Borrow;

use num::one;
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::{
    mem_index::{FUNC_FRAME_START, TABLE_SEG_START, UNIT},
    N_MAX_TABLE_SIZE,
};
use rwasm_executor::Opcode;

use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    memory::{CallStackAddressCols, MemoryCols, TableAddressCols},
    operations::BabyBearWordRangeChecker,
};
const CALL_SP_STACK_SHIFT: u32 = FUNC_FRAME_START;
const TABLE_MEMORY_SHIFT: u32 = TABLE_SEG_START;
const N_ONE_TABLE_MEMORY_LENGTH: u32 = N_MAX_TABLE_SIZE * UNIT;
use super::{CallChip, CallColumns};

impl<AB> Air<AB> for CallChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        //TODO: add signature check and rangecheck

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &CallColumns<AB::Var> = (*local).borrow();
        builder.assert_bool(local.is_call);
        builder.assert_bool(local.is_call_indirect);
        builder.assert_bool(local.is_call_internal);
        builder.assert_bool(local.is_return);

        let opcode = local.is_call * AB::Expr::from_canonical_u32(Opcode::Call(0).code()) +
            local.is_call_internal * AB::Expr::from_canonical_u32(Opcode::CallInternal(0).code()) +
            local.is_call_indirect * AB::Expr::from_canonical_u32(Opcode::CallIndirect(0).code()) +
            local.is_return * AB::Expr::from_canonical_u32(Opcode::Return.code());

        let is_real =
            local.is_call + local.is_call_indirect + local.is_call_internal + local.is_return;
        let is_call_ins =
            local.is_call + local.is_call_indirect + local.is_call_internal + local.is_return;
        builder.receive_instruction(
            local.shard,
            local.clk,
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            AB::Expr::zero(),
            opcode,
            local.opcode_aux_val,
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );
        builder.receive_call(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.opcode,
            local.call_sp,
            local.next_call_sp,
            local.func_ref,
            local.table_id,
            local.table_idx,
            is_call_ins.clone(),
        );

        builder.send_program(
            local.pc.reduce::<AB>() + AB::Expr::one(),
            AB::Expr::from_canonical_u32(Opcode::TableGet(0u16).code()),
            Word::extend_var::<AB>(local.table_id),
            local.is_call_indirect,
        );

        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.pc,
            local.pc_range_checker,
            is_call_ins.clone(),
        );
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            is_call_ins.clone(),
        );
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::from_canonical_u8(1),
            local.next_call_sp_addr.value::<AB>(),
            &local.call_stack_access,
            local.is_call + local.is_call_indirect + local.is_call_internal,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.call_sp_addr.value::<AB>(),
            &local.call_stack_access,
            local.is_return - local.not_real_return,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.table_id * AB::Expr::from_canonical_u32(N_ONE_TABLE_MEMORY_LENGTH) +
                local.table_idx * AB::Expr::from_canonical_u32(UNIT) +
                AB::Expr::from_canonical_u32(TABLE_MEMORY_SHIFT),
            &local.table_access,
            local.is_call_indirect,
        );

        builder.when(local.not_real_return).assert_zero(local.call_sp);
        builder.when(local.not_real_return).assert_one(local.is_return);

        builder
            .when(local.is_call_internal)
            .assert_eq(local.func_ref, local.opcode_aux_val.reduce::<AB>());
        self.eval_call_sp(builder, local);
        self.eval_next_pc(builder, local);

        builder.when(is_call_ins.clone()).assert_eq(
            local.next_call_sp_addr.value::<AB>(),
            local.next_call_sp * AB::Expr::from_canonical_u32(UNIT) +
                AB::Expr::from_canonical_u32(CALL_SP_STACK_SHIFT),
        );
        builder.when(local.is_return - local.not_real_return).assert_eq(
            local.call_sp_addr.value::<AB>(),
            local.call_sp * AB::Expr::from_canonical_u32(UNIT) +
                AB::Expr::from_canonical_u32(CALL_SP_STACK_SHIFT),
        );
        CallStackAddressCols::<AB::Var>::range_check(builder, local.call_sp_addr);
        CallStackAddressCols::<AB::Var>::range_check(builder, local.next_call_sp_addr);
    }
}

impl CallChip {
    fn eval_call_sp<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CallColumns<AB::Var>) {
        builder
            .when(local.is_call_internal + local.is_call_indirect)
            .assert_eq(local.call_sp + AB::Expr::one(), local.next_call_sp);
        builder
            .when(local.is_return - local.not_real_return)
            .assert_eq(local.call_sp - AB::Expr::one(), local.next_call_sp);

        builder.when(local.is_call_internal).assert_eq(
            AB::Expr::one() + local.pc.reduce::<AB>(),
            (*local.call_stack_access.value()).reduce::<AB>(),
        );

        builder.when(local.is_call_indirect).assert_eq(
            AB::Expr::from_canonical_u32(2u32) + local.pc.reduce::<AB>(),
            (*local.call_stack_access.value()).reduce::<AB>(),
        );
        builder
            .when(local.is_return - local.not_real_return)
            .assert_word_eq(local.next_pc, *local.call_stack_access.value());
    }

    fn eval_next_pc<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CallColumns<AB::Var>) {
        builder.when(local.is_call_internal).assert_word_eq(local.next_pc, local.opcode_aux_val);
        builder
            .when(local.is_call_indirect)
            .assert_word_eq(local.next_pc, *local.table_access.value());
        builder.when(local.is_return - local.not_real_return).assert_eq(
            local.next_pc.reduce::<AB>(),
            local.call_stack_access.value().reduce::<AB>(),
        );
    }
}
