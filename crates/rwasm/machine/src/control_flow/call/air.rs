use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::mem_index::{AddressType, FUNC_FRAME_START, UNIT};
use rwasm_executor::{Opcode, DEFAULT_PC_INC, UNUSED_PC};

use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};

use crate::{air::MemoryAirBuilder, memory::MemoryCols};
use crate::air::SP1CoreAirBuilder;
use crate::{air::WordAirBuilder, operations::BabyBearWordRangeChecker};
const CALL_SP_STACK_SHIFT: u32 = FUNC_FRAME_START;
use super::{CallChip, CallColumns};

impl<AB> Air<AB> for CallChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &CallColumns<AB::Var> = (*local).borrow();
        builder.assert_bool(local.is_call);
        builder.assert_bool(local.is_call_indirect);
        builder.assert_bool(local.is_call_internal);
        builder.assert_bool(local.is_return);

        let opcode = local.is_call * AB::Expr::from_canonical_u32(Opcode::Call(0).code())
            + local.is_call_internal * AB::Expr::from_canonical_u32(Opcode::CallInternal(0).code())
            + local.is_call_indirect * AB::Expr::from_canonical_u32(Opcode::CallIndirect(0).code())
            + local.is_return * AB::Expr::from_canonical_u32(Opcode::Return.code());

        let is_real = local.is_call.clone()
            + local.is_call_indirect.clone()
            + local.is_call_internal
            + local.is_return.clone();
        let is_call_ins = local.is_call.clone()
            + local.is_call_indirect.clone()
            + local.is_call_internal.clone()
            + local.is_return;
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
            local.clk.clone() + AB::Expr::from_canonical_u8(1),
            local.next_call_sp * AB::Expr::from_canonical_u32(UNIT)
                + AB::Expr::from_canonical_u32(CALL_SP_STACK_SHIFT),
            &local.call_stack_access,
            local.is_call + local.is_call_indirect + local.is_call_internal,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk.clone(),
            local.call_sp * AB::Expr::from_canonical_u32(UNIT)
                + AB::Expr::from_canonical_u32(CALL_SP_STACK_SHIFT),
            &local.call_stack_access,
            local.is_return - local.not_real_return,
        );

        builder.when(local.not_real_return).assert_zero(local.call_sp);
        builder.when(local.not_real_return).assert_one(local.is_return);

        builder.when(local.is_call_internal).assert_eq(local.func_ref, local.opcode_aux_val.reduce::<AB>());
        self.eval_call_sp(builder, local);
        
    }
}

impl CallChip {
    fn eval_call_sp<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CallColumns<AB::Var>) {
        builder
            .when(local.is_call_internal)
            .assert_eq(local.call_sp + AB::Expr::one(), local.next_call_sp);
        builder
            .when(local.is_return-local.not_real_return)
            .assert_eq(local.call_sp - AB::Expr::one(), local.next_call_sp);

        builder.when(local.is_call_internal).assert_eq(AB::Expr::one()+local.pc.reduce::<AB>(), (*local.call_stack_access.value()).reduce::<AB>());
        builder.when(local.is_return-local.not_real_return).assert_word_eq(local.next_pc, *local.call_stack_access.value());
    }

    
}
