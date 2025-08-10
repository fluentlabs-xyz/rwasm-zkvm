use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm_executor::{Opcode, DEFAULT_PC_INC, UNUSED_PC};

use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};

use crate::{air::WordAirBuilder, operations::BabyBearWordRangeChecker};

use super::{CallChip, CallColumns};



impl<AB> Air<AB> for CallChip
where
    AB: SP1AirBuilder,
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
            + local.is_call_internal * AB::Expr::from_canonical_u32(Opcode::CallIndirect(0).code())
            + local.is_return * AB::Expr::from_canonical_u32(Opcode::Return.code());

        let is_real = local.is_call.clone() +local.is_call_indirect.clone() +local.is_call_internal+local.is_return.clone();
        let is_call = local.is_call.clone() +local.is_call_indirect.clone() +local.is_call_internal.clone();
       builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            AB::Expr::zero(),
            opcode,
            local.pc,
            local.func_ref,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );
    }
}
