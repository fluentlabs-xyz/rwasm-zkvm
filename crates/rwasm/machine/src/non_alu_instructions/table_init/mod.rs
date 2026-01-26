use std::borrow::Borrow;

use crate::{
    air::{MemoryAirBuilder, ProgramAirBuilder},
    memory::{ElementAddressCols, TableAddressCols},
};

use p3_air::{Air, AirBuilder, BaseAir};

use crate::air::WordAirBuilder;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm_executor::{Opcode, DEFAULT_PC_INC};
use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};
mod column;
mod trace;
use crate::memory::MemoryCols;
pub use column::*;

use rwasm::{
    mem_index::{TypedAddress, UNIT},
    N_MAX_TABLE_SIZE,
};

#[derive(Default)]
pub struct TableInitChip {}
impl<AB> Air<AB> for TableInitChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableInitCols<AB::Var> = (*local).borrow();
        let next: &TableInitCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_non_zero_length);

        builder.when_first_row().assert_one(local.is_first);
        builder.when(local.is_last).assert_one(local.is_real);
        builder.when(local.is_first).assert_one(local.is_real);

        // check transition between events
        builder.when_transition().when(local.is_last).when(next.is_real).assert_one(next.is_first);

        // check in event transitions
        builder.when_transition().when_not(local.is_last).assert_eq(local.is_real, next.is_real);
        builder.when_transition().when_not(local.is_last).assert_eq(local.clk, next.clk);
        builder.when_transition().when_not(local.is_last).assert_eq(local.shard, next.shard);
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.table_idx.value::<AB>(), next.table_idx.value::<AB>());
        builder.when_transition().when_not(local.is_last).assert_word_eq(local.src, next.src);
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_word_eq(local.length.word::<AB>(), next.length.word::<AB>());
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_word_eq(*local.dst_access.value(), *next.dst_access.value());

        builder
            .when(local.is_real)
            .assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());

        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length)
            .assert_zero(local.length.value::<AB>());

        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.src.reduce::<AB>() + local.length.value::<AB>() - AB::Expr::one(),
            local.src_address.value::<AB>(),
        );
        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.dst_access.value().reduce::<AB>() + local.length.value::<AB>() - AB::Expr::one(),
            local.dst_address.value::<AB>(),
        );
        builder.when_transition().when(local.is_real).when_not(local.is_last).assert_eq(
            local.src_address.value::<AB>() + AB::Expr::one(),
            next.src_address.value::<AB>(),
        );
        builder.when_transition().when(local.is_real).when_not(local.is_last).assert_eq(
            local.dst_address.value::<AB>() + AB::Expr::one(),
            next.dst_address.value::<AB>(),
        );

        // check that it does not go out of memory bounds
        builder
            .when(local.is_first)
            .assert_eq(local.dst_access.value().reduce::<AB>(), local.dst_address.value::<AB>());

        ElementAddressCols::<AB::Var>::range_check(builder, local.src_address);
        builder.when(local.is_first).assert_one(local.src_address.is_real::<AB>());
        builder.when(local.is_last).assert_one(local.src_address.is_real::<AB>());

        TableAddressCols::<AB::Var>::range_check(builder, local.dst_address);
        builder.when(local.is_first).assert_one(local.dst_address.is_real::<AB>());
        builder.when(local.is_last).assert_one(local.dst_address.is_real::<AB>());

        TableIdxCols::<AB::Var>::range_check(builder, local.table_idx);
        builder.when(local.is_first).assert_one(local.table_idx.is_real::<AB>());

        LengthCols::<AB::Var>::range_check(builder, local.length);
        builder.when(local.is_first).assert_one(local.length.is_real::<AB>());

        self.eval_memory_access(local, builder);

        builder.send_program(
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(Opcode::TableGet(0).code()),
            local.table_idx.word::<AB>(),
            local.is_first,
        );

        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC) + AB::Expr::one(),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(3 * UNIT),
            AB::Expr::from_canonical_u32(2),
            AB::Expr::from_canonical_u32(Opcode::TableInit(0).code()),
            Word::zero::<AB>(),
            local.src,
            local.length.word::<AB>(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_first,
        );
    }
}

impl TableInitChip {
    fn eval_memory_access<AB: SP1AirBuilder>(
        &self,
        local: &TableInitCols<AB::Var>,
        builder: &mut AB,
    ) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(3 * UNIT),
            &local.dst_access.clone(),
            local.is_first,
        );

        let src_addr = AB::Expr::from_canonical_u32(TypedAddress::Element(0).to_virtual_addr()) +
            local.src_address.value::<AB>() * unit.clone();

        let table_addr = AB::Expr::from_canonical_u32(TypedAddress::Table(0).to_virtual_addr()) +
            (local.dst_address.value::<AB>() +
                local.table_idx.value::<AB>() * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)) *
                unit;

        builder.eval_memory_access(
            local.shard,
            local.clk,
            src_addr,
            &local.src_read_access,
            local.is_non_zero_length,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            table_addr,
            &local.dst_write_access,
            local.is_non_zero_length,
        );
    }
}

impl<F> BaseAir<F> for TableInitChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}
