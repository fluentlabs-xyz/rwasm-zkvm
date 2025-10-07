use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::size_of,
};

use crate::{air::MemoryAirBuilder, operations::field::range::FieldLtCols, utils::zeroed_f_vec};
use generic_array::GenericArray;
use itertools::Itertools;
use num::{BigUint, Zero};

use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};

use crate::air::WordAirBuilder;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, FieldOperation, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_curves::{
    params::{Limbs, NumLimbs},
    weierstrass::{FieldType, FpOpField},
};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{BaseAirBuilder, InteractionScope, MachineAir, Polynomial, SP1AirBuilder};
mod column;
mod trace;
use crate::{
    memory::{value_as_limbs, MemoryCols, MemoryReadCols, MemoryWriteCols},
    operations::field::field_op::FieldOpCols,
};
pub use column::*;
use rwasm::mem_index::{ELEMENT_SEG_END, ELEMENT_SEG_START, TABLE_SEG_END};
use rwasm::{
    mem_index::{AddressType, TABLE_ELEM_SIZE, UNIT},
    N_MAX_TABLE_SIZE,
};
pub use trace::*;
#[derive(Default)]
pub struct TableChip {}
impl<AB> Air<AB> for TableChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableCols<AB::Var> = (*local).borrow();
        let next: &TableCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        builder.when_first_row().assert_one(local.is_first);

        builder.when_transition().when(local.is_last).when(next.is_real).assert_one(next.is_first);

        builder.when_transition().when_not(local.is_last).assert_eq(local.is_real, next.is_real);

        builder.when_transition().when_not(local.is_last).assert_eq(local.clk, next.clk);
        builder.when_transition().when_not(local.is_last).assert_eq(local.shard, next.shard);

        builder.when(local.is_last).assert_one(local.is_real);
        builder.when(local.is_first).assert_one(local.is_real);

        builder
            .when(local.is_real)
            .assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());

        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length)
            .assert_word_zero(*local.length_access.value());

        self.eval_memory_access(local, builder);

        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(SyscallCode::TABLE_INIT.syscall_id() as u32),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_first,
            InteractionScope::Local,
        );
    }
}

impl TableChip {
    fn eval_memory_access<AB: SP1AirBuilder>(&self, local: &TableCols<AB::Var>, builder: &mut AB) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp,
            &local.length_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            &local.src_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(2 * UNIT),
            &local.dst_access.clone(),
            local.is_first,
        );

        let src_addr = AB::Expr::from_canonical_u32(AddressType::Element(0).to_virtual_addr())
            + (local.src_access.value().reduce::<AB>() + local.idx) * unit.clone();

        let table_addr = AB::Expr::from_canonical_u32(AddressType::Table(0).to_virtual_addr())
            + (local.table_idx * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)
                + local.dst_access.value().reduce::<AB>()
                + local.idx)
                * unit;

        builder.eval_memory_access(
            local.shard,
            local.clk,
            src_addr,
            &local.src_read_access,
            local.is_non_zero_length,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::from_canonical_u32(1),
            table_addr,
            &local.dst_write_access,
            local.is_non_zero_length,
        );
    }
}

impl<F> BaseAir<F> for TableChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}
