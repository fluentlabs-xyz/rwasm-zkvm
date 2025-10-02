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
        let local = main.row_slice(0);
        let local: &TableCols<AB::Var> = (*local).borrow();

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp,
            &local.length_access.clone(),
            local.is_real,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            &local.src_access.clone(),
            local.is_real,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(2 * UNIT),
            &local.dst_access.clone(),
            local.is_real,
        );

        self.eval_memory_access(local, builder);

        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(SyscallCode::TABLE_INIT.syscall_id() as u32),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
            InteractionScope::Local,
        );
    }
}

impl TableChip {
    fn eval_memory_access<AB: SP1AirBuilder>(&self, local: &TableCols<AB::Var>, builder: &mut AB) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

        let mut src_addr = AB::Expr::from_canonical_u32(AddressType::Element(0).to_virtual_addr())
            + local.src_access.value().reduce::<AB>() * unit.clone();

        let mut table_addr = AB::Expr::from_canonical_u32(AddressType::Table(0).to_virtual_addr())
            + (local.dst_access.value().reduce::<AB>()
                + local.table_idx * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE))
                * unit.clone();

        let n = local.length_access.value().reduce::<AB>();

        // check case when n = 0
        builder.when_not(local.inner[0].is_real.clone()).assert_zero(n.clone());

        builder
            .when(local.inner[N_MAX_TABLE_SIZE as usize - 1].is_real.clone())
            .assert_eq(n.clone(), AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE - 1));

        for idx in 0..N_MAX_TABLE_SIZE as usize {
            let current = &local.inner[idx];

            builder.assert_bool(current.is_real.clone());

            if idx != N_MAX_TABLE_SIZE as usize - 1 {
                let next = &local.inner[idx + 1];

                let idx = idx as u32;

                builder
                    .when_ne(n.clone(), AB::Expr::from_canonical_u32(idx + 1))
                    .assert_eq(current.is_real.clone(), next.is_real.clone());

                builder
                    .when_ne(current.is_real.clone(), next.is_real.clone())
                    .assert_one(current.is_real.clone());

                builder
                    .when_ne(current.is_real.clone(), next.is_real.clone())
                    .assert_eq(n.clone(), AB::Expr::from_canonical_u32(idx + 1));
            }

            // TODO:(Aliaksei) check memory out of bounds
            // builder
            //     .when_ne(AB::Expr::from_canonical_u32(ELEMENT_SEG_END) - src_addr.clone())
            //     .assert_zero(current.is_real.clone());
            // builder
            //     .when_not(AB::Expr::from_canonical_u32(TABLE_SEG_END) - table_addr.clone())
            //     .assert_zero(current.is_real.clone());

            builder.when(current.is_real.clone()).assert_eq(
                current.src_read_access.value().reduce::<AB>(),
                current.dst_write_access.value().reduce::<AB>(),
            );

            builder.eval_memory_access(
                local.shard,
                local.clk,
                src_addr.clone(),
                &current.src_read_access.clone(),
                current.is_real.clone(),
            );

            builder.eval_memory_access(
                local.shard,
                local.clk + AB::Expr::from_canonical_u32(1),
                table_addr.clone(),
                &current.dst_write_access.clone(),
                current.is_real.clone(),
            );

            src_addr += unit.clone();
            table_addr += unit.clone();
        }
    }
}

impl<F> BaseAir<F> for TableChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}
