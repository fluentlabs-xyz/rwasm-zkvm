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
use sp1_stark::air::{BaseAirBuilder, InteractionScope, MachineAir, Polynomial, SP1AirBuilder,};
use crate::air::WordAirBuilder;
mod column;
mod trace;
use crate::{
    memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols,MemoryCols},
    operations::field::field_op::FieldOpCols,
};
pub use column::*;
use rwasm::mem_index::ELEMENT_SEG_START;
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
        let local_main = main.row_slice(0);
        let local_main: &TableCols<AB::Var> = (*local_main).borrow();
        
        for idx in 0..N_MAX_TABLE_SIZE as usize{
            let local = &local_main.inner[idx];
             self.eval_memory_access(local, builder);
            builder.when(local.is_real.clone()).assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());


            builder.when(local.is_first.clone()).assert_eq(local.length.clone(), local.length_access.value().reduce::<AB>());
            builder.when(local.is_first.clone()).assert_eq(local.src_idx.clone(), local.src_access.value().reduce::<AB>());
            builder.when(local.is_first.clone()).assert_eq(local.dst_idx.clone(), local.dst_access.value().reduce::<AB>());
       
        if idx <N_MAX_TABLE_SIZE as usize -1{
            let next = &local_main.inner[idx+1];
             builder.when(next.is_real.clone()).assert_eq(local.src_idx,next.src_idx);
             builder.when(next.is_real.clone()).assert_eq(local.dst_idx,next.dst_idx);
             builder.when(next.is_real.clone()).assert_eq(local.length,next.length);
             builder.when(next.is_real.clone()).assert_eq(local.idx+AB::Expr::one(),next.idx);
        }
        
        
        
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
}

impl TableChip {
    fn eval_memory_access<AB: SP1AirBuilder>(&self, local: &TableSubCols<AB::Var>, builder: &mut AB) {
        let base_elem_addr =
            AB::Expr::from_canonical_u32(AddressType::Element(0).to_virtual_addr());
        let base_table_addr = AB::Expr::from_canonical_u32(AddressType::Table(0).to_virtual_addr());
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
            &local.dst_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + AB::Expr::from_canonical_u32(2 * UNIT),
            &local.src_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            base_elem_addr + (local.src_idx + local.idx) * AB::Expr::from_canonical_u32(UNIT),
            &local.src_read_access.clone(),
            local.is_real,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk+AB::Expr::from_canonical_u32(1),
            base_table_addr
                + local.table_idx * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)
                + (local.dst_idx + local.idx) * AB::Expr::from_canonical_u32(UNIT),
            &local.dst_write_access.clone(),
            local.is_real,
        );
    }
}

impl<F> BaseAir<F> for TableChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}
