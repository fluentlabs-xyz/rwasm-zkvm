use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::size_of,
};

use crate::{air::MemoryAirBuilder, operations::field::range::FieldLtCols, utils::zeroed_f_vec};
use generic_array::GenericArray;
use itertools::Itertools;
use num::{BigUint, Zero};

use p3_air::{Air, BaseAir, PairBuilder};
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
 mod trace;
 mod column;
pub use trace::*;
pub use column::*;
use crate::{
    memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols},
    operations::field::field_op::FieldOpCols,
   
};
use rwasm::{N_MAX_TABLE_SIZE, mem_index::{AddressType, TABLE_ELEM_SIZE, UNIT}};
use rwasm::mem_index::ELEMENT_SEG_START;
#[derive(Default)]
pub struct TableChip{}
impl<AB> Air<AB> for TableChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableCols<AB::Var> = (*local).borrow();
        let next: &TableCols<AB::Var> = (*next).borrow();

        
        self.eval_memory_access(local, builder);

        builder.receive_syscall(local.shard, local.clk, AB::Expr::zero(),AB::Expr::zero(), AB::Expr::zero(),local.is_first,InteractionScope::Local);


    }

    

}

impl TableChip
 {
    fn eval_memory_access<AB: SP1AirBuilder>(&self, local: &TableCols<AB::Var>, builder: &mut AB) {
        let base_elem_addr = AB::Expr::from_canonical_u32(AddressType::Element(0).to_virtual_addr());
          let base_table_addr = AB::Expr::from_canonical_u32(AddressType::Table(0).to_virtual_addr());
        builder.eval_memory_access(
            local.clk,
            local.shard,
            local.sp,
            &local.length_access,
            local.is_first,
        );

         builder.eval_memory_access(
            local.clk,
            local.shard,
            local.sp+AB::Expr::from_canonical_u32(UNIT),
           &local.dst_access,
            local.is_first,
        );

         builder.eval_memory_access(
            local.clk,
            local.shard,
            local.sp+AB::Expr::from_canonical_u32(2*UNIT),
            &local.src_access,
            local.is_first,
        );

        builder.eval_memory_access(
            local.clk,
            local.shard,
            base_elem_addr+local.src_idx+local.idx,
            &local.src_read_access,
            local.is_real);

         builder.eval_memory_access(
            local.clk,
            local.shard,
            base_table_addr+local.table_idx*AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)+local.dst_idx+local.idx,
            &local.dst_write_access,
            local.is_real);
    }
}

impl<F> BaseAir<F> for TableChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}
