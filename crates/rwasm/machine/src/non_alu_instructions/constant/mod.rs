use crate::utils::pad_rows_fixed;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::{mem_index::UNIT, Opcode};
use rwasm_executor::{ExecutionRecord, Program, DEFAULT_PC_INC};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::SP1AirBuilder, Word};
use std::borrow::{Borrow, BorrowMut};

use sp1_stark::air::MachineAir;

pub const NUM_CONST_COLS: usize = size_of::<ConstCols<u8>>();

#[derive(Default)]
pub struct ConstChip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstCols<T> {
    /// The program counter.
    pub pc: T,

    /// The current stack pointer.
    pub sp: T,

    pub aux_val: Word<T>,

    pub is_const: T,

    pub is_drop: T,
}

impl<F> BaseAir<F> for ConstChip {
    fn width(&self) -> usize {
        NUM_CONST_COLS
    }
}

impl<AB> Air<AB> for ConstChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ConstCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_const);
        builder.assert_bool(local.is_drop);

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp - AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Const(0.into()).code()),
            local.aux_val,
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.aux_val,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_const,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::Drop.code()),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_drop,
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for ConstChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Const".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.const_events.iter() {
            let mut row = [F::zero(); NUM_CONST_COLS];
            let cols: &mut ConstCols<F> = row.as_mut_slice().borrow_mut();

            println!("$$$$$$$$$$$$$$ event.sp:{}", event.sp);

            cols.pc = F::from_canonical_u32(event.pc);
            cols.sp = F::from_canonical_u32(event.sp);

            match event.opcode {
                Opcode::I32Const(aux_val) => {
                    cols.aux_val = event.opcode.aux_value().into();
                    cols.is_const = F::one();
                }
                Opcode::Drop => {
                    cols.is_drop = F::one();
                }
                _ => unreachable!(),
            }

            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_CONST_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_CONST_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.const_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}
