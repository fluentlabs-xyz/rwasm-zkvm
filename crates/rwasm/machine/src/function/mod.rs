use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::{collections::HashMap, mem::offset_of};

use crate::{air::FuncCallAirBuilder, utils::pad_rows_fixed};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::PrimeField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_derive::AlignedBorrow;
use sp1_rwasm_executor::{ExecutionRecord, Program};
use sp1_stark::air::{MachineAir, SP1AirBuilder};

/// The number of preprocessed program columns.
pub const NUM_FUNCCALL_PREPROCESSED_COLS: usize = size_of::<FunccallPreprocessedCols<u8>>();

/// The number of columns for the program multiplicities.
pub const NUM_FUNCCALL_MULT_COLS: usize = size_of::<FunccallMultiplicityCols<u8>>();

/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct FunccallPreprocessedCols<T> {
    pub function:T,
    pub index_by_function:T,
}



/// The column layout for the chip.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct FunccallMultiplicityCols<T> {
    pub shard: T,
    pub multiplicity: T,
}

/// A chip that implements addition for the opcodes ADD and ADDI.
#[derive(Default)]
pub struct FunccallChip;

impl FunccallChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F: PrimeField> MachineAir<F> for FunccallChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Funccall".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_FUNCCALL_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let mut rows = program
            .index_by_offset
            .iter()
            .enumerate()
            .map(|(func,offset)| {
               
                let mut row = [F::zero(); NUM_FUNCCALL_MULT_COLS];
                let cols: &mut FunccallPreprocessedCols<F> = row.as_mut_slice().borrow_mut();
                cols.function =F::from_canonical_usize(func);
                cols.index_by_function = F::from_canonical_u32(*offset);
                row
    
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_FUNCCALL_MULT_COLS],
            program.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        let trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_FUNCCALL_MULT_COLS,
        );

        Some(trace)
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.

        // Collect the number of times each instruction is called from the cpu events.
        // Store it as a map of PC -> count.
        let mut call_counts = HashMap::new();
        input.function_call_events.iter().for_each(|event| {
           
           call_counts.entry((*event).func_index).and_modify(|count| *count += 1).or_insert(1);
        });

        let mut rows = input
            .program
            .index_by_offset
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, _)| {
                let mut row = [F::zero(); NUM_FUNCCALL_MULT_COLS];
                let cols: &mut FunccallMultiplicityCols<F> = row.as_mut_slice().borrow_mut();
                cols.shard = F::from_canonical_u32(input.public_values.execution_shard);
                cols.multiplicity =
                    F::from_canonical_usize(*call_counts.get(&(i as u32)).unwrap_or(&0));
                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_FUNCCALL_MULT_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_FUNCCALL_MULT_COLS)
    }

    fn included(&self, _: &Self::Record) -> bool {
        true
    }
}

impl<F> BaseAir<F> for FunccallChip {
    fn width(&self) -> usize {
        NUM_FUNCCALL_MULT_COLS
    }
}

impl<AB> Air<AB> for FunccallChip
where
    AB: FuncCallAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let prep_local = preprocessed.row_slice(0);
        let prep_local: &FunccallPreprocessedCols<AB::Var> = (*prep_local).borrow();
        let mult_local = main.row_slice(0);
        let mult_local: &FunccallMultiplicityCols<AB::Var> = (*mult_local).borrow();

        // Constrain the interaction with CPU table
        builder.receive_function_call(
            prep_local.function,
            prep_local.index_by_function,
            mult_local.shard,
            mult_local.multiplicity,
        );
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use hashbrown::HashMap;
    use p3_baby_bear::BabyBear;

    use p3_matrix::dense::RowMajorMatrix;
    use sp1_rwasm_executor::{ExecutionRecord, Instruction, Opcode, Program};
    use sp1_stark::air::MachineAir;

    use crate::{function::FunccallChip, program::ProgramChip};

    #[test]
    fn generate_trace() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     add x31, x30, x29
        let instructions = vec![];
        let program = Program::new(instructions, 0,0);
        let shard = ExecutionRecord {
            program: Arc::new(program),
            ..Default::default()
        };
        let chip = FunccallChip::new();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values)
    }
}
