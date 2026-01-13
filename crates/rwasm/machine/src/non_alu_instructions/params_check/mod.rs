use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadCols},
    utils::pad_rows_fixed,
};
use hashbrown::HashMap;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use rwasm::{mem_index::LAST_SIG_ADDR, Opcode};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::SP1AirBuilder, Word};
use std::borrow::{Borrow, BorrowMut};

use sp1_stark::air::MachineAir;

pub const NUM_PARAMS_CHECK_COLS: usize = size_of::<ParamsCheckCols<u8>>();

#[derive(Default)]
pub struct ParamsCheckChip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ParamsCheckCols<T> {
    pub shard: T,

    pub clk: T,

    /// The program counter.
    pub pc: T,

    /// The current stack pointer.
    pub sp: T,

    pub aux_val: Word<T>,

    pub params_access: MemoryReadCols<T>,

    pub is_signature_check: T,
}

impl<F> BaseAir<F> for ParamsCheckChip {
    fn width(&self) -> usize {
        NUM_PARAMS_CHECK_COLS
    }
}

impl<AB> Air<AB> for ParamsCheckChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ParamsCheckCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_signature_check);

        builder.eval_memory_access(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(LAST_SIG_ADDR),
            &local.params_access,
            local.is_signature_check,
        );

        builder
            .when(local.is_signature_check)
            .assert_word_eq(*local.params_access.value(), local.aux_val);

        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::SignatureCheck(0).code()),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.aux_val,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_signature_check,
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for ParamsCheckChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ParamsCheck".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.params_check_events.iter() {
            let mut row = [F::zero(); NUM_PARAMS_CHECK_COLS];
            let cols: &mut ParamsCheckCols<F> = row.as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.pc = F::from_canonical_u32(event.pc);
            cols.sp = F::from_canonical_u32(event.sp);

            cols.params_access.populate(event.params_read_record, output);

            match event.opcode {
                Opcode::SignatureCheck(aux_val) => {
                    cols.aux_val = event.opcode.aux_value().into();
                    cols.is_signature_check = F::one();
                }
                _ => unreachable!(),
            }

            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_PARAMS_CHECK_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_PARAMS_CHECK_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.params_check_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .params_check_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_PARAMS_CHECK_COLS];
                    let cols: &mut ParamsCheckCols<F> = row.as_mut_slice().borrow_mut();

                    cols.params_access.populate(event.params_read_record, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.params_check_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}
