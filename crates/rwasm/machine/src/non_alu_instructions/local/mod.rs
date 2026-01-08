use std::borrow::BorrowMut;

use crate::{
    air::MemoryAirBuilder,
    memory::{MemoryCols, MemoryReadWriteCols, StackAddressCols},
    utils::pad_rows_fixed,
};

use p3_air::{Air, BaseAir};
use p3_matrix::Matrix;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::SP1AirBuilder;
use std::borrow::Borrow;

use p3_air::AirBuilder;
use p3_maybe_rayon::prelude::ParallelIterator;

use hashbrown::HashMap;
use p3_matrix::dense::RowMajorMatrix;
use rayon::slice::ParallelSlice;
use sp1_stark::air::MachineAir;

use p3_field::{AbstractField, PrimeField32};
use rwasm::mem_index::UNIT;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, LocalEvent},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_stark::Word;

pub const NUM_LOCAL_COLS: usize = size_of::<LocalCols<u8>>();

/// Columns representing the state of the Local Chip for a single execution step.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LocalCols<T> {
    /// Current Program Counter.
    pc: T,
    /// Current Stack Pointer.
    sp: T,
    /// Global Clock.
    clk: T,
    /// Execution Shard ID.
    shard: T,
    /// The offset of the local variable relative to the Stack Pointer.
    local_depth: Word<T>,
    /// Helper columns for validating the memory address calculation (Range Checks).
    depth_address: StackAddressCols<T>,
    /// Memory columns for reading/writing the local variable value.
    depth_access: MemoryReadWriteCols<T>,
    /// Flag indicating a LocalSet instruction.
    is_local_set: T,
    /// Flag indicating a LocalGet instruction.
    is_local_get: T,
    /// Flag indicating a LocalTee instruction.
    is_local_tee: T,
}

#[derive(Default)]
pub struct LocalChip;

impl<F> BaseAir<F> for LocalChip {
    fn width(&self) -> usize {
        NUM_LOCAL_COLS
    }
}

impl<AB> Air<AB> for LocalChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LocalCols<AB::Var> = (*local).borrow();

        // Ensure exactly one operation (or none) is active.
        let is_real = local.is_local_get + local.is_local_set + local.is_local_tee;

        builder.assert_bool(local.is_local_get);
        builder.assert_bool(local.is_local_set);
        builder.assert_bool(local.is_local_tee);
        builder.assert_bool(is_real.clone());

        // --- LocalGet Constraint ---
        // Reads a value from the stack at `sp + depth` and pushes it to the top.
        // - SP decreases by UNIT (stack grows down).
        // - op_a (result) receives the value read from memory.
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp - AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::LocalGet(0).code()),
            *local.depth_access.value(), // op_a: result value
            Word::zero::<AB>(),          // op_b: unused
            Word::zero::<AB>(),
            local.local_depth,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_local_get,
        );

        // --- LocalSet Constraint ---
        // Pops a value from the top of the stack and writes it to `sp + depth`.
        // - SP increases by UNIT (stack shrinks).
        // - op_b (input) receives the value to be written.
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::LocalSet(0).code()),
            Word::zero::<AB>(),          // op_a: unused
            *local.depth_access.value(), // op_b: input value
            Word::zero::<AB>(),
            local.local_depth,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_local_set,
        );

        // --- LocalTee Constraint ---
        // Peeks at the top value and writes it to `sp + depth`.
        // - SP remains unchanged.
        // - op_b (input) receives the value to be written.
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::LocalTee(0).code()),
            Word::zero::<AB>(),          // op_a: unused (stack top remains)
            *local.depth_access.value(), // op_b: input value
            Word::zero::<AB>(),
            local.local_depth,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_local_tee,
        );

        // Verify the memory address calculation: address = sp + depth.
        // This ensures the read/write happens at the correct stack offset.
        builder.when(is_real.clone()).assert_eq(
            local.sp + local.local_depth.reduce::<AB>(),
            local.depth_address.value::<AB>(),
        );

        // Enforce range checks on the calculated address to prevent field overflows.
        StackAddressCols::<AB::F>::do_range_check(builder, local.depth_address, is_real.clone());

        // Validate the memory access (Read/Write consistency) at the calculated address.
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp + local.local_depth.reduce::<AB>(),
            &local.depth_access,
            is_real,
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for LocalChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Local".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.local_events.iter() {
            let mut row = [F::zero(); NUM_LOCAL_COLS];
            let cols: &mut LocalCols<F> = row.as_mut_slice().borrow_mut();

            self.event_to_row(event, cols, output);

            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_LOCAL_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_LOCAL_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.local_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .local_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_LOCAL_COLS];
                    let cols: &mut LocalCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.local_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LocalChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &LocalEvent,
        cols: &mut LocalCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.shard = F::from_canonical_u32(event.shard);

        cols.local_depth = event.opcode.aux_value().into();

        cols.depth_access.populate(event.depth_access, blu);

        // Populate the helper columns for the address calculation constraint.
        // Uses wrapping_add to safely simulate the field addition in AIR.
        cols.depth_address.populate(event.sp.wrapping_add(event.opcode.aux_value()), blu, true);

        match event.opcode {
            Opcode::LocalGet(_) => cols.is_local_get = F::one(),
            Opcode::LocalSet(_) => cols.is_local_set = F::one(),
            Opcode::LocalTee(_) => cols.is_local_tee = F::one(),
            _ => unreachable!(),
        }
    }
}
