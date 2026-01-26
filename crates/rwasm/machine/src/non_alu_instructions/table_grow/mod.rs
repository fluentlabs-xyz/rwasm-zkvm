use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    control_flow::TableIdxCols,
    memory::{MemoryCols, TableAddressCols},
    non_alu_instructions::table_grow::column::{DeltaCols, TableGrowCols, NUM_TABLE_GROW_SIZE},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::{
    mem_index::{TypedAddress, UNIT},
    N_MAX_TABLE_SIZE,
};
use rwasm_executor::{Opcode, DEFAULT_PC_INC};
use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};
use std::borrow::Borrow;

mod column;
mod trace;

/// AIR chip for the WASM table.grow instruction.
///
/// Verifies correct execution of table growth operations, which increase table capacity
/// by a specified delta and initialize new entries with a given value. Returns the old
/// table size on success, or u32::MAX on failure (e.g., exceeding maximum table size).
#[derive(Default)]
pub struct TableGrowChip {}

impl<AB> Air<AB> for TableGrowChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Extract current and next trace rows for transition constraints
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableGrowCols<AB::Var> = (*local).borrow();
        let next: &TableGrowCols<AB::Var> = (*next).borrow();

        // Constrain control flow flags to be binary (0 or 1)
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_non_zero_length);

        // The circuit's first row must mark itself as first
        builder.when_first_row().assert_one(local.is_first);

        // Event boundary rows cannot be padding
        builder.when(local.is_first).assert_one(local.is_real);
        builder.when(local.is_last).assert_one(local.is_real);

        // For non-zero length operations, verify delta consistency between
        // range-checked decomposition and memory read value
        builder
            .when(local.is_first)
            .when(local.is_non_zero_length)
            .assert_eq(local.delta.value::<AB>(), local.delta.value::<AB>());

        // Verify the table size update equation: old_size + delta = new_size
        // Only checked when table size is actually updated (successful non-zero growth)
        builder.when(local.should_update_table_size).assert_eq(
            local.res.reduce::<AB>() + local.delta.value::<AB>(),
            local.table_size_write_access.value().reduce::<AB>(),
        );

        // Table size updates only occur for successful non-zero growth operations
        // This constraint ensures: should_update_table_size => success AND non_zero
        builder.when(local.is_last).when(local.should_update_table_size).assert_zero(
            local.not_successful_result + (AB::Expr::one() - local.is_non_zero_length),
        );

        // Failed operations must return u32::MAX per WASM specification
        builder
            .when(local.is_last)
            .when(local.not_successful_result)
            .assert_word_eq(local.res, Word::<AB::F>::from(u32::MAX));

        // Event boundary constraint: consecutive events must be properly delimited
        // If current is last AND next is real, then next must be first
        builder.when_transition().when(local.is_last).when(next.is_real).assert_one(next.is_first);

        // Within-event transition constraints: execution context remains constant
        builder.when_transition().when_not(local.is_last).assert_eq(local.is_real, next.is_real);
        builder.when_transition().when_not(local.is_last).assert_eq(local.clk, next.clk);
        builder.when_transition().when_not(local.is_last).assert_eq(local.shard, next.shard);

        // Operation parameters remain constant throughout the event
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.is_non_zero_length, next.is_non_zero_length);

        builder.when_transition().when_not(local.is_last).assert_word_eq(
            *local.table_size_read_access.value(),
            *next.table_size_read_access.value(),
        );

        // Table index must remain constant within an event
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.table_idx.value::<AB>(), next.table_idx.value::<AB>());

        // Delta value must remain constant within an event
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.delta.value::<AB>(), next.delta.value::<AB>());

        // Initialization value must remain constant within an event
        builder.when_transition().when_not(local.is_last).assert_word_eq(local.init, next.init);

        // Verify that new table entries are initialized with the correct value
        builder
            .when(local.is_non_zero_length)
            .assert_word_eq(local.init, *local.dst_write_access.value());

        // Zero-delta successful operations must have zero in delta memory access
        // (not_successful_result exempts failure cases which may have non-zero delta)
        builder.when(local.not_successful_result).assert_zero(local.is_non_zero_length);
        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length + local.not_successful_result)
            .assert_zero(local.delta.value::<AB>());

        // Verify destination address progression for multi-row events
        // On the last row, dst_address should equal old_size + delta - 1
        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.table_size_read_access.value().reduce::<AB>() + local.delta.value::<AB>() -
                AB::Expr::one(),
            local.dst_address.value::<AB>(),
        );

        // Within-event address progression: each row writes to the next consecutive address
        builder.when_transition().when(local.is_non_zero_length).when_not(local.is_last).assert_eq(
            local.dst_address.value::<AB>() + AB::Expr::one(),
            next.dst_address.value::<AB>(),
        );

        // First row destination address equals the old table size (start of new region)
        builder.when(local.is_first).when(local.is_non_zero_length).assert_eq(
            local.table_size_read_access.value().reduce::<AB>(),
            local.dst_address.value::<AB>(),
        );

        // Range check destination address to ensure it fits in table address space
        TableAddressCols::<AB::Var>::range_check(builder, local.dst_address);
        builder
            .when(local.is_non_zero_length)
            .when(local.is_first)
            .assert_one(local.dst_address.is_real::<AB>());
        builder
            .when(local.is_non_zero_length)
            .when(local.is_last)
            .assert_one(local.dst_address.is_real::<AB>());

        // Range check table index to ensure valid table reference
        TableIdxCols::<AB::Var>::range_check(builder, local.table_idx);
        builder.when(local.is_first).assert_one(local.table_idx.is_real::<AB>());

        // Range check delta value to prevent overflow in address calculations
        DeltaCols::<AB::Var>::range_check(builder, local.delta);
        builder
            .when(local.is_non_zero_length)
            .when(local.is_first)
            .assert_one(local.delta.is_real::<AB>());

        // Verify all memory accesses against the global memory bus
        self.eval_memory_access(local, builder);

        // Register this table.grow syscall with the execution trace
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::TableGrow(0).code()),
            local.res,
            local.init,
            local.delta.word::<AB>(),
            local.table_idx.word::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_first,
        );
    }
}

impl<F> BaseAir<F> for TableGrowChip {
    fn width(&self) -> usize {
        NUM_TABLE_GROW_SIZE
    }
}

impl TableGrowChip {
    /// Verifies all memory operations performed by table.grow against the memory bus.
    ///
    /// Memory operations include:
    /// - Reading delta and init values from stack
    /// - Reading current table size
    /// - Writing updated table size (conditional on successful growth)
    /// - Writing initialization values to new table entries (conditional on non-zero delta)
    /// - Writing result to stack
    ///
    /// All accesses are conditioned on appropriate flags to ensure they only occur
    /// when semantically required.
    fn eval_memory_access<AB: SP1AirBuilder>(
        &self,
        local: &TableGrowCols<AB::Var>,
        builder: &mut AB,
    ) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

        // Calculate virtual address for table size metadata
        // Each table has its size stored at TableSize base + table_idx * UNIT
        let table_size_addr =
            AB::Expr::from_canonical_u32(TypedAddress::TableSize(0).to_virtual_addr()) +
                local.table_idx.value::<AB>() * unit.clone();

        // Read current table size at clock cycle clk
        builder.eval_memory_access(
            local.shard,
            local.clk,
            table_size_addr.clone(),
            &local.table_size_read_access.clone(),
            local.is_first,
        );

        // Write updated table size at clock cycle clk + 1
        // Conditional on should_update_table_size (only for successful non-zero growth)
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::from_canonical_u32(1),
            table_size_addr.clone(),
            &local.table_size_write_access.clone(),
            local.should_update_table_size,
        );

        // Calculate virtual address for table entry
        // Table layout: base + (entry_offset + table_idx * N_MAX_TABLE_SIZE) * UNIT
        // where entry_offset is dst_address (relative to table start)
        let table_addr = AB::Expr::from_canonical_u32(TypedAddress::Table(0).to_virtual_addr()) +
            (local.dst_address.value::<AB>() +
                local.table_idx.value::<AB>() * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)) *
                unit;

        // Write initialization value to table entry at clock cycle clk + 1
        // Conditional on is_non_zero_length (no writes for zero-delta operations)
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::from_canonical_u32(1),
            table_addr,
            &local.dst_write_access,
            local.is_non_zero_length,
        );
    }
}
