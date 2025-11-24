use std::borrow::Borrow;

use crate::air::{MemoryAirBuilder, WordAirBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm_executor::syscalls::SyscallCode;
use sp1_stark::air::{BaseAirBuilder, InteractionScope, SP1AirBuilder};

mod column;
mod trace;

use crate::memory::MemoryCols;
pub use column::*;
use rwasm::{
    mem_index::{TypedAddress, UNIT},
    N_MAX_ELEM_SEGMENTS_BITS, N_MAX_TABLE_SIZE,
};

/// Chip for handling WebAssembly table operations (table.init, table.fill, table.copy)
#[derive(Default)]
pub struct TableCopyChip {}

impl<AB: AirBuilder> Air<AB> for TableCopyChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableInitCols<AB::Var> = (*local).borrow();
        let next: &TableInitCols<AB::Var> = (*next).borrow();

        // Check if current/next row is a real operation (not padding)
        let is_real = local.is_table_init + local.is_table_fill + local.is_table_copy;
        let next_is_real = next.is_table_init + next.is_table_fill + next.is_table_copy;

        // === GROUP 1: Row-level constraints (single row validation) ===
        self.eval_row_constraints(builder, local, is_real.clone());

        // === GROUP 2: Transition constraints (between rows) ===
        self.eval_transition_constraints(builder, local, next, is_real.clone(), next_is_real);

        // === GROUP 3: Address constraints (address validation and increments) ===
        self.eval_address_constraints(builder, local, next, is_real.clone());

        // === GROUP 4: External interactions (memory and syscall) ===
        self.eval_memory_access(local, builder);
        self.eval_syscall_interaction(local, builder);
    }
}

impl TableCopyChip {
    // ============================================================================
    // GROUP 1: Row-level constraints
    // ============================================================================

    /// Evaluates all constraints that apply to individual rows
    fn eval_row_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Boolean constraints
        self.eval_boolean_constraints(builder, local, is_real.clone());

        // Row boundary markers (first/last)
        self.eval_row_boundary_constraints(builder, local, is_real.clone());

        // Operation-specific logic
        self.eval_operation_constraints(builder, local);
    }

    /// Ensures all boolean flag fields contain only 0 or 1
    fn eval_boolean_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Operation type flags
        builder.assert_bool(local.is_table_init);
        builder.assert_bool(local.is_table_fill);
        builder.assert_bool(local.is_table_copy);

        // Event boundary flags
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(is_real);

        // Operation characteristic flags
        builder.assert_bool(local.is_non_zero_length);
        builder.assert_bool(local.should_read_elements);
        builder.assert_bool(local.should_read_src_table);

        // Ensure should_read_elements and should_read_src_table are mutually exclusive
        builder.assert_bool(local.should_read_elements + local.should_read_src_table);
    }

    /// Validates row boundary markers and real row constraints
    fn eval_row_boundary_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // First row in the trace must be marked as first
        builder.when_first_row().assert_one(local.is_first);

        // Last and first rows must always be real operations (not padding)
        builder.when(local.is_last).assert_one(is_real.clone());
        builder.when(local.is_first).assert_one(is_real);
    }

    /// Validates operation-specific constraints (aux_value, length, value propagation)
    fn eval_operation_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Validate aux_value encoding based on operation type
        self.eval_aux_value_encoding(builder, local);

        // Validate length parameter and source reading flags
        self.eval_length_and_source_flags(builder, local);

        // Ensure correct value propagation from source to destination
        self.eval_value_propagation(builder, local);
    }

    /// Validates aux_value encoding based on operation type
    /// - table.init: aux_value = 0 (all elements stored in segment 0)
    /// - table.fill: aux_value = dst_table_idx
    /// - table.copy: aux_value = (dst_table_idx << 16) | src_table_idx
    fn eval_aux_value_encoding<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // table.init: In rwasm, all elements are stored in segment 0
        builder.when(local.is_first).when(local.is_table_init).assert_zero(local.aux_value);

        // table.fill: aux_value encodes destination table index only
        builder
            .when(local.is_first)
            .when(local.is_table_fill)
            .assert_eq(local.aux_value, local.dst_table_idx.value::<AB>());

        // table.copy: aux_value packs both table indices
        builder.when(local.is_first).when(local.is_table_copy).assert_eq(
            local.aux_value,
            local.dst_table_idx.value::<AB>() * AB::Expr::from_canonical_u32(1 << 16) +
                local.src_table_idx.value::<AB>(),
        );
    }

    /// Validates length parameter and source reading flags
    fn eval_length_and_source_flags<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // table.fill never reads from a source (it fills with a single value)
        builder
            .when(local.is_table_fill)
            .assert_zero(local.should_read_elements + local.should_read_src_table);

        // table.init with non-zero length must read from element segment
        builder
            .when(local.is_table_init)
            .when(local.is_non_zero_length)
            .assert_one(local.should_read_elements);

        // table.copy with non-zero length must read from source table
        builder
            .when(local.is_table_copy)
            .when(local.is_non_zero_length)
            .assert_one(local.should_read_src_table);

        // table.copy must always read source table size on first row
        builder
            .when(local.is_table_copy)
            .when(local.is_first)
            .assert_one(local.should_read_src_table_size);

        // table.fill and table.init don't need source table size
        builder
            .when(local.is_table_fill + local.is_table_init)
            .assert_zero(local.should_read_src_table_size);

        // Zero-length operations don't read length parameter on first row
        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length)
            .assert_word_zero(*local.length_access.value());
    }

    /// Ensures correct value propagation from source to destination
    fn eval_value_propagation<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // table.init and table.copy: dst_value = src_value
        builder
            .when(local.is_table_init + local.is_table_copy)
            .assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());

        // table.fill: dst_value = fill_value (from src_access stack parameter)
        builder
            .when(local.is_table_fill)
            .when(local.is_non_zero_length)
            .assert_word_eq(*local.src_access.value(), *local.dst_write_access.value());
    }

    // ============================================================================
    // GROUP 2: Transition constraints
    // ============================================================================

    /// Evaluates all transition constraints between consecutive rows
    fn eval_transition_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
        next_is_real: AB::Expr,
    ) {
        // Event boundary transitions (last -> first)
        self.eval_event_boundaries(builder, local, next, next_is_real.clone());

        // Within-event transitions (constant values)
        self.eval_within_event_transitions(builder, local, next, is_real, next_is_real);
    }

    /// Ensures correct behavior at event boundaries
    fn eval_event_boundaries<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        next_is_real: AB::Expr,
    ) {
        // If current row is last and next is real, next must be first
        builder.when_transition().when(local.is_last).when(next_is_real).assert_one(next.is_first);
    }

    /// Ensures constant values within a single operation event
    fn eval_within_event_transitions<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
        next_is_real: AB::Expr,
    ) {
        // When not last row (within same event), values must remain constant
        builder.when_transition().when_not(local.is_last).assert_eq(is_real, next_is_real);

        builder.when_transition().when_not(local.is_last).assert_eq(local.clk, next.clk);

        builder.when_transition().when_not(local.is_last).assert_eq(local.shard, next.shard);

        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.is_table_init, next.is_table_init);

        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.is_table_fill, next.is_table_fill);

        // Table sizes must remain constant
        builder.when_transition().when_not(local.is_last).assert_word_eq(
            *local.table_dst_size_read_access.value(),
            *next.table_dst_size_read_access.value(),
        );

        builder.when_transition().when_not(local.is_last).assert_word_eq(
            *local.table_src_size_read_access.value(),
            *next.table_src_size_read_access.value(),
        );

        // Table indices must remain constant
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.dst_table_idx.value::<AB>(), next.dst_table_idx.value::<AB>());

        builder
            .when_transition()
            .when_not(local.is_last)
            .when(local.is_table_copy)
            .assert_eq(local.src_table_idx.value::<AB>(), next.src_table_idx.value::<AB>());

        // Stack parameters must remain constant
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_word_eq(*local.src_access.value(), *next.src_access.value());

        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_word_eq(*local.length_access.value(), *next.length_access.value());

        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_word_eq(*local.dst_access.value(), *next.dst_access.value());

        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.is_non_zero_length, next.is_non_zero_length);
    }

    // ============================================================================
    // GROUP 3: Address constraints
    // ============================================================================

    /// Evaluates all address-related constraints
    fn eval_address_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Initial addresses (first row of event)
        self.eval_initial_addresses(builder, local);

        // Address increments (between consecutive rows)
        self.eval_address_increments(builder, local, next, is_real);

        // Final addresses (last row of event)
        self.eval_final_addresses(builder, local);

        // Range checks for all addresses and indices
        self.eval_range_checks(builder, local);
    }

    /// Validates initial address values on first row of event
    fn eval_initial_addresses<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // First row: src_address = src_offset (for table.init and table.copy)
        builder
            .when(local.is_first)
            .when(local.is_table_init + local.is_table_copy)
            .assert_eq(local.src_access.value().reduce::<AB>(), local.src_address.value::<AB>());

        // First row: dst_address = dst_offset (for all operations)
        builder
            .when(local.is_first)
            .assert_eq(local.dst_access.value().reduce::<AB>(), local.dst_address.value::<AB>());
    }

    /// Ensures addresses increment correctly within an operation event
    fn eval_address_increments<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Source address increments by 1 for table.init and table.copy
        builder
            .when_transition()
            .when(local.is_table_init + local.is_table_copy)
            .when_not(local.is_last)
            .assert_eq(
                local.src_address.value::<AB>() + AB::Expr::one(),
                next.src_address.value::<AB>(),
            );

        // Destination address increments by 1 for all operations
        builder.when_transition().when(is_real).when_not(local.is_last).assert_eq(
            local.dst_address.value::<AB>() + AB::Expr::one(),
            next.dst_address.value::<AB>(),
        );
    }

    /// Validates address values at operation boundaries (last row)
    fn eval_final_addresses<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Last row: src_address = src_offset + length - 1 (for table.init/copy)
        builder
            .when(local.is_last)
            .when(local.should_read_elements + local.should_read_src_table)
            .assert_eq(
                local.src_access.value().reduce::<AB>() +
                    local.length_access.value().reduce::<AB>() -
                    AB::Expr::one(),
                local.src_address.value::<AB>(),
            );

        // Last row: dst_address = dst_offset + length - 1 (for all operations)
        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.dst_access.value().reduce::<AB>() + local.length_access.value().reduce::<AB>() -
                AB::Expr::one(),
            local.dst_address.value::<AB>(),
        );
    }

    /// Range checking constraints for addresses and indices
    fn eval_range_checks<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Stack pointer range check
        StackAddressCols::<AB::Var>::range_check(builder, local.sp);
        builder.when(local.is_first).assert_one(local.sp.do_check::<AB>());

        // Source address range check with operation-specific bounds
        self.eval_source_address_range_check(builder, local);

        // Destination address range check
        let dst_table_size_value = local.table_dst_size_read_access.value();
        DynamicDstAddressCols::<AB::Var>::range_check(
            builder,
            local.dst_address,
            dst_table_size_value[0].into(),
            dst_table_size_value[1].into(),
        );
        builder.when(local.is_first).assert_one(local.dst_address.do_check::<AB>());
        builder.when(local.is_last).assert_one(local.dst_address.do_check::<AB>());

        // Table index range checks
        TableIdxCols::<AB::Var>::range_check(builder, local.src_table_idx);
        builder
            .when(local.is_table_copy)
            .when(local.is_first)
            .assert_one(local.src_table_idx.do_check::<AB>());

        TableIdxCols::<AB::Var>::range_check(builder, local.dst_table_idx);
        builder.when(local.is_first).assert_one(local.dst_table_idx.do_check::<AB>());
    }

    /// Helper: Range check source address with operation-specific bounds
    fn eval_source_address_range_check<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Set source address upper bound based on operation type
        // table.copy: source end = source table size
        builder
            .when(local.is_table_copy)
            .assert_eq(local.src_end[0], local.table_src_size_read_access.value()[0]);

        builder
            .when(local.is_table_copy)
            .assert_eq(local.src_end[1], local.table_src_size_read_access.value()[1]);

        // table.init: source end = max element segments size (constant)
        builder.when(local.is_table_init).assert_eq(
            local.src_end[0],
            AB::Expr::from_canonical_u8(N_MAX_ELEM_SEGMENTS_BITS as u8),
        );

        builder.when(local.is_table_init).assert_eq(
            local.src_end[1],
            AB::Expr::from_canonical_u8((N_MAX_ELEM_SEGMENTS_BITS >> 8) as u8),
        );

        // Perform range check on first and last rows
        DynamicSrcAddressCols::<AB::Var>::range_check(
            builder,
            local.src_address,
            local.src_end[0].into(),
            local.src_end[1].into(),
        );

        builder
            .when(local.is_first)
            .when(local.is_table_init + local.is_table_copy)
            .assert_one(local.src_address.do_check::<AB>());

        builder
            .when(local.is_last)
            .when(local.is_table_init + local.is_table_copy)
            .assert_one(local.src_address.do_check::<AB>());
    }

    // ============================================================================
    // GROUP 4: External interactions
    // ============================================================================

    /// Memory access constraints for communicating with memory subsystem
    fn eval_memory_access<AB: SP1AirBuilder>(
        &self,
        local: &TableInitCols<AB::Var>,
        builder: &mut AB,
    ) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

        // Read stack parameters (dst_offset, src_offset, length) from stack
        // Stack layout: sp+0 = length, sp+1 = src_offset, sp+2 = dst_offset
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp.value::<AB>(),
            &local.length_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp.value::<AB>() + AB::Expr::from_canonical_u32(UNIT),
            &local.src_access.clone(),
            local.is_first,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.sp.value::<AB>() + AB::Expr::from_canonical_u32(2 * UNIT),
            &local.dst_access.clone(),
            local.is_first,
        );

        // Read element value from element segment (table.init only)
        let elements_src_addr =
            AB::Expr::from_canonical_u32(TypedAddress::Element(0).to_virtual_addr()) +
                local.src_address.value::<AB>() * unit.clone();
        builder.eval_memory_access(
            local.shard,
            local.clk,
            elements_src_addr,
            &local.src_read_access,
            local.should_read_elements,
        );

        // Read value from source table (table.copy only)
        let src_table_addr = AB::Expr::from_canonical_u32(TypedAddress::Table(0).to_virtual_addr()) +
            (local.src_address.value::<AB>() +
                local.src_table_idx.value::<AB>() *
                    AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)) *
                unit.clone();
        builder.eval_memory_access(
            local.shard,
            local.clk,
            src_table_addr,
            &local.src_read_access,
            local.should_read_src_table,
        );

        // Read source table size (table.copy only, on first row)
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            AB::Expr::from_canonical_u32(TypedAddress::TableSize(0).to_virtual_addr()) +
                local.src_table_idx.value::<AB>() * unit.clone(),
            &local.table_src_size_read_access,
            local.should_read_src_table_size,
        );

        // Read destination table size (all operations, on first row)
        builder.eval_memory_access(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(TypedAddress::TableSize(0).to_virtual_addr()) +
                local.dst_table_idx.value::<AB>() * unit.clone(),
            &local.table_dst_size_read_access,
            local.is_first,
        );

        // Write value to destination table (all operations with length > 0)
        let dst_addr = AB::Expr::from_canonical_u32(TypedAddress::Table(0).to_virtual_addr()) +
            (local.dst_address.value::<AB>() +
                local.dst_table_idx.value::<AB>() *
                    AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)) *
                unit.clone();
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            dst_addr,
            &local.dst_write_access,
            local.is_non_zero_length,
        );
    }

    /// Syscall interaction: communicate with VM's syscall handler
    fn eval_syscall_interaction<AB: SP1AirBuilder>(
        &self,
        local: &TableInitCols<AB::Var>,
        builder: &mut AB,
    ) {
        // Compute syscall ID based on operation type
        let syscall_id = local.is_table_init *
            AB::Expr::from_canonical_u32(SyscallCode::TABLE_INIT.syscall_id() as u32) +
            local.is_table_fill *
                AB::Expr::from_canonical_u32(SyscallCode::TABLE_FILL.syscall_id() as u32) +
            local.is_table_copy *
                AB::Expr::from_canonical_u32(SyscallCode::TABLE_COPY.syscall_id() as u32);

        builder.receive_syscall(
            local.shard,
            local.clk,
            syscall_id,
            AB::Expr::zero(),
            local.aux_value,
            local.is_first,
            InteractionScope::Local,
        );
    }
}

impl<F> BaseAir<F> for TableCopyChip {
    fn width(&self) -> usize {
        NUM_TABLE_COPY_SIZE
    }
}
