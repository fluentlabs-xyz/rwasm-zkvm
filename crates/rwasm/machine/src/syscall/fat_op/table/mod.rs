use std::borrow::Borrow;

use crate::{air::MemoryAirBuilder, memory::ElementAddressCols};

use p3_air::{Air, AirBuilder, BaseAir};

use crate::air::WordAirBuilder;
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
    N_MAX_TABLE_SIZE,
};

#[derive(Default)]
pub struct TableInitFillChip {}
impl<AB> Air<AB> for TableInitFillChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableInitCols<AB::Var> = (*local).borrow();
        let next: &TableInitCols<AB::Var> = (*next).borrow();

        let is_real = local.is_table_init + local.is_table_fill;
        let next_is_real = next.is_table_init + next.is_table_fill;

        self.eval_boolean_constraints(builder, local, is_real.clone());

        // First Row Constraints
        // The first row in the trace must be marked as first
        builder.when_first_row().assert_one(local.is_first);

        self.eval_real_row_constraints(builder, local, is_real.clone());
        self.eval_operation_type_constraints(builder, local);
        self.eval_length_validation_constraints(builder, local);

        // Event Transition Constraints
        // Between events: if last row and next is real, then next must be first
        builder
            .when_transition()
            .when(local.is_last)
            .when(next_is_real.clone())
            .assert_one(next.is_first);

        self.eval_within_event_transition_constraints(
            builder,
            local,
            next,
            is_real.clone(),
            next_is_real,
        );
        self.eval_value_propagation_constraints(builder, local);
        self.eval_address_boundary_constraints(builder, local);
        self.eval_address_increment_constraints(builder, local, next, is_real.clone());
        self.eval_initial_address_constraints(builder, local);
        self.eval_range_check_constraints(builder, local);
        self.eval_memory_access(local, builder);
        self.eval_syscall_interactions(builder, local);
    }
}

impl TableInitFillChip {
    // Boolean Constraints
    fn eval_boolean_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Ensure all flag fields are boolean (0 or 1)
        builder.assert_bool(local.is_table_init);
        builder.assert_bool(local.is_table_fill);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(is_real);
        builder.assert_bool(local.is_non_zero_length);
        builder.assert_bool(local.is_first_table_fill + local.is_first_table_init);
    }

    // Real Row Constraints
    fn eval_real_row_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Last and first rows must be real (either table_init or table_fill)
        builder.when(local.is_last).assert_one(is_real.clone());
        builder.when(local.is_first).assert_one(is_real);
    }

    // Operation Type Constraints
    fn eval_operation_type_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // For table_fill operations: first row must be marked as is_first_table_fill
        builder
            .when(local.is_table_fill)
            .when(local.is_first)
            .assert_one(local.is_first_table_fill);

        // For table_init operations: first row must be marked as is_first_table_init
        builder
            .when(local.is_table_init)
            .when(local.is_first)
            .assert_one(local.is_first_table_init);
    }

    // Length Validation Constraints
    fn eval_length_validation_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // On first row: length field must match length from stack access
        builder
            .when(local.is_first)
            .assert_eq(local.length.value::<AB>(), local.length_access.value().reduce::<AB>());

        // For table_fill: should_read_elements must be zero
        builder.when(local.is_table_fill).assert_zero(local.should_read_elements);

        // For table_init with non-zero length: should_read_elements must be one
        builder
            .when(local.is_table_init)
            .when(local.is_non_zero_length)
            .assert_one(local.should_read_elements);

        // For first row with zero length: length_access must be zero
        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length)
            .assert_word_zero(*local.length_access.value());
    }

    // Within-Event Transition Constraints
    fn eval_within_event_transition_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
        next_is_real: AB::Expr,
    ) {
        // When not last row (within same event), the following must remain constant
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
        builder.when_transition().when_not(local.is_last).assert_word_eq(
            *local.table_size_read_access.value(),
            *next.table_size_read_access.value(),
        );
        builder
            .when_transition()
            .when_not(local.is_last)
            .assert_eq(local.table_idx.value::<AB>(), next.table_idx.value::<AB>());
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

    // Value Propagation Constraints
    fn eval_value_propagation_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // For table_init: value read from source must equal value written to destination
        builder
            .when(local.is_table_init)
            .assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());

        // For table_fill with non-zero length
        builder
            .when(local.is_table_fill)
            .when(local.is_non_zero_length)
            .assert_word_eq(*local.src_access.value(), *local.dst_write_access.value());
    }

    // Address Boundary Constraints
    fn eval_address_boundary_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // On last row with elements to read: src_address must equal src_access + length - 1
        builder.when(local.is_last).when(local.should_read_elements).assert_eq(
            local.src_access.value().reduce::<AB>() + local.length_access.value().reduce::<AB>() -
                AB::Expr::one(),
            local.src_address.value::<AB>(),
        );

        // On last row with non-zero length: dst_address must equal dst_access + length - 1
        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.dst_access.value().reduce::<AB>() + local.length_access.value().reduce::<AB>() -
                AB::Expr::one(),
            local.dst_address.value::<AB>(),
        );
    }

    // Address Increment Constraints
    fn eval_address_increment_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
        next: &TableInitCols<AB::Var>,
        is_real: AB::Expr,
    ) {
        // For table_init within event: src_address increments by 1
        builder.when_transition().when(local.is_table_init).when_not(local.is_last).assert_eq(
            local.src_address.value::<AB>() + AB::Expr::one(),
            next.src_address.value::<AB>(),
        );

        // For any real operation within event: dst_address increments by 1
        builder.when_transition().when(is_real).when_not(local.is_last).assert_eq(
            local.dst_address.value::<AB>() + AB::Expr::one(),
            next.dst_address.value::<AB>(),
        );
    }

    // Initial Address Constraints
    fn eval_initial_address_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // For table_init first row: src_address must equal src_access
        builder
            .when(local.is_first)
            .when(local.is_table_init)
            .assert_eq(local.src_access.value().reduce::<AB>(), local.src_address.value::<AB>());

        // For first row: dst_address must equal dst_access
        builder
            .when(local.is_first)
            .assert_eq(local.dst_access.value().reduce::<AB>(), local.dst_address.value::<AB>());
    }

    // Range Check Constraints
    fn eval_range_check_constraints<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Stack pointer range check
        StackAddressCols::<AB::Var>::range_check(builder, local.sp);
        builder.when(local.is_first).assert_one(local.sp.do_check::<AB>());

        // Element address range check
        ElementAddressCols::<AB::Var>::range_check(builder, local.src_address);
        builder
            .when(local.is_first)
            .when(local.is_table_init)
            .assert_one(local.src_address.do_check::<AB>());
        builder
            .when(local.is_last)
            .when(local.is_table_init)
            .assert_one(local.src_address.do_check::<AB>());

        // Dynamic table address range check
        let table_size_value = local.table_size_read_access.value();
        DynamicTableAddressCols::<AB::Var>::range_check(
            builder,
            local.dst_address,
            table_size_value[0].into(),
            table_size_value[1].into(),
        );
        builder.when(local.is_first).assert_one(local.dst_address.do_check::<AB>());
        builder.when(local.is_last).assert_one(local.dst_address.do_check::<AB>());

        // Table index range check
        TableIdxCols::<AB::Var>::range_check(builder, local.table_idx);
        builder.when(local.is_first).assert_one(local.table_idx.do_check::<AB>());

        // Length range check
        LengthCols::<AB::Var>::range_check(builder, local.length);
        builder.when(local.is_first).assert_one(local.length.do_check::<AB>());
    }

    // Syscall Interactions
    fn eval_syscall_interactions<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &TableInitCols<AB::Var>,
    ) {
        // Receive TABLE_INIT syscall
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(SyscallCode::TABLE_INIT.syscall_id() as u32),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_first_table_init,
            InteractionScope::Local,
        );

        // Receive TABLE_FILL syscall
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(SyscallCode::TABLE_FILL.syscall_id() as u32),
            AB::Expr::zero(),
            local.table_idx.value::<AB>(),
            local.is_first_table_fill,
            InteractionScope::Local,
        );
    }

    // Memory Access Constraints
    fn eval_memory_access<AB: SP1AirBuilder>(
        &self,
        local: &TableInitCols<AB::Var>,
        builder: &mut AB,
    ) {
        let unit = AB::Expr::from_canonical_u32(UNIT);

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

        let src_addr = AB::Expr::from_canonical_u32(TypedAddress::Element(0).to_virtual_addr()) +
            local.src_address.value::<AB>() * unit.clone();

        let table_addr = AB::Expr::from_canonical_u32(TypedAddress::Table(0).to_virtual_addr()) +
            (local.dst_address.value::<AB>() +
                local.table_idx.value::<AB>() * AB::Expr::from_canonical_u32(N_MAX_TABLE_SIZE)) *
                unit.clone();

        builder.eval_memory_access(
            local.shard,
            local.clk,
            src_addr,
            &local.src_read_access,
            local.should_read_elements,
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            AB::Expr::from_canonical_u32(TypedAddress::TableSize(0).to_virtual_addr()) +
                local.table_idx.value::<AB>() * unit,
            &local.table_size_read_access,
            local.is_first,
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

impl<F> BaseAir<F> for TableInitFillChip {
    fn width(&self) -> usize {
        NUM_TABLE_INIT_SIZE
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::print_stdout)]

    use crate::{io::SP1Stdin, rwasm::RwasmAir, utils::run_malicious_test};
    use p3_baby_bear::BabyBear;
    use rwasm_executor::{events::PrecompileEvent, ExecutionRecord, Opcode, Program};
    use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, CpuProver, MachineProver};

    use rwasm_executor::syscalls::SyscallCode;

    #[test]
    fn test_malicious_table_init() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let elements = vec![111u32, 111u32, 111u32, 111u32];

        let program = Program::from_instrs(vec![
            Opcode::I32Const(0.into()),
            Opcode::I32Const(64.into()),
            Opcode::TableGrow(0),
            Opcode::I32Const(62.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableInit(0),
            Opcode::TableGet(0),
            Opcode::I32Const(137.into()),
        ])
        .with_elements(elements);

        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();

            if let Some(event) =
                malicious_record.precompile_events.events.get_mut(&SyscallCode::TABLE_INIT)
            {
                event.first_mut().map(|(_, event)| {
                    let event = if let PrecompileEvent::TableInitFill(event) = event {
                        event
                    } else {
                        unreachable!()
                    };

                    event.d = event.d + 1;

                    event.stack_access[0].value = event.d;
                });
            }

            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));

        assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
    }
}
