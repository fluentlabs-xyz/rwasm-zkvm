use std::borrow::Borrow;

use crate::{
    air::MemoryAirBuilder,
    memory::ElementAddressCols,
};

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
pub struct TableInitChip {}
impl<AB> Air<AB> for TableInitChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TableInitCols<AB::Var> = (*local).borrow();
        let next: &TableInitCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_non_zero_length);

        builder.when_first_row().assert_one(local.is_first);
        builder.when(local.is_last).assert_one(local.is_real);
        builder.when(local.is_first).assert_one(local.is_real);

        builder
            .when(local.is_first)
            .assert_eq(local.length.value::<AB>(), local.length_access.value().reduce::<AB>());

        // check transition between events
        builder.when_transition().when(local.is_last).when(next.is_real).assert_one(next.is_first);

        // check in event transitions
        builder.when_transition().when_not(local.is_last).assert_eq(local.is_real, next.is_real);
        builder.when_transition().when_not(local.is_last).assert_eq(local.clk, next.clk);
        builder.when_transition().when_not(local.is_last).assert_eq(local.shard, next.shard);
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
            .when(local.is_real)
            .assert_word_eq(*local.src_read_access.value(), *local.dst_write_access.value());

        builder
            .when(local.is_first)
            .when_not(local.is_non_zero_length)
            .assert_word_zero(*local.length_access.value());

        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.src_access.value().reduce::<AB>() + local.length_access.value().reduce::<AB>() -
                AB::Expr::one(),
            local.src_address.value::<AB>(),
        );
        builder.when(local.is_last).when(local.is_non_zero_length).assert_eq(
            local.dst_access.value().reduce::<AB>() + local.length_access.value().reduce::<AB>() -
                AB::Expr::one(),
            local.dst_address.value::<AB>(),
        );
        builder.when_transition().when(local.is_real).when_not(local.is_last).assert_eq(
            local.src_address.value::<AB>() + AB::Expr::one(),
            next.src_address.value::<AB>(),
        );
        builder.when_transition().when(local.is_real).when_not(local.is_last).assert_eq(
            local.dst_address.value::<AB>() + AB::Expr::one(),
            next.dst_address.value::<AB>(),
        );

        // check that it does not go out of memory bounds
        builder
            .when(local.is_first)
            .assert_eq(local.src_access.value().reduce::<AB>(), local.src_address.value::<AB>());
        builder
            .when(local.is_first)
            .assert_eq(local.dst_access.value().reduce::<AB>(), local.dst_address.value::<AB>());

        StackAddressCols::<AB::Var>::range_check(builder, local.sp);
        builder.when(local.is_first).assert_one(local.sp.is_real::<AB>());

        ElementAddressCols::<AB::Var>::range_check(builder, local.src_address);
        builder.when(local.is_first).assert_one(local.src_address.is_real::<AB>());
        builder.when(local.is_last).assert_one(local.src_address.is_real::<AB>());

        let table_size_value = local.table_size_read_access.value();

        DynamicTableAddressCols::<AB::Var>::range_check(
            builder,
            local.dst_address,
            table_size_value[0].into(),
            table_size_value[1].into(),
        );
        builder.when(local.is_first).assert_one(local.dst_address.is_real::<AB>());
        builder.when(local.is_last).assert_one(local.dst_address.is_real::<AB>());

        TableIdxCols::<AB::Var>::range_check(builder, local.table_idx);
        builder.when(local.is_first).assert_one(local.table_idx.is_real::<AB>());

        LengthCols::<AB::Var>::range_check(builder, local.length);
        builder.when(local.is_first).assert_one(local.length.is_real::<AB>());

        self.eval_memory_access(local, builder);

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

impl TableInitChip {
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
            local.is_non_zero_length,
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

impl<F> BaseAir<F> for TableInitChip {
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
                    let event = if let PrecompileEvent::TableInit(event) = event {
                        event
                    } else {
                        unreachable!()
                    };

                    event.d = event.d + 1;
                });
            }

            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));

        assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
    }
}
