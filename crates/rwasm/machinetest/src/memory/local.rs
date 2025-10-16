#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rwasm_executor::{ExecutionRecord, Executor};
    use rwasm_machine::{
        memory::MemoryLocalChip, programs::tests::build_elf, rwasm::RwasmAir, utils::setup_logger,
    };
    use sp1_stark::{
        air::{InteractionScope, MachineAir},
        baby_bear_poseidon2::BabyBearPoseidon2,
        debug_interactions_with_all_chips, InteractionKind, SP1CoreOpts, StarkMachine,
    };

    #[test]
    fn test_local_memory_generate_trace() {
        let program = build_elf();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip: MemoryLocalChip = MemoryLocalChip::new();

        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{:?}", mem_event);
        }
    }

    #[test]
    fn test_memory_lookup_interactions() {
        setup_logger();
        let program = build_elf();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let machine: StarkMachine<BabyBearPoseidon2, RwasmAir<BabyBear>> =
            RwasmAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = SP1CoreOpts::default();
        machine.generate_dependencies(
            &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
            &opts,
            None,
        );

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_interactions_with_all_chips::<BabyBearPoseidon2, RwasmAir<BabyBear>>(
                &machine,
                &pkey,
                &[*shard],
                vec![InteractionKind::Memory],
                InteractionScope::Local,
            );
        }
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RwasmAir<BabyBear>>(
            &machine,
            &pkey,
            &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
            vec![InteractionKind::Memory],
            InteractionScope::Global,
        );
    }

    #[test]
    fn test_byte_lookup_interactions() {
        setup_logger();
        let program = build_elf();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let machine = RwasmAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = SP1CoreOpts::default();
        machine.generate_dependencies(
            &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
            &opts,
            None,
        );

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_interactions_with_all_chips::<BabyBearPoseidon2, RwasmAir<BabyBear>>(
                &machine,
                &pkey,
                &[*shard],
                vec![InteractionKind::Memory],
                InteractionScope::Local,
            );
        }
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RwasmAir<BabyBear>>(
            &machine,
            &pkey,
            &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
            vec![InteractionKind::Byte],
            InteractionScope::Global,
        );
    }
}
