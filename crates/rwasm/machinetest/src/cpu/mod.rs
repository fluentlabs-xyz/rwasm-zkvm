#[cfg(test)]
pub mod test {

    #![allow(clippy::print_stdout)]

    use hashbrown::HashMap;
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};
    use rwasm_executor::{
        events::AluEvent, ExecutionRecord, Executor, Opcode, Program, DEFAULT_PC_INC,
    };
    use rwasm_machine::{
        cpu::CpuChip,
        programs,
        rwasm::AddSubChip,
        utils::{uni_stark_prove, uni_stark_verify},
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, MachineProver, SP1CoreOpts,
        StarkGenericConfig,
    };
    use std::sync::LazyLock;

    fn build_elf() -> Program {
        // let x_value: u32 = 0x11;
        // let y_value: u32 = 0x23;
        // let z1_value: u32 = 0x40;
        // let z2_value: u32 = 0x37;
        // let z3_value: u32 = 0x1800;
        // let z4_value: u32 = 0x2;
        // let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;

        let instructions = vec![
            Opcode::I32Const(z6_value.into()),
            // Instruction::I32Const(z5_value.into()),
            // Instruction::I32Const(z4_value.into()),
            // Instruction::I32Const(z3_value.into()),
            // Instruction::I32Const(z2_value.into()),
            // Instruction::I32Const(z1_value.into()),
            // Instruction::I32Const(y_value.into()),
            // Instruction::I32Const(x_value.into()),
            // Instruction::I32Add,
            // Instruction::I32Sub,
            // Instruction::I32Mul,
            // Instruction::I32DivS,
            // Instruction::I32DivU,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let opts = SP1CoreOpts::default();

        let program = build_elf();
        let mut runtime = Executor::new(program, opts);
        runtime.run();
        println!("runtimerecordcpu:{:?}", runtime.record.cpu_events);
        let chip = CpuChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&(runtime.record).defer(), &mut ExecutionRecord::default());

        let proof = uni_stark_prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        let result = uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
        println!("{:?}", result);
    }
}
