#[cfg(test)]
pub mod test {

    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use rwasm_executor::{ExecutionRecord, Executor, Opcode, Program};
    use rwasm_machine::{
        cpu::CpuChip,
        utils::{uni_stark_prove, uni_stark_verify},
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, SP1CoreOpts, StarkGenericConfig,
    };

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

        //  memory_image: BTreeMap::new() };

        Program::from_instrs(instructions)
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let opts = SP1CoreOpts::default();

        let program = build_elf();
        let mut runtime = Executor::new(program, opts);
        runtime.run().unwrap();
        println!("runtimerecordcpu:{:?}", runtime.record.cpu_events);
        let chip = CpuChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&(runtime.record).defer(), &mut ExecutionRecord::default());

        let proof = uni_stark_prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
        println!("{:?}", ());
    }
}
