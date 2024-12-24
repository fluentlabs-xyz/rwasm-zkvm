#[cfg(test)]
mod tests {

    use super::super::*;

    use hashbrown::HashMap;
    use sp1_rwasm_executor::{Opcode, Program, SP_START};
    use sp1_rwasm_machine::utils::setup_logger;

    use super::super::*;
    use super::*;
    use anyhow::Result;
    use build::try_build_plonk_bn254_artifacts_dev;
    use p3_field::PrimeField32;
    use serde::{Deserialize, Serialize};
    use serial_test::serial;
    use std::fs::File;
    use std::io::{Read, Write};

    use rwasm::engine::bytecode::Instruction;
    fn build_elf() -> Program {
        /*
         let t0 = Register::X5;
               let syscall_id = self.register(t0);
               c = self.rr(Register::X11, MemoryAccessPosition::C);
               b = self.rr(Register::X10, MemoryAccessPosition::B);
               let syscall = SyscallCode::from_u32(syscall_id);
        */

        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x11;
        let y_value: u32 = 0x23;
        let z1_value: u32 = 0x3;
        let z2_value: u32 = 0x37;
        let z3_value: u32 = 0x12;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;

        let mut mem = HashMap::new();
        mem.insert(sp_value, x_value);
        mem.insert(sp_value - 4, y_value);
        mem.insert(sp_value - 8, z1_value);
        mem.insert(sp_value - 12, z2_value);
        mem.insert(sp_value - 16, z3_value);
        mem.insert(sp_value - 20, z4_value);
        mem.insert(sp_value - 24, z5_value);
        mem.insert(sp_value - 28, z6_value);
        println!("{:?}", mem);
        let instructions = vec![
            Instruction::I32Add,
            Instruction::I32Sub,
            Instruction::I32Mul,
            Instruction::I32DivS,
            Instruction::I32DivU,
        ];

        let program = Program {
            instructions,
            pc_base: 1,//If it's a shard with "CPU", then `start_pc` should never equal zero
            pc_start: 1,//If it's a shard with "CPU", then `start_pc` should never equal zero
            memory_image: mem,
            preprocessed_shape: None,
        };
        //  memory_image: BTreeMap::new() };

        program
    }

    #[test]
    fn test_rwasm_proof2() {
        let mut program = build_elf();
        setup_logger();
        let prover: SP1Prover = SP1Prover::new();
        let mut opts = SP1ProverOpts::default();
        opts.core_opts.shard_batch_size = 1;
        let context = SP1Context::default();

        tracing::info!("setup elf");
        let (pk, vk) = prover.setup_program(&mut program);

        tracing::info!("prove core");
        let stdin = SP1Stdin::new();
        let core_proof = prover.prove_core_program(&pk, program, &stdin, opts, context);
        tracing::info!("prove core finish");
        match core_proof {
            Ok(_) => {
                tracing::info!("verify core");
                prover.verify(&core_proof.unwrap().proof, &vk).unwrap();
            }
            Err(err) => {
                println!("{}", err);
            }
        }

        println!("done rwasm proof");
    }
}
