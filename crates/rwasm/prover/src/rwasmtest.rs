use hashbrown::HashMap;

use rwasm_executor::{Opcode, Program, SP_START};
use rwasm_machine::utils::setup_logger;

use super::*;
use anyhow::Result;
use build::try_build_plonk_bn254_artifacts_dev;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serial_test::serial;
use std::fs::File;
use std::io::{Read, Write};

pub fn run_rwasm_prover(mut program: Program) {
    setup_logger();
    let prover: SP1Prover = SP1Prover::new();
    let mut opts = SP1ProverOpts::default();
    opts.core_opts.shard_batch_size = 1;
    let context = SP1Context::default();

    tracing::info!("setup elf");
    let (_, pk, vk) = prover.setup_program(&mut program);

    tracing::info!("prove core");
    let stdin = SP1Stdin::new();
    let core_proof = prover.prove_core(&pk, program.clone(), &stdin, opts, context);
    tracing::info!("prove core finish");
    match core_proof {
        Ok(_) => {
            tracing::info!("verify core");
            let result = prover.verify(&core_proof.unwrap().proof, &vk);
            match result {
                Ok(_) => (),
                Err(err) => {
                    println!("err:{}", err);
                    panic!();
                }
            }
        }
        Err(err) => {
            println!("{}", err);
        }
    }

    println!("done rwasm proof");
}
#[cfg(test)]
mod tests {

    use super::super::*;

    use hashbrown::HashMap;

    use rwasm::BranchOffset;
    use rwasm_executor::{Opcode, Program, SP_START};
    use rwasm_machine::program;
    use rwasm_machine::utils::setup_logger;

    use super::super::*;
    use super::*;
    use anyhow::Result;
    use build::try_build_plonk_bn254_artifacts_dev;
    use p3_field::PrimeField32;
    use serde::{Deserialize, Serialize};
    use serial_test::serial;
    use std::fs::File;
    use std::io::{Read, Write};

    fn build_elf() -> Program {
        let x_value: u32 = 0x11;
        let y_value: u32 = 0x23;
        let z1_value: u32 = 0x40;
        let z2_value: u32 = 0x37;
        let z3_value: u32 = 0x1800;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;

        let instructions = vec![
            Opcode::I32Const(z6_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Add,
            Opcode::I32Sub,
            Opcode::I32Mul,
            Opcode::I32DivS,
            Opcode::I32DivU,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_elf2() -> Program {
        let x_value: u32 = 0x11;
        let y_value: u32 = 0x23;
        let z1_value: u32 = 0x3;
        let z2_value: u32 = 0x37;
        let z3_value: u32 = 0x12;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;
        let z7_value: u32 = 0x333333;
        let z8_value: u32 = 0x444444;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32Const(z6_value.into()),
            Opcode::I32Const(z7_value.into()),
            Opcode::I32Const(z8_value.into()),
            Opcode::I32Ne,
            Opcode::I32Eq,
            Opcode::I32GtS,
            Opcode::I32GtU,
            Opcode::I32LeS,
            Opcode::I32LeU,
            Opcode::I32GeS,
            Opcode::I32GeU,
            Opcode::I32LtS,
            Opcode::I32Eqz,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_elf3() -> Program {
        let x_value: u32 = 0x1;
        let y_value: u32 = 0x2;
        let z1_value: u32 = 0x1;
        let z2_value: u32 = 0x2;
        let z3_value: u32 = 0x1;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x1;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32And,
            Opcode::I32Or,
            Opcode::I32Xor,
            Opcode::I32Shl,
            Opcode::I32ShrS,
            Opcode::I32ShrU,
        ];
        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf4() -> Program {
        let addr: u32 = 0x40000;
        let addr_2: u32 = 0x40004;
        let addr_3: u32 = 0x40008;

        let x_value: u32 = 0x10007;
        let x_2_value: u32 = 0x10008;

        let x_3_value: u32 = 0x200AA;
        // let mut mem = HashMap::new();
        // mem.insert(sp_value, addr);
        // mem.insert(sp_value - 4, addr_2);
        // mem.insert(sp_value - 8, 0x10000);
        // mem.insert(sp_value - 12, addr_3);
        // mem.insert(sp_value - 16, 0x10000);
        // mem.insert(addr, x_value);
        // mem.insert(addr_2, x_2_value);
        // mem.insert(addr_3, x_3_value);

        //  println!("{:?}", mem);
        let instructions = vec![
            Opcode::I32Const(0x10.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(x_3_value.into()),
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0),
            Opcode::I32Store16(0),
            Opcode::I32Store8(0),
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(0),
            // Opcode::I32Load16U(0),
            // Opcode::I32Load8U(0),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_load4() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 65551i32 as u32;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16S(0u32),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };
        program
    }
    fn build_elf_load5() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_0005;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store16(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16U(0u32),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };
        program
    }
    fn build_elf5() -> Program {
        let addr: u32 = 0x40000;
        let addr_2: u32 = 0x40004;
        let addr_3: u32 = 0x40008;

        let x_value: u32 = 0x10007;
        let x_2_value: u32 = 0x10008;

        let x_3_value: u32 = 0x200AA;

        let instructions = vec![
            Opcode::I32Const(0x10.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(x_3_value.into()),
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0),
            Opcode::I32Store16(0),
            Opcode::I32Store8(0),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_br() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::Br(4.into()),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_branching() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x0;
        let x_2_value: u32 = 0x10008;
        let x_3_value: u32 = 0x1000C;

        // let mut mem = HashMap::new();
        // mem.insert(sp_value, x_value);
        // mem.insert(sp_value - 4, x_2_value);
        // mem.insert(sp_value - 8, x_3_value);

        //  println!("{:?}", mem);
        let instructions = vec![
            Opcode::Br(20.into()),
            Opcode::I32Add,
            Opcode::I32Add,
            Opcode::I32Add,
            Opcode::BrIfNez(12.into()),
            Opcode::I32Add,
            Opcode::BrIfNez(BranchOffset::from(-8i32)),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_elf_local_const() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1234;
        let x_2_value: u32 = x_value + 5;
        let depth = 5;
        let under_depth = depth - 2;
        let constant = x_value;

        let instructions = vec![
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const((x_value + 123).into()),
            Opcode::I32Const((x_value + 456).into()),
            Opcode::I32Const((x_value + 789).into()),
            Opcode::I32Const((x_value).into()),
            Opcode::LocalGet(depth),
            Opcode::LocalSet(under_depth),
            Opcode::LocalTee(under_depth),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_const_another() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;
        let y_value: u32 = 0x54321;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_store_unaligned() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x0103_0507;

        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(3u32),
        ];
        let program = Program::from_instrs(opcodes);
        program
    }

    fn build_load_unaligned() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1103_0507;

        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(3u32),
        ];
        let program = Program::from_instrs(opcodes);
        program
    }

    // fn build_elf_call() -> Program {
    //     let sp_value: u32 = SP_START;
    //     let x_value: u32 = 0x3;
    //     let y_value: u32 = 0x5;

    //     let mut mem = HashMap::new();
    //     mem.insert(sp_value, x_value);
    //     mem.insert(sp_value - 4, y_value);
    //     let mut functions = vec![12 + 1];

    //     let instructions = vec![
    //         Opcode::CallInternal(1u32.into()),
    //         Opcode::Return(DropKeep::none()),
    //         Opcode::I32Add,
    //         Opcode::Return(DropKeep::none()),
    //     ];

    //     let program = Program::new_with_memory_and_func(instructions, mem, functions, 1, 1);
    //     program
    // }
    // fn build_elf_call2() -> Program {
    //     let sp_value: u32 = SP_START;
    //     let x_value: u32 = 0x3;
    //     let y_value: u32 = 0x5;
    //     let z_value: u32 = 0x7;
    //     let mut mem = HashMap::new();

    //     let mut functions = vec![21];

    //     let instructions = vec![
    //         Opcode::I32Const(x_value.into()),
    //         Opcode::I32Const(y_value.into()),
    //         Opcode::I32Const(z_value.into()),
    //         Opcode::CallInternal(0u32.into()),
    //         Opcode::Return(DropKeep::none()),
    //         Opcode::I32Add,
    //         Opcode::I32Add,
    //         Opcode::Return(DropKeep::none()),
    //     ];

    //     let program = Program::new_with_memory_and_func(instructions, mem, functions.clone(), 1, 1);
    //     for (ins_idx, item) in program.instructions.iter().enumerate() {
    //         println!("ins_idx:{},item:{:?},", ins_idx * 4 + 1, item);
    //     }
    //     println!("functions: {:?}", functions);
    //     program
    // }

    // fn build_elf_call3() -> Program {
    //     let sp_value: u32 = SP_START;
    //     let x_value: u32 = 0x3;
    //     let y_value: u32 = 0x5;
    //     let z_value: u32 = 0x7;
    //     let mut mem = HashMap::new();

    //     let mut functions = vec![13, 17, 25];

    //     let instructions = vec![
    //         Opcode::I32Const(x_value.into()),
    //         Opcode::CallInternal(1u32.into()),
    //         Opcode::Return(DropKeep::none()),
    //         Opcode::Return(DropKeep::none()),
    //         Opcode::CallInternal(0u32.into()),
    //         Opcode::Return(DropKeep::none()),
    //     ];

    //     let program = Program::new_with_memory_and_func(instructions, mem, functions.clone(), 1, 1);
    //     for (ins_idx, item) in program.instructions.iter().enumerate() {
    //         println!("ins_idx:{},item:{:?},", ins_idx * 4 + 1, item);
    //     }
    //     println!("functions: {:?}", functions);
    //     program
    // }

    fn build_elf_skipped_ins() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1234;
        let x_2_value: u32 = x_value + 5;
        let depth = 5 * 4;
        let under_depth = depth - 4;
        let constant = x_value;
        // let mut mem = HashMap::new();
        // mem.insert(sp_value, x_value);
        // mem.insert(sp_value - depth, x_2_value);

        //  println!("{:?}", mem);
        let instructions = vec![Opcode::ConsumeFuel(1), Opcode::SignatureCheck(1), Opcode::Drop];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    #[test]
    fn test_rwasm_proof1() {
        let program = build_elf();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof2() {
        let program = build_elf2();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof3() {
        let program = build_elf3();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof4() {
        let program = build_elf4();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof_load4() {
        let program = build_elf_load4();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof_load5() {
        let program = build_elf_load5();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof5() {
        let program = build_elf5();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_br() {
        let program = build_elf_br();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_branching() {
        let program = build_elf_branching();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_local() {
        let program = build_elf_local_const();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_another1() {
        let program = build_elf_const_another();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_unaligned_store() {
        let program = build_store_unaligned();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_unaligned_load() {
        let program = build_load_unaligned();
        run_rwasm_prover(program);
    }
    // #[test]
    // fn test_rwasm_call_internal_and_return() {
    //     let program = build_elf_call();
    //     run_rwasm_prover(program);
    // }
    // #[test]
    // fn test_rwasm_call_internal_and_return2() {
    //     let program = build_elf_call2();
    //     run_rwasm_prover(program);
    // }

    // #[test]
    // fn test_rwasm_call_internal_and_return3() {
    //     let program = build_elf_call3();
    //     run_rwasm_prover(program);
    // }

    #[test]
    fn test_rwasm_skipped() {
        let program = build_elf_skipped_ins();
        run_rwasm_prover(program);
    }
}
