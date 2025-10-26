// In crates/rwasm/prover/src/rwasmtest/rotate.rs

use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
fn test_i32_rotl() {
    let program = Program::from_instrs(vec![
        Opcode::I32Const(2u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32Shl,
        Opcode::I32ShrU,
        // The actual test program for rotation.
        Opcode::I32Const(0x12345678u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Rotl,
    ]);
    run_rwasm_prover(program);
}

#[test]
fn test_i32_rotr() {
    let program = Program::from_instrs(vec![
        Opcode::I32Const(2u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32Shl,
        Opcode::I32ShrU,
        // The actual test program for rotation.
        Opcode::I32Const(0x12345678u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Rotr,
    ]);
    run_rwasm_prover(program);
}
