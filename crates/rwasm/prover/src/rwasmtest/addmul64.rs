use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_mul64() {
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Mul64,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_add64() {
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Add64,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_mixed_add_mul64_program() {
    let ops = vec![
        // First 64-bit multiplication: 3 * 5
        Opcode::I32Const(3u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Mul64,
        // Second 64-bit multiplication: 7 * 11
        Opcode::I32Const(7u32.into()),
        Opcode::I32Const(11u32.into()),
        Opcode::I32Mul64,
        // First 64-bit addition: 0x137_137 + 0x42
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x42u32.into()),
        Opcode::I32Add64,
        // Second 64-bit addition on the resulting top two 32-bit words
        Opcode::I32Add64,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_mul32() {
    let ops =
        vec![Opcode::I32Const((-151515).into()), Opcode::I32Const((-8).into()), Opcode::I32Mul];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
