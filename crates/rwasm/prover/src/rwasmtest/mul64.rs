use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_mul64() {
    let ops = vec![
        Opcode::I32Const(0x137_137.into()),
        Opcode::I32Const(0x137_137.into()),
        Opcode::I32Mul64,
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
