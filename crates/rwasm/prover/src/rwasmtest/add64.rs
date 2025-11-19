use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

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
pub fn test_base_case_for_add32() {
    let ops =
        vec![Opcode::I32Const((-151515).into()), Opcode::I32Const((-8).into()), Opcode::I32Add];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
