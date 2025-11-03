use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Clz];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case2() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Ctz];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_complex_case() {
    let ops = vec![
        Opcode::I32Const(1.into()),
        Opcode::I32Const(2.into()),
        Opcode::I32Const(3.into()),
        Opcode::I32Const(0x137_137.into()),
        Opcode::I32Clz,
        Opcode::I32Ctz,
        Opcode::I32Eqz,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
