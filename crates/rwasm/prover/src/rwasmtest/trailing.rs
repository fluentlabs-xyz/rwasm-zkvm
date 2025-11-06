use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_popcnt() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Popcnt];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_complex_casefor_popcnt() {
    let ops = vec![
        Opcode::I32Const(1.into()),
        Opcode::I32Const(2.into()),
        Opcode::I32Const(3.into()),
        Opcode::I32Const(0x137_137.into()),
        Opcode::I32Popcnt,
        Opcode::I32Add,
        Opcode::I32Eqz,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_clz() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Clz];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_ctz() {
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
