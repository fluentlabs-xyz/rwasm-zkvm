use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_extend8s() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Extend8S];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_extend16s() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Extend16S];
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
        Opcode::I32Extend8S,
        Opcode::I32Extend16S,
        Opcode::I32Eqz,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
