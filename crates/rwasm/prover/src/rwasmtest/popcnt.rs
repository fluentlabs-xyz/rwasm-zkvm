use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case() {
    let ops = vec![Opcode::I32Const(0x137_137.into()), Opcode::I32Popcnt];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
