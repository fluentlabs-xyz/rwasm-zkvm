use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};
#[test]
fn test_consume_fuel_non_stack() {
    let ops = vec![Opcode::ConsumeFuel(5000)];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
#[test]
fn test_consume_fuel_stack() {
    let ops = vec![Opcode::I32Const(5000.into()), Opcode::ConsumeFuelStack];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
