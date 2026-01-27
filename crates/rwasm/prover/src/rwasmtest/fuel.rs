use crate::rwasmtest::{run_rwasm_prover, run_rwasm_prover_with_fuel_limit};
use rwasm_executor::{Opcode, Program, SP1Context};
#[test]
fn test_consume_fuel_non_stack() {
    let ops = vec![
        Opcode::I32Const(5000.into()),
        Opcode::ConsumeFuel(5000),
        Opcode::ConsumeFuel(5000),
        Opcode::ConsumeFuel(5000),
    ];
    let program = Program::from_instrs(ops);

    run_rwasm_prover_with_fuel_limit(program, u32::MAX as u64);
}

#[test]
fn test_consume_fuel_stack() {
    let ops = vec![Opcode::I32Const(5000.into()), Opcode::ConsumeFuelStack];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
#[should_panic]
fn test_consume_fuel_fail() {
    let ops = vec![
        Opcode::I32Const(5000.into()),
        Opcode::ConsumeFuel(5000),
        Opcode::ConsumeFuel(5000),
        Opcode::ConsumeFuel(5000),
    ];
    let program = Program::from_instrs(ops);

    run_rwasm_prover_with_fuel_limit(program, 5000);
}
