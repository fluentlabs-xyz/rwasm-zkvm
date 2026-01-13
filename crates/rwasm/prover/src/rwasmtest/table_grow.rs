use crate::rwasmtest::run_rwasm_prover;
use rwasm::{Opcode, N_MAX_TABLE_SIZE};
use rwasm_executor::Program;

#[test]
pub fn test_base_case() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(2.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(8.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(8.into()),
        Opcode::TableGrow(1),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::TableGrow(1),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_edge_delta() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(0.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(0.into()),
        Opcode::I32Const((N_MAX_TABLE_SIZE - 1).into()),
        Opcode::TableGrow(0),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_one_delta() {
    let ops = vec![Opcode::I32Const(0.into()), Opcode::I32Const(1.into()), Opcode::TableGrow(0)];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_overflow_delta() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const((1).into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(N_MAX_TABLE_SIZE.into()),
        Opcode::TableGrow(0),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
