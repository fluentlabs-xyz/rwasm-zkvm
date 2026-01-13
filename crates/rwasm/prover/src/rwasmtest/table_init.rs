// TODO: add malicious tests

use crate::rwasmtest::run_rwasm_prover;
use rwasm::N_MAX_TABLE_SIZE;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(64.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(2.into()),
        Opcode::TableInit(0),
        Opcode::TableGet(0),
        Opcode::I32Const(137.into()),
    ];
    let elements = vec![111u32, 111u32, 111u32, 111u32];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}

#[test]
pub fn test_zero_length() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(64.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(0.into()),
        Opcode::TableInit(0),
        Opcode::TableGet(0),
    ];
    let elements = vec![5u32, 7u32, 9u32, 12u32];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}

#[test]
#[should_panic]
pub fn test_table_idx_too_large() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(64.into()),
        Opcode::TableGrow(100),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(0.into()),
        Opcode::TableInit(100),
    ];
    let elements = vec![5u32, 7u32, 9u32, 12u32];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}

#[test]
pub fn test_table_memory_bound() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(N_MAX_TABLE_SIZE.into()),
        Opcode::TableGrow(99),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(1.into()),
        Opcode::TableInit(0),
        Opcode::TableGet(99),
    ];
    let elements = vec![137u32; N_MAX_TABLE_SIZE as usize];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}

#[test]
#[should_panic]
pub fn test_table_memory_out_of_bound() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(N_MAX_TABLE_SIZE.into()),
        Opcode::TableGrow(99),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(N_MAX_TABLE_SIZE.into()),
        Opcode::TableInit(0),
        Opcode::TableGet(99),
    ];
    let elements = vec![137u32; N_MAX_TABLE_SIZE as usize];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}
