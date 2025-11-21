use crate::rwasmtest::run_rwasm_prover;
use rwasm::Opcode;
use rwasm_executor::Program;

#[test]
pub fn test_base_case() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(100.into()),
        Opcode::TableGrow(2),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(2.into()),
        Opcode::I32Const(100.into()),
        Opcode::TableFill(2),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(100.into()),
        Opcode::TableGrow(1),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(80.into()),
        Opcode::TableCopy(2, 1),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_same_table() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(100.into()),
        Opcode::TableGrow(2),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(80.into()),
        Opcode::TableCopy(2, 2),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_zero_length() {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(100.into()),
        Opcode::TableGrow(1),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(64.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(1.into()),
        Opcode::I32Const(0.into()),
        Opcode::TableCopy(0, 1),
        Opcode::TableGet(0),
    ];
    let elements = vec![5u32, 7u32, 9u32, 12u32];
    let program = Program::from_instrs(ops).with_elements(elements);
    run_rwasm_prover(program);
}
