use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

fn build_rwasm_call_indirect() -> Program {
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(2.into()),
        Opcode::TableGrow(1),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(0.into()),
        Opcode::I32Const(2.into()),
        Opcode::TableInit(0),
        Opcode::TableGet(1),
        Opcode::I32Const(1.into()),
        Opcode::CallIndirect(0u32),
        Opcode::TableGet(1),
        Opcode::Return,
        Opcode::I32Const(99.into()),
        Opcode::I32Const(98.into()),
        Opcode::I32Add,
        Opcode::Return,
    ];
    let elements = vec![12u32, 12u32];
    let program = Program::from_instrs(ops).with_elements(elements);
    program
}

#[test]
fn test_rwasm_call_indirect() {
    let program = build_rwasm_call_indirect();
    run_rwasm_prover(program);
}
