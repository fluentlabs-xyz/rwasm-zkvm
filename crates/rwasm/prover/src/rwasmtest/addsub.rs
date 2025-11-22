use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_add() {
    // 0x137137 + 0x137137
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Add,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_sub() {
    // 0x137137 - 0x1
    let ops =
        vec![Opcode::I32Const(0x137_137u32.into()), Opcode::I32Const(1u32.into()), Opcode::I32Sub];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_mixed_add_sub_program() {
    // Logic: (3 + 5) - (2 + 1) = 8 - 3 = 5
    let ops = vec![
        // 1. Calculate 3 + 5
        Opcode::I32Const(3u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Add, // Stack: [8]
        // 2. Calculate 2 + 1
        Opcode::I32Const(2u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32Add, // Stack: [8, 3]
        // 3. Subtract the results: 8 - 3
        Opcode::I32Sub, // Stack: [5]
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
