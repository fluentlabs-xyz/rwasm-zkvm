use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_xor() {
    // Logic: 0x137137 XOR 0x42
    // Tests basic XOR functionality
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x42u32.into()),
        Opcode::I32Xor,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_and() {
    // Logic: 0x137137 AND 0xFFFFF
    // Tests basic AND functionality
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0xFFFFFu32.into()),
        Opcode::I32And,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_or() {
    // Logic: 0x137137 OR 0xDEADBEEF
    // Tests basic OR functionality
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0xDEAD_BEEFu32.into()),
        Opcode::I32Or,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_complex_case() {
    // Logic: Mixed bitwise operations to test state transitions
    // 1. 0xAA XOR 0x55 = 0xFF
    // 2. 0xFF OR 0x0F00 = 0x0FFF
    // 3. 0x0FFF AND 0xF0F0 = 0x00F0
    let ops = vec![
        // Step 1: XOR
        Opcode::I32Const(0xAAu32.into()),
        Opcode::I32Const(0x55u32.into()),
        Opcode::I32Xor, // Stack: [0xFF]
        // Step 2: OR
        Opcode::I32Const(0x0F00u32.into()),
        Opcode::I32Or, // Stack: [0x0FFF]
        // Step 3: AND
        Opcode::I32Const(0xF0F0u32.into()),
        Opcode::I32And, // Stack: [0x00F0]
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
