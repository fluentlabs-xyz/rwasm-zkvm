use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_mul() {
    // Tests a standard multiplication case
    // 0x137137 * 0x137137 = 0x17A46E7669 (u64)
    // Truncated to u32: 0x46E7669
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Mul,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_mixed_add_mul_program() {
    let ops = vec![
        // 1. Standard Multiplication: -3 * 5 = -15
        Opcode::I32Const((-3).into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Mul,
        // 2. Standard Multiplication: 7 * 11 = 77
        Opcode::I32Const(7u32.into()),
        Opcode::I32Const(11u32.into()),
        Opcode::I32Mul,
        // 3. Addition (Stack: [-15, 77] -> [62])
        Opcode::I32Add,
        // 4. Wrapping Multiplication Test
        // 92 * u32::MAX (which is -1 in two's complement)
        // Result should be -92 (wrapping)
        Opcode::I32Const(u32::MAX.into()),
        Opcode::I32Mul,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
