use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_divu() {
    // 100 / 3 = 33
    // 0x64 / 0x03 = 0x21
    let ops = vec![Opcode::I32Const(100.into()), Opcode::I32Const(3.into()), Opcode::I32DivU];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_divs() {
    // -100 / 3 = -33
    // In two's complement:
    // -100 = 0xFFFFFF9C
    // 3 = 0x03
    // Result = 0xFFFFFFDF (-33)
    let ops = vec![Opcode::I32Const((-100).into()), Opcode::I32Const(3.into()), Opcode::I32DivS];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_remu() {
    // 100 % 6 = 4
    let ops = vec![Opcode::I32Const(100.into()), Opcode::I32Const(6.into()), Opcode::I32RemU];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_rems() {
    // -100 % 6 = -4
    // In WASM/x86, the sign of the remainder follows the dividend.
    let ops = vec![Opcode::I32Const((-100).into()), Opcode::I32Const(6.into()), Opcode::I32RemS];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_mixed_div_rem_program() {
    let ops = vec![
        // 1. Signed Division: -50 / 4 = -12
        Opcode::I32Const((-50).into()),
        Opcode::I32Const(4.into()),
        Opcode::I32DivS,
        // 2. Unsigned Remainder on the result
        // We take the previous result (-12 which is 0xFFFFFFF4)
        // and do RemU with 0xFFFFFFF0 (a large unsigned number)
        // 0xFFFFFFF4 % 0xFFFFFFF0 = 4
        Opcode::I32Const(0xFFFFFFF0u32.into()),
        Opcode::I32RemU,
        // 3. Chain with Multiplication: 4 * 20 = 80
        Opcode::I32Const(20.into()),
        Opcode::I32Mul,
        // 4. Signed Remainder: 80 % -9 = 8
        // (80 is positive, so remainder is positive)
        Opcode::I32Const((-9).into()),
        Opcode::I32RemS,
        // 5. Division by "Zero" check (based on your chip logic returning 0)
        // 8 / 0 = 0 (if your chip handles it gracefully) or trap
        // Note: Standard WASM traps here. If your chip returns 0 for div by zero, this passes.
        // If your executor traps, remove this step.
        // Opcode::I32Const(0.into()),
        // Opcode::I32DivU,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
