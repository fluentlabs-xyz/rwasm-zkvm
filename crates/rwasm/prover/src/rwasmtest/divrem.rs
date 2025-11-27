use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_for_div_case() {
    let ops = vec![
        // 1. I32DivS, b=61725, c=12
        Opcode::I32Const(61725.into()),
        Opcode::I32Const(12.into()),
        Opcode::I32DivS,
        // 2. I32RemU, b=74070, c=4294967283
        Opcode::I32Const(74070.into()),
        Opcode::I32Const(4294967283u32.into()),
        Opcode::I32RemU,
        // 3. I32RemS, b=86415, c=14
        Opcode::I32Const(86415.into()),
        Opcode::I32Const(14.into()),
        Opcode::I32RemS,
        // 4. I32DivU, b=98760, c=4294967281
        Opcode::I32Const(98760.into()),
        Opcode::I32Const(4294967281u32.into()),
        Opcode::I32DivU,
        // 5. I32DivS, b=111105, c=16
        Opcode::I32Const(111105.into()),
        Opcode::I32Const(16.into()),
        Opcode::I32DivS,
        // 6. I32RemU, b=123450, c=4294967279
        Opcode::I32Const(123450.into()),
        Opcode::I32Const(4294967279u32.into()),
        Opcode::I32RemU,
        // 7. I32RemS, b=135795, c=18
        Opcode::I32Const(135795.into()),
        Opcode::I32Const(18.into()),
        Opcode::I32RemS,
        // 8. I32DivU, b=148140, c=4294967277
        Opcode::I32Const(148140.into()),
        Opcode::I32Const(4294967277u32.into()),
        Opcode::I32DivU,
        // 9. I32DivS, b=160485, c=20
        Opcode::I32Const(160485.into()),
        Opcode::I32Const(20.into()),
        Opcode::I32DivS,
        // 10. I32RemU, b=172830, c=4294967275
        Opcode::I32Const(172830.into()),
        Opcode::I32Const(4294967275u32.into()),
        Opcode::I32RemU,
        // 11. I32RemS, b=185175, c=22
        Opcode::I32Const(185175.into()),
        Opcode::I32Const(22.into()),
        Opcode::I32RemS,
        // 12. I32DivU, b=197520, c=4294967273
        Opcode::I32Const(197520.into()),
        Opcode::I32Const(4294967273u32.into()),
        Opcode::I32DivU,
        // 13. I32DivS, b=209865, c=24
        Opcode::I32Const(209865.into()),
        Opcode::I32Const(24.into()),
        Opcode::I32DivS,
        // 14. I32RemU, b=222210, c=4294967271
        Opcode::I32Const(222210.into()),
        Opcode::I32Const(4294967271u32.into()),
        Opcode::I32RemU,
        // 15. I32RemS, b=234555, c=26
        Opcode::I32Const(234555.into()),
        Opcode::I32Const(26.into()),
        Opcode::I32RemS,
        // 16. I32DivU, b=246900, c=4294967269
        Opcode::I32Const(246900.into()),
        Opcode::I32Const(4294967269u32.into()),
        Opcode::I32DivU,
        // 17. I32DivS, b=259245, c=28
        Opcode::I32Const(259245.into()),
        Opcode::I32Const(28.into()),
        Opcode::I32DivS,
        // 18. I32RemU, b=271590, c=4294967267
        Opcode::I32Const(271590.into()),
        Opcode::I32Const(4294967267u32.into()),
        Opcode::I32RemU,
        // 19. I32RemS, b=283935, c=30
        Opcode::I32Const(283935.into()),
        Opcode::I32Const(30.into()),
        Opcode::I32RemS,
        // 20. I32DivU, b=296280, c=4294967265
        Opcode::I32Const(296280.into()),
        Opcode::I32Const(4294967265u32.into()),
        Opcode::I32DivU,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
#[should_panic]
pub fn test_base_case_for_divide_by_zero() {
    let ops = vec![Opcode::I32Const(0x8000u32.into()), Opcode::I32Const(0.into()), Opcode::I32DivS];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
#[test]
#[should_panic]
pub fn test_base_case_for_rems_divide_by_zero() {
    let ops = vec![Opcode::I32Const(0x8000u32.into()), Opcode::I32Const(0.into()), Opcode::I32RemS];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
#[test]
#[should_panic]
pub fn test_base_case_for_divs_mintrap() {
    let ops = vec![
        Opcode::I32Const(0x80000000u32.into()),
        Opcode::I32Const(0xffffffffu32.into()),
        Opcode::I32DivS,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_divu_mintrap() {
    let ops = vec![
        Opcode::I32Const(0x80000000u32.into()),
        Opcode::I32Const(0xffffffffu32.into()),
        Opcode::I32DivU,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_for_rems_mintrap() {
    let ops = vec![
        Opcode::I32Const(0x80000000u32.into()),
        Opcode::I32Const(0xffffffffu32.into()),
        Opcode::I32RemS,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
#[test]
pub fn test_base_case_for_remu_mintrap() {
    let ops = vec![
        Opcode::I32Const(0x80000000u32.into()),
        Opcode::I32Const(0xffffffffu32.into()),
        Opcode::I32RemU,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

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
