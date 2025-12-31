use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
pub fn test_base_case_br_unconditional() {
    // Unconditional branch: skip over a "poison" instruction and land at Drop.
    //
    // Layout:
    // 0: push 123
    // 1: br +2     -> jump to pc+2 (skips the next I32Const(0xDEAD))
    // 2: push DEAD (must be skipped)
    // 3: drop      (target)
    let ops = vec![
        Opcode::I32Const(123u32.into()),
        Opcode::Br(2i32.into()),
        Opcode::I32Const(0xDEADu32.into()), // should be skipped
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_brifeqz_taken() {
    // BrIfEqz taken when top-of-stack == 0.
    //
    // 0: push 0
    // 1: br_if_eqz +2  -> taken, skips poison const
    // 2: push BEEF (must be skipped)
    // 3: drop
    let ops = vec![
        Opcode::I32Const(0u32.into()),
        Opcode::BrIfEqz(2i32.into()),
        Opcode::I32Const(0xBEEFu32.into()), // should be skipped
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_brifeqz_not_taken() {
    // BrIfEqz not taken when top-of-stack != 0.
    // We ensure the fall-through path is valid and ends cleanly.
    //
    // 0: push 1
    // 1: br_if_eqz +2  -> not taken
    // 2: drop          -> drops the 1
    let ops = vec![Opcode::I32Const(1u32.into()), Opcode::BrIfEqz(2i32.into())];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_brifnez_taken() {
    // BrIfNez taken when top-of-stack != 0.
    //
    // 0: push 7
    // 1: br_if_nez +2  -> taken, skips poison const
    // 2: push CAFE (must be skipped)
    // 3: drop
    let ops = vec![
        Opcode::I32Const(7u32.into()),
        Opcode::BrIfNez(2i32.into()),
        Opcode::I32Const(0xCAFEu32.into()), // should be skipped
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_base_case_brifnez_not_taken() {
    // BrIfNez not taken when top-of-stack == 0 (fall-through).
    //
    // 0: push 0
    // 1: br_if_nez +2  -> not taken
    // 2: drop
    let ops = vec![
        Opcode::I32Const(0.into()),
        Opcode::I32Const(8.into()),
        Opcode::TableGrow(0),
        Opcode::I32Const(0u32.into()),
        Opcode::BrIfNez(2i32.into()),
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_complex_case_mixed_branches() {
    // Mixed control flow to exercise:
    // - taken conditional
    // - not taken conditional
    // - unconditional branch
    //
    // High-level:
    // 1) Push 0, BrIfEqz taken -> jump over poison1
    // 2) Push 1, BrIfEqz not taken -> fall through and Drop it
    // 3) Unconditional Br to skip poison2
    //
    // The program should always terminate cleanly.
    let ops = vec![
        // Step 1: taken BrIfEqz (a == 0)
        Opcode::I32Const(0u32.into()),
        Opcode::BrIfEqz(2i32.into()),
        Opcode::I32Const(0x1111u32.into()), // poison1 (must be skipped)
        // Step 2: not taken BrIfEqz (a != 0) then Drop
        Opcode::I32Const(1u32.into()),
        Opcode::BrIfEqz(2i32.into()),
        // Step 3: unconditional Br to skip poison2
        Opcode::Br(2i32.into()),
        Opcode::I32Const(0x2222u32.into()), // poison2 (must be skipped)
    ];

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
