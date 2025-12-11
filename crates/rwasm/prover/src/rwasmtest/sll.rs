use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

/// Basic logical left shift: tests I32Shl with a small shift.
#[test]
pub fn test_base_case_shl() {
    // Logic: 0x137137 << 5
    // Tests that the SLL chip can handle a normal, non-edge shift.
    let ops = vec![
        Opcode::I32Const(0x0137_0137u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test that shifting by 0 leaves the value unchanged.
#[test]
pub fn test_shiftleft_by_zero() {
    let ops = vec![
        // 0x21212121 << 0 = 0x21212121
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // 0x81818181 << 0 = 0x81818181
        Opcode::I32Const(0x8181_8181u32.into()),
        Opcode::I32Const(0u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test maximum effective shift (31) and that higher shift counts are masked by 31.
#[test]
pub fn test_shiftleft_amount_masking_and_max_shift() {
    // Wasm: effective shift = c & 31
    let ops = vec![
        // Simple max shift: 0x0000_0001 << 31 = 0x8000_0000
        Opcode::I32Const(0x0000_0001u32.into()),
        Opcode::I32Const(31u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // Masking: (0xFFFF_FFE0 & 31 = 0)
        // 0x21212121 << 0xFFFF_FFE0 == 0x21212121 << 0
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE0u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // Masking: (0xFFFF_FFE1 & 31 = 1)
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE1u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // Masking: (0xFFFF_FFFF & 31 = 31)
        Opcode::I32Const(0x0000_0001u32.into()),
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test a battery of SLL edge cases.
#[test]
pub fn test_shl_edge_cases() {
    // (value, shift)
    let cases: &[(u32, u32)] = &[
        // Zero value
        (0x0000_0000, 0),
        (0x0000_0000, 1),
        (0x0000_0000, 31),
        (0x0000_0000, 0xFFFF_FFFF),
        // One, shifting bits into various positions
        (0x0000_0001, 0),
        (0x0000_0001, 1),
        (0x0000_0001, 7),
        (0x0000_0001, 8),
        (0x0000_0001, 15),
        (0x0000_0001, 16),
        (0x0000_0001, 31),
        (0x0000_0001, 32),          // same as shift by 0
        (0x0000_0001, 0xFFFF_FFFF), // same as shift by 31
        // High bit set
        (0x8000_0000, 0),
        (0x8000_0000, 1),
        (0x8000_0000, 7),
        (0x8000_0000, 31),
        // All ones
        (0xFFFF_FFFF, 0),
        (0xFFFF_FFFF, 1),
        (0xFFFF_FFFF, 7),
        (0xFFFF_FFFF, 8),
        (0xFFFF_FFFF, 15),
        (0xFFFF_FFFF, 16),
        (0xFFFF_FFFF, 31),
        (0xFFFF_FFFF, 0xFFFF_FFE0), // &31 = 0
        (0xFFFF_FFFF, 0xFFFF_FFE1), // &31 = 1
        (0xFFFF_FFFF, 0xFFFF_FFE7), // &31 = 7
        (0xFFFF_FFFF, 0xFFFF_FFEE), // &31 = 14
        (0xFFFF_FFFF, 0xFFFF_FFFF), // &31 = 31
        // Mixed patterns
        (0x2121_2121, 0),
        (0x2121_2121, 1),
        (0x2121_2121, 7),
        (0x2121_2121, 14),
        (0x2121_2121, 31),
        (0x2121_2121, 0xFFFF_FFE0),
        (0x2121_2121, 0xFFFF_FFE1),
        (0x2121_2121, 0xFFFF_FFE7),
        (0x2121_2121, 0xFFFF_FFEE),
        (0x2121_2121, 0xFFFF_FFFF),
        (0x8181_8181, 0),
        (0x8181_8181, 1),
        (0x8181_8181, 7),
        (0x8181_8181, 14),
        (0x8181_8181, 31),
        (0x8181_8181, 0xFFFF_FFE3),
        (0x8181_8181, 0xFFFF_FFFF),
    ];

    let mut ops = Vec::new();
    for &(val, shift) in cases {
        ops.push(Opcode::I32Const(val.into()));
        ops.push(Opcode::I32Const(shift.into()));
        ops.push(Opcode::I32Shl);
        ops.push(Opcode::Drop);
    }

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Mixed SLL sequence to stress state transitions and masking.
#[test]
pub fn test_shl_mixed_sequence() {
    // Sequence:
    // 1. Simple pattern shift
    // 2. Same pattern with masked large shift
    // 3. High-bit patterns with different shift counts
    // 4. Masking on large shift for patterned value
    let ops = vec![
        // 1) 0x21212121 << 7
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(7u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // 2) 0x21212121 << 0xFFFF_FFE7 (same effective as << 7)
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE7u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // 3) 0x00000001 << 31
        Opcode::I32Const(0x0000_0001u32.into()),
        Opcode::I32Const(31u32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
        // 4) 0x81818181 << 0xFFFF_FFFF (same effective as << 31)
        Opcode::I32Const(0x8181_8181u32.into()),
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32Shl,
        Opcode::Drop,
    ];

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
