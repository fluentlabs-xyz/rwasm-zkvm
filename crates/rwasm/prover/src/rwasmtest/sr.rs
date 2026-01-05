use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

/// Basic logical right shift: tests I32ShrU with a small shift.
#[test]
pub fn test_base_case_shru() {
    // Logic: 0x137137 >> 5 (logical)
    // Tests that the SRU chip can handle a normal, non-edge shift.
    let ops = vec![
        Opcode::I32Const(0x137_137u32.into()),
        Opcode::I32Const(5u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Basic arithmetic right shift: tests sign-extension behavior of I32ShrS.
#[test]
pub fn test_base_case_shrs() {
    // Logic: (0x80000000 as i32) >> 1 (arithmetic)
    // Should keep the sign bit and fill with 1s on the left.
    let ops = vec![
        Opcode::I32Const(0x8000_0000u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32ShrS,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test that shifting by 0 leaves the value unchanged (both SRU and SRS).
#[test]
pub fn test_shift_by_zero() {
    let ops = vec![
        // SRU: 0x21212121 >> 0 = 0x21212121
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // SRS: 0x81818181 >> 0 = 0x81818181 (no change, but arithmetic path)
        Opcode::I32Const(0x8181_8181u32.into()),
        Opcode::I32Const(0u32.into()),
        Opcode::I32ShrS,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test maximum effective shift (31) and that higher shift counts are masked by 31.
#[test]
pub fn test_shift_amount_masking_and_max_shift() {
    // These hit:
    //  - shift by 31 directly
    //  - large shift values that differ only in the upper bits but have same (c & 31)
    let ops = vec![
        // SRU simple max shift: 0xFFFF_FFFF >> 31 = 1
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32Const(31u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // SRU: check masking (0xFFFF_FFE0 & 31 = 0)
        // 0x21212121 >> 0xFFFF_FFE0  == 0x21212121 >> 0
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE0u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // SRU: (0xFFFF_FFE1 & 31 = 1)
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE1u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // SRS: masking for negative number
        // 0x80000001 >> 0xFFFF_FFFF  == 0x80000001 >> 31 (arithmetic)
        Opcode::I32Const(0x8000_0001u32.into()),
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32ShrS,
        Opcode::Drop,
    ];
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test a battery of SRU edge cases (unsigned logical shifts).
#[test]
pub fn test_shru_edge_cases() {
    // (value, shift)
    let cases: &[(u32, u32)] = &[
        // Zero shift
        (0xFFFF_8000, 0),
        // Small shifts
        (0xFFFF_8000, 1),
        (0xFFFF_8000, 7),
        (0xFFFF_8000, 14),
        (0xFFFF_8001, 15),
        // All ones
        (0xFFFF_FFFF, 0),
        (0xFFFF_FFFF, 1),
        (0xFFFF_FFFF, 7),
        (0xFFFF_FFFF, 14),
        (0xFFFF_FFFF, 31),
        // Mixed pattern
        (0x2121_2121, 0),
        (0x2121_2121, 1),
        (0x2121_2121, 7),
        (0x2121_2121, 14),
        (0x2121_2121, 31),
        // Masking behavior with large shift operands
        (0x2121_2121, 0xFFFF_FFE0), // &31 = 0
        (0x2121_2121, 0xFFFF_FFE1), // &31 = 1
        (0x2121_2121, 0xFFFF_FFE7), // &31 = 7
        (0x2121_2121, 0xFFFF_FFEE), // &31 = 14
        (0x2121_2121, 0xFFFF_FFFF), // &31 = 31
    ];

    let mut ops = Vec::new();
    for &(val, shift) in cases {
        ops.push(Opcode::I32Const(val.into()));
        ops.push(Opcode::I32Const(shift.into()));
        ops.push(Opcode::I32ShrU);
        ops.push(Opcode::Drop);
    }

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Test a battery of SRS edge cases (signed arithmetic shifts).
#[test]
pub fn test_shrs_edge_cases() {
    // (value, shift)
    let cases: &[(u32, u32)] = &[
        // Zero and positive values
        (0x0000_0000, 0),
        (0x7FFF_FFFF, 0),
        (0x7FFF_FFFF, 1),
        (0x7FFF_FFFF, 7),
        (0x7FFF_FFFF, 14),
        (0x7FFF_FFFF, 31),
        // Negative values (MSB = 1) to test sign extension
        (0x8000_0000, 1),
        (0x8000_0000, 7),
        (0x8000_0000, 14),
        (0x8000_0001, 31),
        // Mixed negative pattern
        (0x8181_8181, 0),
        (0x8181_8181, 1),
        (0x8181_8181, 7),
        (0x8181_8181, 14),
        (0x8181_8181, 31),
        // Masking behavior on SRS with large shift operands
        (0x8000_0001, 0xFFFF_FFE0), // &31 = 0
        (0x8000_0001, 0xFFFF_FFE1), // &31 = 1
        (0x8000_0001, 0xFFFF_FFE7), // &31 = 7
        (0x8000_0001, 0xFFFF_FFEE), // &31 = 14
        (0x8000_0001, 0xFFFF_FFFF), // &31 = 31
    ];

    let mut ops = Vec::new();
    for &(val, shift) in cases {
        ops.push(Opcode::I32Const(val.into()));
        ops.push(Opcode::I32Const(shift.into()));
        ops.push(Opcode::I32ShrS);
        ops.push(Opcode::Drop);
    }

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

/// Mixed SRU/SRS sequence to stress state transitions and sign handling.
#[test]
pub fn test_sr_mixed_sequence() {
    // Sequence:
    // 1. SRU on a positive pattern
    // 2. SRU with masked large shift
    // 3. SRS on a negative value
    // 4. SRS with masked large shift
    let ops = vec![
        // 1) SRU: 0x21212121 >> 7
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(7u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // 2) SRU: 0x21212121 >> 0xFFFF_FFE7 (same as >> 7)
        Opcode::I32Const(0x2121_2121u32.into()),
        Opcode::I32Const(0xFFFF_FFE7u32.into()),
        Opcode::I32ShrU,
        Opcode::Drop,
        // 3) SRS: 0x80000000 >> 1 (sign extension)
        Opcode::I32Const(0x8000_0000u32.into()),
        Opcode::I32Const(1u32.into()),
        Opcode::I32ShrS,
        Opcode::Drop,
        // 4) SRS: 0x81818181 >> 0xFFFF_FFFF (same as >> 31, still negative)
        Opcode::I32Const(0x8181_8181u32.into()),
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32ShrS,
        Opcode::Drop,
    ];

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
