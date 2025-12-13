use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

/// Emit `i32.const b; i32.const c; <op>; drop;`
#[inline]
fn push_cmp(ops: &mut Vec<Opcode>, op: Opcode, b: u32, c: u32) {
    ops.push(Opcode::I32Const(b.into()));
    ops.push(Opcode::I32Const(c.into()));
    ops.push(op);
    ops.push(Opcode::Drop);
}

/// Emit `i32.const x; i32.eqz; drop;`
#[inline]
fn push_eqz(ops: &mut Vec<Opcode>, x: u32) {
    ops.push(Opcode::I32Const(x.into()));
    ops.push(Opcode::I32Eqz);
    ops.push(Opcode::Drop);
}

/// Emit `i32.const b; i32.const c; i32.eq; drop;`
#[inline]
fn push_eq(ops: &mut Vec<Opcode>, b: u32, c: u32) {
    push_cmp(ops, Opcode::I32Eq, b, c)
}

/// Emit `i32.const b; i32.const c; i32.lt_u; drop;`
#[inline]
fn push_ltu(ops: &mut Vec<Opcode>, b: u32, c: u32) {
    push_cmp(ops, Opcode::I32LtU, b, c)
}

/// Emit `i32.const b; i32.const c; i32.lt_s; drop;`
#[inline]
fn push_lts(ops: &mut Vec<Opcode>, b: u32, c: u32) {
    push_cmp(ops, Opcode::I32LtS, b, c)
}
#[test]
pub fn test_ltchip_single() {
    let mut ops: Vec<Opcode> = Vec::new();
    const NEG2: u32 = 0xFFFF_FFFE;

    push_cmp(&mut ops, Opcode::I32LeS, NEG2, NEG2);
    push_cmp(&mut ops, Opcode::I32LeU, 15, 15);
    push_cmp(&mut ops, Opcode::I32GeS, NEG2, NEG2);
    push_cmp(&mut ops, Opcode::I32GeU, 15, 15);

    push_cmp(&mut ops, Opcode::I32LeS, 14, NEG2);
    push_cmp(&mut ops, Opcode::I32LeS, NEG2, 16);
    push_cmp(&mut ops, Opcode::I32LeU, 14, 15);
    push_cmp(&mut ops, Opcode::I32LeU, 15, 16);

    push_cmp(&mut ops, Opcode::I32GeS, 14, NEG2);
    push_cmp(&mut ops, Opcode::I32GeS, NEG2, 16);
    push_cmp(&mut ops, Opcode::I32GeU, 14, 15);
    push_cmp(&mut ops, Opcode::I32GeU, 15, 16);

    push_eq(&mut ops, 0, 0);
    push_eq(&mut ops, NEG2, NEG2);
    push_eq(&mut ops, 22, 22);
    push_eq(&mut ops, 22, NEG2);
    push_eq(&mut ops, NEG2, 22);

    push_cmp(&mut ops, Opcode::I32Ne, 0, 0);
    push_cmp(&mut ops, Opcode::I32Ne, NEG2, NEG2);
    push_cmp(&mut ops, Opcode::I32Ne, 22, 22);
    push_cmp(&mut ops, Opcode::I32Ne, 22, NEG2);
    push_cmp(&mut ops, Opcode::I32Ne, NEG2, 22);

    push_eqz(&mut ops, 0);
    push_eqz(&mut ops, NEG2);
    push_eqz(&mut ops, 22);

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}
#[test]
pub fn test_ltchip_batch_core_edges() {
    // Goal: cover the *critical* edge families but run prover only once.
    //
    // Covers:
    // - eqz: 0 / 1 / max / min / all-ones / random
    // - eq: equal + unequal + sign-boundary pairs
    // - lt_u: equal, boundary around 0x7fff_ffff / 0x8000_0000, max/min, first-diff in each byte
    // - lt_s: pos/pos, neg/neg, neg/pos, pos/neg, i32::MIN/MAX boundaries, masked-top-byte behavior

    const I32_MIN: u32 = 0x8000_0000;
    const I32_MAX: u32 = 0x7FFF_FFFF;
    const NEG1: u32 = 0xFFFF_FFFF;
    const NEG2: u32 = 0xFFFF_FFFE;
    const NEG3: u32 = 0xFFFF_FFFD;

    let mut ops: Vec<Opcode> = Vec::new();

    // ---- EQZ edge cases ----
    for &x in &[0x0000_0000, 0x0000_0001, 0x0000_0002, I32_MAX, I32_MIN, NEG1, 0xDEAD_BEEF] {
        push_eqz(&mut ops, x);
    }

    // ---- EQ edge cases ----
    for &(b, c) in &[
        (0x0000_0000, 0x0000_0000),
        (0x0000_0001, 0x0000_0001),
        (I32_MAX, I32_MAX),
        (I32_MIN, I32_MIN),
        (NEG1, NEG1),
        (0xDEAD_BEEF, 0xDEAD_BEEF),
        // unequal
        (0x0000_0000, 0x0000_0001),
        (0x0000_0001, 0x0000_0000),
        (I32_MAX, I32_MIN),
        (I32_MIN, I32_MAX),
        (0xDEAD_BEEF, 0xDEAD_BEEE),
    ] {
        push_eq(&mut ops, b, c);
    }

    // ---- LTU edge cases (unsigned) ----
    for &(b, c) in &[
        // equal
        (0x0000_0000, 0x0000_0000),
        (NEG1, NEG1),
        (I32_MIN, I32_MIN),
        // small boundaries
        (0x0000_0000, 0x0000_0001),
        (0x0000_0001, 0x0000_0000),
        // sign boundary is just magnitude boundary in unsigned
        (I32_MAX, I32_MIN),
        (I32_MIN, I32_MAX),
        // max vs 0
        (NEG1, 0x0000_0000),
        (0x0000_0000, NEG1),
        // first differing byte in byte3/2/1/0 (MSB->LSB)
        (0x0100_0000, 0x0200_0000), // byte3 differs
        (0x0001_0000, 0x0002_0000), // byte2 differs
        (0x0000_0100, 0x0000_0200), // byte1 differs
        (0x0000_0001, 0x0000_0002), // byte0 differs
        // same MSB region, differ inside (stress "first-diff" selection)
        (0x80FF_0000, 0x80FE_FFFF),
        (0xDEAD_BEEF, 0xDEAD_BEF0),
    ] {
        push_ltu(&mut ops, b, c);
        // also include reverse for symmetry (often catches swapped-operand bugs)
        push_ltu(&mut ops, c, b);
    }

    // ---- LTS edge cases (signed) ----
    for &(b, c) in &[
        // equal
        (0, 0),
        (I32_MAX, I32_MAX),
        (I32_MIN, I32_MIN),
        (NEG1, NEG1),
        // pos/pos
        (1, 2),
        (2, 1),
        // neg/neg
        (NEG3, NEG2), // -3 < -2
        (NEG2, NEG3), // -2 < -3
        // neg/pos (always true)
        (NEG1, 0),
        (I32_MIN, 123),
        // pos/neg (always false)
        (0, NEG1),
        (123, NEG3),
        // MIN/MAX boundary
        (I32_MIN, I32_MAX),
        (I32_MAX, I32_MIN),
        // same sign-bit=1 but different magnitudes (mask+unsigned path)
        (0x8000_0001, 0x8000_0002),
        (0x8000_0002, 0x8000_0001),
    ] {
        push_lts(&mut ops, b, c);
    }

    // Finish with a `drop`-only no-op is unnecessary; program ends cleanly.
    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_ltchip_batch_gt_le_ge_integration_edges() {
    // Second (and last) prover run: integration coverage for GT/LE/GE variants,
    // assuming your executor lowers them to LtChip plumbing (swap operands / negate logic).
    //
    // Keep it small but sharp.

    const I32_MIN: u32 = 0x8000_0000;
    const I32_MAX: u32 = 0x7FFF_FFFF;
    const NEG1: u32 = 0xFFFF_FFFF;

    let mut ops: Vec<Opcode> = Vec::new();

    // Unsigned family
    for &(b, c) in
        &[(0, 0), (0, 1), (1, 0), (NEG1, 0), (0, NEG1), (I32_MIN, I32_MAX), (I32_MAX, I32_MIN)]
    {
        push_cmp(&mut ops, Opcode::I32GtU, b, c);
        push_cmp(&mut ops, Opcode::I32LeU, b, c);
        push_cmp(&mut ops, Opcode::I32GeU, b, c);
    }

    // Signed family
    for &(b, c) in &[
        (0, 0),
        (1, 2),
        (2, 1),
        (NEG1, 0),
        (0, NEG1),
        (I32_MIN, I32_MAX),
        (I32_MAX, I32_MIN),
        (I32_MIN, NEG1),
        (NEG1, I32_MIN),
    ] {
        push_cmp(&mut ops, Opcode::I32GtS, b, c);
        push_cmp(&mut ops, Opcode::I32LeS, b, c);
        push_cmp(&mut ops, Opcode::I32GeS, b, c);
    }

    let program = Program::from_instrs(ops);
    run_rwasm_prover(program);
}

#[test]
pub fn test_ltchip_complex_stack_flow_single_run() {
    // One more run (optional). If you want strictly <=2 runs total, delete this test.
    // This checks "mixed opcodes in one program" scheduling behavior.
    let ops = vec![
        Opcode::I32Const(2u32.into()),
        Opcode::I32Const(3u32.into()),
        Opcode::I32LtS,
        Opcode::Drop,
        Opcode::I32Const(0u32.into()),
        Opcode::I32Const(0xFFFF_FFFFu32.into()),
        Opcode::I32LtU,
        Opcode::Drop,
        Opcode::I32Const(0u32.into()),
        Opcode::I32Eqz,
        Opcode::Drop,
        Opcode::I32Const(0x8000_0000u32.into()),
        Opcode::I32Const(0x7FFF_FFFFu32.into()),
        Opcode::I32Eq,
        Opcode::Drop,
    ];
    run_rwasm_prover(Program::from_instrs(ops));
}
