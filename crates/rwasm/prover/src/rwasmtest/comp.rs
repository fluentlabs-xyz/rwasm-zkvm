use crate::rwasmtest::run_rwasm_prover;
use rwasm_executor::{Opcode, Program};

#[test]
fn test_comparison_x_eq() {
    let x_value: u32 = 0x11;
    let y_value: u32 = 0x23;
    let z1_value: u32 = 0x3;
    let instructions = vec![
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32Const(z1_value.into()),
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32GeU,
        Opcode::I32GeS,
        Opcode::I32LeU,
        Opcode::I32LeS,
    ];
    let program = Program::from_instrs(instructions);
    run_rwasm_prover(program);
}
#[test]
fn test_comparison_gts() {
    let x_value: u32 = 0x11;
    let y_value: u32 = 0x23;
    let z1_value: u32 = 0x3;
    let instructions = vec![
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32Const(z1_value.into()),
        Opcode::I32GtS,
        Opcode::I32LtS,
    ];
    let program = Program::from_instrs(instructions);
    run_rwasm_prover(program);
}

#[test]
fn test_comparison_gtu() {
    let x_value: u32 = 0x11;
    let y_value: u32 = 0x23;
    let z1_value: u32 = 0x3;
    let instructions = vec![
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32Const(z1_value.into()),
        Opcode::I32GtU,
        Opcode::I32LtU,
    ];
    let program = Program::from_instrs(instructions);
    run_rwasm_prover(program);
}

#[test]
fn test_comparison_eq() {
    let x_value: u32 = 0x11;
    let y_value: u32 = 0x23;
    let z1_value: u32 = 0x3;
    let instructions = vec![
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32Const(z1_value.into()),
        Opcode::I32Const(x_value.into()),
        Opcode::I32Eq,
        Opcode::I32Ne,
        Opcode::I32Eqz,
    ];
    let program = Program::from_instrs(instructions);
    run_rwasm_prover(program);
}
