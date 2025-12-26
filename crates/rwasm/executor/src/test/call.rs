
#[cfg(test)]
mod test{
    use fluentbase_types::SysFuncIdx;
use rwasm::mem_index::SP_START;
use sp1_stark::SP1CoreOpts;

use crate::{Executor, Opcode, Program};

#[test]
fn test_call() {
    let sp_value: u32 = SP_START;
    let x_value: u32 = 0x7;
    let y_value: u32 = 0x2;
    let z_value: u32 = 0x1;
    let functions = [0, 24];

    let opcodes = vec![
        Opcode::I32Const(x_value.into()),
        Opcode::I32Const(y_value.into()),
        Opcode::I32Const(z_value.into()),
        Opcode::Call(SysFuncIdx::FUEL as u32),
    ];

    let program = Program::from_instrs(opcodes);
    let mut runtime = Executor::new(program, SP1CoreOpts::default());
    runtime.run().unwrap();
}
}

