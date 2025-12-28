#[cfg(test)]
mod test {
    use rwasm::mem_index::SP_START;
    use sp1_stark::SP1CoreOpts;

    use crate::{Executor, Opcode, Program};

    #[test]
    fn test_consume_fuel() {
        let sp_value: u32 = SP_START;

        let opcodes = vec![Opcode::ConsumeFuel(5000)];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.store.fuel_consumed(), 5000);
    }

    #[test]
    fn test_consume_fuel_stack() {
        let sp_value: u32 = SP_START;

        let opcodes = vec![Opcode::I32Const(5000.into()), Opcode::ConsumeFuelStack];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.store.fuel_consumed(), 5000);
    }
}
