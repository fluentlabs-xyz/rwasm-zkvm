use p3_field::PrimeField;

use rwasm::engine::bytecode::Instruction;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::{iter::once, mem::size_of, vec::IntoIter};
pub const NUM_INSTRUCTION_COLS: usize = size_of::<InstructionCols<u8>>();

/// The column layout for instructions.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct InstructionCols<T> {
    /// The opcode for this cycle.
    pub opcode: T,

    /// The first operand for this instruction.
    pub op_a: Word<T>,

    pub op_b: Word<T>,
}

impl<F: PrimeField> InstructionCols<F> {
    pub fn populate(&mut self, instruction: Instruction) {
        self.opcode = F::from_canonical_u32(instruction.to_op());
        let (_, aux) = instruction.to_opcode_and_aux();
        match (aux) {
            Some(aux) => {
                let hi = (aux >> 32) as u32;
                let lo = aux as u32;
                self.op_a = Word::<F>::from(lo);
                self.op_b = Word::<F>::from(hi);
            }
            None => {
                self.op_a = Word::<F>::from(0u32);
                self.op_b = Word::<F>::from(0u32);
            }
        }
    }
}

impl<T> IntoIterator for InstructionCols<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        once(self.opcode).chain(self.op_a).chain(self.op_b).collect::<Vec<_>>().into_iter()
    }
}
