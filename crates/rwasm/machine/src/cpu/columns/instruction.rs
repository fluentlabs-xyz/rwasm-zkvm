use p3_field::PrimeField;
use p3_util::indices_arr;
use rwasm::Opcode;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::{size_of, transmute};

pub const NUM_INSTRUCTION_COLS: usize = size_of::<InstructionCols<u8>>();
pub const INSTRUCTION_COL_MAP: InstructionCols<usize> = make_col_map();

/// The column layout for instructions.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct InstructionCols<T> {
    /// The opcode for this cycle.
    pub opcode: T,
    pub aux_val: Word<T>,
    pub has_result: T,
    pub is_with_zero_params: T,
    pub is_with_one_param: T,
    pub is_with_two_three_params: T,
    pub is_implemented: T,
}

impl<F: PrimeField> InstructionCols<F> {
    pub fn populate(&mut self, opcode: Opcode) {
        self.opcode = F::from_canonical_u32(opcode.code());
        self.aux_val = Word::from(opcode.aux_value());

        self.has_result = F::from_bool(opcode.has_result());
        self.is_with_zero_params = F::from_bool(opcode.is_with_zero_params());
        self.is_with_one_param = F::from_bool(opcode.is_with_one_param());
        self.is_with_two_three_params = F::from_bool(opcode.is_with_two_params() || opcode.is_with_three_params());
    }
}

/// Creates the column map for the CPU.
const fn make_col_map() -> InstructionCols<usize> {
    let indices_arr = indices_arr::<NUM_INSTRUCTION_COLS>();
    unsafe { transmute::<[usize; NUM_INSTRUCTION_COLS], InstructionCols<usize>>(indices_arr) }
}
