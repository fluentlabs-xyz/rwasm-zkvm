use p3_field::PrimeField;

use p3_util::indices_arr;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::{
    mem::{size_of, transmute},
    vec::IntoIter,
};
pub const NUM_INSTRUCTION_COLS: usize = size_of::<InstructionCols<u8>>();
pub const INSTRUCTION_COL_MAP: InstructionCols<usize> = make_col_map();
use rwasm::Opcode;
/// The column layout for instructions.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct InstructionCols<T> {
    /// The opcode for this cycle.
    pub opcode: T,
    pub aux_val: Word<T>,
    pub is_nullary: T,
    pub is_unary: T,
    pub is_binary: T,
    pub is_memory: T,
    pub is_branching: T,
    pub is_call_ins: T,
    pub is_local: T,
    /// Table selectors for opcodes.
    pub is_alu: T,
    pub is_ordinary_alu: T,
    pub is_comparison_alu: T,
    pub is_skipped: T,
    /// Table selectors for opcodes.
    pub is_ecall: T,

    pub is_unimpl: T,
    pub is_i32les: T,
    pub is_i32leu: T,
    pub is_i32gts: T,
    pub is_i32gtu: T,
    pub is_i32ges: T,
    pub is_i32geu: T,
    pub is_i32eq: T,
    pub is_i32ne: T,
    pub is_i32lts: T,
    pub is_i32ltu: T,
    pub is_i32eqz: T,

    pub is_i32load: T,
    pub is_i32load16s: T,
    pub is_i32load16u: T,
    pub is_i32load8u: T,
    pub is_i32load8s: T,
    pub is_i32store: T,
    pub is_i32store16: T,
    pub is_i32store8: T,

    pub is_br: T,
    pub is_brifnez: T,
    pub is_brifeqz: T,
    pub is_brtable: T,
    pub is_localget: T,
    pub is_localset: T,
    pub is_localtee: T,
    pub is_i32const: T,

    pub is_table_grow: T,

    pub is_callinternal: T,
    pub is_callindirect: T,
    pub is_call: T,
    pub is_return: T,
    pub is_64b_op: T,

    pub is_sig_check: T,
}

impl<F: PrimeField> InstructionCols<F> {
    pub fn populate(&mut self, opcode: Opcode) {
        self.opcode = F::from_canonical_u32(opcode.code());
        self.aux_val = Word::from(opcode.aux_value());
        self.is_nullary = F::from_bool(opcode.is_nullary());
        self.is_unary = F::from_bool(opcode.is_unary_instruction());
        self.is_binary = F::from_bool(opcode.is_binary_instruction());
        self.is_memory = F::from_bool(opcode.is_memory_instruction());
        self.is_branching = F::from_bool(opcode.is_branch_instruction());
        self.is_call_ins = F::from_bool(opcode.is_call_instruction());
        self.is_64b_op = F::from_bool(opcode.is_64b_op());
        match opcode {
            Opcode::LocalGet(_) | Opcode::LocalSet(_) | Opcode::LocalTee(_) => {
                self.is_local = F::one();
                match opcode {
                    Opcode::LocalGet(_) => self.is_localget = F::one(),
                    Opcode::LocalSet(_) => self.is_localset = F::one(),
                    Opcode::LocalTee(_) => self.is_localtee = F::one(),
                    _ => (),
                }
            }
            _ => (),
        }

        if opcode.is_alu_instruction() {
            match opcode {
                Opcode::I32LtS |
                Opcode::I32LtU |
                Opcode::I32GtS |
                Opcode::I32GtU |
                Opcode::I32GeS |
                Opcode::I32GeU |
                Opcode::I32LeS |
                Opcode::I32LeU |
                Opcode::I32Eqz |
                Opcode::I32Eq |
                Opcode::I32Ne => {
                    self.is_comparison_alu = F::one();
                }
                _ => {
                    self.is_ordinary_alu = F::one();
                }
            }
            self.is_alu = F::one();
        } else if opcode.is_ecall_instruction() {
            self.is_ecall = F::one();
        }
        match opcode {
            Opcode::I32Eqz => self.is_i32eqz = F::one(),
            Opcode::I32Eq => self.is_i32eq = F::one(),
            Opcode::I32Ne => self.is_i32ne = F::one(),
            Opcode::I32LtU => self.is_i32ltu = F::one(),
            Opcode::I32LtS => self.is_i32lts = F::one(),
            Opcode::I32GtS => self.is_i32gts = F::one(),
            Opcode::I32GtU => self.is_i32gtu = F::one(),
            Opcode::I32LeS => self.is_i32les = F::one(),
            Opcode::I32LeU => self.is_i32leu = F::one(),
            Opcode::I32GeS => self.is_i32ges = F::one(),
            Opcode::I32GeU => self.is_i32geu = F::one(),
            Opcode::I32Load(_) => self.is_i32load = F::one(),
            Opcode::I32Load16S(_) => self.is_i32load16s = F::one(),
            Opcode::I32Load16U(_) => self.is_i32load16u = F::one(),
            Opcode::I32Load8S(_) => self.is_i32load8s = F::one(),
            Opcode::I32Load8U(_) => self.is_i32load8u = F::one(),
            Opcode::I32Store(_) => self.is_i32store = F::one(),
            Opcode::I32Store16(_) => self.is_i32store16 = F::one(),
            Opcode::I32Store8(_) => self.is_i32store8 = F::one(),
            Opcode::Br(_) => self.is_br = F::one(),
            Opcode::BrIfEqz(_) => self.is_brifeqz = F::one(),
            Opcode::BrIfNez(_) => self.is_brifnez = F::one(),
            Opcode::BrTable(_) => self.is_brtable = F::one(),
            Opcode::LocalGet(_) => self.is_localget = F::one(),
            Opcode::LocalSet(_) => self.is_localset = F::one(),
            Opcode::LocalTee(_) => self.is_localtee = F::one(),
            Opcode::I32Const(_) => self.is_i32const = F::one(),
            Opcode::CallInternal(_) => self.is_callinternal = F::one(),
            Opcode::CallIndirect(_) => self.is_callindirect = F::one(),
            Opcode::Call(_) => self.is_call = F::one(),
            Opcode::Return => self.is_return = F::one(),
            Opcode::ConsumeFuel(_) => self.is_skipped = F::one(),
            Opcode::SignatureCheck(_) => self.is_sig_check = F::one(),
            Opcode::Drop => self.is_skipped = F::one(),
            Opcode::TableGrow(_) => self.is_table_grow = F::one(),

            _ => {}
        }
    }
}

impl<T> IntoIterator for InstructionCols<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let columns = vec![
            self.is_alu,
            self.is_ordinary_alu,
            self.is_comparison_alu,
            self.is_branching,
            self.is_memory,
            self.is_ecall,
            self.is_unimpl,
            self.is_i32lts,
            self.is_i32ltu,
            self.is_i32les,
            self.is_i32leu,
            self.is_i32ges,
            self.is_i32geu,
            self.is_i32gts,
            self.is_i32gtu,
            self.is_i32eq,
            self.is_i32ne,
            self.is_i32eqz,
            self.is_i32load,
            self.is_i32load16s,
            self.is_i32load16u,
            self.is_i32load8s,
            self.is_i32load8u,
            self.is_i32store,
            self.is_i32store16,
            self.is_i32store8,
            self.is_br,
            self.is_brifeqz,
            self.is_brifnez,
            self.is_brtable,
            self.is_localget,
            self.is_localset,
            self.is_localtee,
            self.is_i32const,
            self.is_callinternal,
            self.is_return,
            self.is_skipped,
            self.is_64b_op,
        ];

        columns.into_iter()
    }
}
/// Creates the column map for the CPU.
const fn make_col_map() -> InstructionCols<usize> {
    let indices_arr = indices_arr::<NUM_INSTRUCTION_COLS>();
    unsafe { transmute::<[usize; NUM_INSTRUCTION_COLS], InstructionCols<usize>>(indices_arr) }
}
