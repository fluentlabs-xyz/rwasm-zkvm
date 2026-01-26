#![allow(
    clippy::new_without_default,
    clippy::field_reassign_with_default,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_unwrap,
    clippy::default_constructed_unit_structs,
    clippy::box_default,
    clippy::assign_op_pattern,
    deprecated,
    incomplete_features,
    unused_variables
)]
#![warn(unused_extern_crates)]

pub mod air;
pub mod alu;
pub mod bytes;
pub mod constins;
pub mod control_flow;
pub mod cpu;
pub mod global;
pub mod io;
pub mod memory;
pub mod non_alu_instructions;
pub mod operations;
pub mod program;
pub mod rwasm;
pub mod shape;
#[cfg(feature = "sys")]
pub mod sys;
pub mod syscall;
pub mod utils;

// Re-export the `SP1ReduceProof` struct from rwasm_machine.
//
// This is done to avoid a circular dependency between rwasm_machine and rwasm_executor, and
// enable crates that depend on rwasm_machine to import the `SP1ReduceProof` type directly.
pub mod reduce {
    pub use rwasm_executor::SP1ReduceProof;
}

pub mod programs {
    #[allow(dead_code)]
    #[allow(missing_docs)]
    pub mod tests {

        use rwasm::Opcode;
        use rwasm_executor::Program;

        #[must_use]
        pub fn build_elf() -> Program {
            let x_value: u32 = 0x11;
            let y_value: u32 = 0x23;
            let z1_value: u32 = 0x40;
            let z2_value: u32 = 0x37;
            let z3_value: u32 = 0x1800;
            let z4_value: u32 = 0x2;
            let z5_value: u32 = 0x7;
            let z6_value: u32 = 0x21;

            let instructions = vec![
                Opcode::I32Const(z6_value.into()),
                // Opcode::I32Const(z5_value.into()),
                // Opcode::I32Const(z4_value.into()),
                // Opcode::I32Const(z3_value.into()),
                // Opcode::I32Const(z2_value.into()),
                // Opcode::I32Const(z1_value.into()),
                // Opcode::I32Const(y_value.into()),
                // Opcode::I32Const(x_value.into()),
                // Opcode::I32Add,
                // Opcode::I32Sub,
                // Opcode::I32Mul,
                // Opcode::I32DivS,
                // Opcode::I32DivU,
            ];

            //  memory_image: BTreeMap::new() };

            Program::from_instrs(instructions)
        }

        #[must_use]
        pub fn simple_program() -> Program {
            let instructions = vec![
                Opcode::I32Const(29.into()),
                Opcode::I32Const(5.into()),
                Opcode::I32Const(30.into()),
                Opcode::I32Const(37.into()),
                Opcode::I32Const(31.into()),
                Opcode::I32Const(30.into()),
                Opcode::I32Const(29.into()),
                Opcode::I32Add,
                Opcode::I32Add,
                Opcode::I32Add,
            ];

            //  memory_image: BTreeMap::new() };

            Program::from_instrs(instructions)
        }

        /*

        // /// Get the fibonacci program.
        // ///
        // /// # Panics
        // ///
        // /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn fibonacci_program() -> Program {
        //     Program::from(FIBONACCI_ELF).unwrap()
        // }

        // /// Get the secp256r1 add program.
        // ///
        // /// # Panics
        // ///
        // /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn secp256r1_add_program() -> Program {
        //     Program::from(SECP256R1_ADD_ELF).unwrap()
        // }

        // /// Get the secp256r1 double program.
        // ///
        // /// # Panics
        // ///
        // /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn secp256r1_double_program() -> Program {
        //     Program::from(SECP256R1_DOUBLE_ELF).unwrap()
        // }

        // /// Get the u256x2048 mul program.
        // ///
        // /// # Panics
        // ///
        // /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn u256xu2048_mul_program() -> Program {
        //     Program::from(U256XU2048_MUL_ELF).unwrap()
        // }

        /// Get the SSZ withdrawals program.
        ///
        /// # Panics
        // /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn ssz_withdrawals_program() -> Program {
        //     Program::from(SSZ_WITHDRAWALS_ELF).unwrap()
        // }

        /// Get the panic program.
        ///
        /// # Panics
        ///
        /// This function will panic if the program fails to load.
        // #[must_use]
        // pub fn panic_program() -> Program {
        //     Program::from(PANIC_ELF).unwrap()
        // }

        */

        #[must_use]
        #[allow(clippy::unreadable_literal)]
        pub fn simple_memory_program() -> Program {
            let instructions = vec![
                // Opcode::new(Opcode::ADD, 29, 0, 0x12348765, false, true),
                // // SW and LW
                // Opcode::new(Opcode::SW, 29, 0, 0x27654320, false, true),
                // Opcode::new(Opcode::LW, 28, 0, 0x27654320, false, true),
                // // LBU
                // Opcode::new(Opcode::LBU, 27, 0, 0x27654320, false, true),
                // Opcode::new(Opcode::LBU, 26, 0, 0x27654321, false, true),
                // Opcode::new(Opcode::LBU, 25, 0, 0x27654322, false, true),
                // Opcode::new(Opcode::LBU, 24, 0, 0x27654323, false, true),
                // // LB
                // Opcode::new(Opcode::LB, 23, 0, 0x27654320, false, true),
                // Opcode::new(Opcode::LB, 22, 0, 0x27654321, false, true),
                // // LHU
                // Opcode::new(Opcode::LHU, 21, 0, 0x27654320, false, true),
                // Opcode::new(Opcode::LHU, 20, 0, 0x27654322, false, true),
                // // LU
                // Opcode::new(Opcode::LH, 19, 0, 0x27654320, false, true),
                // Opcode::new(Opcode::LH, 18, 0, 0x27654322, false, true),
                // // SB
                // Opcode::new(Opcode::ADD, 17, 0, 0x38276525, false, true),
                // // Save the value 0x12348765 into address 0x43627530
                // Opcode::new(Opcode::SW, 29, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SB, 17, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::LW, 16, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SB, 17, 0, 0x43627531, false, true),
                // Opcode::new(Opcode::LW, 15, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SB, 17, 0, 0x43627532, false, true),
                // Opcode::new(Opcode::LW, 14, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SB, 17, 0, 0x43627533, false, true),
                // Opcode::new(Opcode::LW, 13, 0, 0x43627530, false, true),
                // // SH
                // // Save the value 0x12348765 into address 0x43627530
                // Opcode::new(Opcode::SW, 29, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SH, 17, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::LW, 12, 0, 0x43627530, false, true),
                // Opcode::new(Opcode::SH, 17, 0, 0x43627532, false, true),
                // Opcode::new(Opcode::LW, 11, 0, 0x43627530, false, true),
            ];
            Program::from_instrs(instructions)
        }
    }
}
