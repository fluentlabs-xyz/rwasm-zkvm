//! Programs that can be executed by the SP1 zkVM.

use std::hash::Hash;
use std::{fs::File, io::Read, str::FromStr};

use crate::RwasmAirId;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::Field;
use p3_field::{AbstractExtensionField, PrimeField32};
use p3_maybe_rayon::prelude::IntoParallelIterator;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

use rwasm::mem_index::AddressType;
use rwasm::{InstructionSet, Opcode, RwasmModule};
use serde::{Deserialize, Serialize};
use sp1_stark::septic_curve::{SepticCurve, SepticCurveComplete};
use sp1_stark::septic_digest::SepticDigest;
use sp1_stark::septic_extension::SepticExtension;
use sp1_stark::InteractionKind;
use sp1_stark::{
    air::{MachineAir, MachineProgram},
    shape::Shape,
};

/// A program that can be executed by the SP1 zkVM.
///
/// Contains a series of opcodes along with the initial memory image. It also contains the
/// start address and base address of the program.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Program {
    pub module: RwasmModule,
    pub memory_image: HashMap<u32, u32>,
    pub preprocessed_shape: Option<Shape<RwasmAirId>>,
}

impl Program {
    /// Create a new [Program].
    #[must_use]
    pub fn new(rwasm_module: RwasmModule) -> Self {
        let memory_image = Program::memory_image(&rwasm_module);
        Self { module: rwasm_module, memory_image, preprocessed_shape: None }
    }

    #[must_use]
    pub fn from_instrs(vec: Vec<Opcode>) -> Self {
        let mut code_section = InstructionSet::new();
        code_section.instr = vec;

        Self {
            module: RwasmModule {
                code_section,
                data_section: vec![],
                elem_section: vec![],
                wasm_section: vec![],
            },
            memory_image: HashMap::new(),
            preprocessed_shape: None,
        }
    }

    #[must_use]
    pub fn with_elements(mut self, elements: Vec<u32>) -> Self {
        self.module.elem_section = elements;
        self
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM.
    ///
    /// # Errors
    ///
    /// This function may return an error if the ELF is not valid.
    pub fn from(input: &[u8]) -> eyre::Result<Self> {
        let (module, _) = RwasmModule::new(input);
        let memory_image = Program::memory_image(&module);
        Ok(Program { module, memory_image, preprocessed_shape: None })
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM from a file path.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file cannot be opened or read.
    // pub fn from_elf(path: &str) -> eyre::Result<Self> {
    //     let mut elf_code = Vec::new();
    //     File::open(path)?.read_to_end(&mut elf_code)?;
    //     Program::from(&elf_code)
    // }

    /// Custom logic for padding the trace to a power of two according to the proof shape.
    pub fn fixed_log2_rows<F: Field, A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        let id = RwasmAirId::from_str(&air.name()).unwrap();
        self.preprocessed_shape.as_ref().map(|shape| {
            shape
                .log2_height(&id)
                .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
        })
    }

    pub fn memory_image(module: &RwasmModule) -> HashMap<u32, u32> {
        let len = module.data_section.len() / 4;
        module
            .data_section
            .chunks(4)
            .zip(0..len as u32)
            .map(|x| (u32::from_le_bytes(x.0.try_into().unwrap()), x.1))
            .collect()
    }
    /// get Opcode by programm counter
    pub fn fetch(&self, pc: u32) -> Opcode {
        self.module.code_section[pc as usize]
    }

    pub fn build_memory_image_vec(&self) -> Vec<(u32, u32)> {
        let mut v_data: Vec<_> = self
            .module
            .data_section
            .windows(4)
            .enumerate()
            .map(|(addr, data)| {
                let addr = addr as u32;
                let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let v_addr = AddressType::Data(addr).to_virtual_addr();
                (v_addr, word)
            })
            .collect();

        v_data.extend(self.module.elem_section.iter().enumerate().map(|(addr, data)| {
            let v_addr = AddressType::Element(addr as u32).to_virtual_addr();
            (v_addr, *data)
        }));
        v_data
    }
}

impl<F: PrimeField32> MachineProgram<F> for Program {
    fn pc_start(&self) -> F {
        F::from_canonical_u32(0u32)
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        let mut digests: Vec<SepticCurveComplete<F>> = self
            .build_memory_image_vec()
            .iter()
            .par_bridge()
            .map(|(addr, word)| {
                let addr = *addr;
                let word = *word;
                let values = [
                    (InteractionKind::Memory as u32) << 16,
                    0,
                    addr as u32,
                    word as u32 & 255,
                    (word >> 8) & 255,
                    (word >> 16) & 255,
                    (word >> 24) & 255,
                ];
                let x_start =
                    SepticExtension::<F>::from_base_fn(|i| F::from_canonical_u32(values[i]));
                let (point, _, _, _) = SepticCurve::<F>::lift_x(x_start);
                SepticCurveComplete::Affine(point.neg())
            })
            .collect();
        digests.push(SepticCurveComplete::Affine(SepticDigest::<F>::zero().0));
        SepticDigest(
            digests.into_par_iter().reduce(|| SepticCurveComplete::Infinity, |a, b| a + b).point(),
        )
    }
}
