//! Programs that can be executed by the SP1 zkVM.

use std::str::FromStr;

use crate::RwasmAirId;

use hashbrown::HashMap;
use p3_field::{AbstractExtensionField, Field, PrimeField32};
use p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelBridge, ParallelIterator};

use rwasm::{mem_index::AddressType, InstructionSet, Opcode, RwasmModule, RwasmModuleInner};
use serde::{Deserialize, Serialize};
use sp1_stark::{
    air::{MachineAir, MachineProgram},
    septic_curve::{SepticCurve, SepticCurveComplete},
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    shape::Shape,
    InteractionKind,
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
        *code_section = vec;

        Self {
            module: RwasmModule::from(RwasmModuleInner {
                code_section,
                data_section: vec![],
                elem_section: vec![],
                hint_section: vec![],
            }),
            memory_image: HashMap::new(),
            preprocessed_shape: None,
        }
    }

    #[must_use]
    pub fn with_elements(mut self, elements: Vec<u32>) -> Self {
        let module = RwasmModule::from(RwasmModuleInner {
            code_section: self.module.code_section.clone(),
            data_section: vec![],
            elem_section: elements,
            hint_section: vec![],
        });
        self.module = module;
        self.memory_image = Program::memory_image(&self.module);
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

    /// Custom logic for padding the trace to a power of two according to the proof shape.
    pub fn fixed_log2_rows<F: Field, A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        let id = RwasmAirId::from_str(&air.name()).unwrap();
        self.preprocessed_shape.as_ref().map(|shape| {
            shape
                .log2_height(&id)
                .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
        })
    }

    #[must_use]
    pub fn memory_image(module: &RwasmModule) -> HashMap<u32, u32> {
        let mut v_data: HashMap<_, _> = module
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

        v_data.extend(module.elem_section.iter().enumerate().map(|(addr, data)| {
            let v_addr = AddressType::Element(addr as u32).to_virtual_addr();
            (v_addr, *data)
        }));
        v_data
    }
    /// get Opcode by programm counter
    pub fn fetch(&self, pc: u32) -> Opcode {
        self.module.code_section[pc as usize]
    }
}

impl<F: PrimeField32> MachineProgram<F> for Program {
    fn pc_start(&self) -> F {
        F::from_canonical_u32(0u32)
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        let mut digests: Vec<SepticCurveComplete<F>> = Program::memory_image(&self.module)
            .iter()
            .par_bridge()
            .map(|(addr, word)| {
                let addr = *addr;
                let word = *word;
                let values = [
                    (InteractionKind::Memory as u32) << 16,
                    0,
                    addr,
                    word & 255,
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
