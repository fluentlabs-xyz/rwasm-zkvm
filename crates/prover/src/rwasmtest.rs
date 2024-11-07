#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    
    use super::super::*;
    
   
    use sp1_core_executor::{Opcode, Program,};
   
    use sp1_core_machine::utils::setup_logger;
  

    use std::fs::File;
    use std::io::{Read, Write};
    use serde::{Deserialize, Serialize};
    use super::*;
    use super::super::*;
    use anyhow::Result;
    use build::try_build_plonk_bn254_artifacts_dev;
    use p3_field::PrimeField32;
    use serial_test::serial;
    
    use rwasm::engine::bytecode::Instruction;
    fn build_elf()->Program{
        /*
          let t0 = Register::X5;
                let syscall_id = self.register(t0);
                c = self.rr(Register::X11, MemoryAccessPosition::C);
                b = self.rr(Register::X10, MemoryAccessPosition::B);
                let syscall = SyscallCode::from_u32(syscall_id);
         */

        
         let sp_value:u32 = 0x00_00_20_00;
         let x_value:u32 = 0x11;
         let y_value:u32 = 0x23;
       
         let mut mem= BTreeMap::new();
        
         mem.insert(sp_value, x_value);
         mem.insert(sp_value-1, y_value);
        
         println!("{:?}",mem);
         let instructions = vec![   Instruction::I32Add, 
           ];
  
         let program = Program{instructions:instructions,
              pc_base:0,
              pc_start:0,
              memory_image: mem };
            //  memory_image: BTreeMap::new() };
            
        program
 
    }
   
   
   
    #[test]
    fn test_rwasm_proof2(){
      
        let program = build_elf();
        setup_logger();
        let prover: SP1Prover = SP1Prover::new();
        let opts = SP1ProverOpts::default();
        let context = SP1Context::default();
    
        tracing::info!("setup elf");
        let (pk, vk) = prover.setup_with_program(&program);
    
        tracing::info!("prove core");
        let stdin = SP1Stdin::new();
        let core_proof = prover.prove_core_with_program(&pk.pk,program, &stdin, opts, context).unwrap();
    
        
        tracing::info!("initializing prover");
      
        
        tracing::info!("setup elf");
      
      
        tracing::info!("prove core");
        
       
        tracing::info!("verify core");
        prover.verify(&core_proof.proof, &vk).unwrap();
        println!("done rwasm proof");
    }
   
}