use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};

use rwasm::{
    is_multi_align,
    mem_index::{TypedAddress, UNIT},
};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::air::MachineAir;

use crate::utils::{next_power_of_two, zeroed_f_vec};

use super::{
    columns::{MemoryInstructionsColumns, NUM_MEMORY_INSTRUCTIONS_COLUMNS},
    MemoryInstructionsChip,
};

impl<F: PrimeField32> MachineAir<F> for MemoryInstructionsChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "MemoryInstrs".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_instr_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.memory_instr_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MEMORY_INSTRUCTIONS_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_MEMORY_INSTRUCTIONS_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_MEMORY_INSTRUCTIONS_COLUMNS).enumerate().for_each(
                    |(j, row)| {
                        let idx = i * chunk_size + j;
                        let cols: &mut MemoryInstructionsColumns<F> = row.borrow_mut();

                        if idx < input.memory_instr_events.len() {
                            let event = &input.memory_instr_events[idx];
                            self.event_to_row(event, cols, &mut blu);
                        }
                    },
                );
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MEMORY_INSTRUCTIONS_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_instr_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MemoryInstructionsChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut MemoryInstructionsColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        cols.shard = F::from_canonical_u32(event.shard);
        assert!(cols.shard != F::zero());
        cols.clk = F::from_canonical_u32(event.clk);
        cols.pc = F::from_canonical_u32(event.pc);

        cols.sp = F::from_canonical_u32(event.sp);

        cols.raw_addr.populate(event.arg1, blu, true);
        let offset: u32 = event.opcode.aux_value();
        cols.instr_offset.populate(offset, blu, true);

        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        // Populate addr_word and addr_aligned columns.
        let memory_addr = event.arg1.wrapping_add(offset);
        let typed_addr = TypedAddress::GlobalMemory(memory_addr - memory_addr % WORD_SIZE as u32);

        let aligned_addr = typed_addr.to_virtual_addr();
        let aligned_addr_hi = aligned_addr + UNIT;
        let virtual_addr = TypedAddress::GlobalMemory(memory_addr).to_virtual_addr();
        let is_multi_aligned = is_multi_align(event.opcode, memory_addr);
        if is_multi_aligned {
            cols.memory_access_hi.populate(event.mem_access_hi.unwrap(), blu);
            if event.opcode.is_memory_load_instruction() {
                cols.is_multi_aligned_load = F::from_bool(true);
            } else {
                cols.is_multi_aligned_store = F::from_bool(true);
            }
        }
        cols.addr_word = virtual_addr.into();
        cols.memory_addr.populate(memory_addr, blu, true);

        cols.addr_aligned = F::from_canonical_u32(aligned_addr);
        // Populate the aa_least_sig_byte_decomp columns.
        assert!(aligned_addr.is_multiple_of(4));
        // Populate the aa_least_sig_byte_decomp columns.
        assert!(aligned_addr_hi.is_multiple_of(4));
        // Populate memory offsets.
        let addr_ls_two_bits = (memory_addr % WORD_SIZE as u32) as u8;

        // for store only
        cols.value = event.arg2.into();

        cols.addr_ls_two_bits = F::from_canonical_u8(addr_ls_two_bits);
        cols.ls_bits_is_one = F::from_bool(addr_ls_two_bits == 1);
        cols.ls_bits_is_two = F::from_bool(addr_ls_two_bits == 2);
        cols.ls_bits_is_three = F::from_bool(addr_ls_two_bits == 3);

        // Add byte lookup event to verify correct calculation of addr_ls_two_bits.
        // blu.add_byte_lookup_event(ByteLookupEvent {
        //     opcode: ByteOpcode::AND,
        //     a1: addr_ls_two_bits as u16,
        //     a2: 0,
        //     b: cols.addr_word[0].as_canonical_u32() as u8,
        //     c: 0b11,
        // });

        println!("%%%%%%%%%%%%%% {} {:?} {:?} {} {}", addr_ls_two_bits, memory_addr, event.opcode, event.arg1, event.arg2);

        // If it is a load instruction, set the unsigned_mem_val column.
        let mem_value = event.mem_access.value();
        if matches!(
            event.opcode,
            Opcode::I32Load(_) |
                Opcode::I32Load16U(_) |
                Opcode::I32Load16S(_) |
                Opcode::I32Load8U(_) |
                Opcode::I32Load8S(_)
        ) {

            cols.value = event.res.into();

            match event.opcode {
                Opcode::I32Load8U(_) | Opcode::I32Load8S(_) => {
                    cols.unsigned_mem_val =
                        (mem_value.to_le_bytes()[addr_ls_two_bits as usize] as u32).into();
                }
                Opcode::I32Load16S(_) | Opcode::I32Load16U(_) => {
                    let value = match addr_ls_two_bits {
                        0 => mem_value & 0x0000FFFF,
                        2 => (mem_value & 0xFFFF0000) >> 16,
                        1 => (mem_value & 0x00FFFF00) >> 8,
                        _ => unreachable!(),
                    };
                    cols.unsigned_mem_val = value.into();
                }
                Opcode::I32Load(_) => {
                    cols.unsigned_mem_val = mem_value.into();
                }
                _ => unreachable!(),
            }

            // For the signed load instructions, we need to check if the loaded value is negative.
            if matches!(event.opcode, Opcode::I32Load8S(_) | Opcode::I32Load16S(_)) {
                let most_sig_mem_value_byte = if matches!(event.opcode, Opcode::I32Load8S(_)) {
                    cols.unsigned_mem_val.to_u32().to_le_bytes()[0]
                } else {
                    cols.unsigned_mem_val.to_u32().to_le_bytes()[1]
                };

                let most_sig_mem_value_bit = most_sig_mem_value_byte >> 7;

                cols.most_sig_byte = F::from_canonical_u8(most_sig_mem_value_byte);
                cols.most_sig_bit = F::from_canonical_u8(most_sig_mem_value_bit);

                // blu.add_byte_lookup_event(ByteLookupEvent {
                //     opcode: ByteOpcode::MSB,
                //     a1: most_sig_mem_value_bit as u16,
                //     a2: 0,
                //     b: most_sig_mem_value_byte,
                //     c: 0,
                // });
            }
        }

        cols.is_i32load8s = F::from_bool(matches!(event.opcode, Opcode::I32Load8S(_)));
        cols.is_i32load8u = F::from_bool(matches!(event.opcode, Opcode::I32Load8U(_)));
        cols.is_i32load16s = F::from_bool(matches!(event.opcode, Opcode::I32Load16S(_)));
        cols.is_i32load16u = F::from_bool(matches!(event.opcode, Opcode::I32Load16U(_)));
        cols.is_i32load = F::from_bool(matches!(event.opcode, Opcode::I32Load(_)));
        cols.is_i32store8 = F::from_bool(matches!(event.opcode, Opcode::I32Store8(_)));
        cols.is_i32store16 = F::from_bool(matches!(event.opcode, Opcode::I32Store16(_)));
        cols.is_i32store = F::from_bool(matches!(event.opcode, Opcode::I32Store(_)));

        cols.most_sig_bytes_zero
            .populate_from_field_element(cols.addr_word[1] + cols.addr_word[2] + cols.addr_word[3]);

        // if cols.most_sig_bytes_zero.result == F::one() {
        //     blu.add_byte_lookup_event(ByteLookupEvent {
        //         opcode: ByteOpcode::LTU,
        //         a1: 1,
        //         a2: 0,
        //         b: 31,
        //         c: cols.addr_word[0].as_canonical_u32() as u8,
        //     });
        // }
    }
}
