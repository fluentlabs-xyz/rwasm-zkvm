use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::Opcode;
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, CpuEvent, MemoryRecordEnum},
    syscalls::SyscallCode,
    ByteOpcode::{self, U16Range},
    ExecutionRecord, Program,
};

use sp1_stark::air::MachineAir;
use tracing::instrument;

use super::{columns::NUM_CPU_COLS, CpuChip};
use crate::{cpu::columns::CpuCols, memory::MemoryCols, utils::zeroed_f_vec};

impl<F: PrimeField32> MachineAir<F> for CpuChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        self.id().to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let n_real_rows = input.cpu_events.len();
        let padded_nb_rows = if let Some(shape) = &input.shape {
            shape.height(&self.id()).unwrap()
        } else if n_real_rows < 16 {
            16
        } else {
            n_real_rows.next_power_of_two()
        };
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_CPU_COLS);

        let chunk_size = std::cmp::max(input.cpu_events.len() / num_cpus::get(), 1);
        println!("input.cpu_events{:?}", input.cpu_events);
        values.chunks_mut(chunk_size * NUM_CPU_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_CPU_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut CpuCols<F> = row.borrow_mut();

                    if idx >= input.cpu_events.len() {
                        cols.is_syscall = F::one();
                    } else {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.cpu_events[idx];
                        let instruction = input.program.fetch(event.pc);
                        self.event_to_row(
                            event,
                            cols,
                            &mut byte_lookup_events,
                            input.public_values.execution_shard,
                            instruction,
                        );
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_CPU_COLS)
    }

    #[instrument(name = "generate cpu dependencies", level = "debug", skip_all)]
    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        // Generate the trace rows for each event.
        let chunk_size = std::cmp::max(input.cpu_events.len() / num_cpus::get(), 1);

        let blu_events: Vec<_> = input
            .cpu_events
            .par_chunks(chunk_size)
            .map(|ops: &[CpuEvent]| {
                // The blu map stores shard -> map(byte lookup event -> multiplicity).
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                ops.iter().for_each(|op| {
                    let mut row = [F::zero(); NUM_CPU_COLS];
                    let cols: &mut CpuCols<F> = row.as_mut_slice().borrow_mut();
                    let instruction = input.program.fetch(op.pc);
                    self.event_to_row::<F>(
                        op,
                        cols,
                        &mut blu,
                        input.public_values.execution_shard,
                        instruction,
                    );
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            shard.contains_cpu()
        }
    }
}

impl CpuChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CpuEvent,
        cols: &mut CpuCols<F>,
        blu_events: &mut impl ByteRecord,
        shard: u32,
        instruction: Opcode,
    ) {
        // Populate shard and clk columns.
        self.populate_shard_clk(cols, event, blu_events, shard);
        self.populate_alu(cols, event, instruction);
        // Populate basic fields.
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.sp = F::from_canonical_u32(event.sp);
        cols.next_sp = F::from_canonical_u32(event.next_sp);
        cols.call_data.call_sp = F::from_canonical_u32(event.call_sp);
        cols.call_data.next_call_sp = F::from_canonical_u32(event.next_call_sp);
        cols.instruction.populate(instruction);

        cols.is_memory = F::from_bool(
            instruction.is_memory_load_instruction() || instruction.is_memory_store_instruction(),
        );
        cols.is_syscall = F::from_bool(instruction.is_ecall_instruction());
        *cols.op_res_access.value_mut() = event.res.into();
        *cols.op_res_hi_access.value_mut() = event.res_hi.into();
        *cols.op_arg1_access.value_mut() = event.arg1.into();
        *cols.op_arg2_access.value_mut() = event.arg2.into();

        cols.shard_to_send = if instruction.is_memory_load_instruction() ||
            instruction.is_memory_store_instruction() ||
            instruction.is_ecall_instruction() ||
            instruction.is_call_instruction()
        {
            cols.shard
        } else {
            F::zero()
        };
        cols.clk_to_send = if instruction.is_memory_load_instruction() ||
            instruction.is_memory_store_instruction() ||
            instruction.is_ecall_instruction() ||
            instruction.is_call_instruction()
        {
            F::from_canonical_u32(event.clk)
        } else {
            F::zero()
        };

        // Populate memory accesses for result (lo and optional hi for 64-bit ops).
        if let Some(record_lo) = event.res_record {
            if instruction.is_ecall_instruction() {
                // For ecall instructions, pass in a dummy byte lookup vector.
                cols.op_res_access.populate(record_lo, &mut Vec::new());
            } else {
                let do_populate =
                    !matches!(instruction, |Opcode::ConsumeFuel(_)| Opcode::ConsumeFuelStack);
                if do_populate {
                    // Lo write
                    cols.op_res_access.populate(record_lo, blu_events);
                    let do_check = !matches!(instruction, Opcode::CallIndirect(_));

                    cols.op_res_addr.populate(
                        event.res_addr.unwrap().to_virtual_addr(),
                        blu_events,
                        do_check,
                    );
                }
            }
        }
        if let Some(record_hi) = event.res_hi_record {
            // Hi write for 64-bit ops

            if instruction.is_64b_op() {
                cols.op_res_hi_access.populate(record_hi, blu_events);
                cols.op_res_hi_addr.populate(
                    event.res_hi_addr.unwrap().to_virtual_addr(),
                    blu_events,
                    true,
                );
            } else {
                debug_assert!(false, "Missing res_hi_record for 64-bit op at pc={}", event.pc);
            }
        }
        // Populate arg1/arg2 memory reads.
        if let Some(MemoryRecordEnum::Read(record)) = event.arg1_record {
            //Do not check for LAST_SIG_ADDR because there is only one address.
            let do_populate =
                !matches!(instruction, Opcode::ConsumeFuel(_) | Opcode::ConsumeFuelStack);
            if do_populate {
                cols.op_arg1_access.populate(record, blu_events);
                let do_check = !matches!(instruction, Opcode::SignatureCheck(_));
                cols.op_arg1_addr.populate(
                    event.arg1_addr.unwrap().to_virtual_addr(),
                    blu_events,
                    do_check,
                );
            }
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.arg2_record {
            let do_populate = !matches!(instruction, Opcode::ConsumeFuel(_));
            cols.op_arg2_access.populate(record, blu_events);
            cols.op_arg2_addr.populate(
                event.arg2_addr.unwrap().to_virtual_addr(),
                blu_events,
                true,
            );
        }

        if instruction.is_ecall_instruction() {
            let syscall_id = match instruction {
                Opcode::Call(_) => instruction.aux_value(),
                Opcode::TableInit(_) => SyscallCode::TABLE_INIT.syscall_id(),
                Opcode::TableGrow(_) => SyscallCode::TABLE_GROW.syscall_id(),
                _ => unimplemented!(),
            };
            let syscall_id = F::from_canonical_u32(syscall_id);
            let num_extra_cycles = match instruction {
                Opcode::TableInit(_) => F::from_canonical_u32(2),
                _ => cols.op_res_access.prev_value[2],
            };
            cols.is_halt =
                F::from_bool(syscall_id == F::from_canonical_u32(SyscallCode::HALT.syscall_id()));
            cols.num_extra_cycles = num_extra_cycles;
        }

        if let Some(call_data) = &&event.call_data {
            println!("event.opcode:{},call_data:{:?}", instruction, event.call_data);
            cols.call_data.signature_id = F::from_canonical_u32(call_data.signature_id);
            cols.call_data.func_ref = F::from_canonical_u32(call_data.func_ref);
            cols.call_data.table_id = F::from_canonical_u32(call_data.table_id);
            cols.call_data.table_idx = F::from_canonical_u32(call_data.table_idx);
            cols.call_data.call_sp_is_zero = F::from_bool(event.call_sp == 0);
        }

        // Populate range checks for lo (and hi if 64-bit).
        let checks = if instruction.is_64b_op() {
            vec![cols.op_res_access, cols.op_res_hi_access]
        } else {
            vec![cols.op_res_access]
        };
        for col in checks {
            let bytes = col.access.value.0.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>();
            blu_events.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: bytes[0] as u8,
                c: bytes[1] as u8,
            });
            blu_events.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: bytes[2] as u8,
                c: bytes[3] as u8,
            });
        }
        // Assert that the instruction is not a no-op.
        cols.is_real = F::one();
    }

    /// Populates the shard and clk related rows.
    fn populate_shard_clk<F: PrimeField>(
        &self,
        cols: &mut CpuCols<F>,
        event: &CpuEvent,
        blu_events: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.shard = F::from_canonical_u32(shard);

        let clk_16bit_limb = (event.clk & 0xffff) as u16;
        let clk_8bit_limb = ((event.clk >> 16) & 0xff) as u8;
        cols.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
        cols.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);

        blu_events.add_byte_lookup_event(ByteLookupEvent::new(U16Range, shard as u16, 0, 0, 0));
        blu_events.add_byte_lookup_event(ByteLookupEvent::new(U16Range, clk_16bit_limb, 0, 0, 0));
        blu_events.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U8Range,
            0,
            0,
            0,
            clk_8bit_limb as u8,
        ));
    }

    fn populate_alu<F: PrimeField>(&self, cols: &mut CpuCols<F>, event: &CpuEvent, opcode: Opcode) {
        match opcode {
            Opcode::I32LtS | Opcode::I32GtS | Opcode::I32GeS | Opcode::I32LeS => {
                let alu = &mut cols.alu_cols;

                let arg1_i32 = event.arg1 as i32;
                let arg2_i32 = event.arg2 as i32;

                alu.arg1_eq_arg2 = F::from_bool(arg1_i32 == arg2_i32);
                alu.arg1_gt_arg2 = F::from_bool(arg1_i32 > arg2_i32);
                alu.arg1_lt_arg2 = F::from_bool(arg1_i32 < arg2_i32);
                alu.res_bool = F::from_canonical_u32(event.res);
            }
            Opcode::I32LtU |
            Opcode::I32GtU |
            Opcode::I32GeU |
            Opcode::I32LeU |
            Opcode::I32Eqz |
            Opcode::I32Eq |
            Opcode::I32Ne => {
                let alu = &mut cols.alu_cols;
                alu.arg1_eq_arg2 = F::from_bool(event.arg1 == event.arg2);
                alu.arg1_gt_arg2 = F::from_bool(event.arg1 > event.arg2);
                alu.arg1_lt_arg2 = F::from_bool(event.arg1 < event.arg2);
                alu.res_bool = F::from_canonical_u32(event.res);
            }
            _ => {}
        }
    }
}
