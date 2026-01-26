use std::cmp;

use rwasm::{is_multi_align, mem_index::UNIT, Opcode};

use crate::{
    events::{AluEvent, BranchEvent, MemInstrEvent},
    Executor, UNUSED_PC,
};
use rwasm::mem_index::GLOBAL_MEM_START;

/// Emit the dependencies for memory opcodes.
pub fn emit_memory_dependencies(executor: &mut Executor, event: MemInstrEvent) {
    let offset = event.opcode.aux_value();
    let memory_addr = event.arg1.wrapping_add(offset);
    // Add event to ALU check to check that addr == b + c
    let add1_event = AluEvent {
        pc: UNUSED_PC,
        sp: 0,
        opcode: Opcode::I32Add,
        a: memory_addr,
        b: event.arg1,
        c: offset,
        code: Opcode::I32Add.code(),
    };

    let add2_event = AluEvent {
        pc: UNUSED_PC,
        sp: 0,
        opcode: Opcode::I32Add,
        a: memory_addr + GLOBAL_MEM_START,
        b: GLOBAL_MEM_START,
        c: memory_addr,
        code: Opcode::I32Add.code(),
    };

    executor.record.add_events.push(add1_event);
    executor.record.add_events.push(add2_event);

    let addr_offset = (memory_addr % UNIT) as u8;

    let mut mem_value = event.mem_access.value();
    if is_multi_align(event.opcode, memory_addr) {
        let val_lo = mem_value as u64;
        let val_hi = event.mem_access_hi.unwrap().value() as u64;
        // Combine [HI:LO] and shift right by offset bits (addr_offset * 8)
        mem_value = ((val_hi << 32 | val_lo) >> (addr_offset * 8)) as u32;
    }

    if matches!(event.opcode, Opcode::I32Load8S(_) | Opcode::I32Load16S(_)) {
        let (unsigned_mem_val, sign_bit, sign_value) = match event.opcode {
            Opcode::I32Load8S(_) => {
                let val = mem_value.to_le_bytes()[addr_offset as usize];
                (val as u32, (val >> 7) & 1, 256)
            }
            Opcode::I32Load16S(_) => {
                // Extract 16-bit half-word based on offset (0 or 2 bytes)
                let shift = (addr_offset & 2) * 8;
                let val = (mem_value >> shift) & 0xFFFF;
                (val, ((val >> 15) & 1) as u8, 65536)
            }
            _ => unreachable!(),
        };

        if sign_bit == 1 {
            executor.record.add_events.push(AluEvent {
                pc: UNUSED_PC,
                sp: 0,
                opcode: Opcode::I32Sub,
                a: event.res,
                b: unsigned_mem_val,
                c: sign_value,
                code: Opcode::I32Sub.code(),
            });
        }
    }
}
/// Emit the dependencies for branch opcodes.
pub fn emit_branch_dependencies(executor: &mut Executor, event: BranchEvent) {
    if event.opcode.is_branch_instruction() {
        let offset = event.opcode.aux_value();
        let a_eq_zero = event.arg1 == 0;
        let a_gt_zero = event.arg1 > 0;
        let a_lt_target = event.arg1 < event.opcode.aux_value() - 1;
        let cmp_ins = Opcode::I32LtU;
        // Add the ALU events for the comparisons

        if let Opcode::BrTable(_) = event.opcode {
            executor.record.lt_events.push(AluEvent {
                pc: UNUSED_PC,
                sp: 0,
                opcode: cmp_ins,
                a: a_lt_target as u32,
                b: event.arg1,
                c: event.opcode.aux_value() - 1,
                code: cmp_ins.code(),
            });
        }

        let branching = match event.opcode {
            Opcode::BrIfEqz(_) => a_eq_zero,
            Opcode::BrIfNez(_) => a_gt_zero,
            Opcode::Br(_) | Opcode::BrTable(_) => true,
            _ => unreachable!(),
        };

        if branching {
            if let Opcode::BrTable(_) = event.opcode {
                let index = event.arg1;
                let targets = event.opcode.aux_value();
                let max_index = targets as usize - 1;
                let normalized_index = cmp::min(index as usize, max_index);
                let offset = 2 * normalized_index + 1;

                let add_event = AluEvent {
                    pc: UNUSED_PC,
                    sp: 0,
                    opcode: Opcode::I32Add,
                    a: event.next_pc,
                    b: event.pc,
                    c: offset as u32,
                    code: Opcode::I32Add.code(),
                };
                executor.record.add_events.push(add_event);
            } else {
                let next_pc = (event.pc).wrapping_add(offset);
                let add_event = AluEvent {
                    pc: UNUSED_PC,
                    sp: 0,
                    opcode: Opcode::I32Add,
                    a: next_pc,
                    b: event.pc,
                    c: offset,
                    code: Opcode::I32Add.code(),
                };
                executor.record.add_events.push(add_event);
            }
        }
    }
}

pub fn emit_fuel_dependencies(executor: &mut Executor) {
    let consumed_fuel = executor.store.fuel_consumed();
    let fuel_limit = u64::MAX;

    let (b, c) = if consumed_fuel >> 32 == fuel_limit >> 32 {
        (consumed_fuel as u32, fuel_limit as u32)
    } else {
        ((consumed_fuel >> 32) as u32, (fuel_limit >> 32) as u32)
    };

    let leu_event = AluEvent {
        pc: UNUSED_PC,
        sp: 0,
        opcode: Opcode::I32LeU,
        a: 1,
        b,
        c,
        code: Opcode::I32LeU.code(),
    };

    executor.record.fuel_limit_leu_event = Some(leu_event);
}
