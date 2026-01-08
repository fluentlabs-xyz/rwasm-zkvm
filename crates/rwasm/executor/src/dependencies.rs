use std::cmp;

use rwasm::{is_multi_align, mem_index::UNIT, Opcode};

use crate::{
    events::{AluEvent, BranchEvent, FuelEvent, I64AluEvent, MemInstrEvent},
    Executor, UNUSED_PC,
};
use rwasm::mem_index::GLOBAL_MEM_START;
/*/// Emits the dependencies for division and remainder operations.
#[allow(clippy::too_many_lines)]
pub fn emit_divrem_dependencies(executor: &mut Executor, event: AluEvent) {
    let (quotient, remainder) = get_quotient_and_remainder(event.b, event.c, event.opcode);
    let c_msb = get_msb(event.c);
    let rem_msb = get_msb(remainder);
    let mut c_neg = 0;
    let mut rem_neg = 0;
    let is_signed_operation = is_signed_operation(event.opcode);
    if is_signed_operation {
        c_neg = c_msb; // same as abs_c_alu_event
        rem_neg = rem_msb; // same as abs_rem_alu_event
    }

    if c_neg == 1 {
        executor.record.add_events.push(AluEvent {
            pc: UNUSED_PC,
            sp: 0,
            opcode: Opcode::I32Add,
            a: 0,
            b: event.c,
            c: (event.c as i32).unsigned_abs(),
            code: Opcode::I32Add.code(),
        });
    }
    if rem_neg == 1 {
        executor.record.add_events.push(AluEvent {
            pc: UNUSED_PC,
            sp: 0,
            opcode: Opcode::I32Add,
            a: 0,
            b: remainder,
            c: (remainder as i32).unsigned_abs(),
            code: Opcode::I32Add.code(),
        });
    }

    let c_times_quotient = {
        if is_signed_operation {
            (((quotient as i32) as i64) * ((event.c as i32) as i64)).to_le_bytes()
        } else {
            ((quotient as u64) * (event.c as u64)).to_le_bytes()
        }
    };
    let lower_word = u32::from_le_bytes(c_times_quotient[0..4].try_into().unwrap());
    let upper_word = u32::from_le_bytes(c_times_quotient[4..8].try_into().unwrap());

    let lower_multiplication = AluEvent {
        pc: UNUSED_PC,
        sp: 0,
        opcode: Opcode::I32Mul,
        a: lower_word,
        c: event.c,
        b: quotient,
        code: Opcode::I32Mul.code(),
    };
    executor.record.mul_events.push(lower_multiplication);

    let upper_multiplication = AluEvent {
        pc: UNUSED_PC,
        sp: 0,
        opcode: Opcode::I32Mul,
        a: upper_word,
        c: event.c,
        b: quotient,
        code: {
            if is_signed_operation {
                I32MULH_CODE
            } else {
                I32MULHU_CODE
            }
        },
    };
    executor.record.mul_events.push(upper_multiplication);

    let lt_event = if is_signed_operation {
        AluEvent {
            pc: UNUSED_PC,
            sp: 0,
            opcode: Opcode::I32LtU,
            a: 1,
            b: (remainder as i32).unsigned_abs(),
            c: u32::max(1, (event.c as i32).unsigned_abs()),
            code: Opcode::I32LtU.code(),
        }
    } else {
        AluEvent {
            pc: UNUSED_PC,
            sp: 0,
            opcode: Opcode::I32LtU,
            a: 1,
            b: remainder,
            c: u32::max(1, event.c),
            code: Opcode::I32LtU.code(),
        }
    };

    if event.c != 0 {
        executor.record.lt_events.push(lt_event);
    }
}
*/
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

pub fn emit_fuel_dependencies(executor: &mut Executor, event: FuelEvent) {
    let is_carryed = {
        let fuel_low = event.fuel as u32;
        fuel_low.checked_add(event.to_consume_fuel).is_none()
    };

    let add_low_event = I64AluEvent {
        pc: UNUSED_PC,
        opcode: Opcode::I32Add64,
        a_lo: event.next_fuel as u32,
        a_hi: is_carryed as u32,
        b: event.fuel as u32,
        c: event.to_consume_fuel,
        code: Opcode::I32Add64.code(),
        res_hi_addr: 0u32,
        res_hi_access: None,
    };
    executor.record.add64_events.push(add_low_event);

    let carry_value = if is_carryed { 1 } else { 0 };
    let add_high_event = I64AluEvent {
        pc: UNUSED_PC,
        opcode: Opcode::I32Add64,
        a_lo: (event.next_fuel >> 32) as u32,
        a_hi: carry_value,
        b: (event.fuel >> 32) as u32,
        c: carry_value,
        code: Opcode::I32Add64.code(),
        res_hi_addr: 0u32,
        res_hi_access: None,
    };
    executor.record.add64_events.push(add_high_event);
}
