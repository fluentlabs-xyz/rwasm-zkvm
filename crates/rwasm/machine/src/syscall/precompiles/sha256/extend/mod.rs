mod air;
mod columns;
mod flags;
mod trace;

pub use columns::*;

/// Implements the SHA extension operation which loops over i = \[16, 63\] and modifies w\[i\] in
/// each iteration. The only input to the syscall is the 4byte-aligned pointer to the w array.
///
/// In the AIR, each SHA extend syscall takes up 48 rows, where each row corresponds to a single
/// iteration of the loop.
#[derive(Default)]
pub struct ShaExtendChip;

impl ShaExtendChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub fn sha_extend(w: &mut [u32]) {
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
}

#[cfg(test)]
pub mod extend_tests {
    #![allow(clippy::print_stdout)]

    /*
    pub fn sha_extend_program() -> Program {
        let w_ptr = 100;
        let mut instructions = vec![Opcode::new(Opcode::ADD, 29, 0, 5, false, true)];
        for i in 0..64 {
            instructions.extend(vec![
                Opcode::new(Opcode::ADD, 30, 0, w_ptr + i * 4, false, true),
                Opcode::new(Opcode::SW, 29, 30, 0, false, true),
            ]);
        }
        instructions.extend(vec![
            Opcode::new(Opcode::ADD, 5, 0, SyscallCode::SHA_EXTEND as u32, false, true),
            Opcode::new(Opcode::ADD, 10, 0, w_ptr, false, true),
            Opcode::new(Opcode::ADD, 11, 0, 0, false, true),
            Opcode::new(Opcode::ECALL, 5, 10, 11, false, false),
        ]);
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_events = vec![AluEvent::new(0, Opcode::ADD, 14, 8, 6, false)];
        let chip = ShaExtendChip::new();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values)
    }

    #[test]
    fn test_sha_prove() {
        utils::setup_logger();
        let program = sha_extend_program();
        let stdin = SP1Stdin::new();
        run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }*/
}
