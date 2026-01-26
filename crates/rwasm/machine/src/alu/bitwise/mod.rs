use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::utils::pad_rows_fixed;
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use rwasm::{mem_index::UNIT, Opcode};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, EmptyByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{MachineAir, SP1AirBuilder},
    Word,
};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the opcodes XOR, OR, and AND.
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct BitwiseCols<T> {
    /// The program counter.
    pub pc: T,

    pub sp: T,

    /// The output operand.
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

    /// If the opcode is XOR.
    pub is_xor: T,

    /// If the opcode is OR.
    pub is_or: T,

    /// If the opcode is AND.
    pub is_and: T,
}

impl<F: PrimeField32> MachineAir<F> for BitwiseChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Bitwise".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = input
            .bitwise_events
            .par_iter()
            .map(|event| {
                let mut row = [F::zero(); NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();

                // Pass the "nil" recorder. The compiler will optimize away the lookup generation
                // logic.
                let mut blu = EmptyByteRecord;
                self.event_to_row(event, cols, &mut blu);
                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two.
        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_BITWISE_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_BITWISE_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bitwise_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    // Here we just create a dummy row to satisfy the signature.
                    // Since `BitwiseCols` is small, this stack allocation is cheap
                    // compared to the `Vec` allocation we avoided in `generate_trace`.
                    let mut row = [F::zero(); NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.bitwise_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BitwiseChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut BitwiseCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();

        cols.a = event.a.into();
        cols.b = event.b.into();
        cols.c = event.c.into();

        cols.is_xor = F::from_bool(event.opcode == Opcode::I32Xor);
        cols.is_or = F::from_bool(event.opcode == Opcode::I32Or);
        cols.is_and = F::from_bool(event.opcode == Opcode::I32And);

        if !blu.is_dummy() {
            for ((b_a, b_b), b_c) in a.into_iter().zip(b).zip(c) {
                blu.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::from(event.opcode),
                    a1: b_a as u16,
                    a2: 0,
                    b: b_b,
                    c: b_c,
                });
            }
        }
    }
}

impl<F> BaseAir<F> for BitwiseChip {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}

impl<AB> Air<AB> for BitwiseChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseCols<AB::Var> = (*local).borrow();

        let is_real = local.is_xor + local.is_or + local.is_and;

        // Get the opcode for the Byte Lookup Table.
        let opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>() +
            local.is_or * ByteOpcode::OR.as_field::<AB::F>() +
            local.is_and * ByteOpcode::AND.as_field::<AB::F>();

        // Send byte lookups.
        // Multiplicity is `is_real` (1 for valid rows, 0 for padding).
        for ((a, b), c) in local.a.into_iter().zip(local.b).zip(local.c) {
            builder.send_byte(opcode.clone(), a, b, c, is_real.clone());
        }

        // Get the CPU opcode.
        let cpu_opcode = local.is_xor * AB::Expr::from_canonical_u32(Opcode::I32Xor.code()) +
            local.is_or * AB::Expr::from_canonical_u32(Opcode::I32Or.code()) +
            local.is_and * AB::Expr::from_canonical_u32(Opcode::I32And.code());

        // Receive the arguments.
        // SAFETY: This checks the following.
        // - `next_pc = pc + 4`
        // - `num_extra_cycles = 0`
        // - `op_a_val` is constrained by the byte lookups when `op_a_not_0 == 1`
        // - `op_a_not_0` is correct, due to the sent `op_a_0` being equal to `1 - op_a_not_0`
        // - `op_a_immutable = 0`
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        // Note that `is_xor + is_or + is_and` is checked to be boolean below.
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            cpu_opcode,
            local.a,
            local.b,
            local.c,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Boolean constraints for selectors.
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(is_real);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};
    use rwasm::Opcode;
    use rwasm_executor::{
        events::{AluEvent, MemoryRecordEnum},
        ExecutionRecord, Program,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    use super::BitwiseChip;
    use crate::{
        alu::BitwiseCols,
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove, uni_stark_verify},
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.bitwise_events =
            vec![AluEvent::new(0, 0, Opcode::I32Xor, 25, 10, 19, Opcode::I32Xor.code())];
        let chip = BitwiseChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        assert_eq!(trace.height(), 16);
        assert_eq!(trace.width(), super::NUM_BITWISE_COLS);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mut shard = ExecutionRecord::default();
        shard.bitwise_events = [
            AluEvent::new(0, 0, Opcode::I32Xor, 25, 10, 19, Opcode::I32Xor.code()),
            AluEvent::new(0, 0, Opcode::I32Or, 27, 10, 19, Opcode::I32Or.code()),
            AluEvent::new(0, 0, Opcode::I32And, 2, 10, 19, Opcode::I32And.code()),
        ]
        .repeat(100);
        let chip = BitwiseChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = uni_stark_prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_bitwise() {
        use core::borrow::BorrowMut;
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;
        const NUM_TESTS: usize = 2;
        let mut rng = thread_rng();
        for &opcode in &[Opcode::I32Xor, Opcode::I32Or, Opcode::I32And] {
            for _ in 0..NUM_TESTS {
                let (op_b, op_c): (u32, u32) = (rng.gen(), rng.gen());
                let correct = match opcode {
                    Opcode::I32Xor => op_b ^ op_c,
                    Opcode::I32Or => op_b | op_c,
                    Opcode::I32And => op_b & op_c,
                    _ => unreachable!(),
                };
                let op_a = correct.wrapping_add(1);

                let program = Program::from_instrs(vec![
                    Opcode::I32Const(5u32.into()),
                    Opcode::I32Const(10u32.into()),
                    Opcode::I32Const(op_b.into()),
                    Opcode::I32Const(op_c.into()),
                    opcode,
                ]);
                let stdin = SP1Stdin::new();

                let malicious = move |prover: &P, record: &mut ExecutionRecord| {
                    let mut malicious_record = record.clone();
                    if malicious_record.cpu_events.len() > 4 {
                        if let Some(MemoryRecordEnum::Write(mut wr)) =
                            malicious_record.cpu_events[4].res_record
                        {
                            wr.value = op_a;
                            malicious_record.cpu_events[4].res_record =
                                Some(MemoryRecordEnum::Write(wr));
                        }
                    }

                    let chip = chip_name!(BitwiseChip, BabyBear);
                    let mut traces = prover.generate_traces(&malicious_record);
                    if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                        let row = trace.row_mut(0);
                        let row: &mut BitwiseCols<BabyBear> = row.borrow_mut();
                        row.a = op_a.into();
                    }
                    traces
                };

                let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
                assert!(matches!(&result, Err(e) if e.is_local_cumulative_sum_failing()));
            }
        }
    }
}
