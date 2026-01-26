use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use rwasm::{mem_index::UNIT, Opcode};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, EmptyByteRecord},
    ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{MachineAir, SP1AirBuilder},
    Word,
};

use crate::{
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
};

/// The number of main trace columns for `AddSubChip`.
pub const NUM_ADD_SUB_COLS: usize = size_of::<AddSubCols<u8>>();

/// A chip that implements addition for the opcode ADD and SUB.
///
/// SUB is basically an ADD with a re-arrangement of the operands and result.
/// E.g. given the standard ALU op variable name and positioning of `a` = `b` OP `c`,
/// `a` = `b` + `c` should be verified for ADD, and `b` = `a` + `c` (e.g. `a` = `b` - `c`)
/// should be verified for SUB.
#[derive(Default)]
pub struct AddSubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct AddSubCols<T> {
    /// The program counter.
    pub pc: T,

    /// The current stack pointer.
    pub sp: T,

    /// Instance of `AddOperation` to handle addition logic in `AddSubChip`'s ALU operations.
    /// It's result will be `a` for the add operation and `b` for the sub operation.
    pub add_operation: AddOperation<T>,

    /// The first input operand.  This will be `b` for add operations and `a` for sub operations.
    pub operand_1: Word<T>,

    /// The second input operand.  This will be `c` for both operations.
    pub operand_2: Word<T>,

    /// Boolean to indicate whether the row is for an add operation.
    pub is_add: T,

    /// Boolean to indicate whether the row is for a sub operation.
    pub is_sub: T,
}

impl<F: PrimeField32> MachineAir<F> for AddSubChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "AddSub".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.add_events.len() + input.sub_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the rows for the trace.
        let num_add_events = input.add_events.len();
        let total_events = num_add_events + input.sub_events.len();

        let chunk_size = std::cmp::max(total_events / num_cpus::get(), 1);
        let padded_nb_rows = <AddSubChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD_SUB_COLS);

        values.chunks_mut(chunk_size * NUM_ADD_SUB_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD_SUB_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddSubCols<F> = row.borrow_mut();

                    if idx < total_events {
                        // Avoid creating a merged vector. Select the source by index.
                        let event = if idx < num_add_events {
                            &input.add_events[idx]
                        } else {
                            &input.sub_events[idx - num_add_events]
                        };

                        // Pass the "nil" recorder. The compiler will optimize away the lookup
                        // generation logic.
                        let mut blu = EmptyByteRecord;
                        self.event_to_row(event, cols, &mut blu);
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_ADD_SUB_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size =
            std::cmp::max((input.add_events.len() + input.sub_events.len()) / num_cpus::get(), 1);

        let event_iter =
            input.add_events.chunks(chunk_size).chain(input.sub_events.chunks(chunk_size));

        let blu_batches = event_iter
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_ADD_SUB_COLS];
                    let cols: &mut AddSubCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.add_events.is_empty() || !shard.sub_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddSubChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut AddSubCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        let is_add = event.opcode == Opcode::I32Add;
        cols.is_add = F::from_bool(is_add);
        cols.is_sub = F::from_bool(event.opcode == Opcode::I32Sub);

        // b + c = a
        // b - c = a ==> a + c = b
        let operand_1 = if is_add { event.b } else { event.a };
        let operand_2 = event.c;

        cols.add_operation.populate(blu, operand_1, operand_2);
        cols.operand_1 = operand_1.into();
        cols.operand_2 = operand_2.into();
    }
}

impl<F> BaseAir<F> for AddSubChip {
    fn width(&self) -> usize {
        NUM_ADD_SUB_COLS
    }
}

impl<AB> Air<AB> for AddSubChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddSubCols<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_add` and `is_sub` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_add + is_sub` is
        // boolean. Therefore, the `opcode` matches the corresponding opcode of the
        // instruction.
        let is_real = local.is_add + local.is_sub;
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_sub);
        builder.assert_bool(is_real.clone());

        let opcode = AB::Expr::from_canonical_u32(Opcode::I32Add.code()) * local.is_add +
            AB::Expr::from_canonical_u32(Opcode::I32Sub.code()) * local.is_sub;

        // Evaluate the addition operation.
        // This is enforced only when `is_real` is true.
        AddOperation::<AB::F>::eval(
            builder,
            local.operand_1,
            local.operand_2,
            local.add_operation,
            is_real,
        );

        // Receive the arguments.  There are separate receives for ADD and SUB.
        // For add, `add_operation.value` is `a`, `operand_1` is `b`, and `operand_2` is `c`.
        // SAFETY: This checks the following. Note that in this case `opcode = Opcode::ADD`
        // - `next_pc = pc + 4`
        // - `num_extra_cycles = 0`
        // - `op_a_val` is constrained by the `AddOperation`
        // - `op_a_immutable = 0`
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            opcode.clone(),
            local.add_operation.value,
            local.operand_1,
            local.operand_2,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_add,
        );

        // For sub, `operand_1` is `a`, `add_operation.value` is `b`, and `operand_2` is `c`.
        // SAFETY: This checks the following. Note that in this case `opcode = Opcode::SUB`
        // - `next_pc = pc + 4`
        // - `num_extra_cycles = 0`
        // - `op_a_val` is constrained by the `AddOperation`
        // - `op_a_immutable = 0`
        // - `is_memory = 0`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            opcode,
            local.operand_1,
            local.add_operation.value,
            local.operand_2,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_sub,
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};
    use rwasm::Opcode;
    use rwasm_executor::{events::AluEvent, ExecutionRecord, DEFAULT_PC_INC, SP_START};
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };
    use std::sync::LazyLock;

    use super::*;
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use core::borrow::Borrow;
    use rwasm_executor::events::MemoryRecordEnum;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_events =
            vec![AluEvent::new(0, SP_START, Opcode::I32Add, 14, 8, 6, Opcode::I32Add.code())];
        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values)
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mut current_sp = SP_START - 2 * UNIT;

        let mut shard = ExecutionRecord::default();
        for i in 0..1 {
            let operand_1 = thread_rng().gen_range(0..u32::MAX);
            let operand_2 = thread_rng().gen_range(0..u32::MAX);
            let result = operand_1.wrapping_add(operand_2);
            shard.add_events.push(AluEvent::new(
                i * DEFAULT_PC_INC,
                current_sp,
                Opcode::I32Add,
                result,
                operand_1,
                operand_2,
                Opcode::I32Add.code(),
            ));
        }
        for i in 0..255 {
            let operand_1 = thread_rng().gen_range(0..u32::MAX);
            let operand_2 = thread_rng().gen_range(0..u32::MAX);
            let result = operand_1.wrapping_sub(operand_2);
            shard.add_events.push(AluEvent::new(
                i * DEFAULT_PC_INC,
                current_sp,
                Opcode::I32Sub,
                result,
                operand_1,
                operand_2,
                Opcode::I32Sub.code(),
            ));
        }

        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    // Helper: convert a 4-byte little-endian Word<F> back to u32.
    fn word_to_u32<F: PrimeField32>(w: &Word<F>) -> u32 {
        let limbs = &w.0;
        limbs[0].as_canonical_u32() |
            (limbs[1].as_canonical_u32() << 8) |
            (limbs[2].as_canonical_u32() << 16) |
            (limbs[3].as_canonical_u32() << 24)
    }

    #[test]
    fn row_encodes_add_correctly() {
        // a = b + c
        let b: u32 = 8;
        let c: u32 = 6;
        let a = b.wrapping_add(c);

        let mut current_sp = SP_START - 2 * UNIT;

        let mut shard = ExecutionRecord::default();
        shard.add_events.push(AluEvent::new(
            0,
            current_sp,
            Opcode::I32Add,
            a, // result 'a'
            b, // operand_1 (b for add)
            c, // operand_2 (c)
            Opcode::I32Add.code(),
        ));

        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // Read row 0 as columns.
        let row0 = &trace.values[0..NUM_ADD_SUB_COLS];
        let cols: &AddSubCols<BabyBear> = row0.borrow();

        // Flags and pc
        assert_eq!(cols.is_add, BabyBear::one());
        assert_eq!(cols.is_sub, BabyBear::zero());
        assert_eq!(cols.pc.as_canonical_u32(), 0);

        // Operands and computed value
        assert_eq!(word_to_u32(&cols.operand_1), b);
        assert_eq!(word_to_u32(&cols.operand_2), c);
        assert_eq!(word_to_u32(&cols.add_operation.value), a);
    }

    #[test]
    fn row_encodes_sub_correctly() {
        // For SUB: b = a + c (since a = b - c)
        let a: u32 = 10;
        let c: u32 = 7;
        let b = a.wrapping_add(c);

        let mut current_sp = SP_START - 2 * UNIT;

        let mut shard = ExecutionRecord::default();
        shard.sub_events.push(AluEvent::new(
            0,
            current_sp,
            Opcode::I32Sub,
            a, // 'a' for sub
            a, // operand_1 is 'a' for sub rows
            c, // operand_2 is 'c'
            Opcode::I32Sub.code(),
        ));

        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // Read row 0 as columns.
        let row0 = &trace.values[0..NUM_ADD_SUB_COLS];
        let cols: &AddSubCols<BabyBear> = row0.borrow();

        // Flags and pc
        assert_eq!(cols.is_add, BabyBear::zero());
        assert_eq!(cols.is_sub, BabyBear::one());
        assert_eq!(cols.pc.as_canonical_u32(), 0);

        // Operands and computed value
        // For sub rows, operand_1 is 'a' and operand_2 is 'c', and add_operation.value equals 'b'.
        assert_eq!(word_to_u32(&cols.operand_1), a);
        assert_eq!(word_to_u32(&cols.operand_2), c);
        assert_eq!(word_to_u32(&cols.add_operation.value), b);
    }

    #[test]
    fn test_malicious_add_sub() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let mut rng = thread_rng();

        for &opcode in &[Opcode::I32Add, Opcode::I32Sub] {
            let (op_b, op_c): (u32, u32) = (rng.gen(), rng.gen());
            let correct = match opcode {
                Opcode::I32Add => op_b.wrapping_add(op_c),
                Opcode::I32Sub => op_b.wrapping_sub(op_c),
                _ => unreachable!(),
            };
            let op_a = correct.wrapping_add(16); // force an incorrect result

            // stack: 5, 10,op_b, op_c, then <add|sub>, then a final add
            let program = Program::from_instrs(vec![
                Opcode::I32Const(5u32.into()),
                Opcode::I32Const(10u32.into()),
                Opcode::I32Const(op_b.into()),
                Opcode::I32Const(op_c.into()),
                opcode,
                Opcode::I32Add,
            ]);
            let stdin = SP1Stdin::new();

            let malicious = move |prover: &P, record: &mut ExecutionRecord| {
                let mut rec = record.clone();

                // The ALU op of interest is the 5th instruction (index 4)
                if rec.cpu_events.len() > 4 {
                    let evt = &mut rec.cpu_events[4];

                    // Keep memory trace consistent with our forged result
                    if let Some(MemoryRecordEnum::Write(mut wr)) = evt.res_record.take() {
                        wr.value = op_a;
                        evt.res_record = Some(MemoryRecordEnum::Write(wr));
                    }
                }

                // Corrupt the corresponding to add/sub micro-event
                match opcode {
                    Opcode::I32Add => {
                        if let Some(add) = rec.add_events.get_mut(0) {
                            add.a = op_a;
                        }
                    }
                    Opcode::I32Sub => {
                        if let Some(sub) = rec.sub_events.get_mut(0) {
                            sub.a = op_a;
                        }
                    }
                    _ => unreachable!(),
                }

                // Generate traces, then poison the AddSubChip row to ensure constraint failure
                let mut traces = prover.generate_traces(&rec);
                let chip = chip_name!(AddSubChip, BabyBear);
                if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                    // add events come before sub events
                    let idx = if matches!(opcode, Opcode::I32Add) { 0 } else { 1 };
                    let row = trace.row_mut(idx);
                    let row: &mut AddSubCols<BabyBear> = row.borrow_mut();
                    row.add_operation.value = op_a.into(); // inject the forged value
                }

                traces
            };

            let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
            let chip = chip_name!(AddSubChip, BabyBear);
            assert!(matches!(result, Err(e) if e.is_constraints_failing(&chip)));
        }
    }
}
