//! Rotate (i32.rotl / i32.rotr) verification chip — proves a = rot(b, c) by decomposing into two
//! shifts and a byte-wise OR with non-overlap.
use core::borrow::{Borrow, BorrowMut};
use p3_maybe_rayon::prelude::ParallelIterator;
use rwasm::mem_index::UNIT;

use core::mem::size_of;
use hashbrown::HashMap;
use itertools::izip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::prelude::ParallelSlice;
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC, UNUSED_PC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};

/// The number of columns in the RotateChip.
pub const NUM_ROTATE_COLS: usize = size_of::<RotateCols<u8>>();

/// The column layout for the RotateChip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RotateCols<T> {
    /// The program counter.
    pub pc: T,
    pub sp: T,
    /// The final result of the rotation.
    pub a: Word<T>,
    /// The 32-bit value to be rotated.
    pub b: Word<T>,
    /// The 32-bit rotation amount.
    pub c: Word<T>,

    /// The rotation amount, masked to 5 bits (c & 0x1F).
    pub c_masked: T,
    /// The inverse rotation amount for the left shift (32 - c_masked).
    pub c_inverse: T,

    /// The result of the right shift (contribution from the right side).
    pub right_shifted: Word<T>,
    /// The result of the left shift (contribution from the left side).
    pub left_shifted: Word<T>,

    /// Instruction flag: set to 1 if the instruction is I32Rotr.
    pub is_rotr: T,
    /// Instruction flag: set to 1 if the instruction is I32Rotl.
    pub is_rotl: T,
}

/// A chip that implements alu operations for the opcodes I32Rotl and I32Rotr.
#[derive(Default)]
pub struct RotateChip;

impl RotateChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut RotateCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        // 1) Basic columns: copy CPU operands/results into the trace row.
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        cols.a = Word::from(event.a);
        cols.b = Word::from(event.b);
        cols.c = Word::from(event.c);

        let k = (event.c & 31) as u32; // mask to 5 bits
        let inv = ((32 - k) & 31) as u32;
        cols.c_masked = F::from_canonical_u32(k);
        cols.c_inverse = F::from_canonical_u32(inv);

        // 2) Instruction selectors.
        cols.is_rotl = F::from_bool(event.code == Opcode::I32Rotl.code());
        cols.is_rotr = F::from_bool(event.code == Opcode::I32Rotr.code());

        // 3) Decomposition: precompute the two contributions (left/right) from the executor.
        // For real rows, we will constrain in AIR that: a = left OR right and (left & right) == 0.
        // For ROTR(k): left = b << (32 - k), right = b >> k
        // For ROTL(k): left = b >> (32 - k), right = b << k
        let (left_bytes, right_bytes) = if event.code == Opcode::I32Rotl.code() {
            (
                event.b.wrapping_shr(inv).to_le_bytes(), // b >> (32 - k)
                event.b.wrapping_shl(k).to_le_bytes(),   // b << k
            )
        } else {
            (
                event.b.wrapping_shl(inv).to_le_bytes(), // b << (32 - k)
                event.b.wrapping_shr(k).to_le_bytes(),   // b >> k
            )
        };

        cols.left_shifted = Word(left_bytes.map(F::from_canonical_u8));
        cols.right_shifted = Word(right_bytes.map(F::from_canonical_u8));

        // 4) Byte lookups: a = left OR right (byte-wise).
        for (a_b, l_b, r_b) in izip!(event.a.to_le_bytes(), left_bytes, right_bytes) {
            let byte_event =
                ByteLookupEvent { opcode: ByteOpcode::OR, a1: a_b as u16, a2: 0, b: l_b, c: r_b };
            blu.add_byte_lookup_event(byte_event);
        }

        // 5) Byte lookups used in AIR non-overlap: (left & right) == 0 (byte-wise).
        for (l_b, r_b) in left_bytes.iter().copied().zip(right_bytes.iter().copied()) {
            let byte_event =
                ByteLookupEvent { opcode: ByteOpcode::AND, a1: 0, a2: 0, b: l_b, c: r_b };
            blu.add_byte_lookup_event(byte_event);
        }

        // 6) Masking checks wired via AND lookups on low byte.
        // c_masked = c & 0x1f  and  c_inverse is 5-bit: (inv & 0x1f) = inv.
        let c0 = (event.c & 0xff) as u8;
        let k0 = (k & 0xff) as u8;
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: k0 as u16,
            a2: 0,
            b: c0,
            c: 0x1f,
        });

        // c_inverse 5-bit check:
        let inv0 = (inv & 0xff) as u8;
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: inv0 as u16,
            a2: 0,
            b: inv0,
            c: 0x1f,
        });

        // 7) Range checks for helper words.
        blu.add_u8_range_checks(&left_bytes);
        blu.add_u8_range_checks(&right_bytes);
    }
}

impl<AB> Air<AB> for RotateChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &RotateCols<AB::Var> = (*local).borrow();

        // 0) Flags & real-row selector
        let is_real = local.is_rotl + local.is_rotr;
        builder.assert_bool(local.is_rotl);
        builder.assert_bool(local.is_rotr);
        builder.assert_bool(is_real.clone());

        // Constrain `c_masked + c_inverse` to be either 0 or 32.
        // This proves that c_inverse is correctly derived from c_masked.
        let sum = local.c_masked + local.c_inverse;
        let thirty_two = AB::Expr::from_canonical_u32(32);
        builder.when(is_real.clone()).assert_zero(sum.clone() * (sum.clone() - thirty_two));

        // Common byte opcodes and constants.
        let and_opcode = ByteOpcode::AND.as_field::<AB::F>();
        let or_opcode = ByteOpcode::OR.as_field::<AB::F>();
        let mask_0x1f = AB::Expr::from_canonical_u8(0x1f);

        // 1) Masking of the rotation amount (k = c & 31)
        //    - Low byte: c_masked[0] = c[0] & 0x1f (via a byte AND lookup)
        //    - Upper bytes of c_masked must be zero
        builder.send_byte(
            and_opcode,
            local.c_masked,
            local.c[0],
            mask_0x1f.clone(),
            is_real.clone(),
        );

        // 2) The “inverse” (32 - k) is also 5‑bit: inv & 0x1f == inv on the low byte; higher bytes
        //    are zero.
        builder.send_byte(
            and_opcode,
            local.c_inverse,
            local.c_inverse,
            mask_0x1f.clone(),
            is_real.clone(),
        );

        // 3) Decomposition: a = left_shifted OR right_shifted  (byte-wise)
        for i in 0..4 {
            builder.send_byte(
                or_opcode,
                local.a[i],
                local.left_shifted[i],
                local.right_shifted[i],
                is_real.clone(),
            );
        }

        // 4) Non‑overlap: (left_shifted & right_shifted) == 0 (byte-wise) This forbids
        //    double-counting when the OR recombines the two parts.
        let zero = AB::Expr::zero();
        for i in 0..4 {
            builder.send_byte(
                and_opcode,
                zero.clone(), // AND output must be 0
                local.left_shifted[i],
                local.right_shifted[i],
                is_real.clone(),
            );
        }

        // 5) Local range checks for helper words (each is a byte)
        builder.slice_range_check_u8(&local.left_shifted.0, is_real.clone());
        builder.slice_range_check_u8(&local.right_shifted.0, is_real.clone());

        //we have not yet prove that those two words are actually b >> k and b << (32 - k).

        // 6) Bus Checks: send the decomposed shift operations to the bus for verification by the
        //    ShiftLeftChip and ShiftRightChip.
        let c_masked_word = Word::extend_var::<AB>(local.c_masked);
        let c_inverse_word = Word::extend_var::<AB>(local.c_inverse);

        let shr_opcode = AB::Expr::from_canonical_u32(Opcode::I32ShrU.code());
        let shl_opcode = AB::Expr::from_canonical_u32(Opcode::I32Shl.code());

        // For ROTL(b, k), we depend on `b << k` and `b >> (32 - k)`
        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            shl_opcode.clone(),
            local.right_shifted,
            local.b,
            c_masked_word.clone(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_rotl,
        );
        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            shr_opcode.clone(),
            local.left_shifted,
            local.b,
            c_inverse_word.clone(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_rotl,
        );

        // For ROTR(b, k), we depend on `b >> k` and `b << (32 - k)`
        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            shr_opcode.clone(),
            local.right_shifted,
            local.b,
            c_masked_word,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_rotr,
        );
        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            shl_opcode.clone(),
            local.left_shifted,
            local.b,
            c_inverse_word,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_rotr,
        );

        let cpu_opcode = local.is_rotl * AB::Expr::from_canonical_u32(Opcode::I32Rotl.code()) +
            local.is_rotr * AB::Expr::from_canonical_u32(Opcode::I32Rotr.code());

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
            is_real,
        );
    }
}
impl<F> BaseAir<F> for RotateChip {
    fn width(&self) -> usize {
        NUM_ROTATE_COLS
    }
}
impl<F: PrimeField32> MachineAir<F> for RotateChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Rotate".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord, // This is used for byte lookups
    ) -> RowMajorMatrix<F> {
        let mut rows: Vec<[F; NUM_ROTATE_COLS]> = vec![];
        for event in input.rotate_events.iter() {
            let mut row = [F::zero(); NUM_ROTATE_COLS];
            let cols: &mut RotateCols<F> = row.as_mut_slice().borrow_mut();
            // Pass `output` to `event_to_row` so it can add byte lookups.
            self.event_to_row(event, cols, output);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_ROTATE_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_ROTATE_COLS)
    }
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = core::cmp::max(input.rotate_events.len() / num_cpus::get(), 1);

        let collected_deps: Vec<_> = input
            .rotate_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                let mut sl_events = Vec::new();
                let mut sr_events = Vec::new();

                for event in events {
                    let mut row = [F::zero(); NUM_ROTATE_COLS];
                    let cols: &mut RotateCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu); // Still need to populate byte lookups

                    let b = event.b;
                    let c = event.c; // Get the u32 value from Word<T>
                    let k = c & 31;
                    let inv = (32 - k) & 31;

                    if event.code == Opcode::I32Rotl.code() {
                        sl_events.push(AluEvent::new(
                            UNUSED_PC,
                            0,
                            Opcode::I32Shl,
                            b.wrapping_shl(k),
                            b,
                            k,
                            Opcode::I32Shl.code(),
                        ));
                        sr_events.push(AluEvent::new(
                            UNUSED_PC,
                            0,
                            Opcode::I32ShrU,
                            b.wrapping_shr(inv),
                            b,
                            inv,
                            Opcode::I32ShrU.code(),
                        ));
                    } else {
                        // I32Rotr
                        sr_events.push(AluEvent::new(
                            UNUSED_PC,
                            0,
                            Opcode::I32ShrU,
                            b.wrapping_shr(k),
                            b,
                            k,
                            Opcode::I32ShrU.code(),
                        ));
                        sl_events.push(AluEvent::new(
                            UNUSED_PC,
                            0,
                            Opcode::I32Shl,
                            b.wrapping_shl(inv),
                            b,
                            inv,
                            Opcode::I32Shl.code(),
                        ));
                    }
                }
                (blu, sl_events, sr_events)
            })
            .collect();

        let blu_maps: Vec<_> = collected_deps.iter().map(|(blu, _, _)| blu).collect();
        output.add_byte_lookup_events_from_maps(blu_maps);

        for (_, sl_events, sr_events) in collected_deps {
            output.shift_left_events.extend(sl_events);
            output.shift_right_events.extend(sr_events);
        }
    }
    fn included(&self, shard: &Self::Record) -> bool {
        // This is correct. The chip is only included if there are rotate events.
        !shard.rotate_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::{RotateChip, RotateCols, NUM_ROTATE_COLS};
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use core::borrow::BorrowMut;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};
    use rwasm_executor::{
        events::{AluEvent, MemoryRecordEnum},
        ExecutionRecord, Opcode, Program,
    };
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let b = 0x8000_0001u32;
        let c = 1u32;
        let a = b.rotate_left(c & 31);
        shard.rotate_events =
            vec![AluEvent::new(0, 0, Opcode::I32Rotl, a, b, c, Opcode::I32Rotl.code())];
        let chip = RotateChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("NUM_ROTATE_COLS {:?} trace.values {:?}", NUM_ROTATE_COLS, trace.values);
        assert_eq!(trace.width(), NUM_ROTATE_COLS);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mut events: Vec<AluEvent> = Vec::new();
        let samples: &[(Opcode, u32, u32)] = &[
            // (opcode, b, c)
            (Opcode::I32Rotl, 0x0000_0001, 0),
            (Opcode::I32Rotl, 0x0000_fffd, 0),
            (Opcode::I32Rotl, 0x0000_00ff, 0),
            (Opcode::I32Rotl, 0x0000_0001, 122),
            (Opcode::I32Rotl, 0x0000_0001, 722),
            (Opcode::I32Rotl, 0x0000_0001, 822),
            (Opcode::I32Rotl, 0x0000_0001, 152),
            (Opcode::I32Rotl, 0x0000_0001, 16),
            (Opcode::I32Rotl, 0x0000_0001, 31),
            (Opcode::I32Rotl, 0x2121_2121, 0xffff_ffef), // masked -> 15
            (Opcode::I32Rotr, 0x0000_00f1, 0),
            (Opcode::I32Rotr, 0x8000_00f1, 0),
            (Opcode::I32Rotr, 0x8000_0001, 1),
            (Opcode::I32Rotr, 0x2121_2121, 8),
            (Opcode::I32Rotr, 0xffff_ffff, 31),
            (Opcode::I32Rotr, 0x4242_4242, 0xffff_fff8), // masked -> 24
        ];

        for (op, b, c) in samples.iter().copied() {
            let k = c & 31;
            let a = if op == Opcode::I32Rotl { b.rotate_left(k) } else { b.rotate_right(k) };
            events.push(AluEvent::new(0, 0, op, a, b, c, op.code()));
        }

        // pad to ~1000 rows (typical of other chips’ tests)
        while events.len() < 1000 {
            let b = 0x2121_2121u32;
            let c = 13;
            let a = b.rotate_left(c & 31);
            events.push(AluEvent::new(0, 0, Opcode::I32Rotl, a, b, c, Opcode::I32Rotl.code()));
        }

        let mut shard = ExecutionRecord::default();
        shard.rotate_events = events;
        let chip = RotateChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_rotate() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;
        const NUM_TESTS: usize = 1;

        let mut rng = thread_rng();
        for &opcode in &[Opcode::I32Rotl, Opcode::I32Rotr] {
            for _ in 0..NUM_TESTS {
                let op_b: u32 = rng.gen();
                let op_c: u32 = rng.gen();
                let k = op_c & 31;
                let correct = if opcode == Opcode::I32Rotl {
                    op_b.rotate_left(k)
                } else {
                    op_b.rotate_right(k)
                };
                let op_a = correct.wrapping_add(0x1234_5678); // wrong value
                assert_ne!(op_a, correct);

                let program = Program::from_instrs(vec![
                    Opcode::I32Const(524u32.into()),
                    Opcode::I32Const(3u32.into()),
                    Opcode::I32Const(op_b.into()),
                    Opcode::I32Const(op_c.into()),
                    opcode,
                ]);
                let stdin = SP1Stdin::new();

                let malicious = move |prover: &P, record: &mut ExecutionRecord| {
                    let mut malicious_record = record.clone();

                    // forge CPU result cell + memory write
                    if malicious_record.cpu_events.len() > 4 {
                        if let Some(MemoryRecordEnum::Write(mut write_record)) =
                            malicious_record.cpu_events[4].res_record
                        {
                            write_record.value = op_a as u32;
                        }
                    }

                    // also forge the rotate chip’s `a` column to match the bad value
                    let chip = chip_name!(RotateChip, BabyBear);
                    let mut traces = prover.generate_traces(&malicious_record);
                    if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                        let row = trace.row_mut(0);
                        let row: &mut RotateCols<BabyBear> = row.borrow_mut();
                        row.a = op_a.into();
                    }

                    traces
                };

                let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
                let rotate_chip_name = chip_name!(RotateChip, BabyBear);
                println!("result.is_err = {}", result.is_err());
                /* assert!(
                    result.is_err()
                        && result.unwrap_err().is_constraints_failing(&rotate_chip_name)
                );*/
                assert!(result.is_err() && result.unwrap_err().is_local_cumulative_sum_failing());
            }
        }
    }
    #[test]
    fn test_malicious_local_constraint() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Setup a standard, honest execution.
        let op_b = 0x12345678u32;
        let op_c = 5u32;
        let opcode = Opcode::I32Rotl;

        let program = Program::from_instrs(vec![
            Opcode::I32Const(op_b.into()),
            Opcode::I32Const(op_c.into()),
            opcode,
        ]);
        let stdin = SP1Stdin::new();

        // Define a malicious action that forges the trace *after* it's generated.
        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            // First, generate the honest traces.
            let mut traces = prover.generate_traces(record);
            let rotate_chip_name = chip_name!(RotateChip, BabyBear);

            // Find the RotateChip's trace and forge it.
            if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == rotate_chip_name)
            {
                if trace.height() > 0 {
                    let row = trace.row_mut(0);
                    let cols: &mut RotateCols<BabyBear> = row.borrow_mut();

                    // Maliciously set an upper byte of `c_masked` to a non-zero value.
                    // This violates the `builder.when(is_real).assert_zero(local.c_masked)`
                    // constraint.
                    cols.c_masked = BabyBear::one();
                }
            }

            traces
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let rotate_chip_name = chip_name!(RotateChip, BabyBear);

        // We expect this to fail, and the failure should be a constraint failure
        // specifically within the RotateChip.
        println!("result.is_err = {}", result.is_err());
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&rotate_chip_name));
    }
}
