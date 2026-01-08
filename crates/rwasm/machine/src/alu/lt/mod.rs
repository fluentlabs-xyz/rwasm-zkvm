use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::{izip, Itertools};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use rwasm::{mem_index::UNIT, Opcode};

use crate::{
    air::WordAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};
use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{MachineAir, SP1AirBuilder},
    Word,
};

use crate::{
    air::WordAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};

// The number of main trace columns for `LtChip`.
pub const NUM_LT_COLS: usize = size_of::<LtCols<u8>>();

// A chip that implements bitwise operations for the opcodes SLT and SLTU.
#[derive(Default)]
pub struct LtChip;

// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LtCols<T> {
    // The program counter.
    pub pc: T,

    // The stack pointer.
    pub sp: T,

    // If the operation is signed (S variants: I32LtS, I32LeS, I32GtS, I32GeS).
    pub is_signed: T,

    // If the opcode is I32LtS or I32LtU (less than).
    pub is_i32lt: T,

    // If the opcode is I32LeS or I32LeU (less or equal).
    pub is_i32le: T,

    // If the opcode is I32GtS or I32GtU (greater than).
    pub is_i32gt: T,

    // If the opcode is I32GeS or I32GeU (greater or equal).
    pub is_i32ge: T,

    // If the opcode is I32Eq (equal).
    pub is_i32eq: T,

    // If the opcode is I32Ne (not equal).
    pub is_i32ne: T,

    // If the opcode is I32Eqz (equal to zero).
    pub is_i32eqz: T,

    // The output operand (result of comparison: 0 or 1).
    pub a: T,

    // The first input operand.
    pub b: Word<T>,

    // The second input operand.
    pub c: Word<T>,

    // Boolean flags to indicate which byte pair differs if the operands are not equal.
    // Only one flag is set to 1, pointing to the most significant differing byte.
    pub byte_flags: [T; 4],

    // The masking b[3] & 0x7F (removes sign bit for signed comparisons).
    pub b_masked: T,

    // The masking c[3] & 0x7F (removes sign bit for signed comparisons).
    pub c_masked: T,

    // An inverse of differing byte if c_comp != b_comp.
    // Used to prove that comparison bytes are different: inv * (b - c) = 1.
    pub not_eq_inv: T,

    // The most significant bit of operand b (sign bit).
    pub msb_b: T,

    // The most significant bit of operand c (sign bit).
    pub msb_c: T,

    // The multiplication msb_b * is_signed (effective sign bit for comparison).
    pub bit_b: T,

    // The multiplication msb_c * is_signed (effective sign bit for comparison).
    pub bit_c: T,

    // The result of the intermediate SLTU operation `b_comp < c_comp`.
    // This is the unsigned less-than comparison on potentially masked values.
    pub sltu: T,

    // A boolean flag for an intermediate comparison (b_comp == c_comp).
    pub is_comp_eq: T,

    // A boolean flag for comparing the sign bits (bit_b == bit_c).
    // Always 1 for unsigned operations.
    pub is_sign_eq: T,

    // The comparison bytes to be looked up via ByteLookupEvent.
    // Contains the first differing byte pair [b_byte, c_byte].
    pub comparison_bytes: [T; 2],

    pub opcode: T,
}

impl LtCols<u32> {
    pub fn from_trace_row<F: PrimeField32>(row: &[F]) -> Self {
        let sized: [u32; NUM_LT_COLS] =
            row.iter().map(|x| x.as_canonical_u32()).collect::<Vec<u32>>().try_into().unwrap();
        *sized.as_slice().borrow()
    }
}

impl<F: PrimeField32> MachineAir<F> for LtChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Lt".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let nb_rows = input.lt_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LT_COLS);
        let chunk_size = core::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_LT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_LT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LtCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.lt_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_LT_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = core::cmp::max(input.lt_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .lt_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_LT_COLS];
                    let cols: &mut LtCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.lt_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LtChip {
    // Create a row from an event.
    //
    // This function processes an AluEvent and fills in the LtCols trace row.
    // It handles all comparison operations (LT, LE, GT, GE, EQ, NE, EQZ) for both
    // signed and unsigned variants.
    //
    // For GT/GE operations, operands are swapped to reuse LT logic:
    // - a > b becomes b < a (swap operands)
    // - a >= b becomes b <= a (swap operands)
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut LtCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        // Check if this is a comparison operation (vs equality operation).
        let is_comparison = matches!(event.code,
            code if code == Opcode::I32LtS.code()
                || code == Opcode::I32LtU.code()
                || code == Opcode::I32LeS.code()
                || code == Opcode::I32LeU.code()
                || code == Opcode::I32GtS.code()
                || code == Opcode::I32GtU.code()
                || code == Opcode::I32GeS.code()
                || code == Opcode::I32GeU.code()
        );

        // Determine if we need to swap operands (for GT/GE operations).
        // GT/GE are implemented by swapping operands and using LT/LE logic.
        let (b_val, c_val) = match event.code {
            code if code == Opcode::I32GtS.code() ||
                code == Opcode::I32GtU.code() ||
                code == Opcode::I32GeS.code() ||
                code == Opcode::I32GeU.code() =>
            {
                // For GT/GE: a > b becomes b < a (swap operands)
                (event.c, event.b)
            }
            _ => (event.b, event.c),
        };

        let a = event.a.to_le_bytes();
        let b = b_val.to_le_bytes();
        let c = c_val.to_le_bytes();

        cols.a = F::from_canonical_u8(a[0]);
        cols.b = Word(b.map(F::from_canonical_u8));
        cols.c = Word(c.map(F::from_canonical_u8));

        // Determine if this is a signed operation.
        let is_signed = matches!(event.code,
            code if code == Opcode::I32LtS.code()
                || code == Opcode::I32LeS.code()
                || code == Opcode::I32GtS.code()
                || code == Opcode::I32GeS.code()
        );

        cols.is_signed = F::from_bool(is_signed);

        // Set operation type flags based on the opcode.
        match event.code {
            code if code == Opcode::I32LtS.code() || code == Opcode::I32LtU.code() => {
                cols.is_i32lt = F::one();
            }
            code if code == Opcode::I32LeS.code() || code == Opcode::I32LeU.code() => {
                cols.is_i32le = F::one();
            }
            code if code == Opcode::I32GtS.code() || code == Opcode::I32GtU.code() => {
                cols.is_i32gt = F::one();
            }
            code if code == Opcode::I32GeS.code() || code == Opcode::I32GeU.code() => {
                cols.is_i32ge = F::one();
            }
            code if code == Opcode::I32Eq.code() => {
                cols.is_i32eq = F::one();
            }
            code if code == Opcode::I32Ne.code() => {
                cols.is_i32ne = F::one();
            }
            code if code == Opcode::I32Eqz.code() => {
                cols.is_i32eqz = F::one();
            }
            _ => {}
        }

        // === Comparison Operations (LT, LE, GT, GE) ===
        // The following code only executes for comparison operations.

        // Mask the most significant byte for signed operations.
        // For signed comparisons, we remove the sign bit: byte & 0x7F.
        let masked_b = b[3] & 0x7f;
        let masked_c = c[3] & 0x7f;
        cols.b_masked = F::from_canonical_u8(masked_b);
        cols.c_masked = F::from_canonical_u8(masked_c);

        // Send byte lookup events to constrain the masking operation.
        // This ensures b_masked = b[3] & 0x7F and c_masked = c[3] & 0x7F.

        if is_comparison {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::AND,
                a1: masked_b as u16,
                a2: 0,
                b: b[3],
                c: 0x7f,
            });
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::AND,
                a1: masked_c as u16,
                a2: 0,
                b: c[3],
                c: 0x7f,
            });
        }

        // Prepare comparison values based on operation type.
        // For unsigned operations: b_comp = b, c_comp = c
        // For signed operations: b_comp = b with sign bit masked, c_comp = c with sign bit masked
        let mut b_comp = b;
        let mut c_comp = c;
        if is_signed {
            b_comp[3] = masked_b;
            c_comp[3] = masked_c;
        }

        // Initial comparison: check if b_comp < c_comp and if they're equal.
        cols.sltu = F::from_bool(b_comp < c_comp);
        cols.is_comp_eq = F::from_bool(b_comp == c_comp);

        // Find the first (most significant) differing byte.
        // Iterate from MSB to LSB and set the corresponding byte_flag when difference is found.
        for (b_byte, c_byte, flag) in
            izip!(b_comp.iter().rev(), c_comp.iter().rev(), cols.byte_flags.iter_mut().rev())
        {
            if c_byte != b_byte {
                // Found the differing byte: set its flag and update SLTU result.
                *flag = F::one();
                cols.sltu = F::from_bool(b_byte < c_byte);

                // Store the comparison result for the byte lookup constraint.
                let b_byte = F::from_canonical_u8(*b_byte);
                let c_byte = F::from_canonical_u8(*c_byte);

                // Calculate the inverse for the inequality proof.
                // This will be used to prove that the bytes are indeed different.
                cols.not_eq_inv = (b_byte - c_byte).inverse();
                cols.comparison_bytes = [b_byte, c_byte];
                break;
            }
        }

        // Extract the sign bits (MSB) from both operands.
        cols.msb_b = F::from_canonical_u8((b[3] >> 7) & 1);
        cols.msb_c = F::from_canonical_u8((c[3] >> 7) & 1);

        // Check if sign bits are equal.
        // For unsigned operations, this is always 1.
        // For signed operations, it's 1 if both have the same sign bit.
        cols.is_sign_eq =
            if is_signed { F::from_bool((b[3] >> 7) == (c[3] >> 7)) } else { F::one() };

        // Compute effective sign bits for the comparison formula.
        // bit_b = msb_b * is_signed (0 for unsigned, MSB for signed)
        // bit_c = msb_c * is_signed (0 for unsigned, MSB for signed)
        cols.bit_b = cols.msb_b * cols.is_signed;
        cols.bit_c = cols.msb_c * cols.is_signed;

        // Compute the comparison result based on operation type.
        // The formula for signed less-than (from Jolt paper):
        // SLT = b_sign * (1 - c_sign) + (b_sign == c_sign) * SLTU(b_masked, c_masked)

        // Send byte lookup event for the SLTU (unsigned less-than) operation.
        // This constrains that sltu equals the correct unsigned comparison result.

        if is_comparison {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: cols.sltu.as_canonical_u32() as u16,
                a2: 0,
                b: cols.comparison_bytes[0].as_canonical_u32() as u8,
                c: cols.comparison_bytes[1].as_canonical_u32() as u8,
            });
        }

        cols.opcode = F::from_canonical_u32(event.opcode.code());
    }
}

impl<F> BaseAir<F> for LtChip {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }
}

impl<AB> Air<AB> for LtChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LtCols<AB::Var> = (*local).borrow();

        // Flag for comparison operations (LT, LE, GT, GE) that require sign/magnitude analysis.
        let is_comparison = local.is_i32lt + local.is_i32le + local.is_i32gt + local.is_i32ge;

        // Flag for equality operations (EQ, NE, EQZ) that only need equality check.
        let is_equality = local.is_i32eq + local.is_i32ne + local.is_i32eqz;

        let is_real = is_comparison.clone() + is_equality;

        builder.assert_bool(local.is_signed);
        builder.assert_bool(local.is_i32lt);
        builder.assert_bool(local.is_i32le);
        builder.assert_bool(local.is_i32gt);
        builder.assert_bool(local.is_i32ge);
        builder.assert_bool(local.is_i32eq);
        builder.assert_bool(local.is_i32ne);
        builder.assert_bool(local.is_i32eqz);
        builder.assert_bool(is_real.clone());

        builder.when(local.is_i32eqz).assert_word_zero(local.c);

        let mut b_comp: Word<AB::Expr> = local.b.map(|x| x.into());
        let mut c_comp: Word<AB::Expr> = local.c.map(|x| x.into());

        // Only mask the MSB for comparison operations.
        // For equality operations, we don't need the masked values.
        b_comp[3] =
            local.b[3] * (AB::Expr::one() - local.is_signed) + local.b_masked * local.is_signed;
        c_comp[3] =
            local.c[3] * (AB::Expr::one() - local.is_signed) + local.c_masked * local.is_signed;

        // Send byte lookups only for comparison operations.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.b_masked,
            local.b[3],
            AB::F::from_canonical_u8(0x7f),
            is_comparison.clone(),
        );
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.c_masked,
            local.c[3],
            AB::F::from_canonical_u8(0x7f),
            is_comparison.clone(),
        );

        builder.assert_eq(local.bit_b, local.msb_b * local.is_signed);
        builder.assert_eq(local.bit_c, local.msb_c * local.is_signed);

        // MSB extraction only needed for comparison operations.
        let inv_128 = AB::F::from_canonical_u32(128).inverse();
        builder
            .when(is_comparison.clone())
            .assert_eq(local.msb_b, (local.b[3] - local.b_masked) * inv_128);
        builder
            .when(is_comparison.clone())
            .assert_eq(local.msb_c, (local.c[3] - local.c_masked) * inv_128);

        builder.assert_bool(local.is_sign_eq);
        builder.when(local.is_sign_eq).assert_eq(local.bit_b, local.bit_c);
        builder
            .when(is_comparison.clone())
            .when_not(local.is_sign_eq)
            .assert_one(local.bit_b + local.bit_c);

        // Compute comparison result only for LT/LE/GT/GE operations.
        let lt_result =
            local.bit_b * (AB::Expr::one() - local.bit_c) + local.is_sign_eq * local.sltu;

        // Compute expected result based on operation type.
        // Equality operations (EQ/NE/EQZ) only use is_comp_eq.
        // Comparison operations (LT/LE/GT/GE) use lt_result combined with is_comp_eq.

        // LT/GT: less/greater than
        let expected_a = (local.is_i32lt + local.is_i32gt) * lt_result.clone()
        // LE/GE: less/greater or equal = LT/GT OR EQ
        + (local.is_i32le + local.is_i32ge) * (lt_result + local.is_comp_eq)
        // EQ/EQZ: equal
        + (local.is_i32eq + local.is_i32eqz) * local.is_comp_eq
        // NE: not equal
        + local.is_i32ne * (AB::Expr::one() - local.is_comp_eq);

        builder.assert_eq(local.a, expected_a);

        // Byte flags are only meaningful for comparison operations.
        // For equality operations, we don't strictly need to check them,
        // but we verify them for consistency.
        let sum_flags =
            local.byte_flags[0] + local.byte_flags[1] + local.byte_flags[2] + local.byte_flags[3];
        builder.assert_bool(sum_flags.clone());

        // For comparison operations: if operands not equal, exactly one byte flag set.
        // For equality operations: this constraint is automatically satisfied since
        // is_comp_eq already reflects equality.
        builder
            .when(is_comparison.clone())
            .assert_eq(AB::Expr::one() - local.is_comp_eq, sum_flags);

        builder.assert_bool(local.is_comp_eq);

        let mut is_inequality_visited = AB::Expr::zero();
        let mut b_comparison_byte = AB::Expr::zero();
        let mut c_comparison_byte = AB::Expr::zero();

        // Only process byte comparison loop for comparison operations.
        for (b_byte, c_byte, &flag) in
            izip!(b_comp.0.iter().rev(), c_comp.0.iter().rev(), local.byte_flags.iter().rev())
        {
            is_inequality_visited = is_inequality_visited.clone() + flag.into();
            b_comparison_byte = b_comparison_byte.clone() + b_byte.clone() * flag;
            c_comparison_byte = c_comparison_byte.clone() + c_byte.clone() * flag;

            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_byte.clone(), c_byte.clone());

            builder.when(local.is_comp_eq).assert_zero(is_inequality_visited.clone());
        }

        let (b_comp_byte, c_comp_byte) = (local.comparison_bytes[0], local.comparison_bytes[1]);

        builder.assert_eq(b_comp_byte, b_comparison_byte);
        builder.assert_eq(c_comp_byte, c_comparison_byte);

        builder
            .when(sum_flags.clone())
            .assert_eq(local.not_eq_inv * (b_comp_byte - c_comp_byte), is_real.clone());

        // Send byte lookup only for comparison operations.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            local.sltu,
            b_comp_byte,
            c_comp_byte,
            is_comparison.clone(),
        );

        // Verify that the opcode matches the operation flags.
        // We check that exactly one flag combination is set and corresponds to the opcode.
        // For comparison operations, verify is_signed matches the opcode.
        builder.when(local.is_i32lt).assert_eq(
            local.opcode,
            local.is_signed * AB::F::from_canonical_u32(Opcode::I32LtS.code()) +
                (AB::Expr::one() - local.is_signed) *
                    AB::F::from_canonical_u32(Opcode::I32LtU.code()),
        );

        builder.when(local.is_i32le).assert_eq(
            local.opcode,
            local.is_signed * AB::F::from_canonical_u32(Opcode::I32LeS.code()) +
                (AB::Expr::one() - local.is_signed) *
                    AB::F::from_canonical_u32(Opcode::I32LeU.code()),
        );

        builder.when(local.is_i32gt).assert_eq(
            local.opcode,
            local.is_signed * AB::F::from_canonical_u32(Opcode::I32GtS.code()) +
                (AB::Expr::one() - local.is_signed) *
                    AB::F::from_canonical_u32(Opcode::I32GtU.code()),
        );

        builder.when(local.is_i32ge).assert_eq(
            local.opcode,
            local.is_signed * AB::F::from_canonical_u32(Opcode::I32GeS.code()) +
                (AB::Expr::one() - local.is_signed) *
                    AB::F::from_canonical_u32(Opcode::I32GeU.code()),
        );

        // For equality operations, directly check the opcode.
        builder
            .when(local.is_i32eq)
            .assert_eq(local.opcode, AB::F::from_canonical_u32(Opcode::I32Eq.code()));

        builder
            .when(local.is_i32ne)
            .assert_eq(local.opcode, AB::F::from_canonical_u32(Opcode::I32Ne.code()));

        builder
            .when(local.is_i32eqz)
            .assert_eq(local.opcode, AB::F::from_canonical_u32(Opcode::I32Eqz.code()));

        // Now use the pre-verified opcode directly.
        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            local.opcode,
            Word::extend_var::<AB>(local.a),
            local.b,
            local.c,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_i32le + local.is_i32lt + local.is_i32eq + local.is_i32ne,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            local.opcode,
            Word::extend_var::<AB>(local.a),
            local.b,
            local.c,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_i32eqz,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            local.opcode,
            Word::extend_var::<AB>(local.a),
            local.c,
            local.b,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_i32ge + local.is_i32gt,
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::LtChip;

    use crate::{
        alu::LtCols,
        io::SP1Stdin,
        rwasm::{CpuChip, RwasmAir},
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
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
        shard.lt_events = vec![AluEvent::new(0, 0, Opcode::I32LtS, 0, 3, 2, Opcode::I32LtS.code())];
        let chip = LtChip::default();
        let generate_trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let trace: RowMajorMatrix<BabyBear> = generate_trace;
        println!("{:?}", trace.width)
    }

    fn prove_babybear_template(shard: &mut ExecutionRecord) {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let chip = LtChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(shard, &mut ExecutionRecord::default());
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn prove_babybear_slt() {
        let mut shard = ExecutionRecord::default();

        const NEG_3: u32 = 0b11111111111111111111111111111101;
        const NEG_4: u32 = 0b11111111111111111111111111111100;
        shard.lt_events = vec![
            // 0 == 3 < 2
            AluEvent::new(0, 0, Opcode::I32LtS, 0, 3, 2, Opcode::I32LtS.code()),
            // 1 == 2 < 3
            AluEvent::new(0, 0, Opcode::I32LtS, 1, 2, 3, Opcode::I32LtS.code()),
            // 0 == 5 < -3
            AluEvent::new(0, 0, Opcode::I32LtS, 0, 5, NEG_3, Opcode::I32LtS.code()),
            // 1 == -3 < 5
            AluEvent::new(0, 0, Opcode::I32LtS, 1, NEG_3, 5, Opcode::I32LtS.code()),
            // 0 == -3 < -4
            AluEvent::new(0, 0, Opcode::I32LtS, 0, NEG_3, NEG_4, Opcode::I32LtS.code()),
            // 1 == -4 < -3
            AluEvent::new(0, 0, Opcode::I32LtS, 1, NEG_4, NEG_3, Opcode::I32LtS.code()),
            // 0 == 3 < 3
            AluEvent::new(0, 0, Opcode::I32LtS, 0, 3, 3, Opcode::I32LtS.code()),
            // 0 == -3 < -3
            AluEvent::new(0, 0, Opcode::I32LtS, 0, NEG_3, NEG_3, Opcode::I32LtS.code()),
            AluEvent::new(
                0,
                0,
                Opcode::I32LtS,
                0,
                1749720339u32,
                3190814577u32,
                Opcode::I32LtS.code(),
            ),
        ];

        prove_babybear_template(&mut shard);
    }

    #[test]
    fn prove_babybear_all_comparisons_single() {
        // One proof that matches the *new executor lowering*:
        //
        // The executor emits ONLY LtChip-primitive events:
        //   - signed  primitive: I32LtS(b,c)  with a = [b <_s c]
        //   - unsigned primitive: I32LtU(b,c) with a = [b <_u c]
        //
        // Lowerings in the executor:
        //   LT*: emit LT(b,c)                      (a = b<c)
        //   GT*: emit LT(c,b)                      (a = b>c)
        //   GE*: emit LT(b,c)                      (witness for b<c ; CPU takes NOT later)
        //   LE*: emit LT(c,b)                      (witness for b>c ; CPU takes NOT later)
        //   EQZ: emit LTU(x,1)                     (a = x==0)
        //   EQ:  emit LTU(b^c,1)                   (a = b==c)
        //   NE:  emit LTU(0,b^c)                   (a = b!=c)
        //
        // This test does NOT try to check CPU's NOT wiring for GE/LE directly here;
        // it verifies that LtChip accepts the exact primitive events that executor produces.

        const I32_MIN: u32 = 0x8000_0000;
        const I32_MAX: u32 = 0x7FFF_FFFF;
        const NEG1: u32 = 0xFFFF_FFFF; // -1
        const NEG2: u32 = 0xFFFF_FFFE; // -2
        const NEG3: u32 = 0xFFFF_FFFD; // -3
        const NEG4: u32 = 0xFFFF_FFFC; // -4

        let mut shard = ExecutionRecord::default();
        let mut evs: Vec<AluEvent> = Vec::new();

        // Push a *primitive* LtChip event (opcode determines signed/unsigned semantics).
        let mut push_lt = |primitive: Opcode, a: u32, b: u32, c: u32| {
            debug_assert!(matches!(primitive, Opcode::I32LtS | Opcode::I32LtU));
            evs.push(AluEvent::new(0, primitive, a, b, c, primitive.code()));
        };

        // --- EQZ(x) -> LTU(x, 1)
        for &x in &[0, 1, 2, I32_MAX, I32_MIN, NEG1, 0xDEAD_BEEF] {
            push_lt(Opcode::I32LtU, u32::from(x == 0), x, 1);
        }

        // --- EQ/NE via XOR lowering
        for &(b, c) in &[
            (0, 0),
            (0, 1),
            (1, 0),
            (NEG1, NEG1),
            (NEG1, 0),
            (0, NEG1),
            (I32_MIN, I32_MAX),
            (I32_MAX, I32_MIN),
            (0xDEAD_BEEF, 0xDEAD_BEEF),
            (0xDEAD_BEEF, 0xDEAD_BEEE),
        ] {
            let x = b ^ c;

            // Eq(b,c): LTU(x,1)  == [x==0]
            push_lt(Opcode::I32LtU, u32::from(x == 0), x, 1);

            // Ne(b,c): LTU(0,x)  == [x!=0]
            push_lt(Opcode::I32LtU, u32::from(x != 0), 0, x);
        }

        // Canonical (b,c) pairs used for LT/GT/GE/LE primitives.
        // Include: equal, boundaries, sign boundary, neg/pos, and first-diff byte3/2/1/0.
        let pairs: &[(u32, u32)] = &[
            (0, 0),
            (0, 1),
            (1, 0),
            (I32_MAX, I32_MIN),
            (I32_MIN, I32_MAX),
            (0, NEG1),
            (NEG1, 0),
            // first differing byte (MSB..LSB)
            (0x0100_0000, 0x0200_0000), // byte3
            (0x0001_0000, 0x0002_0000), // byte2
            (0x0000_0100, 0x0000_0200), // byte1
            (0x0000_0001, 0x0000_0002), // byte0
            (0xDEAD_BEEF, 0xDEAD_BEF0),
            (0x80FF_0000, 0x80FE_FFFF),
            // extra signed stress
            (NEG4, NEG3),
            (NEG3, NEG4),
            (NEG3, NEG2),
            (NEG2, NEG3),
            (0x8000_0001, 0x8000_0002),
            (0x8000_0002, 0x8000_0001),
        ];

        // --- LT primitives (what executor emits for I32LtS/I32LtU)
        for &(b, c) in pairs {
            push_lt(Opcode::I32LtU, u32::from(b < c), b, c);
            push_lt(Opcode::I32LtS, u32::from((b as i32) < (c as i32)), b, c);
        }

        // --- GT primitives (what executor emits for I32GtS/I32GtU): LT(c,b)
        for &(b, c) in pairs {
            push_lt(Opcode::I32LtU, u32::from(b > c), c, b); // c < b  == b > c
            push_lt(Opcode::I32LtS, u32::from((b as i32) > (c as i32)), c, b);
        }

        // --- GE primitives (what executor emits for I32GeS/I32GeU): LT(b,c) witness
        // CPU must later compute GE = 1 - LT(b,c).
        for &(b, c) in pairs {
            push_lt(Opcode::I32LtU, u32::from(b < c), b, c);
            push_lt(Opcode::I32LtS, u32::from((b as i32) < (c as i32)), b, c);
        }

        // --- LE primitives (what executor emits for I32LeS/I32LeU): LT(c,b) witness
        // CPU must later compute LE = 1 - LT(c,b) = 1 - (b > c).
        for &(b, c) in pairs {
            push_lt(Opcode::I32LtU, u32::from(b > c), c, b);
            push_lt(Opcode::I32LtS, u32::from((b as i32) > (c as i32)), c, b);
        }

        shard.lt_events = evs;
        prove_babybear_template(&mut shard);
    }
    #[test]
    fn prove_babybear_sltu() {
        let mut shard = ExecutionRecord::default();

        const LARGE: u32 = 0b11111111111111111111111111111101;
        shard.lt_events = vec![
            // 0 == 3 < 2
            AluEvent::new(0, 0, Opcode::I32LtU, 0, 3, 2, Opcode::I32LtU.code()),
            // 1 == 2 < 3
            AluEvent::new(0, 0, Opcode::I32LtU, 1, 2, 3, Opcode::I32LtU.code()),
            // 0 == LARGE < 5
            AluEvent::new(0, 0, Opcode::I32LtU, 0, LARGE, 5, Opcode::I32LtU.code()),
            // 1 == 5 < LARGE
            AluEvent::new(0, 0, Opcode::I32LtU, 1, 5, LARGE, Opcode::I32LtU.code()),
            // 0 == 0 < 0
            AluEvent::new(0, 0, Opcode::I32LtU, 0, 0, 0, Opcode::I32LtU.code()),
            // 0 == LARGE < LARGE
            AluEvent::new(0, 0, Opcode::I32LtU, 0, LARGE, LARGE, Opcode::I32LtU.code()),
        ];

        prove_babybear_template(&mut shard);
    }
    #[test]
    fn prove_babybear_comparisons() {
        let mut shard = ExecutionRecord::default();

        // Reuse patterns from the SLT/SLTU proofs and cover edge cases.
        const NEG_3: u32 = 0xFFFF_FFFD;
        const NEG_4: u32 = 0xFFFF_FFFC;
        const LARGE: u32 = 0xFFFF_FFFD; // same as NEG_3

        shard.lt_events = vec![
            // ---- I32Eqz ----
            // Encode eqz(x) by unsigned check x < 1.
            AluEvent::new(0, 0, Opcode::I32Eqz, 1, 0, 0, Opcode::I32Eqz.code()), // 0 == 0
            AluEvent::new(0, 0, Opcode::I32Eqz, 0, 5, 0, Opcode::I32Eqz.code()), // 5 != 0
            AluEvent::new(0, 0, Opcode::I32Eqz, 0, 1, 0, Opcode::I32Eqz.code()), // 1 != 0
            AluEvent::new(0, 0, Opcode::I32Eqz, 0, NEG_3, 0, Opcode::I32Eqz.code()), // NEG_3 != 0
            
            AluEvent::new(0, 0, Opcode::I32Eq, 1, 7, 7, Opcode::I32Eq.code()),
            AluEvent::new(0, 0, Opcode::I32Eq, 0, 0, 1, Opcode::I32Eq.code()),
            AluEvent::new(0, 0, Opcode::I32Eq, 1, NEG_3, NEG_3, Opcode::I32Eq.code()),
            AluEvent::new(0, 0, Opcode::I32Eq, 1, 123_456, 123_456, Opcode::I32Eq.code()),
            // // ---- I32LtS (signed) ----
            AluEvent::new(0, 0, Opcode::I32LtS, 0, 3, 2, Opcode::I32LtS.code()), // 3 < 2 ? 0
            AluEvent::new(0, 0, Opcode::I32LtS, 1, 2, 3, Opcode::I32LtS.code()), // 2 < 3 ? 1
            AluEvent::new(0, 0, Opcode::I32LtS, 1, NEG_3, 5, Opcode::I32LtS.code()), // -3 < 5 ? 1
            AluEvent::new(0, 0, Opcode::I32LtS, 0, 5, NEG_3, Opcode::I32LtS.code()), // 5 < -3 ? 0
            // ---- I32GtS (signed) -> encode as c < b with LT(S) (swap operands) ----
            AluEvent::new(0, 0, Opcode::I32GtS, 0, NEG_3, 5, Opcode::I32GtS.code()), // 5 > -3
            AluEvent::new(0, 0, Opcode::I32GtS, 1, NEG_3, NEG_4, Opcode::I32GtS.code()), /* -4 >
                                                                                     //                                                                           * -3 ?
                                                                                     //                                                                           * 0
                                                                                     //                                                                           * (swap: -3 <
                                                                                     //                                                                           * -4) */
            AluEvent::new(0, 0, Opcode::I32GtS, 1, 3, 2, Opcode::I32GtS.code()), /* 2 > 3 ? 0
                                                                                  * (swap: 3 <
                                                                                  * 2) */
            AluEvent::new(0, 0, Opcode::I32GtS, 0, 2, 3, Opcode::I32GtS.code()), /* 3 > 2 ? 1
                                                                                 //                                                                       * (swap: 2 <
                                                                                 //                                                                       * 3) */
            // // ---- I32LtU (unsigned) ----
            AluEvent::new(0, 0, Opcode::I32LtU, 0, 3, 2, Opcode::I32LtU.code()),
            AluEvent::new(0, 0, Opcode::I32LtU, 1, 2, 3, Opcode::I32LtU.code()),
            AluEvent::new(0, 0, Opcode::I32LtU, 0, LARGE, 5, Opcode::I32LtU.code()),
            AluEvent::new(0, 0, Opcode::I32LtU, 1, 5, LARGE, Opcode::I32LtU.code()),
            // // ---- I32GtU (unsigned) -> encode as c < b with LT(U) (swap operands) ----
            AluEvent::new(0, 0, Opcode::I32GtU, 1, LARGE, 5, Opcode::I32GtU.code()), /* 5 > LARGE ? 0 */
            AluEvent::new(0, 0, Opcode::I32GtU, 0, 5, LARGE, Opcode::I32GtU.code()), /* LARGE > 5 ? 1 */
            AluEvent::new(0, 0, Opcode::I32GtU, 0, 2, 3, Opcode::I32GtU.code()), /* 3 > 2 ? 1 (swap: 2 < 3) */
            AluEvent::new(0, 0, Opcode::I32GtU, 1, 3, 2, Opcode::I32GtU.code()), /* 2 > 3 ? 0 (swap: 3 < 2) */
            // // ---- I32LeS (signed) ----
            // // Choose unequal pairs so (b <= c) == (b < c) for these rows.
            AluEvent::new(0, 0, Opcode::I32LeS, 1, NEG_4, NEG_3, Opcode::I32LeS.code()), /* -4 <= -3 */
            AluEvent::new(0, 0, Opcode::I32LeS, 0, NEG_3, NEG_4, Opcode::I32LeS.code()), /* -3 <= -4 ? 0 */
            AluEvent::new(0, 0, Opcode::I32LeS, 1, 2, 3, Opcode::I32LeS.code()),         // 2 <= 3
            AluEvent::new(0, 0, Opcode::I32LeS, 0, 3, 2, Opcode::I32LeS.code()), // 3 <= 2 ? 0
            // // ---- I32GeS (signed) -> encode as c < b (swap operands) ----
            // // Use unequal pairs so (b >= c) == (b > c) == (c < b).
            AluEvent::new(0, 0, Opcode::I32GeS, 0, 2, 3, Opcode::I32GeS.code()), // 3 >= 2
            AluEvent::new(0, 0, Opcode::I32GeS, 1, 3, 2, Opcode::I32GeS.code()), // 2 >= 3 ? 0
            AluEvent::new(0, 0, Opcode::I32GeS, 0, NEG_4, NEG_3, Opcode::I32GeS.code()), // -3 >= -4
            AluEvent::new(0, 0, Opcode::I32GeS, 1, NEG_3, NEG_4, Opcode::I32GeS.code()), /* -4 >= -3 ? 0 */
            // // ---- I32LeU (unsigned) ----
            // // Use unequal pairs so (b <= c) == (b < c).
            AluEvent::new(0, 0, Opcode::I32LeU, 1, 0, 1, Opcode::I32LeU.code()),
            AluEvent::new(0, 0, Opcode::I32LeU, 0, 1, 0, Opcode::I32LeU.code()),
            AluEvent::new(0, 0, Opcode::I32LeU, 1, 5, LARGE, Opcode::I32LeU.code()),
            AluEvent::new(0, 0, Opcode::I32LeU, 0, LARGE, 5, Opcode::I32LeU.code()),
            // // ---- I32GeU (unsigned) -> encode as c < b (swap operands) ----
            AluEvent::new(0, 0, Opcode::I32GeU, 0, 0, 1, Opcode::I32GeU.code()), // 1 >= 0
            AluEvent::new(0, 0, Opcode::I32GeU, 1, 1, 0, Opcode::I32GeU.code()), // 0 >= 1 ? 0
            AluEvent::new(0, 0, Opcode::I32GeU, 0, 5, LARGE, Opcode::I32GeU.code()), // LARGE >= 5
            AluEvent::new(0, 0, Opcode::I32GeU, 1, LARGE, 5, Opcode::I32GeU.code()), /* 5 >= LARGE ?
                                                                                 //                                                                       * 0 */
        ];

        // now prove & verify on LtChip.
        prove_babybear_template(&mut shard);
    }

    #[test]
    fn test_malicious_lt() {
        for opcode in [
            Opcode::I32Eqz,
            Opcode::I32Eq,
            Opcode::I32LtS,
            Opcode::I32GtS,
            Opcode::I32LtU,
            Opcode::I32GtU,
            Opcode::I32LeS,
            Opcode::I32GeS,
            Opcode::I32LeU,
            Opcode::I32GeU,
        ] {
            run_malicious_lt(opcode)
        }
    }

    fn run_malicious_lt(opcode: Opcode) {
        use core::borrow::BorrowMut;
        const NUM_TESTS: usize = 1;

        let mut rng = thread_rng();
        for _ in 0..NUM_TESTS {
            let op_b = rng.gen_range(0..u32::MAX);
            let op_c = rng.gen_range(0..u32::MAX);

            let correct_op_a = if opcode == Opcode::I32LtU {
                op_b < op_c
            } else if opcode == Opcode::I32GtU {
                op_b > op_c
            } else if opcode == Opcode::I32LeU {
                op_b <= op_c
            } else if opcode == Opcode::I32GeU {
                op_b >= op_c
            } else if opcode == Opcode::I32Eq {
                op_b == op_c
            } else if opcode == Opcode::I32LtS {
                (op_b as i32) < (op_c as i32)
            } else if opcode == Opcode::I32GtS {
                (op_b as i32) > (op_c as i32)
            } else if opcode == Opcode::I32LeS {
                (op_b as i32) <= (op_c as i32)
            } else if opcode == Opcode::I32GeS {
                (op_b as i32) >= (op_c as i32)
            } else if opcode == Opcode::I32Eqz {
                op_c == 0
            } else {
                true
            };

            let op_a = !correct_op_a;

            let program = Program::from_instrs(vec![
                Opcode::I32Const(op_b.into()),
                Opcode::I32Const(op_c.into()),
                opcode,
            ]);
            let stdin = SP1Stdin::new();
            type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

            let malicious_trace_pv_generator = move |prover: &P, record: &mut ExecutionRecord| {
                let mut malicious_record = record.clone();
                if malicious_record.cpu_events.len() > 2 {
                    // keep memory write consistent
                    if let Some(MemoryRecordEnum::Write(mut write_record)) =
                        &mut malicious_record.cpu_events[2].res_record
                    {
                        write_record.value = op_a as u32;
                    }
                }

                let mut traces = prover.generate_traces(&malicious_record);
                let lt_chip_name = chip_name!(LtChip, BabyBear);
                if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == lt_chip_name)
                {
                    let row = trace.row_mut(0);
                    let row: &mut LtCols<BabyBear> = row.borrow_mut();
                    row.a = BabyBear::from_bool(op_a); // inject forged value
                }

                traces
            };

            let result =
                run_malicious_test::<P>(program, stdin, Box::new(malicious_trace_pv_generator));

            let chip_name = chip_name!(CpuChip, BabyBear);
            println!("run_malicious_lt for opcode : {:?}", opcode);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_constraints_failing(&chip_name));
        }
    }
}
