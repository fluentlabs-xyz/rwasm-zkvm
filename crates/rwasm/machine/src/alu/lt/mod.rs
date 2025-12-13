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
use rwasm::Opcode;

use rwasm_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{MachineAir, SP1AirBuilder},
    Word,
};

use crate::utils::{next_power_of_two, zeroed_f_vec};

/// The number of main trace columns for `LtChip`.
pub const NUM_LT_COLS: usize = size_of::<LtCols<u8>>();

/// A chip that implements comparisons for I32LtS and I32LtU (used as a building block for others).
#[derive(Default)]
pub struct LtChip;

/// Optimized (degree<=3) column layout.
///
/// Removed (vs original):
/// - op_a_not_0 (always 1 in event_to_row; instead we constrain `a` for all rows)
/// - msb_b, msb_c (derived in AIR from masked bytes)
/// - is_comp_eq (derived as `1 - sum_flags`)
/// - is_sign_eq (derived as `1 - (bit_b-bit_c)^2`)
/// - byte_equality_check[4] (unused)
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LtCols<T> {
    /// The program counter.
    pub pc: T,

    /// If the opcode is SLT (signed).
    pub is_slt: T,

    /// If the opcode is SLTU (unsigned).
    pub is_sltu: T,

    /// The output operand (1 byte; later extended to a word by receive_instruction).
    pub a: T,

    /// The first input operand (little-endian bytes).
    pub b: Word<T>,

    /// The second input operand (little-endian bytes).
    pub c: Word<T>,

    /// Boolean flags indicating which *most-significant* differing byte is selected.
    /// Exactly one is set iff b_comp != c_comp.
    pub byte_flags: [T; 4],

    /// b[3] & 0x7F (only meaningful for signed LT).
    pub b_masked: T,

    /// c[3] & 0x7F (only meaningful for signed LT).
    pub c_masked: T,

    /// Inverse hint for proving inequality of selected bytes when b_comp != c_comp.
    pub not_eq_inv: T,

    /// bit_b = msb(b) * is_slt   (0 for SLTU, sign bit for SLT).
    pub bit_b: T,

    /// bit_c = msb(c) * is_slt
    pub bit_c: T,

    /// sltu = b_comp < c_comp where b_comp/c_comp are described in `eval()`.
    pub sltu: T,

    /// The selected comparison bytes (0,0 if equal; otherwise the first differing byte-pair).
    pub comparison_bytes: [T; 2],
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
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut LtCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();

        cols.pc = F::from_canonical_u32(event.pc);

        cols.a = F::from_canonical_u8(a[0]);
        cols.b = Word(b.map(F::from_canonical_u8));
        cols.c = Word(c.map(F::from_canonical_u8));

        cols.is_slt = F::from_bool(event.code == Opcode::I32LtS.code());
        cols.is_sltu = F::from_bool(event.code == Opcode::I32LtU.code());

        // If this is SLT, mask the MSB of b & c (clear sign bit in the top byte).
        let masked_b = b[3] & 0x7f;
        let masked_c = c[3] & 0x7f;
        cols.b_masked = F::from_canonical_u8(masked_b);
        cols.c_masked = F::from_canonical_u8(masked_c);

        // AND lookups for mask correctness.
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

        // Precompute b_comp/c_comp for software (just to set sltu & find differing byte).
        let mut b_comp = b;
        let mut c_comp = c;
        if event.code == Opcode::I32LtS.code() {
            b_comp[3] = masked_b;
            c_comp[3] = masked_c;
        }

        cols.sltu = F::from_bool(b_comp < c_comp);

        // Set bit_b/bit_c (0 for SLTU, msb for SLT).
        let msb_b = (b[3] >> 7) & 1;
        let msb_c = (c[3] >> 7) & 1;
        cols.bit_b = F::from_canonical_u8(msb_b) * cols.is_slt;
        cols.bit_c = F::from_canonical_u8(msb_c) * cols.is_slt;

        // Set the byte flags: find the first differing byte from MSB->LSB (index 3..0).
        // If equal, all flags stay 0 and comparison_bytes stay (0,0).
        for (b_byte, c_byte, flag) in
            izip!(b_comp.iter().rev(), c_comp.iter().rev(), cols.byte_flags.iter_mut().rev())
        {
            if c_byte != b_byte {
                *flag = F::one();
                cols.sltu = F::from_bool(b_byte < c_byte);

                let b_f = F::from_canonical_u8(*b_byte);
                let c_f = F::from_canonical_u8(*c_byte);

                cols.not_eq_inv = (b_f - c_f).inverse();
                cols.comparison_bytes = [b_f, c_f];
                break;
            }
        }

        // Final output check (debug-only): a = bit_b*(1-bit_c) + is_sign_eq*sltu
        // where is_sign_eq = 1 - (bit_b - bit_c)^2.
        let d = cols.bit_b - cols.bit_c;
        let is_sign_eq = F::one() - d * d;
        let expected_a = cols.bit_b * (F::one() - cols.bit_c) + is_sign_eq * cols.sltu;
        debug_assert_eq!(cols.a, expected_a);

        // LTU lookup on selected bytes (0,0 in equal case).
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: cols.sltu.as_canonical_u32() as u16,
            a2: 0,
            b: cols.comparison_bytes[0].as_canonical_u32() as u8,
            c: cols.comparison_bytes[1].as_canonical_u32() as u8,
        });
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

        // Operation selectors.
        builder.assert_bool(local.is_slt);
        builder.assert_bool(local.is_sltu);
        let is_real = local.is_slt + local.is_sltu;
        builder.assert_bool(is_real.clone());

        // Build b_comp/c_comp in expressions:
        // - for SLTU: b_comp = b, c_comp = c
        // - for SLT:  b_comp[3] = b_masked, c_comp[3] = c_masked (top bit cleared)
        let mut b_comp: Word<AB::Expr> = local.b.map(|x| x.into());
        let mut c_comp: Word<AB::Expr> = local.c.map(|x| x.into());

        b_comp[3] = local.b[3] * local.is_sltu + local.b_masked * local.is_slt;
        c_comp[3] = local.c[3] * local.is_sltu + local.c_masked * local.is_slt;

        // Mask correctness via AND lookup (gated by is_real).
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.b_masked,
            local.b[3],
            AB::F::from_canonical_u8(0x7f),
            is_real.clone(),
        );
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.c_masked,
            local.c[3],
            AB::F::from_canonical_u8(0x7f),
            is_real.clone(),
        );

        // Derive msb expressions from masked bytes:
        // msb = (byte - (byte & 0x7f)) / 128.
        let inv_128 = AB::F::from_canonical_u32(128).inverse();
        let msb_b_expr = (local.b[3] - local.b_masked) * inv_128;
        let msb_c_expr = (local.c[3] - local.c_masked) * inv_128;

        // Constrain bit_b/bit_c and force them boolean (degree<=3 safe).
        builder.assert_eq(local.bit_b, msb_b_expr * local.is_slt);
        builder.assert_eq(local.bit_c, msb_c_expr * local.is_slt);
        builder.assert_bool(local.bit_b);
        builder.assert_bool(local.bit_c);

        // is_sign_eq := 1 - (bit_b - bit_c)^2  (degree 2)
        let d = local.bit_b - local.bit_c;
        let is_sign_eq = AB::Expr::one() - d.clone() * d;

        // Final result (NO gating to keep degree<=3):
        // a = bit_b*(1-bit_c) + is_sign_eq*sltu
        builder.assert_eq(
            local.a,
            local.bit_b * (AB::Expr::one() - local.bit_c) + is_sign_eq * local.sltu,
        );

        // Byte-flag constraints: each boolean; sum is boolean (thus <=1 flag set).
        for f in local.byte_flags.iter() {
            builder.assert_bool(*f);
        }
        let sum_flags =
            local.byte_flags[0] + local.byte_flags[1] + local.byte_flags[2] + local.byte_flags[3];
        builder.assert_bool(sum_flags.clone());

        // Optional hygiene: padding rows should not select a differing byte.
        builder.when_not(is_real.clone()).assert_zero(sum_flags.clone());

        // Enforce "first differing byte" semantics and compute the selected comparison bytes.
        let mut is_inequality_visited = AB::Expr::zero();
        let mut b_selected = AB::Expr::zero();
        let mut c_selected = AB::Expr::zero();

        for (b_byte, c_byte, &flag) in
            izip!(b_comp.0.iter().rev(), c_comp.0.iter().rev(), local.byte_flags.iter().rev())
        {
            // Accumulate visited flags (degree 1).
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            // Select the flagged byte-pair (note b_comp[3]/c_comp[3] are degree-2 expressions).
            b_selected = b_selected.clone() + b_byte.clone() * flag;
            c_selected = c_selected.clone() + c_byte.clone() * flag;

            // Until the first inequality is visited, bytes must be equal.
            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_byte.clone(), c_byte.clone());
        }

        // Constrain stored comparison bytes to match the selected expressions.
        let b_comp_byte = local.comparison_bytes[0];
        let c_comp_byte = local.comparison_bytes[1];
        builder.assert_eq(b_comp_byte, b_selected);
        builder.assert_eq(c_comp_byte, c_selected);

        // If sum_flags == 1 (i.e., not equal), enforce the selected bytes differ using not_eq_inv:
        // sum_flags * (not_eq_inv*(b_comp_byte - c_comp_byte) - is_real) = 0   (degree <= 3)
        builder
            .when(sum_flags.clone())
            .assert_eq(local.not_eq_inv * (b_comp_byte - c_comp_byte), is_real.clone());

        // Constrain sltu via LTU lookup on the selected bytes (gated by is_real).
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            local.sltu,
            b_comp_byte,
            c_comp_byte,
            is_real.clone(),
        );

        // Receive the instruction.
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            local.is_slt * AB::F::from_canonical_u32(Opcode::I32LtS.code()) +
                local.is_sltu * AB::F::from_canonical_u32(Opcode::I32LtU.code()),
            Word::extend_var::<AB>(local.a),
            local.b,
            local.c,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
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
        shard.lt_events = vec![AluEvent::new(0, Opcode::I32LtS, 0, 3, 2, Opcode::I32LtS.code())];
        let chip = LtChip::default();
        let generate_trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let trace: RowMajorMatrix<BabyBear> = generate_trace;
        println!("{:?}", trace.values)
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
            AluEvent::new(0, Opcode::I32LtS, 0, 3, 2, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 1, 2, 3, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 0, 5, NEG_3, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 1, NEG_3, 5, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 0, NEG_3, NEG_4, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 1, NEG_4, NEG_3, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 0, 3, 3, Opcode::I32LtS.code()),
            AluEvent::new(0, Opcode::I32LtS, 0, NEG_3, NEG_3, Opcode::I32LtS.code()),
            AluEvent::new(
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
    fn prove_babybear_sltu() {
        let mut shard = ExecutionRecord::default();

        const LARGE: u32 = 0b11111111111111111111111111111101;
        shard.lt_events = vec![
            AluEvent::new(0, Opcode::I32LtU, 0, 3, 2, Opcode::I32LtU.code()),
            AluEvent::new(0, Opcode::I32LtU, 1, 2, 3, Opcode::I32LtU.code()),
            AluEvent::new(0, Opcode::I32LtU, 0, LARGE, 5, Opcode::I32LtU.code()),
            AluEvent::new(0, Opcode::I32LtU, 1, 5, LARGE, Opcode::I32LtU.code()),
            AluEvent::new(0, Opcode::I32LtU, 0, 0, 0, Opcode::I32LtU.code()),
            AluEvent::new(0, Opcode::I32LtU, 0, LARGE, LARGE, Opcode::I32LtU.code()),
        ];

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
                    malicious_record.cpu_events[2].res = op_a as u32;
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
