use crate::{air::SP1CoreAirBuilder, utils::pad_rows_fixed};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rwasm::Opcode::{I32Extend16S, I32Extend8S};
use rwasm_executor::{events::AluEvent, ExecutionRecord, Program, DEFAULT_PC_INC};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::MachineAir, Word};

pub const NUM_EXTEND_COLS: usize = size_of::<ExtendCols<u8>>();

/// # Extend chip (i32.extend8_s / i32.extend16_s)
///
/// ## Columns (per row)
/// - `pc`: program counter.
/// - `is_extend8s`, `is_extend16s`: opcode selectors; sum is boolean (at most one fires).
/// - `b: Word<T>`: input u32 as 4 bytes `[b0,b1,b2,b3]` (LSB→MSB).
/// - `a: Word<T>`: result u32 as 4 bytes `[a0,a1,a2,a3]` (LSB→MSB).
///
/// ### 8-bit witnesses
/// - `bits8[0..8)`: boolean bits of low 8 of `b`.
/// - `lo8`: low 8 value reconstructed from `bits8`.
/// - `q8`: quotient witnessing `b = lo8 + 2^8 * q8`.
///
/// ### 16-bit witnesses
/// - `bits16[0..16)`: boolean bits of low 16 of `b` (LSB→MSB; `bits16[15]` is sign).
/// - `lo16 = [lo,hi]`: low 16 value as two bytes; `lo16_lo = Σ bits16[0..8)·2^i`, `lo16_hi = Σ
///   bits16[8..16)·2^(i-8)`.
/// - `q16  = [lo,hi]`: upper 16 of `b` as two bytes; witnesses `b = lo16 + 2^16 * q16`.
///
/// ## Constraint sketch (all gated by the active selector)
/// - **Byte reconstruction:** `b = Σ b[i]·256^i`, `a = Σ a[i]·256^i` (used only for linkage).
/// - **8-bit path:**
///   1) `bits8[i] ∈ {0,1}`.
///   2) `lo8 = Σ bits8[i]·2^i`.
///   3) `b = lo8 + 256·q8`.
///   4) Let `s8 = bits8[7]`. Enforce bytes: `a0 = lo8`, `a1 = a2 = a3 = 0xFF·s8`.
/// - **16-bit path:**
///   1) `bits16[i] ∈ {0,1}`.
///   2) `lo16_lo = Σ bits16[0..8)·2^i`, `lo16_hi = Σ bits16[8..16)·2^(i-8)`.
///   3) `lo16 = lo16_lo + 256·lo16_hi` and `b = lo16 + 65536·q16`.
///   4) Let `s16 = bits16[15]`. Enforce bytes: `a0 = lo16_lo`, `a1 = lo16_hi`, `a2 = a3 =
///      0xFF·s16`.
///
/// This byte-wise formulation avoids large constants like `2^32-2^8/2^16` and works over small
/// fields.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ExtendCols<T> {
    pub pc: T,
    pub a: Word<T>, // result as bytes (LSB..MSB)
    pub b: Word<T>, // input word as bytes (LSB..MSB)
    pub is_extend8s: T,
    pub is_extend16s: T,

    // 8-bit witnesses
    pub lo8: T,        // b mod 256
    pub q8: T,         // floor(b / 256)
    pub bits8: [T; 8], // boolean bits of lo8 (LSB..MSB)

    // 16-bit witnesses (store as two bytes to avoid large-field constants)
    pub lo16: [T; 2],    // b mod 65536, as [lo, hi]
    pub q16: [T; 2],     // floor(b / 65536), as [lo, hi]
    pub bits16: [T; 16], // boolean bits of lo16 (LSB..MSB)
}

#[derive(Default)]
pub struct ExtendChip;

impl ExtendChip {
    fn event_to_row<F: PrimeField32>(&self, event: &AluEvent, cols: &mut ExtendCols<F>) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.a = event.a.into();
        cols.b = event.b.into();
        cols.is_extend8s = F::from_bool(event.opcode == I32Extend8S);
        cols.is_extend16s = F::from_bool(event.opcode == I32Extend16S);

        // Concrete witnesses for sanity (not strictly necessary at prover level but handy)
        let b = event.b;
        // 8-bit path
        let lo8 = (b & 0xFF) as u32;
        cols.lo8 = F::from_canonical_u32(lo8);
        cols.q8 = F::from_canonical_u32(b >> 8);
        let mut tmp = lo8;
        for bit in cols.bits8.iter_mut() {
            *bit = F::from_canonical_u32(tmp & 1);
            tmp >>= 1;
        }
        // 16-bit path
        let lo16 = (b & 0xFFFF) as u32;
        let q16 = (b >> 16) as u32;
        cols.lo16 = [F::from_canonical_u32(lo16 & 0xFF), F::from_canonical_u32((lo16 >> 8) & 0xFF)];
        cols.q16 = [F::from_canonical_u32(q16 & 0xFF), F::from_canonical_u32((q16 >> 8) & 0xFF)];
        let mut tmp16 = lo16;
        for bit in cols.bits16.iter_mut() {
            *bit = F::from_canonical_u32(tmp16 & 1);
            tmp16 >>= 1;
        }
    }
}

impl<AB> Air<AB> for ExtendChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ExtendCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_extend8s);
        builder.assert_bool(local.is_extend16s);
        let is_real = local.is_extend8s + local.is_extend16s;
        builder.assert_bool(is_real.clone());

        // Reconstruct b from bytes to a single field expression: b = Σ b[i] * 256^i
        let mut b_expr = AB::Expr::from_canonical_u32(0);
        for (i, byte) in local.b.0.iter().enumerate() {
            let w = AB::Expr::from_canonical_u32(1u32 << (8 * i));
            b_expr = b_expr + (*byte) * w;
        }
        // Reconstruct a from bytes to a single field expression: a = Σ a[i] * 256^i
        let mut a_expr = AB::Expr::from_canonical_u32(0);
        for (i, byte) in local.a.0.iter().enumerate() {
            let w = AB::Expr::from_canonical_u32(1u32 << (8 * i));
            a_expr = a_expr + (*byte) * w;
        }

        let two_pow_8 = AB::Expr::from_canonical_u32(1 << 8);
        let two_pow_16 = AB::Expr::from_canonical_u32(1 << 16);
        let one = AB::Expr::one();

        // === Constraint summary (per active opcode) ===
        // See the module docs above for the full derivation. We gate every assertion with
        // `is_extend8s` / `is_extend16s`. The result `a` is constrained byte-wise.

        // -------- extend8_s path --------
        builder.when(local.is_extend8s).assert_zero({
            // booleanity for bits8
            let mut acc = AB::Expr::from_canonical_u32(0);
            for bit in local.bits8 {
                acc = acc + bit * (one.clone() - bit);
            }
            acc
        });

        // lo8 = Σ bit[i] * 2^i
        let mut lo8_from_bits = AB::Expr::from_canonical_u32(0);
        for (i, bit) in local.bits8.iter().enumerate() {
            let w = AB::Expr::from_canonical_u32(1u32 << i);
            lo8_from_bits = lo8_from_bits + (*bit) * w;
        }
        builder.when(local.is_extend8s).assert_zero(local.lo8 - lo8_from_bits.clone());

        // b = lo8 + 2^8 * q8
        builder
            .when(local.is_extend8s)
            .assert_zero(b_expr.clone() - (local.lo8 + two_pow_8.clone() * local.q8));

        // Byte-wise semantics:
        // a[0] = lo8
        // a[1], a[2], a[3] = 0xFF * msb8
        let a_bytes = &local.a.0;
        let msb8 = local.bits8[7];
        let ff = AB::Expr::from_canonical_u32(0xFF);
        builder.when(local.is_extend8s).assert_zero(a_bytes[0] - lo8_from_bits.clone());
        builder.when(local.is_extend8s).assert_zero(a_bytes[1] - msb8 * ff.clone());
        builder.when(local.is_extend8s).assert_zero(a_bytes[2] - msb8 * ff.clone());
        builder.when(local.is_extend8s).assert_zero(a_bytes[3] - msb8 * ff);

        // -------- extend16_s path --------
        builder.when(local.is_extend16s).assert_zero({
            // booleanity for bits16
            let mut acc = AB::Expr::from_canonical_u32(0);
            for bit in local.bits16 {
                acc = acc + bit * (one.clone() - bit);
            }
            acc
        });

        // Reconstruct 16-bit low and high bytes from bits16
        let mut lo16_low_from_bits = AB::Expr::from_canonical_u32(0);
        for (i, bit) in local.bits16[0..8].iter().enumerate() {
            let w = AB::Expr::from_canonical_u32(1u32 << i);
            lo16_low_from_bits = lo16_low_from_bits + (*bit) * w;
        }
        let mut lo16_high_from_bits = AB::Expr::from_canonical_u32(0);
        for (i, bit) in local.bits16[8..16].iter().enumerate() {
            let w = AB::Expr::from_canonical_u32(1u32 << i);
            lo16_high_from_bits = lo16_high_from_bits + (*bit) * w;
        }
        let lo16_expr =
            lo16_low_from_bits.clone() + two_pow_8.clone() * lo16_high_from_bits.clone();
        // q16 is provided as two bytes
        let q16_expr = local.q16[0] + two_pow_8.clone() * local.q16[1];
        // lo16 matches its bit decomposition
        builder
            .when(local.is_extend16s)
            .assert_zero((local.lo16[0] + two_pow_8.clone() * local.lo16[1]) - lo16_expr.clone());
        // b = lo16 + 2^16 * q16
        builder.when(local.is_extend16s).assert_zero(
            b_expr.clone() - (lo16_expr.clone() + two_pow_16.clone() * q16_expr.clone()),
        );
        let msb16 = local.bits16[15];
        let ff = AB::Expr::from_canonical_u32(0xFF);
        // a[0..2] byte semantics: [lo16_lo, lo16_hi, 0xFF*msb16, 0xFF*msb16]
        builder.when(local.is_extend16s).assert_zero(a_bytes[0] - lo16_low_from_bits.clone());
        builder.when(local.is_extend16s).assert_zero(a_bytes[1] - lo16_high_from_bits.clone());
        builder.when(local.is_extend16s).assert_zero(a_bytes[2] - msb16 * ff.clone());
        builder.when(local.is_extend16s).assert_zero(a_bytes[3] - msb16 * ff);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            local.is_extend8s * AB::Expr::from_canonical_u32(I32Extend8S.code()) +
                local.is_extend16s * AB::Expr::from_canonical_u32(I32Extend16S.code()),
            local.a,
            local.b,
            Word::<AB::Expr>::default(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );
    }
}

impl<F> BaseAir<F> for ExtendChip {
    fn width(&self) -> usize {
        NUM_EXTEND_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ExtendChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Extend".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for event in input.extend_events.iter() {
            let mut row = [F::zero(); NUM_EXTEND_COLS];
            let cols: &mut ExtendCols<F> = row.as_mut_slice().borrow_mut();
            self.event_to_row(event, cols);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_EXTEND_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_EXTEND_COLS)
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {
        // No cross-table byte lookups required for sign-extend.
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.extend_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::{ExtendChip, ExtendCols, NUM_EXTEND_COLS};
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rwasm_executor::{events::AluEvent, ExecutionRecord, Opcode, Program};
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        // A few canonical extend cases
        let cases = [
            (Opcode::I32Extend8S, 0x0000_007Fu32, 0x0000_007Fu32), // +127
            (Opcode::I32Extend8S, 0x0000_0080u32, 0xFFFF_FF80u32), // -128
            (Opcode::I32Extend16S, 0x0000_7FFFu32, 0x0000_7FFFu32), // +32767
            (Opcode::I32Extend16S, 0x0000_8000u32, 0xFFFF_8000u32), // -32768
        ];
        for (op, b, a) in cases {
            shard.extend_events.push(AluEvent::new(0, op, a, b, 0, op.code()));
        }

        let chip = ExtendChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.width(), NUM_EXTEND_COLS);
        // Sign-extend chip does not use cross-table byte lookups
        assert_eq!(output.byte_lookups.len(), 0);
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        let samples: [u32; 10] = [
            0x0000_0000,
            0x0000_007F, // +127
            0x0000_0080, // -128
            0x0000_00FF, // -1 (8-bit)
            0x0000_7FFF, // +32767
            0x0000_8000, // -32768
            0x0000_FFFF, // -1 (16-bit)
            0xDEAD_8001,
            0xBEEF_1234,
            0xFEED_BEEF,
        ];

        for &b in &samples {
            let a8 = (((b as i32) as i8) as i32) as u32;
            let a16 = (((b as i32) as i16) as i32) as u32;
            shard.extend_events.push(AluEvent::new(
                0,
                Opcode::I32Extend8S,
                a8,
                b,
                0,
                Opcode::I32Extend8S.code(),
            ));
            shard.extend_events.push(AluEvent::new(
                0,
                Opcode::I32Extend16S,
                a16,
                b,
                0,
                Opcode::I32Extend16S.code(),
            ));
        }

        let chip = ExtendChip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, ExtendChip>(&config, &chip, &mut challenger, trace);
        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_extend() {
        use core::borrow::BorrowMut;
        let config = BabyBearPoseidon2::new();
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        // Program that triggers both extend ops
        let program = Program::from_instrs(vec![
            Opcode::I32Const(0x0000_0080u32.into()),
            Opcode::I32Extend8S,
            Opcode::I32Const(0x0000_8000u32.into()),
            Opcode::I32Extend16S,
        ]);

        let stdin = SP1Stdin::new();
        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            // Generate honest traces first
            let mut traces = prover.generate_traces(record);
            let chip = chip_name!(ExtendChip, BabyBear);

            if let Some((_, trace)) = traces.iter_mut().find(|(name, _)| *name == chip) {
                if trace.height() > 0 {
                    // Corrupt a boolean bit to violate the (b*(1-b)=0) constraint
                    let row0 = trace.row_mut(0);
                    let cols: &mut ExtendCols<BabyBear> = row0.borrow_mut();
                    cols.bits8[0] = BabyBear::from_canonical_u32(2);
                }
            }

            traces
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip = chip_name!(ExtendChip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip));
    }
}
