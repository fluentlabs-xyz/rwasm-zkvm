use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use rwasm::{Opcode, Opcode::I32Mul64};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, I64AluEvent},
    ByteOpcode, ExecutionRecord, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{BYTE_SIZE, LONG_WORD_SIZE, WORD_SIZE};
use sp1_stark::{air::MachineAir, Word};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};

pub const NUM_MUL64_COLS: usize = size_of::<Mul64Cols<u8>>();
const BYTE_MASK: u8 = 0xff;

pub const fn get_msb(a: [u8; WORD_SIZE]) -> u8 {
    (a[WORD_SIZE - 1] >> (BYTE_SIZE - 1)) & 1
}

#[derive(Default)]
pub struct Mul64Chip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Mul64Cols<T> {
    pub pc: T,
    pub a_lo: Word<T>,
    pub a_hi: Word<T>,
    pub b: Word<T>,
    pub c: Word<T>,
    pub carry: [T; LONG_WORD_SIZE],
    pub product: [T; LONG_WORD_SIZE],
    pub b_msb: T,
    pub c_msb: T,
    pub b_sign_extend: T,
    pub c_sign_extend: T,
    pub is_mul64: T,
    pub is_real: T,
}

impl<F> BaseAir<F> for Mul64Chip {
    fn width(&self) -> usize {
        NUM_MUL64_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Mul64Chip {
    type Record = ExecutionRecord;
    type Program = rwasm_executor::Program;

    fn name(&self) -> String {
        "Mul64".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let events: Vec<_> =
            input.i64_events.iter().filter(|e| e.opcode == Opcode::I32Mul64).collect();

        let nb_rows = events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);

        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL64_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL64_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL64_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut Mul64Cols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                });
            },
        );

        RowMajorMatrix::new(values, NUM_MUL64_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events: Vec<_> =
            input.i64_events.iter().filter(|e| e.opcode == Opcode::I32Mul64).collect();

        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_MUL64_COLS];
                    let cols: &mut Mul64Cols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        shard.i64_events.iter().any(|e| e.opcode == Opcode::I32Mul64)
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl Mul64Chip {
    fn event_to_row<F: PrimeField>(
        &self,
        event: &I64AluEvent,
        cols: &mut Mul64Cols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);

        let a_lo_word = event.a_lo.to_le_bytes();
        let a_hi_word = event.a_hi.to_le_bytes();
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        let mut b = b_word.to_vec();
        let mut c = c_word.to_vec();

        // Handle signs. I32Mul64 is signed.
        let b_msb = get_msb(b_word);
        cols.b_msb = F::from_canonical_u8(b_msb);
        let c_msb = get_msb(c_word);
        cols.c_msb = F::from_canonical_u8(c_msb);

        if b_msb == 1 {
            cols.b_sign_extend = F::one();
            b.resize(LONG_WORD_SIZE, BYTE_MASK);
        } else {
            b.resize(LONG_WORD_SIZE, 0);
        }

        if c_msb == 1 {
            cols.c_sign_extend = F::one();
            c.resize(LONG_WORD_SIZE, BYTE_MASK);
        } else {
            c.resize(LONG_WORD_SIZE, 0);
        }

        blu.add_byte_lookup_events(vec![
            ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: b_msb as u16,
                a2: 0,
                b: b_word[WORD_SIZE - 1],
                c: 0,
            },
            ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: c_msb as u16,
                a2: 0,
                b: c_word[WORD_SIZE - 1],
                c: 0,
            },
        ]);

        let mut product = [0u32; LONG_WORD_SIZE];
        for i in 0..b.len() {
            for j in 0..c.len() {
                if i + j < LONG_WORD_SIZE {
                    product[i + j] += (b[i] as u32) * (c[j] as u32);
                }
            }
        }

        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; LONG_WORD_SIZE];
        for i in 0..LONG_WORD_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < LONG_WORD_SIZE {
                product[i + 1] += carry[i];
            }
            cols.carry[i] = F::from_canonical_u32(carry[i]);
        }

        cols.product = product.map(F::from_canonical_u32);
        cols.a_lo = Word(a_lo_word.map(F::from_canonical_u8));
        cols.a_hi = Word(a_hi_word.map(F::from_canonical_u8));
        cols.b = Word(b_word.map(F::from_canonical_u8));
        cols.c = Word(c_word.map(F::from_canonical_u8));
        cols.is_real = F::one();
        cols.is_mul64 = F::one();

        // Send range checks for the original 4 bytes
        blu.add_u8_range_checks(&b_word);
        blu.add_u8_range_checks(&c_word);

        // Send range checks for the extended upper 4 bytes
        for i in WORD_SIZE..LONG_WORD_SIZE {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: b[i],
                c: 0,
            });
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: c[i],
                c: 0,
            });
        }

        blu.add_u16_range_checks(&carry.map(|x| x as u16));
        blu.add_u8_range_checks(&product.map(|x| x as u8));
    }
}

impl<AB> Air<AB> for Mul64Chip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Mul64Cols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << 8);
        let zero: AB::Expr = AB::F::zero().into();
        let byte_mask = AB::F::from_canonical_u8(BYTE_MASK);

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_mul64);
        builder.when(local.is_real).assert_one(local.is_mul64);

        // MSB checks
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.b_msb,
            local.b[WORD_SIZE - 1],
            zero.clone(),
            local.is_real,
        );
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.c_msb,
            local.c[WORD_SIZE - 1],
            zero.clone(),
            local.is_real,
        );

        // Sign extension calculation
        builder.assert_eq(local.b_sign_extend, local.b_msb);
        builder.assert_eq(local.c_sign_extend, local.c_msb);
        builder.assert_bool(local.b_sign_extend);
        builder.assert_bool(local.c_sign_extend);

        // Sign extend b and c
        let (b, c) = {
            let mut b: Vec<AB::Expr> = vec![zero.clone(); LONG_WORD_SIZE];
            let mut c: Vec<AB::Expr> = vec![zero.clone(); LONG_WORD_SIZE];
            for i in 0..LONG_WORD_SIZE {
                if i < WORD_SIZE {
                    b[i] = local.b[i].into();
                    c[i] = local.c[i].into();
                } else {
                    b[i] = local.b_sign_extend * byte_mask;
                    c[i] = local.c_sign_extend * byte_mask;
                }
            }
            (b, c)
        };

        // Send range checks for original b and c bytes
        builder.slice_range_check_u8(&local.b.0, local.is_real);
        builder.slice_range_check_u8(&local.c.0, local.is_real);

        // Send range checks ONLY for the sign-extended upper 4 bytes
        for i in WORD_SIZE..LONG_WORD_SIZE {
            builder.send_byte(
                ByteOpcode::U8Range.as_field::<AB::F>(),
                AB::Expr::zero(),
                b[i].clone(),
                AB::Expr::zero(),
                local.is_real,
            );
            builder.send_byte(
                ByteOpcode::U8Range.as_field::<AB::F>(),
                AB::Expr::zero(),
                c[i].clone(),
                AB::Expr::zero(),
                local.is_real,
            );
        }

        // Uncarried product
        let mut m: Vec<AB::Expr> = vec![zero.clone(); LONG_WORD_SIZE];
        for i in 0..LONG_WORD_SIZE {
            for j in 0..LONG_WORD_SIZE {
                if i + j < LONG_WORD_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * c[j].clone();
                }
            }
        }

        // Carry propagation
        for i in 0..LONG_WORD_SIZE {
            let mut v = m[i].clone();
            if i > 0 {
                v += local.carry[i - 1].into();
            }
            v -= local.carry[i] * base;
            builder.assert_eq(local.product[i], v);
        }

        // Check result against a_lo and a_hi
        for i in 0..WORD_SIZE {
            builder.when(local.is_real).assert_eq(local.product[i], local.a_lo[i]);
            builder.when(local.is_real).assert_eq(local.product[i + WORD_SIZE], local.a_hi[i]);
        }

        // Range checks
        builder.slice_range_check_u16(&local.carry, local.is_real);
        builder.slice_range_check_u8(&local.product, local.is_real);

        let opcode = local.is_mul64 * AB::F::from_canonical_u32(I32Mul64.code());

        builder.receive_64_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            opcode,
            local.a_lo,
            local.a_hi,
            local.b,
            local.c,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{run_malicious_test, uni_stark_prove as prove, uni_stark_verify as verify},
    };
    use p3_baby_bear::BabyBear;
    use rwasm_executor::{ExecutionRecord, Opcode};
    use sp1_stark::{
        air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, chip_name, CpuProver,
        MachineProver, StarkGenericConfig,
    };

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();
        let b = 2u32;
        let c = 3u32;
        let result = (b as i32 as i64) * (c as i32 as i64);
        shard.i64_events.push(I64AluEvent {
            pc: 0,
            opcode: Opcode::I32Mul64,
            a_lo: result as u32,
            a_hi: (result >> 32) as u32,
            b,
            c,
            code: Opcode::I32Mul64.code(),
            res_hi_addr: 0,
            res_hi_access: None,
        });
        let chip = Mul64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        assert_eq!(trace.height(), 16); // Padded to a minimum of 16
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();
        let mut shard = ExecutionRecord::default();
        let mut output = ExecutionRecord::default();

        let test_cases = vec![
            (0, 0),
            (1, 0),
            (u32::MAX, 0),
            (1, 1),
            (u32::MAX, 1),
            (2, 3),
            (u32::MAX, u32::MAX),
            (1 << 20, 1 << 20),
        ];

        for (b, c) in test_cases {
            let result = (b as i32 as i64) * (c as i32 as i64);
            shard.i64_events.push(I64AluEvent {
                pc: 0,
                opcode: Opcode::I32Mul64,
                a_lo: result as u32,
                a_hi: (result >> 32) as u32,
                b,
                c,
                code: Opcode::I32Mul64.code(),
                res_hi_addr: 0,
                res_hi_access: None,
            });
        }

        let chip = Mul64Chip::default();
        let trace: RowMajorMatrix<BabyBear> = chip.generate_trace(&shard, &mut output);
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_mul64() {
        type P = CpuProver<BabyBearPoseidon2, RwasmAir<BabyBear>>;

        let b = 12345u32;
        let c = 54321u32;
        let correct_res = (b as i32 as i64) * (c as i32 as i64);
        let wrong_res = correct_res.wrapping_add(1);

        let program = rwasm_executor::Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Mul64,
            Opcode::Drop,
            Opcode::Drop,
        ]);
        let stdin = SP1Stdin::new();

        let malicious = move |prover: &P, record: &mut ExecutionRecord| {
            let mut malicious_record = record.clone();
            if let Some(event) =
                malicious_record.i64_events.iter_mut().find(|e| e.opcode == Opcode::I32Mul64)
            {
                event.a_lo = wrong_res as u32;
                event.a_hi = (wrong_res >> 32) as u32;
            }
            prover.generate_traces(&malicious_record)
        };

        let result = run_malicious_test::<P>(program, stdin, Box::new(malicious));
        let chip_name = chip_name!(Mul64Chip, BabyBear);
        assert!(result.is_err() && result.unwrap_err().is_constraints_failing(&chip_name));
    }
}
