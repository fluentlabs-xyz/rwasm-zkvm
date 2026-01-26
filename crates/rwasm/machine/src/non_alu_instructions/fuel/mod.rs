use crate::{air::WordAirBuilder, alu::BYTE_SIZE, utils::pad_rows_fixed};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rwasm::{mem_index::UNIT, Opcode};
use rwasm_executor::{
    events::{ByteLookupEvent, ByteRecord, FuelEvent},
    ExecutionRecord, Program, DEFAULT_PC_INC, UNUSED_PC,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{LONG_WORD_SIZE, WORD_SIZE};
use sp1_stark::{
    air::{BaseAirBuilder, PublicValues, SP1AirBuilder, SP1_PROOF_NUM_PV_ELTS},
    Word,
};
use std::borrow::{Borrow, BorrowMut};

use sp1_stark::air::MachineAir;

pub const NUM_FUEL_COLS: usize = size_of::<FuelCols<u8>>();

#[derive(Default)]
pub struct FuelChip;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct FuelCols<T> {
    /// The program counter.
    pub pc: T,

    /// The current stack pointer.
    pub sp: T,

    pub fuel_consumed: (Word<T>, Word<T>),
    // TODO: find way to use interactions with public values
    pub fuel_limit: (Word<T>, Word<T>),

    pub carry: [T; 8],

    pub delta: Word<T>,

    pub last_hi_is_eq: T,
    pub last_hi_non_eq: T,

    pub is_consume_fuel: T,
    pub is_consume_fuel_stack: T,
}

impl<F> BaseAir<F> for FuelChip {
    fn width(&self) -> usize {
        NUM_FUEL_COLS
    }
}

impl<AB> Air<AB> for FuelChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FuelCols<AB::Var> = (*local).borrow();
        let next: &FuelCols<AB::Var> = (*next).borrow();

        let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        let base = AB::F::from_canonical_u32(1 << BYTE_SIZE);

        let [fuel_limit_low, fuel_limit_hi] = public_values.fuel_limit;

        builder.assert_bool(local.is_consume_fuel);
        builder.assert_bool(local.is_consume_fuel_stack);

        let is_real = local.is_consume_fuel + local.is_consume_fuel_stack;

        let next_is_real = next.is_consume_fuel + next.is_consume_fuel_stack;

        builder.assert_bool(is_real.clone());

        builder.slice_range_check_u8(&local.fuel_consumed.0 .0, is_real.clone());
        builder.slice_range_check_u8(&local.fuel_consumed.1 .0, is_real.clone());

        builder.when(is_real.clone()).assert_word_eq(local.fuel_limit.0, fuel_limit_low);
        builder.when(is_real.clone()).assert_word_eq(local.fuel_limit.1, fuel_limit_hi);

        builder.when_first_row().assert_word_eq(local.fuel_consumed.0, local.delta);

        let mut prev_carry = AB::Expr::zero();
        for i in 0..WORD_SIZE {
            let lhs = local.fuel_consumed.0[i] + next.delta[i].into() + prev_carry.clone();
            let rhs = next.fuel_consumed.0[i].into() + local.carry[i].into() * base;

            builder
                .when_transition()
                .when(is_real.clone())
                .when(next_is_real.clone())
                .assert_eq(lhs, rhs);
            prev_carry = local.carry[i].into();
        }

        for i in WORD_SIZE..LONG_WORD_SIZE {
            let lhs = local.fuel_consumed.1[i - WORD_SIZE] + prev_carry.clone();
            let rhs = next.fuel_consumed.1[i - WORD_SIZE].into() + local.carry[i].into() * base;

            builder
                .when_transition()
                .when(is_real.clone())
                .when(next_is_real.clone())
                .assert_eq(lhs, rhs);
            prev_carry = local.carry[i].into();
        }

        builder
            .when_transition()
            .when(is_real.clone())
            .when(next_is_real.clone())
            .assert_zero(prev_carry);

        builder.when_transition().when_not(is_real.clone()).assert_zero(next_is_real.clone());
        builder
            .when_transition()
            .when(local.last_hi_is_eq + local.last_hi_non_eq)
            .assert_zero(next_is_real.clone());

        builder
            .when_last_row()
            .when(is_real.clone())
            .assert_one(local.last_hi_is_eq + local.last_hi_non_eq);

        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32LeU.code()),
            Word::extend_expr::<AB>(AB::Expr::one()),
            local.fuel_consumed.1,
            local.fuel_limit.1,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.last_hi_non_eq,
        );

        builder.send_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32LeU.code()),
            Word::extend_expr::<AB>(AB::Expr::one()),
            local.fuel_consumed.0,
            local.fuel_limit.0,
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.last_hi_is_eq,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::ConsumeFuel(0).code()),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.delta,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_consume_fuel,
        );

        builder.receive_rwasm_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::ConsumeFuelStack.code()),
            Word::zero::<AB>(),
            local.delta,
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_consume_fuel_stack,
        );
    }
}

impl<F: PrimeField32> MachineAir<F> for FuelChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Fuel".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();

        let last_fuel_consumed =
            input.fuel_events.last().map(|event| event.fuel_consumed).unwrap_or(0);

        for (event, next_event) in input
            .fuel_events
            .iter()
            .zip(input.fuel_events.iter().skip(1).map(Some).chain(std::iter::once(None)))
        {
            let mut row = [F::zero(); NUM_FUEL_COLS];
            let cols: &mut FuelCols<F> = row.as_mut_slice().borrow_mut();

            self.event_to_row(
                event,
                next_event,
                cols,
                &mut Vec::new(),
                input.public_values.fuel_limit,
            );

            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::zero(); NUM_FUEL_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<F>>(), NUM_FUEL_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let nb_rows = input.fuel_events.len();

        if nb_rows == 0 {
            return;
        }

        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let num_chunks = nb_rows.div_ceil(chunk_size);

        let blu_batches = input
            .fuel_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_FUEL_COLS];
                    let cols: &mut FuelCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, None, cols, &mut blu, input.public_values.fuel_limit);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.fuel_events.is_empty()
    }

    fn local_only(&self) -> bool {
        false
    }
}

impl FuelChip {
    fn event_to_row<F: PrimeField>(
        &self,
        event: &FuelEvent,
        next_event: Option<&FuelEvent>,
        cols: &mut FuelCols<F>,
        blu: &mut impl ByteRecord,
        fuel_limit: [u32; 2],
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.sp = F::from_canonical_u32(event.sp);

        match event.opcode {
            Opcode::ConsumeFuel(aux_val) => {
                cols.delta = aux_val.into();
                cols.is_consume_fuel = F::one();
            }
            Opcode::ConsumeFuelStack => {
                cols.delta = event.arg1.into();
                cols.is_consume_fuel_stack = F::one();
            }
            _ => unreachable!(),
        };

        let fuel_consumed_word = event.fuel_consumed.to_le_bytes();

        let next_event_delta = match next_event.map(|x| x.opcode) {
            Some(Opcode::ConsumeFuel(aux_val)) => aux_val,
            Some(Opcode::ConsumeFuelStack) => next_event.unwrap().arg1,
            _ => 0,
        };

        let delta_word = next_event_delta.to_le_bytes();

        let mut carry_in: u16 = 0;
        for i in 0..WORD_SIZE {
            let s = (fuel_consumed_word[i] as u16) + (delta_word[i] as u16) + carry_in;
            let carry_out = s >> 8;

            cols.carry[i] = F::from_canonical_u32(carry_out as u32);
            carry_in = carry_out;
        }

        cols.fuel_consumed =
            ((event.fuel_consumed as u32).into(), ((event.fuel_consumed >> 32) as u32).into());

        cols.fuel_limit = (fuel_limit[0].into(), fuel_limit[1].into());

        if next_event.is_none() {
            let last_hi_is_eq = event.fuel_consumed >> 32 == fuel_limit[1] as u64;
            cols.last_hi_is_eq = F::from_bool(last_hi_is_eq);
            cols.last_hi_non_eq = F::from_bool(!last_hi_is_eq);
        }

        {
            blu.add_u8_range_checks(&event.fuel_consumed.to_le_bytes());
        }
    }
}
