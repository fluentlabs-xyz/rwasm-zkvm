mod stack;

use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm_executor::{ByteOpcode, DEFAULT_CLK_INC};
use sp1_stark::{
    air::{BaseAirBuilder, PublicValues, SP1AirBuilder, SP1_PROOF_NUM_PV_ELTS},
    Word,
};

use crate::{
    air::{MemoryAirBuilder, SP1CoreAirBuilder},
    cpu::{
        columns::{CpuCols, NUM_CPU_COLS},
        CpuChip,
    },
};

impl<AB> Air<AB> for CpuChip
where
    AB: SP1CoreAirBuilder + AirBuilderWithPublicValues,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &CpuCols<AB::Var> = (*local).borrow();
        let next: &CpuCols<AB::Var> = (*next).borrow();

        let is_real = local.instruction.is_with_zero_params +
            local.instruction.is_with_one_param +
            local.instruction.is_with_two_three_params;

        let next_is_real = next.instruction.is_with_zero_params +
            next.instruction.is_with_one_param +
            next.instruction.is_with_two_three_params;

        let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // We represent the `clk` with a 16 bit limb and a 8 bit limb.
        // The range checks for these limbs are done in `eval_shard_clk`.
        let clk =
            AB::Expr::from_canonical_u32(1u32 << 16) * local.clk_8bit_limb + local.clk_16bit_limb;

        // Program constraints.
        // SAFETY: `is_real` is checked to be boolean in `eval_is_real`.
        // The `pc` and `instruction` is taken from the `ProgramChip`, where these are preprocessed.
        builder.send_program(
            local.pc,
            local.instruction.opcode,
            local.instruction.aux_val,
            is_real.clone(),
        );

        self.eval_stack(builder, local, next, clk.clone());

        // Assert the shard and clk to send.  Only the memory and syscall instructions need the
        // actual shard and clk values for memory access evals.
        // SAFETY: The usage of `builder.if_else` requires `is_memory + is_syscall` to be boolean.
        // The correctness of `is_memory` and `is_syscall` will be checked in the opcode specific
        // chips. In these correct cases, `is_memory + is_syscall` will be always boolean.
        let expected_shard_to_send = builder.if_else(
            local.is_memory + local.is_syscall + local.is_complex_opcode,
            local.shard,
            AB::Expr::zero(),
        );
        let expected_clk_to_send = builder.if_else(
            local.is_memory + local.is_syscall + local.is_halt + local.is_complex_opcode,
            clk.clone(),
            AB::Expr::zero(),
        );
        builder.when(is_real.clone()).assert_eq(local.shard_to_send, expected_shard_to_send);
        builder.when(is_real.clone()).assert_eq(local.clk_to_send, expected_clk_to_send);

        builder.when(local.instruction.is_implemented).assert_one(is_real.clone());

        builder.send_rwasm_instruction(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.sp.value::<AB>(),
            local.next_sp.value::<AB>(),
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_arg1_val(),
            local.op_arg2_val(),
            local.instruction.aux_val,
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.is_complex_opcode,
            local.instruction.is_implemented,
        );

        // Check that the shard and clk is updated correctly.
        self.eval_shard_clk(builder, local, next, public_values, clk.clone());

        // // Check that the pc is updated correctly.
        self.eval_pc(builder, local, next, public_values);

        // Check that the is_real flag is correct.
        self.eval_is_real(builder, local, next);

        // Check that when `is_real=0` that all flags that send interactions are zero.
        let not_real = AB::Expr::one() - is_real;
        builder.when(not_real.clone()).assert_zero(AB::Expr::one() - local.is_syscall);
    }
}

impl CpuChip {
    /// Constraints related to the shard and clk.
    ///
    /// This method ensures that all of the shard values are the same and that the clk starts at 0
    /// and is transitioned appropriately.  It will also check that shard values are within 16 bits
    /// and clk values are within 24 bits.  Those range checks are needed for the memory access
    /// timestamp check, which assumes those values are within 2^24.  See
    /// [`MemoryAirBuilder::verify_mem_access_ts`].
    pub(crate) fn eval_shard_clk<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
        clk: AB::Expr,
    ) {
        let is_real = local.instruction.is_with_zero_params +
            local.instruction.is_with_one_param +
            local.instruction.is_with_two_three_params;

        let next_is_real = next.instruction.is_with_zero_params +
            next.instruction.is_with_one_param +
            next.instruction.is_with_two_three_params;

        // Verify the public value's shard.
        builder.when(is_real.clone()).assert_eq(public_values.execution_shard, local.shard);

        // Verify that all shard values are the same.
        builder.when_transition().when(next_is_real.clone()).assert_eq(local.shard, next.shard);

        // Verify that the shard value is within 16 bits.
        // SAFETY: `is_real` is checked to be boolean in `eval_is_real`.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            local.shard,
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Verify that the first row has a clk value of 0.
        builder.when_first_row().assert_zero(clk.clone());

        // We already assert that `local.clk < 2^24`. `num_extra_cycles` is an entry of a word and
        // therefore less than `2^8`, this means that the sum cannot overflow in a 31 bit field.
        // The default clk increment is also `4`, equal to `DEFAULT_PC_INC`.
        let expected_next_clk =
            clk.clone() + AB::Expr::from_canonical_u32(DEFAULT_CLK_INC) + local.num_extra_cycles;

        let next_clk =
            AB::Expr::from_canonical_u32(1u32 << 16) * next.clk_8bit_limb + next.clk_16bit_limb;
        builder.when_transition().when(next_is_real).assert_eq(expected_next_clk, next_clk);

        // Range check that the clk is within 24 bits using it's limb values.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        builder.eval_range_check_24bits(clk, local.clk_16bit_limb, local.clk_8bit_limb, is_real);
    }

    /// Constraints related to the pc for non jump, branch, and halt instructions.
    ///
    /// The function will verify that the pc increments by 4 for all instructions except branch,
    /// jump and halt instructions. Also, it ensures that the pc is carried down to the last row
    /// for non-real rows.
    pub(crate) fn eval_pc<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
    ) {
        let is_real = local.instruction.is_with_zero_params +
            local.instruction.is_with_one_param +
            local.instruction.is_with_two_three_params;

        let next_is_real = next.instruction.is_with_zero_params +
            next.instruction.is_with_one_param +
            next.instruction.is_with_two_three_params;

        // Verify the public value's start pc.
        builder.when_first_row().assert_eq(public_values.start_pc, local.pc);

        // Verify that the next row's `pc` is the current row's `next_pc`.
        builder.when_transition().when(next_is_real.clone()).assert_eq(local.next_pc, next.pc);

        // If the last real row is the last row, verify the public value's next pc.
        builder
            .when_transition()
            .when(is_real - next_is_real)
            .assert_eq(public_values.next_pc, local.next_pc);
    }

    /// Constraints related to the is_real column.
    ///
    /// This method checks that the is_real column is a boolean.  It also checks that the first row
    /// is 1 and once its 0, it never changes value.
    pub(crate) fn eval_is_real<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
    ) {
        let is_real = local.instruction.is_with_zero_params +
            local.instruction.is_with_one_param +
            local.instruction.is_with_two_three_params;

        let next_is_real = next.instruction.is_with_zero_params +
            next.instruction.is_with_one_param +
            next.instruction.is_with_two_three_params;

        // Check the is_real flag.  It should be 1 for the first row.  Once its 0, it should never
        // change value.
        builder.assert_bool(is_real.clone());
        builder.when_first_row().assert_one(is_real.clone());
        builder.when_transition().when_not(is_real.clone()).assert_zero(next_is_real.clone());

        // If we're halting and it's a transition, then the next.is_real should be 0.
        builder.when_transition().when(local.is_halt).assert_zero(next_is_real);
    }
}

impl<F> BaseAir<F> for CpuChip {
    fn width(&self) -> usize {
        NUM_CPU_COLS
    }
}
