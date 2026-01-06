use crate::{air::WordAirBuilder, memory::MemoryCols};
use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::mem_index::{LAST_SIG_ADDR, UNIT};
use rwasm_executor::{ByteOpcode, Opcode, DEFAULT_CLK_INC, DEFAULT_PC_INC};
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
    memory::StackAddressCols,
};
use rwasm_executor::UNUSED_PC;

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

        let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // We represent the `clk` with a 16 bit limb and a 8 bit limb.
        // The range checks for these limbs are done in `eval_shard_clk`.
        let clk =
            AB::Expr::from_canonical_u32(1u32 << 16) * local.clk_8bit_limb + local.clk_16bit_limb;

        // Program constraints.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        // The `pc` and `instruction` is taken from the `ProgramChip`, where these are preprocessed.
        builder.send_program(
            local.pc,
            local.instruction.opcode,
            local.instruction.aux_val,
            local.is_real,
        );

        // Assert the shard and clk to send.  Only the memory and syscall instructions need the
        // actual shard and clk values for memory access evals.
        // SAFETY: The usage of `builder.if_else` requires `is_memory + is_syscall` to be boolean.
        // The correctness of `is_memory` and `is_syscall` will be checked in the opcode specific
        // chips. In these correct cases, `is_memory + is_syscall` will be always boolean.
        let expected_shard_to_send = builder.if_else(
            local.is_memory + local.is_syscall + local.instruction.is_call_ins,
            local.shard,
            AB::Expr::zero(),
        );
        let expected_clk_to_send = builder.if_else(
            local.is_memory + local.is_syscall + local.is_halt + local.instruction.is_call_ins,
            clk.clone(),
            AB::Expr::zero(),
        );
        builder.when(local.is_real).assert_eq(local.shard_to_send, expected_shard_to_send);
        builder.when(local.is_real).assert_eq(local.clk_to_send, expected_clk_to_send);

        self.eval_alu(builder, local);
        self.eval_alu_i64(builder, local, clk.clone());
        self.eval_branching(builder, local);
        self.eval_fuel(builder, local, clk.clone());
        self.eval_call(builder, local, next, clk.clone());
        self.eval_memory(builder, local);
        self.eval_local(builder, local, clk.clone());
        self.eval_ecall(builder, local);

        // Check that the shard and clk is updated correctly.
        self.eval_shard_clk(builder, local, next, public_values, clk.clone());

        // Check that the pc is updated correctly.
        self.eval_pc(builder, local, next, public_values);

        //Check memory for instruction operation
        self.eval_op_memory_sp(builder, local, clk);
        //check sp consistence
        builder.when(local.is_real).when(next.is_real).assert_eq(local.next_sp, next.sp);

        // Always range check the word value in `op_a`, as JUMP instructions and `HINT_LEN` syscall
        // may witness an invalid word and write it to memory.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        builder.slice_range_check_u8(&local.op_res_access.access.value.0, local.is_real);
        builder.slice_range_check_u8(
            &local.op_res_hi_access.access.value.0,
            local.instruction.is_64b_op,
        );

        // Check that the is_real flag is correct.
        self.eval_is_real(builder, local, next);

        // Check that when `is_real=0` that all flags that send interactions are zero.
        let not_real = AB::Expr::one() - local.is_real;
        builder.when(not_real.clone()).assert_zero(AB::Expr::one() - local.is_syscall);

        StackAddressCols::<AB::F>::range_check(builder, local.op_res_addr);
        StackAddressCols::<AB::F>::range_check(builder, local.op_res_hi_addr);
        StackAddressCols::<AB::F>::range_check(builder, local.op_arg1_addr);
        StackAddressCols::<AB::F>::range_check(builder, local.op_arg2_addr);
    }
}

impl CpuChip {
    pub(crate) fn eval_alu_i64<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.send_64_instruction(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_res_hi_val(),
            local.op_arg1_val(),
            local.op_arg2_val(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.instruction.is_64b_op,
        );

        builder.eval_memory_access(
            local.shard,
            clk + AB::Expr::one(),
            local.op_res_hi_addr.value::<AB>(),
            &local.op_res_hi_access,
            local.instruction.is_64b_op,
        );
    }
    pub(crate) fn eval_alu<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CpuCols<AB::Var>) {
        // Send the instruction.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        // The `shard`, `clk`, `pc` are constrained throughout the CpuChip.
        // The `local.instruction.opcode`, `local.instruction.op_a_0` are from the ProgramChip.
        // The `local.op_b_val()` and `local.op_c_val()` are constrained in `eval_registers` in the
        // CpuChip. Therefore, opcode specific chips that will receive this instruction need
        // to the following.
        // - For an instruction with a valid opcode, exactly one opcode specific chip can receive
        //   the instruction.
        // - The `next_pc`, `num_extra_cycles`, `op_a_val`, `op_a_immutable`, `is_memory`,
        //   `is_syscall`, `is_halt` are constrained correctly.
        // Note that in this case, `shard_to_send` and `clk_to_send` will be correctly constrained
        // as well. If `instruction.op_a_0 == 1`, then `eval_registers` enforces `op_a_val()
        // == 0`. Therefore, in this case, `op_a_val` doesn't need to be constrained in the
        // opcode specific chips.
        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_arg1_val(),
            local.op_arg2_val(),
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_ordinary_alu,
        );

        // Calculate a_lt_b <==> a < b (using appropriate signedness).
        let use_signed_comparison = local.instruction.is_i32ges +
            local.instruction.is_i32gts +
            local.instruction.is_i32les +
            local.instruction.is_i32lts;
        let comparison_alu = local.alu_cols;
        let is_comparison = local.instruction.is_comparison_alu;
        // assert that all comparison variable are bool
        builder.when(is_comparison).assert_bool(comparison_alu.res_bool);
        builder
            .when(is_comparison)
            .assert_eq(comparison_alu.res_bool, local.op_res_val().reduce::<AB>());
        builder.when(is_comparison).assert_bool(local.alu_cols.arg1_eq_arg2);
        builder.when(is_comparison).assert_bool(local.alu_cols.arg1_lt_arg2);
        builder.when(is_comparison).assert_bool(local.alu_cols.arg1_gt_arg2);
        builder.when(is_comparison).assert_bool(
            local.alu_cols.arg1_eq_arg2 + local.alu_cols.arg1_gt_arg2 + local.alu_cols.arg1_lt_arg2,
        );
        builder
            .when(comparison_alu.arg1_eq_arg2)
            .assert_word_eq(local.op_arg1_val(), local.op_arg2_val());
        builder
            .when(local.instruction.is_i32lts + local.instruction.is_i32ltu)
            .assert_eq(local.alu_cols.res_bool, local.alu_cols.arg1_lt_arg2);
        builder
            .when(local.instruction.is_i32gts + local.instruction.is_i32gtu)
            .assert_eq(local.alu_cols.res_bool, local.alu_cols.arg1_gt_arg2);

        builder
            .when(local.instruction.is_i32les + local.instruction.is_i32leu)
            .assert_eq(local.alu_cols.res_bool, AB::Expr::one() - local.alu_cols.arg1_gt_arg2);

        builder
            .when(local.instruction.is_i32ges + local.instruction.is_i32geu)
            .assert_eq(local.alu_cols.res_bool, AB::Expr::one() - local.alu_cols.arg1_lt_arg2);
        builder
            .when(local.instruction.is_i32ne)
            .assert_eq(AB::Expr::one() - local.alu_cols.res_bool, local.alu_cols.arg1_eq_arg2);
        builder
            .when(local.instruction.is_i32eqz + local.instruction.is_i32eq)
            .assert_eq(local.alu_cols.res_bool, local.alu_cols.arg1_eq_arg2);
        builder.when(local.instruction.is_i32eqz).assert_word_zero(local.op_arg2_val());

        // Create flags to determine which checks are necessary based on the opcode.
        // A `lt` check is needed for all comparisons except `gt` and `le`.
        let needs_lt_check = is_comparison -
            (local.instruction.is_i32gts +
                local.instruction.is_i32gtu +
                local.instruction.is_i32les +
                local.instruction.is_i32leu);
        // A `gt` check is needed for all comparisons except `lt` and `ge`.
        let needs_gt_check = is_comparison -
            (local.instruction.is_i32lts +
                local.instruction.is_i32ltu +
                local.instruction.is_i32ges +
                local.instruction.is_i32geu);

        let cmp_ins_expr = use_signed_comparison.clone() *
            AB::Expr::from_canonical_u32(Opcode::I32LtS.code()) +
            (AB::Expr::one() - use_signed_comparison.clone()) *
                AB::Expr::from_canonical_u32(Opcode::I32LtU.code());

        // Conditionally send the `lt` check to the ALU table.
        builder.send_instruction_old(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            cmp_ins_expr.clone(),
            Word::extend_var::<AB>(comparison_alu.arg1_lt_arg2),
            local.op_arg1_val(),
            local.op_arg2_val(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            needs_lt_check,
        );

        // Conditionally send the `gt` check to the ALU table.
        builder.send_instruction_old(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            cmp_ins_expr.clone(),
            Word::extend_var::<AB>(comparison_alu.arg1_gt_arg2),
            local.op_arg2_val(), // Operands swapped to check `arg2 < arg1`
            local.op_arg1_val(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            needs_gt_check,
        );
    }

    fn eval_memory<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CpuCols<AB::Var>) {
        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_arg1_val(),
            local.instruction.aux_val,
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_memory,
        );
    }

    fn eval_local<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.op_arg1_addr.value::<AB>(),
            &local.op_arg1_access,
            local.instruction.is_localget,
        );

        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.op_arg1_addr.value::<AB>(),
            &local.op_arg1_access,
            local.instruction.is_localset + local.instruction.is_localtee,
        );

        builder.eval_memory_access(
            local.shard,
            clk.clone() + AB::Expr::one(),
            local.op_res_addr.value::<AB>(),
            &local.op_res_access,
            local.instruction.is_localset + local.instruction.is_localtee,
        );

        builder.when(local.instruction.is_localget).assert_eq(
            local.op_arg1_addr.value::<AB>(),
            local.sp +
                local.instruction.aux_val.reduce::<AB>() * AB::Expr::from_canonical_u32(UNIT) -
                AB::Expr::from_canonical_u32(UNIT),
        );
        builder
            .when(local.instruction.is_localset + local.instruction.is_localtee)
            .assert_eq(local.op_arg1_addr.value::<AB>(), local.sp);

        builder.when(local.instruction.is_localset + local.instruction.is_localtee).assert_eq(
            local.op_res_addr.value::<AB>(),
            local.next_sp +
                local.instruction.aux_val.reduce::<AB>() * AB::Expr::from_canonical_u32(UNIT) -
                AB::Expr::from_canonical_u32(UNIT),
        );

        builder
            .when(local.instruction.is_localset)
            .assert_eq(local.next_sp, local.sp + AB::Expr::from_canonical_u32(UNIT));
        builder.when(local.instruction.is_localtee).assert_eq(local.next_sp, local.sp);
        // assert that what has been read is write to memory
        builder
            .when(local.instruction.is_local)
            .assert_word_eq(local.op_res_val(), local.op_arg1_val());
    }

    pub(crate) fn eval_branching<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
    ) {
        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.instruction.aux_val,
            local.op_arg1_val(),
            Word::zero::<AB>(),
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_br + local.instruction.is_brifeqz + local.instruction.is_brifnez,
        );

        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            Word::zero::<AB>(),
            local.op_arg1_val(),
            local.instruction.aux_val,
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_brtable,
        );
        //op_a_val is always the offset when branch
        builder
            .when(
                local.instruction.is_br +
                    local.instruction.is_brifeqz +
                    local.instruction.is_brifnez,
            )
            .assert_word_eq(local.instruction.aux_val, local.op_res_val());
    }

    pub(crate) fn eval_ecall<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CpuCols<AB::Var>) {
        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_arg1_val(),
            local.instruction.aux_val,
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_ecall,
        );
    }

    pub(crate) fn eval_fuel<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.send_64_instruction(
            local.shard,
            clk.clone(),
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_res_hi_val(),
            Word::zero::<AB>(),
            local.instruction.aux_val,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.instruction.is_consume_fuel,
        );
        builder.send_64_instruction(
            local.shard,
            clk.clone(),
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_res_val(),
            local.op_res_hi_val(),
            Word::zero::<AB>(),
            *local.op_arg2_access.value(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.instruction.is_consume_fuel_stack,
        );
        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.op_arg2_addr.value::<AB>(),
            &local.op_arg2_access,
            local.instruction.is_consume_fuel_stack,
        );
    }

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
        // Verify the public value's shard.
        // builder.when(local.is_real).assert_eq(public_values.execution_shard, local.shard);

        // // Verify that all shard values are the same.
        // builder.when_transition().when(next.is_real).assert_eq(local.shard, next.shard);

        // Verify that the shard value is within 16 bits.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            local.shard,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
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
        builder.when_transition().when(next.is_real).assert_eq(expected_next_clk, next_clk);

        // Range check that the clk is within 24 bits using it's limb values.
        // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
        builder.eval_range_check_24bits(
            clk,
            local.clk_16bit_limb,
            local.clk_8bit_limb,
            local.is_real,
        );
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
        // Verify the public value's start pc.
        builder.when_first_row().assert_eq(public_values.start_pc, local.pc);

        // Verify that the next row's `pc` is the current row's `next_pc`.
        builder.when_transition().when(next.is_real).assert_eq(local.next_pc, next.pc);

        // Verify the public value's next pc.  We need to handle two cases:
        // 1. The last real row is a transition row.
        // 2. The last real row is the last row.

        // If the last real row is a transition row, verify the public value's next pc.
        builder
            .when_transition()
            .when(local.is_real - next.is_real)
            .assert_eq(public_values.next_pc, local.next_pc);

        // If the last real row is the last row, verify the public value's next pc.
        builder.when_last_row().when(local.is_real).assert_eq(public_values.next_pc, local.next_pc);
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
        // Check the is_real flag.  It should be 1 for the first row.  Once its 0, it should never
        // change value.
        builder.assert_bool(local.is_real);
        builder.when_first_row().assert_one(local.is_real);
        builder.when_transition().when_not(local.is_real).assert_zero(next.is_real);

        // If we're halting and it's a transition, then the next.is_real should be 0.
        builder.when_transition().when(local.is_halt).assert_zero(next.is_real);
    }

    pub(crate) fn eval_op_memory_sp<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.eval_memory_access(
            local.shard,
            clk.clone() + AB::Expr::from_canonical_u8(1),
            local.op_res_addr.value::<AB>(),
            &local.op_res_access,
            local.instruction.is_binary +
                local.instruction.is_unary +
                local.instruction.is_i32load +
                local.instruction.is_i32load16s +
                local.instruction.is_i32load16u +
                local.instruction.is_i32load8s +
                local.instruction.is_i32load8u +
                local.instruction.is_localget +
                local.instruction.is_i32const +
                local.instruction.is_64b_op,
        );
        self.eval_op_memory_increase_sp(builder, local, clk.clone());
        self.eval_op_memory_decrease_sp(builder, local, clk.clone());
        self.eval_binary_op_memory_sp(builder, local, clk.clone());
        self.eval_unary_op_memory_sp(builder, local, clk.clone());
    }

    pub(crate) fn eval_op_memory_increase_sp<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder
            .when(local.instruction.is_localget + local.instruction.is_i32const)
            .assert_eq(local.next_sp, local.op_res_addr.value::<AB>());

        builder
            .when(local.instruction.is_localget + local.instruction.is_i32const)
            .assert_eq(local.sp - AB::Expr::from_canonical_u32(UNIT), local.next_sp);
    }

    pub(crate) fn eval_op_memory_decrease_sp<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.eval_memory_access(
            local.shard,
            clk,
            local.sp,
            &local.op_arg1_access,
            local.instruction.is_brifeqz +
                local.instruction.is_brifnez +
                local.instruction.is_brtable +
                local.instruction.is_callindirect,
        );

        builder
            .when(
                local.instruction.is_brifeqz +
                    local.instruction.is_brifnez +
                    local.instruction.is_brtable,
            )
            .assert_eq(local.sp + AB::Expr::from_canonical_u32(UNIT), local.next_sp);
    }

    pub(crate) fn eval_unary_op_memory_sp<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder
            .when(
                local.instruction.is_unary +
                    local.instruction.is_i32load +
                    local.instruction.is_i32load16s +
                    local.instruction.is_i32load16u +
                    local.instruction.is_i32load8s +
                    local.instruction.is_i32load8u,
            )
            .assert_eq(local.op_res_addr.value::<AB>(), local.next_sp);
        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp,
            &local.op_arg1_access,
            local.instruction.is_unary +
                local.instruction.is_i32load +
                local.instruction.is_i32load16s +
                local.instruction.is_i32load16u +
                local.instruction.is_i32load8s +
                local.instruction.is_i32load8u,
        );
        builder
            .when(
                local.instruction.is_unary +
                    local.instruction.is_i32load +
                    local.instruction.is_i32load16s +
                    local.instruction.is_i32load16u +
                    local.instruction.is_i32load8s +
                    local.instruction.is_i32load8u,
            )
            .assert_eq(local.sp, local.next_sp);
    }

    pub(crate) fn eval_binary_op_memory_sp<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder
            .when(local.instruction.is_binary)
            .assert_eq(local.op_res_addr.value::<AB>(), local.next_sp);

        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp + AB::Expr::from_canonical_u8(4),
            &local.op_arg1_access,
            local.instruction.is_binary +
                local.instruction.is_64b_op +
                local.instruction.is_i32store +
                local.instruction.is_i32store16 +
                local.instruction.is_i32store8, // + local.instruction.is_table_grow,
        );

        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            local.sp,
            &local.op_arg2_access,
            local.instruction.is_binary +
                local.instruction.is_64b_op +
                local.instruction.is_i32store +
                local.instruction.is_i32store16 +
                local.instruction.is_i32store8, // + local.instruction.is_table_grow,
        );

        builder
            .when(
                local.instruction.is_binary +
                    local.instruction.is_i32store +
                    local.instruction.is_i32store16 +
                    local.instruction.is_i32store8,
            )
            .assert_eq(
                local.sp + AB::Expr::from_canonical_u32(UNIT),
                local.op_arg1_addr.value::<AB>(),
            );
        builder
            .when(
                local.instruction.is_binary +
                    local.instruction.is_i32store +
                    local.instruction.is_i32store16 +
                    local.instruction.is_i32store8,
            )
            .assert_eq(local.sp, local.op_arg2_addr.value::<AB>());
        builder
            .when(local.instruction.is_binary)
            .assert_eq(local.sp + AB::Expr::from_canonical_u32(UNIT), local.next_sp);
        builder
            .when(
                local.instruction.is_i32store +
                    local.instruction.is_i32store16 +
                    local.instruction.is_i32store8,
            )
            .assert_eq(
                local.sp + AB::Expr::from_canonical_u32(UNIT) + AB::Expr::from_canonical_u32(UNIT),
                local.next_sp,
            );
        // set constr for outputs when is_64b_op
        builder.when(local.instruction.is_64b_op).assert_eq(local.sp, local.next_sp);
        builder
            .when(local.instruction.is_64b_op)
            .assert_eq(local.op_res_hi_addr.value::<AB>(), local.next_sp);
        builder.when(local.instruction.is_64b_op).assert_eq(
            local.op_res_addr.value::<AB>(),
            local.next_sp + AB::Expr::from_canonical_u32(UNIT),
        );
    }
    pub(crate) fn eval_call<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.send_call(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.instruction.opcode,
            local.call_data.call_sp,
            local.call_data.next_call_sp,
            local.call_data.func_ref,
            local.call_data.table_id,
            local.call_data.table_idx,
            local.instruction.is_call +
                local.instruction.is_callinternal +
                local.instruction.is_callindirect +
                local.instruction.is_return,
        );

        builder.send_instruction_old(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.instruction.aux_val,
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.is_memory,
            local.is_syscall,
            local.is_halt,
            local.instruction.is_call +
                local.instruction.is_callinternal +
                local.instruction.is_callindirect +
                local.instruction.is_return,
        );

        builder.eval_memory_access(
            local.shard,
            clk.clone(),
            AB::Expr::from_canonical_u32(LAST_SIG_ADDR),
            &local.op_arg1_access,
            local.instruction.is_sig_check,
        );

        builder.eval_memory_access(
            local.shard,
            clk + AB::Expr::one(),
            AB::Expr::from_canonical_u32(LAST_SIG_ADDR),
            &local.op_res_access,
            local.instruction.is_callindirect,
        );

        builder
            .when(local.instruction.is_callindirect)
            .assert_word_eq(local.instruction.aux_val, *local.op_res_access.value());
        builder
            .when(local.instruction.is_sig_check)
            .assert_word_eq(local.instruction.aux_val, *local.op_arg1_access.value());
        builder
            .when(
                AB::Expr::one() -
                    local.instruction.is_call -
                    local.instruction.is_callinternal -
                    local.instruction.is_callindirect -
                    local.instruction.is_return,
            )
            .assert_eq(local.call_data.call_sp, local.call_data.next_call_sp);
        builder
            .when(local.is_real)
            .when(next.is_real)
            .assert_eq(local.call_data.next_call_sp, next.call_data.call_sp);
        builder
            .when(local.instruction.is_return)
            .when(local.call_data.call_sp_is_zero)
            .assert_zero(next.is_real);
        builder
            .when(local.instruction.is_return)
            .when(local.call_data.call_sp_is_zero)
            .assert_zero(local.call_data.call_sp);
    }
}

impl<F> BaseAir<F> for CpuChip {
    fn width(&self) -> usize {
        NUM_CPU_COLS
    }
}
