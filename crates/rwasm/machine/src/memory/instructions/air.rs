use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::mem_index::{GLOBAL_MEM_START, UNIT};
use sp1_stark::{air::SP1AirBuilder, Word};

use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    memory::MemoryCols,
    operations::{BabyBearWordRangeChecker, IsZeroOperation},
};
use rwasm_executor::{ByteOpcode, Opcode, DEFAULT_PC_INC, UNUSED_PC};

use super::{columns::MemoryInstructionsColumns, MemoryInstructionsChip};

impl<AB> Air<AB> for MemoryInstructionsChip
where
    AB: SP1AirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryInstructionsColumns<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_lb`, `is_lbu`, `is_lh`, `is_lhu`, `is_lw`, `is_sb`, `is_sh`, `is_sw` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the eight selectors, is boolean.
        // Therefore, the `opcode` matches the corresponding opcode.

        let is_real = local.is_i32load
            + local.is_i32load16s
            + local.is_i32load16u
            + local.is_i32load8s
            + local.is_i32load8u
            + local.is_i32store8
            + local.is_i32store16
            + local.is_i32store;

        builder.assert_bool(local.is_i32load8s);
        builder.assert_bool(local.is_i32load8u);
        builder.assert_bool(local.is_i32load16s);
        builder.assert_bool(local.is_i32load16u);
        builder.assert_bool(local.is_i32load);
        builder.assert_bool(local.is_i32store8);
        builder.assert_bool(local.is_i32store16);
        builder.assert_bool(local.is_i32store);
        builder.assert_bool(is_real.clone());

        builder
            .when(local.is_multi_aligned_load)
            .when(local.is_i32load)
            .assert_one(local.ls_bits_is_one + local.ls_bits_is_two + local.ls_bits_is_three);
        builder
            .when(local.is_multi_aligned_load)
            .when(local.is_i32load16s + local.is_i32load16u)
            .assert_one(local.ls_bits_is_three);
        builder
            .when(local.is_multi_aligned_store)
            .when(local.is_i32store)
            .assert_one(local.ls_bits_is_one + local.ls_bits_is_two + local.ls_bits_is_three);
        builder
            .when(local.is_multi_aligned_store)
            .when(local.is_i32store16)
            .assert_one(local.ls_bits_is_three);
        let is_store = local.is_i32store8 + local.is_i32store16 + local.is_i32store;
        let is_load = local.is_i32load
            + local.is_i32load16s
            + local.is_i32load16u
            + local.is_i32load8s
            + local.is_i32load8u;
        

        self.eval_memory_address_and_access::<AB>(
            builder,
            local,
            is_real.clone(),
            is_load,
            is_store,
        );
        self.eval_memory_load::<AB>(builder, local);
        self.eval_memory_store::<AB>(builder, local);

        let opcode = self.compute_opcode::<AB>(local);

        // SAFETY: This checks the following.
        // - `shard`, `clk` are correctly received from the CpuChip
        // - `next_pc = pc + 4`
        // - `num_extra_cycles = 0`
        // - `op_a_immutable = is_sb + is_sh + is_sw`, as store instruction keeps `op_a` immutable
        // - `is_memory = 1`
        // - `is_syscall = 0`
        // - `is_halt = 0`
        // `op_a_value` when the instruction is load still has to be constrained, as well as memory opcode behavior.
        builder.receive_instruction(
            local.shard,
            local.clk,
            local.pc,
            local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::zero(),
            opcode,
            local.res,
            local.raw_addr,
            local.instr_offset,
            AB::Expr::one(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );
    }
}

impl MemoryInstructionsChip {
    /// Computes the opcode based on the instruction selectors.
    pub(crate) fn compute_opcode<AB: SP1AirBuilder>(
        &self,
        local: &MemoryInstructionsColumns<AB::Var>,
    ) -> AB::Expr {
        local.is_i32load8s * AB::Expr::from_canonical_u32(Opcode::I32Load8S(0).code())
            + local.is_i32load8u * AB::Expr::from_canonical_u32(Opcode::I32Load8U(0).code())
            + local.is_i32load16s * AB::Expr::from_canonical_u32(Opcode::I32Load16S(0).code())
            + local.is_i32load16u * AB::Expr::from_canonical_u32(Opcode::I32Load16U(0).code())
            + local.is_i32load * AB::Expr::from_canonical_u32(Opcode::I32Load(0).code())
            + local.is_i32store8 * AB::Expr::from_canonical_u32(Opcode::I32Store8(0).code())
            + local.is_i32store16 * AB::Expr::from_canonical_u32(Opcode::I32Store16(0u32).code())
            + local.is_i32store * AB::Expr::from_canonical_u32(Opcode::I32Store(0).code())
    }

    /// Constrains the addr_aligned, addr_offset, and addr_word memory columns.
    ///
    /// This method will do the following:
    /// 1. Calculate that the unaligned address is correctly computed to be op_b.value + op_c.value.
    /// 2. Calculate that the address offset is address % 4.
    /// 3. Assert the validity of the aligned address given the address offset and the unaligned
    ///    address.
    pub(crate) fn eval_memory_address_and_access<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryInstructionsColumns<AB::Var>,
        is_real: AB::Expr,
        is_load: AB::Expr,
        is_store: AB::Expr,
    ) {
        // Send to the ALU table to verify correct calculation of addr_word.
        builder.send_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Add.code()),
            local.memory_addr,
            local.raw_addr,
            local.instr_offset,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Send to the ALU table to verify correct calculation of addr_word.
        builder.send_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(UNUSED_PC),
            AB::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            AB::Expr::zero(),
            AB::Expr::from_canonical_u32(Opcode::I32Add.code()),
            local.addr_word,
            Word::<AB::Expr>::from(GLOBAL_MEM_START),
            local.memory_addr,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Range check the addr_word to be a valid babybear word. Note that this will also implicitly
        // do a byte range check on the most significant byte.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.addr_word,
            local.addr_word_range_checker,
            is_real.clone(),
        );

        // Check that the 2nd and 3rd addr_word elements are bytes. We already check the most sig
        // byte in the BabyBearWordRangeChecker, and the least sig one in the AND byte lookup below.
        builder.slice_range_check_u8(&local.addr_word.0[1..3], is_real.clone());

        // We check that `addr_word >= 32`, or `addr_word > 31` to avoid registers.
        // Check that if the most significant bytes are zero, then the least significant byte is at least 32.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            AB::Expr::from_canonical_u8(31),
            local.addr_word[0],
            local.most_sig_bytes_zero.result,
        );

        // SAFETY: Check that the above interaction is only sent if one of the opcode flags is set.
        // If `is_real = 0`, then `local.most_sig_bytes_zero.result = 0`, leading to no interaction.
        // Note that when `is_real = 1`, due to `IsZeroOperation`, `local.most_sig_bytes_zero.result` is boolean.
        builder.when(local.most_sig_bytes_zero.result).assert_one(is_real.clone());

        // Check the most_sig_byte_zero flag.  Note that we can simply add up the three most significant bytes
        // and check if the sum is zero.  Those bytes are going to be byte range checked, so the only way
        // the sum is zero is if all bytes are 0.
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.addr_word[1] + local.addr_word[2] + local.addr_word[3],
            local.most_sig_bytes_zero,
            is_real.clone(),
        );

        // Evaluate the addr_offset column and offset flags.
        self.eval_offset_value_flags(builder, local);

        // Assert that reduce(addr_word) == addr_aligned + addr_ls_two_bits.
        builder.when(is_real.clone()).assert_eq::<AB::Expr, AB::Expr>(
            local.addr_aligned + local.addr_ls_two_bits,
            local.addr_word.reduce::<AB>(),
        );

        // Check the correct value of addr_ls_two_bits. Note that this lookup will implicitly do a
        // byte range check on the least sig addr byte.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.addr_ls_two_bits,
            local.addr_word[0],
            AB::Expr::from_canonical_u8(0b11),
            is_real.clone(),
        );

        // For operations that require reading from memory (not registers), we need to read the
        // value into the memory columns.
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.addr_aligned,
            &local.memory_access,
            is_load.clone(),
        );

        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.addr_aligned,
            &local.memory_access_hi,
            local.is_multi_aligned_load.clone(),
        );

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            local.addr_aligned,
            &local.memory_access,
            is_store.clone(),
        );

        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::one(),
            local.addr_aligned+AB::Expr::from_canonical_u32(UNIT),
            &local.memory_access_hi,
            local.is_multi_aligned_store.clone(),
        );

        // On memory load instructions, make sure that the memory value is not changed.
        builder
            .when(
                local.is_i32load8s
                    + local.is_i32load8u
                    + local.is_i32load16u
                    + local.is_i32load16s
                    + local.is_i32load,
            )
            .assert_word_eq(*local.memory_access.value(), *local.memory_access.prev_value());
        builder.when(local.is_multi_aligned_load)
            .assert_word_eq(*local.memory_access_hi.value(), *local.memory_access_hi.prev_value());
    }

    /// Evaluates constraints related to loading from memory.
    pub(crate) fn eval_memory_load<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryInstructionsColumns<AB::Var>,
    ) {
        // Verify the unsigned_mem_value column.
        self.eval_unsigned_mem_value(builder, local);

        // SAFETY: `is_lb + is_lh` is already constrained to be boolean.
        // This is because at most one opcode selector can be turned on.
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.most_sig_bit,
            local.most_sig_byte,
            AB::Expr::zero(),
            local.is_i32load8s + local.is_i32load16s,
        );
        builder.assert_eq(
            local.most_sig_byte,
            local.is_i32load8s * local.unsigned_mem_val[0]
                + local.is_i32load16s * local.unsigned_mem_val[1],
        );

        // These two cases combine for all cases where it's a load instruction and `op_a_0 == 0`.
        // Since the store instructions have `op_a_immutable = 1`, this completely constrains the `op_a`'s value.
    }

    /// Evaluates constraints related to storing to memory.
    pub(crate) fn eval_memory_store<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryInstructionsColumns<AB::Var>,
    ) {
        // Get the memory offset flags.
        self.eval_offset_value_flags(builder, local);
        // Compute the offset_is_zero flag.  The other offset flags are already constrained by the
        // method `eval_memory_address_and_access`, which is called in
        // `eval_memory_address_and_access`.
        let offset_is_zero =
            AB::Expr::one() - local.ls_bits_is_one - local.ls_bits_is_two - local.ls_bits_is_three;

        // Compute the expected stored value for a SB instruction.
        let one = AB::Expr::one();
        let a_val = local.res;
        let mem_val = *local.memory_access.value();
        let prev_mem_val = *local.memory_access.prev_value();
        let mem_val_hi = *local.memory_access_hi.value();
        let prev_mem_val_hi = *local.memory_access_hi.prev_value();
        let sb_expected_stored_value = Word([
            a_val[0] * offset_is_zero.clone()
                + (one.clone() - offset_is_zero.clone()) * prev_mem_val[0],
            a_val[0] * local.ls_bits_is_one.clone()
                + (one.clone() - local.ls_bits_is_one.clone()) * prev_mem_val[1],
            a_val[0] * local.ls_bits_is_two
                + (one.clone() - local.ls_bits_is_two) * prev_mem_val[2],
            a_val[0] * local.ls_bits_is_three
                + (one.clone() - local.ls_bits_is_three) * prev_mem_val[3],
        ]);
        builder
            .when(local.is_i32store8)
            .assert_word_eq(mem_val.map(|x| x.into()), sb_expected_stored_value);

        // // When the instruction is SH, make sure both offset one and three are off.
        // builder
        //     .when(local.is_i32store16)
        //     .assert_zero(local.ls_bits_is_one + local.ls_bits_is_three);

        // // When the instruction is SW, ensure that the offset is 0.
        // builder.when(local.is_i32store).assert_one(offset_is_zero.clone());

        // Compute the expected stored value for a SH instruction.
       
        let ls_bits_is_two = local.ls_bits_is_two;
        let ls_bits_is_three = local.ls_bits_is_three;
        let ls_bits_is_one = local.ls_bits_is_one;
        let store16_expected_stored_value_lw = Word([
            a_val[0] * offset_is_zero.clone()
                + (one.clone() - offset_is_zero.clone()) * prev_mem_val[0],
            a_val[1] * offset_is_zero.clone()
                + (ls_bits_is_two.clone() + ls_bits_is_three.clone()) * prev_mem_val[1]
                + ls_bits_is_one * a_val[0],
            a_val[0] * ls_bits_is_two.clone()
                + (ls_bits_is_one.clone()) * a_val[1]
                + (ls_bits_is_three.clone() + offset_is_zero.clone()) * prev_mem_val[2],
            a_val[1] * ls_bits_is_two.clone()
                + a_val[0]
                    * ls_bits_is_three.clone()+(ls_bits_is_one.clone() + offset_is_zero.clone())
                    * prev_mem_val[3],
        ]);
        let store16_expected_stored_value_hi = Word([
            ls_bits_is_three * a_val[1]
                + (ls_bits_is_one.clone() + offset_is_zero.clone() + ls_bits_is_two.clone())
                    * prev_mem_val_hi[0],
            prev_mem_val_hi[1] *one.clone(),
            prev_mem_val_hi[2]*one.clone(),
            prev_mem_val_hi[3]*one.clone(),
        ]);
        builder
            .when(local.is_i32store16)
            .assert_word_eq(mem_val.map(|x| x.into()), store16_expected_stored_value_lw);
        builder
            .when(local.is_i32store16)
            .assert_word_eq(mem_val_hi.map(|x| x.into()), store16_expected_stored_value_hi);

        let store_expected_stored_value_lw = Word([
            a_val[0] * offset_is_zero.clone() + prev_mem_val[0] * (one.clone() - offset_is_zero.clone()),
            a_val[0] * ls_bits_is_one.clone()
                + a_val[1] * offset_is_zero.clone()
                + prev_mem_val[1] * (ls_bits_is_three.clone() + ls_bits_is_two.clone()),
            a_val[0] * ls_bits_is_two.clone()
                + a_val[1] * ls_bits_is_one.clone()
                + a_val[2] * offset_is_zero.clone()
                + prev_mem_val[2] * ls_bits_is_three.clone(),
            a_val[0] * ls_bits_is_three.clone()
                + a_val[1] * ls_bits_is_two.clone()
                + a_val[2] * ls_bits_is_one.clone()
                + a_val[3] * offset_is_zero.clone(),
        ]);

        let store_expected_stored_value_hi = Word([
            prev_mem_val_hi[0] * offset_is_zero.clone()
                + a_val[3] * ls_bits_is_one.clone()
                + a_val[2] * ls_bits_is_two.clone()
                + a_val[1] * ls_bits_is_three.clone(),
            prev_mem_val_hi[1] * (offset_is_zero.clone() + ls_bits_is_one.clone())
                + a_val[3] * ls_bits_is_two.clone()
                + a_val[2] * ls_bits_is_three.clone(),
            prev_mem_val_hi[2]*(offset_is_zero.clone()+ls_bits_is_one.clone()+ls_bits_is_two.clone())
            +a_val[3]*ls_bits_is_three,
            prev_mem_val_hi[3].into(),
        ]);

        // When the instruction is SW, just use the word without masking.
        builder
            .when(local.is_i32store)
            .assert_word_eq(mem_val.map(|x| x.into()), store_expected_stored_value_lw);
         builder
            .when(local.is_i32store)
            .assert_word_eq(mem_val_hi.map(|x| x.into()), store_expected_stored_value_hi);

    }

    /// This function is used to evaluate the unsigned memory value for the load memory
    /// instructions.
    pub(crate) fn eval_unsigned_mem_value<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryInstructionsColumns<AB::Var>,
    ) {
        let mem_val = *local.memory_access.value();

        // Compute the offset_is_zero flag.  The other offset flags are already constrained by the
        // method `eval_memory_address_and_access`, which is called in
        // `eval_memory_address_and_access`.
        let offset_is_zero =
            AB::Expr::one() - local.ls_bits_is_one - local.ls_bits_is_two - local.ls_bits_is_three;

        // Compute the byte value.
        let mem_byte = mem_val[0] * offset_is_zero.clone()
            + mem_val[1] * local.ls_bits_is_one
            + mem_val[2] * local.ls_bits_is_two
            + mem_val[3] * local.ls_bits_is_three;
        let byte_value = Word::extend_expr::<AB>(mem_byte.clone());

        // When the instruction is LB or LBU, just use the lower byte.
        builder
            .when(local.is_i32load8s + local.is_i32load8u)
            .assert_word_eq(byte_value, local.unsigned_mem_val.map(|x| x.into()));

        // When the instruction is LH or LHU, ensure that offset is either zero or two.
        builder
            .when(local.is_i32load16s + local.is_i32load16u)
            .assert_zero(local.ls_bits_is_one + local.ls_bits_is_three);

        // When the instruction is LW, ensure that the offset is zero.
        builder.when(local.is_i32load).assert_one(offset_is_zero.clone());

        let use_lower_half = offset_is_zero;
        let use_upper_half = local.ls_bits_is_two;
        let half_value = Word([
            use_lower_half.clone() * mem_val[0] + use_upper_half * mem_val[2],
            use_lower_half * mem_val[1] + use_upper_half * mem_val[3],
            AB::Expr::zero(),
            AB::Expr::zero(),
        ]);
        builder
            .when(local.is_i32load16s + local.is_i32load16u)
            .assert_word_eq(half_value, local.unsigned_mem_val.map(|x| x.into()));

        // When the instruction is LW, just use the word.
        builder.when(local.is_i32load).assert_word_eq(mem_val, local.unsigned_mem_val);
    }

    /// Evaluates the offset value flags.
    pub(crate) fn eval_offset_value_flags<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &MemoryInstructionsColumns<AB::Var>,
    ) {
        let offset_is_zero =
            AB::Expr::one() - local.ls_bits_is_one - local.ls_bits_is_two - local.ls_bits_is_three;

        // Assert that the value flags are boolean
        builder.assert_bool(local.ls_bits_is_one);
        builder.assert_bool(local.ls_bits_is_two);
        builder.assert_bool(local.ls_bits_is_three);

        // Assert that only one of the value flags is true
        builder.assert_one(
            offset_is_zero.clone()
                + local.ls_bits_is_one
                + local.ls_bits_is_two
                + local.ls_bits_is_three,
        );

        // Assert that the correct value flag is set
        // SAFETY: Due to the constraints here, at most one of the four flags can be turned on (non-zero).
        // As their sum is constrained to be 1, the only possibility is that exactly one flag is on, with value 1.
        builder.when(offset_is_zero).assert_zero(local.addr_ls_two_bits);
        builder.when(local.ls_bits_is_one).assert_one(local.addr_ls_two_bits);
        builder.when(local.ls_bits_is_two).assert_eq(local.addr_ls_two_bits, AB::Expr::two());
        builder
            .when(local.ls_bits_is_three)
            .assert_eq(local.addr_ls_two_bits, AB::Expr::from_canonical_u8(3));
    }
}
