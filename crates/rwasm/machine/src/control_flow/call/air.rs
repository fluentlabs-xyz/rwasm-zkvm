use super::{CallChip, CallColumns};
use crate::{
    air::{SP1CoreAirBuilder, WordAirBuilder},
    control_flow::{FuncIndex, TableIdxCols},
    memory::{CallStackAddressCols, MemoryCols},
    operations::BabyBearWordRangeChecker,
};
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rwasm::{
    mem_index::{TypedAddress, TABLE_SEG_START, UNIT},
    N_MAX_TABLE_SIZE,
};
use rwasm_executor::Opcode;
use sp1_stark::{
    air::{BaseAirBuilder, SP1AirBuilder},
    Word,
};
use std::borrow::Borrow;

// --- Constants ---
const TABLE_MEMORY_SHIFT: u32 = TABLE_SEG_START;
const N_ONE_TABLE_MEMORY_LENGTH: u32 = N_MAX_TABLE_SIZE * UNIT;

impl<AB> Air<AB> for CallChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        //TODO: add signature check and rangecheck

        // --- 1. Setup Columns and Flags ---
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);
        let local: &CallColumns<AB::Var> = (*local).borrow();
        let next: &CallColumns<AB::Var> = (*next).borrow();

        // Enforce boolean flags for operation types (mutually exclusive)
        builder.assert_bool(local.is_call);
        builder.assert_bool(local.is_call_indirect);
        builder.assert_bool(local.is_call_internal);
        builder.assert_bool(local.is_return);
        builder.assert_bool(local.is_main_return);

        //  Define aggregate flags
        let is_real =
            local.is_call + local.is_call_indirect + local.is_call_internal + local.is_return;
        let next_is_real =
            next.is_call + next.is_call_indirect + next.is_call_internal + next.is_return;
        let is_call_ins = local.is_call + local.is_call_indirect + local.is_call_internal;

        builder.when(local.is_main_return).assert_one(local.is_return);

        // // Ensure is_real is boolean (effectively checks mutual exclusivity)
        builder.assert_bool(is_real.clone());

        // Construct opcode value based on active flag
        let opcode = local.is_call * AB::Expr::from_canonical_u32(Opcode::Call(0).code()) +
            local.is_call_internal * AB::Expr::from_canonical_u32(Opcode::CallInternal(0).code()) +
            local.is_call_indirect * AB::Expr::from_canonical_u32(Opcode::CallIndirect(0).code()) +
            local.is_return * AB::Expr::from_canonical_u32(Opcode::Return.code());

        // // --- 2. Trace Integrity Constraints ---

        // // Prevent "resurrection": if current row is padding (0), next row MUST be padding (0).
        // builder.when_transition().when_not(is_real.clone()).assert_zero(next_is_real.clone());

        // // --- 3. Instruction Bus Interaction ---

        // Case A: Direct Call, Internal Call, Return
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.sp,
            local.sp,
            AB::Expr::zero(),
            opcode.clone(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            Word::zero::<AB>(),
            local.aux_value,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_call + local.is_call_internal + local.is_return,
        );

        // Case B: Indirect Call (CallIndirect)
        // Requires SP increment by UNIT (often to pop/push args or frame setup)
        builder.receive_rwasm_instruction(
            local.shard,
            local.clk,
            local.pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.sp,
            local.sp + AB::Expr::from_canonical_u32(UNIT),
            AB::Expr::zero(),
            opcode,
            Word::zero::<AB>(),
            local.func_index.word::<AB>(),
            Word::zero::<AB>(),
            local.aux_value,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_call_indirect,
        );

        // Sanitize aux_value on return
        builder.when(local.is_return).assert_word_zero(local.aux_value);

        builder
            .when(local.is_call_internal)
            .assert_word_eq(local.aux_value, local.func_index.word::<AB>());

        // --- 4. Sub-program Calls (Lookups) ---

        // For CallIndirect, we must verify the table lookup to get the target function
        builder.send_program(
            local.pc.reduce::<AB>() + AB::Expr::one(), // Next instruction should be TableGet
            AB::Expr::from_canonical_u32(Opcode::TableGet(0u16).code()),
            local.table_idx.word::<AB>(),
            local.is_call_indirect,
        );

        // --- 5. Range Checks ---

        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.pc,
            local.pc_range_checker,
            is_call_ins.clone(),
        );
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            is_call_ins.clone(),
        );
        CallStackAddressCols::<AB::Var>::do_range_check(
            builder,
            local.call_stack_address,
            is_real.clone() - local.is_main_return,
        );
        TableIdxCols::<AB::Var>::do_range_check(builder, local.table_idx, local.is_call_indirect);
        FuncIndex::<AB::Var>::do_range_check(builder, local.func_index, local.is_call_indirect);
        // --- 6. Call Stack Logic (The Core) ---

        // Initial state: Call stack must start at 0
        // builder
        //     .when(is_real.clone())
        //     .when_first_row()
        //     .assert_zero(local.call_stack_address.value::<AB>());

        // Push: When calling, increment stack pointer by UNIT
        builder
            .when(is_call_ins.clone()) // Any call type
            .when(next_is_real.clone())
            .assert_eq(
                local.call_stack_address.value::<AB>() + AB::Expr::from_canonical_u32(UNIT),
                next.call_stack_address.value::<AB>(),
            );

        // Pop: When returning, decrement stack pointer by UNIT
        builder.when(local.is_return).when(next_is_real.clone()).assert_eq(
            local.call_stack_address.value::<AB>() - AB::Expr::from_canonical_u32(UNIT),
            next.call_stack_address.value::<AB>(),
        );

        // Termination state:
        // If this is the last real row, stack must be empty (0)
        // builder
        //     .when(is_real.clone())
        //     .when_not(next_is_real.clone())
        //     .assert_zero(local.call_stack_address.value::<AB>());

        // The last instruction MUST be a Return (to exit main)
        builder.when(is_real).when_not(next_is_real).assert_one(local.is_return);

        // --- 7. Memory Access (Stack & Tables) ---

        // WRITE Return Address (on Call)
        // We write to the *current* stack address. Timestamp is clk+1 so Return can read it later.
        builder.eval_memory_access(
            local.shard,
            local.clk + AB::Expr::from_canonical_u8(1),
            AB::Expr::from_canonical_u32(TypedAddress::FuncFrame(0).to_virtual_addr()) +
                local.call_stack_address.value::<AB>(),
            &local.call_stack_access,
            is_call_ins,
        );

        // READ Return Address (on Return)
        // We read from *current* stack address.
        // CRITICAL: Do NOT read if it's a "fake" return (program end), as stack is empty.
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.call_stack_address.value::<AB>(),
            &local.call_stack_access,
            local.is_return - local.is_main_return,
        );

        // // READ Table (on Indirect Call)
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.table_idx.value::<AB>() * AB::Expr::from_canonical_u32(N_ONE_TABLE_MEMORY_LENGTH)
                + local.func_index.value::<AB>() * AB::Expr::from_canonical_u32(UNIT)
                + AB::Expr::from_canonical_u32(TABLE_MEMORY_SHIFT),
            &local.table_access,
            local.is_call_indirect,
        );

        // --- 8. Specific Logic for Helpers ---

        builder.when(local.is_main_return).assert_one(local.is_return);

        // Delegate PC and Stack value logic to helper methods
        self.eval_call_sp(builder, local);
        self.eval_next_pc(builder, local);
    }
}

impl CallChip {
    /// Evaluates what should be stored in the Call Stack (the Return Address)
    fn eval_call_sp<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CallColumns<AB::Var>) {
        // Internal Call: RetAddr = PC + 1
        builder.when(local.is_call_internal).assert_eq(
            AB::Expr::one() + local.pc.reduce::<AB>(),
            (*local.call_stack_access.value()).reduce::<AB>(),
        );

        // Indirect Call: RetAddr = PC + 2 (skips TableGet)
        builder.when(local.is_call_indirect).assert_eq(
            AB::Expr::from_canonical_u32(2u32) + local.pc.reduce::<AB>(),
            (*local.call_stack_access.value()).reduce::<AB>(),
        );

        // Return: Consistency check.
        // The value read from stack must match our next_pc.
        builder
            .when(local.is_return - local.is_main_return)
            .assert_word_eq(local.next_pc, *local.call_stack_access.value());
    }

    /// Evaluates the next Program Counter (Jump target)
    fn eval_next_pc<AB: SP1AirBuilder>(&self, builder: &mut AB, local: &CallColumns<AB::Var>) {
        // Internal Call: Jump to opcode_aux_val (immediate value)
        builder.when(local.is_call_internal).assert_word_eq(local.next_pc, local.aux_value);

        // Indirect Call: Jump to address fetched from Table
        builder
            .when(local.is_call_indirect)
            .assert_word_eq(local.next_pc, *local.table_access.value());

        // Return: Jump to address read from Stack.
        // DISABLED for fake return (program end) to avoid reading garbage from unconstrained
        // columns.
        builder.when(local.is_return - local.is_main_return).assert_eq(
            local.next_pc.reduce::<AB>(),
            local.call_stack_access.value().reduce::<AB>(),
        );
    }
}
