mod air;
mod columns;
mod trace;

pub use columns::*;
use p3_air::BaseAir;

#[derive(Default)]
pub struct BranchChip;

impl<F> BaseAir<F> for BranchChip {
    fn width(&self) -> usize {
        NUM_BRANCH_COLS
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};

    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use rwasm::Opcode;
    use rwasm_executor::{events::BranchEvent, ExecutionRecord, DEFAULT_PC_INC};

    use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

    use p3_field::AbstractField;
    use sp1_stark::air::MachineAir;
    use std::borrow::BorrowMut;

    fn be(pc: u32, next_pc: u32, opcode: Opcode, res: u32, arg1: u32, arg2: u32) -> BranchEvent {
        BranchEvent { pc, next_pc, opcode, res, arg1, arg2 }
    }

    #[test]
    fn generate_trace_branch() {
        let mut shard = ExecutionRecord::default();

        // Keep this reasonable for CI; bump locally if you want a stress run.
        let n = 200_000usize;

        let mut events = Vec::with_capacity(n);
        for i in 0..n as u32 {
            let pc = 10_000 + i * DEFAULT_PC_INC;

            match i % 4 {
                0 => {
                    // Br: always branches, next_pc = pc + res
                    let off = 16u32;
                    events.push(be(pc, pc + off, Opcode::Br((off as i32).into()), off, 0, 0));
                }
                1 => {
                    // BrIfEqz taken (arg1 == 0), next_pc = pc + res, arg2 must be 0
                    let off = 12u32;
                    events.push(be(pc, pc + off, Opcode::BrIfEqz((off as i32).into()), off, 0, 0));
                }
                2 => {
                    // BrIfEqz not taken (arg1 != 0), fall-through next_pc = pc + DEFAULT_PC_INC
                    let off = 20u32; // res is irrelevant when not branching,(still present)
                    events.push(be(
                        pc,
                        pc + DEFAULT_PC_INC,
                        Opcode::BrIfEqz((off as i32).into()),
                        off,
                        7,
                        0,
                    ));
                }
                _ => {
                    // BrIfNez taken (arg1 != 0), next_pc = pc + res, arg2 must be 0
                    let off = 24u32;
                    events.push(be(pc, pc + off, Opcode::BrIfNez((off as i32).into()), off, 9, 0));
                }
            }
        }

        shard.branch_events = events;

        let chip = BranchChip::default();
        let _trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
    }

    #[test]
    fn prove_babybear_branch() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mut shard = ExecutionRecord::default();

        // Hand-picked cases for each opcode + both taken/not-taken for conditional branches.
        let mut events = vec![
            // Br: always taken. next_pc = pc + res
            be(100, 120, Opcode::Br(20i32.into()), 20, 0, 0),
            // BrIfEqz: taken when arg1 == 0, arg2 must be 0
            be(200, 216, Opcode::BrIfEqz(16i32.into()), 16, 0, 0),
            // BrIfEqz: not taken when arg1 != 0 => next_pc = pc + DEFAULT_PC_INC
            be(300, 300 + DEFAULT_PC_INC, Opcode::BrIfEqz(16i32.into()), 16, 5, 0),
            // BrIfNez: taken when arg1 != 0
            be(400, 420, Opcode::BrIfNez(20i32.into()), 20, 7, 0),
            // BrIfNez: not taken when arg1 == 0 => next_pc = pc + DEFAULT_PC_INC
            be(500, 500 + DEFAULT_PC_INC, Opcode::BrIfNez(20i32.into()), 20, 0, 0),
            // BrTable: aux_value = 5 => target = aux-1 = 4.
            // In-range: index=2 < target => offset = 2*index + 1 = 5, so next_pc = pc + 5.
            // (res is unused by BrTable logic in your chip; can be 0.)
            // Choose arg2=aux_value=5 so default-formula (2*arg2-1) is consistent when
            // out-of-range.
            be(600, 605, Opcode::BrTable(5u32), 0, 2, 5),
            // Out-of-range: index=10 >= target => default offset = 2*arg2 - 1 = 9 => next_pc = pc
            // + 9
            be(700, 709, Opcode::BrTable(5u32), 0, 10, 5),
        ];

        // Pad to avoid a super tiny trace (optional).
        while events.len() < 1024 {
            let pc = 1_000 + (events.len() as u32) * DEFAULT_PC_INC;
            events.push(be(pc, pc + 8, Opcode::BrIfEqz(8i32.into()), 8, 0, 0));
        }

        shard.branch_events = events;

        let chip = BranchChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn test_malicious_branch_trace() {
        // Goal: show that if the AIR does NOT tie `a_eq_zero` to `arg1`,
        // a prover can “lie” about the branch condition.
        //
        // This test mutates the generated trace directly:
        // - Keep arg1 != 0 in the row (so condition should be "NEZ taken")
        // - Force not-branching + a_eq_zero = 1 + next_pc = pc + DEFAULT_PC_INC
        // A sound AIR must reject this.

        let config = BabyBearPoseidon2::new();

        let mut shard = ExecutionRecord::default();

        // Honest event: BrIfNez with arg1 != 0 SHOULD branch to pc + res.
        shard.branch_events = vec![be(
            1000,
            1020, // pc + 20
            Opcode::BrIfNez(20i32.into()),
            20,
            7, // nonzero
            0, // required by your constraints for brifnez/brifeqz
        )];

        let chip = BranchChip::default();
        let mut trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        // ---- mutate row 0 ----
        // If your RowMajorMatrix doesn't expose `values`, change this to the correct accessor.
        let values: &mut [BabyBear] = trace.values.as_mut_slice();
        let row0 = &mut values[0..NUM_BRANCH_COLS];
        let cols: &mut BranchColumns<BabyBear> = row0.borrow_mut();

        // Force "not branching" path (fall-through), and lie about condition.
        cols.is_branching = BabyBear::zero();
        cols.is_branching_table = BabyBear::zero();
        cols.is_branching_non_table = BabyBear::zero();

        // Force next_pc = pc + DEFAULT_PC_INC.
        cols.next_pc = (1000 + DEFAULT_PC_INC).into();

        // Lie: claim arg1 == 0 (even though arg1 is still 7).
        cols.a_eq_zero = BabyBear::one();

        // Prove+verify should FAIL if the AIR is sound.
        let mut challenger = config.challenger();
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        assert!(verify(&config, &chip, &mut challenger, &proof).is_err());
    }
}
