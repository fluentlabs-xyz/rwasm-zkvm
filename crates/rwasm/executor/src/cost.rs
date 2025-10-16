use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_baby_bear::BabyBear;

use crate::RwasmAirId;

const BYTE_NUM_ROWS: u64 = 1 << 16;
const MAX_PROGRAM_SIZE: u64 = 1 << 22;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_riscv_lde_size(
    num_events_per_air: EnumMap<RwasmAirId, u64>,
    costs_per_air: &HashMap<RwasmAirId, u64>,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&RwasmAirId::Byte];

    // Compute the program chip contribution.
    cells += MAX_PROGRAM_SIZE * costs_per_air[&RwasmAirId::Program];

    // Compute the cpu chip contribution.
    cells +=
        (num_events_per_air[RwasmAirId::Cpu]).next_power_of_two() * costs_per_air[&RwasmAirId::Cpu];

    // Compute the addsub chip contribution.
    cells += (num_events_per_air[RwasmAirId::AddSub]).next_power_of_two() *
        costs_per_air[&RwasmAirId::AddSub];

    // Compute the mul chip contribution.
    cells +=
        (num_events_per_air[RwasmAirId::Mul]).next_power_of_two() * costs_per_air[&RwasmAirId::Mul];

    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[RwasmAirId::Bitwise]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Bitwise];

    // Compute the shift left chip contribution.
    cells += (num_events_per_air[RwasmAirId::ShiftLeft]).next_power_of_two() *
        costs_per_air[&RwasmAirId::ShiftLeft];

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[RwasmAirId::ShiftRight]).next_power_of_two() *
        costs_per_air[&RwasmAirId::ShiftRight];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[RwasmAirId::DivRem]).next_power_of_two() *
        costs_per_air[&RwasmAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[RwasmAirId::Lt]).next_power_of_two() * costs_per_air[&RwasmAirId::Lt];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[RwasmAirId::MemoryLocal]).next_power_of_two() *
        costs_per_air[&RwasmAirId::MemoryLocal];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[RwasmAirId::Branch]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Branch];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[RwasmAirId::Call]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Call];

    // Compute the jump chip contribution.
    cells += (num_events_per_air[RwasmAirId::Jump]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Jump];

    // Compute the auipc chip contribution.
    cells += (num_events_per_air[RwasmAirId::Auipc]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Auipc];

    // Compute the memory opcode chip contribution.
    cells += (num_events_per_air[RwasmAirId::MemoryInstrs]).next_power_of_two() *
        costs_per_air[&RwasmAirId::MemoryInstrs];

    // Compute the syscall opcode chip contribution.
    cells += (num_events_per_air[RwasmAirId::SyscallInstrs]).next_power_of_two() *
        costs_per_air[&RwasmAirId::SyscallInstrs];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[RwasmAirId::SyscallCore]).next_power_of_two() *
        costs_per_air[&RwasmAirId::SyscallCore];

    // Compute the global chip contribution.
    cells += (num_events_per_air[RwasmAirId::Global]).next_power_of_two() *
        costs_per_air[&RwasmAirId::Global];

    cells * ((core::mem::size_of::<BabyBear>() << 1) as u64)
}

/// Pads the event counts to account for the worst case jump in events across N cycles.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_rv32im_event_counts(
    mut event_counts: EnumMap<RwasmAirId, u64>,
    num_cycles: u64,
) -> EnumMap<RwasmAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        RwasmAirId::Cpu => *v += num_cycles,
        RwasmAirId::AddSub => *v += 5 * num_cycles,
        RwasmAirId::Mul => *v += 4 * num_cycles,
        RwasmAirId::Bitwise => *v += 3 * num_cycles,
        RwasmAirId::ShiftLeft => *v += num_cycles,
        RwasmAirId::ShiftRight => *v += num_cycles,
        RwasmAirId::DivRem => *v += 4 * num_cycles,
        RwasmAirId::Lt => *v += 2 * num_cycles,
        RwasmAirId::MemoryLocal => *v += 64 * num_cycles,
        RwasmAirId::Branch => *v += 8 * num_cycles,
        RwasmAirId::Call => *v += 8 * num_cycles, //TODO: find exact number of worest case events
        RwasmAirId::Auipc => *v += 3 * num_cycles,
        RwasmAirId::MemoryInstrs => *v += 8 * num_cycles,
        RwasmAirId::SyscallInstrs => *v += num_cycles,
        RwasmAirId::SyscallCore => *v += 2 * num_cycles,
        RwasmAirId::Global => *v += 64 * num_cycles,
        _ => (),
    });
    event_counts
}
