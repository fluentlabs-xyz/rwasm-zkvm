use hashbrown::HashMap;
use itertools::Itertools;

use rwasm_executor::{events::PrecompileLocalMemory, ExecutionRecord, RwasmAirId};
use sp1_stark::MachineRecord;

use crate::memory::NUM_LOCAL_MEMORY_ENTRIES_PER_ROW;

#[derive(Debug, Clone, Copy)]
pub enum ShardKind {
    PackedCore,
    Core,
    GlobalMemory,
    Precompile,
}

pub trait Shapeable {
    fn kind(&self) -> ShardKind;
    fn shard(&self) -> u32;
    fn log2_shard_size(&self) -> usize;
    fn debug_stats(&self) -> HashMap<String, usize>;
    fn core_heights(&self) -> Vec<(RwasmAirId, usize)>;
    fn memory_heights(&self) -> Vec<(RwasmAirId, usize)>;
    /// TODO. Returns all precompile events, assuming there is only one kind in `Self`.
    /// The tuple is of the form `(height, (num_memory_local_events, num_global_events))`
    fn precompile_heights(&self) -> impl Iterator<Item = (RwasmAirId, (usize, usize, usize))>;
}

impl Shapeable for ExecutionRecord {
    fn kind(&self) -> ShardKind {
        let contains_global_memory = !self.global_memory_initialize_events.is_empty() ||
            !self.global_memory_finalize_events.is_empty();
        match (self.contains_cpu(), contains_global_memory) {
            (true, true) => ShardKind::PackedCore,
            (true, false) => ShardKind::Core,
            (false, true) => ShardKind::GlobalMemory,
            (false, false) => ShardKind::Precompile,
        }
    }
    fn shard(&self) -> u32 {
        self.public_values.shard
    }

    fn log2_shard_size(&self) -> usize {
        self.cpu_events.len().next_power_of_two().ilog2() as usize
    }

    fn debug_stats(&self) -> HashMap<String, usize> {
        self.stats()
    }

    fn core_heights(&self) -> Vec<(RwasmAirId, usize)> {
        vec![
            (RwasmAirId::Cpu, self.cpu_events.len()),
            (RwasmAirId::DivRem, self.divrem_events.len()),
            (RwasmAirId::AddSub, self.add_events.len() + self.sub_events.len()),
            (RwasmAirId::Bitwise, self.bitwise_events.len()),
            (RwasmAirId::Rotate, self.rotate_events.len()),
            (RwasmAirId::Trailing, self.trailing_events.len()),
            (RwasmAirId::Extend, self.extend_events.len()),
            (RwasmAirId::Mul, self.mul_events.len()),
            (RwasmAirId::AddMul64, self.i64_events.len() + self.i64_events.len()),
            (RwasmAirId::ShiftRight, self.shift_right_events.len()),
            (RwasmAirId::ShiftLeft, self.shift_left_events.len()),
            (RwasmAirId::Lt, self.lt_events.len()),
            (
                RwasmAirId::MemoryLocal,
                self.get_local_mem_events()
                    .chunks(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW)
                    .into_iter()
                    .count(),
            ),
            (RwasmAirId::MemoryInstrs, self.memory_instr_events.len()),
            (RwasmAirId::Branch, self.branch_events.len()),
            (RwasmAirId::Fuel, self.fuel_events.len()),
            (RwasmAirId::Call, self.call_events.len()),
            (RwasmAirId::Global, self.global_interaction_events.len()),
            (RwasmAirId::SyscallCore, self.syscall_events.len()),
            (RwasmAirId::SyscallInstrs, self.syscall_events.len()),
            (RwasmAirId::Local, self.local_events.len()),
            (RwasmAirId::Const, self.const_events.len()),
            (RwasmAirId::ParamsCheck, self.params_check_events.len()),
            //TODO: is it true for TableGrow/TableInit ?
            (RwasmAirId::TableGrow, self.table_grow_events.len()),
            (RwasmAirId::TableInit, self.table_init_events.len()),
        ]
    }

    fn memory_heights(&self) -> Vec<(RwasmAirId, usize)> {
        vec![
            (RwasmAirId::MemoryGlobalInit, self.global_memory_initialize_events.len()),
            (RwasmAirId::MemoryGlobalFinalize, self.global_memory_finalize_events.len()),
            (
                RwasmAirId::Global,
                self.global_memory_finalize_events.len() +
                    self.global_memory_initialize_events.len(),
            ),
        ]
    }

    fn precompile_heights(&self) -> impl Iterator<Item = (RwasmAirId, (usize, usize, usize))> {
        self.precompile_events.events.iter().filter_map(|(code, events)| {
            // Skip empty events.
            (!events.is_empty()).then_some(())?;
            let id = code.as_air_id()?;
            Some((
                id,
                (
                    events.len() * id.rows_per_event(),
                    events.get_local_mem_events().into_iter().count(),
                    self.global_interaction_events.len(),
                ),
            ))
        })
    }
}
