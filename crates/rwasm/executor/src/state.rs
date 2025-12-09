use std::{
    collections::VecDeque,
    fs::File,
    io::{Seek, Write},
};

use fluentbase_runtime::RuntimeContext;
use hashbrown::HashMap;
use rwasm::RwasmStore;
use serde::{Deserialize, Serialize};
use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkVerifyingKey};

use crate::{
    events::MemoryRecord,
    memory::Memory,
    record::{ExecutionRecord, MemoryAccessRecord},
    syscalls::SyscallCode,
    ExecutorMode, SP1ReduceProof,
};

pub use rwasm::mem_index::SP_START;
// The starting address of function frame
pub const FUNFRAMEP_START: u32 = SP_START + 4096;

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct ExecutionState {
    /// The program counter.
    pub pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// The memory which opcodes operate over. Values contain the memory value and last shard
    /// + timestamp that each memory address was accessed.
    pub memory: Memory<MemoryRecord>,

    /// The global clock keeps track of how many opcodes have been executed through all
    /// shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each opcode that has been
    /// executed in this shard.
    pub clk: u32,
    /// stack pointer
    pub sp: u32,
    /// depth of function frame
    pub depth: u32,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: Memory<u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: VecDeque<Vec<u8>>,

    /// A stream of proofs (reduce vk, proof, verifying key) inputted to the program.
    pub proof_stream:
        Vec<(SP1ReduceProof<BabyBearPoseidon2>, StarkVerifyingKey<BabyBearPoseidon2>)>,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,

    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32) -> Self {
        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            sp: SP_START,
            depth: 0,
            pc: pc_start,
            memory: Memory::new_preallocated(),
            uninitialized_memory: Memory::new_preallocated(),
            input_stream: VecDeque::new(),
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream: Vec::new(),
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
        }
    }

    ///update memory state from rwasm Tracer
    pub fn update_state(&mut self, store: &RwasmStore<RuntimeContext>) {
        let tracer = &store.tracer;
        self.clk = tracer.state.clk;
        self.current_shard = tracer.state.shard;
        self.sp = tracer.state.sp;
        println!("update state");
        println!("tracer:{:?}", tracer);
        for item in tracer.memory_records.iter() {
            println!("addr{},record:{:?},", *item.0, *item.1);
            self.memory.insert(*item.0, *item.1);
        }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// All memory changes since the fork point.
    pub memory_diff: HashMap<u32, Option<MemoryRecord>>,
    /// The original memory access record at the fork point.
    pub op_record: MemoryAccessRecord,
    /// The original execution record at the fork point.
    pub record: ExecutionRecord,
    /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
    /// The cumulative number of unconstrained cycles during the entire execution.
    pub total_unconstrained_cycles: u64,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).unwrap();
        writer.flush()?;
        writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
