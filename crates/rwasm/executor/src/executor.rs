#[cfg(feature = "profiling")]
use crate::profiler::Profiler;
use crate::{
    dependencies::{emit_branch_dependencies, emit_memory_dependencies}, //emit_divrem_dependencies
    estimator::RecordEstimator,
    events::{CallEvent, ConstEvent, I64AluEvent, PrecompileEvent, SyscallEvent},
};
#[cfg(feature = "profiling")]
use std::{fs::File, io::BufWriter};
use std::{str::FromStr, sync::Arc};

use clap::ValueEnum;
use enum_map::EnumMap;
use hashbrown::HashMap;

use rwasm::{
    event::FatOpEvent,
    mem::{MemoryLocalEvent, MemoryRecordEnum},
    CallStack, DataOpEvent, InstructionPtr, Opcode, RwasmExecutor, RwasmStore, TraceCallData,
    TrapCode, ValueStack, ValueStackPtr,
};
use serde::{Deserialize, Serialize};
use sp1_primitives::consts::BABYBEAR_PRIME;
use sp1_stark::{air::PublicValues, SP1CoreOpts};
use strum::IntoEnumIterator;
use thiserror::Error;

use crate::{
    context::{IoOptions, SP1Context},
    // dependencies::{
    //   emit_branch_dependencies, emit_divrem_dependencies,
    //      emit_memory_dependencies,
    // },TODO: redo dependencies
    estimate_riscv_lde_size,
    events::{
        AluEvent, BranchEvent, CpuEvent, MemInstrEvent, MemoryInitializeFinalizeEvent,
        MemoryReadRecord, MemoryRecord, MemoryWriteRecord,
    },
    hook::{HookEnv, HookRegistry},
    memory::{Entry, Memory},
    pad_rv32im_event_counts,
    record::{ExecutionRecord, MemoryAccessRecord},
    report::ExecutionReport,
    state::{ExecutionState, ForkState},
    subproof::SubproofVerifier,
    syscalls::{default_syscall_map, Syscall, SyscallCode, SyscallContext},
    CoreAirId,
    MaximalShapes,
    Program,
    RwasmAirId,
};

/// The default increment for the program counter.  Is used for all opcodes except
/// for branches and jumps.
pub const DEFAULT_PC_INC: u32 = 1;
///The default increment for the clk. we increase clk for two because we have
/// a reading phase and a writing phase.
pub const DEFAULT_CLK_INC: u32 = 2 * DEFAULT_PC_INC;
/// This is used in the `InstrEvent` to indicate that the opcode is not from the CPU.
/// A valid pc should be divisible by 4, so we use 1 to indicate that the pc is not used.
pub const UNUSED_PC: u32 = 1 << 24;

/// The maximum number of opcodes in a program.
pub const MAX_PROGRAM_SIZE: usize = 1 << 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Whether to verify deferred proofs during execution.
pub enum DeferredProofVerification {
    /// Verify deferred proofs during execution.
    Enabled,
    /// Skip verification of deferred proofs
    Disabled,
}

impl From<bool> for DeferredProofVerification {
    fn from(value: bool) -> Self {
        if value {
            DeferredProofVerification::Enabled
        } else {
            DeferredProofVerification::Disabled
        }
    }
}

pub struct RwasmExecutorState {
    pub sp: ValueStackPtr,
    pub ip: InstructionPtr,
}

/// An executor for the SP1 RISC-V zkVM.
///
/// The exeuctor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct Executor<'a> {
    /// The program.
    pub program: Arc<Program>,

    pub value_stack: ValueStack,
    pub call_stack: CallStack,
    pub register_state: Option<RwasmExecutorState>,
    pub store: RwasmStore<()>,

    /// The state of the execution.
    pub state: ExecutionState,

    /// Memory addresses that were touched in this batch of shards. Used to minimize the size of
    /// checkpoints.
    pub memory_checkpoint: Memory<Option<MemoryRecord>>,

    /// Memory addresses that were initialized in this batch of shards. Used to minimize the size
    /// of checkpoints. The value stored is whether or not it had a value at the beginning of
    /// the batch.
    pub uninitialized_memory_checkpoint: Memory<bool>,

    /// Report of the program execution.
    pub report: ExecutionReport,

    /// The mode the executor is running in.
    pub executor_mode: ExecutorMode,

    /// The memory accesses for the current cycle.
    pub memory_accesses: MemoryAccessRecord,

    /// Whether the runtime is in constrained mode or not.
    ///
    /// In unconstrained mode, any events, clock, register, or memory changes are reset after
    /// leaving the unconstrained block. The only thing preserved is writes to the input
    /// stream.
    pub unconstrained: bool,

    /// Whether we should write to the report.
    pub print_report: bool,

    /// Data used to estimate total trace area.
    pub record_estimator: Option<Box<RecordEstimator>>,

    /// Whether we should emit global memory init and finalize events. This can be enabled in
    /// Checkpoint mode and disabled in Trace mode.
    pub emit_global_memory_events: bool,

    /// The maximum size of each shard.
    pub shard_size: u32,

    /// The maximum number of shards to execute at once.
    pub shard_batch_size: u32,

    /// The maximum number of cycles for a syscall.
    pub max_syscall_cycles: u32,

    /// The mapping between syscall codes and their implementations.
    pub syscall_map: HashMap<SyscallCode, Arc<dyn Syscall>>,

    /// The options for the runtime.
    pub opts: SP1CoreOpts,

    /// The maximum number of cpu cycles to use for execution.
    pub max_cycles: Option<u64>,

    /// The current trace of the execution that is being collected.
    pub record: Box<ExecutionRecord>,

    /// The collected records, split by cpu cycles.
    pub records: Vec<Box<ExecutionRecord>>,

    /// Local memory access events.
    pub local_memory_access: HashMap<u32, MemoryLocalEvent>,

    /// A counter for the number of cycles that have been executed in certain functions.
    pub cycle_tracker: HashMap<String, (u64, u32)>,

    /// A buffer for stdout and stderr IO.
    pub io_buf: HashMap<u32, String>,

    /// The ZKVM program profiler.
    ///
    /// Keeps track of the number of cycles spent in each function.
    #[cfg(feature = "profiling")]
    pub profiler: Option<(Profiler, BufWriter<File>)>,

    /// The state of the runtime when in unconstrained mode.
    pub unconstrained_state: Box<ForkState>,

    /// Statistics for event counts.
    pub local_counts: LocalCounts,

    /// Verifier used to sanity check `verify_sp1_proof` during runtime.
    pub subproof_verifier: Option<&'a dyn SubproofVerifier>,

    /// Registry of hooks, to be invoked by writing to certain file descriptors.
    pub hook_registry: HookRegistry<'a>,

    /// The maximal shapes for the program.
    pub maximal_shapes: Option<MaximalShapes>,

    /// The costs of the program.
    pub costs: HashMap<RwasmAirId, u64>,

    /// Skip deferred proof verification. This check is informational only, not related to circuit
    /// correctness.
    pub deferred_proof_verification: DeferredProofVerification,

    /// The frequency to check the stopping condition.
    pub shape_check_frequency: u64,

    /// Early exit if the estimate LDE size is too big.
    pub lde_size_check: bool,

    /// The maximum LDE size to allow.
    pub lde_size_threshold: u64,

    /// The options for the IO.
    pub io_options: IoOptions<'a>,

    /// Temporary event counts for the current shard. This is a field to reuse memory.
    event_counts: EnumMap<RwasmAirId, u64>,
}

/// The different modes the executor can run in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
pub enum ExecutorMode {
    /// Run the execution with no tracing or checkpointing.
    #[default]
    Simple,
    /// Run the execution with checkpoints for memory.
    Checkpoint,
    /// Run the execution with full tracing of events.
    Trace,
    /// Run the execution with full tracing of events and size bounds for shape collection.
    ShapeCollection,
}

/// Information about event counts which are relevant for shape fixing.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LocalCounts {
    /// The event counts.
    pub event_counts: Box<EnumMap<u8, u64>>,
    /// The number of syscalls sent globally in the current shard.
    pub syscalls_sent: usize,
    /// The number of addresses touched in this shard.
    pub local_mem: usize,
}

/// Errors that the [``Executor``] can throw.
#[derive(Error, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionError {
    /// The execution failed with a non-zero exit code.
    #[error("execution failed with exit code {0}")]
    HaltWithNonZeroExitCode(u32),

    /// The execution failed with an invalid memory access.
    #[error("invalid memory access for opcode {0} and address {1}")]
    InvalidMemoryAccess(Opcode, u32),

    /// The execution failed with an unimplemented syscall.
    #[error("unimplemented syscall {0}")]
    UnsupportedSyscall(u32),

    /// The execution failed with a breakpoint.
    #[error("breakpoint encountered")]
    Breakpoint(),

    /// The execution failed with an exceeded cycle limit.
    #[error("exceeded cycle limit of {0}")]
    ExceededCycleLimit(u64),

    /// The execution failed because the syscall was called in unconstrained mode.
    #[error("syscall called in unconstrained mode")]
    InvalidSyscallUsage(u64),

    /// The execution failed with an unimplemented feature.
    #[error("got unimplemented as opcode")]
    Unimplemented(),

    /// The program ended in unconstrained mode.
    #[error("program ended in unconstrained mode")]
    EndInUnconstrained(),

    /// The program ended in unconstrained mode.
    #[error("runtime divided by zero")]
    DividedByZero(),

    /// The unconstrained cycle limit was exceeded.
    #[error("unconstrained cycle limit exceeded")]
    UnconstrainedCycleLimitExceeded(u64),
}

impl<'a> Executor<'a> {
    /// Create a new [``Executor``] from a program and options.
    #[must_use]
    pub fn new(program: Program, opts: SP1CoreOpts) -> Self {
        Self::with_context(program, opts, SP1Context::default())
    }

    /// WARNING: This function's API is subject to change without a major version bump.
    ///
    /// If the feature `"profiling"` is enabled, this sets up the profiler. Otherwise, it does
    /// nothing. The argument `elf_bytes` must describe the same program as `self.program`.
    ///
    /// The profiler is configured by the following environment variables:
    ///
    /// - `TRACE_FILE`: writes Gecko traces to this path. If unspecified, the profiler is disabled.
    /// - `TRACE_SAMPLE_RATE`: The period between clock cycles where samples are taken. Defaults to
    ///   1.
    #[inline]
    #[allow(unused_variables)]
    pub fn maybe_setup_profiler(&mut self, elf_bytes: &[u8]) {
        #[cfg(feature = "profiling")]
        {
            let trace_buf = std::env::var("TRACE_FILE").ok().map(|file| {
                let file = File::create(file).unwrap();
                BufWriter::new(file)
            });

            if let Some(trace_buf) = trace_buf {
                eprintln!("Profiling enabled");

                let sample_rate = std::env::var("TRACE_SAMPLE_RATE")
                    .ok()
                    .and_then(|rate| {
                        eprintln!("Profiling sample rate: {rate}");
                        rate.parse::<u32>().ok()
                    })
                    .unwrap_or(1);

                self.profiler = Some((
                    Profiler::new(elf_bytes, sample_rate as u64)
                        .expect("Failed to create profiler"),
                    trace_buf,
                ));
            }
        }
    }

    /// Create a new runtime from a program, options, and a context.
    #[must_use]
    pub fn with_context(program: Program, opts: SP1CoreOpts, context: SP1Context<'a>) -> Self {
        // Create a shared reference to the program.
        let program = Arc::new(program);

        // Create a default record with the program.
        let record = ExecutionRecord::new(program.clone());

        // Determine the maximum number of cycles for any syscall.
        let syscall_map = default_syscall_map();
        let max_syscall_cycles =
            syscall_map.values().map(|syscall| syscall.num_extra_cycles()).max().unwrap_or(0);

        let hook_registry = context.hook_registry.unwrap_or_default();

        let costs: HashMap<String, usize> =
            serde_json::from_str(include_str!("./artifacts/rv32im_costs.json")).unwrap();
        let costs: HashMap<RwasmAirId, usize> =
            costs.into_iter().map(|(k, v)| (RwasmAirId::from_str(&k).unwrap(), v)).collect();

        let store = RwasmStore::default();

        Self {
            record: Box::new(record),
            records: vec![],
            state: ExecutionState::new(0u32),
            program,
            value_stack: Default::default(),
            call_stack: Default::default(),
            register_state: None,
            memory_accesses: MemoryAccessRecord::default(),
            shard_size: (opts.shard_size as u32) * 4,
            shard_batch_size: opts.shard_batch_size as u32,
            cycle_tracker: HashMap::new(),
            io_buf: HashMap::new(),
            #[cfg(feature = "profiling")]
            profiler: None,
            unconstrained: false,
            unconstrained_state: Box::new(ForkState::default()),
            syscall_map,
            executor_mode: ExecutorMode::Trace,
            emit_global_memory_events: true,
            max_syscall_cycles,
            report: ExecutionReport::default(),
            local_counts: LocalCounts::default(),
            print_report: false,
            record_estimator: None,
            subproof_verifier: context.subproof_verifier,
            hook_registry,
            opts,
            max_cycles: context.max_cycles,
            deferred_proof_verification: context.deferred_proof_verification.into(),
            memory_checkpoint: Memory::default(),
            uninitialized_memory_checkpoint: Memory::default(),
            local_memory_access: HashMap::new(),
            maximal_shapes: None,
            costs: costs.into_iter().map(|(k, v)| (k, v as u64)).collect(),
            shape_check_frequency: 16,
            lde_size_check: false,
            lde_size_threshold: 0,
            event_counts: EnumMap::default(),
            io_options: context.io_options,
            store,
        }
    }

    /// Invokes a hook with the given file descriptor `fd` with the data `buf`.
    ///
    /// # Errors
    ///
    /// If the file descriptor is not found in the [``HookRegistry``], this function will return an
    /// error.
    pub fn hook(&self, fd: u32, buf: &[u8]) -> eyre::Result<Vec<Vec<u8>>> {
        Ok(self
            .hook_registry
            .get(fd)
            .ok_or(eyre::eyre!("no hook found for file descriptor {}", fd))?
            .invoke_hook(self.hook_env(), buf))
    }

    /// Prepare a `HookEnv` for use by hooks.
    #[must_use]
    pub fn hook_env<'b>(&'b self) -> HookEnv<'b, 'a> {
        HookEnv { runtime: self }
    }

    /// Recover runtime state from a program and existing execution state.
    #[must_use]
    pub fn recover(program: Program, state: ExecutionState, opts: SP1CoreOpts) -> Self {
        let mut runtime = Self::new(program, opts);
        runtime.state = state;
        // Disable deferred proof verification since we're recovering from a checkpoint, and the
        // checkpoint creator already had a chance to check the proofs.
        runtime.deferred_proof_verification = DeferredProofVerification::Disabled;
        runtime
    }

    /// Get the current value of a word.
    ///
    /// Assumes `addr` is a valid memory address, not a register.
    #[must_use]
    pub fn word(&mut self, addr: u32) -> u32 {
        #[allow(clippy::single_match_else)]
        let record = self.state.memory.page_table.get(addr);

        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match record {
                Some(record) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                None => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }

        match record {
            Some(record) => record.value,
            None => 0,
        }
    }

    /// Get the current value of a byte.
    ///
    /// Assumes `addr` is a valid memory address, not a register.
    #[must_use]
    pub fn byte(&mut self, addr: u32) -> u8 {
        let word = self.word(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    /// Get the current timestamp for a given memory access position.
    #[must_use]
    pub const fn timestamp(&self) -> u32 {
        self.state.clk
    }

    /// Get the current shard.
    #[must_use]
    #[inline]
    pub fn shard(&self) -> u32 {
        self.state.current_shard
    }

    /// Read a word from memory and create an access record.
    pub fn mr(
        &mut self,
        addr: u32,
        shard: u32,
        timestamp: u32,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        // Check that the memory address is within the babybear field and not within the registers'
        // address space.  Also check that the address is aligned.
        if !addr.is_multiple_of(4) || addr >= BABYBEAR_PRIME {
            panic!("Invalid memory access: addr={addr}");
        }

        // Get the memory record entry.
        let entry = self.state.memory.page_table.entry(addr);
        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if self.unconstrained {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }

        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryRecord = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.page_table.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .page_table
                    .entry(addr)
                    .or_insert_with(|| *value != 0);
                entry.insert(MemoryRecord { value: *value, shard: 0, timestamp: 0 })
            }
        };

        // We update the local memory counter in two cases:
        //  1. This is the first time the address is touched, this corresponds to the condition
        //     record.shard != shard.
        //  2. The address is being accessed in a syscall. In this case, we need to send it. We use
        //     local_memory_access to detect this. *WARNING*: This means that we are counting on the
        //     .is_some() condition to be true only in the SyscallContext.
        if !self.unconstrained && (record.shard != shard || local_memory_access.is_some()) {
            self.local_counts.local_mem += 1;
        }

        if !self.unconstrained {
            if let Some(estimator) = &mut self.record_estimator {
                if record.shard != shard {
                    estimator.current_local_mem += 1;
                }
                let current_touched_compressed_addresses = if local_memory_access.is_some() {
                    &mut estimator.current_precompile_touched_compressed_addresses
                } else {
                    &mut estimator.current_touched_compressed_addresses
                };
                current_touched_compressed_addresses.insert(addr >> 2);
            }
        }

        let prev_record = *record;
        record.shard = shard;
        record.timestamp = timestamp;

        if !self.unconstrained && self.executor_mode == ExecutorMode::Trace {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };

            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = *record;
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record,
                    final_mem_access: *record,
                });
        }

        // Construct the memory read record.
        MemoryReadRecord::new(
            record.value,
            record.shard,
            record.timestamp,
            prev_record.shard,
            prev_record.timestamp,
        )
    }

    /// Write a word to memory and create an access record.
    pub fn mw(
        &mut self,
        addr: u32,
        value: u32,
        shard: u32,
        timestamp: u32,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        // Check that the memory address is within the babybear field and not within the registers'
        // address space.  Also check that the address is aligned.
        if !addr.is_multiple_of(4) || addr >= BABYBEAR_PRIME {
            panic!("Invalid memory access: addr={addr}");
        }

        // Get the memory record entry.
        let entry = self.state.memory.page_table.entry(addr);
        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }
        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if self.unconstrained {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }
        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryRecord = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.page_table.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .page_table
                    .entry(addr)
                    .or_insert_with(|| *value != 0);

                entry.insert(MemoryRecord { value: *value, shard: 0, timestamp: 0 })
            }
        };

        // We update the local memory counter in two cases:
        //  1. This is the first time the address is touched, this corresponds to the condition
        //     record.shard != shard.
        //  2. The address is being accessed in a syscall. In this case, we need to send it. We use
        //     local_memory_access to detect this. *WARNING*: This means that we are counting on the
        //     .is_some() condition to be true only in the SyscallContext.
        if !self.unconstrained && (record.shard != shard || local_memory_access.is_some()) {
            self.local_counts.local_mem += 1;
        }

        if !self.unconstrained {
            if let Some(estimator) = &mut self.record_estimator {
                if record.shard != shard {
                    estimator.current_local_mem += 1;
                }
                let current_touched_compressed_addresses = if local_memory_access.is_some() {
                    &mut estimator.current_precompile_touched_compressed_addresses
                } else {
                    &mut estimator.current_touched_compressed_addresses
                };
                current_touched_compressed_addresses.insert(addr >> 2);
            }
        }

        let prev_record = *record;
        record.value = value;
        record.shard = shard;
        record.timestamp = timestamp;
        if !self.unconstrained && self.executor_mode == ExecutorMode::Trace {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };

            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = *record;
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record,
                    final_mem_access: *record,
                });
        }

        // Construct the memory write record.
        MemoryWriteRecord::new(
            record.value,
            record.shard,
            record.timestamp,
            prev_record.value,
            prev_record.shard,
            prev_record.timestamp,
        )
    }

    /// Emit events for this cycle.
    #[allow(clippy::too_many_arguments)]
    fn emit_events(
        &mut self,
        clk: u32,
        pc: u32,
        next_pc: u32,
        sp: u32,
        next_sp: u32,
        call_sp: u32,
        next_call_sp: u32,
        opcode: Opcode,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
        res: u32,
        res_hi: u32,
        record: MemoryAccessRecord,
        call_data: Option<TraceCallData>,
        fat_op: Option<FatOpEvent>,
        dataop_event: Option<DataOpEvent>,
    ) {
        if opcode.is_memory_instruction() {
            self.emit_cpu(
                clk,
                pc,
                next_pc,
                sp,
                next_sp,
                call_sp,
                next_call_sp,
                arg1,
                opcode.aux_value(),
                arg2,
                res_hi,
                record,
                0u32,
                call_data,
            );
        } else {
            self.emit_cpu(
                clk,
                pc,
                next_pc,
                sp,
                next_sp,
                call_sp,
                next_call_sp,
                arg1,
                arg2,
                res,
                res_hi,
                record,
                0u32,
                call_data,
            );
        }

        if opcode.is_alu_instruction() {
            self.emit_alu_event(pc, opcode, arg1, arg2, res);
        } else if opcode.is_memory_load_instruction() || opcode.is_memory_store_instruction() {
            self.emit_mem_instr_event(opcode, arg1, arg2, res, record);
        } else if opcode.is_branch_instruction() {
            self.emit_branch_event(opcode, arg1, arg2, res, next_pc);
        } else if opcode.is_ecall_instruction() {
            let syscall_code = match opcode {
                Opcode::TableInit(_) => SyscallCode::TABLE_INIT,
                Opcode::TableGrow(_) => SyscallCode::TABLE_GROW,
                _ => syscall_code,
            };
            self.emit_syscall_event(
                clk,
                record.arg1_record,
                syscall_code,
                arg1,
                arg2,
                next_pc,
                fat_op,
            );
        } else if opcode.is_const_instruction() {
            self.emit_const_event(opcode);
        } else if opcode.is_state_instrucition() {
        } else if opcode.is_call_instruction() {
            let call_sp_record = record.call_sp_access;
            match call_data {
                Some(call_data) => {
                    self.emit_call_event(
                        clk,
                        pc,
                        next_pc,
                        opcode,
                        call_sp,
                        next_call_sp,
                        call_data.signature_id,
                        call_data.func_ref,
                        call_data.table_id,
                        call_data.table_idx,
                        call_sp_record,
                        call_data.table_access.map(MemoryRecordEnum::Read),
                        dataop_event,
                    );
                }
                None => {
                    self.emit_call_event(
                        clk,
                        pc,
                        next_pc,
                        opcode,
                        call_sp,
                        next_call_sp,
                        0,
                        opcode.aux_value(),
                        0,
                        0,
                        call_sp_record,
                        None,
                        None,
                    );
                }
            }
        } else if opcode.is_64b_op() {
            self.emit_i64_event(clk, pc, next_pc, opcode, res, res_hi, arg1, arg2, record);
        } else {
            println!("no event :ins:{:?},", opcode);
        }
    }

    /// Emit a CPU event.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn emit_cpu(
        &mut self,
        clk: u32,
        pc: u32,
        next_pc: u32,
        sp: u32,
        next_sp: u32,
        call_sp: u32,
        next_call_sp: u32,
        arg1: u32,
        arg2: u32,
        res: u32,
        res_hi: u32,
        record: MemoryAccessRecord,
        exit_code: u32,
        call_data: Option<TraceCallData>,
    ) {
        self.record.cpu_events.push(CpuEvent {
            clk,
            pc,
            next_pc,
            sp,
            next_sp,
            call_sp,
            next_call_sp,
            res,
            res_record: record.res_record,
            res_addr: record.res_addr,
            res_hi,
            res_hi_record: record.res_hi_record,
            res_hi_addr: record.res_hi_addr,
            arg1,
            arg1_record: record.arg1_record,
            arg1_addr: record.arg1_addr,
            arg2,
            arg2_record: record.arg2_record,
            arg2_addr: record.arg2_addr,
            exit_code,
            call_data,
        });
    }

    /// Emit an ALU event.
    #[allow(clippy::too_many_lines)]
    fn emit_alu_event(&mut self, pc: u32, opcode: Opcode, arg1: u32, arg2: u32, res: u32) {
        let event = AluEvent { pc, opcode, a: res, b: arg1, c: arg2, code: opcode.code() };
        match opcode {
            Opcode::I32Add => {
                self.record.add_events.push(event);
            }
            Opcode::I32Sub => {
                self.record.sub_events.push(event);
            }
            Opcode::I32Xor | Opcode::I32Or | Opcode::I32And => {
                self.record.bitwise_events.push(event);
            }
            Opcode::I32Shl => {
                self.record.shift_left_events.push(event);
            }
            Opcode::I32ShrS | Opcode::I32ShrU => {
                self.record.shift_right_events.push(event);
            }
            Opcode::I32GeS |
            Opcode::I32GtS |
            Opcode::I32GeU |
            Opcode::I32GtU |
            Opcode::I32LeS |
            Opcode::I32LeU |
            Opcode::I32LtS |
            Opcode::I32LtU |
            Opcode::I32Eq |
            Opcode::I32Eqz |
            Opcode::I32Ne => {
                let use_signed_comparison = matches!(
                    opcode,
                    Opcode::I32GeS | Opcode::I32GtS | Opcode::I32LeS | Opcode::I32LtS
                );

                let (lt_res, gt_res, cmp_opcode) = {
                    if use_signed_comparison {
                        (
                            ((event.b as i32) < (event.c as i32)) as u32,
                            ((event.b as i32) > (event.c as i32)) as u32,
                            Opcode::I32LtS,
                        )
                    } else {
                        ((event.b < event.c) as u32, (event.b > event.c) as u32, Opcode::I32LtU)
                    }
                };

                let make_lt = |a, b, c| AluEvent {
                    pc: UNUSED_PC,
                    opcode: cmp_opcode,
                    a,
                    b,
                    c,
                    code: cmp_opcode.code(),
                };

                let lt_comp_event = make_lt(lt_res, event.b, event.c);
                let gt_comp_event = make_lt(gt_res, event.c, event.b);

                let ev_eqz = AluEvent {
                    pc: UNUSED_PC,
                    opcode: Opcode::I32LtU,
                    a: u32::from(event.b == 0),
                    b: event.b,
                    c: 1,
                    code: Opcode::I32LtU.code(),
                };

                match opcode {
                    // Opcodes that only need a "less than" check.
                    Opcode::I32LtS | Opcode::I32LtU => {
                        self.record.lt_events.push(lt_comp_event);
                    }
                    // b > c is equivalent to c < b
                    Opcode::I32GtS | Opcode::I32GtU => {
                        self.record.lt_events.push(gt_comp_event);
                    }
                    // b >= c is equivalent to !(b < c)
                    Opcode::I32GeS | Opcode::I32GeU => {
                        self.record.lt_events.push(lt_comp_event);
                    }
                    // b <= c is equivalent to !(c < b)
                    Opcode::I32LeS | Opcode::I32LeU => {
                        self.record.lt_events.push(gt_comp_event);
                    }
                    // EQZ(x): CPU AIR expects two LtU checks:
                    // 1. LtU(x, 0) -> always false (0)
                    // 2. LtU(0, x) -> true if x > 0 (i.e. x != 0)
                    Opcode::I32Eqz => {
                        self.record.lt_events.push(AluEvent {
                            pc: UNUSED_PC,
                            opcode: Opcode::I32LtU,
                            a: 0, // x < 0 is always false for unsigned
                            b: event.b,
                            c: 0,
                            code: Opcode::I32LtU.code(),
                        });
                        self.record.lt_events.push(AluEvent {
                            pc: UNUSED_PC,
                            opcode: Opcode::I32LtU,
                            a: u32::from(event.b != 0), // 0 < x is true if x != 0
                            b: 0,
                            c: event.b,
                            code: Opcode::I32LtU.code(),
                        });
                    }

                    // EQ(b,c) / NE(b,c): CPU AIR expects two LtU checks:
                    // 1. LtU(b, c)
                    // 2. LtU(c, b)
                    Opcode::I32Eq | Opcode::I32Ne => {
                        self.record.lt_events.push(lt_comp_event);
                        self.record.lt_events.push(gt_comp_event);
                    }
                    _ => unreachable!(),
                }
            }
            Opcode::I32Ctz | Opcode::I32Clz | Opcode::I32Popcnt => {
                self.record.trailing_events.push(event);
            }
            Opcode::I32Mul => {
                self.record.mul_events.push(event);
            }
            Opcode::I32DivS | Opcode::I32DivU | Opcode::I32RemS | Opcode::I32RemU => {
                self.record.divrem_events.push(event);
                //emit_divrem_dependencies(self, event);
            }
            Opcode::I32Rotl | Opcode::I32Rotr => {
                self.record.rotate_events.push(event);
            }
            Opcode::I32Extend8S | Opcode::I32Extend16S => {
                self.record.extend_events.push(event);
            }
            _ => unreachable!(),
        }
    }

    // Emit a memory opcode event.
    #[inline]
    fn emit_mem_instr_event(
        &mut self,
        opcode: Opcode,
        arg1: u32,
        arg2: u32,
        res: u32,
        record: MemoryAccessRecord,
    ) {
        println!("record in emit:{:?}", record.memory.expect("Must have memory access"));
        let event = MemInstrEvent {
            shard: self.shard(),
            clk: self.state.clk,
            pc: self.state.pc,
            opcode,
            raw_addr: arg1,
            offset: opcode.aux_value(),
            res,
            mem_access: record.memory.expect("Must have memory access"),
            mem_access_hi: record.memory_hi,
        };
        println!("mem event:{:?}", event);
        self.record.memory_instr_events.push(event);
        emit_memory_dependencies(self, event);
    }

    // Emit a branch event.
    #[inline]
    fn emit_branch_event(&mut self, opcode: Opcode, arg1: u32, arg2: u32, res: u32, next_pc: u32) {
        let event = BranchEvent { pc: self.state.pc, next_pc, opcode, res, arg1, arg2 };
        println!("br event:{:?}", event);
        self.record.branch_events.push(event);

        emit_branch_dependencies(self, event);
    }

    // /// Emit an AUIPC event.
    // #[inline]
    // fn emit_auipc_event(&mut self, opcode: &Opcode, a: u32, b: u32, c: u32, op_a_0: bool) {
    //     let event = AUIPCEvent::new(self.state.pc, opcode, a, b, c, op_a_0);
    //     self.record.auipc_events.push(event);
    //     emit_auipc_dependency(self, event);
    // }

    /// Create a syscall event.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn syscall_event(
        &self,
        clk: u32,
        a_record: Option<MemoryRecordEnum>,
        op_a_0: Option<bool>,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
        next_pc: u32,
    ) -> SyscallEvent {
        let (write, is_real) = match a_record {
            Some(MemoryRecordEnum::Write(record)) => (record, true),
            _ => (MemoryWriteRecord::default(), false),
        };

        // If op_a_0 is None, then we assume it is not register 0.  Note that this will happen
        // for syscall events that are created within the precompiles' execute function.  Those
        // events will be added to precompile tables, which wouldn't use the op_a_0 field.
        // Note that we can't make the op_a_0 field an Option<bool> in SyscallEvent because
        // of the cbindgen.
        let op_a_0 = op_a_0.unwrap_or(false);

        SyscallEvent {
            shard: self.shard(),
            clk,
            pc: self.state.pc,
            next_pc,
            a_record: write,
            a_record_is_real: is_real,
            op_a_0,
            syscall_code,
            syscall_id: syscall_code.syscall_id(),
            arg1,
            arg2,
        }
    }

    /// Emit a syscall event.
    #[allow(clippy::too_many_arguments)]
    fn emit_syscall_event(
        &mut self,
        clk: u32,
        a_record: Option<MemoryRecordEnum>,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
        next_pc: u32,
        fat_op: Option<FatOpEvent>,
    ) {
        let syscall_event =
            self.syscall_event(clk, a_record, Some(true), syscall_code, arg1, arg2, next_pc);

        self.record.syscall_events.push(syscall_event);
        println!("eventcode: {:?}", syscall_code);
        match syscall_code {
            SyscallCode::HALT => todo!(),
            SyscallCode::WRITE => todo!(),
            SyscallCode::ENTER_UNCONSTRAINED => todo!(),
            SyscallCode::EXIT_UNCONSTRAINED => todo!(),
            SyscallCode::SHA_EXTEND => todo!(),
            SyscallCode::SHA_COMPRESS => todo!(),
            SyscallCode::ED_ADD => todo!(),
            SyscallCode::ED_DECOMPRESS => todo!(),
            SyscallCode::KECCAK_PERMUTE => todo!(),
            SyscallCode::SECP256K1_ADD => todo!(),
            SyscallCode::SECP256K1_DOUBLE => todo!(),
            SyscallCode::SECP256K1_DECOMPRESS => todo!(),
            SyscallCode::BN254_ADD => todo!(),
            SyscallCode::BN254_DOUBLE => todo!(),
            SyscallCode::COMMIT => todo!(),
            SyscallCode::COMMIT_DEFERRED_PROOFS => todo!(),
            SyscallCode::VERIFY_SP1_PROOF => todo!(),
            SyscallCode::BLS12381_DECOMPRESS => todo!(),
            SyscallCode::HINT_LEN => todo!(),
            SyscallCode::HINT_READ => todo!(),
            SyscallCode::UINT256_MUL => todo!(),
            SyscallCode::U256XU2048_MUL => todo!(),
            SyscallCode::BLS12381_ADD => todo!(),
            SyscallCode::BLS12381_DOUBLE => todo!(),
            SyscallCode::BLS12381_FP_ADD => todo!(),
            SyscallCode::BLS12381_FP_SUB => todo!(),
            SyscallCode::BLS12381_FP_MUL => todo!(),
            SyscallCode::BLS12381_FP2_ADD => todo!(),
            SyscallCode::BLS12381_FP2_SUB => todo!(),
            SyscallCode::BLS12381_FP2_MUL => todo!(),
            SyscallCode::BN254_FP_ADD => todo!(),
            SyscallCode::BN254_FP_SUB => todo!(),
            SyscallCode::BN254_FP_MUL => todo!(),
            SyscallCode::BN254_FP2_ADD => todo!(),
            SyscallCode::BN254_FP2_SUB => todo!(),
            SyscallCode::BN254_FP2_MUL => todo!(),
            SyscallCode::SECP256R1_ADD => todo!(),
            SyscallCode::SECP256R1_DOUBLE => todo!(),
            SyscallCode::SECP256R1_DECOMPRESS => todo!(),
            SyscallCode::TABLE_INIT => match fat_op.unwrap() {
                FatOpEvent::TableInit(table_init_event) => self.record.precompile_events.add_event(
                    SyscallCode::TABLE_INIT,
                    syscall_event,
                    PrecompileEvent::TableInit(table_init_event),
                ),
                _ => {
                    unreachable!();
                }
            },
            SyscallCode::TABLE_GROW => match fat_op.unwrap() {
                FatOpEvent::TableGrow(table_grow_event) => self.record.precompile_events.add_event(
                    SyscallCode::TABLE_GROW,
                    syscall_event,
                    PrecompileEvent::TableGrow(table_grow_event),
                ),
                _ => {
                    unreachable!();
                }
            },
        }
    }

    // Emit a branch event.
    #[inline]
    fn emit_const_event(&mut self, opcode: Opcode) {
        let event = ConstEvent { pc: self.state.pc, opcode, value: opcode.aux_value() };
        self.record.const_events.push(event);
    }
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn emit_call_event(
        &mut self,

        clk: u32,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        call_sp: u32,
        next_call_sp: u32,
        signature_id: u32,
        func_ref: u32,
        table_id: u32,
        table_idx: u32,
        call_stack_access: Option<MemoryRecordEnum>,
        table_access: Option<MemoryRecordEnum>,
        dataop_event: Option<DataOpEvent>,
    ) {
        let event = CallEvent {
            shard: self.shard(),
            clk,
            pc,
            next_pc,
            opcode,
            call_sp,
            next_call_sp,
            signature_id,
            func_ref,
            table_id,
            table_idx,
            call_stack_access,
            table_access,
        };
        self.record.call_events.push(event);
        if let Some(event) = dataop_event {
            self.record.dataop_events.push(event)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn emit_i64_event(
        &mut self,

        clk: u32,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        res_lo: u32,
        res_hi: u32,
        arg1: u32,
        arg2: u32,
        memory_access: MemoryAccessRecord,
    ) {
        let event = I64AluEvent {
            pc,
            opcode,
            a_lo: res_lo,
            a_hi: res_hi,
            b: arg1,
            c: arg2,
            code: opcode.code(),
            res_hi_addr: memory_access.res_hi_addr.unwrap().to_virtual_addr(),
            res_hi_access: memory_access.res_hi_record,
        };
        match opcode {
            Opcode::I32Mul64 => self.record.mul64_events.push(event),
            Opcode::I32Add64 => self.record.add64_events.push(event),
            _ => {
                unreachable!();
            }
        }
    }

    /// Execute an ecall opcode.
    #[allow(clippy::type_complexity)]
    #[allow(unreachable_code)]
    #[allow(dead_code)]
    fn execute_ecall(
        &mut self,
    ) -> Result<(u32, u32, u32, u32, u32, SyscallCode, u32), ExecutionError> {
        // We peek at register x5 to get the syscall id. The reason we don't `self.rr` this
        // register is that we write to it later.
        todo!();
        let syscall_id = todo!();
        let c = todo!();
        let b = todo!();
        let syscall = SyscallCode::from_u32(syscall_id);

        if self.print_report && !self.unconstrained {
            self.report.syscall_counts[syscall] += 1;
        }

        // `hint_slice` is allowed in unconstrained mode since it is used to write the hint.
        // Other syscalls are not allowed because they can lead to non-deterministic
        // behavior, especially since many syscalls modify memory in place,
        // which is not permitted in unconstrained mode. This will result in
        // non-zero memory interactions when generating a proof.

        if self.unconstrained &&
            (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
        {
            return Err(ExecutionError::InvalidSyscallUsage(syscall_id as u64));
        }

        // Update the syscall counts.
        let syscall_for_count = syscall.count_map();
        let syscall_count = self.state.syscall_counts.entry(syscall_for_count).or_insert(0);
        *syscall_count += 1;

        let syscall_impl = self.get_syscall(syscall).cloned();
        let mut precompile_rt = SyscallContext::new(self);
        let (a, precompile_next_pc, precompile_cycles, returned_exit_code) =
            if let Some(syscall_impl) = syscall_impl {
                // Executing a syscall optionally returns a value to write to the t0
                // register. If it returns None, we just keep the
                // syscall_id in t0.
                let res = syscall_impl.execute(&mut precompile_rt, syscall, b, c);
                let a = if let Some(val) = res { val } else { syscall_id };

                // If the syscall is `HALT` and the exit code is non-zero, return an error.
                if syscall == SyscallCode::HALT && precompile_rt.exit_code != 0 {
                    return Err(ExecutionError::HaltWithNonZeroExitCode(precompile_rt.exit_code));
                }

                (a, precompile_rt.next_pc, syscall_impl.num_extra_cycles(), precompile_rt.exit_code)
            } else {
                return Err(ExecutionError::UnsupportedSyscall(syscall_id));
            };

        if let (Some(estimator), Some(syscall_id)) =
            (&mut self.record_estimator, syscall.as_air_id())
        {
            let threshold = match syscall_id {
                RwasmAirId::ShaExtend => self.opts.split_opts.sha_extend,
                RwasmAirId::ShaCompress => self.opts.split_opts.sha_compress,
                RwasmAirId::KeccakPermute => self.opts.split_opts.keccak,
                _ => self.opts.split_opts.deferred,
            } as u64;
            let shards = &mut estimator.precompile_records[syscall_id];
            let local_memory_ct =
                estimator.current_precompile_touched_compressed_addresses.len() as u64;
            match shards.last_mut().filter(|shard| shard.0 < threshold) {
                Some((shard_precompile_event_ct, shard_local_memory_ct)) => {
                    *shard_precompile_event_ct += 1;
                    *shard_local_memory_ct += local_memory_ct;
                }
                None => shards.push((1, local_memory_ct)),
            }
            estimator.current_precompile_touched_compressed_addresses.clear();
        }

        // // If the syscall is `EXIT_UNCONSTRAINED`, the memory was restored to pre-unconstrained
        // code // in the execute function, so we need to re-read from x10 and x11.  Just do
        // a peek on the // registers.
        // let (b, c) = if syscall == SyscallCode::EXIT_UNCONSTRAINED {
        //     (self.register(Register::X10), self.register(Register::X11))
        // } else {
        //     (b, c)
        // };

        // Allow the syscall impl to modify state.clk/pc (exit unconstrained does this)
        // self.rw_cpu(t0, a); TODO:check wether we need this
        let clk = self.state.clk;
        self.state.clk += precompile_cycles;

        Ok((a, b, c, clk, precompile_next_pc, syscall, returned_exit_code))
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute_cycle(&mut self, res: Result<bool, TrapCode>) -> Result<bool, ExecutionError> {
        let res = match res {
            Ok(value) => Ok(value),
            Err(err) => {
                return if err == TrapCode::UnreachableCodeReached {
                    Ok(true)
                } else {
                    println!("Err:{},", err);
                    Err(ExecutionError::Unimplemented())
                }
            }
        };

        let op_state = self.store.tracer.logs.last().unwrap();
        let syscall = SyscallCode::default();
        let dataop_event = match op_state.opcode {
            Opcode::CallIndirect(_) => Some(self.store.tracer.data_op_logs.last().unwrap()),
            _ => None,
        };

        self.state.clk = op_state.clk;
        self.state.pc = op_state.pc;
        self.emit_events(
            op_state.clk,
            op_state.pc,
            op_state.next_pc,
            op_state.sp,
            op_state.next_sp,
            op_state.call_sp,
            op_state.next_call_sp,
            op_state.opcode,
            syscall,
            op_state.arg1,
            op_state.arg2,
            op_state.res,
            op_state.res_hi,
            op_state.memory_access,
            op_state.call_state,
            op_state.fat_op.clone(),
            dataop_event.cloned(),
        );

        // Increment the clock.
        self.state.global_clk += 1;

        if self.unconstrained {
            self.unconstrained_state.total_unconstrained_cycles += 1;
        }

        if !self.unconstrained {
            // If there's not enough cycles left for another opcode, move to the next shard.
            let cpu_exit = self.max_syscall_cycles + self.state.clk >= self.shard_size;

            // Every N cycles, check if there exists at least one shape that fits.
            //
            // If we're close to not fitting, early stop the shard to ensure we don't OOM.
            let mut shape_match_found = true;
            if self.state.global_clk.is_multiple_of(self.shape_check_frequency) {
                // Estimate the number of events in the trace.

                // Check if the LDE size is too large.
                if self.lde_size_check {
                    let padded_event_counts =
                        pad_rv32im_event_counts(self.event_counts, self.shape_check_frequency);
                    let padded_lde_size = estimate_riscv_lde_size(padded_event_counts, &self.costs);
                    if padded_lde_size > self.lde_size_threshold {
                        #[allow(clippy::cast_precision_loss)]
                        let size_gib = (padded_lde_size as f64) / (1 << 9) as f64;
                        tracing::warn!(
                            "Stopping shard early since the estimated LDE size is too large: {:.3} GiB",
                            size_gib
                        );
                        shape_match_found = false;
                    }
                }
                // Check if we're too "close" to a maximal shape.
                else if let Some(maximal_shapes) = &self.maximal_shapes {
                    let distance = |threshold: usize, count: usize| {
                        if count != 0 {
                            threshold - count
                        } else {
                            usize::MAX
                        }
                    };

                    shape_match_found = false;

                    for shape in maximal_shapes.iter() {
                        let cpu_threshold = shape[CoreAirId::Cpu];
                        if self.state.clk > ((1 << cpu_threshold) << 2) {
                            continue;
                        }

                        let mut l_infinity = usize::MAX;
                        let mut shape_too_small = false;
                        for air in CoreAirId::iter() {
                            if air == CoreAirId::Cpu {
                                continue;
                            }

                            let threshold = 1 << shape[air];
                            let count = self.event_counts[RwasmAirId::from(air)] as usize;
                            if count > threshold {
                                shape_too_small = true;
                                break;
                            }

                            if distance(threshold, count) < l_infinity {
                                l_infinity = distance(threshold, count);
                            }
                        }

                        if shape_too_small {
                            continue;
                        }

                        if l_infinity >= 32 * (self.shape_check_frequency as usize) {
                            shape_match_found = true;
                            break;
                        }
                    }

                    if !shape_match_found {
                        self.record.counts = Some(self.event_counts);
                        tracing::debug!(
                            "Stopping shard {} to stay within some maximal shape. clk = {} pc = 0x{:x?}",
                            self.shard(),
                            self.state.global_clk,
                            self.state.pc,
                        );
                    }
                }
            }

            if cpu_exit || !shape_match_found {
                self.bump_record();
                self.state.current_shard += 1;
                self.state.clk = 0;
            }

            // If the cycle limit is exceeded, return an error.
            if let Some(max_cycles) = self.max_cycles {
                if self.state.global_clk > max_cycles {
                    return Err(ExecutionError::ExceededCycleLimit(max_cycles));
                }
            }
        }
        res
    }

    /// Bump the record.
    pub fn bump_record(&mut self) {
        if let Some(estimator) = &mut self.record_estimator {
            self.local_counts.local_mem = std::mem::take(&mut estimator.current_local_mem);
            // Self::estimate_riscv_event_counts(
            //     &mut self.event_counts,
            //     (self.state.clk >> 2) as u64,
            //     &self.local_counts,
            // );
            // The above method estimates event counts only for core shards.
            estimator.core_records.push(self.event_counts);
            estimator.current_touched_compressed_addresses.clear();
        }
        self.local_counts = LocalCounts::default();
        // Copy all of the existing local memory accesses to the record's local_memory_access vec.
        if self.executor_mode == ExecutorMode::Trace {
            for (_, event) in self.store.tracer.local_memory_event.drain() {
                self.record.cpu_local_memory_access.push(event);
            }
        }

        let removed_record = std::mem::replace(
            &mut self.record,
            Box::new(ExecutionRecord::new(self.program.clone())),
        );
        let public_values = removed_record.public_values;
        self.record.public_values = public_values;
        self.records.push(removed_record);
    }
    /// Execute up to `self.shard_batch_size` cycles, returning the events emitted and whether the
    /// program ended.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn execute_record(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(Vec<Box<ExecutionRecord>>, bool), ExecutionError> {
        self.executor_mode = ExecutorMode::Trace;
        self.emit_global_memory_events = emit_global_memory_events;
        self.print_report = true;

        let done = self.execute()?; //TODO: fix execute

        Ok((std::mem::take(&mut self.records), done))
    }

    /// Execute up to `self.shard_batch_size` cycles, returning the checkpoint from before execution
    /// and whether the program ended.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn execute_state(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(ExecutionState, PublicValues<u32, u32>, bool), ExecutionError> {
        self.memory_checkpoint.clear();
        self.executor_mode = ExecutorMode::Checkpoint;
        self.emit_global_memory_events = emit_global_memory_events;

        // Clone self.state without memory, uninitialized_memory, proof_stream in it so it's faster.
        let memory = std::mem::take(&mut self.state.memory);
        let uninitialized_memory = std::mem::take(&mut self.state.uninitialized_memory);
        let proof_stream = std::mem::take(&mut self.state.proof_stream);
        let mut checkpoint = tracing::debug_span!("clone").in_scope(|| self.state.clone());
        self.state.memory = memory;
        self.state.uninitialized_memory = uninitialized_memory;
        self.state.proof_stream = proof_stream;

        let done = tracing::debug_span!("execute").in_scope(|| self.execute())?;
        println!("cpu events:{:?}", self.record.cpu_events);
        println!("data_op events{:?}", self.record.dataop_events);
        // Create a checkpoint using `memory_checkpoint`. Just include all memory if `done` since we
        // need it all for MemoryFinalize.
        let next_pc = self.state.pc;
        tracing::debug_span!("create memory checkpoint").in_scope(|| {
            let replacement_memory_checkpoint = Memory::<_>::new_preallocated();
            let replacement_uninitialized_memory_checkpoint = Memory::<_>::new_preallocated();
            let memory_checkpoint =
                std::mem::replace(&mut self.memory_checkpoint, replacement_memory_checkpoint);
            let uninitialized_memory_checkpoint = std::mem::replace(
                &mut self.uninitialized_memory_checkpoint,
                replacement_uninitialized_memory_checkpoint,
            );
            if done && !self.emit_global_memory_events {
                // If it's the last shard, and we're not emitting memory events, we need to include
                // all memory so that memory events can be emitted from the checkpoint. But we need
                // to first reset any modified memory to as it was before the execution.
                checkpoint.memory.clone_from(&self.state.memory);
                memory_checkpoint.into_iter().for_each(|(addr, record)| {
                    if let Some(record) = record {
                        checkpoint.memory.insert(addr, record);
                    } else {
                        checkpoint.memory.remove(addr);
                    }
                });
                checkpoint.uninitialized_memory = self.state.uninitialized_memory.clone();
                // Remove memory that was written to in this batch.
                for (addr, is_old) in uninitialized_memory_checkpoint {
                    if !is_old {
                        checkpoint.uninitialized_memory.remove(addr);
                    }
                }
            } else {
                checkpoint.memory = memory_checkpoint
                    .into_iter()
                    .filter_map(|(addr, record)| record.map(|record| (addr, record)))
                    .collect();
                checkpoint.uninitialized_memory = uninitialized_memory_checkpoint
                    .into_iter()
                    .filter(|&(_, has_value)| has_value)
                    .map(|(addr, _)| (addr, *self.state.uninitialized_memory.get(addr).unwrap()))
                    .collect();
            }
        });
        let mut public_values = self.records.last().as_ref().unwrap().public_values;
        public_values.start_pc = next_pc;
        public_values.next_pc = next_pc;
        if !done {
            self.records.clear();
        }
        Ok((checkpoint, public_values, done))
    }

    #[allow(dead_code)]
    fn initialize(&mut self) {
        self.state.clk = 0;

        tracing::debug!("loading memory image");
        for (addr, value) in self.program.memory_image.iter() {
            self.state.memory.insert(*addr, MemoryRecord { value: *value, shard: 0, timestamp: 0 });
        }
    }

    /// Executes the program without tracing and without emitting events.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run_fast(&mut self) -> Result<(), ExecutionError> {
        self.executor_mode = ExecutorMode::Simple;
        self.print_report = true;
        while !self.execute()? {}

        #[cfg(feature = "profiling")]
        if let Some((profiler, writer)) = self.profiler.take() {
            profiler.write(writer).expect("Failed to write profile to output file");
        }

        Ok(())
    }

    /// Executes the program in checkpoint mode, without emitting the checkpoints.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run_checkpoint(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(), ExecutionError> {
        self.executor_mode = ExecutorMode::Simple;
        self.print_report = true;
        while !self.execute_state(emit_global_memory_events)?.2 {}
        Ok(())
    }

    /// Executes the program and prints the execution report.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run(&mut self) -> Result<(), ExecutionError> {
        self.executor_mode = ExecutorMode::Trace;
        self.print_report = true;
        while !self.execute()? {}

        #[cfg(feature = "profiling")]
        if let Some((profiler, writer)) = self.profiler.take() {
            profiler.write(writer).expect("Failed to write profile to output file");
        }

        Ok(())
    }

    /// Executes up to `self.shard_batch_size` cycles of the program, returning whether the program
    /// has finished.
    pub fn execute(&mut self) -> Result<bool, ExecutionError> {
        // Get the program.
        let program = self.program.clone();

        // Get the current shard.
        let start_shard = self.state.current_shard;

        // If it's the first cycle, initialize the program.
        if self.state.global_clk == 0 {
            // We initialize in tracer

            // self.initialize();
        }

        let unconstrained_cycle_limit =
            std::env::var("UNCONSTRAINED_CYCLE_LIMIT").map(|v| v.parse::<u64>().unwrap()).ok();

        // Loop until we've executed `self.shard_batch_size` shards if `self.shard_batch_size` is
        // set.
        let mut done = false;
        let mut current_shard = self.state.current_shard;
        let mut num_shards_executed = 0;

        self.store.tracer.state.next_shard(); // shard starts with 1;

        loop {
            let rwasm_state = self.register_state.get_or_insert_with(|| {
                let sp = self.value_stack.stack_ptr();
                let ip = InstructionPtr::new((*self.program.module.code_section).as_ptr());
                RwasmExecutorState { sp, ip }
            });
            let res: Result<bool, TrapCode>;
            (res, rwasm_state.ip, rwasm_state.sp) = RwasmExecutor::new(
                &self.program.module,
                &mut self.value_stack,
                rwasm_state.sp,
                &mut self.call_stack,
                rwasm_state.ip,
                &mut self.store,
            )
            .step();

            self.postprocess_syscall();

            let res = self.execute_cycle(res)?;
            if res {
                done = true;
                break;
            }

            // Check if the unconstrained cycle limit was exceeded.
            if let Some(unconstrained_cycle_limit) = unconstrained_cycle_limit {
                if self.unconstrained_state.total_unconstrained_cycles > unconstrained_cycle_limit {
                    return Err(ExecutionError::UnconstrainedCycleLimitExceeded(
                        unconstrained_cycle_limit,
                    ));
                }
            }

            if self.shard_batch_size > 0 && current_shard != self.state.current_shard {
                num_shards_executed += 1;
                current_shard = self.state.current_shard;
                if num_shards_executed == self.shard_batch_size {
                    println!("break bad");
                    break;
                }
            }
        }

        // Get the final public values.
        let public_values = self.record.public_values;
        self.state.update_state(&self.store);
        if done {
            self.postprocess();

            // Push the remaining execution record with memory initialize & finalize events.
            self.bump_record();

            // Flush stdout and stderr.
            if let Some(ref mut w) = self.io_options.stdout {
                if let Err(e) = w.flush() {
                    tracing::error!("failed to flush stdout override: {e}");
                }
            }

            if let Some(ref mut w) = self.io_options.stderr {
                if let Err(e) = w.flush() {
                    tracing::error!("failed to flush stderr override: {e}");
                }
            }
        }

        // Push the remaining execution record, if there are any CPU events.
        if !self.record.cpu_events.is_empty() {
            self.bump_record();
        }

        // Set the global public values for all shards.
        let mut last_next_pc = 0;
        let mut last_exit_code = 0;
        for (i, record) in self.records.iter_mut().enumerate() {
            record.program = program.clone();
            record.public_values = public_values;
            record.public_values.committed_value_digest = public_values.committed_value_digest;
            record.public_values.deferred_proofs_digest = public_values.deferred_proofs_digest;
            record.public_values.execution_shard = start_shard + i as u32;
            if record.cpu_events.is_empty() {
                record.public_values.start_pc = last_next_pc;
                record.public_values.next_pc = last_next_pc;
                record.public_values.exit_code = last_exit_code;
            } else {
                record.public_values.start_pc = record.cpu_events[0].pc;
                record.public_values.next_pc = record.cpu_events.last().unwrap().next_pc;
                record.public_values.exit_code = record.cpu_events.last().unwrap().exit_code;
                last_next_pc = record.public_values.next_pc;
                last_exit_code = record.public_values.exit_code;
            }
        }

        Ok(done)
    }

    fn postprocess(&mut self) {
        // Flush remaining stdout/stderr
        for (fd, buf) in &self.io_buf {
            if !buf.is_empty() {
                match fd {
                    1 => {
                        eprintln!("stdout: {buf}");
                    }
                    2 => {
                        eprintln!("stderr: {buf}");
                    }
                    _ => {}
                }
            }
        }

        // Ensure that all proofs and input bytes were read, otherwise warn the user.
        if self.state.proof_stream_ptr != self.state.proof_stream.len() {
            tracing::warn!(
                "Not all proofs were read. Proving will fail during recursion. Did you pass too
        many proofs in or forget to call verify_sp1_proof?"
            );
        }

        if !self.state.input_stream.is_empty() {
            tracing::warn!("Not all input bytes were read.");
        }

        if let Some(estimator) = &mut self.record_estimator {
            // Mirror the logic below.
            // Register 0 is always init and finalized, so we add 1
            // registers 1..32
            // let touched_reg_ct =
            //     1 + (1..32).filter(|&r| self.state.memory.registers.get(r).is_some()).count();
            let total_mem = self.state.memory.page_table.exact_len(); //TODO: fix estimator
                                                                      // The memory_image is already initialized in the MemoryProgram chip
                                                                      // so we subtract it off. It is initialized in the executor in the `initialize`
                                                                      // function.
            estimator.memory_global_init_events = total_mem
                .checked_sub(self.record.program.module.data_section.len())
                .expect("program memory image should be accounted for in memory exact len")
                as u64;
            estimator.memory_global_finalize_events = total_mem as u64;
        }

        if self.emit_global_memory_events &&
            (self.executor_mode == ExecutorMode::Trace ||
                self.executor_mode == ExecutorMode::Checkpoint)
        {
            // SECTION: Set up all MemoryInitializeFinalizeEvents needed for memory argument.
            let memory_finalize_events = &mut self.record.global_memory_finalize_events;
            memory_finalize_events.reserve_exact(self.state.memory.page_table.estimate_len() + 32);

            // We handle the addr = 0 case separately, as we constrain it to be 0 in the first row
            // of the memory finalize table so it must be first in the array of events.
            let addr_0_record = self.state.memory.get(0);

            let addr_0_final_record = match addr_0_record {
                Some(record) => record,
                None => &MemoryRecord { value: 0, shard: 0, timestamp: 0 },
            };

            memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, addr_0_final_record));

            let memory_initialize_events = &mut self.record.global_memory_initialize_events;
            let addr_0_initialize_event = MemoryInitializeFinalizeEvent::initialize(0, 0, true);
            memory_initialize_events.push(addr_0_initialize_event);

            // let memory_initialize_events = &mut self.record.global_memory_initialize_events;
            // let addr_0_init = MemoryInitializeFinalizeEvent { addr: 0, value: 0, shard: 1,
            // timestamp: 1, used: 1 }; let addr_0_final =
            // MemoryInitializeFinalizeEvent{ addr: 0, value: 0, shard: 1, timestamp: 2, used: 1 };
            // let addr_0_initialize_event =
            //     MemoryInitializeFinalizeEvent::initialize(0, 0, true);
            // println!("addr0init:{:?}",addr_0_initialize_event);
            // memory_initialize_events.push(addr_0_init);
            // memory_finalize_events.push(addr_0_final);

            // Count the number of touched memory addresses manually, since `PagedMemory` doesn't
            // already know its length.
            self.report.touched_memory_addresses = 0;

            for addr in self.state.memory.page_table.keys() {
                if !self.program.memory_image.contains_key(&addr) {
                    println!("addr:{}", addr);
                    self.report.touched_memory_addresses += 1;

                    // Program memory is initialized in the MemoryProgram chip and doesn't require
                    // any events, so we only send init events for other memory
                    // addresses.

                    let initial_value = self.state.uninitialized_memory.get(addr).unwrap_or(&0);
                    let init_event =
                        MemoryInitializeFinalizeEvent::initialize(addr, *initial_value, true);
                    println!("init_event:{:?}", init_event);
                    memory_initialize_events.push(init_event);
                }
                let record = *self.state.memory.get(addr).unwrap();
                let final_event =
                    MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record);
                println!("final_event:{:?}", final_event);
                memory_finalize_events.push(final_event);
            }
        }
    }

    pub fn postprocess_syscall(&mut self) {
        if !self.unconstrained && self.executor_mode == ExecutorMode::Trace {
            // Will need to transfer the existing memory local events in the executor to it's
            // record, and return all the syscall memory local events.  This is similar
            // to what `bump_record` does.

            if let Some(op_state) = self.store.tracer.logs.last() {
                let addrs = match &op_state.fat_op {
                    Some(FatOpEvent::TableInit(event)) => &event.local_mem_access_addr,
                    Some(FatOpEvent::TableGrow(event)) => &event.local_mem_access_addr,
                    _ => &Vec::new(),
                };

                for addr in addrs {
                    let local_mem_access = self.store.tracer.local_memory_event.remove(addr);

                    if let Some(local_mem_access) = local_mem_access {
                        self.record.cpu_local_memory_access.push(local_mem_access);
                    }
                }
            }
        }
    }

    fn get_syscall(&mut self, code: SyscallCode) -> Option<&Arc<dyn Syscall>> {
        self.syscall_map.get(&code)
    }

    // /// Maps the opcode counts to the number of events in each air.
    // fn estimate_riscv_event_counts(
    //     event_counts: &mut EnumMap<RwasmAirId, u64>,
    //     cpu_cycles: u64,
    //     local_counts: &LocalCounts,
    // ) {
    //     let touched_addresses: u64 = local_counts.local_mem as u64;
    //     let syscalls_sent: u64 = local_counts.syscalls_sent as u64;
    //     let opcode_counts: &EnumMap<u8, u64> = &local_counts.event_counts;

    //     // Compute the number of events in the cpu chip.
    //     event_counts[RwasmAirId::Cpu] = cpu_cycles;

    //     // Compute the number of events in the add sub chip.
    //     event_counts[RwasmAirId::AddSub] = opcode_counts[Opcode::ADD] +
    // opcode_counts[Opcode::SUB];

    //     // Compute the number of events in the mul chip.
    //     event_counts[RwasmAirId::Mul] = opcode_counts[Opcode::MUL]
    //         + opcode_counts[Opcode::MULH]
    //         + opcode_counts[Opcode::MULHU]
    //         + opcode_counts[Opcode::MULHSU];

    //     // Compute the number of events in the bitwise chip.
    //     event_counts[RwasmAirId::Bitwise] =
    //         opcode_counts[Opcode::XOR] + opcode_counts[Opcode::OR] + opcode_counts[Opcode::AND];

    //     // Compute the number of events in the shift left chip.
    //     event_counts[RwasmAirId::ShiftLeft] = opcode_counts[Opcode::SLL];

    //     // Compute the number of events in the shift right chip.
    //     event_counts[RwasmAirId::ShiftRight] =
    //         opcode_counts[Opcode::SRL] + opcode_counts[Opcode::SRA];

    //     // Compute the number of events in the divrem chip.
    //     event_counts[RwasmAirId::DivRem] = opcode_counts[Opcode::DIV]
    //         + opcode_counts[Opcode::DIVU]
    //         + opcode_counts[Opcode::REM]
    //         + opcode_counts[Opcode::REMU];

    //     // Compute the number of events in the lt chip.
    //     event_counts[RwasmAirId::Lt] = opcode_counts[Opcode::SLT] + opcode_counts[Opcode::SLTU];

    //     // Compute the number of events in the memory local chip.
    //     event_counts[RwasmAirId::MemoryLocal] =
    //         touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);

    //     // Compute the number of events in the branch chip.
    //     event_counts[RwasmAirId::Branch] = opcode_counts[Opcode::BEQ]
    //         + opcode_counts[Opcode::BNE]
    //         + opcode_counts[Opcode::BLT]
    //         + opcode_counts[Opcode::BGE]
    //         + opcode_counts[Opcode::BLTU]
    //         + opcode_counts[Opcode::BGEU];

    //     // Compute the number of events in the jump chip.
    //     event_counts[RwasmAirId::Jump] = opcode_counts[Opcode::JAL] +
    // opcode_counts[Opcode::JALR];

    //     // Compute the number of events in the auipc chip.
    //     event_counts[RwasmAirId::Auipc] = opcode_counts[Opcode::AUIPC]
    //         + opcode_counts[Opcode::UNIMP]
    //         + opcode_counts[Opcode::EBREAK];

    //     // Compute the number of events in the memory opcode chip.
    //     event_counts[RwasmAirId::MemoryInstrs] = opcode_counts[Opcode::LB]
    //         + opcode_counts[Opcode::LH]
    //         + opcode_counts[Opcode::LW]
    //         + opcode_counts[Opcode::LBU]
    //         + opcode_counts[Opcode::LHU]
    //         + opcode_counts[Opcode::SB]
    //         + opcode_counts[Opcode::SH]
    //         + opcode_counts[Opcode::SW];

    //     // Compute the number of events in the syscall opcode chip.
    //     event_counts[RwasmAirId::SyscallInstrs] = opcode_counts[Opcode::ECALL];

    //     // Compute the number of events in the syscall core chip.
    //     event_counts[RwasmAirId::SyscallCore] = syscalls_sent;

    //     // Compute the number of events in the global chip.
    //     event_counts[RwasmAirId::Global] =
    //         2 * touched_addresses + event_counts[RwasmAirId::SyscallInstrs];

    //     // Adjust for divrem dependencies.
    //     event_counts[RwasmAirId::Mul] += event_counts[RwasmAirId::DivRem];
    //     event_counts[RwasmAirId::Lt] += event_counts[RwasmAirId::DivRem];

    //     // Note: we ignore the additional dependencies for addsub, since they are accounted for
    // in     // the maximal shapes.
    // }

    #[inline]
    #[allow(dead_code)]
    fn log(&mut self, _: &Opcode) {
        #[cfg(feature = "profiling")]
        if let Some((ref mut profiler, _)) = self.profiler {
            if !self.unconstrained {
                profiler.record(self.state.global_clk, self.state.pc as u64);
            }
        }

        if !self.unconstrained && self.state.global_clk.is_multiple_of(10_000_000) {
            tracing::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc);
        }
    }
}

/// Aligns an address to the nearest word below or equal to it.
#[must_use]
pub const fn align(addr: u32) -> u32 {
    addr - addr % 4
}

#[cfg(test)]
mod tests {
    #![allow(clippy::vec_init_then_push)]

    use crate::{align, ExecutionError, Executor, Program};
    use hashbrown::HashMap;

    use rwasm::{
        mem_index::{TypedAddress, SP_START, UNIT},
        BranchOffset, I64ValueSplit, Opcode,
    };
    use sp1_stark::SP1CoreOpts;

    fn peek_stack(rt: &Executor) {
        let start = SP_START;
        for idx in 1..16 {
            let rec = rt.state.memory.get(SP_START - 4 * idx);
            match rec {
                Some(rec) => {
                    println!("addr: {}, pos:{},val:{}", SP_START - 4 * idx, idx, rec.value);
                }
                None => {
                    println!("pos:{},empty", idx);
                }
            }
        }
        for idx in 1..16 {
            let rec = rt.state.memory.get(SP_START + 4 * idx);
            match rec {
                Some(rec) => {
                    println!("Error ! addr: {},pos:-{},val:{}", SP_START + 4 * idx, idx, rec.value);
                }
                None => {
                    println!("pos:-{},empty", idx);
                }
            }
        }
    }

    #[test]
    fn test_align_various() {
        // align() returns the nearest word boundary below or equal to addr
        assert_eq!(align(0), 0);
        assert_eq!(align(1), 0);
        assert_eq!(align(3), 0);
        assert_eq!(align(4), 4);
        assert_eq!(align(7), 4);
        assert_eq!(align(8), 8);
        assert_eq!(align(0xFFFF_FFFF), 0xFFFF_FFFC);
    }
    #[test]
    fn test_word_and_byte_reads() {
        // Directly exercise mw/word/byte helpers (little-endian view)
        let program = Program::from_instrs(vec![]);
        let mut rt = Executor::new(program, SP1CoreOpts::default());

        // Uninitialized words read as 0
        assert_eq!(rt.word(8), 0);

        // Write a word and check byte-level reads
        let val: u32 = 0x11_22_33_44;
        rt.mw(8, val, /* shard= */ 0, /* timestamp= */ 1, None);

        assert_eq!(rt.word(8), val);
        assert_eq!(rt.byte(8), 0x44); // least-significant byte at lowest address
        assert_eq!(rt.byte(9), 0x33);
        assert_eq!(rt.byte(10), 0x22);
        assert_eq!(rt.byte(11), 0x11);
    }
    #[test]
    fn test_cycle_limit_exceeded() {
        // With max_cycles = 0, any execution should exceed the limit on the first step
        let program = Program::from_instrs(vec![Opcode::I32Const(1u32.into())]);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.max_cycles = Some(0);

        let err = rt.run().expect_err("expected cycle limit to be exceeded");
        assert!(matches!(err, ExecutionError::ExceededCycleLimit(0)));
    }

    #[test]
    fn test_branch_ifnez_not_taken() {
        // Mirrors build_elf_branching but uses 0 so the branch is NOT taken,
        // thus both trailing adds execute and final becomes 15.
        let x = 1u32;
        let opcodes = vec![
            Opcode::I32Const(x.into()),       // 1
            Opcode::I32Const((x + 1).into()), // 2
            Opcode::I32Const((x + 2).into()), // 3
            Opcode::I32Add,                   // 2 + 3 = 5
            Opcode::I32Add,                   // 1 + 5 = 6
            Opcode::I32Const(0u32.into()),    // 0 -> branch NOT taken
            Opcode::BrIfNez(BranchOffset::from(16i32)),
            Opcode::I32Const((x + 3).into()), // 4
            Opcode::I32Const((x + 4).into()), // 5
            Opcode::I32Add,                   // 4 + 5 = 9
            Opcode::I32Add,                   // 6 + 9 = 15
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 15);
    }

    /// Branch is TAKEN when condition is non-zero; block is skipped and earlier result remains.
    #[test]
    fn test_branch_ifnez_taken() {
        // Start with the same arithmetic prelude as the not-taken test
        let x = 1u32;
        let opcodes = vec![
            Opcode::I32Const(x.into()),                 // 1
            Opcode::I32Const((x + 1).into()),           // 2
            Opcode::I32Const((x + 2).into()),           // 3
            Opcode::I32Add,                             // 2 + 3 = 5
            Opcode::I32Add,                             // 1 + 5 = 6
            Opcode::I32Const(1u32.into()),              // non-zero -> branch TAKEN
            Opcode::BrIfNez(BranchOffset::from(16i32)), // skip 4-op block below
            // --- skipped block if branch taken ---
            Opcode::I32Const((x + 3).into()), // 4
            Opcode::I32Const((x + 4).into()), // 5
            Opcode::I32Add,                   // 4 + 5 = 9
            Opcode::I32Add,                   // 6 + 9 = 15
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        // since the branch was taken, the 4-op block is skipped; final stays 6
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 6);
    }

    /// Non-zero can be any value (including 0xFFFF_FFFF); ensure branch is taken.
    #[test]
    fn test_branch_ifnez_taken_with_max_nonzero() {
        let x = 2u32;
        let opcodes = vec![
            Opcode::I32Const(x.into()),              // 2
            Opcode::I32Const((x + 1).into()),        // 3
            Opcode::I32Const((x + 2).into()),        // 4
            Opcode::I32Add,                          // 3 + 4 = 7
            Opcode::I32Add,                          // 2 + 7 = 9
            Opcode::I32Const(0xFFFF_FFFFu32.into()), // still non-zero
            Opcode::BrIfNez(BranchOffset::from(16i32)),
            // --- would add 5 and 6 if not skipped ---
            Opcode::I32Const((x + 3).into()), // 5
            Opcode::I32Const((x + 4).into()), // 6
            Opcode::I32Add,                   // 5 + 6 = 11
            Opcode::I32Add,                   // 9 + 11 = 20
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        // branch taken -> block skipped -> result remains 9
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 9);
    }

    /// Two conditional blocks: first NOT taken (executes), second TAKEN (skips).
    #[test]
    fn test_branch_ifnez_two_blocks() {
        let x = 1u32;
        let opcodes = vec![
            // Prelude -> 6
            Opcode::I32Const(x.into()),       // 1
            Opcode::I32Const((x + 1).into()), // 2
            Opcode::I32Const((x + 2).into()), // 3
            Opcode::I32Add,                   // 2 + 3 = 5
            Opcode::I32Add,                   // 1 + 5 = 6
            // First branch: NOT taken (cond = 0) -> execute next 4 ops (block A)
            Opcode::I32Const(0u32.into()),
            Opcode::BrIfNez(BranchOffset::from(16i32)),
            // ---- block A (executes) ----
            Opcode::I32Const((x + 3).into()), // 4
            Opcode::I32Const((x + 4).into()), // 5
            Opcode::I32Add,                   // 4 + 5 = 9
            Opcode::I32Add,                   // 6 + 9 = 15
            // Second branch: TAKEN (cond = 1) -> skip next 4 ops (block B)
            Opcode::I32Const(1u32.into()),
            Opcode::BrIfNez(BranchOffset::from(16i32)),
            // ---- block B (skipped) ----
            Opcode::I32Const(10u32.into()),
            Opcode::I32Const(20u32.into()),
            Opcode::I32Add, // 10 + 20 = 30
            Opcode::I32Add, // would be 15 + 30 = 45 if not skipped
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        // block A executed (+9), block B skipped; final remains 15
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 15);
    }
    // ---  store8 + store8 -> load16U (endianness sanity) ---
    #[test]
    fn test_store8_then_load16u() {
        let sp0 = SP_START;
        let addr: u32 = 0x10000;
        let lo = 0xAAu32;
        let hi = 0xBBu32; // expect 0xBBAA when read as 16-bit little endian
        let expected = (hi << 8) | lo;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            // write low byte
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(lo.into()),
            Opcode::I32Store8(0u32),
            // write high byte
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(hi.into()),
            Opcode::I32Store8(1u32),
            // read 16 bits
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16U(0u32),
            // compare
            Opcode::I32Const(expected.into()),
            Opcode::I32Eq,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
        assert_eq!(sp0, rt.state.sp + 2 * UNIT);
    }

    // ---  LtS vs LtU should diverge for (-1, 1) ---
    #[test]
    fn test_lt_u() {
        let x = 5u32;
        let y = 10u32;
        let opcodes = vec![
            // signed LT should be true
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32LtU,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    #[test]
    fn test_lts() {
        let x = neg(3);
        let y = neg(1);
        let opcodes = vec![Opcode::I32Const(x.into()), Opcode::I32Const(y.into()), Opcode::I32LtS];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    #[test]
    fn test_gts() {
        let x = 5;
        let y = 3;
        let opcodes = vec![Opcode::I32Const(x.into()), Opcode::I32Const(y.into()), Opcode::I32GtS];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    #[test]
    fn test_lts_vs_ltu_diverge() {
        let x = neg(1); // 0xffffffff == -1 (signed)
        let y = 1u32;
        // Check: (x <_s y) == 1 AND (x <_u y) == 0
        let opcodes = vec![
            // signed LT should be true
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32LtS,
            Opcode::I32Const(1u32.into()),
            Opcode::I32Eq,
            // unsigned LT should be false
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32LtU,
            Opcode::I32Const(0u32.into()),
            Opcode::I32Eq,
            // both must be true -> AND == 1
            Opcode::I32And,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    //--- GtS vs GtU should diverge for (0x80000000, 0) ---
    #[test]
    fn test_gts_vs_gtu_diverge() {
        let x = 0x8000_0000u32; // negative if signed
        let y = 0u32;
        // Check: (x >_s y) == 0 AND (x >_u y) == 1
        let opcodes = vec![
            // signed GT should be false
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32GtS,
            Opcode::I32Const(0u32.into()),
            Opcode::I32Eq,
            // unsigned GT should be true
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32GtU,
            Opcode::I32Const(1u32.into()),
            Opcode::I32Eq,
            // both checks true -> AND == 1
            Opcode::I32And,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    // --- shift counts are masked mod 32 (33 -> 1, 65 -> 1) ---
    #[test]
    fn test_shift_count_masking() {
        let sp0 = SP_START;
        let opcodes = vec![
            // 1 << 33 == 1 << (33 % 32) == 2
            Opcode::I32Const(1u32.into()),
            Opcode::I32Const(33u32.into()),
            Opcode::I32Shl,
            Opcode::I32Const(2u32.into()),
            Opcode::I32Eq,
            // 8 >> 65 == 8 >> 1 == 4 (logical)
            Opcode::I32Const(8u32.into()),
            Opcode::I32Const(65u32.into()),
            Opcode::I32ShrU,
            Opcode::I32Const(4u32.into()),
            Opcode::I32Eq,
            // both equalities true
            Opcode::I32And,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
        assert_eq!(sp0, rt.state.sp + 4);
    }

    // --- div/rem identity: a = b*q + r (signed, non-overflow case) ---
    #[test]
    fn test_divrem_identity_signed() {
        let a: u32 = 123; // +123
        let b: u32 = neg(7); // -7
        let opcodes = vec![
            // q = a / b (signed)
            Opcode::I32Const(a.into()),
            Opcode::I32Const(b.into()),
            Opcode::I32DivS, // q
            Opcode::I32Const(b.into()),
            Opcode::I32Mul, // b*q
            // r = a % b (signed)
            Opcode::I32Const(a.into()),
            Opcode::I32Const(b.into()),
            Opcode::I32RemS, // r
            // b*q + r
            Opcode::I32Add,
            // compare to a
            Opcode::I32Const(a.into()),
            Opcode::I32Eq,
        ];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- unaligned mw/mr must panic (addr % 4 != 0) ---
    #[test]
    #[should_panic(expected = "Invalid memory access")]
    fn test_mw_unaligned_panics() {
        let program = Program::from_instrs(vec![]);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        // force unaligned write
        rt.mw(2, 0xDEAD_BEEFu32, 0, 1, None);
    }
    // --- event emission sanity: add should produce CPU & ALU events ---
    #[test]
    fn test_event_emission_add() {
        let opcodes =
            vec![Opcode::I32Const(10u32.into()), Opcode::I32Const(20u32.into()), Opcode::I32Add];
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        // After run(), shards are in rt.records
        let total_cpu: usize = rt.records.iter().map(|r| r.cpu_events.len()).sum();
        let total_add: usize = rt.records.iter().map(|r| r.add_events.len()).sum();

        assert!(total_cpu > 0, "expected CPU events");
        assert!(total_add > 0, "expected at least one add ALU event");
    }

    /// After writing individual bytes, a word read should reconstruct the same value (LE).
    #[test]
    fn test_word_read_after_byte_writes() {
        let program = Program::from_instrs(vec![]);
        let mut rt = Executor::new(program, SP1CoreOpts::default());

        let base: u32 = 8; // 4-byte aligned in the VM's virtual addressing
        let bytes = [0x12u32, 0x34, 0x56, 0x78]; // LE layout
        let mut w = 0u32;

        // Simulate four byte-writes by composing the word and committing via aligned mw()
        for (i, b) in bytes.into_iter().enumerate() {
            let shift = (i as u32) * 8;
            w = (w & !(0xFFu32 << shift)) | ((b & 0xFF) << shift);
            rt.mw(base, w, /* shard= */ 0, /* timestamp= */ (i as u32) + 1, None);
        }

        assert_eq!(rt.word(base), 0x7856_3412, "word should equal assembled bytes (LE)");
    }

    #[test]
    fn test_byte_overwrite_then_word_read() {
        let program = Program::from_instrs(vec![]);
        let mut rt = Executor::new(program, SP1CoreOpts::default());

        let base: u32 = 8; // aligned
        let initial: u32 = 0x1122_3344; // bytes in LE: [44, 33, 22, 11]
        rt.mw(base, initial, 0, 1, None);

        // Overwrite the third byte (offset 2) with 0x99 -> [44, 33, 0x99, 11]
        let expected: u32 = 0x1199_3344;
        let new_w = (initial & !(0xFFu32 << 16)) | (0x99u32 << 16);
        rt.mw(base, new_w, 0, 2, None);

        assert_eq!(rt.word(base), expected);
    }

    #[test]
    fn test_xor() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 37;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Xor, // 5 xor 37 = 32
            Opcode::I32Const((x_value ^ y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_or() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 37;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Or, // 5 or 37 = 32
            Opcode::I32Const((x_value | y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_and() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 37;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32And, // 5 and 37 = 32
            Opcode::I32Const((x_value & y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn test_ori() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 37;
        let z_value: u32 = 42;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Or, // 5 or 37 = 37
            Opcode::I32Or, // 37 or 42  = 47
            Opcode::I32Const((x_value | y_value | z_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_andi() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 37;
        let z_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32And, // 5 and 37 = 32
            Opcode::I32And, // 5 and 4  = 4
            Opcode::I32Const((x_value & y_value & z_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_mul() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let y_value: u32 = 32;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Mul, // 5 * 32 = 160
            Opcode::I32Const((x_value * y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_eq() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;
        let y_value: u32 = 32;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Eq, // check whether x_value is equal y_value
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_ne() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 1;
        let y_value: u32 = 32;
        let z_value: u32 = 1;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Ne, // check whether x_value is not equal y_value
            Opcode::I32Ne,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 0);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_eqz() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Eqz, // check whether x_value is zero
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 0);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_lts_ltu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0;
        let y_value: u32 = 32;
        let z_value: u32 = 233;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32LtS, // check whether signed x_value is less than signed y_value
            Opcode::I32LtU, // check whether unsigned x_value is less than unsigned y_value
        ];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn test_comparisons() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 233;

        for opcode in
            [Opcode::I32GeS, Opcode::I32LeS, Opcode::I32GeU, Opcode::I32LeU, Opcode::I32Eq]
        {
            let opcodes =
                vec![Opcode::I32Const(x_value.into()), Opcode::I32Const(x_value.into()), opcode];

            let program = Program::from_instrs(opcodes);
            let mut runtime = Executor::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
            assert_eq!(sp_value, runtime.state.sp + 4);
        }
        for opcode in
            [Opcode::I32GeS, Opcode::I32LeS, Opcode::I32GeU, Opcode::I32LeU, Opcode::I32Eq]
        {
            let opcodes = vec![
                Opcode::I32Const(neg(x_value).into()),
                Opcode::I32Const(neg(x_value).into()),
                opcode,
            ];

            let program = Program::from_instrs(opcodes);
            let mut runtime = Executor::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
            assert_eq!(sp_value, runtime.state.sp + 4);
        }
        for opcode in [Opcode::I32GeS, Opcode::I32Ne, Opcode::I32GtS] {
            let opcodes = vec![
                Opcode::I32Const(x_value.into()),
                Opcode::I32Const(neg(x_value).into()),
                opcode,
            ];

            let program = Program::from_instrs(opcodes);
            let mut runtime = Executor::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
            assert_eq!(sp_value, runtime.state.sp + 4);
        }
        for opcode in [Opcode::I32LeS, Opcode::I32Ne, Opcode::I32LtS] {
            let opcodes = vec![
                Opcode::I32Const(neg(x_value).into()),
                Opcode::I32Const(x_value.into()),
                opcode,
            ];

            let program = Program::from_instrs(opcodes);
            let mut runtime = Executor::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
            assert_eq!(sp_value, runtime.state.sp + 4);
        }
    }

    #[test]
    fn test_gts_gtu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 21;
        let y_value: u32 = 36;
        let z_value: u32 = 0;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32GtS,
            //  Opcode::I32GtS,
            Opcode::I32GtU,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_ges_geu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 1;
        let y_value: u32 = 233;
        let z_value: u32 = 36;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32GeS,
            Opcode::I32GeU,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn test_les_leu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0;
        let y_value: u32 = 3;
        let z_value: u32 = 9;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32LeS, // check whether signed x_value is less than or equal to signed y_value
            Opcode::I32LeU, /* check whether unsigned x_value is less than or equal to unsigned
                             * y_value */
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_divs_divu() {
        use std::collections::HashMap;

        // Helper: mirror the semantics used in the DivRem chip tests.
        fn compute_expected(opcode: Opcode, b: u32, c: u32) -> u32 {
            match opcode {
                Opcode::I32DivU => {
                    if c == 0 {
                        0
                    } else {
                        b / c
                    }
                }
                Opcode::I32RemU => {
                    if c == 0 {
                        b
                    } else {
                        b % c
                    }
                }
                Opcode::I32DivS => {
                    if c == 0 {
                        0
                    } else {
                        (b as i32).wrapping_div(c as i32) as u32
                    }
                }
                Opcode::I32RemS => {
                    if c == 0 {
                        b
                    } else {
                        (b as i32).wrapping_rem(c as i32) as u32
                    }
                }
                _ => panic!("unexpected opcode in div/rem test: {:?}", opcode),
            }
        }

        let sp_value: u32 = SP_START;

        // Same edge-case set as the DivRemChip proof test.
        let instructions: &[(u32, u32)] = &[
            // --- Basic Identity ---
            (0, 1),   // 0 / 1 = 0
            (1, 1),   // 1 / 1 = 1
            (50, 50), // x / x = 1
            // --- Basic Arithmetic / remainder sanity ---
            (100, 3), // 100 / 3, remainder 1
            (1, 2),   // 1 / 2: quotient 0, remainder 1
            // --- Unsigned Boundaries ---
            (u32::MAX, 1),            // Max / 1
            (u32::MAX, u32::MAX),     // Max / Max
            (u32::MAX, 2),            // Large / Small
            (u32::MAX, u32::MAX - 1), // Max / (Max-1)
            (1, u32::MAX),            // Small / Large
            // --- Signed Boundaries (Two's Complement) ---
            (i32::MIN as u32, 1),               // INT_MIN / 1
            (i32::MAX as u32, 1),               // INT_MAX / 1
            (i32::MIN as u32, i32::MIN as u32), // INT_MIN / INT_MIN
            // --- Signed Overflow-style Case (wrapping semantics) ---
            //(i32::MIN as u32, u32::MAX), // INT_MIN / -1

            // --- Signed Negative Operand Combos ---
            ((-5i32) as u32, 2),              // -5 / 2
            (5, (-2i32) as u32),              // 5 / -2
            ((-5i32) as u32, (-2i32) as u32), // -5 / -2
            // --- Division by Zero Cases ---
            //   (100, 0),
            //   (0, 0),
            //   (u32::MAX, 0),

            // Extra mixed sign edge cases
            //   (0x80000000u32, -1i32 as u32),
            (-1i32 as u32, 0x80000000u32),
        ];

        let opcodes = [Opcode::I32DivU, Opcode::I32DivS, Opcode::I32RemU, Opcode::I32RemS];

        for &(b, c) in instructions {
            for &op in &opcodes {
                println!("next test: {:?} with b = {:#010x}, c = {:#010x}", op, b, c);
                // For each (b, c, op) we run a tiny program: push b, push c, apply op.
                let program = Program::from_instrs(vec![
                    Opcode::I32Const(b.into()),
                    Opcode::I32Const(c.into()),
                    op,
                ]);

                let mut runtime = Executor::new(program, SP1CoreOpts::default());
                runtime.run().unwrap();

                // Top-of-stack is at runtime.state.sp (stack grows down).
                let top =
                    runtime.state.memory.get(runtime.state.sp).expect("stack top must exist").value;

                let expected = compute_expected(op, b, c);

                assert_eq!(
                    top, expected,
                    "div/rem result mismatch for {:?} with b = {:#010x}, c = {:#010x}",
                    op, b, c
                );

                // Stack pointer should have moved by exactly one 32-bit word:
                // initial SP = SP_START, final SP = SP_START - 4  =>  SP_START == sp + 4
                assert_eq!(
                    sp_value,
                    runtime.state.sp + 4,
                    "unexpected SP movement for {:?} with b = {:#010x}, c = {:#010x}",
                    op,
                    b,
                    c
                );
            }
        }
    }
    #[test]
    fn test_rems_remu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 320;
        let y_value: u32 = 13;
        let z_value: u32 = 5;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32RemS,
            Opcode::I32RemU,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value % (y_value % z_value)
        );
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn test_shl() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 2;
        let y_value: u32 = 2;
        let z_value: u32 = 3;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Shl, // y_value is shifted left by z_value
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value << (y_value << z_value)
        );
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn test_shr_shru() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 256;
        let y_value: u32 = 2;
        let z_value: u32 = 3;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32ShrS, // y_value is shifted right by z_value
            Opcode::I32ShrU, //
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value >> (y_value >> z_value)
        );
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    fn simple_opcode_test(opcode: Opcode, expected: u32, a: u32, b: u32) {
        let sp_value: u32 = SP_START;
        let x_value: u32 = a;
        let y_value: u32 = b;
        let opcodes =
            vec![Opcode::I32Const(x_value.into()), Opcode::I32Const(y_value.into()), opcode];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        println!("opxxx:{}", opcode);
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, expected);
    }
    fn simple_opcode_test_expect_error(opcode: Opcode, expected: u32, a: u32, b: u32) {
        let sp_value: u32 = SP_START;
        let x_value: u32 = a;
        let y_value: u32 = b;
        let opcodes =
            vec![Opcode::I32Const(x_value.into()), Opcode::I32Const(y_value.into()), opcode];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        let error = runtime.run().expect_err("expected error");
        println!("expected execution error :{:?}", error);
    }
    #[test]
    #[allow(clippy::unreadable_literal)]
    fn multiplication_tests() {
        simple_opcode_test(Opcode::I32Mul, 0x00001200, 0x00007e00, 0xb6db6db7);
        simple_opcode_test(Opcode::I32Mul, 0x00001240, 0x00007fc0, 0xb6db6db7);
        simple_opcode_test(Opcode::I32Mul, 0x00000000, 0x00000000, 0x00000000);
        simple_opcode_test(Opcode::I32Mul, 0x00000001, 0x00000001, 0x00000001);
        simple_opcode_test(Opcode::I32Mul, 0x00000015, 0x00000003, 0x00000007);
        simple_opcode_test(Opcode::I32Mul, 0x00000000, 0x00000000, 0xffff8000);
        simple_opcode_test(Opcode::I32Mul, 0x00000000, 0x80000000, 0x00000000);
        simple_opcode_test(Opcode::I32Mul, 0x00000000, 0x80000000, 0xffff8000);
        simple_opcode_test(Opcode::I32Mul, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d);
        simple_opcode_test(Opcode::I32Mul, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab);
        simple_opcode_test(Opcode::I32Mul, 0x00000000, 0xff000000, 0xff000000);
        simple_opcode_test(Opcode::I32Mul, 0x00000001, 0xffffffff, 0xffffffff);
        simple_opcode_test(Opcode::I32Mul, 0xffffffff, 0xffffffff, 0x00000001);
        simple_opcode_test(Opcode::I32Mul, 0xffffffff, 0x00000001, 0xffffffff);
    }
    #[test]
    fn division_tests() {
        simple_opcode_test(Opcode::I32DivU, 3, 20, 6);
        simple_opcode_test(Opcode::I32DivU, 715_827_879, u32::MAX - 20 + 1, 6);
        simple_opcode_test(Opcode::I32DivU, 0, 20, u32::MAX - 6 + 1);
        simple_opcode_test(Opcode::I32DivU, 0, u32::MAX - 20 + 1, u32::MAX - 6 + 1);

        simple_opcode_test(Opcode::I32DivU, 1 << 31, 1 << 31, 1);
        simple_opcode_test(Opcode::I32DivU, 0, 1 << 31, u32::MAX - 1 + 1);

        //divide by zero case
        simple_opcode_test_expect_error(Opcode::I32DivU, u32::MAX, 1 << 31, 0);
        simple_opcode_test_expect_error(Opcode::I32DivU, u32::MAX, 1, 0);
        simple_opcode_test_expect_error(Opcode::I32DivU, u32::MAX, 0, 0);
        simple_opcode_test_expect_error(Opcode::I32DivS, neg(1), 0, 0);

        simple_opcode_test(Opcode::I32DivS, 3, 18, 6);
        simple_opcode_test(Opcode::I32DivS, neg(6), neg(24), 4);
        simple_opcode_test(Opcode::I32DivS, neg(2), 16, neg(8));

        //TODO check: Overflow cases
        /*
        (i32.rem_s x y)
         If y == 0: trap
         If x == INT_MIN and y == -1: the result is 0 (not a trap — division would trap, but rem doesn’t)
         Otherwise: x % y with signed behavior
        */
        simple_opcode_test_expect_error(Opcode::I32DivS, 1 << 31, 1 << 31, neg(1));
        simple_opcode_test(Opcode::I32RemS, 0, 1 << 31, neg(1));
    }
    #[test]
    fn remainder_tests() {
        simple_opcode_test(Opcode::I32RemS, 7, 16, 9);
        simple_opcode_test(Opcode::I32RemS, neg(4), neg(22), 6);
        simple_opcode_test(Opcode::I32RemS, 1, 25, neg(3));
        simple_opcode_test(Opcode::I32RemS, neg(2), neg(22), neg(4));
        simple_opcode_test(Opcode::I32RemS, 0, 873, 1);
        simple_opcode_test(Opcode::I32RemS, 0, 873, neg(1));
        //error: mod by zero
        simple_opcode_test_expect_error(Opcode::I32RemS, 5, 5, 0);
        simple_opcode_test_expect_error(Opcode::I32RemS, neg(5), neg(5), 0);
        simple_opcode_test_expect_error(Opcode::I32RemS, 0, 0, 0);

        simple_opcode_test(Opcode::I32RemU, 4, 18, 7);
        simple_opcode_test(Opcode::I32RemU, 6, neg(20), 11);
        simple_opcode_test(Opcode::I32RemU, 23, 23, neg(6));
        simple_opcode_test(Opcode::I32RemU, neg(21), neg(21), neg(11));

        //error: mod by zero
        simple_opcode_test_expect_error(Opcode::I32RemU, 5, 5, 0);
        simple_opcode_test_expect_error(Opcode::I32RemU, neg(1), neg(1), 0);
        simple_opcode_test_expect_error(Opcode::I32RemU, 0, 0, 0);
    }
    #[test]
    #[allow(clippy::unreadable_literal)]
    fn shift_tests() {
        simple_opcode_test(Opcode::I32Shl, 0x00000001, 0x00000001, 0);
        simple_opcode_test(Opcode::I32Shl, 0x00000002, 0x00000001, 1);
        simple_opcode_test(Opcode::I32Shl, 0x00000080, 0x00000001, 7);
        simple_opcode_test(Opcode::I32Shl, 0x00004000, 0x00000001, 14);
        simple_opcode_test(Opcode::I32Shl, 0x80000000, 0x00000001, 31);
        simple_opcode_test(Opcode::I32Shl, 0xffffffff, 0xffffffff, 0);
        simple_opcode_test(Opcode::I32Shl, 0xfffffffe, 0xffffffff, 1);
        simple_opcode_test(Opcode::I32Shl, 0xffffff80, 0xffffffff, 7);
        simple_opcode_test(Opcode::I32Shl, 0xffffc000, 0xffffffff, 14);
        simple_opcode_test(Opcode::I32Shl, 0x80000000, 0xffffffff, 31);
        simple_opcode_test(Opcode::I32Shl, 0x21212121, 0x21212121, 0);
        simple_opcode_test(Opcode::I32Shl, 0x42424242, 0x21212121, 1);
        simple_opcode_test(Opcode::I32Shl, 0x90909080, 0x21212121, 7);
        simple_opcode_test(Opcode::I32Shl, 0x48484000, 0x21212121, 14);
        simple_opcode_test(Opcode::I32Shl, 0x80000000, 0x21212121, 31);
        simple_opcode_test(Opcode::I32Shl, 0x21212121, 0x21212121, 0xffffffe0);
        simple_opcode_test(Opcode::I32Shl, 0x42424242, 0x21212121, 0xffffffe1);
        simple_opcode_test(Opcode::I32Shl, 0x90909080, 0x21212121, 0xffffffe7);
        simple_opcode_test(Opcode::I32Shl, 0x48484000, 0x21212121, 0xffffffee);
        simple_opcode_test(Opcode::I32Shl, 0x00000000, 0x21212120, 0xffffffff);

        simple_opcode_test(Opcode::I32ShrU, 0xffff8000, 0xffff8000, 0);
        simple_opcode_test(Opcode::I32ShrU, 0x7fffc000, 0xffff8000, 1);
        simple_opcode_test(Opcode::I32ShrU, 0x01ffff00, 0xffff8000, 7);
        simple_opcode_test(Opcode::I32ShrU, 0x0003fffe, 0xffff8000, 14);
        simple_opcode_test(Opcode::I32ShrU, 0x0001ffff, 0xffff8001, 15);
        simple_opcode_test(Opcode::I32ShrU, 0xffffffff, 0xffffffff, 0);
        simple_opcode_test(Opcode::I32ShrU, 0x7fffffff, 0xffffffff, 1);
        simple_opcode_test(Opcode::I32ShrU, 0x01ffffff, 0xffffffff, 7);
        simple_opcode_test(Opcode::I32ShrU, 0x0003ffff, 0xffffffff, 14);
        simple_opcode_test(Opcode::I32ShrU, 0x00000001, 0xffffffff, 31);
        simple_opcode_test(Opcode::I32ShrU, 0x21212121, 0x21212121, 0);
        simple_opcode_test(Opcode::I32ShrU, 0x10909090, 0x21212121, 1);
        simple_opcode_test(Opcode::I32ShrU, 0x00424242, 0x21212121, 7);
        simple_opcode_test(Opcode::I32ShrU, 0x00008484, 0x21212121, 14);
        simple_opcode_test(Opcode::I32ShrU, 0x00000000, 0x21212121, 31);
        simple_opcode_test(Opcode::I32ShrU, 0x21212121, 0x21212121, 0xffffffe0);
        simple_opcode_test(Opcode::I32ShrU, 0x10909090, 0x21212121, 0xffffffe1);
        simple_opcode_test(Opcode::I32ShrU, 0x00424242, 0x21212121, 0xffffffe7);
        simple_opcode_test(Opcode::I32ShrU, 0x00008484, 0x21212121, 0xffffffee);
        simple_opcode_test(Opcode::I32ShrU, 0x00000000, 0x21212121, 0xffffffff);

        simple_opcode_test(Opcode::I32ShrS, 0x00000000, 0x00000000, 0);
        simple_opcode_test(Opcode::I32ShrS, 0xc0000000, 0x80000000, 1);
        simple_opcode_test(Opcode::I32ShrS, 0xff000000, 0x80000000, 7);
        simple_opcode_test(Opcode::I32ShrS, 0xfffe0000, 0x80000000, 14);
        simple_opcode_test(Opcode::I32ShrS, 0xffffffff, 0x80000001, 31);
        simple_opcode_test(Opcode::I32ShrS, 0x7fffffff, 0x7fffffff, 0);
        simple_opcode_test(Opcode::I32ShrS, 0x3fffffff, 0x7fffffff, 1);
        simple_opcode_test(Opcode::I32ShrS, 0x00ffffff, 0x7fffffff, 7);
        simple_opcode_test(Opcode::I32ShrS, 0x0001ffff, 0x7fffffff, 14);
        simple_opcode_test(Opcode::I32ShrS, 0x00000000, 0x7fffffff, 31);
        simple_opcode_test(Opcode::I32ShrS, 0x81818181, 0x81818181, 0);
        simple_opcode_test(Opcode::I32ShrS, 0xc0c0c0c0, 0x81818181, 1);
        simple_opcode_test(Opcode::I32ShrS, 0xff030303, 0x81818181, 7);
        simple_opcode_test(Opcode::I32ShrS, 0xfffe0606, 0x81818181, 14);
        simple_opcode_test(Opcode::I32ShrS, 0xffffffff, 0x81818181, 31);
    }
    fn neg(a: u32) -> u32 {
        u32::MAX - a + 1
    }

    #[test]
    fn test_store() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime
                .state
                .memory
                .get(TypedAddress::GlobalMemory(addr).to_virtual_addr())
                .unwrap()
                .value,
            x_value
        );
        assert_eq!(sp_value, runtime.state.sp + UNIT);
    }

    #[test]
    fn test_store16() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_0005;
        let y_value: u32 = 0xFFFF_0008;
        let y_actually: u32 = (y_value & 0x0000_FFFF) << 16;

        let addr: u32 = 0x10000;

        //discuss why Opcode::I32Store16(0.into()),Opcode::I32Store16(1.into()) are not working if
        // they are subsequent
        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store16(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Store16(2u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let v_addr = TypedAddress::GlobalMemory(addr).to_virtual_addr();
        println!("v_addr");
        println!("stack val:{:x}", runtime.state.memory.get(v_addr).unwrap().value);
        assert_eq!(
            runtime.state.memory.get(v_addr).unwrap().value,
            (x_value & 0x0000_FFFF) + ((y_value & 0x0000_FFFF) << 16)
        );
        assert_eq!(sp_value, runtime.state.sp + UNIT);
    }

    #[test]
    fn test_store8() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_0001;
        let y_value: u32 = 0xFFFF_0002;
        let z_value: u32 = 0xFFFF_0003;
        let t_value: u32 = 0xFFFF_0004;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store8(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Store8(1u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Store8(2u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(t_value.into()),
            Opcode::I32Store8(3u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let v_addr = TypedAddress::GlobalMemory(addr).to_virtual_addr();
        assert_eq!(
            runtime.state.memory.get(v_addr).unwrap().value,
            ((x_value & 0x0000_00FF) +
                ((y_value & 0x0000_00FF) << 8) +
                ((z_value & 0x0000_00FF) << 16) +
                ((t_value & 0x0000_00FF) << 24))
        );
        assert_eq!(sp_value, runtime.state.sp + UNIT);
    }

    fn simple_memory_load_opcode_test(mem: HashMap<u32, u32>, opcodes: Vec<Opcode>, expected: u32) {
        let program = Program::from_instrs(opcodes);

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, expected);
    }

    #[test]
    fn test_simple_memory_opcode() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFF1_0005;
        let y_value: u32 = 0xFFF2_0008;
        let z_value: u32 = 0xFFF3_000A;
        let t_value: u32 = 0xFFF4_000B;
        let addr: u32 = 0xDD_0000;

        let mem_opcodes = [
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(t_value.into()),
        ];

        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store(0.into())],
        //     addr + 0,
        //     t_value,
        // );
        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store(16.into())],
        //     addr + 16,
        //     t_value,
        // );

        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store16(0.into())],
        //     addr,
        //     (t_value & 0x0000_FFFF),
        // );
        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store16(2.into())],
        //     addr,
        //     (t_value & 0x0000_FFFF) << 16,
        // );

        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store8(0.into())],
        //     addr,
        //     (t_value & 0x0000_00FF),
        // );
        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store8(1.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 8,
        // );
        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store8(2.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 16,
        // );
        // simple_memory_store_opcode_test(
        //     mem_opcodes.clone(),
        //     vec![Opcode::I32Store8(3.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 24,
        // );

        /*simple_memory_store_opcode_test(
            mem_opcodes.clone(),
            vec![Opcode::I32Store16(0.into()), Opcode::I32Store16(1.into())],
            addr,
            (z_value & 0x0000_FFFF) + ((t_value & 0x0000_FFFF) << 16),
        );

          simple_memory_store_opcode_test(
              mem_opcodes.clone(),
              vec![
                  Opcode::I32Store8(0.into()),
                  Opcode::I32Store8(1.into()),
                  Opcode::I32Store8(2.into()),
                  Opcode::I32Store8(3.into()),
              ],
              addr,
              (x_value & 0x0000_00FF)
                  + ((y_value & 0x0000_00FF) << 8)
                  + ((z_value & 0x0000_00FF) << 16)
                  + ((t_value & 0x0000_00FF) << 24),
          );*/

        /*mem.insert(addr, x_value);
        mem.insert(addr + 4, x_value + 4);
        mem.insert(addr + 8, x_value + 8);
        mem.insert(addr + 12, x_value + 12);
        mem.insert(addr + 160, x_value + 16);

        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load(0.into())],
            x_value,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load(4.into())],
            x_value + 4,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load(8.into())],
            x_value + 8,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load(12.into())],
            x_value + 12,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load(160.into())],
            x_value + 16,
        );

        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load16S(0.into())],
            x_value & 0x0000_FFFF,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load16U(0.into())],
            x_value & 0x0000_FFFF,
        );

        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load16U(2.into())],
            (x_value & 0xFFFF_0000) >> 16,
        );
        let value = (x_value & 0xFFFF_0000) >> 16;
        let expected = ((value as i16) as i32) as u32;
        println!("expected u32 {} : actual i16 {}", expected, (value as i16));
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load16S(2.into())],
            ((value as i16) as i32) as u32,
        );

        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8U(0.into())],
            (x_value & 0x0000_00FF),
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8U(1.into())],
            (x_value & 0x0000_FF) >> 8,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8U(2.into())],
            (x_value & 0x00FF_FFFF) >> 16,
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8U(3.into())],
            (x_value & 0xFF00_FFFF) >> 24,
        );

        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8S(0.into())],
            (x_value & 0x0000_00FF),
        );
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8S(1.into())],
            (x_value & 0x0000_FF) >> 8,
        );

        let value = (x_value & 0x00FF_FFFF) >> 16;
        let expected = ((value as i8) as i32) as u32;
        println!("expected u32 {} : actual i8 {}", expected, (value as i8));
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8S(2.into())],
            expected,
        );

        let value = (x_value & 0xFF00_FFFF) >> 24;
        let expected = ((value as i8) as i32) as u32;
        println!("expected u32 {} : actual i8 {}", expected, (value as i8));
        simple_memory_load_opcode_test(
            mem.clone(),
            vec![Opcode::I32Load8S(3.into())],
            expected,
        );*/
    }

    #[test]
    fn test_load32() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 5;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(0u32),
        ];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value);
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }

    #[test]
    fn test_store_unaligned() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x0103_0507;

        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(3u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(3u32),
        ];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        peek_stack(&runtime);
        println!("stack pointer: {}", runtime.state.sp);

        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value);
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }

    #[test]
    fn test_load_unaligned() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1103_0507;

        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(3u32),
        ];
        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 0x11);
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }
    #[test]
    fn test_load16u() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_0005;
        let addr: u32 = 0x10000;

        //work on order
        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store16(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16U(0u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value & 0x0000_FFFF
        );
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }
    #[test]
    fn test_load16s_normal() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 65551i32 as u32;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16S(0u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value & 0x0000_ffff
        );
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }
    #[test]
    fn test_load16s() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = (-5i16) as i32 as u32;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store16(0u32), //I32Store16S
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16U(0u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value & 0x0000_ffff
        );
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }

    #[test]
    fn test_load8u() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_05FF;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load8U(1u32),
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());

        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            (x_value & 0x0000_FF00) >> 8
        );
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }

    #[test]
    fn test_load8s() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xDFFF_00FF;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load8S(3u32), //I32Load8S
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value as i8,
            ((x_value & 0xff00_0000) >> 24) as i8
        );
        assert_eq!(sp_value, runtime.state.sp + 2 * UNIT);
    }

    #[test]
    fn test_br() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Shl,
            Opcode::I32Const(x_value.into()),
            Opcode::I32Shl,
            Opcode::I32Const(x_value.into()),
            Opcode::Br(1.into()),
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());

        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, ((1 << 1) << 1) << 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }

    #[test]
    fn build_elf_branching() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const((x_value + 1).into()),
            Opcode::I32Const((x_value + 2).into()),
            Opcode::I32Add,
            Opcode::I32Add,
            Opcode::I32Const((5).into()),
            Opcode::BrIfNez(BranchOffset::from(16i32)),
            Opcode::I32Const((x_value + 3).into()),
            Opcode::I32Const((x_value + 4).into()),
            Opcode::I32Add,
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();

        println!("initial.sp {} , state.sp {}", sp_value, runtime.state.sp);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 6);
    }

    #[test]
    fn test_local_get() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;

        let opcodes = vec![
            Opcode::I32Const((x_value + 5).into()),
            Opcode::I32Const((x_value + 4).into()),
            Opcode::I32Const((x_value + 3).into()),
            Opcode::I32Const((x_value + 2).into()),
            Opcode::I32Const((x_value + 1).into()),
            Opcode::I32Const((x_value).into()),
            Opcode::LocalGet(6u32),
            Opcode::LocalGet(6u32),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        peek_stack(&runtime);
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value + 5 + x_value + 4
        );
        assert_eq!(sp_value, runtime.state.sp + 7 * 4);
    }

    #[test]
    fn test_local_set() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 80;

        let opcodes = vec![
            Opcode::I32Const((x_value + 1).into()),
            Opcode::I32Const((x_value + 2).into()),
            Opcode::I32Const((x_value + 3).into()),
            Opcode::I32Const((x_value + 4).into()),
            Opcode::I32Const((x_value + 1).into()),
            Opcode::I32Const((x_value + 2).into()),
            Opcode::I32Const((x_value + 3).into()),
            Opcode::I32Const((x_value + 4).into()),
            Opcode::I32Const((x_value + 5).into()),
            Opcode::I32Const((x_value + 6).into()),
            Opcode::I32Const((x_value).into()),
            Opcode::LocalSet(3u32),
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        peek_stack(&runtime);
        println!("after sp: {}", runtime.state.sp);
        println!("after pos: {}", (SP_START - runtime.state.sp) / 4);
        assert_eq!(runtime.state.memory.get(runtime.state.sp + 2 * UNIT).unwrap().value, x_value);
    }
    #[test]
    fn test_locals() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 1;
        let y_value: u32 = 22;
        let z_value: u32 = 6;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::LocalGet(2u32),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        println!("before {}", runtime.state.sp);
        runtime.run().unwrap();
        println!("after {}", runtime.state.sp);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, y_value + z_value);
    }

    #[test]
    fn test_local_tee() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const((x_value + 2).into()),
            Opcode::I32Const((x_value + 3).into()),
            Opcode::I32Const((x_value + 4).into()),
            Opcode::I32Const((x_value + 7).into()),
            Opcode::LocalTee(4u32), /* get last element and put it into address (where address =
                                     * last sp + 16) */
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.sp, sp_value - 20);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value + 7);
    }

    #[test]
    fn test_i32const() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;
        let opcodes = vec![Opcode::I32Const(x_value.into())];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.sp, sp_value - 4);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value);
    }

    #[test]
    fn test_call_internal_and_return() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x7;
        let y_value: u32 = 0x2;
        let z_value: u32 = 0x1;
        let functions = [0, 24];

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::CallInternal(6u32),
            Opcode::I32Sub,
            Opcode::Return,
            Opcode::I32Add,
            Opcode::Return,
        ];

        let program = Program::from_instrs(opcodes);
        //  memory_image: BTreeMap::new() };
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value - (y_value + z_value)
        );
    }

    #[test]
    fn test_call_indirect() {
        let ops = vec![
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableGrow(0),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableInit(0),
            Opcode::TableGet(0),
            Opcode::I32Const(1.into()),
            Opcode::CallIndirect(0u32),
            Opcode::TableGet(0),
            Opcode::Return,
            Opcode::I32Const(99.into()),
            Opcode::I32Const(98.into()),
            Opcode::I32Add,
            Opcode::Return,
        ];
        let elements = vec![12u32, 12u32];
        let program = Program::from_instrs(ops).with_elements(elements);

        let mut rt = Executor::new(program, SP1CoreOpts::default());

        rt.run().unwrap();
    }
    #[test]
    fn test_i32constwith_add() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;
        let y_value: u32 = 0x54321;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.sp, sp_value - 4);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value + y_value);
    }

    #[test]
    fn test_runstate() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;
        let y_value: u32 = 0x54321;

        let opcodes = vec![Opcode::I32Const(x_value.into())];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.execute().unwrap();
        println!("record:{:?}", runtime.record);
        println!("records:{:?}", runtime.records);
        assert_eq!(runtime.state.sp, sp_value - 4);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value);
    }
    #[test]
    fn test_call_chain_incrementers() {
        // inc(x) = x + 1
        let inc_fn = vec![
            Opcode::I32Const(1u32.into()),
            Opcode::I32Add,
            Opcode::Return, // function returns here
        ];

        // Main: x -> inc -> inc -> (== expected) -> Return
        // Count main ops carefully and include the final Return to avoid fall-through.
        // main ops: I32Const(x), CallInternal, CallInternal, I32Const(expected), I32Eq, Return  =>
        // 6
        let main_len = 6;
        let inc_pos = main_len as u32; // function starts right after main

        let x = 41u32;
        let expected = x + 2;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::CallInternal(inc_pos)); // x+1
        ops.push(Opcode::CallInternal(inc_pos)); // (x+1)+1
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq); // 1 if equal
        ops.push(Opcode::Return); // <-- prevent fall-through into inc_fn

        // append function
        ops.extend(inc_fn);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    #[test]
    fn test_store_then_load_via_function() {
        let addr: u32 = 0x10000; // inside 2 pages (0..=0x1FFFF), 32-bit store is safe
        let val: u32 = 0xDEAD_BEEF;

        // function expects: [ ..., addr, addr, value ]
        // does: store(addr, value); load(addr); return
        let fun = vec![Opcode::I32Store(0u32), Opcode::I32Load(0u32), Opcode::Return];

        // main ops:
        //   MemoryGrow(2)
        //   push addr, addr, val
        //   CallInternal(fun_pos)
        //   == val
        //   Return   <-- prevent fall-through into 'fun'
        let main_len = 2 /*grow*/ + 3 /*pushes*/ + 1 /*call*/ + 2 /*eq*/ + 1 /*ret*/;
        let fun_pos = main_len as u32;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(2.into()));
        ops.push(Opcode::MemoryGrow);

        // push in order: addr, addr, value (value must be on top for store)
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(val.into()));

        ops.push(Opcode::CallInternal(fun_pos));

        // compare with expected
        ops.push(Opcode::I32Const(val.into()));
        ops.push(Opcode::I32Eq);

        // stop main; don't fall through into the function body
        ops.push(Opcode::Return);

        // append function
        ops.extend(fun);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);

        // sanity: at least one mem event and one call event
        let mem_events: usize = rt.records.iter().map(|r| r.memory_instr_events.len()).sum();
        let call_events: usize = rt.records.iter().map(|r| r.call_events.len()).sum();
        assert!(mem_events >= 2);
        assert!(call_events >= 1);
    }
    // --- call mix: xor + and + final combine through calls ---
    #[test]
    fn test_call_mix_bitwise_and_arith() {
        let xor_fn = vec![Opcode::I32Xor, Opcode::Return]; // len 2
        let and_fn = vec![Opcode::I32And, Opcode::Return]; // len 2

        let a = 0xAAAA5555u32;
        let b = 0x0F0F0F0Fu32;
        let c = 0xF0F0FF00u32;
        let d = 0x00FF00FFu32;
        let expected = (a ^ b).wrapping_add(c & d);

        // main has:
        //   a,b,CallInternal(xor)            -> 3
        //   c,d,CallInternal(and)            -> 3
        //   I32Add                           -> 1
        //   I32Const(expected), I32Eq        -> 2
        //   Return                           -> 1
        // total = 10
        let main_len = 10u32;
        let xor_pos = main_len; // function 1 starts right after main
        let and_pos = xor_pos + xor_fn.len() as u32; // function 2 follows

        let mut ops = Vec::new();
        // (a ^ b) via xor_fn
        ops.push(Opcode::I32Const(a.into()));
        ops.push(Opcode::I32Const(b.into()));
        ops.push(Opcode::CallInternal(xor_pos));

        // (c & d) via and_fn
        ops.push(Opcode::I32Const(c.into()));
        ops.push(Opcode::I32Const(d.into()));
        ops.push(Opcode::CallInternal(and_pos));

        // sum them
        ops.push(Opcode::I32Add);

        // compare with expected
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);

        // IMPORTANT: prevent fall-through into function code
        ops.push(Opcode::Return);

        // append functions
        ops.extend(xor_fn);
        ops.extend(and_fn);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    #[test]
    fn test_stack_after_nested_calls() {
        // inc(z) = z + 1
        let inc_fn = vec![Opcode::I32Const(1u32.into()), Opcode::I32Add, Opcode::Return]; // len = 3

        // f(x,y) = (x + y) + 1  ==  I32Add ; CallInternal(inc_pos) ; Return
        // We'll compute inc_pos after setting main_len.
        // Placeholder only to get its length (3):
        let f_body_len = 3;

        // main has: I32Const(x), I32Const(y), CallInternal(f_pos), I32Const(expected), I32Eq,
        // Return
        let main_len = 6u32;

        // final layout: [ main | f | inc ]
        let f_pos = main_len;
        let inc_pos = f_pos + f_body_len as u32;

        // build f with the correct inc_pos
        let f = vec![Opcode::I32Add, Opcode::CallInternal(inc_pos), Opcode::Return];

        let x = 10u32;
        let y = 31u32;
        let expected = x + y + 1;

        // build main
        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::CallInternal(f_pos));
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return); // <-- prevent fall-through into f

        // append functions
        ops.extend(f);
        ops.extend(inc_fn);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        let sp0 = rt.state.sp;
        rt.run().unwrap();

        // one boolean result left on stack
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
        assert_eq!(rt.state.sp, sp0 - 4);
    }

    // --- multiple calls with memory byte assembly inside a function ---
    #[test]
    fn test_call_build_u32_from_bytes() {
        // build32(addr, b0,b1,b2,b3): store8 at offsets 0..3, then load32
        let fun = vec![
            Opcode::I32Store8(0u32), // pops (value, addr)
            Opcode::I32Store8(1u32),
            Opcode::I32Store8(2u32),
            Opcode::I32Store8(3u32),
            Opcode::I32Load(0u32), // pops addr, pushes word
            Opcode::Return,
        ];

        // main = grow(4), push 9 items (addr sequencing below), call, cmp, return
        // total main_len = 2 (grow) + 9 (pushes) + 1 (call) + 2 (cmp) + 1 (ret) = 15
        let main_len = 15u32;
        let fun_pos = main_len;

        let addr: u32 = 0x30000; // needs 4 pages (4 * 64KiB = 262,144 bytes)
        let b0 = 0x11u32;
        let b1 = 0x22u32;
        let b2 = 0x33u32;
        let b3 = 0x44u32;
        let expected = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);

        let mut ops = Vec::new();
        // Allocate 4 pages
        ops.push(Opcode::I32Const(4.into()));
        ops.push(Opcode::MemoryGrow);

        // Stack on function entry must be (top right):
        // [ addr(load), addr(b3), b3, addr(b2), b2, addr(b1), b1, addr(b0), b0 ]
        // so each store8 pops value then addr in order b0, b1, b2, b3; and one addr remains for
        // load.
        ops.push(Opcode::I32Const(addr.into())); // for final load (bottom-most)
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(b3.into()));
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(b2.into()));
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(b1.into()));
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(b0.into()));

        // Call the function
        ops.push(Opcode::CallInternal(fun_pos));

        // Compare with expected, then return to avoid fall-through into function body
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // Append function
        ops.extend(fun);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- chain calls: (((x+1)+1)*2) then >> 1 => x+2 (sanity of order) ---
    #[test]
    fn test_chain_calls_and_shifts() {
        let inc = vec![Opcode::I32Const(1u32.into()), Opcode::I32Add, Opcode::Return]; // len 3

        let times2 = vec![Opcode::I32Const(1u32.into()), Opcode::I32Shl, Opcode::Return]; // len 3

        // main: const x, call inc, call inc, call times2, const 1, shrU, const expected, eq, return
        // => 9
        let main_len = 9u32;

        let inc1_pos = main_len;
        let inc2_pos = inc1_pos + inc.len() as u32;
        let times2_pos = inc2_pos + inc.len() as u32;

        let x = 100u32;
        let expected = x + 2;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into())); // x
        ops.push(Opcode::CallInternal(inc1_pos)); // x+1
        ops.push(Opcode::CallInternal(inc2_pos)); // x+2
        ops.push(Opcode::CallInternal(times2_pos)); // (x+2)*2
        ops.push(Opcode::I32Const(1u32.into())); // shift by 1
        ops.push(Opcode::I32ShrU); // ((x+2)*2) >> 1 == x+2
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return); // prevent fall-through

        // append functions
        ops.extend(inc.clone());
        ops.extend(inc);
        ops.extend(times2);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- function that conditionally skips work with Br (simple jump) ---
    #[test]
    fn test_function_with_simple_br_skip() {
        use rwasm::BranchOffset;

        // f(x,y): (x+y); Br(2) to skip the next insn; Return
        let f = vec![
            Opcode::I32Add,                       // (x + y)
            Opcode::Br(BranchOffset::from(2i32)), // skip over I32Const(999) to Return
            Opcode::I32Const(999u32.into()),      // should be skipped
            Opcode::Return,
        ];

        // main: x,y; call f; const expected; eq; return  => 6 ops
        let main_len = 6u32;
        let f_pos = main_len;

        let x = 7u32;
        let y = 8u32;
        let expected = x + y;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::CallInternal(f_pos));
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return); // prevent fall-through

        ops.extend(f);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- many events sanity: multiple calls + mem ops + alu; check counts ---
    #[test]
    fn test_many_events_and_calls_sanity() {
        // helpers
        let add = vec![Opcode::I32Add, Opcode::Return]; // len 2
        let mul = vec![Opcode::I32Mul, Opcode::Return]; // len 2
        let shl1 = vec![Opcode::I32Const(1u32.into()), Opcode::I32Shl, Opcode::Return]; // len 3

        // Use 0x40000 => needs >= 5 pages (5*64KiB = 327,680)
        let addr: u32 = 0x40000;

        // main:
        // grow(5) |
        // push addrL, addrS |
        // (2,3, call add) |
        // (4,5, call add) |
        // call mul |
        // store v3 @ addrS |
        // load @ addrL |
        // call shl1 |
        // const 0 ; gtU |
        // return
        let main_len = 17u32;

        let add_pos = main_len;
        let mul_pos = add_pos + add.len() as u32;
        let shl_pos = mul_pos + mul.len() as u32;

        let mut ops = Vec::new();
        // grow
        ops.push(Opcode::I32Const(5.into()));
        ops.push(Opcode::MemoryGrow);

        // two copies of addr: one for load (left behind), one for store consumption
        ops.push(Opcode::I32Const(addr.into())); // addrL
        ops.push(Opcode::I32Const(addr.into())); // addrS

        // v1 = (2+3)
        ops.push(Opcode::I32Const(2u32.into()));
        ops.push(Opcode::I32Const(3u32.into()));
        ops.push(Opcode::CallInternal(add_pos));

        // v2 = (4+5)
        ops.push(Opcode::I32Const(4u32.into()));
        ops.push(Opcode::I32Const(5u32.into()));
        ops.push(Opcode::CallInternal(add_pos));

        // v3 = v1 * v2  (stack: [addrL, addrS, v3])
        ops.push(Opcode::CallInternal(mul_pos));

        // store: needs [ ..., addr, value ] with value on top -> OK: top is v3, below is addrS
        ops.push(Opcode::I32Store(0u32)); // pops v3, addrS; stack now [addrL]

        // load back from addrL
        ops.push(Opcode::I32Load(0u32)); // pops addrL, pushes v3

        // << 1 via helper
        ops.push(Opcode::CallInternal(shl_pos));

        // check > 0 (unsigned)
        ops.push(Opcode::I32Const(0u32.into()));
        ops.push(Opcode::I32GtU);

        // end main
        ops.push(Opcode::Return);

        // append functions
        ops.extend(add);
        ops.extend(mul);
        ops.extend(shl1);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);

        // sanity on events
        let calls: usize = rt.records.iter().map(|r| r.call_events.len()).sum();
        let alus: usize = rt
            .records
            .iter()
            .map(|r| {
                r.add_events.len() +
                    r.mul_events.len() +
                    r.bitwise_events.len() +
                    r.shift_left_events.len() +
                    r.shift_right_events.len()
            })
            .sum();
        let mems: usize = rt.records.iter().map(|r| r.memory_instr_events.len()).sum();
        assert!(calls >= 3); // add, add, mul, shl1 -> 4 actually
        assert!(alus >= 3); // add/mul/shl/gtu
        assert!(mems >= 2); // store + load
    }
    // --- Fibonacci n=9 via iterative step function and CallInternal ---
    #[test]
    fn test_fibonacci_n9_callinternal() {
        let base: u32 = 0x10000;
        let addr_tmp = base;
        let addr_a = base + 4; // will hold F(n)
        let addr_b = base + 8; // will hold F(n+1)
        let addr_n = base + 12;

        // step(): (a,b,n) -> (b, a+b, n-1); returns new n
        let step_fn = vec![
            // tmp = a + b
            Opcode::I32Const(addr_tmp.into()),
            Opcode::I32Const(addr_a.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Add,
            Opcode::I32Store(0u32),
            // a = b
            Opcode::I32Const(addr_a.into()),
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Store(0u32),
            // b = tmp
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Const(addr_tmp.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Store(0u32),
            // n = n - 1
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(1u32.into()),
            Opcode::I32Sub,
            Opcode::I32Store(0u32),
            // return n
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Load(0u32),
            Opcode::Return,
        ];

        let mut ops = Vec::new();
        // memory
        ops.push(Opcode::I32Const(2.into()));
        ops.push(Opcode::MemoryGrow);

        // a=0
        ops.push(Opcode::I32Const(addr_a.into()));
        ops.push(Opcode::I32Const(0u32.into()));
        ops.push(Opcode::I32Store(0u32));
        // b=1
        ops.push(Opcode::I32Const(addr_b.into()));
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::I32Store(0u32));
        // n=9
        ops.push(Opcode::I32Const(addr_n.into()));
        ops.push(Opcode::I32Const(9u32.into()));
        ops.push(Opcode::I32Store(0u32));

        // 9 iterations: call step, drop returned n
        let mut call_sites = Vec::<usize>::new();
        for _ in 0..9 {
            call_sites.push(ops.len());
            ops.push(Opcode::CallInternal(0u32)); // patched later
            ops.push(Opcode::Drop);
        }

        // compare a with 34 (F9)
        ops.push(Opcode::I32Const(addr_a.into()));
        ops.push(Opcode::I32Load(0u32));
        ops.push(Opcode::I32Const(34u32.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // patch function position
        let step_pos = ops.len() as u32;
        for idx in call_sites {
            ops[idx] = Opcode::CallInternal(step_pos);
        }
        ops.extend(step_fn);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- Nested calls: f1 -> f2 -> f3 (3 levels) ---
    #[test]
    fn test_nested_three_level_calls() {
        // f3(x): return 2*x
        let f3 = vec![Opcode::I32Const(1u32.into()), Opcode::I32Shl, Opcode::Return]; // len 3

        // f2(x,y): return f3(x) + y
        let f2 = vec![
            Opcode::CallInternal(0u32), // patched to f3_pos
            Opcode::I32Add,
            Opcode::Return,
        ]; // len 3

        // f1(x,y,z): return f2(x,y) + z
        let f1 = vec![
            Opcode::CallInternal(0u32), // patched to f2_pos
            Opcode::I32Add,
            Opcode::Return,
        ]; // len 3

        // main: push z, y, x (so top=x,y below,z bottom) ; call f1 ; cmp ; return
        let x = 3u32;
        let y = 5u32;
        let z = 7u32;
        let expected = (2 * x) + y + z; // 18

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(z.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::CallInternal(0u32)); // patch to f1_pos
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // compute positions
        let f1_pos = ops.len() as u32; // after main
        let mut f1_patched = f1.clone();
        let f2_pos = f1_pos + f1.len() as u32; // after f1
        let mut f2_patched = f2.clone();
        let f3_pos = f2_pos + f2.len() as u32; // after f2

        // patch call targets
        f1_patched[0] = Opcode::CallInternal(f2_pos);
        f2_patched[0] = Opcode::CallInternal(f3_pos);
        ops[3] = Opcode::CallInternal(f1_pos);

        // append functions
        ops.extend(f1_patched);
        ops.extend(f2_patched);
        ops.extend(f3);

        // run
        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // --- Direct call using Opcode::CallInternal (f_add) ---
    #[test]
    fn test_call_direct_add() {
        // f_add(x,y) = x + y
        let f_add = vec![Opcode::I32Add, Opcode::Return]; // len = 2

        // main: push x, y; Call(f_add); const expected; eq; return  => 6 ops
        let main_len = 6u32;
        let f_add_pos = main_len;

        let x = 12u32;
        let y = 30u32;
        let expected = x + y;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        // If your runtime expects a function index instead of a byte position,
        ops.push(Opcode::CallInternal(f_add_pos));
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return); // prevent fall-through

        // append the callee
        ops.extend(f_add);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    // --- Fibonacci n=25 via iterative step function and CallInternal ---
    #[test]
    fn test_fibonacci_n25_callinternal() {
        let base: u32 = 0x10000;
        let addr_tmp = base;
        let addr_a = base + 4; // F(n)
        let addr_b = base + 8; // F(n+1)
        let addr_n = base + 12;

        // step(): (a,b,n) -> (b, a+b, n-1); returns new n (ignored by caller)
        let step_fn = vec![
            // tmp = a + b
            Opcode::I32Const(addr_tmp.into()),
            Opcode::I32Const(addr_a.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Add,
            Opcode::I32Store(0u32),
            // a = b
            Opcode::I32Const(addr_a.into()),
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Store(0u32),
            // b = tmp
            Opcode::I32Const(addr_b.into()),
            Opcode::I32Const(addr_tmp.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Store(0u32),
            // n = n - 1
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(1u32.into()),
            Opcode::I32Sub,
            Opcode::I32Store(0u32),
            // return n
            Opcode::I32Const(addr_n.into()),
            Opcode::I32Load(0u32),
            Opcode::Return,
        ];

        let mut ops = Vec::new();
        // memory
        ops.push(Opcode::I32Const(2.into()));
        ops.push(Opcode::MemoryGrow);

        // init a=0, b=1, n=25
        ops.push(Opcode::I32Const(addr_a.into()));
        ops.push(Opcode::I32Const(0u32.into()));
        ops.push(Opcode::I32Store(0u32));

        ops.push(Opcode::I32Const(addr_b.into()));
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::I32Store(0u32));

        ops.push(Opcode::I32Const(addr_n.into()));
        ops.push(Opcode::I32Const(25u32.into()));
        ops.push(Opcode::I32Store(0u32));

        // 25 iterations of step(); drop returned n each time
        let mut call_sites = Vec::<usize>::new();
        for _ in 0..25 {
            call_sites.push(ops.len());
            ops.push(Opcode::CallInternal(0u32)); // patched later
            ops.push(Opcode::Drop);
        }

        // compare a with F25 = 75025, return
        ops.push(Opcode::I32Const(addr_a.into()));
        ops.push(Opcode::I32Load(0u32));
        ops.push(Opcode::I32Const(75025u32.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // patch function position and append
        let step_pos = ops.len() as u32;
        for idx in call_sites {
            ops[idx] = Opcode::CallInternal(step_pos);
        }
        ops.extend(step_fn);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    //Cross-page unaligned store/load with calls
    #[test]
    fn test_cross_page_unaligned_store_load_with_calls() {
        // helpers
        let add = vec![Opcode::I32Add, Opcode::Return]; // len 2
        let shl2 = vec![Opcode::I32Const(2u32.into()), Opcode::I32Shl, Opcode::Return]; // len 3

        let a = 15u32;
        let b = 27u32;
        let expected = (a + b) << 2;

        // choose aligned base, use unaligned offset = 3
        let base: u32 = 0x4FFFC;

        let mut ops = Vec::new();

        // make sure address is valid
        ops.push(Opcode::I32Const(6u32.into()));
        ops.push(Opcode::MemoryGrow);

        // keep addr under the value for store: push addr first
        ops.push(Opcode::I32Const(base.into()));

        // compute v = (a+b) via add
        ops.push(Opcode::I32Const(a.into()));
        ops.push(Opcode::I32Const(b.into()));
        let call_add_idx = ops.len();
        ops.push(Opcode::CallInternal(0u32)); // patch later

        // then <<2 via shl2
        let call_shl_idx = ops.len();
        ops.push(Opcode::CallInternal(0u32)); // patch later

        // store unaligned at base+3 (store pops value, then addr)
        ops.push(Opcode::I32Store(3u32));

        // load back and compare
        ops.push(Opcode::I32Const(base.into()));
        ops.push(Opcode::I32Load(3u32));
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // append helpers and patch the calls
        let add_pos = ops.len() as u32;
        ops.extend(add);
        let shl2_pos = ops.len() as u32;
        ops.extend(shl2);

        ops[call_add_idx] = Opcode::CallInternal(add_pos);
        ops[call_shl_idx] = Opcode::CallInternal(shl2_pos);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // Dot product length=8 via stateful step function (loads, stores, ptr++), FIXED store order.
    #[test]
    fn test_dot_product_len8_via_step_function() {
        // memory layout
        let base: u32 = 0x20000;
        let pa = base; // ptr A
        let pb = base + 4; // ptr B
        let acc = base + 8; // accumulator
        let rem = base + 12; // remaining
        let arr_a = base + 64; // A[8]
        let arr_b = base + 64 + 8 * 4; // B[8]

        // step():
        //   acc = acc + (*pa * *pb)
        //   pa += 4; pb += 4; rem -= 1
        //   return rem
        let step = vec![
            // ---- acc = acc + (*pa * *pb) ----
            // Push acc address first so it's under the computed value at store time.
            Opcode::I32Const(acc.into()),
            // *pa
            Opcode::I32Const(pa.into()),
            Opcode::I32Load(0u32), // load pa (pointer)
            Opcode::I32Load(0u32), // load *pa
            // *pb
            Opcode::I32Const(pb.into()),
            Opcode::I32Load(0u32), // load pb (pointer)
            Opcode::I32Load(0u32), // load *pb
            // *pa * *pb
            Opcode::I32Mul,
            // + acc
            Opcode::I32Const(acc.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Add,
            // store to acc (stack: ... addr(acc), value)
            Opcode::I32Store(0u32),
            // ---- pa += 4 ----
            Opcode::I32Const(pa.into()),
            Opcode::I32Const(pa.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(4u32.into()),
            Opcode::I32Add,
            Opcode::I32Store(0u32),
            // ---- pb += 4 ----
            Opcode::I32Const(pb.into()),
            Opcode::I32Const(pb.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(4u32.into()),
            Opcode::I32Add,
            Opcode::I32Store(0u32),
            // ---- rem -= 1 ----
            Opcode::I32Const(rem.into()),
            Opcode::I32Const(rem.into()),
            Opcode::I32Load(0u32),
            Opcode::I32Const(1u32.into()),
            Opcode::I32Sub,
            Opcode::I32Store(0u32),
            // return rem
            Opcode::I32Const(rem.into()),
            Opcode::I32Load(0u32),
            Opcode::Return,
        ];

        // expected: A=[1..8], B=[8..1] => dot = 120
        let a_vals = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let b_vals = [8u32, 7, 6, 5, 4, 3, 2, 1];

        let mut ops = Vec::new();
        // grow memory
        ops.push(Opcode::I32Const(3u32.into()));
        ops.push(Opcode::MemoryGrow);

        // init pointers
        ops.push(Opcode::I32Const(pa.into()));
        ops.push(Opcode::I32Const(arr_a.into()));
        ops.push(Opcode::I32Store(0u32));
        ops.push(Opcode::I32Const(pb.into()));
        ops.push(Opcode::I32Const(arr_b.into()));
        ops.push(Opcode::I32Store(0u32));
        // acc=0, rem=8
        ops.push(Opcode::I32Const(acc.into()));
        ops.push(Opcode::I32Const(0u32.into()));
        ops.push(Opcode::I32Store(0u32));
        ops.push(Opcode::I32Const(rem.into()));
        ops.push(Opcode::I32Const(8u32.into()));
        ops.push(Opcode::I32Store(0u32));

        // write arrays
        for (i, v) in a_vals.iter().enumerate() {
            let addr = arr_a + (i as u32) * 4;
            ops.push(Opcode::I32Const(addr.into()));
            ops.push(Opcode::I32Const((*v).into()));
            ops.push(Opcode::I32Store(0u32));
        }
        for (i, v) in b_vals.iter().enumerate() {
            let addr = arr_b + (i as u32) * 4;
            ops.push(Opcode::I32Const(addr.into()));
            ops.push(Opcode::I32Const((*v).into()));
            ops.push(Opcode::I32Store(0u32));
        }

        // 8 calls (drop returned rem)
        let mut call_sites = Vec::<usize>::new();
        for _ in 0..8 {
            call_sites.push(ops.len());
            ops.push(Opcode::CallInternal(0u32));
            ops.push(Opcode::Drop);
        }

        // check acc == 120
        ops.push(Opcode::I32Const(acc.into()));
        ops.push(Opcode::I32Load(0u32));
        ops.push(Opcode::I32Const(120u32.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // patch function position
        let step_pos = ops.len() as u32;
        for idx in call_sites {
            ops[idx] = Opcode::CallInternal(step_pos);
        }
        ops.extend(step);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // 3) Branch-free piecewise select using compares and arithmetic masks — FIXED
    #[test]
    fn test_piecewise_select_without_branches() {
        let scratch: u32 = 0x18000; // holds 'lt' (0 or 1)

        let x = 13u32;
        let y = 25u32;
        let left = (y - x) << 3; // 96
        let expected = left;

        let mut ops = Vec::new();

        // grow so 0x18000 is valid (2 pages = 128 KiB)
        ops.push(Opcode::I32Const(2u32.into()));
        ops.push(Opcode::MemoryGrow);

        // store lt at scratch: push addr first, then compute lt => stack [addr, lt]
        ops.push(Opcode::I32Const(scratch.into())); // addr under value
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::I32LtU); // lt
        ops.push(Opcode::I32Store(0u32)); // pops value, then addr

        // compute left = (y-x)<<3 and multiply by lt
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Sub);
        ops.push(Opcode::I32Const(3u32.into()));
        ops.push(Opcode::I32Shl); // left
        ops.push(Opcode::I32Const(scratch.into()));
        ops.push(Opcode::I32Load(0u32)); // lt
        ops.push(Opcode::I32Mul); // left * lt

        // compute right = (x-y)>>1 and multiply by (1-lt)
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::I32Sub);
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::I32ShrU); // right
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::I32Const(scratch.into()));
        ops.push(Opcode::I32Load(0u32)); // lt
        ops.push(Opcode::I32Sub); // 1 - lt
        ops.push(Opcode::I32Mul); // (1-lt)*right

        // add both branches and compare
        ops.push(Opcode::I32Add);
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }
    // Sign vs zero extension mix: Load8S + Load8U on the same byte and combine (unchanged)
    #[test]
    fn test_sign_and_zero_extension_mix() {
        let addr: u32 = 0x12000;
        // byte 0xF0 = 240; signed as i8 -> -16; -16 + 240 = 224
        let expected = 224u32;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(2u32.into()));
        ops.push(Opcode::MemoryGrow);

        // store 0x000000F0 at addr
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(0x0000_00F0u32.into()));
        ops.push(Opcode::I32Store(0u32));

        // Load8S + Load8U, add
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Load8S(0u32)); // (i8)240 == -16 (as u32 two's complement)
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Load8U(0u32)); // 240
        ops.push(Opcode::I32Add); // 224

        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    #[test]
    fn test_table_init() {
        let ops = vec![
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableGrow(0),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(1.into()),
            Opcode::TableInit(0),
            Opcode::TableGet(0),
        ];
        let elements = vec![5u32, 7u32];
        let program = Program::from_instrs(ops).with_elements(elements);

        let mut rt = Executor::new(program, SP1CoreOpts::default());

        rt.run().unwrap();
    }

    /// Boundary check: a 32‑bit load that starts inside the last page but
    /// crosses the end of the page must trap. This exercises the VM's
    /// out‑of‑bounds path (which none of the existing tests hit).
    #[test]
    fn test_oob_load_crosses_page_end_traps() {
        // Allocate exactly one 64KiB page.
        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::MemoryGrow);

        // Address such that addr + 4 overflows the single page (65536 bytes).
        let addr = (64 * 1024u32) - 2; // 65534

        // Attempt a 32‑bit load at addr -> must trap as OOB.
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Load(0u32));

        // If the load does not trap, we would return; but we expect an error.
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        let res = rt.run();
        assert!(res.is_err(), "expected out-of-bounds 32-bit load to trap");
        if let Err(e) = res {
            // Make debugging easier when it fails on CI.
            eprintln!("oob load produced error: {:?}", e);
        }
    }

    /// Boundary check: a 32-bit store that starts inside the last page but
    /// crosses the page end must trap. Complements the load OOB test by
    /// exercising the write-side error path.
    #[test]
    fn test_oob_store_crosses_page_end_traps() {
        // Grow memory to exactly one 64KiB page.
        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::MemoryGrow);

        // Choose an address near the very end so addr..addr+3 exceeds bounds.
        let addr = (64 * 1024u32) - 1; // 65535

        // Push address first so value is on top for store (store pops value, then addr).
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(0xDEAD_BEEFu32.into()));
        ops.push(Opcode::I32Store(0u32));

        // If it didn't trap (it should), execution would continue to Return.
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        let res = rt.run();
        assert!(res.is_err(), "expected out-of-bounds 32-bit store to trap");
        if let Err(e) = res {
            eprintln!("oob store produced error: {:?}", e);
        }
    }

    /// OOB via non-zero offset: base address is inside the page but
    /// `addr + offset + (size-1)` crosses the end. Ensures offset is
    /// included in bounds checking for loads.
    #[test]
    fn test_oob_load_with_nonzero_offset_traps() {
        // Grow to exactly one 64KiB page.
        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::MemoryGrow);

        // Choose base so base is valid, but base + 3 (offset) + 3 (size-1) crosses.
        let base = (64 * 1024u32) - 4; // 65532
        let offset = 3u32; // base + offset = 65535; needs 4 bytes -> OOB

        // Attempt a 32-bit load with non-zero offset -> must trap.
        ops.push(Opcode::I32Const(base.into()));
        ops.push(Opcode::I32Load(offset));

        // If not trapped, we'd return; but we expect an error.
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        let res = rt.run();
        assert!(res.is_err(), "expected OOB due to non-zero offset crossing page end");
        if let Err(e) = res {
            eprintln!("oob load (with offset) produced error: {:?}", e);
        }
    }

    /* /// Stack underflow: executing I32Add with fewer than two stack values
    /// must panic in the current rwasm backend (it does not return Err).
    /// This test documents that behavior explicitly.
    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn test_stack_underflow_add_traps() {
        let ops = vec![
            // No pushes
            Opcode::I32Add, // requires two operands -> underflow
            Opcode::Return,
        ];

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        // `run()` will panic before returning due to value-stack underflow.
        let _ = rt.run();
    }*/

    /// Sign-extension vs zero-extension on 16-bit loads: for the halfword
    /// 0x8000, Load16S yields 0xFFFF8000 and Load16U yields 0x00008000.
    /// Adding them must wrap to 0 in u32 arithmetic. This covers 16-bit
    /// sign extension and wrapping add semantics.
    #[test]
    fn test_load16s_plus_load16u_wraps_to_zero() {
        let addr: u32 = 0x14000;

        let mut ops = Vec::new();
        // Ensure memory covers addr
        ops.push(Opcode::I32Const(2u32.into()));
        ops.push(Opcode::MemoryGrow);

        // Store 0x00008000 at addr (low 16 bits are 0x8000)
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Const(0x0000_8000u32.into()));
        ops.push(Opcode::I32Store(0u32));

        // Load16S (=> 0xFFFF8000) and Load16U (=> 0x00008000), add -> 0
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Load16S(0u32));
        ops.push(Opcode::I32Const(addr.into()));
        ops.push(Opcode::I32Load16U(0u32));
        ops.push(Opcode::I32Add);

        // Compare with 0 and return
        ops.push(Opcode::I32Const(0u32.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    /// Success-path boundary check: writing a single byte at the very last
    /// address of a 64KiB page (65535) must SUCCEED, and reading it back
    /// via Load8U must yield the same value. This complements the OOB tests
    /// by exercising the inclusive upper bound for 1-byte accesses.
    #[test]
    fn test_store8_at_last_byte_succeeds() {
        let last: u32 = (64 * 1024) - 1; // 65535
        let byte: u32 = 0xAB;

        let mut ops = Vec::new();
        // Allocate exactly one page
        ops.push(Opcode::I32Const(1u32.into()));
        ops.push(Opcode::MemoryGrow);

        // Store8 at the last byte (value must be on top; store pops (value, addr))
        ops.push(Opcode::I32Const(last.into()));
        ops.push(Opcode::I32Const(byte.into()));
        ops.push(Opcode::I32Store8(0u32));

        // Load8U back from the same address and compare
        ops.push(Opcode::I32Const(last.into()));
        ops.push(Opcode::I32Load8U(0u32));
        ops.push(Opcode::I32Const(byte.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        let program = Program::from_instrs(ops);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 1);
    }

    #[test]
    fn test_i32rotl_masks_and_rotates() {
        // Ensure ROTL masks the shift by 31 and rotates correctly.
        let b: u32 = 0x2121_2121u32;
        let c: u32 = 0xffff_ffefu32; // masks to 15
        let expected: u32 = b.rotate_left(c & 31);

        let program = Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Rotl,
        ]);

        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        let top = rt.state.memory.get(rt.state.sp).unwrap().value;
        assert_eq!(top, expected, "I32Rotl result mismatch (masking or rotation incorrect)");
    }

    #[test]
    fn test_i32rotr_basic() {
        // Basic ROTR by 1: 0x8000_0001 -> 0xC000_0000 (but we check via rotate_right to avoid
        // literal mistakes)
        let b: u32 = 0x8000_0001u32;
        let c: u32 = 1u32;
        let expected: u32 = b.rotate_right(c & 31);

        let program = Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Rotr,
        ]);

        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        let top = rt.state.memory.get(rt.state.sp).unwrap().value;
        assert_eq!(top, expected, "I32Rotr result mismatch");
    }
    #[test]
    fn test_i32rot() {
        let sp0 = SP_START;
        let b: u32 = 0x8000_0001u32;
        let c: u32 = 7u32;
        let program = Program::from_instrs(vec![
            Opcode::I32Const(b.into()),
            Opcode::I32Const(c.into()),
            Opcode::I32Rotr,
            Opcode::I32Const(c.into()),
            Opcode::I32Rotl,
        ]);

        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        let top = rt.state.memory.get(rt.state.sp).unwrap().value;
        assert_eq!(top, b, "recovery result mismatch");
        assert_eq!(sp0, rt.state.sp + UNIT);
    }

    #[test]
    fn test_i32popcnt() {
        fn check(a: u32) {
            let sp0 = SP_START;
            let program = Program::from_instrs(vec![Opcode::I32Const(a.into()), Opcode::I32Popcnt]);

            let mut rt = Executor::new(program, SP1CoreOpts::default());
            rt.run().unwrap();

            let expected = a.count_ones();
            let top = rt.state.memory.get(rt.state.sp).unwrap().value;
            assert_eq!(top, expected, "I32Popcnt result mismatch for a={:#x}", a);
            // One 32-bit value pushed
            assert_eq!(sp0, rt.state.sp + UNIT);
        }

        // Edge cases
        check(0x0000_0000); // 0
        check(0xFFFF_FFFF); // 32
        check(0x0000_0001); // 1
        check(0x8000_0000); // 1
        check(0x7FFF_FFFF); // 31

        // Patterns
        check(0xAAAA_AAAA); // 16 ones
        check(0x5555_5555); // 16 ones

        // Random-ish sanity values
        check(0x0137_0137);
        check(0xDEAD_BEEF);
    }
    #[test]
    fn test_i32clz() {
        fn check(a: u32) {
            let sp0 = SP_START;
            let program = Program::from_instrs(vec![Opcode::I32Const(a.into()), Opcode::I32Clz]);

            let mut rt = Executor::new(program, SP1CoreOpts::default());
            rt.run().unwrap();

            let expected = a.leading_zeros(); // WASM spec: clz(0) = 32
            let top = rt.state.memory.get(rt.state.sp).unwrap().value;
            assert_eq!(top, expected, "I32Clz result mismatch for a={:#x}", a);
            // One 32-bit value pushed
            assert_eq!(sp0, rt.state.sp + UNIT);
        }

        // Edge cases
        check(0x0000_0000); // 32
        check(0x0000_0001); // 31
        check(0x8000_0000); // 0

        // Boundary & pattern cases
        check(0x0000_FFFF); // 16
        check(0x00F0_0000); // 8
        check(0x7FFF_FFFF); // 1
        check(0xFFFF_0000); // 0

        // Random-ish sanity values
        check(0x0137_0137);
        check(0xDEAD_BEEF);
    }
    #[test]
    fn test_i32ctz() {
        fn check(a: u32) {
            let sp0 = SP_START;
            let program = Program::from_instrs(vec![Opcode::I32Const(a.into()), Opcode::I32Ctz]);

            let mut rt = Executor::new(program, SP1CoreOpts::default());
            rt.run().unwrap();

            let expected = a.trailing_zeros(); // Rust matches WASM semantics: ctz(0) = 32
            let top = rt.state.memory.get(rt.state.sp).unwrap().value;
            assert_eq!(top, expected, "I32Ctz result mismatch for a={:#x}", a);
            // One 32-bit value pushed
            assert_eq!(sp0, rt.state.sp + UNIT);
        }

        // Edge cases
        check(0x0000_0000); // ctz(0) = 32
        check(0x0000_0001); // 0
        check(0x0000_0002); // 1
        check(0x0000_0004); // 2
        check(0x8000_0000); // 31

        // Boundary & pattern cases
        check(0x0001_0000); // 16
        check(0xFFFF_0000); // 16 (low half zero)
        check(0x00F0_0000); // 20

        // Random-ish sanity values
        check(0x0137_0137);
        check(0xDEAD_BEEF);
    }

    #[test]
    fn test_i32add64() {
        fn check(a: u32, b: u32) {
            let sp0 = SP_START;
            let program = Program::from_instrs(vec![
                Opcode::I32Const(a.into()),
                Opcode::I32Const(b.into()),
                Opcode::I32Add64,
            ]);
            let mut rt = Executor::new(program, SP1CoreOpts::default());
            rt.run().unwrap();

            let sum = (a as u64) + (b as u64);
            let lo = (sum & 0xFFFF_FFFF) as u32;
            let hi = (sum >> 32) as u32;

            // Convention for 64-bit ops: top-of-stack = HI, next = LO.
            assert_eq!(
                rt.state.memory.get(rt.state.sp).unwrap().value,
                hi,
                "HI mismatch for a={:#x}, b={:#x}",
                a,
                b
            );
            assert_eq!(
                rt.state.memory.get(rt.state.sp + 4).unwrap().value,
                lo,
                "LO mismatch for a={:#x}, b={:#x}",
                a,
                b
            );
            // Two 32-bit words were produced
            assert_eq!(sp0, rt.state.sp + 2 * UNIT);
        }

        // Basic and edge cases
        check(0, 0); // 0 + 0 -> (hi=0, lo=0)
        check(u32::MAX, 0); // max + 0
        check(u32::MAX, 1); // carry into HI -> (hi=1, lo=0)
        check(u32::MAX, u32::MAX); // 0xFFFF_FFFF + 0xFFFF_FFFF -> (hi=1, lo=0xFFFF_FFFE)
        check(0x7FFF_FFFF, 1); // boundary without HI carry -> (hi=0, lo=0x8000_0000)

        // Cross terms around the carry boundary
        check(0xFFFF_0000, 0x0000_FFFF); // no HI carry -> (hi=0, lo=0xFFFF_FFFF)
        check(0xFFFF_0001, 0x0000_FFFF); // exact 2^32 -> (hi=1, lo=0)

        // Symmetry / commutativity sanity
        check(0x1234_5678, 0x9ABC_DEF0);
        check(0x9ABC_DEF0, 0x1234_5678);
    }
    #[test]
    fn test_i32mul64() {
        fn check(a: u32, b: u32) {
            let sp0 = SP_START;
            let program = Program::from_instrs(vec![
                Opcode::I32Const(a.into()),
                Opcode::I32Const(b.into()),
                Opcode::I32Mul64,
            ]);
            let mut rt = Executor::new(program, SP1CoreOpts::default());
            rt.run().unwrap();

            let prod = (a as i64).wrapping_mul(b as i64);
            let (lo, hi) = prod.split_into_i32_tuple();

            // After 64-bit ops, convention is: top-of-stack = HI, next = LO (see test_i32add64).
            assert_eq!(
                rt.state.memory.get(rt.state.sp).unwrap().value,
                hi as u32,
                "HI mismatch for a={:#x}, b={:#x}",
                a,
                b
            );
            assert_eq!(
                rt.state.memory.get(rt.state.sp + 4).unwrap().value,
                lo as u32,
                "LO mismatch for a={:#x}, b={:#x}",
                a,
                b
            );
            // Two 32-bit words remain on the stack
            assert_eq!(sp0, rt.state.sp + 2 * UNIT);
        }

        // Edge & sanity cases
        check(0, 0); // zero * zero
        check(u32::MAX, 1); // max * 1
        check(u32::MAX, u32::MAX); // max * max -> hi = 0xFFFF_FFFE, lo = 1
        check(0x8000_0000, 2); // 2^31 * 2 = 2^32 -> hi=1, lo=0
        check(0xFFFF_0000, 0x0000_FFFF); // cross terms
                                         // Random-ish sanity and commutativity
        check(0x1234_5678, 0x9ABC_DEF0);
        check(0x9ABC_DEF0, 0x1234_5678);
    }
    #[test]
    fn test_i32extend8s() {
        {
            // Keep any cases you like; this one hits the boundary (-128)
            let cases = [
                0x0000_0000,    // 0  -> 0
                0x0000_7FFF,    // +32767 stays +32767
                0x0000_8000,    // -32768 -> 0xFFFF_8000
                0x0000_FFFF,    // -1     -> 0xFFFF_FFFF
                0xDEAD_8001u32, // 0x8001 -> -32767 -> 0xFFFF_8001
                0xBEEF_1234u32, // 0x1234 -> +4660  -> 0x0000_1234
            ];

            for &input in &cases {
                println!("next test of test_i32extend8s is {}", input);
                // Mirror: i32_extend8_s(x) = (x as i8) as i32
                let expected: u32 = ((input as i8) as i32) as u32;

                let program =
                    Program::from_instrs(vec![Opcode::I32Const(input.into()), Opcode::I32Extend8S]);

                let mut rt = Executor::new(program, SP1CoreOpts::default());
                rt.run().unwrap();

                let top = rt.state.memory.get(rt.state.sp + 3).unwrap().value;
                assert_eq!(top, expected, "I32Extend8S({:#010x}) mismatch", input);
            }
        }
    }
    #[test]
    fn test_i32extend16s() {
        {
            // Sign-extend from the low 16 bits into i32, push as u32.
            // Include boundary values and cases where high bits are noisy.
            let cases: [u32; 6] = [
                0x0000_0000, // 0  -> 0
                0x0000_7FFF, // +32767 stays +32767
                0x0000_8000, // -32768 -> 0xFFFF_8000
                0x0000_FFFF, // -1     -> 0xFFFF_FFFF
                0xDEAD_8001, // 0x8001 -> -32767 -> 0xFFFF_8001
                0xBEEF_1234, // 0x1234 -> +4660  -> 0x0000_1234
            ];
            for &input in &cases {
                let expected: u32 = ((input as i16) as i32) as u32;
                let program = Program::from_instrs(vec![
                    Opcode::I32Const(input.into()),
                    Opcode::I32Extend16S,
                ]);

                let mut rt = Executor::new(program, SP1CoreOpts::default());
                rt.run().unwrap();

                let top = rt.state.memory.get(rt.state.sp).unwrap().value;
                assert_eq!(top, expected, "I32Extend16S({:#010x}) mismatch", input);
            }
        }
    }

    #[test]
    fn test_sig_id_check() {
        let ops = vec![
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableGrow(0),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()),
            Opcode::TableInit(0),
            Opcode::TableGet(0),
            Opcode::I32Const(1.into()),
            Opcode::CallIndirect(1u32),
            Opcode::TableGet(0),
            Opcode::SignatureCheck(1),
            Opcode::Return,
            Opcode::I32Const(99.into()),
            Opcode::I32Const(98.into()),
            Opcode::I32Add,
            Opcode::Return,
        ];

        let elements = vec![12u32, 12u32];
        let program = Program::from_instrs(ops).with_elements(elements);

        let mut rt = Executor::new(program, SP1CoreOpts::default());

        rt.run().unwrap();
    }
    /// Helper to run a program and return the value at the top of the stack.
    fn run_and_get_stack_top(opcodes: Vec<Opcode>) -> u32 {
        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();
        rt.state.memory.get(rt.state.sp).unwrap().value
    }

    #[test]
    fn test_add_simple() {
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add, // 32 + 4 = 36
        ];

        let result = run_and_get_stack_top(opcodes);
        assert_eq!(result, x_value + y_value);
    }

    #[test]
    fn test_sub_simple() {
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Sub, // 32 - 4 = 28
        ];

        let result = run_and_get_stack_top(opcodes);
        assert_eq!(result, x_value - y_value);
    }

    #[test]
    fn test_add_overflow_wraps() {
        // Edge Case: u32::MAX + 1 should wrap to 0
        let sp0 = SP_START;
        let opcodes =
            vec![Opcode::I32Const(u32::MAX.into()), Opcode::I32Const(1u32.into()), Opcode::I32Add];

        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        // Verify Value
        assert_eq!(rt.state.memory.get(rt.state.sp).unwrap().value, 0);
        // Verify Stack Pointer (Should be SP_START + 4 bytes, as 2 args popped, 1 pushed)
        assert_eq!(sp0, rt.state.sp + 4);
    }

    #[test]
    fn test_sub_underflow_wraps() {
        // Edge Case: 0 - 1 should wrap to u32::MAX
        let expected = u32::MAX;
        let opcodes = vec![
            Opcode::I32Const(0u32.into()),
            Opcode::I32Const(1u32.into()),
            Opcode::I32Sub,
            // Check result immediately
            Opcode::I32Const(expected.into()),
            Opcode::I32Eq,
        ];

        let result = run_and_get_stack_top(opcodes);
        assert_eq!(result, 1); // 1 means True (Equality check passed)
    }

    #[test]
    fn test_add_negative_chained() {
        // Logic: 4 + (-1) + 5.
        // In u32 two's complement, -1 is 0xFFFF_FFFF.
        let x_value: u32 = 4;
        let y_value: u32 = 0xFFFF_FFFF; // -1
        let z_value: u32 = 5;
        let expected_res = x_value.wrapping_add(y_value).wrapping_add(z_value); // 8

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Add, // Stack: [4, ( -1 + 5 )] -> [4, 4]
            Opcode::I32Add, // Stack: [8]
            Opcode::I32Const(expected_res.into()),
            Opcode::I32Eq,
        ];

        let result = run_and_get_stack_top(opcodes);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_stack_operations_eq_drop() {
        // Tests that the stack pointer moves correctly after Add, Eq, and Drop
        let sp0 = SP_START;
        let x = 10;
        let y = 20;

        let opcodes = vec![
            Opcode::I32Const(x.into()),       // SP - 4
            Opcode::I32Const(y.into()),       // SP - 8
            Opcode::I32Add,                   // Pop 2, Push 1 -> SP - 4 (Value 30)
            Opcode::I32Const((x + y).into()), // SP - 8
            Opcode::I32Eq,                    // Pop 2, Push 1 -> SP - 4 (Value 1)
            Opcode::Drop,                     // Pop 1 -> SP Back to Start
        ];

        let program = Program::from_instrs(opcodes);
        let mut rt = Executor::new(program, SP1CoreOpts::default());
        rt.run().unwrap();

        // Assert Stack Pointer returned to start
        assert_eq!(sp0, rt.state.sp);
    }

    #[test]
    fn test_comprehensive_edge_cases() {
        // This test combines Identity, Inverse, and Zero properties
        // 1. Identity: x + 0 = x
        // 2. Inverse: x + y - y = x
        // 3. Zero Sub: x - 0 = x
        // 4. Self Sub: x - x = 0

        let x = 123456u32;
        let y = 789u32;

        let opcodes = vec![
            // Case 1: x + 0 == x
            Opcode::I32Const(x.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Add,
            Opcode::I32Const(x.into()),
            Opcode::I32Eq, // Stack: [1]
            // Case 2: x + y - y == x
            Opcode::I32Const(x.into()),
            Opcode::I32Const(y.into()),
            Opcode::I32Add,
            Opcode::I32Const(y.into()),
            Opcode::I32Sub,
            Opcode::I32Const(x.into()),
            Opcode::I32Eq, // Stack: [1, 1]
            // Case 3: x - 0 == x
            Opcode::I32Const(x.into()),
            Opcode::I32Const(0.into()),
            Opcode::I32Sub,
            Opcode::I32Const(x.into()),
            Opcode::I32Eq, // Stack: [1, 1, 1]
            // Case 4: x - x == 0
            Opcode::I32Const(x.into()),
            Opcode::I32Const(x.into()),
            Opcode::I32Sub,
            Opcode::I32Const(0.into()),
            Opcode::I32Eq, // Stack: [1, 1, 1, 1]
            // Verify all checks passed (AND them together)
            Opcode::I32And,
            Opcode::I32And,
            Opcode::I32And, // Stack: [1]
        ];

        let result = run_and_get_stack_top(opcodes);
        assert_eq!(result, 1, "One of the comprehensive edge cases failed");
    }
}
