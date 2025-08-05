#[cfg(feature = "profiling")]
use crate::profiler::Profiler;
use crate::{
    dependencies::{emit_branch_dependencies, emit_divrem_dependencies, emit_memory_dependencies},
    estimator::RecordEstimator,
    events::{ConstEvent, SysStateEvent, SyscallEvent},
    syscalls, SP_START,
};
use std::rc::Rc;
#[cfg(feature = "profiling")]
use std::{fs::File, io::BufWriter};
use std::{str::FromStr, sync::Arc};

use clap::ValueEnum;
use enum_map::EnumMap;
use hashbrown::HashMap;

use rwasm::{
    always_failing_syscall_handler, mem::MemoryLocalEvent, CallStack, ExecutionEngine, ExecutorConfig, ImportLinker, InstructionPtr, Opcode, RwasmExecutor, RwasmModule, RwasmStore, Store, Tracer, TrapCode, ValueStack, ValueStackPtr
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
    // },TODO: redo dependecies
    estimate_riscv_lde_size,
    events::{
        AluEvent, BranchEvent, CpuEvent, MemInstrEvent, MemoryInitializeFinalizeEvent,
        MemoryReadRecord, MemoryRecord, MemoryRecordEnum, MemoryWriteRecord,
        NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC,
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
///The default increment for the clk. we increase clk for two becaseu we have
/// a reading phase and a writing phase.
pub const DEFAULT_CLK_INC: u32 = 2*DEFAULT_PC_INC;
/// This is used in the `InstrEvent` to indicate that the opcode is not from the CPU.
/// A valid pc should be divisible by 4, so we use 1 to indicate that the pc is not used.
pub const UNUSED_PC: u32 = 1<<24;

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

struct RwasmExecutorState {
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

    /// Memory addresses that were initialized in this batch of shards. Used to minimize the size of
    /// checkpoints. The value stored is whether or not it had a value at the beginning of the batch.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum ExecutorMode {
    /// Run the execution with no tracing or checkpointing.
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
    /// If the feature `"profiling"` is enabled, this sets up the profiler. Otherwise, it does nothing.
    /// The argument `elf_bytes` must describe the same program as `self.program`.
    ///
    /// The profiler is configured by the following environment variables:
    ///
    /// - `TRACE_FILE`: writes Gecko traces to this path. If unspecified, the profiler is disabled.
    /// - `TRACE_SAMPLE_RATE`: The period between clock cycles where samples are taken. Defaults to 1.
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

        let rwasm_config = ExecutorConfig::default();
        let store = RwasmStore::new(
            rwasm_config,
            // TODO(dmitry123): "use import linker from fluentbase once tracer is merged"
            Arc::new(ImportLinker::default()),
            (),
            // TODO(dmitry123): "use syscall handler from runtime"
            always_failing_syscall_handler,
        );

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
        if addr % 4 != 0 || addr >= BABYBEAR_PRIME {
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
        //  1. This is the first time the address is touched, this corresponds to the
        //     condition record.shard != shard.
        //  2. The address is being accessed in a syscall. In this case, we need to send it. We use
        //     local_memory_access to detect this. *WARNING*: This means that we are counting
        //     on the .is_some() condition to be true only in the SyscallContext.
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
        if addr % 4 != 0 || addr >= BABYBEAR_PRIME {
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
        //  1. This is the first time the address is touched, this corresponds to the
        //     condition record.shard != shard.
        //  2. The address is being accessed in a syscall. In this case, we need to send it. We use
        //     local_memory_access to detect this. *WARNING*: This means that we are counting
        //     on the .is_some() condition to be true only in the SyscallContext.
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
        pc:u32,
        next_pc: u32,
        sp: u32,
        next_sp:u32,
        opcode: Opcode,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
        res: u32,
        record: MemoryAccessRecord,
    ) {
        println!("emit cpu");
        if !opcode.is_memory_instruction(){
              self.emit_cpu(clk,pc, next_pc, sp, next_sp,arg1, arg2, res, record, 0u32);
        } else{
            self.emit_cpu(clk,pc, next_pc, sp, next_sp,arg1, opcode.aux_value(), arg2, record, 0u32);
        }


        if opcode.is_alu_instruction() {
            self.emit_alu_event(pc,opcode, arg1, arg2, res);
        } else if opcode.is_memory_load_instruction() || opcode.is_memory_store_instruction() {
            self.emit_mem_instr_event(opcode, arg1, arg2, res, record);
        } else if opcode.is_branch_instruction() {
            self.emit_branch_event(opcode, arg1, arg2, res, next_pc);
        } else if opcode.is_ecall_instruction() {
            self.emit_syscall_event(clk, record.arg1_record, syscall_code, arg2, res, next_pc);
        } else if opcode.is_const_instruction() {
            self.emit_const_event(opcode);
        } else if opcode.is_state_instrucition() {
            #[cfg(debug_assertions)]
            {
                println!("sys_state_event not generated here");
            }
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
        pc:u32,
        next_pc: u32,
        sp: u32,
        next_sp:u32,
        arg1: u32,
        arg2: u32,
        res: u32,
        record: MemoryAccessRecord,
        exit_code: u32,
    ) {
        self.record.cpu_events.push(CpuEvent {
            clk,
            pc,
            next_pc,
            sp,
            next_sp,
            res,
            res_record: record.res_record,
            arg1,
            arg1_record: record.arg1_record,
            arg2,
            arg2_record: record.arg2_record,
            exit_code,
        });
    }

    /// Emit an ALU event.
    #[allow(clippy::too_many_lines)]
    fn emit_alu_event(&mut self,pc:u32 , opcode: Opcode, arg1: u32, arg2: u32, res: u32) {
        let event =
            AluEvent { pc,opcode, a: res, b: arg1, c: arg2, code: opcode.code() };
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
            Opcode::I32GeS
            | Opcode::I32GtS
            | Opcode::I32GeU
            | Opcode::I32GtU
            | Opcode::I32LeS
            | Opcode::I32LeU
            | Opcode::I32LtS
            | Opcode::I32LtU
            | Opcode::I32Eq
            | Opcode::I32Eqz
            | Opcode::I32Ne => {
                let use_signed_comparison = matches!(
                    opcode,
                    Opcode::I32GeS | Opcode::I32GtS | Opcode::I32LeS | Opcode::I32LtS
                );
                let arg1_lt_arg2 = if use_signed_comparison {
                    (event.b as i32) < (event.c as i32)
                } else {
                    event.b < event.c
                };
                let arg1_gt_arg2 = if use_signed_comparison {
                    (event.b as i32) > (event.c as i32)
                } else {
                    event.b > event.c
                };
                let cmp_ins = {
                    if use_signed_comparison {
                        Opcode::I32LtS
                    } else {
                        Opcode::I32LtU
                    }
                };
                let lt_comp_event = AluEvent {
                    pc:UNUSED_PC,
                    opcode: cmp_ins,
                    a: arg1_lt_arg2 as u32,
                    b: event.b,
                    c: event.c,
                    code: cmp_ins.code(),
                };
                let gt_comp_event = AluEvent {
                    pc: UNUSED_PC,
                    opcode: cmp_ins,
                    a: arg1_gt_arg2 as u32,
                    b: event.c,
                    c: event.b,
                    code: cmp_ins.code(),
                };
                if opcode==Opcode::I32LtS{
                     println!("gt event:{:?}",gt_comp_event);
               println!("lt event:{:?}",lt_comp_event);
                }

                self.record.lt_events.push(gt_comp_event);

                self.record.lt_events.push(lt_comp_event);
            }
            Opcode::I32Mul => {
                self.record.mul_events.push(event);
            }
            Opcode::I32DivS | Opcode::I32DivU | Opcode::I32RemS | Opcode::I32RemU => {
                self.record.divrem_events.push(event);
                emit_divrem_dependencies(self, event);
            }
            Opcode::I32Rotl | Opcode::I32Rotr => {
                todo!();
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
        println!("mem event:{:?}",event);
        self.record.memory_instr_events.push(event);
        emit_memory_dependencies(self, event);
    }

    // Emit a branch event.
    #[inline]
    fn emit_branch_event(&mut self, opcode: Opcode, arg1: u32, arg2: u32, res: u32, next_pc: u32) {
        let event = BranchEvent { pc: self.state.pc, next_pc, opcode, res, arg1, arg2 };
         println!("br event:{:?}",event);
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
        // for syscall events that are created within the precompiles' execute function.  Those events will be
        // added to precompile tables, which wouldn't use the op_a_0 field.  Note that we can't make
        // the op_a_0 field an Option<bool> in SyscallEvent because of the cbindgen.
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
    ) {
        let syscall_event =
            self.syscall_event(clk, a_record, Some(true), syscall_code, arg1, arg2, next_pc);

        self.record.syscall_events.push(syscall_event);
    }

    // Emit a branch event.
    #[inline]
    fn emit_const_event(&mut self, opcode: Opcode) {
        let event = ConstEvent { pc: self.state.pc, opcode, value: opcode.aux_value() };
        self.record.const_events.push(event);
    }

    // Emit a branch event.
    #[inline]
    fn emit_sys_state_event(
        &mut self,
        opcode: Opcode,
        fuel: u32,
        next_fuel: u32,
        max_memory: u32,
        next_max_memory: u32,
    ) {
        let event = SysStateEvent::new(opcode, fuel, next_fuel, max_memory, next_max_memory);
        self.record.sys_state_events.push(event);
    }

    /// Execute an ecall opcode.
    #[allow(clippy::type_complexity)]
    fn execute_ecall(
        &mut self,
    ) -> Result<(u32, u32, u32, u32, u32, SyscallCode, u32), ExecutionError> {
        // We peek at register x5 to get the syscall id. The reason we don't `self.rr` this
        // register is that we write to it later.
        let t0 = todo!();
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

        if self.unconstrained
            && (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
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

        // // If the syscall is `EXIT_UNCONSTRAINED`, the memory was restored to pre-unconstrained code
        // // in the execute function, so we need to re-read from x10 and x11.  Just do a peek on the
        // // registers.
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

        self.state.clk=op_state.clk;
        self.state.pc=op_state.pc;
        self.emit_events(
            op_state.clk,
            op_state.pc,
            op_state.next_pc,
            op_state.sp,
            op_state.next_sp,
            op_state.opcode,
            syscall,
            op_state.arg1,
            op_state.arg2,
            op_state.res,
            op_state.memory_access,
        );
        // if op_state.opcode.is_state_instrucition() {
        //TODO: generate sys_state_event here
        // }

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
            if self.state.global_clk % self.shape_check_frequency == 0 {
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
                        (count != 0).then(|| threshold - count).unwrap_or(usize::MAX)
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
        println!("pre_execute_records:{:?}", self.records);
        println!("pre_execute_record:{:?}", self.record);

        let done = self.execute()?; //TODO: fix execute
        println!("post_execute_record:{:?}", self.records);
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

    fn initialize(&mut self) {
        self.state.clk = 0;

        tracing::debug!("loading memory image");
        for item in self.program.memory_image.iter() {
            let addr = item.1;
            let value = item.0;
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
            self.initialize();
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
                let ip = InstructionPtr::new(self.program.module.code_section.instr.as_ptr());
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
            let res = self.execute_cycle(res)?;
            println!("self.record.cpuevent:{:?}",self.record.cpu_events);
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
            let total_mem = self.state.memory.page_table.exact_len(); //TODO: fix esitmator
                                                                      // The memory_image is already initialized in the MemoryProgram chip
                                                                      // so we subtract it off. It is initialized in the executor in the `initialize` function.
            estimator.memory_global_init_events = total_mem
                .checked_sub(self.record.program.module.data_section.len())
                .expect("program memory image should be accounted for in memory exact len")
                as u64;
            estimator.memory_global_finalize_events = total_mem as u64;
        }

        if self.emit_global_memory_events
            && (self.executor_mode == ExecutorMode::Trace
                || self.executor_mode == ExecutorMode::Checkpoint)
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
            // let addr_0_init = MemoryInitializeFinalizeEvent { addr: 0, value: 0, shard: 1, timestamp: 1, used: 1 };
            // let addr_0_final =  MemoryInitializeFinalizeEvent{ addr: 0, value: 0, shard: 1, timestamp: 2, used: 1 };
            // let addr_0_initialize_event =
            //     MemoryInitializeFinalizeEvent::initialize(0, 0, true);
            // println!("addr0init:{:?}",addr_0_initialize_event);
            // memory_initialize_events.push(addr_0_init);
            // memory_finalize_events.push(addr_0_final);


            // Count the number of touched memory addresses manually, since `PagedMemory` doesn't
            // already know its length.
            self.report.touched_memory_addresses = 0;

            for addr in self.state.memory.page_table.keys() {
                println!("addr:{}",addr);
                self.report.touched_memory_addresses += 1;

                // Program memory is initialized in the MemoryProgram chip and doesn't require any
                // events, so we only send init events for other memory addresses.

                let initial_value = self.state.uninitialized_memory.get(addr).unwrap_or(&0);
                let init_event = MemoryInitializeFinalizeEvent::initialize(
                    addr,
                    *initial_value,
                    true,
                );
                 println!("init_event:{:?}",init_event);
                memory_initialize_events.push(init_event);


                let record = *self.state.memory.get(addr).unwrap();
                let final_event =MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record);
                println!("final_event:{:?}",final_event);
                memory_finalize_events
                    .push(final_event);
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
    //     event_counts[RwasmAirId::AddSub] = opcode_counts[Opcode::ADD] + opcode_counts[Opcode::SUB];

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
    //     event_counts[RwasmAirId::Jump] = opcode_counts[Opcode::JAL] + opcode_counts[Opcode::JALR];

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

    //     // Note: we ignore the additional dependencies for addsub, since they are accounted for in
    //     // the maximal shapes.
    // }

    #[inline]
    fn log(&mut self, _: &Opcode) {
        #[cfg(feature = "profiling")]
        if let Some((ref mut profiler, _)) = self.profiler {
            if !self.unconstrained {
                profiler.record(self.state.global_clk, self.state.pc as u64);
            }
        }

        if !self.unconstrained && self.state.global_clk % 10_000_000 == 0 {
            tracing::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc);
        }
    }
}

impl Default for ExecutorMode {
    fn default() -> Self {
        Self::Simple
    }
}
fn peek_stack(rt: &Executor) {
    let start = SP_START;
    for idx in (1..16) {
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
    for idx in (1..16) {
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

/// Aligns an address to the nearest word below or equal to it.
#[must_use]
pub const fn align(addr: u32) -> u32 {
    addr - addr % 4
}

#[cfg(test)]
mod tests {
    use super::peek_stack;
    use crate::{align, Executor, Program};
    use hashbrown::HashMap;

    use rwasm::{mem_index::{AddressType, SP_START, UNIT}, BranchOffset, Op, Opcode};
    use sp1_stark::SP1CoreOpts;

    #[test]
    fn test_add() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add, // 32 + 4 = 36
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value + y_value);
        println!("initial sp_value {} and last state.sp {}", sp_value, runtime.state.sp);
        assert_eq!(sp_value, runtime.state.sp +4 );
    }
    #[test]
    fn test_add_eq() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add, // 32 + 4 = 36
            Opcode::I32Const((x_value+y_value).into()),
            Opcode::I32Eq, //stack has now 1
            //Opcode::Drop,
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        println!("initial sp_value {} and last state.sp {}", sp_value, runtime.state.sp);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp+4);
    }
    #[test]
    fn test_add_eq_drop() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add, // 32 + 4 = 36
            Opcode::I32Const((x_value+y_value).into()),
            Opcode::I32Eq, //stack has now 1
            Opcode::Drop, // no stack elements
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        println!("initial sp_value {} and last state.sp {}", sp_value, runtime.state.sp);
        assert_eq!(sp_value, runtime.state.sp);
    }
    #[test]
    fn test_sub() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 32;
        let y_value: u32 = 4;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Sub, // 32 - 4 = 28
            Opcode::I32Const((x_value-y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
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
            Opcode::I32Const((x_value^y_value).into()),
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
            Opcode::I32Const((x_value&y_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_addi_negative() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 4;
        let y_value: u32 = 0xFFFF_FFFF;
        let z_value: u32 = 5;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32Add,
            Opcode::I32Add,
            Opcode::I32Const((x_value - 1 + z_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            1
        );
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
            Opcode::I32Const(( x_value | y_value | z_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            1
        );
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
            Opcode::I32Const(( x_value & y_value & z_value).into()),
            Opcode::I32Eq, //stack has now 1
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            1
        );
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
            Opcode::I32Const(( x_value * y_value).into()),
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
        let x_value: u32 = 321;
        let y_value: u32 = 233;
        let z_value: u32 = 0;

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32GeS, // check whether signed x_value is greater than or equal to signed y_value
            Opcode::I32GeU, // check whether unsigned x_value is greater than or equal to unsigned y_value
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
            Opcode::I32LeU, // check whether unsigned x_value is less than or equal to unsigned y_value
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, 1);
        assert_eq!(sp_value, runtime.state.sp + 4);
    }
    #[test]
    fn test_divs_divu() {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 320;
        let y_value: u32 = 10;
        let z_value: u32 = 2;
        let mut mem = HashMap::new();
        mem.insert(sp_value - 8, x_value);
        mem.insert(sp_value - 4, y_value);
        mem.insert(sp_value, z_value);

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::I32DivS, // divide x_value by y_value and return quotient (x and y are signed)
            Opcode::I32DivU, // divide x_value by y_value and return quotient
        ];

        let program = Program::from_instrs(opcodes);
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        assert_eq!(
            runtime.state.memory.get(runtime.state.sp).unwrap().value,
            x_value / (y_value / z_value)
        );
        assert_eq!(sp_value, runtime.state.sp + 4);
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
                .get(AddressType::GlobalMemory(addr).to_virtual_addr())
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

        //discuss why Opcode::I32Store16(0.into()),Opcode::I32Store16(1.into()) are not working if they are subsequent
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
        let v_addr = AddressType::GlobalMemory(addr).to_virtual_addr();
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
        let v_addr = AddressType::GlobalMemory(addr).to_virtual_addr();
        assert_eq!(
            runtime.state.memory.get(v_addr).unwrap().value,
            ((x_value & 0x0000_00FF)
                + ((y_value & 0x0000_00FF) << 8)
                + ((z_value & 0x0000_00FF) << 16)
                + ((t_value & 0x0000_00FF) << 24))
        );
        assert_eq!(sp_value, runtime.state.sp + UNIT);
    }

    fn simple_memory_load_opcode_test(
        mut mem: HashMap<u32, u32>,
        opcodes: Vec<Opcode>,
        expected: u32,
    ) {
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
        let z_value: u32 = 0xFFF3_000a;
        let t_value: u32 = 0xFFF4_000b;
        let addr: u32 = 0xDD_0000;

        let memOpcodes = vec![
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
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store(0.into())],
        //     addr + 0,
        //     t_value,
        // );
        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store(16.into())],
        //     addr + 16,
        //     t_value,
        // );

        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store16(0.into())],
        //     addr,
        //     (t_value & 0x0000_FFFF),
        // );
        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store16(2.into())],
        //     addr,
        //     (t_value & 0x0000_FFFF) << 16,
        // );

        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store8(0.into())],
        //     addr,
        //     (t_value & 0x0000_00FF),
        // );
        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store8(1.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 8,
        // );
        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store8(2.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 16,
        // );
        // simple_memory_store_opcode_test(
        //     memOpcodes.clone(),
        //     vec![Opcode::I32Store8(3.into())],
        //     addr,
        //     (t_value & 0x0000_00FF) << 24,
        // );

        /*simple_memory_store_opcode_test(
            memOpcodes.clone(),
            vec![Opcode::I32Store16(0.into()), Opcode::I32Store16(1.into())],
            addr,
            (z_value & 0x0000_FFFF) + ((t_value & 0x0000_FFFF) << 16),
        );

          simple_memory_store_opcode_test(
              memOpcodes.clone(),
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
        let x_value: u32 = 0xdFFF_00FF;
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
            Opcode::LocalTee(4u32), // get last element and put it into address (where address = last sp + 16)
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

    // #[test]
    // fn test_call_internal_and_return() {
    //     let sp_value: u32 = SP_START;
    //     let x_value: u32 = 0x3;
    //     let y_value: u32 = 0x5;
    //     let z_value: u32 = 0x7;
    //     let mut functions = vec![0, 24];

    //     let opcodes = vec![
    //         Opcode::I32Const(x_value.into()),
    //         Opcode::I32Const(y_value.into()),
    //         Opcode::I32Const(z_value.into()),
    //         Opcode::CallInternal(1u32.into()),
    //         // Opcode::Return(DropKeep::none()),
    //         Opcode::I32Add,
    //         Opcode::I32Add,
    //         Opcode::Return,
    //     ];

    //     let program = Program::new_with_memory_and_func(opcodes, HashMap::new(), functions, 0, 0);
    //     //  memory_image: BTreeMap::new() };
    //     let mut runtime = Executor::new(program, SP1CoreOpts::default());
    //     runtime.run().unwrap();
    //     assert_eq!(
    //         runtime.state.memory.get(runtime.state.sp).unwrap().value,
    //         x_value + y_value + z_value
    //     );
    // }
    #[test]
    fn test_i32constwithAdd() {
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
        runtime.execute();
        println!("record:{:?}", runtime.record);
        println!("records:{:?}", runtime.records);
        assert_eq!(runtime.state.sp, sp_value - 4);
        assert_eq!(runtime.state.memory.get(runtime.state.sp).unwrap().value, x_value);
    }
}
