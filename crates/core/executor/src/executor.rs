use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::Arc,
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use sp1_stark::SP1CoreOpts;

use thiserror::Error;

use crate::{
    context::SP1Context,
    events::{
        create_alu_lookup_id, create_alu_lookups, AluEvent, CpuEvent, ExecMemoryRecords, LookupId, MemoryAccessPosition, MemoryInitializeFinalizeEvent, MemoryReadRecord, MemoryRecord, MemoryWriteRecord
    },
    hook::{HookEnv, HookRegistry},
    memory::{Entry, PagedMemory},
    record::{ExecutionRecord, MemoryAccessRecord},
    report::ExecutionReport,
    state::{ExecutionState, ForkState},
    subproof::{DefaultSubproofVerifier, SubproofVerifier},
    syscalls::{default_syscall_map, Syscall, SyscallCode, SyscallContext},
     Opcode, Program, Register,
};
use rwasm::engine::bytecode::Instruction;

/// An executor for the SP1 RISC-V zkVM.
///
/// The exeuctor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct Executor<'a> {
    /// The program.
    pub program: Arc<Program>,

    /// The state of the execution.
    pub state: ExecutionState,

    /// The current trace of the execution that is being collected.
    pub record: ExecutionRecord,

    /// The collected records, split by cpu cycles.
    pub records: Vec<ExecutionRecord>,

    /// The memory accesses for the current cycle.
    pub memory_accesses: MemoryAccessRecord,

    /// The maximum size of each shard.
    pub shard_size: u32,

    /// The maximimum number of shards to execute at once.
    pub shard_batch_size: u32,

    /// A counter for the number of cycles that have been executed in certain functions.
    pub cycle_tracker: HashMap<String, (u64, u32)>,

    /// A buffer for stdout and stderr IO.
    pub io_buf: HashMap<u32, String>,

    /// A buffer for writing trace events to a file.
    pub trace_buf: Option<BufWriter<File>>,

    /// Whether the runtime is in constrained mode or not.
    ///
    /// In unconstrained mode, any events, clock, register, or memory changes are reset after
    /// leaving the unconstrained block. The only thing preserved is writes to the input
    /// stream.
    pub unconstrained: bool,

    /// The state of the runtime when in unconstrained mode.
    pub unconstrained_state: ForkState,

    /// The mapping between syscall codes and their implementations.
    pub syscall_map: HashMap<SyscallCode, Arc<dyn Syscall>>,

    /// The maximum number of cycles for a syscall.
    pub max_syscall_cycles: u32,

    /// The mode the executor is running in.
    pub executor_mode: ExecutorMode,

    /// Report of the program execution.
    pub report: ExecutionReport,

    /// Whether we should write to the report.
    pub print_report: bool,

    /// Verifier used to sanity check `verify_sp1_proof` during runtime.
    pub subproof_verifier: Arc<dyn SubproofVerifier + 'a>,

    /// Registry of hooks, to be invoked by writing to certain file descriptors.
    pub hook_registry: HookRegistry<'a>,

    /// The options for the runtime.
    pub opts: SP1CoreOpts,

    /// The maximum number of cpu cycles to use for execution.
    pub max_cycles: Option<u64>,

    /// Memory addresses that were touched in this batch of shards. Used to minimize the size of
    /// checkpoints.
    pub memory_checkpoint: PagedMemory<Option<MemoryRecord>>,
}

/// The different modes the executor can run in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorMode {
    /// Run the execution with no tracing or checkpointing.
    Simple,
    /// Run the execution with checkpoints for memory.
    Checkpoint,
    /// Run the execution with full tracing of events.
    Trace,
}

/// Errors that the [``Executor``] can throw.
#[derive(Error, Debug, Serialize, Deserialize)]
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
}

macro_rules! assert_valid_memory_access {
    ($addr:expr, $position:expr) => {
        #[cfg(not(debug_assertions))]
        {}
    };
}

impl<'a> Executor<'a> {
    /// Create a new [``Executor``] from a program and options.
    #[must_use]
    pub fn new(program: Program, opts: SP1CoreOpts) -> Self {
        Self::with_context(program, opts, SP1Context::default())
    }

    /// Create a new runtime from a program, options, and a context.
    ///
    /// # Panics
    ///
    /// This function may panic if it fails to create the trace file if `TRACE_FILE` is set.
    #[must_use]
    pub fn with_context(program: Program, opts: SP1CoreOpts, context: SP1Context<'a>) -> Self {
        // Create a shared reference to the program.
        let program = Arc::new(program);

        // Create a default record with the program.
        let record = ExecutionRecord { program: program.clone(), ..Default::default() };

        // If `TRACE_FILE`` is set, initialize the trace buffer.
        let trace_buf = if let Ok(trace_file) = std::env::var("TRACE_FILE") {
            let file = File::create(trace_file).unwrap();
            Some(BufWriter::new(file))
        } else {
            None
        };

        // Determine the maximum number of cycles for any syscall.
        let syscall_map = default_syscall_map();
        let max_syscall_cycles =
            syscall_map.values().map(|syscall| syscall.num_extra_cycles()).max().unwrap_or(0);

        let subproof_verifier =
            context.subproof_verifier.unwrap_or_else(|| Arc::new(DefaultSubproofVerifier::new()));
        let hook_registry = context.hook_registry.unwrap_or_default();

        Self {
            record,
            records: vec![],
            state: ExecutionState::new(program.pc_start),
            program,
            memory_accesses: MemoryAccessRecord::default(),
            shard_size: (opts.shard_size as u32) * 4,
            shard_batch_size: opts.shard_batch_size as u32,
            cycle_tracker: HashMap::new(),
            io_buf: HashMap::new(),
            trace_buf,
            unconstrained: false,
            unconstrained_state: ForkState::default(),
            syscall_map,
            executor_mode: ExecutorMode::Trace,
            max_syscall_cycles,
            report: ExecutionReport::default(),
            print_report: false,
            subproof_verifier,
            hook_registry,
            opts,
            max_cycles: context.max_cycles,
            memory_checkpoint: PagedMemory::new_preallocated(),
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
        runtime
    }

  

   
    /// Get the current value of a word.
    #[must_use]
    pub fn word(&mut self, addr: u32) -> u32 {
        #[allow(clippy::single_match_else)]
        let record = self.state.memory.get(addr);

        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match record {
                Some(record) => {
                    self.memory_checkpoint.entry(addr).or_insert_with(|| Some(*record));
                }
                None => {
                    self.memory_checkpoint.entry(addr).or_insert(None);
                }
            }
        }

        match record {
            Some(record) => record.value,
            None => 0,
        }
    }

    /// Get the current value of a byte.
    #[must_use]
    pub fn byte(&mut self, addr: u32) -> u8 {
        let word = self.word(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    /// Get the current timestamp for a given memory access position.
    #[must_use]
    pub const fn timestamp(&self, position: &MemoryAccessPosition) -> u32 {
        self.state.clk + *position as u32
    }

    /// Get the current shard.
    #[must_use]
    #[inline]
    pub fn shard(&self) -> u32 {
        self.state.current_shard
    }

    /// Get the current channel.
    #[must_use]
    #[inline]
    pub fn channel(&self) -> u8 {
        self.state.channel
    }

    /// Read a word from memory and create an access record.
    pub fn mr(&mut self, addr: u32, shard: u32, timestamp: u32) -> MemoryReadRecord {
        // Get the memory record entry.
        let entry = self.state.memory.entry(addr);
        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.entry(addr).or_insert(None);
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
                let value = self.state.uninitialized_memory.get(&addr).unwrap_or(&0);
                entry.insert(MemoryRecord { value: *value, shard: 0, timestamp: 0 })
            }
        };
        let value = record.value;
        let prev_shard = record.shard;
        let prev_timestamp = record.timestamp;
        record.shard = shard;
        record.timestamp = timestamp;

        // Construct the memory read record.
        MemoryReadRecord::new(value, shard, timestamp, prev_shard, prev_timestamp)
    }

    /// Write a word to memory and create an access record.
    pub fn mw(&mut self, addr: u32, value: u32, shard: u32, timestamp: u32) -> MemoryWriteRecord {
        // Get the memory record entry.
        let entry = self.state.memory.entry(addr);
        if self.executor_mode == ExecutorMode::Checkpoint || self.unconstrained {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.entry(addr).or_insert(None);
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
                let value = self.state.uninitialized_memory.get(&addr).unwrap_or(&0);

                entry.insert(MemoryRecord { value: *value, shard: 0, timestamp: 0 })
            }
        };
        let prev_value = record.value;
        let prev_shard = record.shard;
        let prev_timestamp = record.timestamp;
        record.value = value;
        record.shard = shard;
        record.timestamp = timestamp;

        // Construct the memory write record.
        MemoryWriteRecord::new(value, shard, timestamp, prev_value, prev_shard, prev_timestamp)
    }

  

 

    /// Emit a CPU event.
    #[allow(clippy::too_many_arguments)]
    fn emit_cpu(
        &mut self,
        shard: u32,
        channel: u8,
        clk: u32,
        next_clk:u32,
        pc: u32,
        next_pc: u32,
        sp: u32,
        next_sp:u32,
        instruction: Instruction,
        
        exec_record:ExecMemoryRecords,
        
        exit_code: u32,
        lookup_id: LookupId,
        syscall_lookup_id: LookupId,
    ) {
        let cpu_event = CpuEvent {
            shard,
            channel,
            clk,
            next_clk,
            pc,
            next_pc,
            instruction,
            sp,
            next_sp,
            exec_memory_list:exec_record,
            exit_code,
            alu_lookup_id: lookup_id,
            syscall_lookup_id,
            memory_add_lookup_id: create_alu_lookup_id(),
            memory_sub_lookup_id: create_alu_lookup_id(),
            branch_lt_lookup_id: create_alu_lookup_id(),
            branch_gt_lookup_id: create_alu_lookup_id(),
            branch_add_lookup_id: create_alu_lookup_id(),
            jump_jal_lookup_id: create_alu_lookup_id(),
            jump_jalr_lookup_id: create_alu_lookup_id(),
            auipc_lookup_id: create_alu_lookup_id(),
            
        };

        self.record.cpu_events.push(cpu_event);
    }

    /// Emit an ALU event.
    fn emit_alu(&mut self, clk: u32, opcode: Opcode, a: u32, b: u32, c: u32, lookup_id: LookupId) {
        let event = AluEvent {
            lookup_id,
            shard: self.shard(),
            clk,
            channel: self.channel(),
            opcode,
            a,
            b,
            c,
            sub_lookups: create_alu_lookups(),
        };
        match opcode {
            Opcode::ADD => {
                self.record.add_events.push(event);
            }
            Opcode::SUB => {
                self.record.sub_events.push(event);
            }
            Opcode::XOR | Opcode::OR | Opcode::AND => {
                self.record.bitwise_events.push(event);
            }
            Opcode::SLL => {
                self.record.shift_left_events.push(event);
            }
            Opcode::SRL | Opcode::SRA => {
                self.record.shift_right_events.push(event);
            }
            Opcode::SLT | Opcode::SLTU => {
                self.record.lt_events.push(event);
            }
            Opcode::MUL | Opcode::MULHU | Opcode::MULHSU | Opcode::MULH => {
                self.record.mul_events.push(event);
            }
            Opcode::DIVU | Opcode::REMU | Opcode::DIV | Opcode::REM => {
                self.record.divrem_events.push(event);
            }
            _ => {}
        }
    }

    fn fetch_binary32_args_from_stack(&mut self,sp:u32)->(MemoryReadRecord,MemoryReadRecord){
        let x_record = self.mr(sp, self.shard(),self.state.clk);

        self.state.clk+=1;
        let y_record = self.mr(sp-1, self.shard(),self.state.clk);
        (x_record,y_record)
    }

    fn update_stack_after_binary32(&mut self,res:u32)->MemoryWriteRecord{
        self.state.clk+=1;
        let sp = self.state.sp;
        self.state.sp-=1;
        let next_sp = self.state.sp;
        self.mw(next_sp, res, self.shard(), self.state.clk)
    }
    /// Fetch the instruction at the current program counter.
    fn fetch(&self) -> Instruction {
        let idx = ((self.state.pc - self.program.pc_base) / 4) as usize;
        self.program.instructions[idx]
    }

    /// Execute the given instruction over the current state of the runtime.
    #[allow(clippy::too_many_lines)]
    fn execute_instruction(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let mut pc = self.state.pc;
        let mut clk = self.state.clk;
        let mut sp = self.state.sp;
        let mut next_sp = self.state.sp;
        let mut exit_code = 0u32;

        let mut next_pc = self.state.pc.wrapping_add(4);

        let mut arg1:u32; let mut arg1_hi:u32;
        let mut arg2:u32; let mut arg2_hi:u32;
        let mut arg3:u32; let mut arg3_hi:u32;
       
       
        if self.executor_mode == ExecutorMode::Trace {
            self.memory_accesses = MemoryAccessRecord::default();
        }
        let lookup_id = if self.executor_mode == ExecutorMode::Trace {
            create_alu_lookup_id()
        } else {
            LookupId::default()
        };
        let syscall_lookup_id = if self.executor_mode == ExecutorMode::Trace {
            create_alu_lookup_id()
        } else {
            LookupId::default()
        };

        // if self.print_report && !self.unconstrained {
        //     self.report.opcode_counts[instruction] += 1;
        // }
        //TODO: fix report
        let arg1_record:MemoryReadRecord;
        let arg2_record:MemoryReadRecord;
        let arg3_record:MemoryReadRecord;
        let res_record:MemoryWriteRecord;
        let exec_memory_records = ExecMemoryRecords::new();
        match instruction{
            // Arithmetic instructions.
            Instruction::I32Add=>{
                (arg1_record,arg2_record) = self.fetch_binary32_args_from_stack(sp);
                let arg1 = arg1_record.value;
                let arg2 = arg2_record.value;
                let res = arg1.wrapping_add(arg2);
                res_record = self.update_stack_after_binary32(res)
            }
           
            Instruction::Call(syscall_id)=>{
                let syscall_id = syscall_id.to_u32();
                  // System instructions.
             {
                // We peek at register x5 to get the syscall id. The reason we don't `self.rr` this
                // register is that we write to it later.
               
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

                let syscall_impl = self.get_syscall(syscall).cloned();
                let mut precompile_rt = SyscallContext::new(self);
                precompile_rt.syscall_lookup_id = syscall_lookup_id;
                let (precompile_next_pc, precompile_cycles, returned_exit_code) =
                    if let Some(syscall_impl) = syscall_impl {
                        // Executing a syscall optionally returns a value to write to the t0
                        // register. If it returns None, we just keep the
                        // syscall_id in t0.
                        
                      
                        // If the syscall is `HALT` and the exit code is non-zero, return an error.
                        if syscall == SyscallCode::HALT && precompile_rt.exit_code != 0 {
                            return Err(ExecutionError::HaltWithNonZeroExitCode(
                                precompile_rt.exit_code,
                            ));
                        }

                        (
                            precompile_rt.next_pc,
                            syscall_impl.num_extra_cycles(),
                            precompile_rt.exit_code,
                        )
                    } else {
                        return Err(ExecutionError::UnsupportedSyscall(syscall_id));
                    };

                // Allow the syscall impl to modify state.clk/pc (exit unconstrained does this)
                clk = self.state.clk;
                pc = self.state.pc;
                sp = self.state.sp;
                next_sp = self.state.sp;
                // self.rw(t0, a); TODO writeback syscall result to top of stack
                next_pc = precompile_next_pc;
                self.state.clk += precompile_cycles;
                exit_code = returned_exit_code;

                // Update the syscall counts.
                let syscall_for_count = syscall.count_map();
                let syscall_count = self.state.syscall_counts.entry(syscall_for_count).or_insert(0);
                let (threshold, multiplier) = match syscall_for_count {
                    SyscallCode::KECCAK_PERMUTE => (self.opts.split_opts.keccak, 24),
                    SyscallCode::SHA_EXTEND => (self.opts.split_opts.sha_extend, 48),
                    SyscallCode::SHA_COMPRESS => (self.opts.split_opts.sha_compress, 80),
                    _ => (self.opts.split_opts.deferred, 1),
                };
                let nonce = (((*syscall_count as usize) % threshold) * multiplier) as u32;
                self.record.nonce_lookup.insert(syscall_lookup_id, nonce);
                *syscall_count += 1;
            }
            }
            Instruction::LocalGet(local_depth) => todo!(),
            Instruction::LocalSet(local_depth) => todo!(),
            Instruction::LocalTee(local_depth) => todo!(),
            Instruction::BrIfEqz(branch_offset) => todo!(),
            Instruction::BrIfNez(branch_offset) => todo!(),
            Instruction::BrAdjust(branch_offset) => todo!(),
            Instruction::BrAdjustIfNez(branch_offset) => todo!(),
            Instruction::BrTable(branch_table_targets) => todo!(),
            Instruction::Unreachable => todo!(),
            Instruction::ConsumeFuel(block_fuel) => todo!(),
            Instruction::Return(drop_keep) => todo!(),
            Instruction::ReturnIfNez(drop_keep) => todo!(),
            Instruction::ReturnCallInternal(compiled_func) => todo!(),
            Instruction::ReturnCall(func_idx) => todo!(),
            Instruction::ReturnCallIndirect(signature_idx) => todo!(),
            Instruction::CallInternal(compiled_func) => todo!(),
            Instruction::Call(func_idx) => todo!(),
            Instruction::CallIndirect(signature_idx) => todo!(),
            Instruction::SignatureCheck(signature_idx) => todo!(),
            Instruction::Drop => todo!(),
            Instruction::Select => todo!(),
            Instruction::GlobalGet(global_idx) => todo!(),
            Instruction::GlobalSet(global_idx) => todo!(),
            Instruction::I32Load(address_offset) => todo!(),
            Instruction::I64Load(address_offset) => todo!(),
            Instruction::F32Load(address_offset) => todo!(),
            Instruction::F64Load(address_offset) => todo!(),
            Instruction::I32Load8S(address_offset) => todo!(),
            Instruction::I32Load8U(address_offset) => todo!(),
            Instruction::I32Load16S(address_offset) => todo!(),
            Instruction::I32Load16U(address_offset) => todo!(),
            Instruction::I64Load8S(address_offset) => todo!(),
            Instruction::I64Load8U(address_offset) => todo!(),
            Instruction::I64Load16S(address_offset) => todo!(),
            Instruction::I64Load16U(address_offset) => todo!(),
            Instruction::I64Load32S(address_offset) => todo!(),
            Instruction::I64Load32U(address_offset) => todo!(),
            Instruction::I32Store(address_offset) => todo!(),
            Instruction::I64Store(address_offset) => todo!(),
            Instruction::F32Store(address_offset) => todo!(),
            Instruction::F64Store(address_offset) => todo!(),
            Instruction::I32Store8(address_offset) => todo!(),
            Instruction::I32Store16(address_offset) => todo!(),
            Instruction::I64Store8(address_offset) => todo!(),
            Instruction::I64Store16(address_offset) => todo!(),
            Instruction::I64Store32(address_offset) => todo!(),
            Instruction::MemorySize => todo!(),
            Instruction::MemoryGrow => todo!(),
            Instruction::MemoryFill => todo!(),
            Instruction::MemoryCopy => todo!(),
            Instruction::MemoryInit(data_segment_idx) => todo!(),
            Instruction::DataDrop(data_segment_idx) => todo!(),
            Instruction::TableSize(table_idx) => todo!(),
            Instruction::TableGrow(table_idx) => todo!(),
            Instruction::TableFill(table_idx) => todo!(),
            Instruction::TableGet(table_idx) => todo!(),
            Instruction::TableSet(table_idx) => todo!(),
            Instruction::TableCopy(table_idx) => todo!(),
            Instruction::TableInit(element_segment_idx) => todo!(),
            Instruction::ElemDrop(element_segment_idx) => todo!(),
            Instruction::RefFunc(func_idx) => todo!(),
            Instruction::I32Const(untyped_value) => todo!(),
            Instruction::I64Const(untyped_value) => todo!(),
            Instruction::F32Const(untyped_value) => todo!(),
            Instruction::F64Const(untyped_value) => todo!(),
            Instruction::ConstRef(const_ref) => todo!(),
            Instruction::I32Eqz => todo!(),
            Instruction::I32Eq => todo!(),
            Instruction::I32Ne => todo!(),
            Instruction::I32LtS => todo!(),
            Instruction::I32LtU => todo!(),
            Instruction::I32GtS => todo!(),
            Instruction::I32GtU => todo!(),
            Instruction::I32LeS => todo!(),
            Instruction::I32LeU => todo!(),
            Instruction::I32GeS => todo!(),
            Instruction::I32GeU => todo!(),
            Instruction::I64Eqz => todo!(),
            Instruction::I64Eq => todo!(),
            Instruction::I64Ne => todo!(),
            Instruction::I64LtS => todo!(),
            Instruction::I64LtU => todo!(),
            Instruction::I64GtS => todo!(),
            Instruction::I64GtU => todo!(),
            Instruction::I64LeS => todo!(),
            Instruction::I64LeU => todo!(),
            Instruction::I64GeS => todo!(),
            Instruction::I64GeU => todo!(),
            Instruction::F32Eq => todo!(),
            Instruction::F32Ne => todo!(),
            Instruction::F32Lt => todo!(),
            Instruction::F32Gt => todo!(),
            Instruction::F32Le => todo!(),
            Instruction::F32Ge => todo!(),
            Instruction::F64Eq => todo!(),
            Instruction::F64Ne => todo!(),
            Instruction::F64Lt => todo!(),
            Instruction::F64Gt => todo!(),
            Instruction::F64Le => todo!(),
            Instruction::F64Ge => todo!(),
            Instruction::I32Clz => todo!(),
            Instruction::I32Ctz => todo!(),
            Instruction::I32Popcnt => todo!(),
            Instruction::I32Sub => todo!(),
            Instruction::I32Mul => todo!(),
            Instruction::I32DivS => todo!(),
            Instruction::I32DivU => todo!(),
            Instruction::I32RemS => todo!(),
            Instruction::I32RemU => todo!(),
            Instruction::I32And => todo!(),
            Instruction::I32Or => todo!(),
            Instruction::I32Xor => todo!(),
            Instruction::I32Shl => todo!(),
            Instruction::I32ShrS => todo!(),
            Instruction::I32ShrU => todo!(),
            Instruction::I32Rotl => todo!(),
            Instruction::I32Rotr => todo!(),
            Instruction::I64Clz => todo!(),
            Instruction::I64Ctz => todo!(),
            Instruction::I64Popcnt => todo!(),
            Instruction::I64Add => todo!(),
            Instruction::I64Sub => todo!(),
            Instruction::I64Mul => todo!(),
            Instruction::I64DivS => todo!(),
            Instruction::I64DivU => todo!(),
            Instruction::I64RemS => todo!(),
            Instruction::I64RemU => todo!(),
            Instruction::I64And => todo!(),
            Instruction::I64Or => todo!(),
            Instruction::I64Xor => todo!(),
            Instruction::I64Shl => todo!(),
            Instruction::I64ShrS => todo!(),
            Instruction::I64ShrU => todo!(),
            Instruction::I64Rotl => todo!(),
            Instruction::I64Rotr => todo!(),
            Instruction::F32Abs => todo!(),
            Instruction::F32Neg => todo!(),
            Instruction::F32Ceil => todo!(),
            Instruction::F32Floor => todo!(),
            Instruction::F32Trunc => todo!(),
            Instruction::F32Nearest => todo!(),
            Instruction::F32Sqrt => todo!(),
            Instruction::F32Add => todo!(),
            Instruction::F32Sub => todo!(),
            Instruction::F32Mul => todo!(),
            Instruction::F32Div => todo!(),
            Instruction::F32Min => todo!(),
            Instruction::F32Max => todo!(),
            Instruction::F32Copysign => todo!(),
            Instruction::F64Abs => todo!(),
            Instruction::F64Neg => todo!(),
            Instruction::F64Ceil => todo!(),
            Instruction::F64Floor => todo!(),
            Instruction::F64Trunc => todo!(),
            Instruction::F64Nearest => todo!(),
            Instruction::F64Sqrt => todo!(),
            Instruction::F64Add => todo!(),
            Instruction::F64Sub => todo!(),
            Instruction::F64Mul => todo!(),
            Instruction::F64Div => todo!(),
            Instruction::F64Min => todo!(),
            Instruction::F64Max => todo!(),
            Instruction::F64Copysign => todo!(),
            Instruction::I32WrapI64 => todo!(),
            Instruction::I32TruncF32S => todo!(),
            Instruction::I32TruncF32U => todo!(),
            Instruction::I32TruncF64S => todo!(),
            Instruction::I32TruncF64U => todo!(),
            Instruction::I64ExtendI32S => todo!(),
            Instruction::I64ExtendI32U => todo!(),
            Instruction::I64TruncF32S => todo!(),
            Instruction::I64TruncF32U => todo!(),
            Instruction::I64TruncF64S => todo!(),
            Instruction::I64TruncF64U => todo!(),
            Instruction::F32ConvertI32S => todo!(),
            Instruction::F32ConvertI32U => todo!(),
            Instruction::F32ConvertI64S => todo!(),
            Instruction::F32ConvertI64U => todo!(),
            Instruction::F32DemoteF64 => todo!(),
            Instruction::F64ConvertI32S => todo!(),
            Instruction::F64ConvertI32U => todo!(),
            Instruction::F64ConvertI64S => todo!(),
            Instruction::F64ConvertI64U => todo!(),
            Instruction::F64PromoteF32 => todo!(),
            Instruction::I32Extend8S => todo!(),
            Instruction::I32Extend16S => todo!(),
            Instruction::I64Extend8S => todo!(),
            Instruction::I64Extend16S => todo!(),
            Instruction::I64Extend32S => todo!(),
            Instruction::I32TruncSatF32S => todo!(),
            Instruction::I32TruncSatF32U => todo!(),
            Instruction::I32TruncSatF64S => todo!(),
            Instruction::I32TruncSatF64U => todo!(),
            Instruction::I64TruncSatF32S => todo!(),
            Instruction::I64TruncSatF32U => todo!(),
            Instruction::I64TruncSatF64S => todo!(),
            Instruction::I64TruncSatF64U => todo!(),
            Instruction::Br(branch_offset) => todo!(),
        
        }
        // Update the program counter.
        self.state.pc = next_pc;
        let next_clk = self.state.clk;
        
        let next_sp = self.state.sp;

        let channel = self.channel();
        
        // Update the channel to the next cycle.
        if !self.unconstrained {
            self.state.channel = (self.state.channel + 1) % NUM_BYTE_LOOKUP_CHANNELS;
        }

        // Emit the CPU event for this cycle.
        if self.executor_mode == ExecutorMode::Trace {
            self.emit_cpu(
                self.shard(),
                channel,
                clk,
                next_clk,
                pc,
                next_pc,
                sp,
                next_sp,
                *instruction,
               exec_memory_records,
             
                exit_code,
                lookup_id,
                syscall_lookup_id,
            );
        };
        Ok(())
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    fn execute_cycle(&mut self) -> Result<bool, ExecutionError> {
        // Fetch the instruction at the current program counter.
        let instruction = self.fetch();

        // Log the current state of the runtime.
        self.log(&instruction);

        // Execute the instruction.
        self.execute_instruction(&instruction)?;

        // Increment the clock.
        self.state.global_clk += 1;

        // If there's not enough cycles left for another instruction, move to the next shard.
        // We multiply by 4 because clk is incremented by 4 for each normal instruction.
        if !self.unconstrained && self.max_syscall_cycles + self.state.clk >= self.shard_size {
            self.state.current_shard += 1;
            self.state.clk = 0;
            self.state.channel = 0;

            self.bump_record();
        }

        // If the cycle limit is exceeded, return an error.
        if let Some(max_cycles) = self.max_cycles {
            if self.state.global_clk >= max_cycles {
                return Err(ExecutionError::ExceededCycleLimit(max_cycles));
            }
        }

        let done = self.state.pc == 0
            || self.state.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32;
        if done && self.unconstrained {
            log::error!("program ended in unconstrained mode at clk {}", self.state.global_clk);
            return Err(ExecutionError::EndInUnconstrained());
        }
        Ok(done)
    }

    /// Bump the record.
    pub fn bump_record(&mut self) {
        let removed_record =
            std::mem::replace(&mut self.record, ExecutionRecord::new(self.program.clone()));
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
    pub fn execute_record(&mut self) -> Result<(Vec<ExecutionRecord>, bool), ExecutionError> {
        self.executor_mode = ExecutorMode::Trace;
        self.print_report = true;
        let done = self.execute()?;
        Ok((std::mem::take(&mut self.records), done))
    }

    /// Execute up to `self.shard_batch_size` cycles, returning the checkpoint from before execution
    /// and whether the program ended.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn execute_state(&mut self) -> Result<(ExecutionState, bool), ExecutionError> {
        self.memory_checkpoint.clear();
        self.executor_mode = ExecutorMode::Checkpoint;

        // Take memory out of state before cloning it so that memory is not cloned.
        let memory = std::mem::take(&mut self.state.memory);
        let mut checkpoint = tracing::info_span!("clone").in_scope(|| self.state.clone());
        self.state.memory = memory;

        let done = tracing::info_span!("execute").in_scope(|| self.execute())?;
        // Create a checkpoint using `memory_checkpoint`. Just include all memory if `done` since we
        // need it all for MemoryFinalize.
        tracing::info_span!("create memory checkpoint").in_scope(|| {
            let memory_checkpoint = std::mem::take(&mut self.memory_checkpoint);
            if done {
                // If we're done, we need to include all memory. But we need to reset any modified
                // memory to as it was before the execution.
                checkpoint.memory.clone_from(&self.state.memory);
                memory_checkpoint.into_iter().for_each(|(addr, record)| {
                    if let Some(record) = record {
                        checkpoint.memory.insert(addr, record);
                    } else {
                        checkpoint.memory.remove(addr);
                    }
                });
            } else {
                checkpoint.memory = memory_checkpoint
                    .into_iter()
                    .filter_map(|(addr, record)| record.map(|record| (addr, record)))
                    .collect();
            }
        });
        Ok((checkpoint, done))
    }

    fn initialize(&mut self) {
        self.state.clk = 0;
        self.state.channel = 0;

        tracing::debug!("loading memory image");
        for (&addr, value) in &self.program.memory_image {
            self.state.memory.insert(addr, MemoryRecord { value: *value, shard: 0, timestamp: 0 });
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
        Ok(())
    }

    /// Executes up to `self.shard_batch_size` cycles of the program, returning whether the program
    /// has finished.
    fn execute(&mut self) -> Result<bool, ExecutionError> {
        // Get the program.
        let program = self.program.clone();

        // Get the current shard.
        let start_shard = self.state.current_shard;

        // If it's the first cycle, initialize the program.
        if self.state.global_clk == 0 {
            self.initialize();
        }

        // Loop until we've executed `self.shard_batch_size` shards if `self.shard_batch_size` is
        // set.
        let mut done = false;
        let mut current_shard = self.state.current_shard;
        let mut num_shards_executed = 0;
        loop {
            if self.execute_cycle()? {
                done = true;
                break;
            }

            if self.shard_batch_size > 0 && current_shard != self.state.current_shard {
                num_shards_executed += 1;
                current_shard = self.state.current_shard;
                if num_shards_executed == self.shard_batch_size {
                    break;
                }
            }
        }

        // Get the final public values.
        let public_values = self.record.public_values;

        // Push the remaining execution record, if there are any CPU events.
        if !self.record.cpu_events.is_empty() {
            self.bump_record();
        }

        if done {
            self.postprocess();

            // Push the remaining execution record with memory initialize & finalize events.
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
                        println!("stdout: {buf}");
                    }
                    2 => {
                        println!("stderr: {buf}");
                    }
                    _ => {}
                }
            }
        }

        // Flush trace buf
        if let Some(ref mut buf) = self.trace_buf {
            buf.flush().unwrap();
        }

        // Ensure that all proofs and input bytes were read, otherwise warn the user.
        // if self.state.proof_stream_ptr != self.state.proof_stream.len() {
        //     panic!(
        //         "Not all proofs were read. Proving will fail during recursion. Did you pass too
        // many proofs in or forget to call verify_sp1_proof?"     );
        // }
        if self.state.input_stream_ptr != self.state.input_stream.len() {
            tracing::warn!("Not all input bytes were read.");
        }

        // SECTION: Set up all MemoryInitializeFinalizeEvents needed for memory argument.
        let memory_finalize_events = &mut self.record.memory_finalize_events;

        // We handle the addr = 0 case separately, as we constrain it to be 0 in the first row
        // of the memory finalize table so it must be first in the array of events.
        let addr_0_record = self.state.memory.get(0);

        let addr_0_final_record = match addr_0_record {
            Some(record) => record,
            None => &MemoryRecord { value: 0, shard: 0, timestamp: 1 },
        };
        memory_finalize_events
            .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, addr_0_final_record));

        let memory_initialize_events = &mut self.record.memory_initialize_events;
        let addr_0_initialize_event =
            MemoryInitializeFinalizeEvent::initialize(0, 0, addr_0_record.is_some());
        memory_initialize_events.push(addr_0_initialize_event);

        // Count the number of touched memory addresses manually, since `PagedMemory` doesn't
        // already know its length.
        self.report.touched_memory_addresses = 0;
        for addr in self.state.memory.keys() {
            self.report.touched_memory_addresses += 1;
            if addr == 0 {
                // Handled above.
                continue;
            }

            // Program memory is initialized in the MemoryProgram chip and doesn't require any
            // events, so we only send init events for other memory addresses.
            if !self.record.program.memory_image.contains_key(&addr) {
                let initial_value = self.state.uninitialized_memory.get(&addr).unwrap_or(&0);
                memory_initialize_events.push(MemoryInitializeFinalizeEvent::initialize(
                    addr,
                    *initial_value,
                    true,
                ));
            }

            let record = *self.state.memory.get(addr).unwrap();
            memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record));
        }
    }

    fn get_syscall(&mut self, code: SyscallCode) -> Option<&Arc<dyn Syscall>> {
        self.syscall_map.get(&code)
    }

    #[inline]
    fn log(&mut self, _: &Instruction) {
        // Write the current program counter to the trace buffer for the cycle tracer.
        if let Some(ref mut buf) = self.trace_buf {
            if !self.unconstrained {
                buf.write_all(&u32::to_be_bytes(self.state.pc)).unwrap();
            }
        }

        if !self.unconstrained && self.state.global_clk % 10_000_000 == 0 {
            log::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc);
        }
    }
}

impl Default for ExecutorMode {
    fn default() -> Self {
        Self::Simple
    }
}

// TODO: FIX
/// Aligns an address to the nearest word below or equal to it.
#[must_use]
pub const fn align(addr: u32) -> u32 {
    addr - addr % 4
}

// TODO: FIX
/// The number of different byte lookup channels.
pub const NUM_BYTE_LOOKUP_CHANNELS: u8 = 16;
