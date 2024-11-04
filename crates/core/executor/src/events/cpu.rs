use serde::{Deserialize, Serialize};

use rwasm::engine::bytecode::Instruction;

use super::{memory::MemoryRecordEnum, LookupId, MemoryReadRecord,MemoryWriteRecord};

/// CPU Event.
///
/// This object encapsulates the information needed to prove a CPU operation. This includes its
/// shard, channel, opcode, operands, and other relevant information.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct CpuEvent {
    /// The shard number.
    pub shard: u32,
    /// The channel number.
    pub channel: u8,
    /// The clock cycle.
    pub clk: u32,
    /// The program counter.
    /// 
    pub next_clk:u32,
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,

    pub sp: u32,
    pub next_sp:u32,
    /// The instruction.
    pub instruction: Instruction,
    /// The first operand.
   
    pub exec_memory_list:ExecMemoryRecords,
    /// The exit code.
    pub exit_code: u32,
    /// The ALU lookup id.
    pub alu_lookup_id: LookupId,
    /// The syscall lookup id.
    pub syscall_lookup_id: LookupId,
    /// The memory add lookup id.
    pub memory_add_lookup_id: LookupId,
    /// The memory sub lookup id.
    pub memory_sub_lookup_id: LookupId,
    /// The branch gt lookup id.
    pub branch_gt_lookup_id: LookupId,
    /// The branch lt lookup id.
    pub branch_lt_lookup_id: LookupId,
    /// The branch add lookup id.
    pub branch_add_lookup_id: LookupId,
    /// The jump jal lookup id.
    pub jump_jal_lookup_id: LookupId,
    /// The jump jalr lookup id.
    pub jump_jalr_lookup_id: LookupId,
    /// The auipc lookup id.
    pub auipc_lookup_id: LookupId,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize,Default)]
pub struct ExecMemoryRecords{
    arg1: u32,
    arg1_record: Option<MemoryReadRecord>,
    arg2:u32,
    arg2_record:Option<MemoryReadRecord>,
    arg3:u32,
    arg3_record:Option<MemoryReadRecord>,
    res:u32,
    res_record:Option<MemoryWriteRecord>,
}
impl ExecMemoryRecords {
    pub fn new()->Self{
        ExecMemoryRecords{
            arg1: 0,
            arg1_record:None,
            arg2: 0,
            arg2_record: None,
            arg3:0,
            arg3_record:None,
            res: 0,
            res_record: None,
        }
    }
}