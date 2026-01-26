pub use rwasm_chips::*;

use core::fmt;

use hashbrown::{HashMap, HashSet};
use p3_field::PrimeField32;
use rwasm_executor::{ExecutionRecord, Program, RwasmAirId};
use sp1_curves::weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField};
use sp1_stark::{
    air::{InteractionScope, MachineAir, SP1_PROOF_NUM_PV_ELTS},
    Chip, InteractionKind, StarkGenericConfig, StarkMachine,
};
use strum_macros::{EnumDiscriminants, EnumIter};

use crate::{
    alu::{AddMul64Chip, TrailingChip},
    bytes::trace::NUM_ROWS as BYTE_CHIP_NUM_ROWS,
    control_flow::{BranchChip, CallChip},
    global::GlobalChip,
    memory::{MemoryChipType, MemoryInstructionsChip, MemoryLocalChip},
    non_alu_instructions::{
        ConstChip, ExtendChip, LocalChip, ParamsCheckChip, TableGrowChip, TableInitChip,
    },
    shape::Shapeable,
    syscall::{
        instructions::SyscallInstrsChip,
        precompiles::fptower::{Fp2AddSubAssignChip, Fp2MulAssignChip, FpOpChip},
    },
};

/// A module for importing all the different Rwasm chips.
pub(crate) mod rwasm_chips {
    pub use crate::{
        alu::{
            AddSubChip, BitwiseChip, DivRemChip, LtChip, MulChip, RotateChip, ShiftLeft,
            ShiftRightChip,
        },
        bytes::ByteChip,
        cpu::CpuChip,
        memory::MemoryGlobalChip,
        program::ProgramChip,
        syscall::{
            chip::SyscallChip,
            precompiles::{
                edwards::{EdAddAssignChip, EdDecompressChip},
                keccak256::KeccakPermuteChip,
                sha256::{ShaCompressChip, ShaExtendChip},
                u256x2048_mul::U256x2048MulChip,
                uint256::Uint256MulChip,
                weierstrass::{
                    WeierstrassAddAssignChip, WeierstrassDecompressChip,
                    WeierstrassDoubleAssignChip,
                },
            },
        },
    };
    pub use sp1_curves::{
        edwards::{ed25519::Ed25519Parameters, EdwardsCurve},
        weierstrass::{
            bls12_381::Bls12381Parameters, bn254::Bn254Parameters, secp256k1::Secp256k1Parameters,
            secp256r1::Secp256r1Parameters, SwCurve,
        },
    };
}

/// The maximum log number of shards in core.
pub const MAX_LOG_NUMBER_OF_SHARDS: usize = 16;

/// The maximum number of shards in core.
pub const MAX_NUMBER_OF_SHARDS: usize = 1 << MAX_LOG_NUMBER_OF_SHARDS;

/// An AIR for encoding Rwasm execution.
///
/// This enum contains all the different AIRs that are used in the Sp1 Rwasm IOP. Each variant is
/// a different AIR that is used to encode a different part of the Rwasm execution, and the
/// different AIR variants have a joint lookup argument.
#[derive(sp1_derive::MachineAirRwasm, EnumDiscriminants)]
#[strum_discriminants(derive(Hash, EnumIter))]
pub enum RwasmAir<F: PrimeField32> {
    /// An AIR that contains a preprocessed program table and a lookup for the instructions.
    Program(ProgramChip),
    /// An AIR for the RwasmCPU. Each row represents a cpu cycle.
    Cpu(CpuChip),
    /// An AIR for the RWASM Add and SUB instruction.
    Add(AddSubChip),
    /// An AIR for Rwasm Bitwise instructions.
    Bitwise(BitwiseChip),
    /// An AIR for Rwasm Mul instruction.
    Mul(MulChip),
    /// An AIR for Rwasm Add64 and Mul64 instructions.
    AddMul64(AddMul64Chip),
    /// An AIR for Rwasm Div and Rem instructions.
    DivRem(DivRemChip),
    /// An AIR for Rwasm Lt instruction.
    Lt(LtChip),
    /// An AIR for Rwasm SLL instruction.
    ShiftLeft(ShiftLeft),
    /// An AIR for Rwasm SRL and SRA instruction.
    ShiftRight(ShiftRightChip),
    /// An AIR for WASM Rotl, Rotr instruction.
    Rotate(RotateChip),
    /// An AIR for WASM Trailing and Popcnt instructions.
    Trailing(TrailingChip),
    /// An AIR for WASM Extend instructions.
    Extend(ExtendChip),
    /// An AIR for Rwasm memory instructions.
    Memory(MemoryInstructionsChip),
    /// An AIR for Rwasm branch instructions.
    Branch(BranchChip),
    /// An AIR for Rwasm branch instructions.
    Call(CallChip),
    /// An AIR for Rwasm ecall instructions.
    SyscallInstrs(SyscallInstrsChip),
    /// A lookup table for byte operations.
    ByteLookup(ByteChip<F>),
    /// A table for initializing the global memory state.
    MemoryGlobalInit(MemoryGlobalChip),
    /// A table for finalizing the global memory state.
    MemoryGlobalFinal(MemoryGlobalChip),
    /// A table for the local memory state.
    MemoryLocal(MemoryLocalChip),
    /// A table for all the syscall invocations.
    SyscallCore(SyscallChip),
    /// A table for all the precompile invocations.
    SyscallPrecompile(SyscallChip),
    /// A table for all the global interactions.
    Global(GlobalChip),
    /// A precompile for sha256 extend.
    Sha256Extend(ShaExtendChip),
    /// A precompile for sha256 compress.
    Sha256Compress(ShaCompressChip),
    /// A precompile for addition on the Elliptic curve ed25519.
    Ed25519Add(EdAddAssignChip<EdwardsCurve<Ed25519Parameters>>),
    /// A precompile for decompressing a point on the Edwards curve ed25519.
    Ed25519Decompress(EdDecompressChip<Ed25519Parameters>),
    /// A precompile for decompressing a point on the K256 curve.
    K256Decompress(WeierstrassDecompressChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for decompressing a point on the P256 curve.
    P256Decompress(WeierstrassDecompressChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256k1.
    Secp256k1Add(WeierstrassAddAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256k1.
    Secp256k1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256r1.
    Secp256r1Add(WeierstrassAddAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256r1.
    Secp256r1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for the Keccak permutation.
    KeccakP(KeccakPermuteChip),
    /// A precompile for addition on the Elliptic curve bn254.
    Bn254Add(WeierstrassAddAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bn254.
    Bn254Double(WeierstrassDoubleAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for addition on the Elliptic curve bls12_381.
    Bls12381Add(WeierstrassAddAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bls12_381.
    Bls12381Double(WeierstrassDoubleAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for uint256 mul.
    Uint256Mul(Uint256MulChip),
    /// A precompile for u256x2048 mul.
    U256x2048Mul(U256x2048MulChip),
    /// A precompile for decompressing a point on the BLS12-381 curve.
    Bls12381Decompress(WeierstrassDecompressChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for BLS12-381 fp operation.
    Bls12381Fp(FpOpChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 multiplication.
    Bls12381Fp2Mul(Fp2MulAssignChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 addition/subtraction.
    Bls12381Fp2AddSub(Fp2AddSubAssignChip<Bls12381BaseField>),
    /// A precompile for BN-254 fp operation.
    Bn254Fp(FpOpChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 multiplication.
    Bn254Fp2Mul(Fp2MulAssignChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 addition/subtraction.
    Bn254Fp2AddSub(Fp2AddSubAssignChip<Bn254BaseField>),

    TableInit(TableInitChip),

    TableGrow(TableGrowChip),

    Const(ConstChip),
    Local(LocalChip),

    ParamsCheck(ParamsCheckChip),
}

impl<F: PrimeField32> RwasmAir<F> {
    pub fn id(&self) -> RwasmAirId {
        RwasmAirId::from(RwasmAirDiscriminants::from(self))
    }

    pub fn machine<SC: StarkGenericConfig<Val = F>>(config: SC) -> StarkMachine<SC, Self> {
        let chips = Self::chips();
        StarkMachine::new(config, chips, SP1_PROOF_NUM_PV_ELTS, true)
    }

    /// Get all the different Rwasm AIRs.
    pub fn chips() -> Vec<Chip<F, Self>> {
        let (chips, _) = Self::get_chips_and_costs();
        chips
    }

    /// Get all the costs of the different Rwasm AIRs.
    pub fn costs() -> HashMap<String, u64> {
        let (_, costs) = Self::get_chips_and_costs();
        costs
    }

    /// Get all the different Rwasm AIRs and their costs.
    pub fn get_airs_and_costs() -> (Vec<Self>, HashMap<String, u64>) {
        let (chips, costs) = Self::get_chips_and_costs();
        (chips.into_iter().map(|chip| chip.into_inner()).collect(), costs)
    }

    /// Get all the different Rwasm chips and their costs.
    pub fn get_chips_and_costs() -> (Vec<Chip<F, Self>>, HashMap<String, u64>) {
        let mut costs: HashMap<String, u64> = HashMap::new();

        // The order of the chips is used to determine the order of trace generation.
        let mut chips = vec![];
        let cpu = Chip::new(RwasmAir::Cpu(CpuChip::default()));
        costs.insert(cpu.name(), cpu.cost());
        chips.push(cpu);

        let program = Chip::new(RwasmAir::Program(ProgramChip::default()));
        costs.insert(program.name(), program.cost());
        chips.push(program);

        let sha_extend = Chip::new(RwasmAir::Sha256Extend(ShaExtendChip::default()));
        costs.insert(sha_extend.name(), sha_extend.cost());
        chips.push(sha_extend);

        let sha_compress = Chip::new(RwasmAir::Sha256Compress(ShaCompressChip::default()));
        costs.insert(sha_compress.name(), sha_compress.cost());
        chips.push(sha_compress);

        let ed_add_assign = Chip::new(RwasmAir::Ed25519Add(EdAddAssignChip::<
            EdwardsCurve<Ed25519Parameters>,
        >::new()));
        costs.insert(ed_add_assign.name(), ed_add_assign.cost());
        chips.push(ed_add_assign);

        let ed_decompress = Chip::new(RwasmAir::Ed25519Decompress(EdDecompressChip::<
            Ed25519Parameters,
        >::default()));
        costs.insert(ed_decompress.name(), ed_decompress.cost());
        chips.push(ed_decompress);

        let k256_decompress = Chip::new(RwasmAir::K256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256k1Parameters>,
        >::with_lsb_rule()));
        costs.insert(k256_decompress.name(), k256_decompress.cost());
        chips.push(k256_decompress);

        let secp256k1_add_assign = Chip::new(RwasmAir::Secp256k1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256k1Parameters>,
        >::new()));
        costs.insert(secp256k1_add_assign.name(), secp256k1_add_assign.cost());
        chips.push(secp256k1_add_assign);

        let secp256k1_double_assign =
            Chip::new(RwasmAir::Secp256k1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256k1Parameters>,
            >::new()));
        costs.insert(secp256k1_double_assign.name(), secp256k1_double_assign.cost());
        chips.push(secp256k1_double_assign);

        let p256_decompress = Chip::new(RwasmAir::P256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256r1Parameters>,
        >::with_lsb_rule()));
        costs.insert(p256_decompress.name(), p256_decompress.cost());
        chips.push(p256_decompress);

        let secp256r1_add_assign = Chip::new(RwasmAir::Secp256r1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256r1Parameters>,
        >::new()));
        costs.insert(secp256r1_add_assign.name(), secp256r1_add_assign.cost());
        chips.push(secp256r1_add_assign);

        let secp256r1_double_assign =
            Chip::new(RwasmAir::Secp256r1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256r1Parameters>,
            >::new()));
        costs.insert(secp256r1_double_assign.name(), secp256r1_double_assign.cost());
        chips.push(secp256r1_double_assign);

        let keccak_permute = Chip::new(RwasmAir::KeccakP(KeccakPermuteChip::new()));
        costs.insert(keccak_permute.name(), keccak_permute.cost());
        chips.push(keccak_permute);

        let bn254_add_assign = Chip::new(RwasmAir::Bn254Add(WeierstrassAddAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_add_assign.name(), bn254_add_assign.cost());
        chips.push(bn254_add_assign);

        let bn254_double_assign = Chip::new(RwasmAir::Bn254Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_double_assign.name(), bn254_double_assign.cost());
        chips.push(bn254_double_assign);

        let bls12381_add = Chip::new(RwasmAir::Bls12381Add(WeierstrassAddAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_add.name(), bls12381_add.cost());
        chips.push(bls12381_add);

        let bls12381_double = Chip::new(RwasmAir::Bls12381Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_double.name(), bls12381_double.cost());
        chips.push(bls12381_double);

        let uint256_mul = Chip::new(RwasmAir::Uint256Mul(Uint256MulChip::default()));
        costs.insert(uint256_mul.name(), uint256_mul.cost());
        chips.push(uint256_mul);

        let u256x2048_mul = Chip::new(RwasmAir::U256x2048Mul(U256x2048MulChip::default()));
        costs.insert(u256x2048_mul.name(), u256x2048_mul.cost());
        chips.push(u256x2048_mul);

        let bls12381_fp = Chip::new(RwasmAir::Bls12381Fp(FpOpChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp.name(), bls12381_fp.cost());
        chips.push(bls12381_fp);

        let bls12381_fp2_addsub =
            Chip::new(RwasmAir::Bls12381Fp2AddSub(Fp2AddSubAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_addsub.name(), bls12381_fp2_addsub.cost());
        chips.push(bls12381_fp2_addsub);

        let bls12381_fp2_mul =
            Chip::new(RwasmAir::Bls12381Fp2Mul(Fp2MulAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_mul.name(), bls12381_fp2_mul.cost());
        chips.push(bls12381_fp2_mul);

        let bn254_fp = Chip::new(RwasmAir::Bn254Fp(FpOpChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp.name(), bn254_fp.cost());
        chips.push(bn254_fp);

        let bn254_fp2_addsub =
            Chip::new(RwasmAir::Bn254Fp2AddSub(Fp2AddSubAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_addsub.name(), bn254_fp2_addsub.cost());
        chips.push(bn254_fp2_addsub);

        let bn254_fp2_mul =
            Chip::new(RwasmAir::Bn254Fp2Mul(Fp2MulAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_mul.name(), bn254_fp2_mul.cost());
        chips.push(bn254_fp2_mul);

        let bls12381_decompress =
            Chip::new(RwasmAir::Bls12381Decompress(WeierstrassDecompressChip::<
                SwCurve<Bls12381Parameters>,
            >::with_lexicographic_rule()));
        costs.insert(bls12381_decompress.name(), bls12381_decompress.cost());
        chips.push(bls12381_decompress);

        let syscall_core = Chip::new(RwasmAir::SyscallCore(SyscallChip::core()));
        costs.insert(syscall_core.name(), syscall_core.cost());
        chips.push(syscall_core);

        let syscall_precompile = Chip::new(RwasmAir::SyscallPrecompile(SyscallChip::precompile()));
        costs.insert(syscall_precompile.name(), syscall_precompile.cost());
        chips.push(syscall_precompile);

        let div_rem = Chip::new(RwasmAir::DivRem(DivRemChip::default()));
        costs.insert(div_rem.name(), div_rem.cost());
        chips.push(div_rem);

        let rotate = Chip::new(RwasmAir::Rotate(RotateChip::default()));
        costs.insert(rotate.name(), rotate.cost());
        chips.push(rotate);

        let trailing = Chip::new(RwasmAir::Trailing(TrailingChip::default()));
        costs.insert(trailing.name(), trailing.cost());
        chips.push(trailing);

        let extend = Chip::new(RwasmAir::Extend(ExtendChip::default()));
        costs.insert(extend.name(), extend.cost());
        chips.push(extend);

        let add_sub = Chip::new(RwasmAir::Add(AddSubChip::default()));
        costs.insert(add_sub.name(), add_sub.cost());
        chips.push(add_sub);

        let bitwise = Chip::new(RwasmAir::Bitwise(BitwiseChip::default()));
        costs.insert(bitwise.name(), bitwise.cost());
        chips.push(bitwise);

        let mul = Chip::new(RwasmAir::Mul(MulChip::default()));
        costs.insert(mul.name(), mul.cost());
        chips.push(mul);

        let addmul64 = Chip::new(RwasmAir::AddMul64(AddMul64Chip::default()));
        costs.insert(addmul64.name(), addmul64.cost());
        chips.push(addmul64);

        let shift_right = Chip::new(RwasmAir::ShiftRight(ShiftRightChip::default()));
        costs.insert(shift_right.name(), shift_right.cost());
        chips.push(shift_right);

        let shift_left = Chip::new(RwasmAir::ShiftLeft(ShiftLeft::default()));
        costs.insert(shift_left.name(), shift_left.cost());
        chips.push(shift_left);

        let lt = Chip::new(RwasmAir::Lt(LtChip::default()));
        costs.insert(lt.name(), lt.cost());
        chips.push(lt);

        let memory_instructions = Chip::new(RwasmAir::Memory(MemoryInstructionsChip::default()));
        costs.insert(memory_instructions.name(), memory_instructions.cost());
        chips.push(memory_instructions);

        let branch = Chip::new(RwasmAir::Branch(BranchChip::default()));
        costs.insert(branch.name(), branch.cost());
        chips.push(branch);

        let call = Chip::new(RwasmAir::Call(CallChip::default()));
        costs.insert(call.name(), call.cost());
        chips.push(call);

        let syscall_instrs = Chip::new(RwasmAir::SyscallInstrs(SyscallInstrsChip::default()));
        costs.insert(syscall_instrs.name(), syscall_instrs.cost());
        chips.push(syscall_instrs);

        let memory_global_init = Chip::new(RwasmAir::MemoryGlobalInit(MemoryGlobalChip::new(
            MemoryChipType::Initialize,
        )));
        costs.insert(memory_global_init.name(), memory_global_init.cost());
        chips.push(memory_global_init);

        let memory_global_finalize =
            Chip::new(RwasmAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)));
        costs.insert(memory_global_finalize.name(), memory_global_finalize.cost());
        chips.push(memory_global_finalize);

        let memory_local = Chip::new(RwasmAir::MemoryLocal(MemoryLocalChip::new()));
        costs.insert(memory_local.name(), memory_local.cost());
        chips.push(memory_local);

        let global = Chip::new(RwasmAir::Global(GlobalChip));
        costs.insert(global.name(), global.cost());
        chips.push(global);

        let byte = Chip::new(RwasmAir::ByteLookup(ByteChip::default()));
        costs.insert(byte.name(), byte.cost());
        chips.push(byte);

        let table_init = Chip::new(RwasmAir::TableInit(TableInitChip::default()));
        costs.insert(table_init.name(), table_init.cost());
        chips.push(table_init);

        let table_grow = Chip::new(RwasmAir::TableGrow(TableGrowChip::default()));
        costs.insert(table_grow.name(), table_grow.cost());
        chips.push(table_grow);

        let const_chip = Chip::new(RwasmAir::Const(ConstChip::default()));
        costs.insert(const_chip.name(), const_chip.cost());
        chips.push(const_chip);

        let local_chip = Chip::new(RwasmAir::Local(LocalChip::default()));
        costs.insert(local_chip.name(), local_chip.cost());
        chips.push(local_chip);

        let params_check_chip = Chip::new(RwasmAir::ParamsCheck(ParamsCheckChip::default()));
        costs.insert(params_check_chip.name(), params_check_chip.cost());
        chips.push(params_check_chip);

        assert_eq!(chips.len(), costs.len(), "chips and costs must have the same length",);

        (chips, costs)
    }

    /// Get the heights of the preprocessed chips for a given program.
    pub(crate) fn preprocessed_heights(program: &Program) -> Vec<(RwasmAirId, usize)> {
        vec![
            (RwasmAirId::Program, program.module.code_section.len()),
            (RwasmAirId::Byte, BYTE_CHIP_NUM_ROWS),
        ]
    }

    /// Get the heights of the chips for a given execution record.
    pub fn core_heights(record: &ExecutionRecord) -> Vec<(RwasmAirId, usize)> {
        record.core_heights()
    }

    pub(crate) fn get_all_core_airs() -> Vec<Self> {
        vec![
            RwasmAir::Cpu(CpuChip::default()),
            RwasmAir::Add(AddSubChip::default()),
            RwasmAir::Bitwise(BitwiseChip::default()),
            RwasmAir::Mul(MulChip::default()),
            RwasmAir::AddMul64(AddMul64Chip::default()),
            RwasmAir::DivRem(DivRemChip::default()),
            RwasmAir::Lt(LtChip::default()),
            RwasmAir::ShiftLeft(ShiftLeft::default()),
            RwasmAir::ShiftRight(ShiftRightChip::default()),
            RwasmAir::Rotate(RotateChip::default()),
            RwasmAir::Trailing(TrailingChip::default()),
            RwasmAir::Extend(ExtendChip::default()),
            RwasmAir::Memory(MemoryInstructionsChip::default()),
            RwasmAir::Branch(BranchChip::default()),
            RwasmAir::Call(CallChip::default()),
            RwasmAir::SyscallInstrs(SyscallInstrsChip::default()),
            RwasmAir::MemoryLocal(MemoryLocalChip::new()),
            RwasmAir::Global(GlobalChip),
            RwasmAir::SyscallCore(SyscallChip::core()),
            RwasmAir::Const(ConstChip::default()),
            RwasmAir::Local(LocalChip::default()),
            RwasmAir::TableGrow(TableGrowChip::default()),
            RwasmAir::TableInit(TableInitChip::default()),
        ]
    }

    pub(crate) fn memory_init_final_airs() -> Vec<Self> {
        vec![
            RwasmAir::MemoryGlobalInit(MemoryGlobalChip::new(MemoryChipType::Initialize)),
            RwasmAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)),
            RwasmAir::Global(GlobalChip),
        ]
    }

    /// Returns the upper bound of the number of memory events per row of each precompile. Used in
    /// shape-fitting.
    pub(crate) fn precompile_airs_with_memory_events_per_row(
    ) -> impl Iterator<Item = (RwasmAirId, usize)> {
        let mut airs: HashSet<_> = Self::get_airs_and_costs().0.into_iter().collect();

        // Remove the core airs.
        for core_air in Self::get_all_core_airs() {
            airs.remove(&core_air);
        }

        // Remove the memory init/finalize airs.
        for memory_air in Self::memory_init_final_airs() {
            airs.remove(&memory_air);
        }

        // Remove the syscall, program, and byte lookup airs.
        airs.remove(&Self::SyscallPrecompile(SyscallChip::precompile()));
        airs.remove(&Self::Program(ProgramChip::default()));
        airs.remove(&Self::ByteLookup(ByteChip::default()));

        airs.into_iter().map(|air| {
            let chip = Chip::new(air);
            let local_mem_events_per_row: usize = chip
                .sends()
                .iter()
                .chain(chip.receives())
                .filter(|interaction| {
                    interaction.kind == InteractionKind::Memory &&
                        interaction.scope == InteractionScope::Local
                })
                .count();

            (chip.into_inner().id(), local_mem_events_per_row)
        })
    }
}

impl<F: PrimeField32> PartialEq for RwasmAir<F> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl<F: PrimeField32> Eq for RwasmAir<F> {}

impl<F: PrimeField32> core::hash::Hash for RwasmAir<F> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl<F: PrimeField32> fmt::Debug for RwasmAir<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl From<RwasmAirDiscriminants> for RwasmAirId {
    fn from(value: RwasmAirDiscriminants) -> Self {
        match value {
            RwasmAirDiscriminants::Program => RwasmAirId::Program,
            RwasmAirDiscriminants::Cpu => RwasmAirId::Cpu,
            RwasmAirDiscriminants::Add => RwasmAirId::AddSub,
            RwasmAirDiscriminants::Bitwise => RwasmAirId::Bitwise,
            RwasmAirDiscriminants::Mul => RwasmAirId::Mul,
            RwasmAirDiscriminants::AddMul64 => RwasmAirId::AddMul64,
            RwasmAirDiscriminants::DivRem => RwasmAirId::DivRem,
            RwasmAirDiscriminants::Lt => RwasmAirId::Lt,
            RwasmAirDiscriminants::ShiftLeft => RwasmAirId::ShiftLeft,
            RwasmAirDiscriminants::ShiftRight => RwasmAirId::ShiftRight,
            RwasmAirDiscriminants::Memory => RwasmAirId::MemoryInstrs,
            RwasmAirDiscriminants::Branch => RwasmAirId::Branch,
            RwasmAirDiscriminants::Call => RwasmAirId::Call,
            RwasmAirDiscriminants::SyscallInstrs => RwasmAirId::SyscallInstrs,
            RwasmAirDiscriminants::ByteLookup => RwasmAirId::Byte,
            RwasmAirDiscriminants::MemoryGlobalInit => RwasmAirId::MemoryGlobalInit,
            RwasmAirDiscriminants::MemoryGlobalFinal => RwasmAirId::MemoryGlobalFinalize,
            RwasmAirDiscriminants::MemoryLocal => RwasmAirId::MemoryLocal,
            RwasmAirDiscriminants::SyscallCore => RwasmAirId::SyscallCore,
            RwasmAirDiscriminants::SyscallPrecompile => RwasmAirId::SyscallPrecompile,
            RwasmAirDiscriminants::Global => RwasmAirId::Global,
            RwasmAirDiscriminants::Sha256Extend => RwasmAirId::ShaExtend,
            RwasmAirDiscriminants::Sha256Compress => RwasmAirId::ShaCompress,
            RwasmAirDiscriminants::Ed25519Add => RwasmAirId::EdAddAssign,
            RwasmAirDiscriminants::Ed25519Decompress => RwasmAirId::EdDecompress,
            RwasmAirDiscriminants::K256Decompress => RwasmAirId::Secp256k1Decompress,
            RwasmAirDiscriminants::P256Decompress => RwasmAirId::Secp256r1Decompress,
            RwasmAirDiscriminants::Secp256k1Add => RwasmAirId::Secp256k1AddAssign,
            RwasmAirDiscriminants::Secp256k1Double => RwasmAirId::Secp256k1DoubleAssign,
            RwasmAirDiscriminants::Secp256r1Add => RwasmAirId::Secp256r1AddAssign,
            RwasmAirDiscriminants::Secp256r1Double => RwasmAirId::Secp256r1DoubleAssign,
            RwasmAirDiscriminants::KeccakP => RwasmAirId::KeccakPermute,
            RwasmAirDiscriminants::Bn254Add => RwasmAirId::Bn254AddAssign,
            RwasmAirDiscriminants::Bn254Double => RwasmAirId::Bn254DoubleAssign,
            RwasmAirDiscriminants::Bls12381Add => RwasmAirId::Bls12381AddAssign,
            RwasmAirDiscriminants::Bls12381Double => RwasmAirId::Bls12381DoubleAssign,
            RwasmAirDiscriminants::Uint256Mul => RwasmAirId::Uint256MulMod,
            RwasmAirDiscriminants::U256x2048Mul => RwasmAirId::U256XU2048Mul,
            RwasmAirDiscriminants::Bls12381Decompress => RwasmAirId::Bls12381Decompress,
            RwasmAirDiscriminants::Bls12381Fp => RwasmAirId::Bls12381FpOpAssign,
            RwasmAirDiscriminants::Bls12381Fp2Mul => RwasmAirId::Bls12381Fp2MulAssign,
            RwasmAirDiscriminants::Bls12381Fp2AddSub => RwasmAirId::Bls12381Fp2AddSubAssign,
            RwasmAirDiscriminants::Bn254Fp => RwasmAirId::Bn254FpOpAssign,
            RwasmAirDiscriminants::Bn254Fp2Mul => RwasmAirId::Bn254Fp2MulAssign,
            RwasmAirDiscriminants::Bn254Fp2AddSub => RwasmAirId::Bn254Fp2AddSubAssign,
            RwasmAirDiscriminants::TableInit => RwasmAirId::TableInit,
            RwasmAirDiscriminants::TableGrow => RwasmAirId::TableGrow,
            RwasmAirDiscriminants::Rotate => RwasmAirId::Rotate,
            RwasmAirDiscriminants::Trailing => RwasmAirId::Trailing,
            RwasmAirDiscriminants::Extend => RwasmAirId::Extend,
            RwasmAirDiscriminants::Const => RwasmAirId::Const,
            RwasmAirDiscriminants::Local => RwasmAirId::Local,
            RwasmAirDiscriminants::ParamsCheck => RwasmAirId::ParamsCheck,
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
#[allow(clippy::print_stdout)]
pub mod tests {

    use crate::{
        io::SP1Stdin,
        rwasm::RwasmAir,
        utils::{self, run_test},
    };

    use crate::programs::tests::*;
    use hashbrown::HashMap;
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use rwasm_executor::RwasmAirId;
    use sp1_stark::{air::MachineAir, CpuProver};
    use strum::IntoEnumIterator;

    //TODO(Aliaksei): fix ignored tests

    #[test]
    #[ignore]
    fn test_primitives_and_machine_air_names_match() {
        let chips = RwasmAir::<BabyBear>::chips();
        for (a, b) in chips.iter().zip_eq(RwasmAirId::iter()) {
            assert_eq!(a.name(), b.to_string());
        }
    }

    #[test]
    #[ignore]
    fn core_air_cost_consistency() {
        // Load air costs from file
        let file = std::fs::File::open("../executor/src/artifacts/rv32im_costs.json").unwrap();
        let costs: HashMap<String, u64> = serde_json::from_reader(file).unwrap();
        // Compare with costs computed by machine
        let machine_costs = RwasmAir::<BabyBear>::costs();
        assert_eq!(costs, machine_costs);
    }

    #[test]
    #[ignore]
    fn write_core_air_costs() {
        let costs = RwasmAir::<BabyBear>::costs();
        println!("{:?}", costs);
        // write to file
        // Create directory if it doesn't exist
        let dir = std::path::Path::new("../executor/src/artifacts");
        if !dir.exists() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let file = std::fs::File::create(dir.join("rv32im_costs.json")).unwrap();
        serde_json::to_writer_pretty(file, &costs).unwrap();
    }

    #[test]
    fn test_simple_prove() {
        utils::setup_logger();
        let program = simple_program();
        let stdin = SP1Stdin::new();
        run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }

    // #[test]
    // fn test_shift_prove() {
    //     utils::setup_logger();
    //     let shift_ops = [Opcode::SRL, Opcode::SRA, Opcode::SLL];
    //     let operands =
    //         [(1, 1), (1234, 5678), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX), (u32::MAX,
    // 0)];     for shift_op in shift_ops.iter() {
    //         for op in operands.iter() {
    //             let instructions = vec![
    //                 Opcode::new(Opcode::ADD, 29, 0, op.0, false, true),
    //                 Opcode::new(Opcode::ADD, 30, 0, op.1, false, true),
    //                 Opcode::new(*shift_op, 31, 29, 3, false, false),
    //             ];
    //             let program = Program::from_instrs(instructions);
    //             let stdin = SP1Stdin::new();
    //             run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //         }
    //     }
    // }

    // #[test]
    // fn test_sub_prove() {
    //     utils::setup_logger();
    //     let instructions = vec![
    //         Opcode::new(Opcode::ADD, 29, 0, 5, false, true),
    //         Opcode::new(Opcode::ADD, 30, 0, 8, false, true),
    //         Opcode::new(Opcode::SUB, 31, 30, 29, false, false),
    //     ];
    //     let program = Program::new(instructions, 0, 0);
    //     let stdin = SP1Stdin::new();
    //     run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    // }

    // #[test]
    // fn test_add_prove() {
    //     setup_logger();
    //     let instructions = vec![
    //         Opcode::new(Opcode::ADD, 29, 0, 5, false, true),
    //         Opcode::new(Opcode::ADD, 30, 0, 8, false, true),
    //         Opcode::new(Opcode::ADD, 31, 30, 29, false, false),
    //     ];
    //     let program = Program::new(instructions, 0, 0);
    //     let stdin = SP1Stdin::new();
    //     run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    // }

    // #[test]
    // fn test_mul_prove() {
    //     let mul_ops = [Opcode::MUL, Opcode::MULH, Opcode::MULHU, Opcode::MULHSU];
    //     utils::setup_logger();
    //     let operands =
    //         [(1, 1), (1234, 5678), (8765, 4321), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX)];
    //     for mul_op in mul_ops.iter() {
    //         for operand in operands.iter() {
    //             let instructions = vec![
    //                 Opcode::new(Opcode::ADD, 29, 0, operand.0, false, true),
    //                 Opcode::new(Opcode::ADD, 30, 0, operand.1, false, true),
    //                 Opcode::new(*mul_op, 31, 30, 29, false, false),
    //             ];
    //             let program = Program::new(instructions, 0, 0);
    //             let stdin = SP1Stdin::new();
    //             run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //         }
    //     }
    // }

    // #[test]
    // fn test_lt_prove() {
    //     setup_logger();
    //     let less_than = [Opcode::SLT, Opcode::SLTU];
    //     for lt_op in less_than.iter() {
    //         let instructions = vec![
    //             Opcode::new(Opcode::ADD, 29, 0, 5, false, true),
    //             Opcode::new(Opcode::ADD, 30, 0, 8, false, true),
    //             Opcode::new(*lt_op, 31, 30, 29, false, false),
    //         ];
    //         let program = Program::new(instructions, 0, 0);
    //         let stdin = SP1Stdin::new();
    //         run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //     }
    // }

    // #[test]
    // fn test_bitwise_prove() {
    //     setup_logger();
    //     let bitwise_opcodes = [Opcode::XOR, Opcode::OR, Opcode::AND];

    //     for bitwise_op in bitwise_opcodes.iter() {
    //         let instructions = vec![
    //             Opcode::new(Opcode::ADD, 29, 0, 5, false, true),
    //             Opcode::new(Opcode::ADD, 30, 0, 8, false, true),
    //             Opcode::new(*bitwise_op, 31, 30, 29, false, false),
    //         ];
    //         let program = Program::new(instructions, 0, 0);
    //         let stdin = SP1Stdin::new();
    //         run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //     }
    // }

    // #[test]
    // fn test_divrem_prove() {
    //     setup_logger();
    //     let div_rem_ops = [
    //         Opcode::I32DivS,
    //         Opcode::I32DivSU,
    //         Opcode::I32RemS,
    //         Opcode::I32RemU,
    //     ];
    //     let operands = [
    //         (1, 1),
    //         (123, 456 * 789),
    //         (123 * 456, 789),
    //         (0xffff * (0xffff - 1), 0xffff),
    //         (u32::MAX - 5, u32::MAX - 7),
    //     ];
    //     for div_rem_op in div_rem_ops.iter() {
    //         for op in operands.iter() {
    //             let instructions = vec![
    //                 Opcode::new(Opcode::ADD, 29, 0, op.0, false, true),
    //                 Opcode::new(Opcode::ADD, 30, 0, op.1, false, true),
    //                 Opcode::new(*div_rem_op, 31, 29, 30, false, false),
    //             ];
    //             let program = Program::new(instructions, 0, 0);
    //             let stdin = SP1Stdin::new();
    //             run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //         }
    //     }
    // }

    // #[test]
    // fn test_fibonacci_prove_simple() {
    //     setup_logger();
    //     let program = fibonacci_program();
    //     let stdin = SP1Stdin::new();
    //     run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    // }

    // #[test]
    // fn test_fibonacci_prove_checkpoints() {
    //     setup_logger();

    //     let program = fibonacci_program();
    //     let stdin = SP1Stdin::new();
    //     let mut opts = SP1CoreOpts::default();
    //     opts.shard_size = 1024;
    //     opts.shard_batch_size = 2;

    //     let config = BabyBearPoseidon2::new();
    //     let machine = RwasmAir::machine(config);
    //     let prover = CpuProver::new(machine);
    //     let (pk, vk) = prover.setup(&program);
    //     prove_core::<_, _>(
    //         &prover,
    //         &pk,
    //         &vk,
    //         program,
    //         &stdin,
    //         opts,
    //         SP1Context::default(),
    //         None,
    //         None,
    //     )
    //     .unwrap();
    // }

    // // #[test]
    // // fn test_fibonacci_prove_batch() {
    // //     setup_logger();
    // //     let program = fibonacci_program();
    // //     let stdin = SP1Stdin::new();

    // //     let opts = SP1CoreOpts::default();
    // //     let config = BabyBearPoseidon2::new();
    // //     let machine = RiscvAir::machine(config);
    // //     let prover = CpuProver::new(machine);
    // //     let (pk, vk) = prover.setup(&program);
    // //     prove_core::<_, _>(
    // //         &prover,
    // //         &pk,
    // //         &vk,
    // //         program,
    // //         &stdin,
    // //         opts,
    // //         SP1Context::default(),
    // //         None,
    // //         None,
    // //     )
    // //     .unwrap();
    // // }

    // #[test]
    // fn test_simple_memory_program_prove() {
    //     setup_logger();
    //     let program = simple_memory_program();
    //     let stdin = SP1Stdin::new();
    //     run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    // }

    // #[test]
    // fn test_ssz_withdrawal() {
    //     setup_logger();
    //     let program = ssz_withdrawals_program();
    //     let stdin = SP1Stdin::new();
    //     run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    // }

    // #[test]
    // fn test_key_serde() {
    //     let program = ssz_withdrawals_program();
    //     let config = BabyBearPoseidon2::new();
    //     let machine = RwasmAir::machine(config);
    //     let (pk, vk) = machine.setup(&program);

    //     let serialized_pk = bincode::serialize(&pk).unwrap();
    //     let deserialized_pk: StarkProvingKey<BabyBearPoseidon2> =
    //         bincode::deserialize(&serialized_pk).unwrap();
    //     assert_eq!(pk.commit, deserialized_pk.commit);
    //     assert_eq!(pk.pc_start, deserialized_pk.pc_start);
    //     assert_eq!(pk.traces, deserialized_pk.traces);
    //     assert_eq!(pk.data.root(), deserialized_pk.data.root());
    //     assert_eq!(pk.chip_ordering, deserialized_pk.chip_ordering);
    //     assert_eq!(pk.local_only, deserialized_pk.local_only);

    //     let serialized_vk = bincode::serialize(&vk).unwrap();
    //     let deserialized_vk: StarkVerifyingKey<BabyBearPoseidon2> =
    //         bincode::deserialize(&serialized_vk).unwrap();
    //     assert_eq!(vk.commit, deserialized_vk.commit);
    //     assert_eq!(vk.pc_start, deserialized_vk.pc_start);
    //     assert_eq!(vk.chip_information.len(), deserialized_vk.chip_information.len());
    //     for (a, b) in vk.chip_information.iter().zip(deserialized_vk.chip_information.iter()) {
    //         assert_eq!(a.0, b.0);
    //         assert_eq!(a.1.log_n, b.1.log_n);
    //         assert_eq!(a.1.shift, b.1.shift);
    //         assert_eq!(a.2.height, b.2.height);
    //         assert_eq!(a.2.width, b.2.width);
    //     }
    //     assert_eq!(vk.chip_ordering, deserialized_vk.chip_ordering);
    // }
}
