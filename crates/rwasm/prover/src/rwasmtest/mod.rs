mod addmul64;
mod addsub;
mod bitwise;
mod branch;
mod call;
mod comp;
mod divrem;
mod extend;
mod lt;
mod mul;
mod fuel;
mod rotate;
mod sr;
mod table_grow;
mod table_init;
mod trailing;
use rwasm_executor::Program;
use rwasm_machine::utils::setup_logger;

use super::*;

pub fn run_rwasm_prover(mut program: Program) {
    setup_logger();
    let prover: SP1Prover = SP1Prover::new();
    let mut opts = SP1ProverOpts::default();
    opts.core_opts.shard_batch_size = 1;
    let context = SP1Context::default();

    tracing::info!("setup elf");
    let (_, pk, vk) = prover.setup_program(&mut program);

    println!("setup_program {:?}", program);

    tracing::info!("prove core");
    let stdin = SP1Stdin::new();
    let core_proof = prover.prove_core(&pk, program.clone(), &stdin, opts, context);
    tracing::info!("prove core finish");
    match core_proof {
        Ok(_) => {
            tracing::info!("verify core");
            let result = prover.verify(&core_proof.unwrap().proof, &vk);
            match result {
                Ok(_) => (),
                Err(err) => {
                    println!("err:{}", err);
                    panic!();
                }
            }
        }
        Err(err) => {
            println!("{}", err);
        }
    }

    println!("done rwasm proof");
}
#[cfg(test)]
mod tests {

    use super::*;
    use rwasm_executor::{Opcode, Program, SP_START};

    fn build_elf() -> Program {
        let x_value: u32 = 0x11;
        let y_value: u32 = 0x23;
        let z1_value: u32 = 0x40;
        let z2_value: u32 = 0x37;
        let z3_value: u32 = 0x1800;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;

        let instructions = vec![
            Opcode::I32Const(z6_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Add,
            Opcode::I32Sub,
            Opcode::I32Mul,
            Opcode::I32DivS,
            Opcode::I32DivU,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_elf2() -> Program {
        let x_value: u32 = 0x11;
        let y_value: u32 = 0x23;
        let z1_value: u32 = 0x3;
        let z2_value: u32 = 0x37;
        let z3_value: u32 = 0x12;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x7;
        let z6_value: u32 = 0x21;
        let z7_value: u32 = 0x333333;
        let z8_value: u32 = 0x444444;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32Const(z6_value.into()),
            Opcode::I32Const(z7_value.into()),
            Opcode::I32Const(z8_value.into()),
            Opcode::I32Ne,
            Opcode::I32Eq,
            Opcode::I32GtS,
            Opcode::I32GtU,
            Opcode::I32LeS,
            Opcode::I32LeU,
            Opcode::I32GeS,
            Opcode::I32GeU,
            Opcode::I32LtS,
            Opcode::I32Eqz,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_elf3() -> Program {
        let x_value: u32 = 0x1;
        let y_value: u32 = 0x2;
        let z1_value: u32 = 0x1;
        let z2_value: u32 = 0x2;
        let z3_value: u32 = 0x1;
        let z4_value: u32 = 0x2;
        let z5_value: u32 = 0x1;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z1_value.into()),
            Opcode::I32Const(z2_value.into()),
            Opcode::I32Const(z3_value.into()),
            Opcode::I32Const(z4_value.into()),
            Opcode::I32Const(z5_value.into()),
            Opcode::I32And,
            Opcode::I32Or,
            Opcode::I32Xor,
            Opcode::I32Shl,
            Opcode::I32ShrS,
            Opcode::I32ShrU,
        ];
        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf4() -> Program {
        let addr: u32 = 0x40000;
        let addr_2: u32 = 0x40004;
        let addr_3: u32 = 0x40008;

        let x_value: u32 = 0x10007;
        let x_2_value: u32 = 0x10008;

        let x_3_value: u32 = 0x200AA;
        // let mut mem = HashMap::new();
        // mem.insert(sp_value, addr);
        // mem.insert(sp_value - 4, addr_2);
        // mem.insert(sp_value - 8, 0x10000);
        // mem.insert(sp_value - 12, addr_3);
        // mem.insert(sp_value - 16, 0x10000);
        // mem.insert(addr, x_value);
        // mem.insert(addr_2, x_2_value);
        // mem.insert(addr_3, x_3_value);

        //  println!("{:?}", mem);
        let instructions = vec![
            Opcode::I32Const(0x10.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(x_3_value.into()),
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0),
            Opcode::I32Store16(0),
            Opcode::I32Store8(0),
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(0),
            // Opcode::I32Load16U(0),
            // Opcode::I32Load8U(0),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_load4() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 65551i32 as u32;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16S(0u32),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };
        program
    }
    fn build_elf_load5() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0xFFFF_0005;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store16(0u32),
            Opcode::I32Const(addr.into()),
            Opcode::I32Load16U(0u32),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };
        program
    }
    fn build_elf5() -> Program {
        let addr: u32 = 0x40000;
        let addr_2: u32 = 0x40004;
        let addr_3: u32 = 0x40008;

        let x_value: u32 = 0x10007;
        let x_2_value: u32 = 0x10008;

        let x_3_value: u32 = 0x200AA;

        let instructions = vec![
            Opcode::I32Const(0x10.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr_3.into()),
            Opcode::I32Const(x_3_value.into()),
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const(addr_2.into()),
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(0),
            Opcode::I32Store16(0),
            Opcode::I32Store8(0),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_br() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::Br(3.into()),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_brifnez() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::BrIfNez(3.into()),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_brifeqz() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(0.into()),
            Opcode::BrIfEqz(3.into()),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_nobr_brifeqz() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(1.into()),
            Opcode::BrIfEqz(3.into()),
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_nobr_brifnez() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(0.into()),
            Opcode::BrIfNez(3.into()),
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_brtable_index() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(1.into()),
            Opcode::BrTable(3u32),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_brtable_max_index() -> Program {
        let x_value: u32 = 0x1;
        let addr: u32 = 0x10000;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(10.into()),
            Opcode::BrTable(2u32),
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
        ];

        let program = Program::from_instrs(instructions);
        program
    }

    fn build_elf_local_const() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x1234;
        let x_2_value: u32 = x_value + 5;
        let depth = 5;
        let under_depth = depth - 2;
        let constant = x_value;

        let instructions = vec![
            Opcode::I32Const(x_2_value.into()),
            Opcode::I32Const((x_value + 123).into()),
            Opcode::I32Const((x_value + 456).into()),
            Opcode::I32Const((x_value + 789).into()),
            Opcode::I32Const((x_value).into()),
            Opcode::LocalGet(depth),
            Opcode::LocalSet(under_depth),
            Opcode::LocalTee(under_depth),
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }
    fn build_elf_const_another() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x12345;
        let y_value: u32 = 0x54321;

        let instructions = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Add,
        ];

        let program = Program::from_instrs(instructions);
        //  memory_image: BTreeMap::new() };

        program
    }

    fn build_store_unaligned() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x0103_0507;

        let addr: u32 = 0x10000;

        let opcodes = vec![
            Opcode::I32Const(2.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Store(3u32),
        ];
        let program = Program::from_instrs(opcodes);
        program
    }

    fn build_load_unaligned() -> Program {
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
        program
    }

    fn build_rwasm_call_internal() -> Program {
        let sp_value: u32 = SP_START;
        let x_value: u32 = 0x7;
        let y_value: u32 = 0x2;
        let z_value: u32 = 0x1;
        let functions = vec![0, 24];

        let opcodes = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(y_value.into()),
            Opcode::I32Const(z_value.into()),
            Opcode::CallInternal(6u32.into()),
            Opcode::I32Sub,
            Opcode::Return,
            Opcode::I32Add,
            Opcode::Return,
        ];

        let program = Program::from_instrs(opcodes);
        program
    }

    #[test]
    fn test_rwasm_proof1() {
        let program = build_elf();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof2() {
        let program = build_elf2();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof3() {
        let program = build_elf3();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof4() {
        let program = build_elf4();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof_load4() {
        let program = build_elf_load4();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof_load5() {
        let program = build_elf_load5();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_proof5() {
        let program = build_elf5();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_br() {
        let program = build_elf_br();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_rwasm_brifnez() {
        let program = build_elf_brifnez();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_brifeqz() {
        let program = build_elf_brifeqz();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_no_brifeqz() {
        let program = build_elf_nobr_brifeqz();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_no_brifnez() {
        let program = build_elf_nobr_brifnez();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_brtable_index() {
        let program = build_elf_brtable_index();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_brtable_max_index() {
        let program = build_elf_brtable_max_index();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_local() {
        let program = build_elf_local_const();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_another1() {
        let program = build_elf_const_another();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_unaligned_store() {
        let program = build_store_unaligned();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_unaligned_load() {
        let program = build_load_unaligned();
        run_rwasm_prover(program);
    }

    #[test]
    fn test_rwasm_call_internal() {
        let program = build_rwasm_call_internal();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_call_direct_add() {
        let program = build_test_call_direct_add();
        run_rwasm_prover(program);
    }
    fn build_test_call_direct_add() -> Program {
        // f_add(x,y) = x + y
        let f_add = vec![rwasm::Opcode::I32Add, rwasm::Opcode::Return]; // len = 2

        // main: push x, y; Call(f_add); const expected; eq; return  => 6 ops
        let main_len = 6u32;
        let f_add_pos = main_len;

        let x = 12u32;
        let y = 30u32;
        let expected = x + y;

        let mut ops = Vec::new();
        ops.push(rwasm::Opcode::I32Const(x.into()));
        ops.push(rwasm::Opcode::I32Const(y.into()));
        // If your runtime expects a function index instead of a byte position,
        ops.push(rwasm::Opcode::CallInternal(f_add_pos.into()));
        ops.push(rwasm::Opcode::I32Const(expected.into()));
        ops.push(rwasm::Opcode::I32Eq);
        ops.push(rwasm::Opcode::Return); // prevent fall-through

        // append the callee
        ops.extend(f_add);

        let program = Program::from_instrs(ops);
        program
    }
    fn build_test_fibonacci_n25_callinternal() -> Program {
        let base: u32 = 0x10000;
        let addr_tmp = base + 0;
        let addr_a = base + 4; // F(n)
        let addr_b = base + 8; // F(n+1)
        let addr_n = base + 12;

        // step(): (a,b,n) -> (b, a+b, n-1); returns new n (ignored by caller)
        let step_fn = vec![
            // tmp = a + b
            rwasm::Opcode::I32Const(addr_tmp.into()),
            rwasm::Opcode::I32Const(addr_a.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::I32Const(addr_b.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::I32Add,
            rwasm::Opcode::I32Store(0u32),
            // a = b
            rwasm::Opcode::I32Const(addr_a.into()),
            rwasm::Opcode::I32Const(addr_b.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::I32Store(0u32),
            // b = tmp
            rwasm::Opcode::I32Const(addr_b.into()),
            rwasm::Opcode::I32Const(addr_tmp.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::I32Store(0u32),
            // n = n - 1
            rwasm::Opcode::I32Const(addr_n.into()),
            rwasm::Opcode::I32Const(addr_n.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::I32Const(1u32.into()),
            rwasm::Opcode::I32Sub,
            rwasm::Opcode::I32Store(0u32),
            // return n
            rwasm::Opcode::I32Const(addr_n.into()),
            rwasm::Opcode::I32Load(0u32),
            rwasm::Opcode::Return,
        ];

        let mut ops = Vec::new();
        // memory
        ops.push(rwasm::Opcode::I32Const(2.into()));
        ops.push(rwasm::Opcode::MemoryGrow);

        // init a=0, b=1, n=25
        ops.push(rwasm::Opcode::I32Const(addr_a.into()));
        ops.push(rwasm::Opcode::I32Const(0u32.into()));
        ops.push(rwasm::Opcode::I32Store(0u32));

        ops.push(rwasm::Opcode::I32Const(addr_b.into()));
        ops.push(rwasm::Opcode::I32Const(1u32.into()));
        ops.push(rwasm::Opcode::I32Store(0u32));

        ops.push(rwasm::Opcode::I32Const(addr_n.into()));
        ops.push(rwasm::Opcode::I32Const(25u32.into()));
        ops.push(rwasm::Opcode::I32Store(0u32));

        // 25 iterations of step(); drop returned n each time
        let mut call_sites = Vec::<usize>::new();
        for _ in 0..25 {
            call_sites.push(ops.len());
            ops.push(rwasm::Opcode::CallInternal(0u32.into())); // patched later
            ops.push(rwasm::Opcode::Drop);
        }

        // compare a with F25 = 75025, return
        ops.push(rwasm::Opcode::I32Const(addr_a.into()));
        ops.push(rwasm::Opcode::I32Load(0u32));
        ops.push(rwasm::Opcode::I32Const(75025u32.into()));
        ops.push(rwasm::Opcode::I32Eq);
        ops.push(rwasm::Opcode::Return);

        // patch function position and append
        let step_pos = ops.len() as u32;
        for idx in call_sites {
            ops[idx] = rwasm::Opcode::CallInternal(step_pos.into());
        }
        ops.extend(step_fn);

        let program = Program::from_instrs(ops);
        program
    }

    #[test]
    fn test_fibonacci_n25_callinternal() {
        let program = build_test_fibonacci_n25_callinternal();
        run_rwasm_prover(program);
    }
    #[test]
    fn test_nested_three_level_calls() {
        // f3(x): return 2*x
        let f3 = vec![
            rwasm::Opcode::I32Const(1u32.into()),
            rwasm::Opcode::I32Shl,
            rwasm::Opcode::Return,
        ]; // len 3

        // f2(x,y): return f3(x) + y
        let f2 = vec![
            rwasm::Opcode::CallInternal(0u32.into()), // patched to f3_pos
            rwasm::Opcode::I32Add,
            rwasm::Opcode::Return,
        ]; // len 3

        // f1(x,y,z): return f2(x,y) + z
        let f1 = vec![
            rwasm::Opcode::CallInternal(0u32.into()), // patched to f2_pos
            rwasm::Opcode::I32Add,
            rwasm::Opcode::Return,
        ]; // len 3

        // main: push z, y, x (so top=x,y below,z bottom) ; call f1 ; cmp ; return
        let x = 3u32;
        let y = 5u32;
        let z = 7u32;
        let expected = (2 * x) + y + z; // 18

        let mut ops = Vec::new();
        ops.push(rwasm::Opcode::I32Const(z.into()));
        ops.push(rwasm::Opcode::I32Const(y.into()));
        ops.push(rwasm::Opcode::I32Const(x.into()));
        ops.push(rwasm::Opcode::CallInternal(0u32.into())); // patch to f1_pos
        ops.push(rwasm::Opcode::I32Const(expected.into()));
        ops.push(rwasm::Opcode::I32Eq);
        ops.push(rwasm::Opcode::Return);

        // compute positions
        let f1_pos = ops.len() as u32; // after main
        let mut f1_patched = f1.clone();
        let f2_pos = f1_pos + f1.len() as u32; // after f1
        let mut f2_patched = f2.clone();
        let f3_pos = f2_pos + f2.len() as u32; // after f2

        // patch call targets
        f1_patched[0] = rwasm::Opcode::CallInternal(f2_pos.into());
        f2_patched[0] = rwasm::Opcode::CallInternal(f3_pos.into());
        ops[3] = rwasm::Opcode::CallInternal(f1_pos.into());

        // append functions
        ops.extend(f1_patched);
        ops.extend(f2_patched);
        ops.extend(f3);

        // run
        let program = Program::from_instrs(ops);
        run_rwasm_prover(program);
    }
    // #[test]
    // fn test_rwasm_call_internal_and_return() {
    //     let program = build_elf_call();
    //     run_rwasm_prover(program);
    // }
    // #[test]
    // fn test_rwasm_call_internal_and_return2() {
    //     let program = build_elf_call2();
    //     run_rwasm_prover(program);
    // }

    // #[test]
    // fn test_rwasm_call_internal_and_return3() {
    //     let program = build_elf_call3();
    //     run_rwasm_prover(program);
    // }

    /* #[test]
    fn test_rwasm_skipped() {
        let program = build_elf_skipped_ins();
        run_rwasm_prover(program);
    }*/

    fn build_elf_load8u() -> Program {
        let addr: u32 = 0x10000;
        let value: u32 = 0xABCD_00EF; // low byte = 0xEF
        let ops = vec![
            Opcode::I32Const(2u32.into()),
            Opcode::MemoryGrow,
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(value.into()),
            Opcode::I32Store(0u32),
            // load the lowest byte back unsigned
            Opcode::I32Const(addr.into()),
            Opcode::I32Load8U(0u32),
            Opcode::Drop, // discard loaded byte; proof just needs a valid trace
        ];
        Program::from_instrs(ops)
    }

    #[test]
    fn test_rwasm_load8u() {
        let program = build_elf_load8u();
        run_rwasm_prover(program);
    }

    fn build_elf_mem_overwrite() -> Program {
        // Write 32-bit word, then overwrite its second byte via Store8 with offset=1,
        // finally read back the 32-bit word (exercise mixed width stores/loads).
        let addr: u32 = 0x10020;
        let word: u32 = 0x0102_0304; // bytes: 04 03 02 01 (LE)
        let ops = vec![
            Opcode::I32Const(2u32.into()),
            Opcode::MemoryGrow,
            // store initial 32-bit word
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(word.into()),
            Opcode::I32Store(0u32),
            // overwrite byte at addr+1 with 0xAA
            Opcode::I32Const(addr.into()),
            Opcode::I32Const(0xAAu32.into()),
            Opcode::I32Store8(1u32),
            // read back full 32-bit word (unaltered semantics validated by executor)
            Opcode::I32Const(addr.into()),
            Opcode::I32Load(0u32),
            Opcode::Drop,
        ];
        Program::from_instrs(ops)
    }

    #[test]
    fn test_rwasm_mem_overwrite() {
        let program = build_elf_mem_overwrite();
        run_rwasm_prover(program);
    }

    fn build_test_call_direct_mul() -> Program {
        // f_mul(x,y) = x * y
        let f_mul = vec![rwasm::Opcode::I32Mul, rwasm::Opcode::Return]; // len = 2

        // main: push x, y; Call(f_mul); const expected; eq; return
        let main_len = 6u32;
        let f_mul_pos = main_len;

        let x = 7u32;
        let y = 9u32;
        let expected = x * y;

        let mut ops = Vec::new();
        ops.push(Opcode::I32Const(x.into()));
        ops.push(Opcode::I32Const(y.into()));
        ops.push(Opcode::CallInternal(f_mul_pos.into()));
        ops.push(Opcode::I32Const(expected.into()));
        ops.push(Opcode::I32Eq);
        ops.push(Opcode::Return);

        // append callee
        ops.extend(f_mul);

        Program::from_instrs(ops)
    }

    #[test]
    fn test_call_direct_mul() {
        let program = build_test_call_direct_mul();
        run_rwasm_prover(program);
    }

    fn build_elf_brtable_zero_index() -> Program {
        // Use BrTable with index 0; ensure enough stack for 3 shifts if fallthrough occurs.
        let x_value: u32 = 1;
        let ops = vec![
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()),
            Opcode::I32Const(x_value.into()), // extra operand so 3 shifts are safe
            Opcode::I32Const(0u32.into()),    // table index = 0
            Opcode::BrTable(3u32),            // table size = 3
            // these would be skipped if correct table mapping branches;
            // if fallthrough, they still have enough operands.
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::I32Shl,
            Opcode::Return,
        ];
        Program::from_instrs(ops)
    }

    #[test]
    fn test_rwasm_brtable_zero_index() {
        let program = build_elf_brtable_zero_index();
        run_rwasm_prover(program);
    }

    fn build_table_grow_and_get() -> Program {
        let ops = vec![
            Opcode::I32Const(0.into()),
            Opcode::I32Const(2.into()), // delta = 2
            Opcode::TableGrow(0),
            Opcode::I32Const(1.into()),
            Opcode::TableGet(0),
            Opcode::Drop,
            Opcode::Return,
        ];
        Program::from_instrs(ops)
    }

    #[test]
    fn test_table_grow_and_get() {
        let program = build_table_grow_and_get();
        run_rwasm_prover(program);
    }
}
