use std::{fs, path::Path};

use hashbrown::HashMap;

use rwasm::{
    CallStack, ExecutionEngine, RwasmExecutor, RwasmModule, Store, ValueStack,
};

use crate::executor;

pub const FIB_REC_PATH: &str = "../examples/hello/hello.wat";
pub const FIB_PATH: &str = "../examples/hello/hello_copy.wat";
pub const SIMPLE_PATH: &str = "../examples/hello/simple.wat";
pub fn read_wat(file: &Path) -> Vec<u8> {
    let file_bytes = fs::read(file).unwrap();
    let wasm_binary: Vec<u8>;

    wasm_binary = wat::parse_bytes(&file_bytes).unwrap().to_vec();

    wasm_binary
}
