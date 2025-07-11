#![allow(
    clippy::new_without_default,
    clippy::field_reassign_with_default,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_unwrap,
    clippy::default_constructed_unit_structs,
    clippy::box_default,
    clippy::assign_op_pattern,
    deprecated,
    incomplete_features
)]
#![warn(unused_extern_crates)]

mod memory;
mod cpu;



// Re-export the `SP1ReduceProof` struct from rwasm_machine.
//
// This is done to avoid a circular dependency between rwasm_machine and rwasm_executor, and
// enable crates that depend on rwasm_machine to import the `SP1ReduceProof` type directly.
pub mod reduce {
    pub use rwasm_executor::SP1ReduceProof;
}

#[cfg(test)]
pub mod test {
    
   #[test]
   fn simple(){

   }
}
