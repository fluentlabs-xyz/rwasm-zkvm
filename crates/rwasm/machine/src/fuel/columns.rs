use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::size_of;

use crate::memory::MemoryReadWriteCols;

pub const NUM_FUEL_COLS: usize = size_of::<FuelColumns<u8>>();

/// The column layout for fuel consumption.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct FuelColumns<T> {
    pub clk: T,
    pub shard: T,
    pub pc: Word<T>,
    pub next_pc: Word<T>,
    pub sp: T,
    pub fuel_consumed_low_record: MemoryReadWriteCols<T>,
    pub fuel_consumed_high_record: MemoryReadWriteCols<T>,
    pub to_consume_fuel: Word<T>,
    pub next_consumed_fuel_low_record: MemoryReadWriteCols<T>,
    pub next_consumed_fuel_high_record: MemoryReadWriteCols<T>,
    pub is_consume_fuel: T,
    pub is_consume_fuel_stack: T,
    pub is_carryed: T, /* This indicates whether there is a carry when adding to_consume_fuel to
                        * fuel_consumed */
}
