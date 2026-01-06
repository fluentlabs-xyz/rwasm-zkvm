mod air;
mod columns;
mod trace;

pub use columns::*;
use p3_air::BaseAir;

#[derive(Default)]
pub struct FuelChip;

impl<F> BaseAir<F> for FuelChip {
    fn width(&self) -> usize {
        NUM_FUEL_COLS
    }
}
