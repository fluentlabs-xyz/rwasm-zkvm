mod air;
mod columns;
mod trace;

pub use columns::*;
use p3_air::BaseAir;

#[derive(Default)]
pub struct CallChip;

impl<F> BaseAir<F> for CallChip {
    fn width(&self) -> usize {
        NUM_CALL_COLS
    }
}
