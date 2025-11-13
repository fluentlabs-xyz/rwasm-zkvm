use std::iter::once;

use p3_air::AirBuilder;
use sp1_stark::{
    air::{AirInteraction, BaseAirBuilder, InteractionScope},
    InteractionKind, Word,
};

/// A trait which contains methods related to program interactions in an AIR.
pub trait ProgramAirBuilder: BaseAirBuilder {
    /// Sends an instruction.
    fn send_program(
        &mut self,
        pc: impl Into<Self::Expr>,
        opcode: impl Into<Self::Expr>,
        aux_val: Word<impl Into<Self::Expr>>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let values = once(pc.into())
            .chain(once(opcode.into()))
            .chain(aux_val.0.into_iter().map(Into::into))
            .collect();

        self.send(
            AirInteraction::new(values, multiplicity.into(), InteractionKind::Program),
            InteractionScope::Local,
        );
    }

    /// Receives an instruction.
    fn receive_program(
        &mut self,
        pc: impl Into<Self::Expr>,
        opcode: impl Into<Self::Expr>,
        aux_val: Word<impl Into<Self::Expr>>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let values: Vec<<Self as AirBuilder>::Expr> = once(pc.into())
            .chain(once(opcode.into()))
            .chain(aux_val.0.into_iter().map(Into::into))
            .collect();

        self.receive(
            AirInteraction::new(values, multiplicity.into(), InteractionKind::Program),
            InteractionScope::Local,
        );
    }
}
