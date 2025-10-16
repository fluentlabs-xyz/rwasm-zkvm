mod air;

pub use air::*;

#[cfg(test)]
mod tests {

    use sp1_curves::{params::FieldParameters, uint256::U256Field, utils::biguint_from_limbs};

    #[test]
    fn test_uint256_modulus() {
        assert_eq!(biguint_from_limbs(U256Field::MODULUS), U256Field::modulus());
    }
}
