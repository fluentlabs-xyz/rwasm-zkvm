use std::{hash::Hash, str::FromStr};

use hashbrown::HashMap;

use rwasm::Opcode;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::RwasmAirId;

/// Serialize a `HashMap<u32, V>` as a `Vec<(u32, V)>`.
pub fn serialize_hashmap_as_vec<K: Eq + Hash + Serialize, V: Serialize, S: Serializer>(
    map: &HashMap<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    Serialize::serialize(&map.iter().collect::<Vec<_>>(), serializer)
}

/// Deserialize a `Vec<(u32, V)>` as a `HashMap<u32, V>`.
pub fn deserialize_hashmap_as_vec<
    'de,
    K: Eq + Hash + Deserialize<'de>,
    V: Deserialize<'de>,
    D: Deserializer<'de>,
>(
    deserializer: D,
) -> Result<HashMap<K, V>, D::Error> {
    let seq: Vec<(K, V)> = Deserialize::deserialize(deserializer)?;
    Ok(seq.into_iter().collect())
}

/// Returns `true` if the given `opcode` is a signed operation.
#[must_use]
pub fn is_signed_operation(ins: Opcode) -> bool {
    ins == Opcode::I32DivS || ins == Opcode::I32RemS
}

/// Calculate the correct `quotient` and `remainder` for the given `b` and `c` per Rwasm spec.
#[must_use]
pub fn get_quotient_and_remainder(b: u32, c: u32, ins: Opcode) -> (u32, u32) {
    if c == 0 {
        // When c is 0, the quotient is 2^32 - 1 and the remainder is b regardless of whether we
        // perform signed or unsigned division.
        (u32::MAX, b)
    } else if is_signed_operation(ins) {
        ((b as i32).wrapping_div(c as i32) as u32, (b as i32).wrapping_rem(c as i32) as u32)
    } else {
        (b.wrapping_div(c), b.wrapping_rem(c))
    }
}

/// Calculate the most significant bit of the given 32-bit integer `a`, and returns it as a u8.
#[must_use]
pub const fn get_msb(a: u32) -> u8 {
    ((a >> 31) & 1) as u8
}

/// Load the cost of each air from the predefined JSON.
#[must_use]
pub fn rv32im_costs() -> HashMap<RwasmAirId, usize> {
    let costs: HashMap<String, usize> =
        serde_json::from_str(include_str!("./artifacts/rv32im_costs.json")).unwrap();
    costs.into_iter().map(|(k, v)| (RwasmAirId::from_str(&k).unwrap(), v)).collect()
}

///  these are pseudo rwasm opcode for chips
pub const I32MULH_CODE: u32 = 0x0101;
pub const I32MULHU_CODE: u32 = 0x0102;
pub const I32MULHSU_CODE: u32 = 0x0103;
