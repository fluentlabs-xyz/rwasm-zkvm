mod shapeable;

pub use shapeable::*;

use std::{collections::BTreeMap, marker::PhantomData, str::FromStr};

use hashbrown::HashMap;
use itertools::Itertools;
use num::Integer;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_util::log2_ceil_usize;
use rwasm_executor::{ExecutionRecord, Program, RwasmAirId};
use sp1_stark::{
    air::MachineAir,
    shape::{OrderedShape, Shape, ShapeCluster},
};
use thiserror::Error;

use super::rwasm::rwasm_chips::{ByteChip, ProgramChip, SyscallChip};
use crate::{
    control_flow::CallChip,
    global::GlobalChip,
    memory::{MemoryLocalChip, NUM_LOCAL_MEMORY_ENTRIES_PER_ROW},
    rwasm::RwasmAir,
};

/// The set of maximal shapes.
///
/// These shapes define the "worst-case" shapes for typical shards that are proving `rv32im`
/// execution. We use a variant of a cartesian product of the allowed log heights to generate
/// smaller shapes from these ones.
const MAXIMAL_SHAPES: &[u8] = include_bytes!("maximal_shapes.json");

/// The set of tiny shapes.
///
/// These shapes are used to optimize performance for smaller programs.
const SMALL_SHAPES: &[u8] = include_bytes!("small_shapes.json");

/// A configuration for what shapes are allowed to be used by the prover.
#[derive(Debug)]
pub struct CoreShapeConfig<F: PrimeField32> {
    partial_preprocessed_shapes: ShapeCluster<RwasmAirId>,
    partial_core_shapes: BTreeMap<usize, Vec<ShapeCluster<RwasmAirId>>>,
    partial_memory_shapes: ShapeCluster<RwasmAirId>,
    partial_precompile_shapes: HashMap<RwasmAirId, (usize, Vec<usize>)>,
    partial_small_shapes: Vec<ShapeCluster<RwasmAirId>>,
    costs: HashMap<RwasmAirId, usize>,
    _data: PhantomData<F>,
}

impl<F: PrimeField32> CoreShapeConfig<F> {
    /// Fix the preprocessed shape of the proof.
    pub fn fix_preprocessed_shape(&self, program: &mut Program) -> Result<(), CoreShapeError> {
        // If the preprocessed shape is already fixed, return an error.
        if program.preprocessed_shape.is_some() {
            return Err(CoreShapeError::PreprocessedShapeAlreadyFixed);
        }

        // Get the heights of the preprocessed chips and find a shape that fits.
        let preprocessed_heights = RwasmAir::<F>::preprocessed_heights(program);
        let preprocessed_shape = self
            .partial_preprocessed_shapes
            .find_shape(&preprocessed_heights)
            .ok_or(CoreShapeError::PreprocessedShapeError)?;

        // Set the preprocessed shape.
        program.preprocessed_shape = Some(preprocessed_shape);

        Ok(())
    }

    /// Fix the shape of the proof.
    pub fn fix_shape(&self, record: &mut ExecutionRecord) -> Result<(), CoreShapeError> {
        if record.program.preprocessed_shape.is_none() {
            return Err(CoreShapeError::PreprocessedShapeMissing);
        }
        if record.shape.is_some() {
            return Err(CoreShapeError::ShapeAlreadyFixed);
        }

        // Set the shape of the chips with prepcoded shapes to match the preprocessed shape from the
        // program.
        record.shape.clone_from(&record.program.preprocessed_shape);

        let shape = self.find_shape(record)?;
        record.shape.as_mut().unwrap().extend(shape);
        Ok(())
    }

    /// TODO move this into the executor crate
    pub fn find_shape<R: Shapeable>(
        &self,
        record: &R,
    ) -> Result<Shape<RwasmAirId>, CoreShapeError> {
        match record.kind() {
            // If this is a packed "core" record where the cpu events are alongisde the memory init
            // and finalize events, try to fix the shape using the tiny shapes.
            ShardKind::PackedCore => {
                // Get the heights of the core airs in the record.
                let mut heights = record.core_heights();
                heights.extend(record.memory_heights());

                let (cluster_index, shape, _) = self
                    .minimal_cluster_shape(self.partial_small_shapes.iter().enumerate(), &heights)
                    .ok_or_else(|| {
                        // No shape found, so return an error.
                        CoreShapeError::ShapeError(
                            heights
                                .iter()
                                .map(|(air, height)| (air.to_string(), log2_ceil_usize(*height)))
                                .collect(),
                        )
                    })?;

                let shard = record.shard();
                tracing::debug!("Shard Lifted: Index={}, Cluster={}", shard, cluster_index);
                for (air, height) in heights.iter() {
                    if shape.contains(air) {
                        tracing::debug!(
                            "Chip {:<20}: {:<3} -> {:<3}",
                            air,
                            log2_ceil_usize(*height),
                            shape.log2_height(air).unwrap(),
                        );
                    }
                }
                Ok(shape)
            }
            ShardKind::Core => {
                // If this is a normal "core" record, try to fix the shape as such.

                // Get the heights of the core airs in the record.
                let heights = record.core_heights();

                // Try to find the smallest shape fitting within at least one of the candidate
                // shapes.
                let log2_shard_size = record.log2_shard_size();

                let (cluster_index, shape, _) = self
                    .minimal_cluster_shape(
                        self.partial_core_shapes
                            .range(log2_shard_size..)
                            .flat_map(|(_, clusters)| clusters.iter().enumerate()),
                        &heights,
                    )
                    // No shape found, so return an error.
                    .ok_or_else(|| CoreShapeError::ShapeError(record.debug_stats()))?;

                let shard = record.shard();
                tracing::debug!("Shard Lifted: Index={}, Cluster={}", shard, cluster_index);

                for (air, height) in heights.iter() {
                    if shape.contains(air) {
                        tracing::debug!(
                            "Chip {:<20}: {:<3} -> {:<3}",
                            air,
                            log2_ceil_usize(*height),
                            shape.log2_height(air).unwrap(),
                        );
                    }
                }
                Ok(shape)
            }
            ShardKind::GlobalMemory => {
                // If the record is a does not have the CPU chip and is a global memory
                // init/finalize record, try to fix the shape as such.
                let heights = record.memory_heights();
                let shape = self
                    .partial_memory_shapes
                    .find_shape(&heights)
                    .ok_or(CoreShapeError::ShapeError(record.debug_stats()))?;
                Ok(shape)
            }
            ShardKind::Precompile => {
                // Try to fix the shape as a precompile record.
                for (&air, (memory_events_per_row, allowed_log2_heights)) in
                    self.partial_precompile_shapes.iter()
                {
                    // Filter to check that the shard and shape air match.

                    let Some((height, num_memory_local_events, num_global_events)) =
                        record.precompile_heights().find_map(|x| (x.0 == air).then_some(x.1))
                    else {
                        continue;
                    };
                    println!(
                        "air: {},height{}, num_memory_local_events {}, num_global_events{}",
                        air, height, num_memory_local_events, num_global_events
                    );
                    for allowed_log2_height in allowed_log2_heights {
                        let allowed_height = 1 << allowed_log2_height;
                        if height as u32 <= allowed_height as u32 {
                            for shape in self.get_precompile_shapes(
                                air,
                                *memory_events_per_row,
                                *allowed_log2_height,
                            ) {
                                let mem_events_height = shape[2].1;
                                let global_events_height = shape[3].1;
                                if num_memory_local_events
                                    .div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW) <=
                                    (1 << mem_events_height) &&
                                    num_global_events <= (1 << global_events_height)
                                {
                                    let mut actual_shape: Shape<RwasmAirId> = Shape::default();
                                    actual_shape.extend(
                                        shape
                                            .iter()
                                            .map(|x| (RwasmAirId::from_str(&x.0).unwrap(), x.1)),
                                    );
                                    println!("actual shape{:?}", actual_shape);
                                    return Ok(actual_shape);
                                }
                            }
                        }
                    }
                    tracing::error!(
                        "Cannot find shape for precompile {:?}, height {:?}, and mem events {:?}",
                        air,
                        height,
                        num_memory_local_events
                    );
                    return Err(CoreShapeError::ShapeError(record.debug_stats()));
                }
                Err(CoreShapeError::PrecompileNotIncluded(record.debug_stats()))
            }
        }
    }

    /// Returns the area, cluster index, and shape of the minimal shape from candidates that fit a
    /// given collection of heights.
    pub fn minimal_cluster_shape<'a, N, I>(
        &self,
        indexed_shape_clusters: I,
        heights: &[(RwasmAirId, usize)],
    ) -> Option<(N, Shape<RwasmAirId>, usize)>
    where
        I: IntoIterator<Item = (N, &'a ShapeCluster<RwasmAirId>)>,
    {
        // Try to find a shape fitting within at least one of the candidate shapes.
        indexed_shape_clusters
            .into_iter()
            .filter_map(|(i, cluster)| {
                // println!("height:{:?},",heights);
                let shape = cluster.find_shape(heights)?;
                // println!("shape:{:?}",shape);
                let area = self.estimate_lde_size(&shape);
                Some((i, shape, area))
            })
            .min_by_key(|x| x.2) // Find minimum by area.
    }

    // TODO: this function is atrocious, fix this
    fn get_precompile_shapes(
        &self,
        air_id: RwasmAirId,
        memory_events_per_row: usize,
        allowed_log2_height: usize,
    ) -> Vec<[(String, usize); 4]> {
        // TODO: This is a temporary fix to the shape, concretely fix this
        (1..=4 * air_id.rows_per_event())
            .rev()
            .map(|rows_per_event| {
                let num_local_mem_events =
                    ((1 << allowed_log2_height) * memory_events_per_row).div_ceil(rows_per_event);
                [
                    (air_id.to_string(), allowed_log2_height),
                    (
                        RwasmAir::<F>::SyscallPrecompile(SyscallChip::precompile()).name(),
                        (((1 << allowed_log2_height) as usize)
                            .div_ceil(air_id.rows_per_event())
                            .next_power_of_two()
                            .ilog2() as usize)
                            .max(4),
                    ),
                    (
                        RwasmAir::<F>::MemoryLocal(MemoryLocalChip::new()).name(),
                        (num_local_mem_events
                            .div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW)
                            .next_power_of_two()
                            .ilog2() as usize)
                            .max(4),
                    ),
                    (
                        RwasmAir::<F>::Global(GlobalChip).name(),
                        ((2 * num_local_mem_events +
                            (1 << allowed_log2_height).div_ceil(&air_id.rows_per_event()))
                        .next_power_of_two()
                        .ilog2() as usize)
                            .max(4),
                    ),
                ]
            })
            .filter(|shape| shape[3].1 <= 22)
            .collect::<Vec<_>>()
    }

    fn generate_all_shapes_from_allowed_log_heights(
        allowed_log_heights: impl IntoIterator<Item = (String, Vec<Option<usize>>)>,
    ) -> impl Iterator<Item = OrderedShape> {
        allowed_log_heights
            .into_iter()
            .map(|(name, heights)| heights.into_iter().map(move |height| (name.clone(), height)))
            .multi_cartesian_product()
            .map(|iter| {
                iter.into_iter()
                    .filter_map(|(name, maybe_height)| {
                        maybe_height.map(|log_height| (name, log_height))
                    })
                    .collect::<OrderedShape>()
            })
    }

    pub fn all_shapes(&self) -> impl Iterator<Item = OrderedShape> + '_ {
        let preprocessed_heights = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, heights)| (air.to_string(), heights.clone()))
            .collect::<HashMap<_, _>>();

        let mut memory_heights = self
            .partial_memory_shapes
            .iter()
            .map(|(air, heights)| (air.to_string(), heights.clone()))
            .collect::<HashMap<_, _>>();
        memory_heights.extend(preprocessed_heights.clone());

        let precompile_only_shapes = self.partial_precompile_shapes.iter().flat_map(
            move |(&air, (mem_events_per_row, allowed_log_heights))| {
                allowed_log_heights.iter().flat_map(move |allowed_log_height| {
                    self.get_precompile_shapes(air, *mem_events_per_row, *allowed_log_height)
                })
            },
        );

        let precompile_shapes =
            Self::generate_all_shapes_from_allowed_log_heights(preprocessed_heights.clone())
                .flat_map(move |preprocessed_shape| {
                    precompile_only_shapes.clone().map(move |precompile_shape| {
                        preprocessed_shape
                            .clone()
                            .into_iter()
                            .chain(precompile_shape)
                            .collect::<OrderedShape>()
                    })
                });

        self.partial_core_shapes
            .values()
            .flatten()
            .chain(self.partial_small_shapes.iter())
            .flat_map(move |allowed_log_heights| {
                Self::generate_all_shapes_from_allowed_log_heights({
                    let mut log_heights = allowed_log_heights
                        .iter()
                        .map(|(air, heights)| (air.to_string(), heights.clone()))
                        .collect::<HashMap<_, _>>();
                    log_heights.extend(preprocessed_heights.clone());
                    log_heights
                })
            })
            .chain(Self::generate_all_shapes_from_allowed_log_heights(memory_heights))
            .chain(precompile_shapes)
    }

    pub fn maximal_core_shapes(&self, max_log_shard_size: usize) -> Vec<Shape<RwasmAirId>> {
        let max_shard_size: usize = core::cmp::max(
            1 << max_log_shard_size,
            1 << self.partial_core_shapes.keys().min().unwrap(),
        );

        let log_shard_size = max_shard_size.ilog2() as usize;
        debug_assert_eq!(1 << log_shard_size, max_shard_size);
        let max_preprocessed = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, allowed_heights)| {
                (air.to_string(), allowed_heights.last().unwrap().unwrap())
            })
            .collect::<HashMap<_, _>>();

        let max_core_shapes =
            self.partial_core_shapes[&log_shard_size].iter().map(|allowed_log_heights| {
                max_preprocessed
                    .clone()
                    .into_iter()
                    .chain(allowed_log_heights.iter().flat_map(|(air, allowed_heights)| {
                        allowed_heights
                            .last()
                            .unwrap()
                            .map(|log_height| (air.to_string(), log_height))
                    }))
                    .map(|(air, log_height)| (RwasmAirId::from_str(&air).unwrap(), log_height))
                    .collect::<Shape<RwasmAirId>>()
            });

        max_core_shapes.collect()
    }

    pub fn maximal_core_plus_precompile_shapes(
        &self,
        max_log_shard_size: usize,
    ) -> Vec<Shape<RwasmAirId>> {
        let max_preprocessed = self
            .partial_preprocessed_shapes
            .iter()
            .map(|(air, allowed_heights)| {
                (air.to_string(), allowed_heights.last().unwrap().unwrap())
            })
            .collect::<HashMap<_, _>>();

        let precompile_only_shapes = self.partial_precompile_shapes.iter().flat_map(
            move |(&air, (mem_events_per_row, allowed_log_heights))| {
                self.get_precompile_shapes(
                    air,
                    *mem_events_per_row,
                    *allowed_log_heights.last().unwrap(),
                )
            },
        );

        let precompile_shapes: Vec<Shape<RwasmAirId>> = precompile_only_shapes
            .map(|x| {
                max_preprocessed
                    .clone()
                    .into_iter()
                    .chain(x)
                    .map(|(air, log_height)| (RwasmAirId::from_str(&air).unwrap(), log_height))
                    .collect::<Shape<RwasmAirId>>()
            })
            .filter(|shape| shape.log2_height(&RwasmAirId::Global).unwrap() < 21)
            .collect();

        self.maximal_core_shapes(max_log_shard_size).into_iter().chain(precompile_shapes).collect()
    }

    pub fn estimate_lde_size(&self, shape: &Shape<RwasmAirId>) -> usize {
        // println!("shape:{:?},",shape);
        shape.iter().map(|(air, height)| self.costs[air] * (1 << height) as usize).sum()
    }

    // TODO: cleanup..
    pub fn small_program_shapes(&self) -> Vec<OrderedShape> {
        self.partial_small_shapes
            .iter()
            .map(|log_heights| {
                OrderedShape::from_log2_heights(
                    &log_heights
                        .iter()
                        .filter(|(_, v)| v[0].is_some())
                        .map(|(k, v)| (k.to_string(), v.last().unwrap().unwrap()))
                        .chain(vec![
                            (MachineAir::<BabyBear>::name(&ProgramChip), 19),
                            (MachineAir::<BabyBear>::name(&ByteChip::default()), 16),
                            (MachineAir::<BabyBear>::name(&CallChip::default()), 16),
                        ])
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

impl<F: PrimeField32> Default for CoreShapeConfig<F> {
    fn default() -> Self {
        // Load the maximal shapes.
        let mut maximal_shapes: BTreeMap<usize, Vec<Shape<RwasmAirId>>> =
            serde_json::from_slice(MAXIMAL_SHAPES).unwrap();
        let small_shapes: Vec<Shape<RwasmAirId>> = serde_json::from_slice(SMALL_SHAPES).unwrap();
        let mut new_shapes = BTreeMap::new();
        for (i, vec) in maximal_shapes.iter() {
            let mut new_vec = vec![];
            for item in vec.iter() {
                let mut item = (*item).clone();
                item.inner.insert(RwasmAirId::Call, 14);
                new_vec.push(item);
            }
            new_shapes.insert(*i, new_vec);
        }
        maximal_shapes = new_shapes;
        // Set the allowed preprocessed log2 heights.
        let allowed_preprocessed_log2_heights = HashMap::from([
            (RwasmAirId::Program, vec![Some(19), Some(20), Some(21), Some(22)]),
            (RwasmAirId::Byte, vec![Some(16)]),
        ]);

        // Generate the clusters from the maximal shapes and register them indexed by log2 shard
        //  size.
        let blacklist = [
            27, 33, 47, 68, 75, 102, 104, 114, 116, 118, 137, 138, 139, 144, 145, 153, 155, 157,
            158, 169, 170, 171, 184, 185, 187, 195, 216, 243, 252, 275, 281, 282, 285,
        ];
        let mut core_allowed_log2_heights = BTreeMap::new();
        for (log2_shard_size, maximal_shapes) in maximal_shapes {
            let mut clusters = vec![];

            for (i, maximal_shape) in maximal_shapes.iter().enumerate() {
                // WARNING: This must be tuned carefully.
                //
                // This is current hardcoded, but in the future it should be computed dynamically.
                if log2_shard_size == 21 && blacklist.contains(&i) {
                    continue;
                }

                let cluster = derive_cluster_from_maximal_shape(maximal_shape);
                clusters.push(cluster);
            }

            core_allowed_log2_heights.insert(log2_shard_size, clusters);
        }

        // Set the memory init and finalize heights.
        let memory_allowed_log2_heights = HashMap::from(
            [
                (
                    RwasmAirId::MemoryGlobalInit,
                    vec![None, Some(10), Some(16), Some(18), Some(19), Some(20), Some(21)],
                ),
                (
                    RwasmAirId::MemoryGlobalFinalize,
                    vec![None, Some(10), Some(16), Some(18), Some(19), Some(20), Some(21)],
                ),
                (RwasmAirId::Global, vec![None, Some(11), Some(17), Some(19), Some(21), Some(22)]),
            ]
            .map(|(air, log_heights)| (air, log_heights)),
        );

        let mut precompile_allowed_log2_heights = HashMap::new();
        let precompile_heights = (3..21).collect::<Vec<_>>();
        for (air, memory_events_per_row) in
            RwasmAir::<F>::precompile_airs_with_memory_events_per_row()
        {
            precompile_allowed_log2_heights
                .insert(air, (memory_events_per_row, precompile_heights.clone()));
        }

        Self {
            partial_preprocessed_shapes: ShapeCluster::new(allowed_preprocessed_log2_heights),
            partial_core_shapes: core_allowed_log2_heights,
            partial_memory_shapes: ShapeCluster::new(memory_allowed_log2_heights),
            partial_precompile_shapes: precompile_allowed_log2_heights,
            partial_small_shapes: small_shapes
                .into_iter()
                .map(|x| {
                    ShapeCluster::new(x.into_iter().map(|(k, v)| (k, vec![Some(v)])).collect())
                })
                .collect(),
            costs: serde_json::from_str(include_str!("rv32im_costs.json")).unwrap(),
            _data: PhantomData,
        }
    }
}

fn derive_cluster_from_maximal_shape(shape: &Shape<RwasmAirId>) -> ShapeCluster<RwasmAirId> {
    // We first define a heuristic to derive the log heights from the maximal shape.
    let log2_gap_from_21 = 21 - shape.log2_height(&RwasmAirId::Cpu).unwrap();
    let min_log2_height_threshold = 18 - log2_gap_from_21;
    let log2_height_buffer = 10;
    let heuristic = |maximal_log2_height: Option<usize>, min_offset: usize| {
        if let Some(maximal_log2_height) = maximal_log2_height {
            let tallest_log2_height = std::cmp::max(maximal_log2_height, min_log2_height_threshold);
            let shortest_log2_height = tallest_log2_height.saturating_sub(min_offset);

            let mut range =
                (shortest_log2_height..=tallest_log2_height).map(Some).collect::<Vec<_>>();

            if shortest_log2_height > maximal_log2_height {
                range.insert(0, Some(shortest_log2_height));
            }

            range
        } else {
            vec![None, Some(log2_height_buffer)]
        }
    };

    let mut maybe_log2_heights = HashMap::new();

    let cpu_log_height = shape.log2_height(&RwasmAirId::Cpu);
    maybe_log2_heights.insert(RwasmAirId::Cpu, heuristic(cpu_log_height, 0));

    let addsub_log_height = shape.log2_height(&RwasmAirId::AddSub);
    maybe_log2_heights.insert(RwasmAirId::AddSub, heuristic(addsub_log_height, 0));

    let lt_log_height = shape.log2_height(&RwasmAirId::Lt);
    maybe_log2_heights.insert(RwasmAirId::Lt, heuristic(lt_log_height, 0));

    let memory_local_log_height = shape.log2_height(&RwasmAirId::MemoryLocal);
    maybe_log2_heights.insert(RwasmAirId::MemoryLocal, heuristic(memory_local_log_height, 0));

    let divrem_log_height = shape.log2_height(&RwasmAirId::DivRem);
    maybe_log2_heights.insert(RwasmAirId::DivRem, heuristic(divrem_log_height, 1));

    let bitwise_log_height = shape.log2_height(&RwasmAirId::Bitwise);
    maybe_log2_heights.insert(RwasmAirId::Bitwise, heuristic(bitwise_log_height, 1));

    let rotate_log_height = shape.log2_height(&RwasmAirId::Rotate);
    maybe_log2_heights.insert(RwasmAirId::Rotate, heuristic(rotate_log_height, 1));

    let trialing_log_height = shape.log2_height(&RwasmAirId::Trailing);
    maybe_log2_heights.insert(RwasmAirId::Trailing, heuristic(trialing_log_height, 1));

    let extend_log_height = shape.log2_height(&RwasmAirId::Extend);
    maybe_log2_heights.insert(RwasmAirId::Extend, heuristic(extend_log_height, 1));

    let addmul64_log_height = shape.log2_height(&RwasmAirId::AddMul64);
    maybe_log2_heights.insert(RwasmAirId::AddMul64, heuristic(addmul64_log_height, 1));

    let mul_log_height = shape.log2_height(&RwasmAirId::Mul);
    maybe_log2_heights.insert(RwasmAirId::Mul, heuristic(mul_log_height, 1));

    let shift_right_log_height = shape.log2_height(&RwasmAirId::ShiftRight);
    maybe_log2_heights.insert(RwasmAirId::ShiftRight, heuristic(shift_right_log_height, 1));

    let shift_left_log_height = shape.log2_height(&RwasmAirId::ShiftLeft);
    maybe_log2_heights.insert(RwasmAirId::ShiftLeft, heuristic(shift_left_log_height, 1));

    let memory_instrs_log_height = shape.log2_height(&RwasmAirId::MemoryInstrs);
    maybe_log2_heights.insert(RwasmAirId::MemoryInstrs, heuristic(memory_instrs_log_height, 0));

    let auipc_log_height = shape.log2_height(&RwasmAirId::Auipc);
    maybe_log2_heights.insert(RwasmAirId::Auipc, heuristic(auipc_log_height, 0));

    let branch_log_height = shape.log2_height(&RwasmAirId::Branch);
    maybe_log2_heights.insert(RwasmAirId::Branch, heuristic(branch_log_height, 0));
    let fuel_log_height = shape.log2_height(&RwasmAirId::Fuel);
    maybe_log2_heights.insert(RwasmAirId::Fuel, heuristic(fuel_log_height, 0));
    let call_log_height = shape.log2_height(&RwasmAirId::Call);
    maybe_log2_heights.insert(RwasmAirId::Call, heuristic(call_log_height, 0));

    let jump_log_height = shape.log2_height(&RwasmAirId::Jump);
    maybe_log2_heights.insert(RwasmAirId::Jump, heuristic(jump_log_height, 0));

    let syscall_core_log_height = shape.log2_height(&RwasmAirId::SyscallCore);
    maybe_log2_heights.insert(RwasmAirId::SyscallCore, heuristic(syscall_core_log_height, 0));

    let syscall_instrs_log_height = shape.log2_height(&RwasmAirId::SyscallInstrs);
    maybe_log2_heights.insert(RwasmAirId::SyscallInstrs, heuristic(syscall_instrs_log_height, 0));

    let global_log_height = shape.log2_height(&RwasmAirId::Global);
    maybe_log2_heights.insert(RwasmAirId::Global, heuristic(global_log_height, 1));

    let const_log_height = shape.log2_height(&RwasmAirId::Const);
    maybe_log2_heights.insert(RwasmAirId::Const, heuristic(const_log_height, 1));

    let local_log_height = shape.log2_height(&RwasmAirId::Local);
    maybe_log2_heights.insert(RwasmAirId::Local, heuristic(local_log_height, 1));

    let table_grow_log_height = shape.log2_height(&RwasmAirId::TableGrow);
    maybe_log2_heights.insert(RwasmAirId::TableGrow, heuristic(table_grow_log_height, 1));

    let table_init_log_height = shape.log2_height(&RwasmAirId::TableInit);
    maybe_log2_heights.insert(RwasmAirId::TableInit, heuristic(table_init_log_height, 1));

    let params_check_log_height = shape.log2_height(&RwasmAirId::ParamsCheck);
    maybe_log2_heights.insert(RwasmAirId::ParamsCheck, heuristic(params_check_log_height, 1));

    assert!(maybe_log2_heights.len() >= shape.len(), "not all chips were included in the shape");

    ShapeCluster::new(maybe_log2_heights)
}

#[derive(Debug, Error)]
pub enum CoreShapeError {
    #[error("no preprocessed shape found")]
    PreprocessedShapeError,
    #[error("Preprocessed shape already fixed")]
    PreprocessedShapeAlreadyFixed,
    #[error("no shape found {0:?}")]
    ShapeError(HashMap<String, usize>),
    #[error("Preprocessed shape missing")]
    PreprocessedShapeMissing,
    #[error("Shape already fixed")]
    ShapeAlreadyFixed,
    #[error("Precompile not included in allowed shapes {0:?}")]
    PrecompileNotIncluded(HashMap<String, usize>),
}

pub fn create_dummy_program(shape: &Shape<RwasmAirId>) -> Program {
    let mut program = Program::from_instrs(vec![]);
    program.preprocessed_shape = Some(shape.clone());
    program
}

pub fn create_dummy_record(shape: &Shape<RwasmAirId>) -> ExecutionRecord {
    let program = std::sync::Arc::new(create_dummy_program(shape));
    let mut record = ExecutionRecord::new(program);
    record.shape = Some(shape.clone());
    record
}

#[cfg(test)]
pub mod tests {
    #![allow(clippy::print_stdout)]

    use hashbrown::HashSet;
    use sp1_stark::{Dom, MachineProver, StarkGenericConfig};

    use super::*;

    fn try_generate_dummy_proof<SC: StarkGenericConfig, P: MachineProver<SC, RwasmAir<SC::Val>>>(
        prover: &P,
        shape: &Shape<RwasmAirId>,
    ) where
        SC::Val: PrimeField32,
        Dom<SC>: core::fmt::Debug,
    {
        let program = create_dummy_program(shape);
        let record = create_dummy_record(shape);

        // Try doing setup.
        let (pk, _) = prover.setup(&program);

        // Try to generate traces.
        let main_traces = prover.generate_traces(&record);

        // Try to commit the traces.
        let main_data = prover.commit(&record, main_traces);

        let mut challenger = prover.machine().config().challenger();

        // Try to "open".
        prover.open(&pk, main_data, &mut challenger).unwrap();
    }

    #[test]
    #[ignore]
    fn test_making_shapes() {
        use p3_baby_bear::BabyBear;
        let shape_config = CoreShapeConfig::<BabyBear>::default();
        let num_shapes = shape_config.all_shapes().collect::<HashSet<_>>().len();
        println!("There are {} core shapes", num_shapes);
        assert!(num_shapes < 1 << 24);
    }

    #[test]
    fn test_dummy_record() {
        use crate::utils::setup_logger;
        use p3_baby_bear::BabyBear;
        use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, CpuProver};

        type SC = BabyBearPoseidon2;
        type A = RwasmAir<BabyBear>;

        setup_logger();

        let preprocessed_log_heights = [(RwasmAirId::Program, 10), (RwasmAirId::Byte, 16)];

        let core_log_heights = [
            (RwasmAirId::Cpu, 11),
            (RwasmAirId::DivRem, 11),
            (RwasmAirId::AddSub, 10),
            (RwasmAirId::Bitwise, 10),
            (RwasmAirId::Mul, 10),
            (RwasmAirId::AddMul64, 12),
            (RwasmAirId::ShiftRight, 10),
            (RwasmAirId::ShiftLeft, 10),
            (RwasmAirId::Lt, 10),
            (RwasmAirId::MemoryLocal, 10),
            (RwasmAirId::SyscallCore, 10),
            (RwasmAirId::Global, 10),
        ];

        let height_map =
            preprocessed_log_heights.into_iter().chain(core_log_heights).collect::<HashMap<_, _>>();

        let shape = Shape::new(height_map);

        // Try generating preprocessed traces.
        let config = SC::default();
        let machine = A::machine(config);
        let prover = CpuProver::new(machine);

        try_generate_dummy_proof(&prover, &shape);
    }
}
