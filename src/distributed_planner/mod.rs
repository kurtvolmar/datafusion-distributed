mod batch_coalescing_below_network_boundaries;
mod distributed_config;
mod distributed_physical_optimizer_rule;
mod insert_broadcast;
mod network_boundary;
mod plan_annotator;
mod task_estimator;

pub(crate) use batch_coalescing_below_network_boundaries::batch_coalescing_below_network_boundaries;
pub use distributed_config::DistributedConfig;
pub use distributed_physical_optimizer_rule::DistributedPhysicalOptimizerRule;
pub use network_boundary::{NetworkBoundary, NetworkBoundaryExt};
pub(crate) use task_estimator::set_distributed_task_estimator;
pub use task_estimator::{TaskCountAnnotation, TaskEstimation, TaskEstimator};

// @NetworkBoundaryStrategy: new module and re-exports for pluggable network boundary strategies.
mod network_boundary_strategy;

pub(crate) use network_boundary_strategy::set_distributed_network_boundary_strategy;

#[rustfmt::skip]
pub use network_boundary_strategy::{
    CombinedNetworkBoundaryStrategy, NetworkBoundaryAnnotation, NetworkBoundaryContext,
    NetworkBoundaryStrategy,
};
pub use plan_annotator::PlanOrNetworkBoundary;
