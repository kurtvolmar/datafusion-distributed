// @NetworkBoundaryStrategy: new module for pluggable network boundary strategies so custom strategies can override or extend default boundary placement.
use crate::DistributedConfig;
use crate::common::require_one_child;
use crate::distributed_planner::plan_annotator::PlanOrNetworkBoundary;
use datafusion::common::plan_err;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::ExecutionPlan;
use std::fmt::Debug;
use std::ops::AddAssign;
use std::sync::Arc;
use uuid::Uuid;

/// Annotation metadata about a network boundary for a plan node.
///
/// Returned by [`NetworkBoundaryStrategy::annotate_network_boundary`] to describe
/// what kind of network boundary (if any) is needed and optional hints about
/// the output task count.
#[derive(Debug, Clone)]
pub struct NetworkBoundaryAnnotation {
    /// The type of network boundary required (Shuffle, Coalesce, Broadcast, Extension).
    /// If None, no network boundary is needed for this plan node.
    pub required_network_boundary: Option<PlanOrNetworkBoundary>,

    /// Optional hint for the output task count after this boundary is applied.
    ///
    /// - If `None`, DFD will calculate the output task count using default cardinality scaling.
    /// - If `Some(n)`, DFD will use `n` as the output task count for this stage.
    ///
    /// This allows strategies which know their output task count to override the generic
    /// cardinality-based calculation in the plan annotator.
    pub output_tasks: Option<usize>,
}

/// Context provided to network boundary strategies when deciding how to place boundaries.
///
/// This struct contains all the information a strategy needs to make decisions about
/// how to transform the plan at a network boundary point.
#[derive(Debug)]
pub struct NetworkBoundaryContext<'a> {
    /// The type of network boundary required at this point.
    pub boundary_type: &'a PlanOrNetworkBoundary,
    /// The already-distributed children to be wrapped by the network boundary.
    pub new_children: Arc<dyn ExecutionPlan>,
    /// The query ID for this execution.
    pub query_id: Uuid,
    /// The stage ID for the boundary being created.
    pub stage_id: usize,
    /// Number of tasks in the current stage (above the boundary).
    pub task_count: usize,
    /// Number of tasks in the input stage (below the boundary).
    pub input_task_count: usize,
    /// The DataFusion configuration options.
    pub config: &'a ConfigOptions,
}

/// Strategy for placing network boundaries in a distributed execution plan.
///
/// When a network boundary is needed (e.g., after hash repartition or before coalesce),
/// strategies are invoked in order. The first strategy to return annotation with a boundary wins.
///
/// Strategies should return `None` to defer to the next strategy in the chain.
/// Custom strategies can be registered to override default behavior.
pub trait NetworkBoundaryStrategy: Debug + Send + Sync {
    /// Annotates a plan node with network boundary metadata.
    ///
    /// Returns `Some(NetworkBoundaryAnnotation)` if this strategy detects a boundary is needed,
    /// or `None` to defer to the next strategy.
    ///
    /// The annotation can optionally include an `output_tasks` hint to override DFD's
    /// default task count calculation.
    fn annotate_network_boundary(
        &self,
        plan: &dyn ExecutionPlan,
    ) -> Option<NetworkBoundaryAnnotation>;

    /// Apply this strategy to place a network boundary. Return `Ok(None)` to defer to next strategy.
    fn apply_boundary(
        &self,
        context: &NetworkBoundaryContext<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>;
}

/// Combines multiple [`NetworkBoundaryStrategy`] implementations.
///
/// Strategies are tried in order for both annotation and boundary application.
/// The first strategy to return an annotation with a boundary will be used.
#[derive(Clone)]
pub struct CombinedNetworkBoundaryStrategy {
    pub(crate) strategies: Vec<Arc<dyn NetworkBoundaryStrategy>>,
}

impl From<Vec<Arc<dyn NetworkBoundaryStrategy>>> for CombinedNetworkBoundaryStrategy {
    fn from(strategies: Vec<Arc<dyn NetworkBoundaryStrategy>>) -> Self {
        Self { strategies }
    }
}

impl Default for CombinedNetworkBoundaryStrategy {
    fn default() -> Self {
        Self { strategies: vec![] }
    }
}

impl Debug for CombinedNetworkBoundaryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedNetworkBoundaryStrategy")
            .field("strategies_count", &self.strategies.len())
            .finish()
    }
}

impl NetworkBoundaryStrategy for CombinedNetworkBoundaryStrategy {
    fn annotate_network_boundary(
        &self,
        plan: &dyn ExecutionPlan,
    ) -> Option<NetworkBoundaryAnnotation> {
        for strategy in &self.strategies {
            if let Some(annotation) = strategy.annotate_network_boundary(plan) {
                return Some(annotation);
            }
        }
        None
    }

    fn apply_boundary(
        &self,
        context: &NetworkBoundaryContext<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        for strategy in &self.strategies {
            if let Some(result) = strategy.apply_boundary(context)? {
                return Ok(Some(result));
            }
        }
        Ok(None)
    }
}

/// Applies an extension network boundary by building context and delegating to the configured
/// strategy. Used by the distributed physical optimizer rule when it encounters
/// `PlanOrNetworkBoundary::Extension`.
pub(crate) fn apply_extension_boundary(
    d_cfg: &DistributedConfig,
    boundary_type: &PlanOrNetworkBoundary,
    new_children: Vec<Arc<dyn ExecutionPlan>>,
    query_id: Uuid,
    stage_id: &mut usize,
    task_count: usize,
    max_child_task_count: Option<usize>,
    cfg: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    let context = NetworkBoundaryContext {
        boundary_type,
        new_children: require_one_child(new_children)?,
        query_id,
        stage_id: *stage_id,
        task_count,
        input_task_count: max_child_task_count.unwrap_or(1),
        config: cfg,
    };
    match d_cfg
        .__private_network_boundary_strategy
        .apply_boundary(&context)?
    {
        Some(custom_plan) => {
            // TODO: revisit the stage incrementation, since a strategy can insert 0 to many stages.
            stage_id.add_assign(1);
            Ok(custom_plan)
        }
        None => plan_err!("Extension boundary not handled by any strategy"),
    }
}

/// Runs network boundary strategies after default detection. Returns an optional boundary type
/// and an optional task count. When the task count is Some, the annotator should use it and return early.
pub(crate) fn apply_network_boundary_strategy(
    d_cfg: &DistributedConfig,
    plan: &Arc<dyn ExecutionPlan>,
) -> (
    Option<PlanOrNetworkBoundary>,
    Option<crate::TaskCountAnnotation>,
) {
    let strategy_annotation = d_cfg
        .__private_network_boundary_strategy
        .annotate_network_boundary(plan.as_ref());
    let boundary = strategy_annotation
        .as_ref()
        .and_then(|a| a.required_network_boundary.clone());
    let task_count =
        strategy_annotation.and_then(|a| a.output_tasks.map(crate::TaskCountAnnotation::Desired));
    (boundary, task_count)
}

/// Helper function to add a network boundary strategy to the session config.
/// This is used by the DistributedExt trait implementation.
pub(crate) fn set_distributed_network_boundary_strategy(
    cfg: &mut datafusion::prelude::SessionConfig,
    strategy: impl NetworkBoundaryStrategy + 'static,
) {
    use crate::config_extension_ext::set_distributed_option_extension;
    use crate::distributed_planner::DistributedConfig;

    let opts = cfg.options_mut();
    if let Some(distributed_cfg) = opts.extensions.get_mut::<DistributedConfig>() {
        distributed_cfg
            .__private_network_boundary_strategy
            .strategies
            .push(Arc::new(strategy));
    } else {
        let mut combined = CombinedNetworkBoundaryStrategy::default();
        combined.strategies.push(Arc::new(strategy));
        set_distributed_option_extension(
            cfg,
            DistributedConfig {
                __private_network_boundary_strategy: combined,
                ..Default::default()
            },
        );
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::distributed_planner::insert_broadcast::insert_broadcast_execs;
    use crate::distributed_planner::plan_annotator::{PlanOrNetworkBoundary, annotate_plan};
    use crate::test_utils::in_memory_channel_resolver::InMemoryWorkerResolver;
    use crate::test_utils::plans::{TestPlanOptions, base_session_builder, context_with_query};
    use crate::{
        DistributedConfig, DistributedExt, DistributedPhysicalOptimizerRule, assert_snapshot,
        display_plan_ascii,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::SessionStateBuilder;
    use datafusion::physical_plan::displayable;
    use datafusion::physical_plan::empty::EmptyExec;

    #[test]
    fn test_combined_network_strategy_first_strategy_wins() {
        use datafusion::physical_plan::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;

        let mut combined = CombinedNetworkBoundaryStrategy::default();
        // Add two PassthroughStrategy instances with different extension names
        combined.strategies.insert(
            0,
            Arc::new(PassthroughStrategy::new(
                |plan| {
                    plan.as_any()
                        .downcast_ref::<RepartitionExec>()
                        .map(|repartition| {
                            matches!(repartition.partitioning(), Partitioning::Hash(_, _))
                        })
                        .unwrap_or(false)
                },
                "wrap_hash_repartition_0",
                3,
            )),
        );
        combined.strategies.insert(
            1,
            Arc::new(PassthroughStrategy::new(
                |plan| {
                    plan.as_any()
                        .downcast_ref::<RepartitionExec>()
                        .map(|repartition| {
                            matches!(repartition.partitioning(), Partitioning::Hash(_, _))
                        })
                        .unwrap_or(false)
                },
                "wrap_hash_repartition_1",
                3,
            )),
        );

        // Test with a Hash RepartitionExec plan
        let plan = hash_repartition_plan();
        let result = combined.annotate_network_boundary(plan.as_ref());

        // First strategy should win and return Extension("wrap_hash_repartition_0")
        assert!(matches!(
            result
                .as_ref()
                .and_then(|a| a.required_network_boundary.as_ref()),
            Some(PlanOrNetworkBoundary::Extension("wrap_hash_repartition_0"))
        ));
        assert_eq!(result.as_ref().and_then(|a| a.output_tasks), Some(3));
    }

    #[test]
    fn test_combined_network_strategy_continues_until_match() {
        use datafusion::physical_plan::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;

        let mut combined = CombinedNetworkBoundaryStrategy::default();
        // First strategy matches EmptyExec (won't match Hash RepartitionExec)
        combined.strategies.insert(
            0,
            Arc::new(PassthroughStrategy::new(
                |plan| plan.as_any().is::<EmptyExec>(),
                "wrap_empty_exec",
                3,
            )),
        );
        // Second strategy matches Hash RepartitionExec
        combined.strategies.insert(
            1,
            Arc::new(PassthroughStrategy::new(
                |plan| {
                    plan.as_any()
                        .downcast_ref::<RepartitionExec>()
                        .map(|repartition| {
                            matches!(repartition.partitioning(), Partitioning::Hash(_, _))
                        })
                        .unwrap_or(false)
                },
                "wrap_hash_repartition",
                3,
            )),
        );

        // Test with Hash RepartitionExec - first strategy won't match, second strategy should match
        let plan = hash_repartition_plan();
        let result = combined.annotate_network_boundary(plan.as_ref());
        assert!(matches!(
            result
                .as_ref()
                .and_then(|a| a.required_network_boundary.as_ref()),
            Some(PlanOrNetworkBoundary::Extension("wrap_hash_repartition"))
        ));
    }

    #[tokio::test]
    async fn test_extension_boundary_strategy() {
        use datafusion::physical_plan::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;

        let query = r#"
        SELECT count(*), "RainToday" FROM weather GROUP BY "RainToday" ORDER BY count(*)
        "#;
        let annotated = annotate_test_plan(query, TestPlanOptions::default(), |b| {
            let mut state = b.build();
            let config = state.config_mut();
            let d_cfg = DistributedConfig::from_config_options_mut(config.options_mut()).unwrap();
            d_cfg.__private_network_boundary_strategy.strategies =
                vec![Arc::new(PassthroughStrategy::new(
                    |plan| {
                        plan.as_any()
                            .downcast_ref::<RepartitionExec>()
                            .map(|repartition| {
                                matches!(repartition.partitioning(), Partitioning::Hash(_, _))
                            })
                            .unwrap_or(false)
                    },
                    "wrap_hash_repartition",
                    3,
                )) as Arc<dyn NetworkBoundaryStrategy>];
            SessionStateBuilder::new_from_existing(state)
        })
        .await;
        assert_snapshot!(annotated, @r"
        ProjectionExec: task_count=Maximum(1)
          SortPreservingMergeExec: task_count=Maximum(1)
            [NetworkBoundary] Coalesce: task_count=Maximum(1)
              SortExec: task_count=Desired(3)
                ProjectionExec: task_count=Desired(3)
                  AggregateExec: task_count=Desired(3)
                    [NetworkBoundary] Extension(wrap_hash_repartition): task_count=Desired(3)
                      RepartitionExec: task_count=Desired(3)
                        AggregateExec: task_count=Desired(3)
                          DataSourceExec: task_count=Desired(3)
        ")
    }

    #[tokio::test]
    async fn test_extension_boundary_with_custom_strategy() {
        use datafusion::physical_plan::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;

        let query = r#"
        SELECT count(*), "RainToday" FROM weather GROUP BY "RainToday" ORDER BY count(*)
        "#;
        let plan = explain_test_plan(query, TestPlanOptions::default(), true, |b| {
            let mut state = b.build();
            let config = state.config_mut();
            let d_cfg = DistributedConfig::from_config_options_mut(config.options_mut()).unwrap();
            d_cfg.__private_network_boundary_strategy.strategies =
                vec![Arc::new(PassthroughStrategy::new(
                    |plan| {
                        plan.as_any()
                            .downcast_ref::<RepartitionExec>()
                            .map(|repartition| {
                                matches!(repartition.partitioning(), Partitioning::Hash(_, _))
                            })
                            .unwrap_or(false)
                    },
                    "wrap_hash_repartition",
                    3,
                )) as Arc<dyn NetworkBoundaryStrategy>];
            SessionStateBuilder::new_from_existing(state)
                .with_distributed_worker_resolver(InMemoryWorkerResolver::new(3))
        })
        .await;

        assert_snapshot!(plan, @r"
        ┌───── DistributedExec ── Tasks: t0:[p0] 
        │ ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday]
        │   SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST]
        │     [Stage 2] => NetworkCoalesceExec: output_partitions=12, input_tasks=3
        └──────────────────────────────────────────────────
          ┌───── Stage 2 ── Tasks: t0:[p0..p3] t1:[p0..p3] t2:[p0..p3] 
          │ SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true]
          │   ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))]
          │     AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
          │       PassthroughExec
          │         RepartitionExec: partitioning=Hash([RainToday@0], 4), input_partitions=1
          │           AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
          │             PartitionIsolatorExec: t0:[p0,__,__] t1:[__,p0,__] t2:[__,__,p0]
          │               DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
          └──────────────────────────────────────────────────
        ");
    }

    // --- Helpers (duplicated from plan_annotator and distributed_physical_optimizer_rule
    // test modules to avoid tail-of-file merge conflicts when upstream adds tests there) ---

    fn hash_repartition_plan() -> Arc<dyn ExecutionPlan> {
        use datafusion::physical_expr::expressions::Column as PhysicalColumn;
        use datafusion::physical_plan::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let child = Arc::new(EmptyExec::new(schema.clone()));
        let partitioning = Partitioning::Hash(
            vec![Arc::new(PhysicalColumn::new("a", 0))
                as Arc<dyn datafusion::physical_expr::PhysicalExpr>],
            4,
        );
        Arc::new(RepartitionExec::try_new(child, partitioning).unwrap())
    }

    /// Test strategy that wraps specific ExecutionPlan nodes with a PassthroughExec.
    /// Useful for testing custom boundary insertion without transformation.
    #[derive(Clone)]
    pub struct PassthroughStrategy {
        /// Function to check if a plan matches the target type
        matcher: Arc<dyn Fn(&dyn ExecutionPlan) -> bool + Send + Sync>,
        /// Extension name to use for the boundary
        extension_name: &'static str,
        /// Number of output tasks
        output_tasks: usize,
    }

    impl std::fmt::Debug for PassthroughStrategy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PassthroughStrategy")
                .field("extension_name", &self.extension_name)
                .field("output_tasks", &self.output_tasks)
                .finish()
        }
    }

    impl PassthroughStrategy {
        /// Creates a new PassthroughStrategy with a custom matcher function
        pub fn new<F>(matcher: F, extension_name: &'static str, output_tasks: usize) -> Self
        where
            F: Fn(&dyn ExecutionPlan) -> bool + Send + Sync + 'static,
        {
            Self {
                matcher: Arc::new(matcher),
                extension_name,
                output_tasks,
            }
        }
    }

    impl NetworkBoundaryStrategy for PassthroughStrategy {
        fn annotate_network_boundary(
            &self,
            plan: &dyn ExecutionPlan,
        ) -> Option<NetworkBoundaryAnnotation> {
            if (self.matcher)(plan) {
                Some(NetworkBoundaryAnnotation {
                    required_network_boundary: Some(PlanOrNetworkBoundary::Extension(
                        self.extension_name,
                    )),
                    output_tasks: Some(self.output_tasks),
                })
            } else {
                None
            }
        }

        fn apply_boundary(
            &self,
            context: &NetworkBoundaryContext<'_>,
        ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
            if let PlanOrNetworkBoundary::Extension(name) = context.boundary_type {
                if *name == self.extension_name {
                    // Wrap with PassthroughExec to demonstrate custom boundary insertion
                    return Ok(Some(Arc::new(PassthroughExec::new(
                        context.new_children.clone(),
                    ))));
                }
            }
            Ok(None)
        }
    }

    /// A no-op ExecutionPlan that passes through data from its child unchanged.
    /// Useful for testing scenarios where you need a wrapper node that doesn't transform data.
    #[derive(Debug)]
    pub struct PassthroughExec {
        child: Arc<dyn ExecutionPlan>,
    }

    impl PassthroughExec {
        pub fn new(child: Arc<dyn ExecutionPlan>) -> Self {
            Self { child }
        }
    }

    impl ExecutionPlan for PassthroughExec {
        fn name(&self) -> &str {
            "PassthroughExec"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
            self.child.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.child]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(children[0].clone())))
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<datafusion::execution::TaskContext>,
        ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
            // Simply delegate to the child
            self.child.execute(partition, context)
        }
    }

    impl std::fmt::Display for PassthroughExec {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "PassthroughExec")
        }
    }

    impl datafusion::physical_plan::DisplayAs for PassthroughExec {
        fn fmt_as(
            &self,
            _t: datafusion::physical_plan::DisplayFormatType,
            f: &mut std::fmt::Formatter,
        ) -> std::fmt::Result {
            write!(f, "PassthroughExec")
        }
    }

    async fn annotate_test_plan(
        query: &str,
        options: TestPlanOptions,
        configure: impl FnOnce(SessionStateBuilder) -> SessionStateBuilder,
    ) -> String {
        let builder = base_session_builder(
            options.target_partitions,
            options.num_workers,
            options.broadcast_enabled,
        );
        let builder = configure(builder);
        let (ctx, query) = context_with_query(builder, query).await;
        let df = ctx.sql(&query).await.unwrap();
        let mut plan = df.create_physical_plan().await.unwrap();

        plan = insert_broadcast_execs(plan, ctx.state_ref().read().config_options().as_ref())
            .expect("failed to insert broadcasts");

        let annotated = annotate_plan(plan, ctx.state_ref().read().config_options().as_ref())
            .expect("failed to annotate plan");
        format!("{annotated:?}")
    }

    async fn explain_test_plan(
        query: &str,
        options: TestPlanOptions,
        use_optimizer: bool,
        configure: impl FnOnce(SessionStateBuilder) -> SessionStateBuilder,
    ) -> String {
        let mut builder = base_session_builder(
            options.target_partitions,
            options.num_workers,
            options.broadcast_enabled,
        );
        if use_optimizer {
            builder =
                builder.with_physical_optimizer_rule(Arc::new(DistributedPhysicalOptimizerRule));
        }
        let builder = configure(builder);
        let (ctx, query) = context_with_query(builder, query).await;
        let df = ctx.sql(&query).await.unwrap();
        let physical_plan = df.create_physical_plan().await.unwrap();

        if use_optimizer {
            display_plan_ascii(physical_plan.as_ref(), false)
        } else {
            format!("{}", displayable(physical_plan.as_ref()).indent(true))
        }
    }
}
