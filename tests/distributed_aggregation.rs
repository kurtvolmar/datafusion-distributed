#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::physical_plan::{displayable, execute_stream};
    use datafusion::prelude::SessionContext;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::test_utils::parquet::register_parquet_tables;
    use datafusion_distributed::test_utils::session_context::register_temp_parquet_table;
    use datafusion_distributed::{DefaultSessionBuilder, assert_snapshot, display_plan_ascii};
    use futures::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;
    use uuid::Uuid;

    #[tokio::test]
    async fn distributed_aggregation() -> Result<(), Box<dyn Error>> {
        let (ctx_distributed, _guard, _) = start_localhost_context(3, DefaultSessionBuilder).await;

        let query =
            r#"SELECT count(*), "RainToday" FROM weather GROUP BY "RainToday" ORDER BY count(*)"#;

        let ctx = SessionContext::default();
        *ctx.state_ref().write().config_mut() = ctx_distributed.copied_config();
        register_parquet_tables(&ctx).await?;
        let df = ctx.sql(query).await?;
        let physical = df.create_physical_plan().await?;
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();

        register_parquet_tables(&ctx_distributed).await?;
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        assert_snapshot!(physical_str,
            @r"
        ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday]
          SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST]
            SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true]
              ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))]
                AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
                  RepartitionExec: partitioning=Hash([RainToday@0], 3), input_partitions=3
                    AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
                      DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
        ",
        );

        assert_snapshot!(physical_distributed_str,
            @r"
        ┌───── DistributedExec ── Tasks: t0:[p0]
        │ ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday]
        │   SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST]
        │     [Stage 2] => NetworkCoalesceExec: output_partitions=6, input_tasks=2
        └──────────────────────────────────────────────────
          ┌───── Stage 2 ── Tasks: t0:[p0..p2] t1:[p0..p2]
          │ SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true]
          │   ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))]
          │     AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
          │       [Stage 1] => NetworkShuffleExec: output_partitions=3, input_tasks=3
          └──────────────────────────────────────────────────
            ┌───── Stage 1 ── Tasks: t0:[p0..p5] t1:[p0..p5] t2:[p0..p5]
            │ RepartitionExec: partitioning=Hash([RainToday@0], 6), input_partitions=3
            │   AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
            │     DistributedLeafExec: DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
            └──────────────────────────────────────────────────
        ",
        );

        let batches = pretty_format_batches(
            &execute_stream(physical, ctx.task_ctx())?
                .try_collect::<Vec<_>>()
                .await?,
        )?;

        assert_snapshot!(batches, @r"
        +----------+-----------+
        | count(*) | RainToday |
        +----------+-----------+
        | 66       | Yes       |
        | 300      | No        |
        +----------+-----------+
        ");

        let batches_distributed = pretty_format_batches(
            &execute_stream(physical_distributed, ctx.task_ctx())?
                .try_collect::<Vec<_>>()
                .await?,
        )?;
        assert_snapshot!(batches_distributed, @r"
        +----------+-----------+
        | count(*) | RainToday |
        +----------+-----------+
        | 66       | Yes       |
        | 300      | No        |
        +----------+-----------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn distributed_aggregation_head_node_partitioned() -> Result<(), Box<dyn Error>> {
        let (ctx_distributed, _guard, _) = start_localhost_context(6, DefaultSessionBuilder).await;

        let query = r#"SELECT count(*), "RainToday" FROM weather GROUP BY "RainToday""#;

        let ctx = SessionContext::default();
        *ctx.state_ref().write().config_mut() = ctx_distributed.copied_config();
        register_parquet_tables(&ctx).await?;
        let df = ctx.sql(query).await?;
        let physical = df.create_physical_plan().await?;
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();

        register_parquet_tables(&ctx_distributed).await?;
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        assert_snapshot!(physical_str,
            @r"
        ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday]
          AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
            RepartitionExec: partitioning=Hash([RainToday@0], 3), input_partitions=3
              AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
                DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
        ",
        );

        assert_snapshot!(physical_distributed_str,
            @r"
        ┌───── DistributedExec ── Tasks: t0:[p0]
        │ CoalescePartitionsExec
        │   [Stage 2] => NetworkCoalesceExec: output_partitions=6, input_tasks=2
        └──────────────────────────────────────────────────
          ┌───── Stage 2 ── Tasks: t0:[p0..p2] t1:[p0..p2]
          │ ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday]
          │   AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
          │     [Stage 1] => NetworkShuffleExec: output_partitions=3, input_tasks=3
          └──────────────────────────────────────────────────
            ┌───── Stage 1 ── Tasks: t0:[p0..p5] t1:[p0..p5] t2:[p0..p5]
            │ RepartitionExec: partitioning=Hash([RainToday@0], 6), input_partitions=3
            │   AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
            │     DistributedLeafExec: DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
            └──────────────────────────────────────────────────
        ",
        );

        Ok(())
    }

    /// Test that multiple first_value() aggregations work correctly in distributed queries.
    // TODO: Once https://github.com/apache/datafusion/pull/18303 is merged, this test will lose
    //       meaning, since the PR above will mask the underlying problem. Different queries or
    //       a new approach must be used in this case.
    #[tokio::test]
    async fn test_multiple_first_value_aggregations() -> Result<(), Box<dyn Error>> {
        let (ctx, _guard, _) = start_localhost_context(3, DefaultSessionBuilder).await;

        let schema = Arc::new(Schema::new(vec![
            Field::new("group_id", DataType::Int32, false),
            Field::new("trace_id", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        // Create 2 batches that will be stored as separate parquet files
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["trace1", "trace2"])),
                Arc::new(Int32Array::from(vec![100, 200])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![3, 4])),
                Arc::new(StringArray::from(vec!["trace3", "trace4"])),
                Arc::new(Int32Array::from(vec![300, 400])),
            ],
        )?;

        let file1 =
            register_temp_parquet_table("records_part1", schema.clone(), vec![batch1], &ctx)
                .await?;
        let file2 =
            register_temp_parquet_table("records_part2", schema.clone(), vec![batch2], &ctx)
                .await?;

        // Create a partitioned table by registering multiple files
        let temp_dir = std::env::temp_dir();
        let table_dir = temp_dir.join(format!("partitioned_table_{}", Uuid::new_v4()));
        std::fs::create_dir(&table_dir)?;
        std::fs::copy(&file1, table_dir.join("part1.parquet"))?;
        std::fs::copy(&file2, table_dir.join("part2.parquet"))?;

        // Register the directory as a partitioned table
        ctx.register_parquet(
            "records_partitioned",
            table_dir.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await?;

        let query = r#"SELECT group_id, first_value(trace_id) AS fv1, first_value(value) AS fv2
                       FROM records_partitioned 
                       GROUP BY group_id 
                       ORDER BY group_id"#;

        let df = ctx.sql(query).await?;
        let physical = df.create_physical_plan().await?;

        // Execute distributed query
        let batches_distributed = execute_stream(physical, ctx.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;

        let actual_result = pretty_format_batches(&batches_distributed)?;
        let expected_result = "\
+----------+--------+-----+
| group_id | fv1    | fv2 |
+----------+--------+-----+
| 1        | trace1 | 100 |
| 2        | trace2 | 200 |
| 3        | trace3 | 300 |
| 4        | trace4 | 400 |
+----------+--------+-----+";

        // Print them out, the error message from `assert_eq` is otherwise hard to read.
        println!("{expected_result}");
        println!("{actual_result}");

        // Compare against result. The regression this is testing for would have NULL values in
        // the second and third column.
        assert_eq!(actual_result.to_string(), expected_result,);

        Ok(())
    }

    #[tokio::test]
    async fn distributed_aggregation_explain_analyze() -> Result<(), Box<dyn Error>> {
        let (ctx_distributed, _guard, _) = start_localhost_context(3, DefaultSessionBuilder).await;

        let query =
            r#"EXPLAIN ANALYZE SELECT count(*), "RainToday" FROM weather GROUP BY "RainToday" ORDER BY count(*)"#;

        let ctx = SessionContext::default();
        *ctx.state_ref().write().config_mut() = ctx_distributed.copied_config();
        register_parquet_tables(&ctx).await?;
        let df = ctx.sql(query).await?;
        let physical = df.create_physical_plan().await?;
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();

        register_parquet_tables(&ctx_distributed).await?;
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        assert_snapshot!(physical_str,
            @"
        AnalyzeExec verbose=false
          ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday]
            SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST]
              SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true]
                ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))]
                  AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
                    RepartitionExec: partitioning=Hash([RainToday@0], 3), input_partitions=3
                      AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
                        DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
        ",
        );

        assert_snapshot!(physical_distributed_str,
            @"
        ┌───── DistributedExec ── Tasks: t0:[p0]
        │ AnalyzeExec verbose=false
        │   ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday]
        │     SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST]
        │       [Stage 2] => NetworkCoalesceExec: output_partitions=6, input_tasks=2
        └──────────────────────────────────────────────────
          ┌───── Stage 2 ── Tasks: t0:[p0..p2] t1:[p0..p2]
          │ SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true]
          │   ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))]
          │     AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
          │       [Stage 1] => NetworkShuffleExec: output_partitions=3, input_tasks=3
          └──────────────────────────────────────────────────
            ┌───── Stage 1 ── Tasks: t0:[p0..p5] t1:[p0..p5] t2:[p0..p5]
            │ RepartitionExec: partitioning=Hash([RainToday@0], 6), input_partitions=1
            │   AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))]
            │     PartitionIsolatorExec: tasks=3 partitions=3
            │       DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet
            └──────────────────────────────────────────────────
        ",
        );

        let batches = pretty_format_batches(
            &execute_stream(physical, ctx.task_ctx())?
                .try_collect::<Vec<_>>()
                .await?,
        )?;

        assert_snapshot!(batches, @"
        +-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | plan_type         | plan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
        +-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        | Plan with Metrics | ProjectionExec: expr=[count(*)@0 as count(*), RainToday@1 as RainToday], metrics=[output_rows=2, elapsed_compute=1.46µs, output_bytes=48.0 B, output_batches=1, expr_0_eval_time=166ns, expr_1_eval_time=84ns]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
        |                   |   SortPreservingMergeExec: [count(Int64(1))@2 ASC NULLS LAST], metrics=[output_rows=2, elapsed_compute=196.12µs, output_bytes=64.0 B, output_batches=1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
        |                   |     SortExec: expr=[count(*)@0 ASC NULLS LAST], preserve_partitioning=[true], metrics=[output_rows=2, elapsed_compute=9.75µs, output_bytes=0.0 B, output_batches=0, spill_count=0, spilled_bytes=0.0 B, spilled_rows=0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
        |                   |       ProjectionExec: expr=[count(Int64(1))@1 as count(*), RainToday@0 as RainToday, count(Int64(1))@1 as count(Int64(1))], metrics=[output_rows=2, elapsed_compute=4.13µs, output_bytes=192.0 B, output_batches=2, expr_0_eval_time=334ns, expr_1_eval_time=210ns, expr_2_eval_time=251ns]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        |                   |         AggregateExec: mode=FinalPartitioned, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))], metrics=[output_rows=2, elapsed_compute=89.21µs, output_bytes=192.0 B, output_batches=2, spill_count=0, spilled_bytes=0.0 B, spilled_rows=0, peak_mem_used=1.90 K, aggregate_arguments_time=1.46µs, aggregation_time=48.88µs, emitting_time=4.04µs, time_calculating_group_ids=3.58µs]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
        |                   |           RepartitionExec: partitioning=Hash([RainToday@0], 3), input_partitions=3, metrics=[output_rows=6, elapsed_compute=91.25µs, output_bytes=384.0 KB, output_batches=2, spill_count=0, spilled_bytes=0.0 B, spilled_rows=0, fetch_time=4.74ms, repartition_time=278.96µs, send_time=21.17µs]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
        |                   |             AggregateExec: mode=Partial, gby=[RainToday@0 as RainToday], aggr=[count(Int64(1))], metrics=[output_rows=6, elapsed_compute=495.50µs, output_bytes=288.0 B, output_batches=3, spill_count=0, spilled_bytes=0.0 B, spilled_rows=0, skipped_aggregation_rows=0, peak_mem_used=7.18 K, aggregate_arguments_time=68.21µs, aggregation_time=41.83µs, emitting_time=12.42µs, time_calculating_group_ids=181.62µs, reduction_factor=1.6% (6/366)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
        |                   |               DataSourceExec: file_groups={3 groups: [[/testdata/weather/result-000000.parquet], [/testdata/weather/result-000001.parquet], [/testdata/weather/result-000002.parquet]]}, projection=[RainToday], file_type=parquet, metrics=[output_rows=366, elapsed_compute=3ns, output_bytes=8.2 KB, output_batches=3, files_ranges_pruned_statistics=3 total → 3 matched, row_groups_pruned_statistics=3 total → 3 matched, row_groups_pruned_bloom_filter=3 total → 3 matched, page_index_pages_pruned=0 total → 0 matched, page_index_rows_pruned=0 total → 0 matched, limit_pruned_row_groups=0 total → 0 matched, batches_split=0, bytes_scanned=3.12 K, file_open_errors=0, file_scan_errors=0, num_predicate_creation_errors=0, predicate_evaluation_errors=0, pushdown_rows_matched=0, pushdown_rows_pruned=0, predicate_cache_inner_records=0, predicate_cache_records=0, bloom_filter_eval_time=6ns, metadata_load_time=259.59µs, page_index_eval_time=6ns, row_pushdown_eval_time=6ns, statistics_eval_time=6ns, time_elapsed_opening=518.29µs, time_elapsed_processing=1.20ms, time_elapsed_scanning_total=3.91ms, time_elapsed_scanning_until_data=3.49ms, scan_efficiency_ratio=6.8% (3.12 K/45.82 K)] |
        |                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
        +-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        ");

        let batches_distributed = pretty_format_batches(
            &execute_stream(physical_distributed, ctx.task_ctx())?
                .try_collect::<Vec<_>>()
                .await?,
        )?;
        assert_snapshot!(batches_distributed, @r"
        +----------+-----------+
        | count(*) | RainToday |
        +----------+-----------+
        | 66       | Yes       |
        | 300      | No        |
        +----------+-----------+
        ");

        Ok(())
    }
}
