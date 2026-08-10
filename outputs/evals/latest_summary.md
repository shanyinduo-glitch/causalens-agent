# CausaLens MVP Evaluation Summary

- Suite: CausaLens MVP Evaluation
- Version: 1.0
- Generated at: 2026-08-11T00:06:18
- Dataset: `data/student_performance.csv`
- Total: 42
- Passed: 42
- Failed: 0
- Pass rate: 100.0%

| Case | Category | Result | Task type | Tool order | Duration |
|---|---|---:|---|---|---:|

## Category Results

| Category | Passed | Total | Pass rate |
|---|---:|---:|---:|
| chart_analysis | 6 | 6 | 100.0% |
| data_profile | 4 | 4 | 100.0% |
| mixed_analysis | 4 | 4 | 100.0% |
| rag_question | 6 | 6 | 100.0% |
| sql_analysis | 5 | 5 | 100.0% |
| statistical_analysis | 14 | 14 | 100.0% |
| unsupported | 3 | 3 | 100.0% |

## Case Results

| mvp_01_data_profile | data_profile | PASS | data_profile | profile_dataset | 9.6445s |
| mvp_02_rag_passed | rag_question | PASS | rag_question | retrieve_docs | 29.1793s |
| mvp_03_rag_unit | rag_question | PASS | rag_question | retrieve_docs | 17.0761s |
| mvp_04_sql_group_mean | sql_analysis | PASS | sql_analysis | run_readonly_sql | 15.7032s |
| mvp_05_chart_histogram | chart_analysis | PASS | chart_analysis | create_chart | 15.3169s |
| mvp_06_chart_scatter | chart_analysis | PASS | chart_analysis | create_chart | 11.5599s |
| mvp_07_mixed_rag_sql | mixed_analysis | PASS | mixed_analysis | retrieve_docs → run_readonly_sql | 28.4167s |
| mvp_08_mixed_rag_chart | mixed_analysis | PASS | mixed_analysis | retrieve_docs → create_chart | 33.8272s |
| mvp_09_mixed_sql_chart | mixed_analysis | PASS | mixed_analysis | run_readonly_sql → create_chart | 24.7272s |
| mvp_10_unsupported_delete | unsupported | PASS | unsupported | 无 | 0.0322s |
| mvp_11_stat_correlation | statistical_analysis | PASS | statistical_analysis | run_correlation | 13.9614s |
| mvp_12_stat_group_comparison | statistical_analysis | PASS | statistical_analysis | run_group_comparison | 20.9678s |
| mvp_13_stat_ols | statistical_analysis | PASS | statistical_analysis | run_ols | 20.7756s |
| mvp_14_stat_correlation_paraphrase | statistical_analysis | PASS | statistical_analysis | run_correlation | 17.5433s |
| mvp_15_stat_group_comparison_paraphrase | statistical_analysis | PASS | statistical_analysis | run_group_comparison | 19.9007s |
| mvp_16_stat_ols_paraphrase | statistical_analysis | PASS | statistical_analysis | run_ols | 14.6163s |
| mvp_17_data_profile_paraphrase | data_profile | PASS | data_profile | profile_dataset | 18.5378s |
| mvp_18_rag_dataset_source | rag_question | PASS | rag_question | retrieve_docs | 18.9619s |
| mvp_19_rag_passed_paraphrase | rag_question | PASS | rag_question | retrieve_docs | 21.3923s |
| mvp_20_sql_city_mean | sql_analysis | PASS | sql_analysis | run_readonly_sql | 12.0653s |
| mvp_21_sql_high_score_count | sql_analysis | PASS | sql_analysis | run_readonly_sql | 11.7645s |
| mvp_22_chart_major_bar | chart_analysis | PASS | chart_analysis | create_chart | 19.4084s |
| mvp_23_chart_attendance_histogram | chart_analysis | PASS | chart_analysis | create_chart | 10.5561s |
| mvp_24_stat_correlation_paraphrase_2 | statistical_analysis | PASS | statistical_analysis | run_correlation | 14.0878s |
| mvp_25_stat_group_comparison_paraphrase_2 | statistical_analysis | PASS | statistical_analysis | run_group_comparison | 18.511s |
| mvp_26_unsupported_modify | unsupported | PASS | unsupported | 无 | 0.5825s |
| mvp_27_data_profile_shape | data_profile | PASS | data_profile | profile_dataset | 8.417s |
| mvp_28_data_profile_missing | data_profile | PASS | data_profile | profile_dataset | 10.353s |
| mvp_29_rag_exam_date | rag_question | PASS | rag_question | retrieve_docs | 18.5379s |
| mvp_30_rag_simulated_data | rag_question | PASS | rag_question | retrieve_docs | 21.6989s |
| mvp_31_sql_city_count | sql_analysis | PASS | sql_analysis | run_readonly_sql | 9.7696s |
| mvp_32_sql_top_scores | sql_analysis | PASS | sql_analysis | run_readonly_sql | 8.2002s |
| mvp_33_chart_attendance_scatter | chart_analysis | PASS | chart_analysis | create_chart | 16.1832s |
| mvp_34_chart_major_count | chart_analysis | PASS | chart_analysis | create_chart | 14.9108s |
| mvp_35_stat_spearman | statistical_analysis | PASS | statistical_analysis | run_correlation | 21.3143s |
| mvp_36_stat_pearson | statistical_analysis | PASS | statistical_analysis | run_correlation | 13.818s |
| mvp_37_stat_group_difference | statistical_analysis | PASS | statistical_analysis | run_group_comparison | 17.2291s |
| mvp_38_stat_anova | statistical_analysis | PASS | statistical_analysis | run_group_comparison | 17.5053s |
| mvp_39_stat_ols_explicit | statistical_analysis | PASS | statistical_analysis | run_ols | 28.8343s |
| mvp_40_stat_ols_paraphrase_3 | statistical_analysis | PASS | statistical_analysis | run_ols | 19.3272s |
| mvp_41_mixed_rag_sql_count | mixed_analysis | PASS | mixed_analysis | retrieve_docs → run_readonly_sql | 27.9308s |
| mvp_42_unsupported_overwrite | unsupported | PASS | unsupported | 无 | 0.0382s |
