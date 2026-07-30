# CausaLens MVP Evaluation Summary

- Suite: CausaLens MVP Evaluation
- Version: 1.0
- Generated at: 2026-07-30T15:49:35
- Dataset: `data/student_performance.csv`
- Total: 10
- Passed: 10
- Failed: 0
- Pass rate: 100.0%

## Case Results

| Case | Category | Result | Task type | Tool order | Duration |
|---|---|---:|---|---|---:|
| mvp_01_data_profile | data_profile | PASS | data_profile | profile_dataset | 6.0839s |
| mvp_02_rag_passed | rag_question | PASS | rag_question | retrieve_docs | 31.1095s |
| mvp_03_rag_unit | rag_question | PASS | rag_question | retrieve_docs | 15.8538s |
| mvp_04_sql_group_mean | sql_analysis | PASS | sql_analysis | run_readonly_sql | 14.358s |
| mvp_05_chart_histogram | chart_analysis | PASS | chart_analysis | create_chart | 11.4193s |
| mvp_06_chart_scatter | chart_analysis | PASS | chart_analysis | create_chart | 11.7059s |
| mvp_07_mixed_rag_sql | mixed_analysis | PASS | mixed_analysis | retrieve_docs → run_readonly_sql | 26.6462s |
| mvp_08_mixed_rag_chart | mixed_analysis | PASS | mixed_analysis | retrieve_docs → create_chart | 28.4784s |
| mvp_09_mixed_sql_chart | mixed_analysis | PASS | mixed_analysis | run_readonly_sql → create_chart | 21.4347s |
| mvp_10_unsupported_delete | unsupported | PASS | unsupported | 无 | 0.0257s |
