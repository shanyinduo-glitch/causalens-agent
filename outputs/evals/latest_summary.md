# CausaLens MVP Evaluation Summary

- Suite: CausaLens MVP Evaluation
- Version: 1.0
- Generated at: 2026-07-28T18:14:58
- Dataset: `data/student_performance.csv`
- Total: 10
- Passed: 10
- Failed: 0
- Pass rate: 100.0%

## Case Results

| Case | Category | Result | Task type | Tool order | Duration |
|---|---|---:|---|---|---:|
| mvp_01_data_profile | data_profile | PASS | data_profile | profile_dataset | 9.5974s |
| mvp_02_rag_passed | rag_question | PASS | rag_question | retrieve_docs | 27.8021s |
| mvp_03_rag_unit | rag_question | PASS | rag_question | retrieve_docs | 14.2366s |
| mvp_04_sql_group_mean | sql_analysis | PASS | sql_analysis | run_readonly_sql | 12.1663s |
| mvp_05_chart_histogram | chart_analysis | PASS | chart_analysis | create_chart | 14.5849s |
| mvp_06_chart_scatter | chart_analysis | PASS | chart_analysis | create_chart | 7.4381s |
| mvp_07_mixed_rag_sql | mixed_analysis | PASS | mixed_analysis | retrieve_docs → run_readonly_sql | 24.5741s |
| mvp_08_mixed_rag_chart | mixed_analysis | PASS | mixed_analysis | retrieve_docs → create_chart | 23.6292s |
| mvp_09_mixed_sql_chart | mixed_analysis | PASS | mixed_analysis | run_readonly_sql → create_chart | 13.001s |
| mvp_10_unsupported_delete | unsupported | PASS | unsupported | 无 | 0.0274s |
