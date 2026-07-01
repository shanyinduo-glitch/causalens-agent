from causalens.tools.chart_tool import create_chart


bar_result = create_chart(
    dataset_path="data/student_performance.csv",
    chart_type="bar",
    x_column="major",
    y_column="score",
    aggregation="mean",
    title="Average Score by Major",
)

print("\n=== Bar Chart Result ===")
print(bar_result.model_dump())


scatter_result = create_chart(
    dataset_path="data/student_performance.csv",
    chart_type="scatter",
    x_column="study_hours",
    y_column="score",
    title="Study Hours vs Score",
)

print("\n=== Scatter Chart Result ===")
print(scatter_result.model_dump())


histogram_result = create_chart(
    dataset_path="data/student_performance.csv",
    chart_type="histogram",
    x_column="score",
    title="Score Distribution",
)

print("\n=== Histogram Result ===")
print(histogram_result.model_dump())


line_result = create_chart(
    dataset_path="data/student_performance.csv",
    chart_type="line",
    x_column="exam_date",
    y_column="score",
    aggregation="mean",
    title="Average Score by Exam Date",
)

print("\n=== Line Chart Result ===")
print(line_result.model_dump())