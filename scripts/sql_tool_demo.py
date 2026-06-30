from causalens.tools.sql_tool import run_readonly_sql


queries = [
    "SELECT COUNT(*) AS total_rows FROM dataset",
    """
    SELECT major, COUNT(*) AS n
    FROM dataset
    GROUP BY major
    ORDER BY n DESC
    """,
    """
    SELECT major, AVG(score) AS avg_score
    FROM dataset
    GROUP BY major
    ORDER BY avg_score DESC
    """,
    """
    SELECT *
    FROM dataset
    ORDER BY score DESC
    LIMIT 5
    """,
    """
    SELECT city, AVG(study_hours) AS avg_hours
    FROM dataset
    GROUP BY city
    ORDER BY avg_hours DESC
    """,
]

for index, query in enumerate(queries, start=1):
    print(f"\n{'=' * 20} Query {index} {'=' * 20}")
    print(query)

    result = run_readonly_sql(
        dataset_path="data/student_performance.csv",
        query=query,
    )

    print(result.model_dump())