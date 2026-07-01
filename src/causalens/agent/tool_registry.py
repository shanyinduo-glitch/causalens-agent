from causalens.tools.data_profile import profile_dataset
from causalens.tools.sql_tool import run_readonly_sql
from causalens.tools.chart_tool import create_chart


TOOL_REGISTRY = {
    "profile_dataset": profile_dataset,
    "run_readonly_sql": run_readonly_sql,
    "create_chart": create_chart,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_readonly_sql",
            "description": (
                "Run one read-only SQL query against the CSV dataset. "
                "The CSV is always available in SQL as a table named 'dataset'. "
                "Use this tool for filtering, grouping, counting, aggregation, "
                "sorting, ranking, Top-N queries, and row-level lookup. "
                "Only use read-only SQL starting with SELECT, WITH, or EXPLAIN. "
                "Do not use INSERT, UPDATE, DELETE, DROP, CREATE, or multiple SQL statements. "
                "For questions about highest, lowest, Top-N, or individual rows, "
                "use ORDER BY and LIMIT in the SQL query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Path to the CSV dataset. Use the dataset path "
                            "provided in the user context."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "One read-only SQL query. The CSV must always be referenced "
                            "as the table name 'dataset'."
                        ),
                    },
                },
                "required": ["dataset_path", "query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Create a PNG chart from the CSV dataset and return the chart path "
                "plus the real data summary used to create the chart. "
                "Use bar charts for category comparisons, scatter charts for the "
                "relationship between two numeric columns, histograms for the "
                "distribution of one numeric column, and line charts for trends "
                "over a date column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Path to the CSV dataset. Use the dataset path "
                            "provided in the user context."
                        ),
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "bar",
                            "scatter",
                            "histogram",
                            "line",
                        ],
                        "description": (
                            "bar for category comparison; "
                            "scatter for two numeric columns; "
                            "histogram for one numeric column distribution; "
                            "line for date-based trends."
                        ),
                    },
                    "x_column": {
                        "type": "string",
                        "description": (
                            "For bar: category column. "
                            "For scatter: numeric x-axis column. "
                            "For histogram: numeric column. "
                            "For line: date column."
                        ),
                    },
                    "y_column": {
                        "type": "string",
                        "description": (
                            "Required for scatter. "
                            "Required for bar and line when aggregation is mean or sum. "
                            "Do not provide it for histogram or count charts."
                        ),
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "mean",
                            "sum",
                            "count",
                        ],
                        "description": (
                            "Required for bar and line charts. "
                            "For count, y_column is not needed. "
                            "For mean or sum, y_column must be numeric."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Optional English chart title. Prefer English titles "
                            "to avoid local font compatibility issues."
                        ),
                    },
                },
                "required": [
                    "dataset_path",
                    "chart_type",
                    "x_column",
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": (
                "读取 CSV 数据集并返回行数、列数、字段名、数据类型、"
                "缺失值统计和数值列描述性统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "CSV 文件路径，例如 data/sample.csv",
                    }
                },
                "required": ["dataset_path"],
            },
        },
    }
]