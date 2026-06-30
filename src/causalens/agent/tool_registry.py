from causalens.tools.data_profile import profile_dataset
from causalens.tools.sql_tool import run_readonly_sql


TOOL_REGISTRY = {
    "profile_dataset": profile_dataset,
    "run_readonly_sql": run_readonly_sql,
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