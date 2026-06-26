from causalens.tools.data_profile import profile_dataset


TOOL_REGISTRY = {
    "profile_dataset": profile_dataset,
}


TOOLS = [
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