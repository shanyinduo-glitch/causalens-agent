from causalens.agent.tool_registry import TOOL_REGISTRY, TOOLS
from causalens.tools.data_profile import profile_dataset


def test_profile_dataset_is_registered():
    """工具名称应能找到真实的 Python 函数。"""

    assert "profile_dataset" in TOOL_REGISTRY
    assert TOOL_REGISTRY["profile_dataset"] is profile_dataset
    assert callable(TOOL_REGISTRY["profile_dataset"])


def test_profile_dataset_tool_description_exists():
    """给模型看的 TOOLS 说明中应包含 profile_dataset。"""

    function_tools = [
        tool["function"]
        for tool in TOOLS
        if tool.get("type") == "function"
    ]

    tool_names = {tool["name"] for tool in function_tools}

    assert "profile_dataset" in tool_names


def test_profile_dataset_requires_dataset_path():
    """profile_dataset 的工具说明必须要求 dataset_path 参数。"""

    profile_tool = next(
        tool["function"]
        for tool in TOOLS
        if tool["function"]["name"] == "profile_dataset"
    )

    parameters = profile_tool["parameters"]

    assert parameters["type"] == "object"
    assert parameters["required"] == ["dataset_path"]
    assert parameters["properties"]["dataset_path"]["type"] == "string"