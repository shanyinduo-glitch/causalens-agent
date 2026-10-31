from causalens.agent.tool_registry import (
    TOOL_REGISTRY,
    TOOLS,
)


def test_week6_statistical_tools_are_registered():
    expected_tools = {
        "run_correlation",
        "run_group_comparison",
        "run_ols",
    }

    assert expected_tools.issubset(
        TOOL_REGISTRY.keys()
    )

    for tool_name in expected_tools:
        assert callable(
            TOOL_REGISTRY[tool_name]
        )


def test_week6_statistical_tools_are_exposed_to_model():
    exposed_tool_names = {
        tool["function"]["name"]
        for tool in TOOLS
    }

    expected_tools = {
        "run_correlation",
        "run_group_comparison",
        "run_ols",
    }

    assert expected_tools.issubset(
        exposed_tool_names
    )


def test_causal_estimators_are_not_exposed_to_model():
    exposed_tool_names = {
        tool["function"]["name"]
        for tool in TOOLS
    }

    assert "run_dml" not in exposed_tool_names
    assert "run_aipw" not in exposed_tool_names
    assert "run_causal_forest" not in exposed_tool_names
    assert "run_difference_in_differences" not in exposed_tool_names
    assert "run_causal_estimators" not in exposed_tool_names
    assert (
        "check_causal_readiness"
        not in exposed_tool_names
    )
