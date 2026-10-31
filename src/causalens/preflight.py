from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from causalens.config import get_free_model_candidates, get_max_llm_calls_per_run


REQUIRED_MODULES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "econml": "econml",
    "pydantic": "pydantic",
}


def _api_key_is_configured(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    return not any(
        marker in normalized
        for marker in ("your_", "replace_me", "example", "placeholder")
    )


def run_preflight(workspace_path: str | Path | None = None) -> dict[str, Any]:
    """Perform local checks only. Never sends a request or exposes the API key."""

    checks: list[dict[str, Any]] = []
    expected_python = sys.version_info[:2] == (3, 12)
    checks.append(
        {
            "name": "python_version",
            "passed": expected_python,
            "blocking_for": ["offline", "llm"] if not expected_python else [],
            "detail": f"Python {sys.version_info.major}.{sys.version_info.minor}",
        }
    )

    missing_modules: list[str] = []
    versions: dict[str, str] = {}
    for module_name, distribution_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(distribution_name)
            continue
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution_name] = "installed (version unavailable)"
    checks.append(
        {
            "name": "causal_dependencies",
            "passed": not missing_modules,
            "blocking_for": ["offline", "llm"] if missing_modules else [],
            "detail": {
                "installed_versions": versions,
                "missing": missing_modules,
            },
        }
    )

    workspace = Path(workspace_path or Path.cwd()).resolve()
    writable = workspace.is_dir() and os.access(workspace, os.W_OK)
    checks.append(
        {
            "name": "workspace_writable",
            "passed": writable,
            "blocking_for": ["offline", "llm"] if not writable else [],
            "detail": str(workspace),
        }
    )

    raw_key = os.getenv("DASHSCOPE_API_KEY", "")
    key_configured = _api_key_is_configured(raw_key)
    checks.append(
        {
            "name": "dashscope_api_key",
            "passed": key_configured,
            "blocking_for": [] if key_configured else ["llm"],
            "detail": "configured" if key_configured else "missing or placeholder",
        }
    )

    config_errors: list[str] = []
    candidates: tuple[str, ...] = ()
    call_budget: int | None = None
    try:
        candidates = get_free_model_candidates()
        call_budget = get_max_llm_calls_per_run()
    except RuntimeError as exc:
        config_errors.append(str(exc))
    checks.append(
        {
            "name": "free_model_configuration",
            "passed": not config_errors,
            "blocking_for": [] if not config_errors else ["llm"],
            "detail": {
                "candidates": list(candidates),
                "candidate_count": len(candidates),
                "max_llm_calls_per_run": call_budget,
                "errors": config_errors,
            },
        }
    )

    offline_ready = all(
        check["passed"]
        for check in checks
        if "offline" in check["blocking_for"] or not check["blocking_for"]
    )
    llm_ready = all(
        check["passed"]
        for check in checks
        if "llm" in check["blocking_for"] or not check["blocking_for"]
    )
    return {
        "offline_contract_runner_ready": offline_ready,
        "natural_language_agent_ready": llm_ready,
        "checks": checks,
        "warnings": [
            "本检查不会联网，无法核验模型是否仍有免费额度。请继续在百炼控制台"
            "启用“免费额度用完即停”，并只把确认免费的模型写入候选列表。"
        ],
        "network_calls": 0,
        "secrets_exposed": False,
    }
