from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field


IdentificationStatus = Literal[
    "not_assessed",
    "identified",
    "blocked",
]


class IdentificationAssessment(BaseModel):
    """基于用户提供 DAG 的后门识别检查结果。"""

    status: IdentificationStatus
    identified: bool | None
    dag_provided: bool
    dag_valid: bool | None
    treatment: str
    outcome: str
    adjustment_set: list[str]
    dag_edges: list[tuple[str, str]] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)


def _build_graph(
    edges: list[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in edges:
        children[source].add(target)
        parents[target].add(source)
        nodes.update((source, target))
    return children, parents, nodes


def _find_cycle(
    children: dict[str, set[str]],
    nodes: set[str],
) -> list[str] | None:
    state: dict[str, int] = {node: 0 for node in nodes}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for child in children.get(node, set()):
            if state[child] == 0:
                cycle = visit(child)
                if cycle:
                    return cycle
            elif state[child] == 1:
                start = stack_index[child]
                return [*stack[start:], child]
        stack.pop()
        stack_index.pop(node, None)
        state[node] = 2
        return None

    for node in sorted(nodes):
        if state[node] == 0:
            cycle = visit(node)
            if cycle:
                return cycle
    return None


def _descendants(
    start: str,
    children: dict[str, set[str]],
) -> set[str]:
    found: set[str] = set()
    queue = deque(children.get(start, set()))
    while queue:
        node = queue.popleft()
        if node in found:
            continue
        found.add(node)
        queue.extend(children.get(node, set()))
    return found


def _ancestors(
    starts: set[str],
    parents: dict[str, set[str]],
) -> set[str]:
    found = set(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        for parent in parents.get(node, set()):
            if parent not in found:
                found.add(parent)
                queue.append(parent)
    return found


def _has_directed_path(
    source: str,
    target: str,
    children: dict[str, set[str]],
) -> bool:
    return target in _descendants(source, children)


def _has_open_backdoor_path(
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    edges: list[tuple[str, str]],
) -> bool:
    """
    在删除 treatment 出边后的图上，用祖先图道德化判定 d-separation。

    若 treatment 与 outcome 在移除 adjustment_set 后仍连通，则至少存在
    一条未被阻断的后门路径。
    """

    backdoor_edges = [
        (source, target)
        for source, target in edges
        if source != treatment
    ]
    children, parents, _ = _build_graph(backdoor_edges)
    relevant = _ancestors(
        {treatment, outcome, *adjustment_set},
        parents,
    )

    undirected: dict[str, set[str]] = defaultdict(set)
    for source, target in backdoor_edges:
        if source in relevant and target in relevant:
            undirected[source].add(target)
            undirected[target].add(source)

    # 道德化：共同父节点之间增加无向边。
    for child in relevant:
        child_parents = sorted(parents.get(child, set()) & relevant)
        for index, left in enumerate(child_parents):
            for right in child_parents[index + 1 :]:
                undirected[left].add(right)
                undirected[right].add(left)

    blocked = set(adjustment_set)
    if treatment in blocked or outcome in blocked:
        return True

    visited = set(blocked)
    queue = deque([treatment])
    visited.add(treatment)
    while queue:
        node = queue.popleft()
        if node == outcome:
            return True
        for neighbor in undirected.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _minimal_adjustment_sets(
    *,
    treatment: str,
    outcome: str,
    edges: list[tuple[str, str]],
    children: dict[str, set[str]],
    nodes: set[str],
    maximum_candidates: int = 12,
) -> tuple[list[list[str]], str | None]:
    """枚举小型用户 DAG 的最小后门调整集，规模过大时安全跳过。"""

    forbidden = {treatment, outcome, *_descendants(treatment, children)}
    candidates = sorted(nodes - forbidden)
    if len(candidates) > maximum_candidates:
        return [], (
            "DAG has too many eligible adjustment candidates for exhaustive "
            f"enumeration ({len(candidates)} > {maximum_candidates})."
        )
    minimal: list[list[str]] = []
    for size in range(len(candidates) + 1):
        for subset_tuple in combinations(candidates, size):
            subset = set(subset_tuple)
            if any(set(existing).issubset(subset) for existing in minimal):
                continue
            if not _has_open_backdoor_path(
                treatment,
                outcome,
                subset,
                edges,
            ):
                minimal.append(list(subset_tuple))
    return minimal, None


def assess_causal_identification(
    *,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    dag_edges: list[tuple[str, str]] | None,
) -> IdentificationAssessment:
    """验证 DAG，并检查给定调整集是否满足后门准则。"""

    normalized_edges = list(dag_edges or [])
    if not normalized_edges:
        return IdentificationAssessment(
            status="not_assessed",
            identified=None,
            dag_provided=False,
            dag_valid=None,
            treatment=treatment,
            outcome=outcome,
            adjustment_set=list(adjustment_set),
            warnings=[
                "未提供因果 DAG，因此无法验证调整集是否阻断全部后门路径。"
            ],
        )

    children, _, nodes = _build_graph(normalized_edges)
    blocking_issues: list[str] = []
    warnings: list[str] = []

    cycle = _find_cycle(children, nodes)
    if cycle:
        blocking_issues.append(
            "提供的因果图不是 DAG，检测到有向环："
            + " → ".join(cycle)
        )

    missing_core_nodes = sorted({treatment, outcome} - nodes)
    if missing_core_nodes:
        blocking_issues.append(
            "DAG 缺少核心变量：" + ", ".join(missing_core_nodes)
        )

    missing_adjustment_nodes = sorted(set(adjustment_set) - nodes)
    if missing_adjustment_nodes:
        blocking_issues.append(
            "调整集中的变量未出现在 DAG："
            + ", ".join(missing_adjustment_nodes)
        )

    if blocking_issues:
        return IdentificationAssessment(
            status="blocked",
            identified=False,
            dag_provided=True,
            dag_valid=False,
            treatment=treatment,
            outcome=outcome,
            adjustment_set=list(adjustment_set),
            dag_edges=normalized_edges,
            blocking_issues=blocking_issues,
            warnings=warnings,
            diagnostics={"cycle": cycle},
        )

    descendants = _descendants(treatment, children)
    post_treatment_adjustments = sorted(set(adjustment_set) & descendants)
    if post_treatment_adjustments:
        blocking_issues.append(
            "调整集包含 treatment 的后代（可能是中介或处理后变量）："
            + ", ".join(post_treatment_adjustments)
        )

    open_backdoor = _has_open_backdoor_path(
        treatment,
        outcome,
        set(adjustment_set),
        normalized_edges,
    )
    if open_backdoor:
        blocking_issues.append(
            "给定调整集未能阻断 treatment 与 outcome 之间的全部后门路径。"
        )

    directed_effect_path = _has_directed_path(
        treatment,
        outcome,
        children,
    )
    if not directed_effect_path:
        warnings.append(
            "DAG 中不存在 treatment 指向 outcome 的有向路径；"
            "请确认研究问题或图结构是否正确。"
        )

    minimal_sets, enumeration_warning = _minimal_adjustment_sets(
        treatment=treatment,
        outcome=outcome,
        edges=normalized_edges,
        children=children,
        nodes=nodes,
    )
    if enumeration_warning:
        warnings.append(enumeration_warning)
    adjustment_is_minimal = any(
        set(candidate) == set(adjustment_set) for candidate in minimal_sets
    )
    if not open_backdoor and minimal_sets and not adjustment_is_minimal:
        warnings.append(
            "The supplied adjustment set identifies the effect but is not minimal; "
            "inspect the suggested minimal sets to avoid unnecessary adjustment."
        )

    identified = not blocking_issues
    return IdentificationAssessment(
        status="identified" if identified else "blocked",
        identified=identified,
        dag_provided=True,
        dag_valid=True,
        treatment=treatment,
        outcome=outcome,
        adjustment_set=list(adjustment_set),
        dag_edges=normalized_edges,
        blocking_issues=blocking_issues,
        warnings=warnings,
        diagnostics={
            "treatment_descendants": sorted(descendants),
            "post_treatment_adjustments": post_treatment_adjustments,
            "open_backdoor_path": open_backdoor,
            "directed_effect_path": directed_effect_path,
            "minimal_adjustment_sets": minimal_sets,
            "adjustment_set_minimal": adjustment_is_minimal,
        },
    )
