#!/usr/bin/env python3
"""Exact enumeration and MH transition checks on a four-cycle.

Anchors: u, v
Optional bridge vertices: x, y
Graph edges: ux, xv, vy, yu
No diagonals.

The feasible space includes non-anchor leaves, as in the current
mathematical definition. This script enumerates exact probabilities
and transitions; it does not run Monte Carlo sampling.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path


ANCHORS = {"u", "v"}
VERTICES = {"u", "v", "x", "y"}


def canonical_edge(u, v):
    return tuple(sorted((u, v)))


WEIGHTS = {
    canonical_edge("u", "x"): 1.0,
    canonical_edge("x", "v"): 1.2,
    canonical_edge("v", "y"): 1.4,
    canonical_edge("y", "u"): 1.1,
}

GRAPH_EDGES = tuple(sorted(WEIGHTS))

# First hold with probability 0.5. Otherwise select a move type.
HOLD_PROBABILITY = 0.5
MOVE_PROBABILITIES = {
    "exchange": 0.5,
    "birth": 0.25,
    "death": 0.25,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def vertices_of(tree):
    return {v for edge in tree for v in edge}


def neighbors_of(tree):
    neighbors = {v: set() for v in vertices_of(tree)}
    for u, v in tree:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def reachable(start, edges):
    neighbors = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in neighbors.get(u, ()):
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return visited


def is_feasible(tree):
    nodes = vertices_of(tree)
    if not ANCHORS.issubset(nodes):
        return False
    if not nodes.issubset(VERTICES):
        return False
    if not set(tree).issubset(WEIGHTS):
        return False
    if len(tree) != len(nodes) - 1:
        return False
    return reachable(next(iter(nodes)), tree) == nodes


def enumerate_trees():
    trees = []
    for size in range(1, len(GRAPH_EDGES) + 1):
        for edges in combinations(GRAPH_EDGES, size):
            tree = frozenset(edges)
            if is_feasible(tree):
                trees.append(tree)
    return sorted(trees, key=lambda t: (len(t), tuple(sorted(t))))


def cost(tree):
    return sum(WEIGHTS[edge] for edge in sorted(tree))


def edge_string(tree):
    return ";".join(f"{u}-{v}" for u, v in sorted(tree))


def add_mass(distribution, tree, mass):
    distribution[tree] = distribution.get(tree, 0.0) + mass


def exchange_distribution(tree):
    """Uniform removed edge, then uniform alternative crossing edge."""
    distribution = {}
    nodes = vertices_of(tree)

    for old_edge in sorted(tree):
        remaining = tree - {old_edge}
        side = reachable(old_edge[0], remaining)

        choices = [
            edge for edge in GRAPH_EDGES
            if set(edge).issubset(nodes)
            and edge not in tree
            and ((edge[0] in side) != (edge[1] in side))
        ]

        if not choices:
            add_mass(distribution, tree, 1.0 / len(tree))
            continue

        for new_edge in choices:
            proposed = frozenset(remaining | {new_edge})
            require(is_feasible(proposed), "Invalid exchange proposal.")
            add_mass(
                distribution, proposed,
                1.0 / len(tree) / len(choices),
            )

    return distribution


def birth_distribution(tree):
    """Uniform eligible (edge, unused bridge) insertion."""
    nodes = vertices_of(tree)
    choices = []

    for u, v in sorted(tree):
        for x in sorted(VERTICES - nodes - ANCHORS):
            left = canonical_edge(u, x)
            right = canonical_edge(x, v)
            if left in WEIGHTS and right in WEIGHTS:
                proposed = frozenset(
                    (tree - {canonical_edge(u, v)}) | {left, right}
                )
                require(is_feasible(proposed), "Invalid birth proposal.")
                choices.append(proposed)

    if not choices:
        return {tree: 1.0}

    distribution = {}
    for proposed in choices:
        add_mass(distribution, proposed, 1.0 / len(choices))
    return distribution


def death_distribution(tree):
    """Uniform eligible degree-two bridge removal."""
    neighbors = neighbors_of(tree)
    choices = []

    for x in sorted(vertices_of(tree) - ANCHORS):
        if len(neighbors[x]) != 2:
            continue

        u, v = sorted(neighbors[x])
        shortcut = canonical_edge(u, v)
        if shortcut not in WEIGHTS:
            continue

        proposed = frozenset(
            (tree - {canonical_edge(u, x), canonical_edge(x, v)})
            | {shortcut}
        )
        require(is_feasible(proposed), "Invalid death proposal.")
        choices.append(proposed)

    if not choices:
        return {tree: 1.0}

    distribution = {}
    for proposed in choices:
        add_mass(distribution, proposed, 1.0 / len(choices))
    return distribution


def write_csv(path, rows):
    require(bool(rows), f"No rows to write: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    args = parser.parse_args()

    lam = args.lambda_value
    require(math.isfinite(lam) and lam > 0, "Lambda must be positive.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir()
    print(f"[OUTPUT] {out}", flush=True)

    trees = enumerate_trees()
    n = len(trees)
    index = {tree: i for i, tree in enumerate(trees)}
    print(f"[ENUMERATION] {n} feasible trees", flush=True)

    costs = [cost(tree) for tree in trees]

    # Shift energies for stable normalization; probabilities are unchanged.
    minimum_cost = min(costs)
    relative_weights = [
        math.exp(-lam * (value - minimum_cost)) for value in costs
    ]
    normalizer = sum(relative_weights)
    probabilities = [value / normalizer for value in relative_weights]
    log_z = -lam * minimum_cost + math.log(normalizer)

    proposal = [[0.0] * n for _ in range(n)]
    eligibility_rows = []

    for i, tree in enumerate(trees):
        distributions = {
            "exchange": exchange_distribution(tree),
            "birth": birth_distribution(tree),
            "death": death_distribution(tree),
        }

        for move, distribution in distributions.items():
            require(
                abs(sum(distribution.values()) - 1.0) < 1e-12,
                f"Unnormalized {move} proposal.",
            )
            eligibility_rows.append({
                "state": f"T{i + 1:02d}",
                "move": move,
                "nonself_proposal_available": any(
                    proposed != tree for proposed in distribution
                ),
            })

            for proposed, mass in distribution.items():
                require(proposed in index, "Proposal outside state space.")
                j = index[proposed]
                proposal[i][j] += MOVE_PROBABILITIES[move] * mass

    # Exact MH transition matrix for the complete proposal mixture.
    # Unavailable moves and rejected proposals remain at the current state.
    kernel = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j or proposal[i][j] == 0:
                continue

            reverse = proposal[j][i]
            if reverse == 0:
                acceptance = 0.0
            else:
                log_ratio = (
                    -lam * (costs[j] - costs[i])
                    + math.log(reverse)
                    - math.log(proposal[i][j])
                )
                acceptance = math.exp(min(0.0, log_ratio))

            kernel[i][j] = (
                (1.0 - HOLD_PROBABILITY)
                * proposal[i][j]
                * acceptance
            )
        kernel[i][i] = 1.0 - sum(kernel[i])

    # Every positive non-self MH transition must have a reverse transition.
    for i in range(n):
        for j in range(n):
            require(
                (kernel[i][j] > 0) == (kernel[j][i] > 0),
                "Asymmetric transition support.",
            )

    # Connected components of this reversible transition support are
    # the communicating classes.
    classes = []
    unseen = set(range(n))
    while unseen:
        start = min(unseen)
        component = {start}
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j != i and kernel[i][j] > 0 and j not in component:
                    component.add(j)
                    stack.append(j)
        unseen -= component
        classes.append(sorted(component))

    class_of = {
        state: number
        for number, component in enumerate(classes, 1)
        for state in component
    }
    class_mass = {
        number: sum(probabilities[i] for i in component)
        for number, component in enumerate(classes, 1)
    }

    # Exact structural and numerical checks.
    require(n == 6, f"Expected 6 feasible trees; found {n}.")
    require(
        sorted(map(len, classes)) == [1, 1, 4],
        f"Unexpected communicating classes: {classes}",
    )
    require(all(is_feasible(tree) for tree in trees), "Invalid state.")

    max_row_residual = max(
        abs(sum(row) - 1.0) for row in kernel
    )
    max_stationarity_residual = max(
        abs(
            sum(probabilities[i] * kernel[i][j] for i in range(n))
            - probabilities[j]
        )
        for j in range(n)
    )

    flow_rows = []
    max_balance_residual = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if kernel[i][j] == 0 and kernel[j][i] == 0:
                continue
            forward = probabilities[i] * kernel[i][j]
            reverse = probabilities[j] * kernel[j][i]
            residual = abs(forward - reverse)
            tolerance = 1e-12 + 1e-10 * max(abs(forward), abs(reverse))
            require(residual <= tolerance, "Detailed balance failed.")
            max_balance_residual = max(max_balance_residual, residual)
            flow_rows.append({
                "from": f"T{i + 1:02d}",
                "to": f"T{j + 1:02d}",
                "forward_flow": forward,
                "reverse_flow": reverse,
                "absolute_residual": residual,
                "tolerance": tolerance,
            })

    require(max_row_residual <= 1e-12, "Kernel rows do not sum to one.")
    require(
        max_stationarity_residual <= 1e-12,
        "Exact stationarity check failed.",
    )
    require(
        all(kernel[i][i] >= HOLD_PROBABILITY - 1e-12 for i in range(n)),
        "Holding probability check failed.",
    )

    state_rows = []
    for i, tree in enumerate(trees):
        number = class_of[i]
        state_rows.append({
            "state": f"T{i + 1:02d}",
            "vertices": ";".join(sorted(vertices_of(tree))),
            "edges": edge_string(tree),
            "n_bridges": len(vertices_of(tree) - ANCHORS),
            "cost": costs[i],
            "lambda": lam,
            "full_target_probability": probabilities[i],
            "communicating_class": number,
            "class_target_mass": class_mass[number],
            "conditional_target_probability": (
                probabilities[i] / class_mass[number]
            ),
            "inaccessible_target_mass": 1.0 - class_mass[number],
        })

    transition_rows = [
        {
            "from": f"T{i + 1:02d}",
            "to": f"T{j + 1:02d}",
            "proposal_probability": proposal[i][j],
            "transition_probability": kernel[i][j],
        }
        for i in range(n)
        for j in range(n)
    ]

    write_csv(out / "01_graph_edges.csv", [
        {"u": u, "v": v, "weight": WEIGHTS[(u, v)]}
        for u, v in GRAPH_EDGES
    ])
    write_csv(out / "02_exact_tree_probabilities.csv", state_rows)
    write_csv(out / "03_transition_matrix_long.csv", transition_rows)
    write_csv(out / "04_detailed_balance_flows.csv", flow_rows)
    write_csv(out / "05_move_availability.csv", eligibility_rows)

    qc = {
        "graph": "four-cycle without diagonals",
        "anchors": sorted(ANCHORS),
        "optional_bridges": sorted(VERTICES - ANCHORS),
        "allows_nonanchor_leaves": True,
        "lambda": lam,
        "log_partition_function": log_z,
        "n_feasible_trees": n,
        "n_communicating_classes": len(classes),
        "class_sizes": [len(component) for component in classes],
        "hold_probability": HOLD_PROBABILITY,
        "conditional_move_probabilities": MOVE_PROBABILITIES,
        "max_kernel_row_sum_residual": max_row_residual,
        "max_stationarity_residual": max_stationarity_residual,
        "max_detailed_balance_residual": max_balance_residual,
        "status": "PASS",
        "monte_carlo_sampling_performed": False,
    }
    (out / "06_exact_validation_qc.json").write_text(
        json.dumps(qc, indent=2) + "\n"
    )

    lines = [
        "FOUR-CYCLE EXACT VALIDATION",
        "",
        f"Lambda: {lam}",
        f"Feasible trees: {n}",
        f"Communicating classes: {len(classes)}",
        "",
    ]
    for number, component in enumerate(classes, 1):
        names = ", ".join(f"T{i + 1:02d}" for i in component)
        lines.append(
            f"Class {number}: {names}; "
            f"target mass={class_mass[number]:.8f}; "
            f"inaccessible mass={1.0 - class_mass[number]:.8f}"
        )
    lines.extend([
        "",
        "No birth/death moves are available because the graph has no triangles.",
        "Edge exchanges preserve the selected vertex set.",
        "The full target is stationary, but the chain is not irreducible.",
        "A chain started in one class can target only its conditional distribution.",
        "No Monte Carlo sampling or biological validation was performed.",
        "",
        "All implemented exact checks passed.",
    ])
    summary = "\n".join(lines) + "\n"
    (out / "07_summary.txt").write_text(summary)
    print(summary, flush=True)
    print(f"[DONE] Results saved in {out}", flush=True)


if __name__ == "__main__":
    main()
