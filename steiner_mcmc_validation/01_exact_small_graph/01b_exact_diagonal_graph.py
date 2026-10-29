#!/usr/bin/env python3
"""Exact validation on the four-cycle with diagonal u-v.

Reuses the Stage 01 proposal and enumeration functions.
No Monte Carlo sampling is performed.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path


def load_core():
    source = Path(__file__).with_name("01_exact_small_graph.py")
    if not source.is_file():
        raise FileNotFoundError(source)

    spec = importlib.util.spec_from_file_location("exact_core", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Modify this imported module in memory only.
    # The original script and its four-cycle graph remain unchanged.
    module.WEIGHTS[module.canonical_edge("u", "v")] = 1.5
    module.GRAPH_EDGES = tuple(sorted(module.WEIGHTS))
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    args = parser.parse_args()

    core = load_core()
    require = core.require
    lam = args.lambda_value
    require(math.isfinite(lam) and lam > 0, "Lambda must be positive.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir()
    print(f"[OUTPUT] {out}", flush=True)

    trees = core.enumerate_trees()
    n = len(trees)
    index = {tree: i for i, tree in enumerate(trees)}
    costs = [core.cost(tree) for tree in trees]

    minimum_cost = min(costs)
    weights = [
        math.exp(-lam * (value - minimum_cost))
        for value in costs
    ]
    normalizer = sum(weights)
    probabilities = [value / normalizer for value in weights]
    log_z = -lam * minimum_cost + math.log(normalizer)

    proposal = [[0.0] * n for _ in range(n)]
    move_rows = []

    for i, tree in enumerate(trees):
        distributions = {
            "exchange": core.exchange_distribution(tree),
            "birth": core.birth_distribution(tree),
            "death": core.death_distribution(tree),
        }

        for move, distribution in distributions.items():
            require(
                abs(sum(distribution.values()) - 1.0) <= 1e-12,
                f"Unnormalized {move} proposal.",
            )

            for proposed, conditional_probability in distribution.items():
                require(proposed in index, "Proposal outside feasible space.")
                j = index[proposed]

                contribution = (
                    core.MOVE_PROBABILITIES[move]
                    * conditional_probability
                )
                proposal[i][j] += contribution

                move_rows.append({
                    "from": f"T{i + 1:02d}",
                    "to": f"T{j + 1:02d}",
                    "move": move,
                    "changes_state": i != j,
                    "probability_given_move": conditional_probability,
                    "mixture_contribution": contribution,
                })

    kernel = [[0.0] * n for _ in range(n)]
    pair_rows = []

    for i in range(n):
        require(
            abs(sum(proposal[i]) - 1.0) <= 1e-12,
            "Proposal row does not sum to one.",
        )

        for j in range(n):
            if i == j or proposal[i][j] == 0:
                continue

            forward = proposal[i][j]
            reverse = proposal[j][i]
            require(reverse > 0, "Non-self proposal has no reverse.")

            log_ratio = (
                -lam * (costs[j] - costs[i])
                + math.log(reverse)
                - math.log(forward)
            )
            acceptance = math.exp(min(0.0, log_ratio))

            kernel[i][j] = (
                (1.0 - core.HOLD_PROBABILITY)
                * forward
                * acceptance
            )

            pair_rows.append({
                "from": f"T{i + 1:02d}",
                "to": f"T{j + 1:02d}",
                "cost_change": costs[j] - costs[i],
                "q_forward": forward,
                "q_reverse": reverse,
                "proposal_ratio": reverse / forward,
                "mh_acceptance": acceptance,
                "transition_probability": kernel[i][j],
            })

        kernel[i][i] = 1.0 - sum(kernel[i])

    # Identify communicating classes from reversible transition support.
    for i in range(n):
        for j in range(n):
            require(
                (kernel[i][j] > 0) == (kernel[j][i] > 0),
                "Asymmetric transition support.",
            )

    classes = []
    unseen = set(range(n))

    while unseen:
        start = min(unseen)
        component = {start}
        stack = [start]

        while stack:
            i = stack.pop()
            for j in range(n):
                if kernel[i][j] > 0 and j not in component:
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

    # Structural checks include non-anchor leaves.
    require(n == 15, f"Expected 15 feasible trees; found {n}.")
    require(all(core.is_feasible(tree) for tree in trees), "Invalid tree.")
    require(
        len(classes) == 1,
        f"Expected one communicating class; found {len(classes)}.",
    )

    available_moves = {
        row["move"]
        for row in move_rows
        if row["changes_state"]
    }
    require(
        available_moves == {"exchange", "birth", "death"},
        f"Missing a non-self move type: {available_moves}",
    )

    asymmetric_pairs = [
        row for row in pair_rows
        if abs(row["q_forward"] - row["q_reverse"]) > 1e-12
    ]
    require(
        bool(asymmetric_pairs),
        "No asymmetric proposal pair exercised the MH correction.",
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

    require(max_row_residual <= 1e-12, "Invalid transition row sums.")
    require(
        max_stationarity_residual <= 1e-12,
        "Stationarity check failed.",
    )
    require(
        all(
            kernel[i][i] >= core.HOLD_PROBABILITY - 1e-12
            for i in range(n)
        ),
        "Holding probability check failed.",
    )

    state_rows = []
    for i, tree in enumerate(trees):
        number = class_of[i]
        state_rows.append({
            "state": f"T{i + 1:02d}",
            "vertices": ";".join(sorted(core.vertices_of(tree))),
            "edges": core.edge_string(tree),
            "n_bridges": len(core.vertices_of(tree) - core.ANCHORS),
            "cost": costs[i],
            "lambda": lam,
            "full_target_probability": probabilities[i],
            "communicating_class": number,
            "conditional_target_probability": (
                probabilities[i] / class_mass[number]
            ),
            "inaccessible_target_mass": 1.0 - class_mass[number],
        })

    core.write_csv(out / "01_graph_edges.csv", [
        {"u": u, "v": v, "weight": core.WEIGHTS[(u, v)]}
        for u, v in core.GRAPH_EDGES
    ])
    core.write_csv(out / "02_exact_tree_probabilities.csv", state_rows)
    core.write_csv(out / "03_transition_matrix_long.csv", [
        {
            "from": f"T{i + 1:02d}",
            "to": f"T{j + 1:02d}",
            "proposal_probability": proposal[i][j],
            "transition_probability": kernel[i][j],
        }
        for i in range(n)
        for j in range(n)
    ])
    core.write_csv(out / "04_detailed_balance_flows.csv", flow_rows)
    core.write_csv(out / "05_move_proposals.csv", move_rows)
    core.write_csv(out / "06_mh_proposal_ratios.csv", pair_rows)

    qc = {
        "graph": "four-cycle with diagonal u-v",
        "anchors": sorted(core.ANCHORS),
        "optional_bridges": sorted(core.VERTICES - core.ANCHORS),
        "allows_nonanchor_leaves": True,
        "lambda": lam,
        "log_partition_function": log_z,
        "n_feasible_trees": n,
        "n_communicating_classes": len(classes),
        "class_sizes": [len(component) for component in classes],
        "nonself_move_types": sorted(available_moves),
        "n_directed_asymmetric_proposal_pairs": len(asymmetric_pairs),
        "hold_probability": core.HOLD_PROBABILITY,
        "conditional_move_probabilities": core.MOVE_PROBABILITIES,
        "max_kernel_row_sum_residual": max_row_residual,
        "max_stationarity_residual": max_stationarity_residual,
        "max_detailed_balance_residual": max_balance_residual,
        "status": "PASS",
        "monte_carlo_sampling_performed": False,
    }
    (out / "07_exact_validation_qc.json").write_text(
        json.dumps(qc, indent=2) + "\n"
    )

    summary = "\n".join([
        "FOUR-CYCLE WITH DIAGONAL: EXACT VALIDATION",
        "",
        f"Lambda: {lam}",
        f"Feasible trees: {n}",
        f"Communicating classes: {len(classes)}",
        f"Class sizes: {[len(component) for component in classes]}",
        f"Non-self move types: {', '.join(sorted(available_moves))}",
        f"Asymmetric directed proposal pairs: {len(asymmetric_pairs)}",
        "",
        f"Maximum row-sum residual: {max_row_residual:.3e}",
        f"Maximum stationarity residual: {max_stationarity_residual:.3e}",
        f"Maximum detailed-balance residual: {max_balance_residual:.3e}",
        "",
        "The enumerated transition graph is connected for this graph.",
        "Birth/death proposals change selected bridge membership.",
        "Asymmetric proposals use the full MH proposal-ratio correction.",
        "Explicit holding makes the finite irreducible chain aperiodic.",
        "Therefore the full target is the unique stationary distribution.",
        "This result is specific to this graph and implemented kernel.",
        "No Monte Carlo sampling or biological validation was performed.",
        "",
        "All implemented exact checks passed.",
        "",
    ])
    (out / "08_summary.txt").write_text(summary)
    print(summary, flush=True)
    print(f"[DONE] Results saved in {out}", flush=True)


if __name__ == "__main__":
    main()
