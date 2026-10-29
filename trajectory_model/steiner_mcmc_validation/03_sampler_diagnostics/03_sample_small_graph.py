#!/usr/bin/env python3
"""Sample the diagonal toy graph and compare with exact probabilities.

Uses actual move proposals and MH accept/reject steps, rather than
drawing directly from the exact target or transition matrix.

Earlier exact-validation outputs are read-only.
"""

import argparse
import csv
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_core(base):
    source = base / "01_exact_small_graph/01_exact_small_graph.py"
    require(source.is_file(), f"Missing script: {source}")
    spec = importlib.util.spec_from_file_location("exact_core", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    # In-memory diagonal graph; source file is never changed.
    core.WEIGHTS[core.canonical_edge("u", "v")] = 1.5
    core.GRAPH_EDGES = tuple(sorted(core.WEIGHTS))
    return core, source


def save_figure(fig, folder, stem):
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=60000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    require(
        0 <= args.burn_in < args.iterations,
        "Burn-in must be smaller than iterations.",
    )
    base = args.base_dir.resolve()
    core, source = load_core(base)
    lam = 1.0

    # Select the latest successful exact diagonal run.
    exact_root = base / "01_exact_small_graph/outputs/diagonal_graph"
    candidates = []
    for qc_path in sorted(exact_root.glob("*/07_exact_validation_qc.json")):
        qc = json.loads(qc_path.read_text())
        if qc.get("status") == "PASS" and qc.get("lambda") == lam:
            candidates.append(qc_path.parent)
    require(candidates, "No successful exact diagonal run at lambda=1.")
    exact_run = candidates[-1]
    reference = read_csv(exact_run / "02_exact_tree_probabilities.csv")
    print(f"[EXACT REFERENCE] {exact_run}", flush=True)

    trees = core.enumerate_trees()
    n = len(trees)
    require(n == 15, "Expected 15 feasible trees.")
    index = {tree: i for i, tree in enumerate(trees)}
    costs = np.array([core.cost(tree) for tree in trees])
    bridges = np.array([
        len(core.vertices_of(tree) - core.ANCHORS) for tree in trees
    ])

    probabilities = np.exp(-lam * (costs - costs.min()))
    probabilities /= probabilities.sum()

    # Check that sampling uses exactly the saved reference state space.
    require(len(reference) == n, "Reference state-count mismatch.")
    for i, row in enumerate(reference):
        require(row["state"] == f"T{i + 1:02d}", "State-order mismatch.")
        require(
            row["edges"] == core.edge_string(trees[i]),
            "Reference edge-set mismatch.",
        )
        require(
            abs(float(row["full_target_probability"]) - probabilities[i])
            <= 1e-12,
            "Reference probability mismatch.",
        )

    stage = base / "03_sampler_diagnostics"
    out = stage / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    figures = out / "figures"
    figures.mkdir()
    print(f"[OUTPUT] {out}", flush=True)

    # Cache each move-specific proposal distribution.
    move_names = ["exchange", "birth", "death"]
    move_functions = [
        core.exchange_distribution,
        core.birth_distribution,
        core.death_distribution,
    ]
    move_probabilities = np.array([
        core.MOVE_PROBABILITIES[name] for name in move_names
    ])

    choices = {}
    proposal = np.zeros((n, n))

    for i, tree in enumerate(trees):
        for m, function in enumerate(move_functions):
            distribution = function(tree)
            targets = np.array(
                [index[t] for t in distribution], dtype=int
            )
            masses = np.array(list(distribution.values()), dtype=float)
            require(abs(masses.sum() - 1) <= 1e-12, "Invalid proposal.")
            choices[(i, m)] = (targets, masses)
            for j, mass in zip(targets, masses):
                proposal[i, j] += move_probabilities[m] * mass

    # Deliberately distinct initial trees: direct anchor edge,
    # the route through x, and a maximum-cost feasible tree.
    starts = [
        index[frozenset({core.canonical_edge("u", "v")})],
        index[frozenset({
            core.canonical_edge("u", "x"),
            core.canonical_edge("x", "v"),
        })],
        int(np.argmax(costs)),
    ]
    require(len(set(starts)) == 3, "Starting trees must be distinct.")

    seed_sequences = np.random.SeedSequence(args.seed).spawn(3)
    retained_chains = []
    trace_rows = []
    acceptance_rows = []
    frequency_rows = []
    comparison_rows = []
    chain_summaries = []

    for chain_number, (start, seed) in enumerate(
        zip(starts, seed_sequences), 1
    ):
        rng = np.random.default_rng(seed)
        state = start
        states = np.empty(args.iterations, dtype=int)
        counts = {
            name: {"selected": 0, "unavailable": 0, "accepted": 0}
            for name in move_names
        }

        for iteration in range(args.iterations):
            held = rng.random() < core.HOLD_PROBABILITY

            if not held:
                m = int(rng.choice(3, p=move_probabilities))
                name = move_names[m]
                if iteration >= args.burn_in:
                    counts[name]["selected"] += 1

                targets, masses = choices[(state, m)]
                proposed = int(rng.choice(targets, p=masses))

                if proposed == state:
                    if iteration >= args.burn_in:
                        counts[name]["unavailable"] += 1
                else:
                    reverse = proposal[proposed, state]
                    require(reverse > 0, "Missing reverse proposal.")
                    log_ratio = (
                        -lam * (costs[proposed] - costs[state])
                        + math.log(reverse)
                        - math.log(proposal[state, proposed])
                    )
                    uniform = max(rng.random(), np.finfo(float).tiny)
                    if math.log(uniform) < min(0.0, log_ratio):
                        state = proposed
                        if iteration >= args.burn_in:
                            counts[name]["accepted"] += 1

            # Save repeated states, including holds and rejections.
            states[iteration] = state

        retained = states[args.burn_in:]
        retained_chains.append(retained)
        empirical = np.bincount(retained, minlength=n) / len(retained)
        tv = 0.5 * np.abs(empirical - probabilities).sum()
        max_error = np.abs(empirical - probabilities).max()

        chain_summaries.append({
            "chain": chain_number,
            "initial_state": f"T{start + 1:02d}",
            "initial_edges": core.edge_string(trees[start]),
            "retained_draws": len(retained),
            "states_visited_after_burn_in": len(np.unique(retained)),
            "total_variation_distance": float(tv),
            "maximum_absolute_probability_error": float(max_error),
        })
        print(
            f"[CHAIN {chain_number}] Start=T{start + 1:02d}; "
            f"visited={len(np.unique(retained))}/{n}; TV={tv:.6f}",
            flush=True,
        )

        for iteration, state_id in enumerate(states, 1):
            trace_rows.append({
                "chain": chain_number,
                "iteration": iteration,
                "retained": iteration > args.burn_in,
                "state": f"T{state_id + 1:02d}",
                "cost": float(costs[state_id]),
                "n_bridges": int(bridges[state_id]),
            })

        for i in range(n):
            frequency_rows.append({
                "chain": chain_number,
                "state": f"T{i + 1:02d}",
                "edges": core.edge_string(trees[i]),
                "exact_probability": float(probabilities[i]),
                "sampled_probability": float(empirical[i]),
                "absolute_error": float(abs(empirical[i] - probabilities[i])),
            })

        for name, values in counts.items():
            selected = values["selected"]
            available = selected - values["unavailable"]
            acceptance_rows.append({
                "chain": chain_number,
                "move": name,
                "selected_after_burn_in": selected,
                "unavailable_after_burn_in": values["unavailable"],
                "accepted_after_burn_in": values["accepted"],
                "unavailable_fraction": (
                    values["unavailable"] / selected if selected else None
                ),
                "acceptance_given_available": (
                    values["accepted"] / available if available else None
                ),
                "acceptance_per_selected_move": (
                    values["accepted"] / selected if selected else None
                ),
            })

    # Exact and empirical edge/bridge inclusion probabilities.
    feature_names = []
    feature_indicators = []

    for edge in core.GRAPH_EDGES:
        feature_names.append(f"Edge {edge[0]}-{edge[1]}")
        feature_indicators.append([edge in tree for tree in trees])
    for vertex in sorted(core.VERTICES - core.ANCHORS):
        feature_names.append(f"Bridge {vertex}")
        feature_indicators.append([
            vertex in core.vertices_of(tree) for tree in trees
        ])

    indicators = np.asarray(feature_indicators, dtype=float)
    exact_support = indicators @ probabilities

    for c, retained in enumerate(retained_chains, 1):
        support = indicators[:, retained].mean(axis=1)
        for j, name in enumerate(feature_names):
            comparison_rows.append({
                "chain": c,
                "feature": name,
                "exact_inclusion_probability": float(exact_support[j]),
                "sampled_inclusion_probability": float(support[j]),
            })

    write_csv(out / "01_chain_traces.csv", trace_rows)
    write_csv(out / "02_tree_probability_comparison.csv", frequency_rows)
    write_csv(out / "03_acceptance_by_move.csv", acceptance_rows)
    write_csv(out / "04_edge_bridge_inclusion.csv", comparison_rows)
    write_csv(out / "05_chain_summary.csv", chain_summaries)

    plt.rcParams.update({
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = ["#2878B5", "#D9822B", "#35945B"]

    # Figure 1: the two specified input graphs.
    positions = {
        "u": (0, 1), "x": (1, 1),
        "y": (0, 0), "v": (1, 0),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    diagonal = core.canonical_edge("u", "v")

    for ax, include_diagonal, title in zip(
        axes, [False, True],
        ["(a) Four-cycle", "(b) Four-cycle with diagonal"],
    ):
        for edge in core.GRAPH_EDGES:
            if edge == diagonal and not include_diagonal:
                continue
            u, v = edge
            a, b = positions[u], positions[v]
            ax.plot(
                [a[0], b[0]], [a[1], b[1]],
                color="#687782", linewidth=1.8, zorder=1,
            )
            midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            ax.text(
                *midpoint, f"{core.WEIGHTS[edge]:.1f}",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=2),
                zorder=3,
            )
        for vertex, position in positions.items():
            anchor = vertex in core.ANCHORS
            ax.scatter(
                *position, s=300,
                marker="o" if anchor else "s",
                color="#C84B43" if anchor else "#CBD5DE",
                edgecolor="black", zorder=4,
            )
            ax.text(*position, vertex, ha="center", va="center", zorder=5)
        ax.set_title(title)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#C84B43",
               label="Anchors"),
        Line2D([], [], marker="s", linestyle="", color="#CBD5DE",
               label="Optional bridges"),
    ], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Synthetic candidate graphs")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, figures, "01_candidate_graphs")

    # Figure 2: exact versus sampled tree probabilities.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(n)
    ax.bar(x, probabilities, color="0.86", label="Exact target")
    for c, retained in enumerate(retained_chains):
        empirical = np.bincount(retained, minlength=n) / len(retained)
        ax.plot(
            x, empirical, "o", ms=4, color=colors[c],
            label=f"Chain {c + 1}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i + 1:02d}" for i in range(n)], rotation=45)
    ax.set_xlabel("Feasible tree")
    ax.set_ylabel("Probability")
    ax.set_title(r"Exact and sampled tree probabilities ($\lambda=1$)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    save_figure(fig, figures, "02_exact_vs_sampled_probabilities")

    # Figure 3: traces. Subsample only for display; all draws are saved.
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for c, retained in enumerate(retained_chains):
        rows = trace_rows[
            c * args.iterations:(c + 1) * args.iterations
        ]
        display_indices = np.unique(np.linspace(
            0, args.iterations - 1,
            min(1500, args.iterations), dtype=int,
        ))
        iterations = display_indices + 1
        for column, key in enumerate(["cost", "n_bridges"]):
            ax = axes[c, column]
            values = [rows[i][key] for i in display_indices]
            ax.plot(iterations, values, lw=0.6, color=colors[c])
            ax.axvline(args.burn_in, color="black", ls="--", lw=1)
            ax.set_ylabel(
                f"Chain {c + 1}\n"
                + ("Cost" if column == 0 else "Bridge count")
            )
            if column == 1:
                ax.set_yticks([0, 1, 2])
    axes[0, 0].set_title("(a) Tree cost")
    axes[0, 1].set_title("(b) Selected bridge count")
    for ax in axes[-1]:
        ax.set_xlabel("Iteration")
    fig.suptitle("Chain traces; dashed line marks the end of burn-in")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, figures, "03_chain_traces")

    # Figure 4: edge and bridge inclusion.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(feature_names))
    ax.bar(x, exact_support, color="0.86", label="Exact target")
    for c, retained in enumerate(retained_chains):
        support = indicators[:, retained].mean(axis=1)
        ax.plot(x, support, "o", color=colors[c], label=f"Chain {c + 1}")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Inclusion probability")
    ax.set_title(r"Edge and bridge inclusion probabilities ($\lambda=1$)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    save_figure(fig, figures, "04_edge_bridge_inclusion")

    manifest = {
        "graph": "four-cycle with diagonal u-v",
        "lambda": lam,
        "iterations_per_chain": args.iterations,
        "burn_in": args.burn_in,
        "n_chains": 3,
        "seed": args.seed,
        "exact_reference_directory": str(exact_run),
        "core_script": str(source),
        "core_script_sha256": file_hash(source),
        "sampling_script_sha256": file_hash(Path(__file__)),
        "hold_probability": core.HOLD_PROBABILITY,
        "move_probabilities": core.MOVE_PROBABILITIES,
        "thinning": "none",
        "trace_plot": "up to 1500 displayed iterations; full traces saved",
        "empirical_results": "descriptive; no arbitrary pass threshold",
        "scope": "synthetic sampling comparison, not biological validation",
    }
    (out / "06_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    lines = [
        "SMALL-GRAPH MCMC SAMPLING",
        "",
        f"Lambda: {lam}",
        f"Iterations per chain: {args.iterations}",
        f"Burn-in: {args.burn_in}",
        f"Retained draws per chain: {args.iterations - args.burn_in}",
        "No thinning; holds and rejections are retained.",
        "",
    ]
    for row in chain_summaries:
        lines.append(
            f"Chain {row['chain']}: start={row['initial_state']}; "
            f"visited={row['states_visited_after_burn_in']}/{n}; "
            f"TV distance={row['total_variation_distance']:.6f}; "
            f"max probability error="
            f"{row['maximum_absolute_probability_error']:.6f}"
        )
    lines.extend([
        "",
        "Four figures saved as both PDF and PNG.",
        "Sampling discrepancies are descriptive, not formal convergence tests.",
        "Autocorrelation, effective sample size, and lambda sensitivity",
        "have not yet been assessed by this script.",
        "",
    ])
    summary = "\n".join(lines)
    (out / "07_summary.txt").write_text(summary)
    print(summary, flush=True)
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
