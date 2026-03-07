#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
from contextlib import redirect_stdout
from io import StringIO

import numpy as np


def _cell_code(cells, idx):
    return "".join(cells[idx]["source"])


def _strip_lines(code, startswith_prefixes=(), contains_tokens=()):
    out = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in startswith_prefixes):
            continue
        if any(tok in line for tok in contains_tokens):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def build_runner(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    code_graph = _strip_lines(
        _cell_code(cells, 2),
        contains_tokens=("plt.", "plot_results(", "display("),
    )
    code_consts = _cell_code(cells, 4)
    code_n_convert = _strip_lines(_cell_code(cells, 9), startswith_prefixes=("print(",))

    code_fn = _cell_code(cells, 10)
    code_eta = _cell_code(cells, 11)
    code_stage2_iter = _cell_code(cells, 12)
    code_stage2 = _cell_code(cells, 13)

    code_stage3_setup = _cell_code(cells, 16)
    code_nearest = _cell_code(cells, 17)
    code_lists = _cell_code(cells, 21)
    code_stage3_main = _cell_code(cells, 22)
    code_stage3_result = _cell_code(cells, 23)

    runner_code = "\n".join(
        [
            code_graph,
            code_consts,
            code_fn,
        ]
    )

    def run_trial(params):
        ns = {"math": math, "np": np}
        capture = StringIO()
        with redirect_stdout(capture):
            exec(runner_code, ns)

            # Override initial parameters from section 2
            ns["d1_отн"] = params["d1_отн"]
            ns["c_а1_отн"] = params["c_а1_отн"]
            ns["с_а_отн"] = params["с_а_отн"]
            ns["Hт_ср_отн"] = params["Hт_ср_отн"]
            ns["H_т1"] = params["H_т1"]
            ns["R_ср1"] = 0.5

            # Re-run section 2 + 3 with trial parameters
            exec(code_n_convert, ns)
            exec(code_eta, ns)
            exec(code_stage2_iter, ns)
            exec(code_stage2, ns)
            exec(code_stage3_setup, ns)
            exec(code_nearest, ns)
            # Equivalent of non-plotting part from notebook cell 18
            N = ns["N"]
            ns["R_ср1_list"] = np.linspace(ns["R_ср1"], ns["R_ср1"], N)
            if N <= 6:
                ns["Kh"] = np.linspace(0.99, 0.94, N)
            else:
                ns["Kh"] = np.concatenate([
                    np.linspace(0.99, 0.94, 6),
                    np.full((N - 6), 0.94)
                ])
            # Equivalent of non-printing calculations from notebook cell 19
            ns["α_1_вна"] = math.radians(90)
            ns["α_2_вна"] = math.atan(ns["c_а1_отн"] / ns["c_1u_отн"])
            exec(code_lists, ns)
            exec(code_stage3_main, ns)
            exec(code_stage3_result, ns)

        keys = ("π_ла_полн", "η_ла_полн", "π_к_полн", "η_к_полн")
        metrics = {k: float(ns[k]) for k in keys}
        return metrics

    return run_trial


def frange(start, stop, step):
    values = np.arange(start, stop + step * 0.5, step)
    return [round(float(v), 10) for v in values]


def score(metrics, targets):
    # Relative L1 error with a small safeguard for near-zero targets
    total = 0.0
    details = {}
    for key, target in targets.items():
        denom = max(abs(target), 1e-9)
        err = abs(metrics[key] - target) / denom
        details[key] = err
        total += err
    return total, details


def build_targets(args):
    # Compare only by compressor targets (as requested).
    if args.target_pi_k is not None and args.target_eta_k is not None:
        return {
            "π_к_полн": args.target_pi_k,
            "η_к_полн": args.target_eta_k,
        }

    # Safe defaults from technical assignment + section 1 formula
    k = 1.4
    pi_k = 18.2
    eta_pol = 0.92
    eta_k = (pi_k ** ((k - 1) / k) - 1) / (pi_k ** ((k - 1) / (k * eta_pol)) - 1)
    return {
        "π_к_полн": pi_k,
        "η_к_полн": eta_k,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Полный перебор комбинаций начальных параметров (по заданной сетке) для main.ipynb."
    )
    parser.add_argument("--notebook", default="main.ipynb")
    parser.add_argument("--step-d1", type=float, default=0.05)
    parser.add_argument("--step-ca1", type=float, default=0.05)
    parser.add_argument("--step-ca-last", type=float, default=0.05)
    parser.add_argument("--step-Hsr", type=float, default=0.05)
    parser.add_argument("--step-H1", type=float, default=0.01)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--target-pi-k", type=float, default=None)
    parser.add_argument("--target-eta-k", type=float, default=None)
    parser.add_argument("--show-errors", type=int, default=3)
    parser.add_argument("--max-error-pct", type=float, default=None)
    parser.add_argument("--export-file", default="подбор_комбинаций.csv")
    args = parser.parse_args()

    run_trial = build_runner(args.notebook)

    # Ranges from section 2 of notebook
    ranges = {
        "d1_отн": frange(0.3, 0.6, args.step_d1),
        "c_а1_отн": frange(0.4, 0.7, args.step_ca1),
        "с_а_отн": frange(0.25, 0.6, args.step_ca_last),
        "Hт_ср_отн": frange(0.1, 0.4, args.step_Hsr),
        "H_т1": frange(0.15, 0.25, args.step_H1),
    }

    all_combinations = list(itertools.product(
        ranges["d1_отн"],
        ranges["c_а1_отн"],
        ranges["с_а_отн"],
        ranges["Hт_ср_отн"],
        ranges["H_т1"],
    ))

    targets = build_targets(args)

    print("Целевые параметры для сравнения:")
    for k, v in targets.items():
        print(f"  {k} = {v:.6f}")

    results = []
    failed = 0
    shown_errors = 0

    for idx, combo in enumerate(all_combinations, 1):
        params = {
            "d1_отн": combo[0],
            "c_а1_отн": combo[1],
            "с_а_отн": combo[2],
            "Hт_ср_отн": combo[3],
            "H_т1": combo[4],
        }

        try:
            metrics = run_trial(params)
            total_score, detail_score = score(metrics, targets)

            row = {
                "score": total_score,
                "params": params,
                "metrics": metrics,
                "detail_score": detail_score,
            }
            results.append(row)

        except Exception as exc:
            failed += 1
            if shown_errors < args.show_errors:
                shown_errors += 1
                print(f"Ошибка #{shown_errors} для params={params}: {type(exc).__name__}: {exc}")

        if idx % 500 == 0 or idx == len(all_combinations):
            print(f"Обработано {idx}/{len(all_combinations)}; ошибок: {failed}")

    results.sort(key=lambda x: x["score"])

    filtered_results = results
    if args.max_error_pct is not None:
        threshold = args.max_error_pct / 100.0
        filtered_results = [
            r for r in results
            if r["detail_score"]["π_к_полн"] <= threshold
            and r["detail_score"]["η_к_полн"] <= threshold
        ]

    # Export from best to worst (sorted by total error).
    with open(args.export_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "total_error_pct",
            "d1_отн",
            "c_а1_отн",
            "с_а_отн",
            "Hт_ср_отн",
            "H_т1",
            "π_к_полн",
            "η_к_полн",
            "err_π_к_pct",
            "err_η_к_pct",
        ])
        for rank, item in enumerate(filtered_results, 1):
            writer.writerow([
                rank,
                item["score"] * 100.0,
                item["params"]["d1_отн"],
                item["params"]["c_а1_отн"],
                item["params"]["с_а_отн"],
                item["params"]["Hт_ср_отн"],
                item["params"]["H_т1"],
                item["metrics"]["π_к_полн"],
                item["metrics"]["η_к_полн"],
                item["detail_score"]["π_к_полн"] * 100.0,
                item["detail_score"]["η_к_полн"] * 100.0,
            ])

    print(f"\nЭкспортировано вариантов: {len(filtered_results)} -> {args.export_file}")
    if args.max_error_pct is not None:
        print(
            "Фильтр по ошибкам: "
            f"err_π_к <= {args.max_error_pct:.3f}% и err_η_к <= {args.max_error_pct:.3f}%"
        )

    print("\nЛучшие комбинации:")
    for rank, item in enumerate(filtered_results[:args.top], 1):
        print(f"\n#{rank}: score={item['score']:.8f}")
        print("  params:")
        for k, v in item["params"].items():
            print(f"    {k} = {v}")
        print("  по целевым параметрам:")
        pi_k_val = item["metrics"]["π_к_полн"]
        eta_k_val = item["metrics"]["η_к_полн"]
        pi_k_err_pct = item["detail_score"]["π_к_полн"] * 100
        eta_k_err_pct = item["detail_score"]["η_к_полн"] * 100
        print(f"    π_к_полн = {pi_k_val:.6f}; ошибка = {pi_k_err_pct:.3f}%")
        print(f"    η_к_полн = {eta_k_val:.6f}; ошибка = {eta_k_err_pct:.3f}%")
        print(f"    суммарная ошибка (π_к + η_к) = {item['score'] * 100:.3f}%")


if __name__ == "__main__":
    main()
