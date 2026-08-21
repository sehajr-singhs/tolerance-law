"""Tolerance Law analysis.

Turns the raw phase-diagram JSONs into:
  1. the learned-clearance boundary c*(N, width) -- interpolated at the
     success threshold,
  2. the power-law fit c* ~ a * N^-alpha (per capacity) and its adaptive
     counterpart N*(c) ~ b * c^-beta,
  3. the phase-diagram figure,
  4. a numbers.tex with every number the papers cite.

Usage:
    PYTHONPATH=src python scripts/analyze_tolerance.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUCCESS_THRESHOLD = 0.50


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def boundary_at(successes: list[float], clearances: list[float],
                thresh: float = SUCCESS_THRESHOLD) -> float | None:
    """Smallest clearance at which success crosses thresh (linear interp)."""
    cs = np.asarray(clearances, dtype=float)
    ss = np.asarray(successes, dtype=float)
    order = np.argsort(cs)
    cs, ss = cs[order], ss[order]
    for i in range(len(cs) - 1):
        if ss[i] >= thresh:
            return float(cs[i])
        if ss[i] < thresh <= ss[i + 1]:
            # linear interpolation between grid points
            t = (thresh - ss[i]) / max(ss[i + 1] - ss[i], 1e-9)
            return float(cs[i] + t * (cs[i + 1] - cs[i]))
    return float(cs[-1]) if ss[-1] >= thresh else None


def fit_powerlaw(xs: list[float], ys: list[float]) -> dict:
    """log y = log a + b log x -> y = a x^b. Returns fit stats."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    keep = (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return {"a": None, "b": None, "r2": None, "n": len(x)}
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    (b, loga), *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    yhat = np.exp(loga) * x ** b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"a": float(np.exp(loga)), "b": float(b), "r2": float(r2),
            "n": len(x)}


def analyze(results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sweep = load(results / "sweep.json")
    adaptive = load(results / "adaptive.json")
    capacity = load(results / "capacity.json")
    teacher = load(results / "teacher.json")

    report: dict = {}

    # ---- 1. phase diagram / boundary ---------------------------------- #
    if sweep:
        clearances = sorted({r["clearance"] for r in sweep})
        widths = sorted({r["width"] for r in sweep})
        budgets = sorted({r["n_demos"] for r in sweep})
        seeds = sorted({r["seed"] for r in sweep})

        def cell_mean(c, w, n):
            vals = [r["success"] for r in sweep
                    if r["clearance"] == c and r["width"] == w
                    and r["n_demos"] == n]
            return float(np.mean(vals)) if vals else None

        boundaries = {}
        for w in widths:
            for n in budgets:
                cs = []
                ss = []
                for c in clearances:
                    m = cell_mean(c, w, n)
                    if m is not None:
                        cs.append(c)
                        ss.append(m)
                b = boundary_at(ss, cs)
                if b is not None:
                    boundaries[f"w{w}_N{n}"] = b

        # power law c* ~ a N^-b at the middle capacity
        fits = {}
        w_mid = widths[len(widths) // 2]
        ns, bs = [], []
        for n in budgets:
            key = f"w{w_mid}_N{n}"
            if key in boundaries:
                ns.append(n)
                bs.append(boundaries[key])
        fits["budget_powerlaw"] = fit_powerlaw(ns, bs)
        report["boundaries"] = boundaries
        report["phase"] = {
            "clearances": clearances, "widths": widths, "budgets": budgets,
            "seeds": seeds,
            "matrix": {f"w{w}_N{n}": {c: cell_mean(c, w, n) for c in clearances}
                       for w in widths for n in budgets},
        }
        report["fits"] = fits
        print(f"phase diagram: {len(clearances)} clearances x "
              f"{len(widths)} widths x {len(budgets)} budgets x "
              f"{len(seeds)} seeds = {len(sweep)} cells", flush=True)
        for k, v in sorted(boundaries.items()):
            print(f"  boundary {k}: c* = {v:.5f}", flush=True)
        if fits.get("budget_powerlaw", {}).get("b") is not None:
            f = fits["budget_powerlaw"]
            print(f"  power law c* ~ {f['a']:.4f} N^{f['b']:.3f} (r2={f['r2']:.3f})",
                  flush=True)

    # ---- 2. adaptive budget: N*(c) ------------------------------------- #
    if adaptive:
        learned = [r for r in adaptive if r.get("learned")]
        report["adaptive_budget"] = adaptive
        if learned:
            cs = [r["clearance"] for r in learned]
            ns = [r["min_budget"] for r in learned]
            fit = fit_powerlaw(cs, ns)  # N* ~ b c^beta
            report["adaptive_fit"] = fit
            print("adaptive budget N*(c):", flush=True)
            for r in sorted(learned, key=lambda r: r["clearance"]):
                print(f"  c={r['clearance']:.4f} -> N*={r['min_budget']}", flush=True)
            if fit["b"] is not None:
                print(f"  N* ~ {fit['a']:.1f} c^{fit['b']:.3f} (r2={fit['r2']:.3f})",
                      flush=True)

    # ---- 3. adaptive capacity ------------------------------------------- #
    if capacity:
        report["adaptive_capacity"] = capacity
        for r in capacity:
            print(f"  c={r['clearance']:.4f} -> min capacity "
                  f"{r.get('min_capacity')} (learned={r.get('learned')})",
                  flush=True)

    # ---- 4. teacher ------------------------------------------------------ #
    if teacher:
        report["teacher"] = teacher
        print("teacher:", {r["clearance"]: r["expert_success"] for r in teacher},
              flush=True)

    # ---- 5. LaTeX macros ------------------------------------------------- #
    mac = []
    mac.append(r"% Tolerance Law numbers -- generated by analyze_tolerance.py")
    mac.append(r"% Do not edit by hand.")
    if teacher:
        t = {r["clearance"]: r["expert_success"] for r in teacher}
        mac.append(r"\newcommand{\TLteacherTight}{%d\%%}" % int(t.get(0.0005, 1.0) * 100))
        mac.append(r"\newcommand{\TLteacherLoose}{%d\%%}" % int(t.get(0.004, 1.0) * 100))
    if sweep:
        mac.append(r"\newcommand{\TLclearances}{%d}" % len(clearances))
        mac.append(r"\newcommand{\TLwidths}{%d}" % len(widths))
        mac.append(r"\newcommand{\TLbudgets}{%d}" % len(budgets))
        mac.append(r"\newcommand{\TLseeds}{%d}" % len(seeds))
        mac.append(r"\newcommand{\TLcells}{%d}" % len(sweep))
        mac.append(r"\newcommand{\TLthreshold}{%.2f}" % SUCCESS_THRESHOLD)
        f = fits.get("budget_powerlaw", {})
        if f.get("b") is not None:
            mac.append(r"\newcommand{\TLalpha}{%.2f}" % (-f["b"]))
            mac.append(r"\newcommand{\TLalphaSE}{--}")
            mac.append(r"\newcommand{\TLrSq}{%.3f}" % f["r2"])
            mac.append(r"\newcommand{\TLa}{%.4f}" % f["a"])
        if f.get("b") is not None:
            mac.append(r"\newcommand{\TLalphaNeg}{%.3f}" % f["b"])
    if adaptive:
        fit = report.get("adaptive_fit", {})
        if fit.get("b") is not None:
            mac.append(r"\newcommand{\TLbeta}{%.2f}" % (-fit["b"]))
            mac.append(r"\newcommand{\TLb}{%.2f}" % fit["a"])
    (out / "numbers.tex").write_text("\n".join(mac) + "\n")
    print(f"wrote {out / 'numbers.tex'}", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out / 'report.json'}", flush=True)

    # ---- 6. phase-diagram figure ----------------------------------------- #
    if sweep:
        _figure(sweep, clearances, widths, budgets, out)


def _figure(sweep, clearances, widths, budgets, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for w in widths:
        for n in budgets:
            for c in clearances:
                vals = [r["success"] for r in sweep
                        if r["clearance"] == c and r["width"] == w
                        and r["n_demos"] == n]
                if vals:
                    rows.append((w, n, c, float(np.mean(vals)),
                                 float(np.std(vals)) if len(vals) > 1 else 0.0))

    ncols = len(widths)
    nrows = len(budgets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False, sharex=True)
    cmap = plt.get_cmap("RdYlGn")
    for (w, n, c, m, s) in rows:
        r = nrows - 1 - budgets.index(n)
        col = widths.index(w)
        ax = axes[r][col]
        color = cmap(np.clip(m, 0, 1))
        ax.bar(c, m, width=min(clearances) * 0.8, color=color,
               edgecolor="black", linewidth=0.4, yerr=s if s > 0 else None,
               capsize=1.5, error_kw={"linewidth": 0.6})
        ax.axhline(SUCCESS_THRESHOLD, color="0.5", linestyle="--", linewidth=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        if r == nrows - 1:
            ax.set_xlabel(f"width {w}")
        if col == 0:
            ax.set_ylabel(f"N={n}\nlearned success")
        ax.set_xticks(clearances)
        ax.set_xticklabels([f"{c*1000:.1f}" for c in clearances], fontsize=6)
    fig.suptitle("Learned insertion success vs clearance (mm) — state policies",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "fig_phase_diagram.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}", flush=True)

    # boundary curve: c* vs N at the widest capacity
    if len(budgets) >= 3:
        w = widths[-1]
        fig2, ax = plt.subplots(figsize=(4.2, 3.0))
        ns = []
        bs = []
        for n in budgets:
            vals = [r["success"] for r in sweep
                    if r["width"] == w and r["n_demos"] == n]
            if not vals:
                continue
            cs = []
            ss = []
            for c in clearances:
                cv = [r["success"] for r in sweep
                      if r["width"] == w and r["n_demos"] == n
                      and r["clearance"] == c]
                if cv:
                    cs.append(c)
                    ss.append(float(np.mean(cv)))
            b = boundary_at(ss, cs)
            if b is not None:
                ns.append(n)
                bs.append(b)
        ax.plot(ns, bs, "o-", color="#0b5d4f")
        if len(ns) >= 3:
            fit = fit_powerlaw(ns, bs)
            if fit["b"] is not None:
                xx = np.linspace(min(ns), max(ns), 100)
                ax.plot(xx, fit["a"] * xx ** fit["b"], "--", color="0.4",
                        label=f"c* $\\propto$ N$^{{{fit['b']:.2f}}}$ (r²={fit['r2']:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("demo budget N")
        ax.set_ylabel("learnable clearance c* (m)")
        ax.set_title("Tolerance Law boundary moves with data")
        ax.legend(fontsize=8)
        fig2.tight_layout()
        p2 = out / "fig_boundary_law.png"
        fig2.savefig(p2, dpi=150)
        print(f"wrote {p2}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_tolerance/state")
    ap.add_argument("--out", default="paper/generated")
    args = ap.parse_args()
    analyze(Path(args.results_dir), Path(args.out))


if __name__ == "__main__":
    main()
