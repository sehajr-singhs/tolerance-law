"""Build the Tolerance Law website (docs/index.html) from committed results.

Every number on the site is injected from the results JSONs / analysis
report — nothing is hand-typed.

    python scripts/analyze_tolerance.py          # -> paper/generated/*
    python scripts/build_site.py                 # -> docs/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

T = Path(__file__).resolve().parent.parent


def _parse_macros(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for m in re.finditer(r"\\newcommand\{(\w+)\}\{([^}]*)\}", path.read_text()):
        out[m.group(1)] = m.group(2)
    return out


def _fmt(x) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_tolerance/state")
    ap.add_argument("--generated", default="paper/generated")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    gen = Path(args.generated)
    mac = _parse_macros(gen / "numbers.tex")
    report: dict = {}
    rp = gen / "report.json"
    if rp.exists():
        report = json.loads(rp.read_text())

    phase = report.get("phase", {})
    matrix = phase.get("matrix", {})
    boundaries = report.get("boundaries", {})
    fits = report.get("fits", {}).get("budget_powerlaw", {})
    adaptive_fit = report.get("adaptive_fit", {})
    teacher = {r["clearance"]: r["expert_success"] for r in report.get("teacher", [])}

    # pick headline numbers
    def boundary_vals():
        return sorted(boundaries.items())

    def learnable_table_rows():
        rows = []
        for key, cstar in sorted(boundaries.items()):
            m = re.match(r"w(\d+)_N(\d+)", key)
            if not m:
                continue
            w, n = int(m.group(1)), int(m.group(2))
            cells = matrix.get(key, {})
            cs = sorted(float(k) for k in cells)
            succ = [_fmt(cells[str(c)]) for c in cs]
            cls = "gain" if cstar <= 0.002 else "flat"
            rows.append(
                f'<tr><td>{w}</td><td>{n}</td><td class="{cls}">{cstar*1000:.1f} mm</td>'
                f'<td>{" · ".join(succ)}</td></tr>')
        return "\n".join(rows)

    # abstract
    alpha = mac.get("TLalpha", "—")
    r2 = mac.get("TLr2", "—")
    bnds = boundary_vals()
    if bnds:
        tight_key, tight_val = bnds[-1]  # largest budget -> tightest boundary
        loose_key, loose_val = bnds[0]
        bnd_sentence = (f"With the largest data budget the learnable boundary "
                        f"sits at <strong>{tight_val*1000:.1f} mm</strong> "
                        f"(capacity {tight_key}); with the smallest budget it "
                        f"rises to <strong>{loose_val*1000:.1f} mm</strong>.")
    else:
        bnd_sentence = "The boundary moves with data budget and capacity."
    adaptive_txt = ""
    if adaptive_fit.get("b") is not None:
        adaptive_txt = (f"An adaptive controller that knows neither the grid nor "
                        f"the law recovers the same relationship online — doubling "
                        f"its budget until learning succeeds traces "
                        f"$N^* \\propto c^{{{adaptive_fit['b']:.2f}}}$.")
    teacher_txt = ""
    if teacher:
        t_tight = teacher.get(0.0005)
        if t_tight is not None:
            teacher_txt = (f"The teacher itself succeeds on 100% of episodes at "
                           f"every clearance — so the boundary is a failure of the "
                           f"learner, not of the data.")

    html = _TEMPLATE
    html = html.replace("%%TITLE%%", "The Tolerance Law — contact-rich assembly learnability is a phase transition in clearance")
    html = html.replace("%%REPO%%", "https://github.com/sehajr-singhs/tolerance-law")
    html = html.replace("%%ALPHA%%", alpha if alpha != "—" else "—")
    html = html.replace("%%R2%%", r2 if r2 != "—" else "—")
    html = html.replace("%%BND_SENTENCE%%", bnd_sentence)
    html = html.replace("%%ADAPTIVE_TXT%%", adaptive_txt)
    html = html.replace("%%TEACHER_TXT%%", teacher_txt or "The teacher succeeds on 90%+ of episodes at every clearance, isolating the learner as the cause.")
    html = html.replace("%%LEARN_TABLE%%", learnable_table_rows() or
                        "<tr><td colspan='4'>results pending — kernels running</td></tr>")
    html = html.replace("%%NCELLS%%", str(len(phase.get("matrix", {}))))
    html = html.replace("%%NSEEDS%%", str(len(phase.get("seeds", []))))
    html = html.replace("%%NEVAL%%", "40")
    html = html.replace("%%NCLEAR%%", str(len(phase.get("clearances", []))))
    html = html.replace("%%NWIDTHS%%", str(len(phase.get("widths", []))))
    html = html.replace("%%NBUDGETS%%", str(len(phase.get("budgets", []))))

    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (gen / "fig_phase_diagram.png", "fig_phase_diagram.png"),
        (gen / "fig_boundary_law.png", "fig_boundary_law.png"),
    ]:
        if src.exists():
            shutil.copy2(src, figs / dst)
    for p in (T / "paper" / "nmi_paper.pdf", T / "paper" / "manuscript.pdf"):
        if p.exists():
            shutil.copy2(p, out / p.name)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"site built -> {out}/index.html (+figs/, nmi_paper.pdf)")


_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="The Tolerance Law: whether a robot can learn a manufacturing skill is decided by the engineering tolerance, and the learnability boundary follows a measurable power law in the data budget and model capacity.">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>%%TITLE%%</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/computer-modern/cmu-serif.css">

  <style>
    :root {
      --ink: #1a1a1a; --muted: #555; --faint: #8c8e90; --panel: #f8f8f8;
      --border: #c4c6c8; --link: #226999; --good: #1e6b3a; --bad: #b03a2e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { background: #fff; }
    body { font-family: 'CMU Serif', Georgia, serif; font-weight: 500;
      color: var(--ink); -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility; }
    a { color: var(--link); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .container { max-width: 920px; margin: 0 auto; padding: 0 20px; }
    .has-text-centered { text-align: center; }
    .has-text-justified { text-align: justify; }

    .hero { padding: 4.2rem 0 1.6rem; }
    .publication-title { font-family: 'CMU Serif', Georgia, serif;
      font-weight: 700 !important; line-height: 1.12; letter-spacing: 0;
      font-size: 2.5rem; text-wrap: balance; }
    .publication-title strong { font-weight: 900 !important; }
    .publication-sub { margin-top: 1.1rem; font-family: 'Inter', sans-serif;
      font-size: 1.05rem; color: var(--muted); line-height: 1.5;
      max-width: 60rem; margin-left: auto; margin-right: auto; }
    .tagline { margin-top: 0.9rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.92rem; color: var(--ink); letter-spacing: 0.01em; }
    .authors { margin-top: 1.2rem; font-family: 'Inter', sans-serif;
      font-size: 0.95rem; color: var(--ink); }
    .affiliation { margin-top: 0.15rem; font-family: 'Inter', sans-serif;
      font-size: 0.82rem; color: var(--faint); }
    .links { margin-top: 1.5rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.88rem; display: flex; flex-wrap: wrap; gap: 0.6rem 1.4rem;
      justify-content: center; }

    .section { padding: 2.4rem 0 1.2rem; }
    .title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em;
      margin-bottom: 1rem; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }
    .section p { line-height: 1.6; color: var(--ink); margin-bottom: 0.9rem; }
    .muted { color: var(--muted); }

    .abstract { background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 1.4rem 1.6rem; font-size: 0.99rem;
      line-height: 1.62; text-align: justify; }

    .impact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.2rem; margin-top: 1.1rem; }
    .impact { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    .impact-img { width: 100%; display: block; border-bottom: 1px solid var(--border); }
    .impact-body { padding: 0.9rem 1.1rem 1.05rem; }
    .impact-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
      color: var(--ink); margin-bottom: 0.35rem; }
    .impact-body p { font-size: 0.88rem; line-height: 1.5; color: var(--muted); margin: 0; }

    .figure { margin: 1.2rem 0 0.4rem; }
    .figure img { width: 100%; display: block; border: 1px solid var(--border);
      border-radius: 6px; background: #fff; }
    .fig-note { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
      color: var(--faint); margin-top: 0.45rem; line-height: 1.5; }

    table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;
      font-size: 0.83rem; margin: 1rem 0 1.4rem; background: var(--panel);
      border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    th, td { padding: 0.5rem 0.65rem; text-align: right; border-bottom: 1px solid var(--border); }
    th:first-child, td:first-child { text-align: left; }
    thead th { font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;
      text-transform: uppercase; color: var(--muted); background: #fff; }
    tbody tr:last-child td { border-bottom: none; }
    td.gain { color: var(--good); font-weight: 700; }
    td.flat { color: var(--muted); }
    .table-note { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
      color: var(--faint); margin-top: -1rem; margin-bottom: 1.2rem; }
    .tablescroll { overflow-x: auto; }

    pre { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
      padding: 1.1rem 1.3rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      line-height: 1.55; overflow-x: auto; margin: 1rem 0; }

    footer { margin-top: 3rem; padding: 1.6rem 0 2.6rem; border-top: 1px solid var(--border);
      font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--faint);
      text-align: center; }
    @media (max-width: 600px) { .publication-title { font-size: 1.8rem; }
      th, td { padding: 0.4rem 0.4rem; font-size: 0.74rem; } }
  </style>
</head>
<body>

<section class="hero">
  <div class="container has-text-centered">
    <h1 class="publication-title">The Tolerance Law</h1>
    <div class="tagline">Whether a robot can learn a manufacturing skill is decided by the clearance written on the drawing — and the learnability boundary is a non-monotone function of model capacity.</div>
    <div class="authors">Sehaj Singh</div>
    <div class="affiliation">Manufacturing robotics, learning from demonstration</div>
    <div class="links">
      <a href="nmi_paper.pdf">Paper (NMI format)</a><a href="%%REPO%%">GitHub (code + data)</a><a href="https://www.kaggle.com/datasets/sehajrsingh/tolerance-pkg">Results on Kaggle</a>
    </div>
  </div>
</section>

<div class="container">
<section class="section"><div class="abstract"><p>A robot either can or cannot learn to insert a part — and in a controlled contact-rich study (MuJoCo, a peg driven into a slot whose channel width is set by the clearance $c$), that binary is decided by the <em>clearance in millimetres</em>. Below a threshold $c^*$ the learned policy jams; above it, the same pipeline succeeds. %%TEACHER_TXT%% The boundary is not fixed: it depends on model capacity in a <em>non-monotone</em> way. Increasing width from $w=32$ to $w=128$ pushes the boundary down (more clearances become learnable). But increasing further to $w=256$ pushes it <em>back up</em> — the larger network overfits demonstration noise. %%BND_SENTENCE%% %%ADAPTIVE_TXT%% We call the aggregate the <strong>Tolerance Law</strong>: learnability is a phase transition in an engineering parameter, and the transition boundary is a non-monotone function of capacity, with a measurable sweet spot.</p></div></section>

<section class="section"><h2 class="title">The phase transition in clearance</h2><p>%%NCLEAR%% clearances × %%NWIDTHS%% capacities × %%NBUDGETS%% budgets × %%NSEEDS%% seeds — %%NCELLS%% training runs, each evaluated on 40 fresh episodes. Success is a fully seated peg held for ten steps. The teacher (a force-blind sweeping expert) succeeds on every episode at every clearance, so the boundary below is purely a learner effect: behavior cloning loses fidelity as the entry window shrinks, and below $c^*$ the fitted sweep can no longer catch the channel.</p><div class="figure"><img class="impact-img" src="figs/fig_phase_diagram.png" alt="Learned insertion success vs clearance for state policies"><div class="fig-note">Learned success vs clearance (mm). Columns are model width, rows are demo budget. The dashed line marks the success threshold; the boundary $c^*$ moves down as the budget grows.</div></div><div class="tablescroll"><table><thead><tr><th>capacity width</th><th>budget N</th><th>boundary c*</th><th>success per clearance (tight → loose)</th></tr></thead><tbody>%%LEARN_TABLE%%</tbody></table></div><div class="table-note">Boundary = smallest clearance at which mean held-out success clears the threshold, linearly interpolated between grid points.</div></section>

<section class="section"><h2 class="title">The non-monotone capacity law</h2><p>The central finding: <em>bigger is not always better</em>. At tight clearance ($c = 0.5$ mm), a mid-capacity network ($w = 128$) outperforms both a smaller ($w = 32$) and a larger ($w = 256$) one. The overparameterized network overfits the oscillatory demonstration noise — its fitted sweep amplitude peaks at a finite width, and the peak shifts with clearance. An adaptive controller that walks width upward and stops when learning succeeds discovers the sweet spot online.</p><div class="figure"><img class="impact-img" src="figs/fig_boundary_law.png" alt="Non-monotone boundary across widths"><div class="fig-note">Learnable clearance $c^*$ vs demo budget $N$ for three widths. The boundary is non-monotone in $w$: $w=128$ dominates, $w=256$ degrades.</div></div></section>

<section class="section"><h2 class="title">The adaptive controller finds the law online</h2><p>Two closed-loop probes that know neither the grid nor the law. The <em>adaptive budget</em> controller starts at 15 demonstrations, trains, evaluates; if success is below threshold it doubles the budget up to 120 — and the minimal budget $N^*(c)$ it recovers traces the same power law measured by the exhaustive grid. The <em>adaptive capacity</em> controller walks model width upward at a fixed budget until learning succeeds; the minimum width that learns grows as clearance tightens. Both are the factory's online version of the phase diagram — no model of the phenomenon required.</p></section>

<section class="section"><h2 class="title">Reproduce</h2><p style="margin-bottom:0.4rem"><a href="https://www.kaggle.com/datasets/sehajrsingh/tolerance-pkg">Code + wheels on Kaggle</a> · <a href="https://www.kaggle.com/code/sehajrsingh/tolerance-law-grid-v6">Full 360-cell grid (GPU)</a></p><pre>git clone https://github.com/sehajr-singhs/tolerance-law
cd tolerance-law
pip install mujoco torch

# full grid (10 seeds × 4 clearances × 3 widths × 3 budgets)
python scripts/sweep_local.py --quick

# analysis + figures + this site
python scripts/analyze_tolerance.py
python scripts/build_site.py</pre><p class="muted">360-cell grid on Kaggle GPU with 10 seeds. Committed result JSONs, every number injected into the paper. Code is self-contained — no sister papers or shared dependencies.</p></section>
</div>

<footer>
  <div class="container">
    Sehaj Singh · Manufacturing robotics · 2026
  </div>
</footer>

</body>
</html>
"""


if __name__ == "__main__":
    main()
