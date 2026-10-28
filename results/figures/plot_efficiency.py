#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Efficiency figure: cost against accuracy, one panel per real database.

x = cross-hop median large-query relative error (the class the paper reports;
``hop2_re`` defines it per schema), y = synthesis-phase wall clock in seconds,
both logarithmic.  Six methods, six budgets each (points joined in budget
order), so a method's curve reads "how much accuracy its budget buys, and at
what time cost".

Sources
-------
chainnpm : the reported row-level quota-calibration arm
           (``szquota_common.szquota_records()``, i.e. results/arms/absorb or
           tmp/absorb_proto/campaign_szquota{2}); the szquota arm is what every
           Chain-NPM number in the paper uses.
others   : released archive results/by_dataset (pertable -> PrivMRF,
           lavaprop -> PrivLava, ...), canonicalised by p7_common.

Timing convention (matches the paper's efficiency claims)
---------------------------------------------------------
y is the SYNTHESIS phase: ``runtime_sec`` (model + marginals + sampling); the
evaluation is excluded.  Where a cell has no ``runtime_sec`` (PrivLava's
V-shaped cells), the fallback order is model_sec+synthesize_sec, then
``total_runtime_sec`` -- the fallback actually used is recorded per series in
the coverage report, because ``total_runtime_sec`` includes the evaluation.
PrivPetal's ``runtime_sec`` is that method's own wall clock on A800 GPU(s)
while the other five are single-CPU; the asymmetry is conservative for
Chain-NPM and is stated in the caption.

The figure refuses to be drawn if any series is missing a budget it should
have, or if the layout audit finds text outside the canvas / colliding /
over another panel's plot area."""
import argparse
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(RESULTS_DIR)
sys.path.insert(0, os.path.join(RESULTS_DIR, "tables"))
from p7_common import hop2_re, DATASETS, EPS_ORDER       # noqa: E402
import szquota_common                                    # noqa: E402
from plot_controlled_2panel import audit_layout          # noqa: E402

BASE_NAME = os.path.join(HERE, "fig_efficiency")
COVERAGE_TXT = os.path.join(HERE, "plot_efficiency_coverage.txt")

# same series identity as plot_dataset_sweep / plot_main_effect, so the paper's
# figures read as one family
META = {
    "privmrf":   ("PrivMRF",   "#be123c", "v"),
    "privbayes": ("PrivBayes", "#166534", "^"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D"),
    "privpetal": ("PrivPetal", "#b45309", "s"),
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o"),
}
SERIES_ORDER = ["privmrf", "privbayes", "pbpgm", "lavaprop", "privpetal",
                "chainnpm"]
EPS_LABELS = ["0.1", "0.2", "0.4", "0.8", "1.6", "3.2"]


def runtime_of(d):
    """(seconds, which-field) for the synthesis phase, or (None, why)."""
    v = d.get("runtime_sec")
    if isinstance(v, (int, float)) and v > 0:
        return float(v), "runtime_sec"
    a, b = d.get("model_sec"), d.get("synthesize_sec")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a + b > 0:
        return float(a + b), "model+synthesize_sec"
    t = d.get("total_runtime_sec")
    if isinstance(t, (int, float)) and t > 0:
        return float(t), "total_runtime_sec(eval included)"
    return None, "missing"


def load(arm):
    """(dataset, series, eps) -> {'re': {seed: v}, 'rt': {seed: v}},
    plus the field actually used per series and the seeds per cell."""
    rows = (szquota_common.szquota_records() if arm == "szquota"
            else None)
    if rows is None:
        from p7_common import load_rows
        rows = load_rows()
    per = defaultdict(lambda: {"re": {}, "rt": {}})
    fields = defaultdict(set)
    for d in rows:
        if d.get("mode") != "natural":
            continue
        series = d.get("method")
        if series not in META:
            continue
        ds, eps = d["_dataset"], d.get("eps")
        v = hop2_re(d)
        if v is not None:
            per[(ds, series, eps)]["re"][d.get("seed")] = float(v)
        rt, which = runtime_of(d)
        if rt is not None:
            per[(ds, series, eps)]["rt"][d.get("seed")] = rt
            fields[series].add(which)
    return per, fields


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 6.6,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 0.9,
        "lines.markersize": 2.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def decade_ticks(lo, hi):
    """Decade ticks inside [lo, hi] with short plain labels (see the caller)."""
    import math
    d0 = int(math.floor(math.log10(lo)))
    d1 = int(math.ceil(math.log10(hi)))
    vals = [10.0 ** k for k in range(d0, d1 + 1) if lo <= 10.0 ** k <= hi]
    labels = [("%gk" % (v / 1000.0)) if v >= 1000 else ("%g" % v)
              for v in vals]
    return vals, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["szquota", "released"], default="szquota",
                    help="chainnpm rows: reported szquota arm (default) or "
                         "the released archive")
    args = ap.parse_args()

    per, fields = load(args.arm)
    problems, notes = [], []

    # --- completeness: every series must carry every budget ----------------
    grid = defaultdict(list)                       # (ds, series) -> eps list
    for (ds, series, eps) in per:
        grid[(ds, series)].append(eps)
    for ds in DATASETS:
        for series in SERIES_ORDER:
            have = sorted(e for e in grid.get((ds, series), [])
                          if per[(ds, series, e)]["re"] and
                          per[(ds, series, e)]["rt"])
            missing = [e for e in EPS_ORDER if e not in have]
            if missing:
                problems.append("%s/%s misses budget(s) %s"
                                % (ds, series, missing))
    if problems:
        sys.exit("refusing to plot an incomplete grid:\n  " +
                 "\n  ".join(problems))

    apply_style()
    fig = plt.figure(figsize=(7.03, 2.16))
    gs = fig.add_gridspec(1, 4, left=0.058, right=0.995, top=0.795,
                          bottom=0.30, wspace=0.30)
    axes = []
    xs_all, ys_all = [], []
    table_rows = []
    for i, ds in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        for series in SERIES_ORDER:
            label, color, marker = META[series]
            xs, ys = [], []
            for eps in EPS_ORDER:
                cell = per.get((ds, series, eps))
                if not cell:
                    continue
                if not cell["re"] or not cell["rt"]:
                    continue
                x = st.median(cell["re"].values())
                y = st.median(cell["rt"].values())
                xs.append(x)
                ys.append(y)
                xs_all.append(x)
                ys_all.append(y)
                table_rows.append((ds, label, eps, x, y, len(cell["rt"])))
            ax.plot(xs, ys, color=color, marker=marker, markersize=2.6,
                    zorder=3, label=label)
            ax.plot(xs, ys, color=color, linewidth=0.0, marker=marker,
                    markersize=2.6, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        shape = "deep chain" if ds == "financial" else "V-shaped"
        ax.set_title("%s (%s)" % (ds, shape), pad=3.0)
        ax.grid(True, which="major", color="#dddddd", linewidth=0.4, zorder=1)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("synthesis wall clock (s)")
        ax.set_xlabel("cross-hop query RE")

    # one shared legend under the panels, six columns
    handles = [Line2D([], [], color=META[s][1], marker=META[s][2],
                      markersize=2.8, linewidth=0.9, label=META[s][0])
               for s in SERIES_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               handletextpad=0.35, columnspacing=1.30, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 0.015))

    # identical axes on every panel, one tick per decade with plain labels:
    # matplotlib's default log locator packs $3\times10^{-2}$-style labels until
    # they overlap, which the audit rejects
    xlo, xhi = min(xs_all) * 0.5, max(xs_all) * 2.2
    ylo, yhi = min(ys_all) * 0.5, max(ys_all) * 2.2
    xt, xl = decade_ticks(xlo, xhi)
    yt, yl = decade_ticks(ylo, yhi)
    for ax in axes:
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    audit_layout(fig, axes, "efficiency")

    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)

    # --- coverage report ---------------------------------------------------
    with open(COVERAGE_TXT, "w") as f:
        f.write("fig_efficiency -- arm=%s\n" % args.arm)
        f.write("x = cross-hop median large-query RE (hop2_re), "
                "y = synthesis wall clock (s); medians over seeds\n")
        f.write("timing fields used per series: %s\n"
                % {s: sorted(fields.get(s, [])) for s in SERIES_ORDER})
        f.write("\n%-11s %-10s %5s %10s %12s %5s\n"
                % ("dataset", "method", "eps", "RE", "time_s", "seeds"))
        for ds, label, eps, x, y, n in sorted(table_rows):
            f.write("%-11s %-10s %5s %10.5f %12.3f %5d\n"
                    % (ds, label, eps, x, y, n))
    for s in SERIES_ORDER:
        flds = sorted(fields.get(s, []))
        if any("eval included" in x for x in flds):
            notes.append("%s: some cells fall back to total_runtime_sec "
                         "(evaluation included)" % META[s][0])
    print("wrote %s.pdf/.png and %s" % (BASE_NAME, COVERAGE_TXT))
    print("series timing fields: %s"
          % {META[s][0]: sorted(fields.get(s, [])) for s in SERIES_ORDER})
    for n in notes:
        print("note: %s" % n)


if __name__ == "__main__":
    main()