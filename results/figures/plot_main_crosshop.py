#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main cross-hop figure for the Chain-NPM manuscript.

Four datasets (2 x 2 panels) x six methods x six privacy budgets: the median
2-hop query relative error (``re.re_median_large``) over the archived seeds,
with per-budget min-max error bands.  Natural (non-planted) mode only.

Data source
-----------
``results/by_dataset/<dataset>/*.json`` -- the confirmed archival subset of
the full result grid (see ``results/README.md``).  This script deliberately
never reads ``tmp/results``.

Series conventions (manuscript labels)
--------------------------------------
``chainnpm`` -> Chain-NPM, ``privpetal`` -> PrivPetal, ``lavaprop`` ->
PrivLava, ``privbayes`` -> PrivBayes, ``pbpgm`` -> PB-PGM, and
``privmrf`` -> PerTable, the manuscript's per-table baseline (per-table
PrivMRF with author-protocol FK re-linking). The series *absorbs* the
``pertable`` label; on a duplicate cell ``pertable`` wins. The archive
records this series under the ``pertable`` label, while ``p7_summary.tsv``
still calls it ``privmrf`` — that alias is documented in
``results/tables/tab_re_appendix.tsv`` and ``results/README.md``.

Deduplication rules (guards against the historical degradation bug)
-------------------------------------------------------------------
* A record only ever occupies a dedup key ``(dataset, series, eps, seed)``
  if it carries a usable 2-hop metric; a degraded record without
  ``re.re_median_large[<hop key>]`` never crowds out a valid record that is
  read later.
* ``*_sub*`` files, ``legacy``/``degraded`` files and macOS ``._`` files are
  excluded up front and reported.
* The archive already excludes ``denorm`` (user decision).

Outputs
-------
``fig_main_crosshop_natural.pdf`` / ``.png``  (paper figure, shared log y)
``plot_main_crosshop_coverage.txt``           (per-cell seed coverage report)
``--yscale per-panel`` additionally writes ``..._perpanel.png`` (review aid).

Usage
-----
    python plot_main_crosshop.py
    python plot_main_crosshop.py --yscale per-panel
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths (derived from this file; nothing absolute is hard-coded)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # results/figures
RESULTS_DIR = os.path.dirname(HERE)                        # results
DATA_DIR = os.path.join(RESULTS_DIR, "by_dataset")
BASE_NAME = "fig_main_crosshop_natural"
COVERAGE_TXT = os.path.join(HERE, "plot_main_crosshop_coverage.txt")

DATASETS = ["financial", "imdb", "instacart", "movielens"]
EPS_ORDER = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
EXPECTED_SEEDS = list(range(42, 52))                       # 42..51, n = 10

# chain (financial) and vshape (the rest) name the 2-hop query class differently
HOP2_KEY = {ds: ("2hop_topbot" if ds == "financial" else "cross_r1r2")
            for ds in DATASETS}

# series -> display label, colour, marker, line style
# (colour set = dataviz categorical palette; markers + two dash patterns are
#  the secondary encoding required by the palette validator's CVD warnings)
META = {
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o", "-"),
    "privpetal": ("PrivPetal", "#b45309", "s", "-"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D", "-"),
    "privmrf":   ("PerTable",  "#be123c", "v", (0, (4.0, 1.6))),
    "privbayes": ("PrivBayes", "#166534", "^", "-"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P", (0, (1.3, 1.3))),
}
SERIES_ORDER = list(META)

# JSON method label -> series key
SOURCE_METHODS = {
    "chainnpm": "chainnpm",
    "privpetal": "privpetal",
    "lavaprop": "lavaprop",
    "privmrf": "privmrf",
    "pertable": "privmrf",
    "privbayes": "privbayes",
    "pbpgm": "pbpgm",
}
# lower rank wins on a duplicate cell
SOURCE_RANK = {"privmrf": {"pertable": 0, "privmrf": 1}}

EXCLUDE_PATTERNS = ("legacy", "degraded")


# ---------------------------------------------------------------------------
# Loading / de-duplication
# ---------------------------------------------------------------------------
def exclude_reason(path):
    """Return a reason string if the file must not enter the grid."""
    name = os.path.basename(path)
    if name.startswith("._"):
        return "macOS resource fork"
    if "_sub" in name:
        return "'_sub' subset run"
    for pat in EXCLUDE_PATTERNS:
        if pat in name:
            return "marker '%s'" % pat
    return None


def hop2_value(rec, ds):
    """2-hop median RE of a record, or None when unusable/degraded.

    ``ds`` is the archive folder name and decides which 2-hop query class the
    dataset uses; a mismatch with the record's own ``dataset`` field is
    reported by the caller instead of silently picking one.
    """
    reobj = rec.get("re")
    if not isinstance(reobj, dict):
        return None
    large = reobj.get("re_median_large")
    if not isinstance(large, dict) or not large:
        return None
    val = large.get(HOP2_KEY.get(ds))
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def load_grid(data_dir):
    """Read the archive and return (cells, info).

    cells: {(ds, series, eps, seed): value}
    info:  dict with exclusion/degradation bookkeeping for the coverage report.
    """
    cells = {}
    chosen_label = {}
    info = {"excluded": [], "degraded": [], "unknown_method": [],
            "wrong_mode": [], "bad_eps": [], "unreadable": [],
            "dataset_mismatch": [],
            "files": 0, "duplicates": [], "replaced": []}

    for ds in DATASETS:
        for path in sorted(glob.glob(os.path.join(data_dir, ds, "*.json"))):
            info["files"] += 1
            reason = exclude_reason(path)
            if reason:
                info["excluded"].append((path, reason))
                continue
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except Exception as exc:                      # pragma: no cover
                info["unreadable"].append((path, str(exc)))
                continue
            if not isinstance(rec, dict):
                info["unreadable"].append((path, "not a JSON object"))
                continue
            if rec.get("mode") != "natural":
                info["wrong_mode"].append((path, rec.get("mode")))
                continue

            if rec.get("dataset") != ds:
                info["dataset_mismatch"].append((path, rec.get("dataset")))

            series = SOURCE_METHODS.get(rec.get("method"))
            if series is None:
                info["unknown_method"].append(
                    (path, rec.get("method")))
                continue

            val = hop2_value(rec, ds)
            if val is None:
                # degraded record: reported, but never occupies a dedup key
                info["degraded"].append((path, series, rec.get("eps"),
                                         rec.get("seed")))
                continue

            eps = rec.get("eps")
            key_eps = None
            for cand in EPS_ORDER:
                if eps is not None and abs(float(eps) - cand) < 1e-9:
                    key_eps = cand
                    break
            if key_eps is None:
                info["bad_eps"].append((path, eps))
                continue

            key = (ds, series, key_eps, rec.get("seed"))
            if key not in cells:
                cells[key] = val
                chosen_label[key] = rec.get("method")
                continue
            # duplicate cell: keep the higher-priority source label
            rank = SOURCE_RANK.get(series, {})
            new_rank = rank.get(rec.get("method"), 99)
            keep_rank = rank.get(chosen_label.get(key), 99)
            info["duplicates"].append((key, path))
            if new_rank < keep_rank:
                info["replaced"].append((key, path))
                cells[key] = val
                chosen_label[key] = rec.get("method")
    return cells, info


def aggregate(cells):
    """{(ds, series): {eps: [values]}} plus the seeds behind each cell."""
    vals = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(list)
    for (ds, series, eps, seed), v in cells.items():
        vals[(ds, series)][eps].append(v)
        seeds[(ds, series, eps)].append(seed)
    for k in vals:
        for eps in vals[k]:
            vals[k][eps] = [x for _, x in
                            sorted(zip(seeds[(k[0], k[1], eps)],
                                       vals[k][eps]))]
    for k in seeds:
        seeds[k] = sorted(seeds[k])
    return vals, seeds


def dataset_scale(cells, data_dir):
    """Total ground-truth row count per dataset, taken from the archive."""
    scale = {}
    for ds in DATASETS:
        for path in sorted(glob.glob(os.path.join(data_dir, ds, "*.json"))):
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            counts = rec.get("counts") or {}
            gt = [v for k, v in counts.items()
                  if k.endswith("_gt") and isinstance(v, (int, float))]
            if gt:
                scale[ds] = int(sum(gt))
                break
    return scale


def dataset_shape(cells, data_dir):
    shape = {}
    for ds in DATASETS:
        for path in sorted(glob.glob(os.path.join(data_dir, ds, "*.json"))):
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if rec.get("shape"):
                shape[ds] = rec["shape"]
                break
    return shape


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def fmt_scale(n):
    if n is None:
        return "?"
    return "%.2fM" % (n / 1e6) if n >= 1e6 else "%dK" % round(n / 1e3)


def write_coverage(out_path, cells, seeds, info, scale, shapes,
                   data_dir=DATA_DIR):
    """Write the per-cell seed coverage report to ``out_path``.

    NOTE: every loop below binds its own file variable (``fp``); a loop
    variable named like the output path used to shadow it and made this
    function overwrite an *input* JSON with the report.
    """
    def vals_of(ds, series):
        out = defaultdict(list)
        for (d, s, eps, _seed), v in cells.items():
            if d == ds and s == series:
                out[eps].append(v)
        return out

    lines = []
    lines.append("Coverage report -- %s (natural mode, 2-hop median RE)"
                 % BASE_NAME)
    lines.append("source: %s" % data_dir)
    lines.append("records read: %d files | excluded %d | degraded %d | "
                 "wrong mode %d | unknown method %d | unexpected eps %d | "
                 "unreadable %d"
                 % (info["files"], len(info["excluded"]),
                    len(info["degraded"]), len(info["wrong_mode"]),
                    len(info["unknown_method"]), len(info["bad_eps"]),
                    len(info["unreadable"])))
    lines.append("cells in grid: %d (expected %d = %d datasets x %d series "
                 "x %d budgets x %d seeds)"
                 % (len(cells), len(DATASETS) * len(SERIES_ORDER)
                    * len(EPS_ORDER) * len(EXPECTED_SEEDS),
                    len(DATASETS), len(SERIES_ORDER), len(EPS_ORDER),
                    len(EXPECTED_SEEDS)))
    lines.append("")

    missing_cells, short_cells, single_cells, over_cells = [], [], [], []
    lines.append("-- per-cell seed coverage (n = seeds present) --")
    header = ("dataset    series     " +
              "".join("%9s" % ("eps=%g" % e) for e in EPS_ORDER) + "   total")
    lines.append(header)
    lines.append("-" * len(header))
    for ds in DATASETS:
        for series in SERIES_ORDER:
            row = []
            tot = 0
            for eps in EPS_ORDER:
                got = seeds.get((ds, series, eps), [])
                n = len(got)
                tot += n
                row.append("%9s" % str(n))
                if n == 0:
                    missing_cells.append((ds, series, eps))
                elif n == 1:
                    single_cells.append((ds, series, eps, got))
                elif n < len(EXPECTED_SEEDS):
                    short_cells.append((ds, series, eps, got))
                elif n > len(EXPECTED_SEEDS):
                    over_cells.append((ds, series, eps, got))
            lines.append("%-10s %-10s" % (ds, META[series][0]) +
                         "".join(row) + "%8d" % tot)
    lines.append("")

    lines.append("-- medians / min / max actually plotted --")
    lines.append("dataset    series     " +
                 "".join("%26s" % ("eps=%g" % e) for e in EPS_ORDER))
    for ds in DATASETS:
        for series in SERIES_ORDER:
            cells_txt = []
            for eps in EPS_ORDER:
                v = None
                key = (ds, series, eps)
                got = seeds.get(key)
                if got:
                    v = [cells[(ds, series, eps, s)] for s in got]
                if v:
                    cells_txt.append("%8.3f[%6.3f,%6.3f]" %
                                     (float(np.median(v)), min(v), max(v)))
                else:
                    cells_txt.append("%26s" % "n/a")
            lines.append("%-10s %-10s" % (ds, META[series][0]) +
                         "".join(cells_txt))
    lines.append("")

    lines.append("-- anomalies --")
    if not (missing_cells or short_cells or single_cells or over_cells):
        lines.append("none: every (dataset, series, eps) cell carries all %d "
                     "seeds %s" % (len(EXPECTED_SEEDS),
                                   "%d-%d" % (EXPECTED_SEEDS[0],
                                              EXPECTED_SEEDS[-1])))
    for ds, series, eps in missing_cells:
        lines.append("MISSING  %s / %s / eps=%g: no seed at all "
                     "(series absent from this panel)" % (ds, META[series][0], eps))
    for ds, series, eps, got in short_cells:
        gone = [s for s in EXPECTED_SEEDS if s not in got]
        lines.append("SHORT    %s / %s / eps=%g: n=%d, missing seeds %s"
                     % (ds, META[series][0], eps, len(got), gone))
    for ds, series, eps, got in single_cells:
        lines.append("SINGLETON %s / %s / eps=%g: n=1 (seed %s) -> drawn as a "
                     "single marker, never silently joined by a line"
                     % (ds, META[series][0], eps, got[0]))
    for ds, series, eps, got in over_cells:
        lines.append("EXTRA    %s / %s / eps=%g: n=%d seeds %s"
                     % (ds, META[series][0], eps, len(got), got))
    lines.append("")

    lines.append("-- excluded / degraded inputs --")
    if (not info["excluded"] and not info["degraded"]
            and not info["wrong_mode"] and not info["unknown_method"]
            and not info["bad_eps"] and not info["unreadable"]
            and not info["dataset_mismatch"]):
        lines.append("none")
    for fp, reason in info["excluded"]:
        lines.append("EXCLUDED %s (%s)" % (os.path.relpath(fp, RESULTS_DIR),
                                           reason))
    for fp, series, eps, seed in info["degraded"]:
        lines.append("DEGRADED %s (series %s, eps=%s, seed=%s): no "
                     "re.re_median_large metric -> kept out of every dedup key"
                     % (os.path.relpath(fp, RESULTS_DIR), series, eps, seed))
    for fp, mode in info["wrong_mode"]:
        lines.append("WRONG-MODE %s (mode=%s)"
                     % (os.path.relpath(fp, RESULTS_DIR), mode))
    for fp, method in info["unknown_method"]:
        lines.append("UNKNOWN-METHOD %s (method=%s)"
                     % (os.path.relpath(fp, RESULTS_DIR), method))
    for fp, eps in info["bad_eps"]:
        lines.append("BAD-EPS %s (eps=%s)"
                     % (os.path.relpath(fp, RESULTS_DIR), eps))
    for fp, err in info["unreadable"]:
        lines.append("UNREADABLE %s (%s)"
                     % (os.path.relpath(fp, RESULTS_DIR), err))
    for fp, ds_field in info["dataset_mismatch"]:
        lines.append("MISMATCH %s: folder vs record 'dataset' field = %s "
                     "(folder wins)" % (os.path.relpath(fp, RESULTS_DIR),
                                        ds_field))
    lines.append("")

    lines.append("-- plotted envelope vs paper y-axis limits --")
    lines.append("paper figure y limits (log10 axis): %.3f .. %.1f" % Y_LIM)
    gmin = gmax = None
    for ds in DATASETS:
        per_series = [vals_of(ds, s) for s in SERIES_ORDER]
        vals_ds = [v for pts in per_series for e in pts for v in pts[e]]
        if not vals_ds:
            continue
        gmin = min(vals_ds) if gmin is None else min(gmin, min(vals_ds))
        gmax = max(vals_ds) if gmax is None else max(gmax, max(vals_ds))
        lines.append("%-10s min-max band envelope %.4f .. %.4f"
                     % (ds, min(vals_ds), max(vals_ds)))
    inside = (gmin is not None and gmin >= Y_LIM[0] and gmax <= Y_LIM[1])
    lines.append("global envelope %.4f .. %.4f -> %s"
                 % (gmin, gmax,
                    "inside the axis limits (no clipping)"
                    if inside else "OUTSIDE the axis limits -- FIX Y_LIM"))
    lines.append("")

    lines.append("-- panel scale (ground-truth rows, from record 'counts') --")
    for ds in DATASETS:
        lines.append("%-10s shape=%-7s rows=%s" % (ds, shapes.get(ds, "?"),
                                                   fmt_scale(scale.get(ds))))
    lines.append("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {
        "out_path": out_path,
        "missing": missing_cells,
        "short": short_cells,
        "single": single_cells,
        "over": over_cells,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


Y_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
# the whole envelope (0.035 .. 9.7) sits strictly inside these limits, with a
# visible margin at both ends so no band ever appears cut by the frame
Y_LIM = (0.02, 20.0)


def draw_panel(ax, vals, ds, shared_y):
    drawn = []
    for series in SERIES_ORDER:
        label, color, marker, ls = META[series]
        pts = vals.get((ds, series))
        if not pts:
            continue
        # contiguous runs of budgets that carry >= 2 seeds; a run is drawn as
        # one polyline + min-max band.  A budget with a single seed breaks the
        # run, so no line is ever drawn across (or through) a lone marker.
        runs, cur = [], []
        singles = []
        for e in EPS_ORDER:
            v = pts.get(e)
            if v is None:
                if cur:
                    runs.append(cur)
                    cur = []
            elif len(v) >= 2:
                cur.append(e)
            else:
                singles.append(e)
                if cur:
                    runs.append(cur)
                    cur = []
        if cur:
            runs.append(cur)

        first = True
        for run in runs:
            med = [float(np.median(pts[e])) for e in run]
            lo = [float(min(pts[e])) for e in run]
            hi = [float(max(pts[e])) for e in run]
            ax.fill_between(run, lo, hi, color=color, alpha=0.09,
                            linewidth=0, zorder=1)
            # thin envelope lines keep each band readable where bands overlap
            ax.plot(run, lo, color=color, lw=0.55, alpha=0.6, zorder=2)
            ax.plot(run, hi, color=color, lw=0.55, alpha=0.6, zorder=2)
            h, = ax.plot(run, med, color=color, ls=ls, marker=marker,
                         markersize=4.2, markerfacecolor=color,
                         markeredgecolor="white", markeredgewidth=0.5,
                         label=label if first else None, zorder=3)
            drawn.append(h)
            first = False
        if singles:
            # n = 1: a lone marker, never silently joined by a line
            med = [float(np.median(pts[e])) for e in singles]
            ax.plot(singles, med, ls="none", marker=marker, markersize=4.6,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.1, zorder=4,
                    label=label if first else None)

    ax.set_xscale("log", base=2)
    ax.set_xticks(EPS_ORDER)
    ax.set_xticklabels(["%g" % e for e in EPS_ORDER])
    ax.set_xlim(0.082, 4.1)
    ax.minorticks_off()

    ax.axhline(1.0, color="#9a9a9a", lw=0.7, ls=":", zorder=0)

    if shared_y:
        ax.set_yscale("log")
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(["%g" % t for t in Y_TICKS])
    else:
        allv = [v for s in SERIES_ORDER
                for e in vals.get((ds, s), {})
                for v in vals[(ds, s)][e]]
        lo = min(allv) * 0.85
        hi = max(allv) * 1.12
        ax.set_ylim(lo, hi)

    ax.grid(True, which="major", axis="both", color="#c9c9c9",
            alpha=0.35, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    return drawn


def render(vals, scale, shapes, out_pdf, out_png, shared_y=True):
    apply_style()
    if shared_y:
        env = [v for s in SERIES_ORDER for ds in DATASETS
               for e in vals.get((ds, s), {}) for v in vals[(ds, s)][e]]
        if env and (min(env) < Y_LIM[0] or max(env) > Y_LIM[1]):
            sys.exit("plotted envelope %.4f..%.4f falls outside the paper "
                     "y limits %s: refusing to clip silently"
                     % (min(env), max(env), Y_LIM))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6),
                             sharex=True, sharey=shared_y)
    axes = np.asarray(axes).ravel()
    handles = []
    for ax, ds in zip(axes, DATASETS):
        handles = draw_panel(ax, vals, ds, shared_y) or handles
        shape, rows = shapes.get(ds), scale.get(ds)
        # never print a placeholder: fall back to the bare dataset name
        title = ("%s (%s, %s rows)" % (ds, shape, fmt_scale(rows))
                 if shape and rows else ds)
        ax.set_title(title, pad=3.5)
    # set_yticklabels/set_xticklabels re-enable labels on the inner panels, so
    # switch them off explicitly: y labels only in the left column, x labels
    # only in the bottom row
    for i, ax in enumerate(axes):
        row, col = divmod(i, 2)
        ax.tick_params(axis="y", labelleft=(col == 0))
        ax.tick_params(axis="x", labelbottom=(row == 1))
    for ax in axes[2:]:
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
    for ax in axes[::2]:
        ax.set_ylabel("2-hop query RE (median)")

    fig.legend(handles, [META[s][0] for s in SERIES_ORDER],
               loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=len(SERIES_ORDER), frameon=False,
               handletextpad=0.5, columnspacing=1.3,
               markerscale=1.15, borderaxespad=0.0)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.875, bottom=0.095,
                        hspace=0.22, wspace=0.05)
    if out_pdf:
        fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_pdf, out_png


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="directory holding <dataset>/*.json (default: %s)"
                         % DATA_DIR)
    ap.add_argument("--outdir", default=HERE,
                    help="output directory (default: %s)" % HERE)
    ap.add_argument("--yscale", choices=["shared-log", "per-panel"],
                    default="shared-log",
                    help="'shared-log' = paper figure (one shared log y "
                         "range for all panels); 'per-panel' = per-panel "
                         "linear y range (review aid only)")
    args = ap.parse_args(argv)

    cells, info = load_grid(args.data_dir)
    if not cells:
        sys.exit("no usable records under %s" % args.data_dir)
    vals, seeds = aggregate(cells)
    scale = dataset_scale(cells, args.data_dir)
    shapes = dataset_shape(cells, args.data_dir)

    cov = write_coverage(os.path.join(args.outdir,
                                      os.path.basename(COVERAGE_TXT)),
                         cells, seeds, info, scale, shapes,
                         data_dir=args.data_dir)
    if not (os.path.isfile(cov["out_path"])
            and os.path.getsize(cov["out_path"]) > 0):
        sys.exit("coverage report was not written to %s" % cov["out_path"])
    print("wrote %s" % cov["out_path"])

    if args.yscale == "shared-log":
        # paper figure: one shared y range across all four panels
        pdf = os.path.join(args.outdir, BASE_NAME + ".pdf")
        png = os.path.join(args.outdir, BASE_NAME + ".png")
        render(vals, scale, shapes, pdf, png, shared_y=True)
        print("wrote %s\nwrote %s" % (pdf, png))
    else:
        # review aid only: per-panel linear y range, no PDF
        png = os.path.join(args.outdir, BASE_NAME + "_perpanel.png")
        render(vals, scale, shapes, None, png, shared_y=False)
        print("wrote %s (review aid, not the paper figure)" % png)
    print("cells=%d  missing=%d  short=%d  singleton=%d  extra=%d"
          % (len(cells), len(cov["missing"]), len(cov["short"]),
             len(cov["single"]), len(cov["over"])))
    # numerical digest, for cross-checking against results/tables/p7_summary.tsv
    print("\n== median 2-hop RE (n) per dataset x series x eps ==")
    for ds in DATASETS:
        for series in SERIES_ORDER:
            pts = vals.get((ds, series), {})
            if not pts:
                print("%-10s %-10s absent" % (ds, META[series][0]))
                continue
            txt = "  ".join("%.1f:%.3f(n=%d)" % (e, float(np.median(pts[e])),
                                                 len(pts[e]))
                            for e in EPS_ORDER if e in pts)
            print("%-10s %-10s %s" % (ds, META[series][0], txt))
    return 0


if __name__ == "__main__":
    sys.exit(main())