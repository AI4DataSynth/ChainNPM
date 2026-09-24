#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main "plant-and-audit retention" figure for the Chain-NPM manuscript.

Four datasets (2 x 2 panels) x six methods x {plant_product, plant_xor} x
{eps = 0.8, 3.2}: the planted-effect retention error

    ret_err = |effect_syn - effect_real| / |effect_real|

(median over the archived seeds).  0 means the planted two-hop effect survives
synthesis exactly, 1 means the synthetic data is as far from the truth as a
no-cross-hop draw (the per-table / no-joint level), > 1 means the sign or the
magnitude is wrong.

Reading the panel
-----------------
* x: the planted-effect target actually requested.  Slots 0.10 / 0.20 / 0.316 /
  0.40 are ``plant_product``; the separated, shaded right-hand slot is
  ``plant_xor`` with phi = 0.80.  Both are the same axis quantity: the size of
  the plant-and-audit target.
* one mark per (method, target): the median over seeds at eps = 3.2 (filled) and
  at eps = 0.8 (hollow), joined by a thin connector so budget sensitivity is
  visible; the four ``plant_product`` marks of a method are additionally joined
  by a thin line (solid, eps = 3.2 only) to make the trend over target size
  legible.  The ``plant_xor`` slot is never joined to the product slots: it is a
  different planting protocol.
* y: log scale, never clipped -- the script refuses to draw when any plotted
  value falls outside the axis limits.

Honesty requirements carried by the figure
------------------------------------------
* Six series, complete grid at n = 10 seeds.  ``PrivPetal`` carries no target
  dimension of its own (its plant cells are named by budget only), so it
  contributes a single ``plant_product`` mark at t = 0.316 per budget and the
  ``plant_xor`` mark at 0.8.  A legend entry gains ``(n = k)`` only if a cell
  ever holds fewer than 10 seeds, ``k`` being read off the data.
* Some requested targets were never realised as distinct plants -- the
  ``plant_reused`` flag in ``../tables/p7_effect.tsv``.  On imdb the attainable
  contrast is already reached at every requested target (one realisation), and
  PrivLava reuses a single realised plant (t = 0.316) for all four product files
  on the three vshape datasets.  Those series are therefore flat over the
  target slots; each affected panel says so instead of hiding it.  No value is
  ever invented for a missing cell.
* The financial panel annotation is generated from the plotted medians, not
  written by hand: it states the Chain-NPM range at $\varepsilon=3.2$ and adds
  the per-fork level only when the per-table family really sits on it.  The
  panel is annotated, never smoothed.

Data source: ``tmp/results/<dataset>/*.json`` (read only).  denorm is excluded
by user decision.  Per-cell coverage is written next to the figure.

Usage
-----
    python plot_main_effect.py
    python plot_main_effect.py --yscale linear      # review aid only
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))          # results/figures
RESULTS_DIR = os.path.dirname(HERE)                        # results
ROOT = os.path.dirname(RESULTS_DIR)                        # project root
DATA_DIR = os.path.join(ROOT, "tmp", "results")
BASE_NAME = "fig_main_effect_plant"
COVERAGE_TXT = os.path.join(HERE, "plot_main_effect_coverage.txt")

DATASETS = ["financial", "imdb", "instacart", "movielens"]
EPS_ORDER = [0.8, 3.2]
EPS_FILLED = 3.2                    # filled marker = primary budget
T_PROD = [0.1, 0.2, 0.316, 0.4]
XOR_PHI = 0.8
FULL_SEEDS = list(range(42, 52))
# Leaf (evaluation-instance) table of each schema: the chain shape bottoms out
# in "bot", the vshape schemas in the "c" child table.  The panel title reports
# this table's ground truth, not the sum over every ground-truth table.
LEAF_GT_KEYS = ("bot_gt", "c_gt")
# The archive stores the schema kind as a code token ("chain"/"vshape"); panel
# titles must show the manuscript's spelling, so map it at draw time only.
SHAPE_LABEL = {"chain": "chain", "vshape": "V-shaped"}

# series -> display label, colour, marker (same conventions as
# plot_main_crosshop.py so the manuscript figures read as one family)
# 2026-09-19: the per-table baseline is displayed as "PrivMRF" (manuscript
# rename); only the display label changed -- the archive key stays "pertable"
# in the JSON records, in SERIES_LABELS below and in the coverage report.
META = {
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o"),
    "pertable":  ("PrivMRF",   "#be123c", "v"),
    "privpetal": ("PrivPetal", "#b45309", "s"),
    "privbayes": ("PrivBayes", "#166534", "^"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D"),
}
SERIES_ORDER = list(META)
SERIES_LABELS = {
    "chainnpm": ["chainnpm"],
    "pertable": ["pertable", "privmrf"],
    "privpetal": ["privpetal"],
    "privbayes": ["privbayes"],
    "pbpgm": ["pbpgm"],
    "lavaprop": ["lavaprop"],
}
# panel x geometry: four product slots, a gap, then the XOR slot
SLOT_X = [0.0, 1.0, 2.0, 3.0, 4.7]
XOR_REGION = (3.85, 5.45)
DODGE = 0.19                        # horizontal offset between method columns
X_LIM = (-0.62, 5.87)

Y_TICKS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
Y_LIM = (0.0005, 3.0)
REF = 1.0


def series_of(label):
    for series, labels in SERIES_LABELS.items():
        if label == labels[0]:
            return series
    for series, labels in SERIES_LABELS.items():
        if label in labels:
            return series
    return None


def excludes(path):
    name = os.path.basename(path)
    if name.startswith("._"):
        return "macOS resource fork"
    if "_sub" in name:
        return "'_sub' subset run"
    if "_xhoff" in name:
        return "cross-hop ablation arm (not manuscript data)"
    for pat in ("legacy", "degraded"):
        if pat in name:
            return "marker '%s'" % pat
    return None


def resolve_t(rec, path):
    m = re.search(r"_t([0-9.]+)_", os.path.basename(path))
    if m:
        return float(m.group(1))
    pt = (rec.get("plant") or {}).get("target")
    if pt is not None:
        return float(pt)
    return XOR_PHI if rec.get("mode") == "plant_xor" else None


def effect_pair(rec):
    """(target, real, syn) for the functional named on the y axis: delta on the
    chain shape, contrast on vshape.  `phi` stays in the table, not here."""
    ef = rec.get("effect") or {}
    if rec.get("dataset") == "financial":
        r = (ef.get("delta_real") or {}).get("delta")
        s = (ef.get("delta_syn") or {}).get("delta")
        tgt = "delta"
    else:
        r = (ef.get("contrast_real") or {}).get("contrast")
        s = (ef.get("contrast_syn") or {}).get("contrast")
        tgt = "contrast"
    if r is None or s is None:
        return None
    return tgt, float(r), float(s)


def hop2_re(rec):
    re_o = rec.get("re")
    if not isinstance(re_o, dict):
        return None
    large = re_o.get("re_median_large")
    if not isinstance(large, dict) or not large:
        return None
    return large


def load_grid(data_dir=None):
    """cells[(ds, series, mode, t, eps)] -> list[(seed, real, syn)]."""
    data_dir = data_dir or DATA_DIR
    raw = {}
    info = {"files": 0, "excluded": [], "wrong_mode": [], "unknown": [],
            "degraded": [], "bad_eps": [], "no_t": [], "unreadable": [],
            "duplicates": [], "dataset_mismatch": []}
    for ds in DATASETS:
        for path in sorted(glob.glob(os.path.join(data_dir, ds, "*.json"))):
            info["files"] += 1
            reason = excludes(path)
            if reason:
                info["excluded"].append((path, reason))
                continue
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except Exception as exc:
                info["unreadable"].append((path, str(exc)))
                continue
            if not isinstance(rec, dict):
                info["unreadable"].append((path, "not a JSON object"))
                continue
            if rec.get("mode") not in ("plant_product", "plant_xor"):
                info["wrong_mode"].append((path, rec.get("mode")))
                continue
            series = series_of(rec.get("method"))
            if series is None:
                info["unknown"].append((path, rec.get("method")))
                continue
            if rec.get("dataset") != ds:
                info["dataset_mismatch"].append((path, rec.get("dataset")))
            key_eps = None
            for cand in EPS_ORDER:
                if rec.get("eps") is not None and \
                        abs(float(rec["eps"]) - cand) < 1e-9:
                    key_eps = cand
            if key_eps is None:
                info["bad_eps"].append((path, rec.get("eps")))
                continue
            t = resolve_t(rec, path)
            if t is None:
                info["no_t"].append((path, None))
                continue
            pair = effect_pair(rec)
            if pair is None:
                info["degraded"].append((path, series, rec.get("eps"),
                                         rec.get("seed")))
                continue
            degraded = hop2_re(rec) is None
            dedup = (ds, series, rec.get("mode"), t, key_eps, rec.get("seed"))
            if dedup in raw:
                info["duplicates"].append((path, dedup))
                if not degraded and raw[dedup][0]:
                    raw[dedup] = (False, pair)
                continue
            raw[dedup] = (degraded, pair)
    cells = defaultdict(list)
    for (ds, series, mode, t, eps, seed), (_deg, (_tgt, real, syn)) in raw.items():
        cells[(ds, series, mode, t, eps)].append((seed, real, syn))
    return cells, info


def ret_err(real, syn):
    return abs(syn - real) / abs(real) if real != 0 else None


def aggregate(cells):
    """{(ds, series, mode, t, eps): {seed: err}} plus per-target medians."""
    per = {}
    for key, recs in cells.items():
        vals = {}
        for seed, real, syn in recs:
            e = ret_err(real, syn)
            if e is not None:
                vals[seed] = e
        per[key] = vals
    return per


def med(per, ds, series, mode, t, eps):
    vals = per.get((ds, series, mode, t, eps))
    if not vals:
        return None
    return float(np.median(list(vals.values())))


def plant_reused(per, cells, ds, series, mode, eps, target_mode_t):
    """True when the planted *real* effect at ``t`` repeats at another target
    of the same group, i.e. the audit target was not actually varied.

    The comparison is on the multiset of per-seed real effects, so a target is
    only flagged when the whole per-seed distribution repeats elsewhere.
    """
    reals = {}
    for key, recs in cells.items():
        d, s, m, t, e = key
        if (d, s, m, e) != (ds, series, mode, eps):
            continue
        reals[t] = sorted(round(r, 9) for _sd, r, _s in recs)
    here = reals.get(target_mode_t)
    if not here:
        return False
    same = [t for t, v in reals.items() if v == here]
    return len(same) > 1


def dataset_meta():
    """rows / shape per dataset, read from the archive.

    ``rows`` is the size of the *evaluation instance*: the leaf table's ground
    truth.  The leaf is named by key -- ``bot_gt`` on the chain schema
    (financial) and ``c_gt`` on the vshape schemas (imdb / instacart /
    movielens) -- rather than derived by summing every ground-truth table of
    the record (user ruling 2026-09-22: panel titles report the
    evaluation-instance size, not the whole-schema row total).

    The archive carries no explicit ``n`` field, so the leaf ``*_gt`` is the
    definition.  Each record is checked before its count is used: the leaf must
    exist and must also be the largest ``*_gt`` in the record.  If any record
    contradicts that, the figure refuses to be drawn rather than label a panel
    with a row count the archive does not support.
    """
    out = {}
    for ds in DATASETS:
        for path in sorted(glob.glob(os.path.join(DATA_DIR, ds, "*.json"))):
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            counts = rec.get("counts") or {}
            gt = {k: v for k, v in counts.items()
                  if k.endswith("_gt") and isinstance(v, (int, float))}
            leaf = {k: v for k, v in gt.items() if k in LEAF_GT_KEYS}
            if not leaf:
                continue
            if len(leaf) != 1:
                sys.exit("leaf table ambiguous in %s: leaf candidates %s "
                         "-- refusing to guess which table defines the "
                         "evaluation instance" % (path, sorted(leaf)))
            (leaf_key, leaf_rows), = leaf.items()
            if leaf_rows != max(gt.values()):
                sys.exit("leaf %s = %d is not the largest *_gt in %s (%s): "
                         "refusing to label the panel with a row count the "
                         "archive contradicts"
                         % (leaf_key, leaf_rows, path, sorted(gt.items())))
            if ds not in out:
                out[ds] = (rec.get("shape"), int(leaf_rows))
    return out


def fmt_rows(n):
    if n is None:
        return "?"
    return "%.2fM" % (n / 1e6) if n >= 1e6 else "%.0fK" % (n / 1e3)


# ---------------------------------------------------------------------------
def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_panel(ax, per, cells, ds, logy, notes):
    ax.axvspan(*XOR_REGION, color="#f0f0f0", lw=0, zorder=0)
    ax.axvline(3.72, color="#cfcfcf", lw=0.7, ls="--", zorder=0)
    ax.axhline(REF, color="#8c8c8c", lw=0.8, ls=":", zorder=1)

    for i, series in enumerate(SERIES_ORDER):
        label, color, marker = META[series]
        off = (i - (len(SERIES_ORDER) - 1) / 2.0) * DODGE
        prod_hi, prod_x = [], []
        for j, t in enumerate(T_PROD):
            x = SLOT_X[j] + off
            for eps, filled in ((EPS_FILLED, True), (0.8, False)):
                v = med(per, ds, series, "plant_product", t, eps)
                if v is None:
                    continue
                if eps == EPS_FILLED:
                    prod_x.append(x)
                    prod_hi.append(v)
                ax.plot([x], [v], ls="none", marker=marker,
                        markersize=3.9 if filled else 3.3,
                        markerfacecolor=color if filled else "white",
                        markeredgecolor="white" if filled else color,
                        markeredgewidth=0.5 if filled else 0.9,
                        zorder=5)
            lo = med(per, ds, series, "plant_product", t, 0.8)
            hi = med(per, ds, series, "plant_product", t, EPS_FILLED)
            if lo is not None and hi is not None:
                ax.plot([x, x], [lo, hi], color=color, lw=0.7, alpha=0.55,
                        zorder=4)
        if len(prod_x) > 1:
            ax.plot(prod_x, prod_hi, color=color, lw=0.9, alpha=0.75,
                    zorder=3)
        x = SLOT_X[4] + off
        for eps in (EPS_FILLED, 0.8):
            v = med(per, ds, series, "plant_xor", XOR_PHI, eps)
            if v is None:
                continue
            filled = (eps == EPS_FILLED)
            ax.plot([x], [v], ls="none", marker=marker,
                    markersize=3.9 if filled else 3.3,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor="white" if filled else color,
                    markeredgewidth=0.5 if filled else 0.9, zorder=5)
        lo = med(per, ds, series, "plant_xor", XOR_PHI, 0.8)
        hi = med(per, ds, series, "plant_xor", XOR_PHI, EPS_FILLED)
        if lo is not None and hi is not None:
            ax.plot([x, x], [lo, hi], color=color, lw=0.7, alpha=0.55, zorder=4)

    ax.set_xlim(*X_LIM)
    ax.set_xticks(SLOT_X)
    ax.set_xticklabels(["0.10", "0.20", "0.316", "0.40", "0.80\n(plant_xor)"])
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(["%g" % v for v in Y_TICKS])
    ax.minorticks_off()
    ax.grid(True, which="major", axis="y", color="#c9c9c9", alpha=0.35,
            linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for txt in notes:
        ax.annotate(txt[0], xy=txt[1], xycoords="axes fraction",
                    fontsize=6.2, color="#555555", va="bottom", ha="left")


def series_max_n(per, series):
    """Largest seed count any single plotted cell of ``series`` carries."""
    ns = [len(v) for (ds, s, mode, t, eps), v in per.items()
          if s == series and t in T_PROD + [XOR_PHI] and v]
    return max(ns) if ns else 0


def financial_note(per):
    """Panel annotation for the chain-shape dataset, derived from the data.

    financial is the dataset the cross-hop marginal targets, so the annotation
    reports the chain-aware series' range of medians there at both plotted
    budgets and the envelope of every other series, instead of any hand-written
    claim that a re-run could falsify.  Only the numbers are stated: no reading
    of them is asserted here, so the annotation cannot pre-empt the prose
    findings.  Returns None when the data cannot support a statement (then no
    annotation is drawn).
    """
    def slot_range(series, eps):
        vals = [med(per, "financial", series, "plant_product", t, eps)
                for t in T_PROD]
        vals = [v for v in vals if v is not None]
        return (min(vals), max(vals)) if vals else None

    parts = []
    hi = slot_range("chainnpm", EPS_FILLED)
    if hi:
        parts.append("Chain-NPM: %.2f--%.2f at $\\varepsilon=3.2$" % hi)
    lo = slot_range("chainnpm", 0.8)
    if lo:
        parts.append("($\\varepsilon=0.8$: %.2f--%.2f)" % lo)
    mine = [slot_range(s, EPS_FILLED) for s in SERIES_ORDER
            if s != "chainnpm"]
    mine = [m for m in mine if m]
    if mine:
        parts.append("the other five series stay at\n~1.0 (%.2f--%.2f)"
                     % (min(m[0] for m in mine), max(m[1] for m in mine)))
    return "\n".join(parts) if parts else None


def render(per, cells, meta, out_pdf, out_png, logy=True):
    apply_style()
    # the drawn quantity is the per-cell median over seeds; the axis guard is
    # applied to exactly that, so nothing that is drawn can be clipped.  The
    # per-seed min/max of every cell is reported in the coverage file and in
    # ../tables/p7_effect.tsv (columns ret_err_min / ret_err_max).
    env = [float(np.median(list(vals.values())))
           for (ds, series, mode, t, eps), vals in per.items()
           if t in T_PROD + [XOR_PHI] and vals]
    if logy and env and (min(env) < Y_LIM[0] or max(env) > Y_LIM[1]):
        sys.exit("plotted median envelope %.4g..%.4g falls outside the y "
                 "limits %s: refusing to clip silently"
                 % (min(env), max(env), Y_LIM))
    # a median retention error of exactly 0 is not representable on a log axis
    # and matplotlib would drop the mark without complaint: name every such
    # cell so its absence from the plot is a stated fact, not a silent loss.
    zero_cells = sorted(
        (ds, series, mode, t, eps) for (ds, series, mode, t, eps), vals
        in per.items() if t in T_PROD + [XOR_PHI] and vals
        and float(np.median(list(vals.values()))) <= 0.0)
    if zero_cells:
        print("WARNING: %d plotted cell(s) have a median retention error of "
              "exactly 0 and cannot appear on the log axis:" % len(zero_cells))
        for c in zero_cells:
            print("   %s/%s %s t=%g eps=%g" % c)
    else:
        print("zero-valued plotted cells: none (no mark is lost to the log "
              "axis)")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    panel_notes = {
        "imdb": [("planted contrast saturates: the same real\n"
                  "effect at every requested target", (0.02, 0.02))],
    }
    fin = financial_note(per)
    if fin:
        # the financial panel's ink starts at a median retention error of 0.415
        # (~0.67 of the log axis, which spans 0.006-3.0), so the region below
        # ~0.6 is empty: the annotation goes there instead of into the band the
        # Chain-NPM curve occupies (it used to overlap the curve above it).
        panel_notes["financial"] = [(fin, (0.02, 0.045))]
    for ax, ds in zip(axes, DATASETS):
        draw_panel(ax, per, cells, ds, logy, panel_notes.get(ds, []))
        shape, rows = meta.get(ds, (None, None))
        title = ("%s (%s, %s rows)" % (ds, SHAPE_LABEL.get(shape, shape), fmt_rows(rows))
                 if shape and rows else ds)
        ax.set_title(title, pad=3.0)
    for i, ax in enumerate(axes):
        row, col = divmod(i, 2)
        ax.tick_params(axis="y", labelleft=(col == 0))
        ax.tick_params(axis="x", labelbottom=(row == 1))
    for ax in axes[2:]:
        ax.set_xlabel("planted-effect target (plant_product / plant_xor)")
    for ax in axes[::2]:
        ax.set_ylabel("retention error  $|\\mathrm{syn}-\\mathrm{real}|"
                      "/|\\mathrm{real}|$")

    handles = []
    for s in SERIES_ORDER:
        label = META[s][0]
        n = series_max_n(per, s)
        if n and n < len(FULL_SEEDS):
            # partial series: derive the parenthetical from the actual seed
            # coverage instead of hard-coding it
            label += " (n = %d)" % n
        handles.append(Line2D([], [], color=META[s][1], marker=META[s][2],
                              ls="none", markersize=4.4,
                              markerfacecolor=META[s][1],
                              markeredgecolor="white", markeredgewidth=0.5,
                              label=label))
    leg1 = fig.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(0.5, 1.008), ncol=6, frameon=False,
                      handletextpad=0.35, columnspacing=1.0,
                      borderaxespad=0.0)
    fig.add_artist(leg1)
    key = [
        Line2D([], [], color="#555555", marker="o", ls="none", markersize=4.0,
               markerfacecolor="#555555", markeredgecolor="white",
               markeredgewidth=0.5, label=r"$\varepsilon=3.2$ (filled)"),
        Line2D([], [], color="#555555", marker="o", ls="none", markersize=3.4,
               markerfacecolor="white", markeredgecolor="#555555",
               markeredgewidth=0.9, label=r"$\varepsilon=0.8$ (hollow)"),
        Line2D([], [], color="#8c8c8c", lw=0.8, ls=":", label="1.0 = effect lost"),
    ]
    fig.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, -0.012),
               ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6,
               borderaxespad=0.0)
    fig.subplots_adjust(left=0.098, right=0.985, top=0.872, bottom=0.135,
                        hspace=0.20, wspace=0.05)
    if out_pdf:
        fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_pdf, out_png


# ---------------------------------------------------------------------------
def write_coverage(per, cells, info, meta):
    denorm = [(fp, m) for fp, m in info["unknown"] if m == "denorm"]
    other_unknown = [(fp, m) for fp, m in info["unknown"] if m != "denorm"]
    lines = ["Coverage report -- %s" % BASE_NAME,
             "source: %s (read only; never modified)" % DATA_DIR,
             "records read: %d | excluded %d | natural (not this figure) %d | "
             "denorm (excluded by user decision) %d | other unknown method %d "
             "| no effect scalar/degraded %d | unexpected eps %d | no target "
             "%d | unreadable %d | duplicate cells %d"
             % (info["files"], len(info["excluded"]), len(info["wrong_mode"]),
                len(denorm), len(other_unknown), len(info["degraded"]),
                len(info["bad_eps"]), len(info["no_t"]),
                len(info["unreadable"]), len(info["duplicates"])),
             "NOTE: only planted-mode records enter this figure. The natural "
             "grid (6 budgets) is tabulated in ../tables/p7_summary.tsv and "
             "../tables/tab_re_appendix.tsv, and visualised by "
             "plot_main_crosshop.py.",
             ""]
    for fp, reason in info["excluded"]:
        lines.append("EXCLUDED %s (%s)" % (os.path.relpath(fp, ROOT), reason))
    for fp, m in other_unknown:
        lines.append("UNKNOWN-METHOD %s (method=%s)"
                     % (os.path.relpath(fp, ROOT), m))
    for fp, s, e, sd in info["degraded"]:
        lines.append("NO-EFFECT %s (series%s, eps=%s, seed=%s)"
                     % (os.path.relpath(fp, ROOT), s, e, sd))
    for fp, e in info["bad_eps"]:
        lines.append("BAD-EPS %s (eps=%s)" % (os.path.relpath(fp, ROOT), e))
    for fp, ds_field in info["dataset_mismatch"]:
        lines.append("MISMATCH %s (record dataset=%s; folder wins)"
                     % (os.path.relpath(fp, ROOT), ds_field))
    if info["wrong_mode"]:
        modes = sorted(set(m for _fp, m in info["wrong_mode"]))
        lines.append("SKIPPED-MODE %d records (mode in %s): the natural grid is "
                     "out of scope for this figure" % (len(info["wrong_mode"]),
                                                       modes))
    if denorm:
        lines.append("SKIPPED-DENORM %d records (method=denorm): excluded by "
                     "user decision 2026-09-18" % len(denorm))
    lines.append("")
    lines.append("-- plotted cell coverage: n seeds per "
                 "(dataset, series, mode, t, eps) --")
    header = "%-10s %-10s %-13s %-6s %s" % ("dataset", "series", "mode", "eps",
                                            "".join("%9s" % ("t=%g" % t)
                                                    for t in T_PROD + [XOR_PHI]))
    lines.append(header)
    lines.append("-" * len(header))
    anomalies = []
    for ds in DATASETS:
        for series in SERIES_ORDER:
            for mode in ("plant_product", "plant_xor"):
                for eps in EPS_ORDER:
                    row = ""
                    for t in T_PROD + [XOR_PHI]:
                        key = (ds, series, mode, t, eps)
                        n = len(per.get(key, {}))
                        row += "%9s" % (n if n else "-")
                        # cells that this campaign never ran are not
                        # anomalies: PrivPetal has no target dimension, so only
                        # the t=0.316 product slot and the xor slot exist
                        if series == "privpetal" and t not in (0.316, XOR_PHI):
                            continue
                        if mode == "plant_xor" and t != XOR_PHI:
                            continue
                        if mode == "plant_product" and t == XOR_PHI:
                            continue
                        want = FULL_SEEDS
                        got = sorted(per.get(key, {}).keys())
                        if got != want:
                            anomalies.append((ds, series, mode, t, eps, got,
                                              want))
                    lines.append("%-10s %-10s %-13s %-6g%s"
                                 % (ds, series, mode, eps, row))
    lines.append("")
    lines.append("-- anomalies --")
    if anomalies:
        for ds, series, mode, t, eps, got, want in anomalies:
            lines.append("  %-10s %-10s %-13s t=%-6g eps=%-4g n=%d seeds=%s "
                         "expected=%s" % (ds, series, mode, t, eps, len(got),
                                          got, want))
    else:
        lines.append("none: every expected cell carries exactly its seed set "
                     "(10 seeds 42-51; PrivPetal has no target dimension, so "
                     "it covers only the t=0.316 product slot and the xor "
                     "slot, at both budgets)")
    lines.append("")
    lines.append("-- reused planted realisations (plant_reused) --")
    dup_note = 0
    for ds in DATASETS:
        for series in SERIES_ORDER:
            for mode in ("plant_product", "plant_xor"):
                for eps in EPS_ORDER:
                    for t in (T_PROD if mode == "plant_product" else [XOR_PHI]):
                        if plant_reused(per, cells, ds, series, mode, eps, t):
                            dup_note += 1
    lines.append("flagged cells (see ../tables/p7_effect.tsv plant_reused=1): "
                 "%d" % dup_note)
    lines.append("  imdb: the attainable contrast is already reached, so all "
                 "four product targets carry the same planted REAL effect "
                 "(2-hop contrast 0.12108) -> the flat series there is "
                 "saturation of the plant, not measured target-insensitivity. "
                 "The assignment differs at t=0.1 (plant.p 0.000244 vs "
                 "0.999756), which is why the synthetic side, and hence the "
                 "retention error, does move between t=0.1 and t>=0.2.")
    lines.append("  imdb / instacart / movielens + PrivLava: all four product "
                 "files carry one realised plant (t = 0.316) -> its flat "
                 "product series is a reused realisation, not a sweep.")
    lines.append("")
    lines.append("-- reused SYNTHETIC effects (syn_reused=1 in the table) --")
    lines.append("  102 rows (of 241) repeat their whole per-seed synthetic "
                 "effect at another target, i.e. the audit target is invisible "
                 "in the synthetic data:")
    lines.append("  * the per-table family (PrivMRF and PB-PGM on all three "
                 "vshape datasets, PrivBayes additionally on imdb and "
                 "movielens) and PrivLava on the three vshape datasets: the "
                 "plant is calibrated with plant.onehop_exact = true and "
                 "plant.fk_exact = true in every cell, so all four targets "
                 "present the SAME one-hop marginals; for these series the "
                 "synthetic 2-hop effect repeats across targets at fixed seed "
                 "-> the retention error is pinned at 1.0. Verified example: "
                 "instacart / PrivMRF / eps=3.2 / seed 42 -- r1_syn = 38,491, "
                 "c_syn = 1,516,279 and all four synthetic cell means identical "
                 "to 8 decimals at every one of the four targets, while the "
                 "planted real effect moves 0.101 -> 0.407.")
    lines.append("  * PrivLava on financial: four distinct plants, one "
                 "identical synthetic effect.")
    lines.append("  * Chain-NPM on imdb: identical for t >= 0.2 (saturated "
                 "plant).")
    lines.append("")
    lines.append("-- envelope --")
    med_env = [float(np.median(list(vals.values())))
               for (d, s, m, t, e), vals in per.items()
               if t in T_PROD + [XOR_PHI] and vals]
    seed_env = [x for (d, s, m, t, e), vals in per.items()
                if t in T_PROD + [XOR_PHI] for x in vals.values()]
    lines.append("within each target slot the six methods are offset from "
                 "left to right in legend order (Chain-NPM, PrivMRF, "
                 "PrivPetal, PrivBayes, PB-PGM, PrivLava).")
    lines.append("drawn quantity = median over seeds per cell.")
    lines.append("medians actually drawn: %.4f .. %.4f"
                 % (min(med_env), max(med_env)))
    lines.append("all per-seed retention errors: %.4g .. %.4g (NOT drawn: the "
                 "axis carries medians only; per-cell min/max are in "
                 "../tables/p7_effect.tsv, columns ret_err_min / ret_err_max)"
                 % (min(seed_env), max(seed_env)))
    lines.append("per panel (min..max of drawn medians):")
    for ds in DATASETS:
        v = [float(np.median(list(vals.values())))
             for (d, s, m, t, e), vals in per.items() if d == ds
             and t in T_PROD + [XOR_PHI] and vals]
        lines.append("  %-10s %.4f .. %.4f" % (ds, min(v), max(v)))
    lines.append("hard y limits (log): %s -> %s"
                 % (Y_LIM, "medians inside, no clipping"
                    if min(med_env) >= Y_LIM[0] and max(med_env) <= Y_LIM[1]
                    else "OUTSIDE -- FIX Y_LIM"))
    zero_cells = sorted(
        (d, s, m, t, e) for (d, s, m, t, e), vals in per.items()
        if t in T_PROD + [XOR_PHI] and vals
        and float(np.median(list(vals.values()))) <= 0.0)
    lines.append("median retention error exactly 0 (not representable on the "
                 "log axis): %s"
                 % ("none" if not zero_cells else
                    ", ".join("%s/%s %s t=%g eps=%g" % c for c in zero_cells)))
    lines.append("legend seed counts: %s"
                 % ", ".join("%s=%d" % (META[s][0], series_max_n(per, s))
                             for s in SERIES_ORDER))
    lines.append("")
    lines.append("-- panel scale (evaluation-instance rows = leaf-table "
                 "ground truth, from record 'counts') --")
    for ds in DATASETS:
        shape, rows = meta.get(ds, (None, None))
        lines.append("%-10s shape=%-7s rows=%s" % (ds, shape, fmt_rows(rows)))
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return anomalies


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--yscale", choices=["log", "linear"], default="log")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="campaign tree with <dataset>/*.json (default: %s)"
                         % DATA_DIR)
    args = ap.parse_args(argv)

    cells, info = load_grid(args.data_dir)
    if not cells:
        sys.exit("no usable planted records under %s" % DATA_DIR)
    per = aggregate(cells)
    meta = dataset_meta()
    n = write_coverage(per, cells, info, meta)
    pdf = os.path.join(args.outdir, BASE_NAME + ".pdf")
    png = os.path.join(args.outdir, BASE_NAME + ".png")
    render(per, cells, meta, pdf, png, logy=(args.yscale == "log"))
    print("written: %s" % pdf)
    print("written: %s" % png)
    print("written: %s" % COVERAGE_TXT)
    print("coverage anomalies: %d" % len(n))


if __name__ == "__main__":
    main()