#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新 Fig 7：自然模式下四个真实库的 2-hop 相对误差 vs 隐私预算 ε。

版式（与 PrivPetal 实验节同族）
  跨双栏figure*，1 行 × 4 列 panel = {financial, imdb, instacart, movielens}；
  每 panel 六条方法曲线；x = 六档 ε ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2}；
  y = 大计数 2-hop 合取查询的中位相对误差（10 seeds），带 10 seeds 的 min–max 带。
  纵轴为断轴（zigzag）：下段 [0, 1.0]，上段 [1.0, 4.5]，因为绝大多数中位数落在
  1.0 以下而 chainnpm/financial 在低预算下达到 4.34。

面板标题里的规模是**评估实例**的规模：叶子表自己的 `*_gt`（chain 的 bot /
vshape 的 c）。归档记录没有 `n` 字段，各表的行数只能来自 `counts` 的 `*_gt`，
所以评估实例的行数就是叶子表 `_gt`；此前用的「全部 `*_gt` 之和」把中间维表也
算了进去（imdb 1.97M vs 叶子 1.22M），不是评估实例。

数据源：results/tables/p7_common.py（优先现场战役树 tmp/results，退化到
results/by_dataset），与 p7_summary.tsv / p7_stats.json 走完全同一条代码路径。

自检（不通过则不出图）
  每个 (dataset, series, eps) 由本脚本重算的中位数，必须与 p7_summary.tsv 的
  median 逐格一致（1e-6）；任何被绘制的值（中位数或 min/max 带端点）都必须落在
  两段轴的范围之内，否则拒绝绘制而不是静默裁切。
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))          # results/figures
RESULTS_DIR = os.path.dirname(HERE)                        # results
ROOT = os.path.dirname(RESULTS_DIR)                        # project root
sys.path.insert(0, os.path.join(RESULTS_DIR, "tables"))
from p7_common import load_rows, hop2_re, DATASETS, EPS_ORDER  # noqa: E402
import szquota_common  # noqa: E402

BASE_NAME = os.path.join(HERE, "fig_dataset_sweep")
SUMMARY_TSV = os.path.join(RESULTS_DIR, "tables", "p7_summary.tsv")
COVERAGE_TXT = os.path.join(HERE, "plot_dataset_sweep_coverage.txt")

# 与 plot_main_effect.py / plot_main_crosshop.py 完全一致的系列约定，
# 使正文各图读起来是一个图族
META = {
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o"),
    "privmrf":   ("PrivMRF",   "#be123c", "v"),
    "privpetal": ("PrivPetal", "#b45309", "s"),
    "privbayes": ("PrivBayes", "#166534", "^"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D"),
}
SERIES_ORDER = list(META)

# 断轴几何。szquota 后期数据（p7_summary_szquota）绝大多数中位数落在 1.0 以下，
# 仅 privpetal/instacart 低预算（1.207/1.032）留在上段，故上段收到 [1.0, 2.0]；
# released 期为 9.5（chainnpm/financial 低预算 9.03 上界已不存在）。
Y_LO = (0.0, 1.0)          # 下段：绝大多数中位数所在
Y_HI = (1.0, 2.0)          # 上段：仅剩的少数 >1 格（privpetal instacart 低预算）
Y_TICKS_LO = [0.0, 0.25, 0.5, 0.75, 1.0]
Y_TICKS_HI = [1.25, 1.5, 1.75, 2.0]
EPS_LABELS = ["0.1", "0.2", "0.4", "0.8", "1.6", "3.2"]

# 叶子表（评估实例所在表）的行数键：chain 的 bot、vshape 的 c。
LEAF_GT_KEYS = ("bot_gt", "c_gt")


def fmt_rows(n):
    if n is None:
        return "?"
    return "%.2fM" % (n / 1e6) if n >= 1e6 else "%.0fK" % (n / 1e3)


def leaf_rows(counts):
    """叶子表 `_gt` = 评估实例规模；返回 (key, value, why)。

    `key` 为 None 表示无法确定（`why` 说明原因），调用方据此拒绝出图。归档
    记录里 `*_gt` 是各 ground-truth 表的行数、没有 `n` 字段，所以评估实例的
    规模只能取叶子表自己的 `_gt`。只认 `bot_gt` / `c_gt` 两个键名：一个都不
    出现或同时出现都拒绝出图，不猜。一致性检查：叶子表必须同时是记录里
    `*_gt` 的最大者，否则说明叶子键与本库的 shape 对不上，同样拒绝。
    """
    gt = {k: float(x) for k, x in (counts or {}).items()
          if k.endswith("_gt") and isinstance(x, (int, float))}
    hits = [k for k in LEAF_GT_KEYS if k in gt]
    if len(hits) != 1:
        return None, None, ("expected exactly one leaf key out of %s in "
                            "'counts', found %s (keys present: %s)"
                            % (list(LEAF_GT_KEYS), hits, sorted(gt)))
    lk = hits[0]
    mx = max(gt, key=lambda k: gt[k])
    if gt[lk] < gt[mx]:
        return None, None, ("leaf %s (=%d) is not the largest _gt in the "
                            "record; the largest is %s (=%d) -- refusing to "
                            "guess which table holds the evaluation instance"
                            % (lk, gt[lk], mx, gt[mx]))
    return lk, gt[lk], ""


def load(arm):
    """(dataset, series, eps) -> {seed: re}；各库的 shape 与评估实例行数。

    返回 (per, meta, scale_problems)。meta[ds] = (shape, leaf_gt, leaf_key)，
    其中 leaf_gt 是叶子表自己的 `_gt`（评估实例规模），由 leaf_rows() 判定；
    任何无法判定或跨记录不一致的情况都进 scale_problems，由 main() 拒绝出图。
    arm='szquota' 时 chainnpm 行换成 szquota 臂网格（szquota_common）。
    """
    rows = (szquota_common.szquota_records() if arm == 'szquota'
            else load_rows())
    per = defaultdict(dict)
    meta = {}
    scale_problems = []
    leaf_seen = defaultdict(set)
    leaf_bad = {}
    for d in rows:
        ds = d["_dataset"]
        if d.get("mode") != "natural":
            continue
        v = hop2_re(d)
        if v is None:
            continue
        key = (d["_dataset"], d["method"], d.get("eps"))
        per[key][d.get("seed")] = float(v)
        lk, lv, why = leaf_rows(d.get("counts"))
        if lk is None:
            leaf_bad.setdefault(ds, why)   # 每个库只报一次
            continue
        leaf_seen[ds].add(lv)
        if ds not in meta:
            meta[ds] = (d.get("shape"), int(lv), lk)
    for ds, why in sorted(leaf_bad.items()):
        scale_problems.append("%s: %s" % (ds, why))
    for ds, vals in sorted(leaf_seen.items()):
        if len(vals) > 1:
            scale_problems.append(
                "leaf `_gt` disagrees across the records of %s: %s"
                % (ds, sorted(int(x) for x in vals)))
    for ds in DATASETS:
        if ds not in meta:
            scale_problems.append("no natural record with a leaf `_gt` for %s"
                                  % ds)
    return per, meta, scale_problems


def read_summary(arm):
    """p7_summary.tsv -> {(ds, series, eps): median}；szquota 臂对拍 szquota summary"""
    path = (os.path.join(RESULTS_DIR, 'tables', 'p7_summary_szquota.tsv')
            if arm == 'szquota' else SUMMARY_TSV)
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split("\t")
    out = {}
    for r in csv.DictReader(lines, delimiter="\t", fieldnames=hdr):
        if r["dataset"] == "dataset":
            continue
        out[(r["dataset"], r["series"], float(r["eps"]))] = float(r["median"])
    return out


def self_check(per, summary):
    """中位数逐格对拍；返回 (problems, notes)。"""
    problems = []
    checked = 0
    for ds in DATASETS:
        for s in SERIES_ORDER:
            for eps in EPS_ORDER:
                vals = per.get((ds, s, eps))
                if not vals:
                    problems.append("missing cell %s/%s/eps=%s" % (ds, s, eps))
                    continue
                med = float(np.median(list(vals.values())))
                ref = summary.get((ds, s, eps))
                if ref is None:
                    problems.append("no p7_summary row %s/%s/eps=%s"
                                    % (ds, s, eps))
                    continue
                if abs(med - ref) > 1e-6:
                    problems.append("median mismatch %s/%s/eps=%s: %.9f vs %.9f"
                                    % (ds, s, eps, med, ref))
                checked += 1
                if len(vals) != 10:
                    problems.append("seed count %d != 10 at %s/%s/eps=%s"
                                    % (len(vals), ds, s, eps))
    return problems, checked


def axis_range_check(per):
    """所有将被绘制的值必须落在两段轴的并集之内。"""
    lo, hi = Y_LO[0], Y_HI[1]
    bad = []
    for key, vals in per.items():
        arr = list(vals.values())
        for v in [np.median(arr), min(arr), max(arr)]:
            if not (lo <= v <= hi):
                bad.append((key, float(v)))
    return bad


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def break_marks(ax_hi, ax_lo, d=0.013):
    kw = dict(color="k", clip_on=False, lw=0.75, zorder=5)
    for x0, x1 in ((0.0, 2 * d), (1.0 - 2 * d, 1.0)):
        ax_hi.plot((x0, x0 + d), (-d, +d), transform=ax_hi.transAxes, **kw)
        ax_lo.plot((x0, x0 + d), (1 - d, 1 + d), transform=ax_lo.transAxes,
                   **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--arm', choices=['released', 'szquota'], default='released',
                    help="Chain-NPM series source: released (archive) or the "
                         "szquota arm (campaign grids); default released")
    a = ap.parse_args()
    per, meta, scale_problems = load(a.arm)
    summary = read_summary(a.arm)

    problems, checked = self_check(per, summary)
    problems = scale_problems + problems
    out_of_range = axis_range_check(per)
    notes = []
    notes.append("panel title basis: leaf-table `_gt` = evaluation-instance "
                 "rows (chain: bot_gt, vshape: c_gt), NOT the sum of every "
                 "`*_gt` in the record")
    notes.append("cells cross-checked against %s: %d "
                 "(expected %d = 4 datasets x 6 series x 6 eps)"
                 % ('p7_summary_szquota.tsv' if a.arm == 'szquota'
                    else 'p7_summary.tsv',
                    checked, len(DATASETS) * len(SERIES_ORDER) * len(EPS_ORDER)))
    if out_of_range:
        notes.append("OUT OF AXIS RANGE: %s" % out_of_range[:10])
    if problems:
        notes.append("PROBLEMS:")
        notes.extend("  " + p for p in problems)
    else:
        notes.append("none: every plotted median matches the summary table "
                     "and every value lies inside one of the two axis segments")
    if problems or out_of_range:
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        raise SystemExit("refusing to draw: see %s" % COVERAGE_TXT)

    apply_style()
    fig = plt.figure(figsize=(7.05, 2.45))
    gs = GridSpec(2, len(DATASETS), figure=fig, height_ratios=[1.5, 2.2],
                  hspace=0.13, wspace=0.20,
                  left=0.072, right=0.995, top=0.795, bottom=0.155)
    x = np.arange(len(EPS_ORDER), dtype=float)

    for j, ds in enumerate(DATASETS):
        ax_hi = fig.add_subplot(gs[0, j])
        ax_lo = fig.add_subplot(gs[1, j])
        for ax, ylim, ticks in ((ax_hi, Y_HI, Y_TICKS_HI),
                                (ax_lo, Y_LO, Y_TICKS_LO)):
            ax.set_xlim(-0.42, len(EPS_ORDER) - 0.58)
            ax.set_ylim(*ylim)
            ax.set_yticks(ticks)
            ax.tick_params(axis="y", direction="out")
            ax.grid(axis="y", color="#d9d9d9", lw=0.45, zorder=0)
            ax.set_axisbelow(True)
        ax_hi.spines["bottom"].set_visible(False)
        ax_hi.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_lo.spines["top"].set_visible(False)
        break_marks(ax_hi, ax_lo)

        for s in SERIES_ORDER:
            label, color, marker = META[s]
            med, lo, hi = [], [], []
            for eps in EPS_ORDER:
                vals = list((per.get((ds, s, eps)) or {}).values())
                if not vals:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                med.append(float(np.median(vals)))
                lo.append(min(vals)); hi.append(max(vals))
            med = np.array(med); lo = np.array(lo); hi = np.array(hi)
            for ax in (ax_hi, ax_lo):
                ax.fill_between(x, lo, hi, color=color, alpha=0.15,
                                linewidth=0, zorder=1)
                ax.plot(x, med, color=color, marker=marker, ls="-", lw=1.15,
                        markersize=3.2, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.4,
                        zorder=3, label=label)

        shape, rows, _leaf_key = meta.get(ds, (None, None, None))
        title = ("%s (%s rows)" % (ds, fmt_rows(rows))
                 if rows else ds)
        ax_hi.set_title(title, pad=2.6)
        if j == 0:
            ax_lo.set_yticks(Y_TICKS_LO)
        else:
            ax_lo.tick_params(axis="y", labelleft=False)
            ax_hi.tick_params(axis="y", labelleft=False)
        if j == 0:
            pass
        ax_lo.set_xticks(x)
        ax_lo.set_xticklabels(EPS_LABELS)

    fig.text(0.013, 0.50, "2-hop relative error",
             rotation=90, va="center", ha="left", fontsize=8.8)
    fig.text(0.545, 0.072, r"privacy budget $\varepsilon$",
             ha="center", va="top", fontsize=9.0)

    handles = [Line2D([], [], color=META[s][1], marker=META[s][2], ls="-",
                      lw=1.15, markersize=3.4, markeredgecolor="white",
                      markeredgewidth=0.4, label=META[s][0])
               for s in SERIES_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0.0)

    out_pdf = BASE_NAME + ".pdf"
    out_png = BASE_NAME + ".png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
        fh.write("\n-- panel scale (leaf-table `_gt` = evaluation-instance "
                 "rows; NOT the sum of all ground-truth tables) --\n")
        for ds in DATASETS:
            shape, rows, leaf = meta.get(ds, (None, None, None))
            fh.write("%-10s shape=%-7s leaf=%-7s rows=%s\n"
                     % (ds, shape, leaf if leaf else "?", fmt_rows(rows)))
        fh.write("\n-- plotted medians --\n")
        fh.write("%-10s %-10s %s\n" % ("dataset", "series",
                                       " ".join("%7s" % e for e in EPS_LABELS)))
        for ds in DATASETS:
            for s in SERIES_ORDER:
                cells = []
                for eps in EPS_ORDER:
                    vals = per.get((ds, s, eps))
                    cells.append("%7.3f" % np.median(list(vals.values()))
                                 if vals else "      -")
                fh.write("%-10s %-10s %s\n" % (ds, s, " ".join(cells)))
    print("written:", out_pdf)
    print("written:", out_png)
    print("written:", COVERAGE_TXT)
    print("# " + notes[0])
    print("# " + notes[-1])


if __name__ == "__main__":
    main()