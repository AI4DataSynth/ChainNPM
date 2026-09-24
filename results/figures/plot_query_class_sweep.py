#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新 Fig 8：按查询类别分层的自然模式精度（跨四个真实库取中位数）。

版式
  跨双栏figure*，1 行 × 4 列 panel = 四个查询类别
    (a) 2-hop 合取   chain: 2hop_topbot               vshape: cross_r1r2
    (b) 1-hop        chain: 1hop_topmid, 1hop_midbot  vshape: 1hop_r1c, 1hop_r2c
    (c) 0-hop 边际   chain: 0hop_top, 0hop_bot        vshape: 0hop_r1, 0hop_r2, 0hop_c
    (d) 尺寸分布 TV  chain: bot_per_mid, mid_per_top  vshape: size_tv 标量
  每 panel 六条方法曲线，x = 六档 ε ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2}。

聚合口径（三级，逐级都写进覆盖报告，避免读者误读）
  1. 同 (dataset, series, eps, seed, 类别) 内：对该库该 shape 下该类别的**全部指标键取算术均值**
     （键数不同：chain 的 0-hop 有 2 键、vshape 有 3 键；TV 在 chain 是 2 键、vshape 是标量）；
  2. 同 (dataset, series, eps, 类别) 内：对 10 个 seeds 取中位数；
  3. 同一 (series, eps, 类别)：对四个真实库取中位数 -> 这就是画出来的点。

诚实性约束
  * 任何 (series, eps, 类别, dataset) 只要缺格，就在覆盖报告里逐条列出；缺格不参与第 3 级中位数，
    但会统计「实际参与库数」，不把它当作 4 库齐备。
  * y 轴按 panel 各自量纲设置，不做跨 panel 的视觉比较；每 panel 的轴标签写明所画量。
  * p7_summary.tsv 只覆盖 2-hop，因此本图对 (a) 做一次对拍自检；(b)(c)(d) 无同源汇总表可比，
    只能自检内部一致性（键齐全性、seed 数）。
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RESULTS_DIR, "tables"))
from p7_common import load_rows, DATASETS, EPS_ORDER  # noqa: E402
import szquota_common  # noqa: E402

BASE_NAME = os.path.join(HERE, "fig_query_class_sweep")
COVERAGE_TXT = os.path.join(HERE, "plot_query_class_sweep_coverage.txt")

META = {
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o"),
    "privmrf":   ("PrivMRF",   "#be123c", "v"),
    "privpetal": ("PrivPetal", "#b45309", "s"),
    "privbayes": ("PrivBayes", "#166534", "^"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D"),
}
SERIES_ORDER = list(META)

CHAIN = "chain"
CLASS_KEYS = {
    # 类别 -> {shape: [re.re_median_large 的键]}
    "2hop":       {CHAIN: ["2hop_topbot"], "vshape": ["cross_r1r2"]},
    "1hop":       {CHAIN: ["1hop_topmid", "1hop_midbot"],
                   "vshape": ["1hop_r1c", "1hop_r2c"]},
    "0hop":       {CHAIN: ["0hop_top", "0hop_bot"],
                   "vshape": ["0hop_r1", "0hop_r2", "0hop_c"]},
}
# 尺寸分布 TV：走 size_tv，键名与上不同，单独处理
TV_KEYS = {CHAIN: ["bot_per_mid", "mid_per_top"], "vshape": []}
CLASS_ORDER = ["2hop", "1hop", "0hop", "tv"]
CLASS_TITLE = {"2hop": "2-hop conjunctive", "1hop": "1-hop predicate",
               "0hop": "0-hop marginal", "tv": "size-distribution TV"}
CLASS_YLABEL = {"2hop": "relative error", "1hop": "relative error",
                "0hop": "relative error", "tv": "total variation"}
EPS_LABELS = ["0.1", "0.2", "0.4", "0.8", "1.6", "3.2"]


def shape_of(d):
    return "chain" if (d.get("shape") == "chain" or
                       d["_dataset"] == "financial") else "vshape"


def class_value(d, klass):
    """该记录在给定类别上的标量：可用键取算术均值；无键返回 None。"""
    if klass == "tv":
        tv = d.get("size_tv")
        if isinstance(tv, dict):
            vals = [v for k, v in tv.items() if k in TV_KEYS[CHAIN]
                    and isinstance(v, (int, float))]
        elif isinstance(tv, (int, float)):
            vals = [float(tv)]
        else:
            vals = []
    else:
        m = (d.get("re") or {}).get("re_median_large")
        keys = CLASS_KEYS[klass].get(shape_of(d), [])
        vals = [m[k] for k in keys if isinstance(m, dict) and
                isinstance(m.get(k), (int, float))]
    if not vals:
        return None
    return float(np.mean(vals))


def collect(arm):
    """(series, eps, klass, dataset) -> {seed: value}"""
    per = defaultdict(dict)
    keys_seen = defaultdict(set)
    rows = (szquota_common.szquota_records() if arm == 'szquota'
            else load_rows())
    for d in rows:
        if d.get("mode") != "natural":
            continue
        ds = d["_dataset"]
        s = d["method"]
        eps = d.get("eps")
        for klass in CLASS_ORDER:
            v = class_value(d, klass)
            if v is None:
                continue
            per[(s, eps, klass, ds)][d.get("seed")] = v
            keys_seen[(ds, klass)].add(shape_of(d))
    return per, keys_seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--arm', choices=['released', 'szquota'], default='released',
                    help="Chain-NPM series source: released (archive) or the "
                         "szquota arm (campaign grids); default released")
    a = ap.parse_args()
    per, keys_seen = collect(a.arm)
    notes = []
    missing = []
    common = {}           # (klass, eps) -> 六方法共同有格的库列表
    plotted = {}          # (series, eps, klass) -> (values over common,), n
    for klass in CLASS_ORDER:
        for eps in EPS_ORDER:
            d_common = [ds for ds in DATASETS
                        if all(per.get((s, eps, klass, ds))
                               for s in SERIES_ORDER)]
            common[(klass, eps)] = d_common
        for s in SERIES_ORDER:
            for eps in EPS_ORDER:
                perds = []
                for ds in common[(klass, eps)]:
                    cell = per[(s, eps, klass, ds)]
                    perds.append(float(np.median(list(cell.values()))))
                plotted[(s, eps, klass)] = (perds, len(perds))
    # 缺格清单（仍然逐条留痕，但不参与绘制）
    for klass in CLASS_ORDER:
        for s in SERIES_ORDER:
            for eps in EPS_ORDER:
                for ds in DATASETS:
                    cell = per.get((s, eps, klass, ds))
                    if not cell:
                        missing.append((klass, s, eps, ds))
                    elif len(cell) != 10:
                        missing.append((klass, s, eps,
                                        "%s(seed n=%d)" % (ds, len(cell))))

    notes.append("cells present per class (series x eps x datasets):")
    for klass in CLASS_ORDER:
        cnt = sum(1 for s in SERIES_ORDER for eps in EPS_ORDER
                  for ds in DATASETS if per.get((s, eps, klass, ds)))
        notes.append("  %-5s %3d / %d" % (klass, cnt,
                                          len(SERIES_ORDER) * len(EPS_ORDER)
                                          * len(DATASETS)))
    notes.append("")
    notes.append("!! basis rule: a panel plots, for every series, the median "
                 "over ONLY the datasets where ALL SIX series report that "
                 "class, so no line inside a panel is aggregated over a "
                 "different dataset set than its neighbours")
    for klass in CLASS_ORDER:
        sets = sorted({tuple(common[(klass, e)]) for e in EPS_ORDER})
        notes.append("  %-5s common basis per eps: %s"
                     % (klass, " ; ".join(
                         "eps=%s: %s" % (e, common[(klass, e)])
                         for e in EPS_ORDER)))
        if len(sets) != 1:
            notes.append("       NOTE: the common basis differs across eps "
                         "for this class")
    if missing:
        notes.append("missing or short cells, NOT plotted (%d):" % len(missing))
        notes.extend("  %s" % (m,) for m in missing[:60])
    else:
        notes.append("none: every (series, eps, class, dataset) cell holds "
                     "exactly 10 seeds")

    # 对拍：2-hop 类别在四库上的中位数应与对应臂的 summary 一致（同级聚合）
    import csv
    summary_path = (os.path.join(RESULTS_DIR, "tables",
                                 "p7_summary_szquota.tsv")
                    if a.arm == 'szquota'
                    else os.path.join(RESULTS_DIR, "tables",
                                      "p7_summary.tsv"))
    summ = {}
    with open(summary_path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split("\t")
    for r in csv.DictReader(lines, delimiter="\t", fieldnames=hdr):
        if r["dataset"] != "dataset":
            summ[(r["dataset"], r["series"], float(r["eps"]))] = float(r["median"])
    bad = []
    for ds in DATASETS:
        for s in SERIES_ORDER:
            for eps in EPS_ORDER:
                cell = per.get((s, eps, "2hop", ds))
                if not cell:
                    continue
                med = float(np.median(list(cell.values())))
                ref = summ.get((ds, s, eps))
                if ref is None or abs(med - ref) > 1e-6:
                    bad.append((ds, s, eps, med, ref))
    notes.append("2-hop cross-check against p7_summary.tsv: %s"
                 % ("OK" if not bad else "MISMATCH %s" % bad[:5]))
    if bad:
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        raise SystemExit("refusing to draw: 2-hop panel disagrees with "
                         "p7_summary.tsv; see %s" % COVERAGE_TXT)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.6, "axes.titlesize": 8.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(7.05, 1.95),
                             gridspec_kw=dict(left=0.062, right=0.995,
                                              top=0.80, bottom=0.165,
                                              wspace=0.24))
    x = np.arange(len(EPS_ORDER), dtype=float)

    for ax, klass in zip(axes, CLASS_ORDER):
        for s in SERIES_ORDER:
            vals = []
            for eps in EPS_ORDER:
                perds, n = plotted[(s, eps, klass)]
                vals.append(float(np.median(perds)) if perds else np.nan)
            ax.plot(x, np.array(vals), color=META[s][1], marker=META[s][2],
                    ls="-", lw=1.15, markersize=3.2,
                    markerfacecolor=META[s][1], markeredgecolor="white",
                    markeredgewidth=0.4, label=META[s][0], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(EPS_LABELS)
        ax.set_title(CLASS_TITLE[klass], pad=2.6)
        ax.set_ylabel(CLASS_YLABEL[klass], labelpad=1.6)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.35, len(EPS_ORDER) - 0.65)
    if CLASS_ORDER[-1] == "tv":
        axes[-1].set_ylim(0.0, None)

    handles = [Line2D([], [], color=META[s][1], marker=META[s][2], ls="-",
                      lw=1.15, markersize=3.4, markeredgecolor="white",
                      markeredgewidth=0.4, label=META[s][0])
               for s in SERIES_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0.0)
    fig.text(0.545, 0.045, r"privacy budget $\varepsilon$", ha="center",
             va="top", fontsize=8.8)

    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
        fh.write("\n-- plotted values (median over the datasets that have the "
                 "cell) --\n")
        fh.write("%-5s %-10s %s\n" % ("class", "series",
                                      " ".join("%8s" % e for e in EPS_LABELS)))
        for klass in CLASS_ORDER:
            for s in SERIES_ORDER:
                row = []
                for eps in EPS_ORDER:
                    perds, n = plotted[(s, eps, klass)]
                    v = float(np.median(perds)) if perds else float("nan")
                    row.append(("%8.3f" % v) if perds else "       -")
                fh.write("%-5s %-10s %s\n" % (klass, s, " ".join(row)))
            fh.write("\n")
        fh.write("\n-- datasets actually contributing per class --\n")
        for klass in CLASS_ORDER:
            for ds in DATASETS:
                fh.write("  %-5s %-10s shapes=%s\n"
                         % (klass, ds, sorted(keys_seen[(ds, klass)])))
    print("written:", BASE_NAME + ".pdf")
    print("written:", COVERAGE_TXT)
    for n in notes:
        print("#", n)


if __name__ == "__main__":
    main()