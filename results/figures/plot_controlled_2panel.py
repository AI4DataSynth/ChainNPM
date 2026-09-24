#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""受控基准合并图（替换原 Fig 7 fig:evidence 与 Fig 8 fig:recovery）。

版式
  跨双栏figure*，1 行 × 2 列 panel；
  (a) Δ_syn/Δ_real vs ε  —— 效应恢复，六方法（含 PrivMRF 三次运行的范围带）；
  (b) 2-hop 查询 RE vs ε —— 查询精度，同一批六方法。
  x 三档 ε ∈ {0.8, 3.2, 8.0}；每点 3 次运行，竖段给出运行跨度。

为什么合并
  原 fig:evidence（四个基线、无 Chain-NPM）与原 fig:recovery（同一 Δ 恢复、
  六方法）画的是**同一个量**，只有方法集不同——读稿时被判定为内容重叠。
  合并后 (a) 承载效应恢复、(b) 承载查询精度，两 panel 各自回答一个不同的问题。

诚实性约束（两个 panel 讲述的方向并不一致，图注不得模糊这一点）
  * (a) 中四个 per-FK 基线在 0 附近（效应被结构性丢弃），Chain-NPM 接近 1；
  * (b) 中 Chain-NPM 在 ε=0.8 **高于**四个基线（0.79 vs 0.19–0.41），
    到 ε≥3.2 才反超——查询精度与效应恢复是两个不同的问题，
    该交叉必须由图注明确说明，不能靠视觉掩盖。
  * 自检：四个基线的 Δ 中位数必须与正文一致（±0.011），否则拒绝出图。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.join(ROOT, "tmp")
BASE_NAME = os.path.join(HERE, "fig_controlled_2panel")
COVERAGE_TXT = os.path.join(HERE, "plot_controlled_2panel_coverage.txt")

EPS = [("0.8", 0.8), ("3.2", 3.2), ("8.0", 8.0)]
BASE_KEYS = ["pertable", "lava_asis", "lava_prop", "denorm"]
CHAIN_KEYS = ["chain_direct", "chain_mixture"]
# 正文口径的基线 Δ 中位数（ε=0.8/3.2/8.0），用于自检
# 2026-09-21：修复"逐表基线 ε 盲"后按 V2 实测重设（tab:gap 同步为
# 0.00/-0.02/-0.09 与 0.21/0.29/0.21）；此值必须与 main.tex 的 tab:gap 一致，
# 否则脚本会拒绝绘图（见下方 SystemExit）。
EXP_DELTA = {"pertable": [0.00, -0.02, -0.09], "lava_asis": [0.02, 0.01, 0.03],
             "lava_prop": [0.02, -0.004, -0.03], "denorm": [0.21, 0.29, 0.21]}
STYLES = [
    ("pertable",     "PrivMRF",         "#2b2b2b", "o", "-"),
    ("lava_asis",    "Lava-AsIs",       "#6e6e6e", "^", "-"),
    ("lava_prop",    "Lava-Prop",       "#a8a8a8", "s", "-"),
    ("denorm",       "PrivMRF (denorm)", "#1f4e8c", "D", "--"),
    ("chain_direct", "Chain-NPM-direct", "#c0392b", "v", "-"),
    ("chain_mixture", "Chain-NPM-mixture", "#e67e22", "P", "-"),
]


def load():
    out = {}
    for eps_str, _ in EPS:
        b = json.load(open(os.path.join(TMP,
                       "multihop_final_eps%s.json" % eps_str)))["results"][eps_str]
        c = json.load(open(os.path.join(TMP,
                       "chain_results_eps%s.json" % eps_str)))["results"][eps_str]
        out[eps_str] = dict(b, **c)
    return out


def main():
    res = load()
    notes = []
    bad = []
    for k, want in EXP_DELTA.items():
        got = [res[e][k]["delta_ratio"] for e, _ in EPS]
        if not all(abs(g - w) < 0.011 for g, w in zip(got, want)):
            bad.append((k, ["%.3f" % g for g in got], want))
    notes.append("self-check baseline delta_ratio medians vs manuscript: %s"
                 % ("OK" if not bad else "MISMATCH %s" % bad))
    notes.append("")
    notes.append("-- every plotted value --")
    notes.append("%-14s %-12s %8s %8s %8s" % ("method", "metric",
                                              *[e for e, _ in EPS]))
    for key, label, _c, _mk, _ls in STYLES:
        for f in ("delta_ratio", "2hop"):
            notes.append("%-14s %-12s %8.4f %8.4f %8.4f"
                         % (label, f, *[res[e][key][f] for e, _ in EPS]))
    notes.append("")
    notes.append("!! panel (b) crossing: at eps=0.8 the two Chain-NPM variants "
                 "have HIGHER 2-hop query RE than the four per-FK baselines "
                 "(direct %.3f / mixture %.3f vs %.3f--%.3f); from eps>=3.2 "
                 "they are far lower. Query accuracy and effect recovery are "
                 "different questions and the caption must say so."
                 % (res["0.8"]["chain_direct"]["2hop"],
                    res["0.8"]["chain_mixture"]["2hop"],
                    min(res["0.8"][k]["2hop"] for k in BASE_KEYS),
                    max(res["0.8"][k]["2hop"] for k in BASE_KEYS)))
    if bad:
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        raise SystemExit("refusing to draw: baseline deltas disagree with the "
                         "manuscript; see %s" % COVERAGE_TXT)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.8, "axes.titlesize": 8.4, "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35),
                             gridspec_kw=dict(left=0.092, right=0.995,
                                              top=0.815, bottom=0.165,
                                              wspace=0.19))
    xpos = [0, 1, 2]
    xt = [e for e, _ in EPS]

    # (a) Δ recovery
    ax = axes[0]
    runs = [res[e]["pertable"]["runs"] for e, _ in EPS]
    lo = [min(r["delta_ratio"] for r in rr) for rr in runs]
    hi = [max(r["delta_ratio"] for r in rr) for rr in runs]
    ax.fill_between(xpos, lo, hi, color="#bfbfbf", alpha=0.35, lw=0,
                    label="PrivMRF run range", zorder=1)
    for key, label, c, mk, ls in STYLES:
        med = [res[e][key]["delta_ratio"] for e, _ in EPS]
        rr = [[r["delta_ratio"] for r in res[e][key]["runs"]] for e, _ in EPS]
        ax.plot(xpos, med, color=c, ls=ls, lw=1.3, marker=mk, markersize=4.4,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        for x, rv in zip(xpos, rr):
            ax.plot([x, x], [min(rv), max(rv)], color=c, lw=1.0, alpha=0.35,
                    zorder=2, solid_capstyle="round")
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(-0.28, 1.22)
    ax.set_ylabel(r"$\Delta_{\mathrm{syn}}/\Delta_{\mathrm{real}}$",
                  labelpad=1.6)
    ax.set_title("(a) effect recovery", pad=2.6)

    # (b) 2-hop query RE
    ax = axes[1]
    for key, label, c, mk, ls in STYLES:
        med = [res[e][key]["2hop"] for e, _ in EPS]
        rr = [[r["2hop"] for r in res[e][key]["runs"]] for e, _ in EPS]
        ax.plot(xpos, med, color=c, ls=ls, lw=1.3, marker=mk, markersize=4.4,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        for x, rv in zip(xpos, rr):
            ax.plot([x, x], [min(rv), max(rv)], color=c, lw=1.0, alpha=0.35,
                    zorder=2, solid_capstyle="round")
    ax.set_ylim(0.0, 0.88)
    ax.set_ylabel("2-hop query RE", labelpad=1.6)
    ax.set_title("(b) query accuracy", pad=2.6)

    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels(xt)
        ax.set_xlabel(r"privacy budget $\varepsilon$", labelpad=1.6)
        ax.set_xlim(-0.22, 2.22)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.3,
                      markersize=4.4, markeredgecolor="white",
                      markeredgewidth=0.5, label=label)
               for _k, label, c, mk, ls in STYLES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, handletextpad=0.3, columnspacing=0.8,
               borderaxespad=0.0)

    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    print("written:", BASE_NAME + ".pdf")
    print("written:", COVERAGE_TXT)
    for n in notes[:3]:
        print("#", n)
    print("#", notes[-1])


if __name__ == "__main__":
    main()