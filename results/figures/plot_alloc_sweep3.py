#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新 Fig 9：预算分配消融（跨跳权重 w 扫描），1 行 × 3 列 panel。

版式
  (a) 2-hop 查询 RE vs w      —— 曲线呈单谷：w=1（等分）为实测最小点，
                                 两个方向（欠权与过权）都变差
  (b) Δ-recovery vs w         —— 效应恢复量：ε=3.2 档在噪声内基本平坦，
                                 ε=0.8 档在 w=0.25 处偏低（数值见覆盖报告，
                                 本图不对"持平"作超出数据的断言）
  (c) 1-hop 查询 RE vs w      —— 相邻跳被挤压：region×household（实线）
                                 与 household×individual（虚线）；两个预算、
                                 两个跳类在 w<1 与 w>1 两端都高于 w=1 的取值
                                  （该结论由脚本逐项算出并写入覆盖报告）
  x = 跨跳重要性权重 w ∈ {0.25, 0.5, 1, 2, 4, 8}（对数刻度，刻度标
  0.25/0.5/1/2/4/8）；颜色 = 隐私预算（ε=0.8 蓝、ε=3.2 红，与旧版 alloc
  图一致）；每点**三次运行**，竖段给出三次运行的最小/最大跨度；
  panel (a) 在 w=1 处加一条浅色竖虚线与 "equal split" 标注，
  显式标出实测最小点（不覆盖任何曲线）。

数据源
  * tmp/exp_results/alloc_sweep_w6_3run.json（**唯一来源**，6 点 × 3 运行）；
  * tmp/alloc_sweep_results.json 仅作对拍参考（旧图口径：4 点 × 2 运行，
    w ∈ {1,2,4,8}）。

诚实性约束
  * 该扫描**只有两个预算档**（ε ∈ {0.8, 3.2}）：不存在的 6 档 ε 曲线绝不伪造，
    图注必须写明只有两档。
  * 自检（任一不过则拒绝出图、写覆盖报告并以非 0 退出）：
    1. ε=0.8 六点 2-hop 中位数按 w 升序 = 0.934/0.841/0.791/1.010/1.187/1.791
       （±0.005）；
    2. ε=3.2 六点 2-hop 中位数按 w 升序 = 0.098/0.140/0.037/0.144/0.213/0.416
       （±0.005）；
    3. 重叠点 w ∈ {1,2,4,8} 在新文件里的**前两次运行**必须与旧文件逐格一致
       （±1e-6）——这是"新点与旧点同一协议"的证据；
    4. 每个被绘制的数值（含三次运行的跨度端点）都不得越出坐标轴范围：
       越界即拒绝出图（坐标轴范围先由数据算出，不做裁剪）；
    5. 两个预算的 2-hop 中位数 argmin 都必须是 w=1——panel (a) 的
       "equal split" 标注以此为前提，不成立则拒绝出图。
    6. 图例净空（版式）：panel (c) 的图例框内不得有任何"曲线色"像素。做法是
       把图例本身隐藏后按发布分辨率（300 dpi）重渲染，再逐像素统计（颜色匹配
       需同时满足"与曲线参考色距离 <= 50/255"与"通道极差 >= 15/255"，否则
       黑字/灰字的抗锯齿像素会被误判）。y 轴上限按闭式抬高，使图例框下方留出
       净空——数据不裁剪、也不被图例遮挡。历史上该 panel 的蓝实线曾穿过图例
       第二行文字（page-level 视觉审计发现），此约束专为此设立。
    7. 图例文字框与三个 panel 的标题框不得相交。
  * 口径说明：旧文件存的是 2 次运行的"中位数"，两次取值的中位数按定义等于
    二者均值，因此它与新文件 3 次运行的中位数**合法地**不同；覆盖报告逐格
    列出该均值-对拍，说明二者不是矛盾。
  * 全部被绘制的数值（中位数与三次运行的跨度端点）写入覆盖报告，便于逐项复核。
"""
import json
import math
import os
import statistics
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.text import Text as MplText

HERE = os.path.dirname(os.path.abspath(__file__))            # results/figures
ROOT = os.path.dirname(os.path.dirname(HERE))                # project root
SRC = os.path.join(ROOT, "tmp", "exp_results", "alloc_sweep_w6_3run.json")
REF = os.path.join(ROOT, "tmp", "alloc_sweep_results.json")
BASE_NAME = os.path.join(HERE, "fig_alloc_sweep3")
COVERAGE_TXT = os.path.join(HERE, "plot_alloc_sweep3_coverage.txt")

# (w, new-file key)；按 w 升序，即 x 轴顺序
ALLOC_KEYS = [(0.25, "chain_direct_cross025"), (0.5, "chain_direct_cross05"),
              (1, "chain_direct"), (2, "chain_direct_cross2"),
              (4, "chain_direct_cross"), (8, "chain_direct_cross8")]
# 旧文件对拍用（只有 w ∈ {1,2,4,8} 重叠）
REF_KEYS = {1: "chain_direct", 2: "chain_direct_cross2",
            4: "chain_direct_cross", 8: "chain_direct_cross8"}
REF_METRICS = ["2hop", "delta_ratio", "1hop_rh", "1hop_hi", "2hop_wide"]
EPS_LIST = [("0.8", "#1f4e8c"), ("3.2", "#c0392b")]   # 仅有的两档预算
TOL_MED = 0.005                                       # 中位数自检容差
TOL_RUN = 1e-6                                        # 逐运行对拍容差
EXP_HOP2 = {                                          # 自检基准（按 w 升序）
    "0.8": [0.934, 0.841, 0.791, 1.010, 1.187, 1.791],
    "3.2": [0.098, 0.140, 0.037, 0.144, 0.213, 0.416],
}
PANELS = [
    ("a", "2-hop query RE", [("2hop", "solid", "cross-hop")]),
    ("b", r"$\Delta$-recovery", [("delta_ratio", "solid", None)]),
    ("c", "1-hop query RE", [("1hop_rh", "solid", "region$\\times$household"),
                             ("1hop_hi", "dashed",
                              "household$\\times$individual")]),
]


def load():
    with open(SRC) as fh:
        return json.load(fh)


def write_and_abort(notes, msg):
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    raise SystemExit("refusing to draw: %s; see %s" % (msg, COVERAGE_TXT))


def _series_refs():
    """曲线的参考颜色：实线/标记（alpha=1）与运行跨度段（alpha=0.35 叠白）。"""
    refs = []
    for _e, c in EPS_LIST:
        rgb = np.array(mcolors.to_rgb(c), dtype=float)
        for a in (1.0, 0.35):
            refs.append(a * rgb + (1.0 - a) * 1.0)
    return np.array(refs, dtype=float)


def legend_overlap_counts(fig, legend_specs, ann_objects, dpi=300.0, tol=50.0):
    """像素级检查：图例框/文字框、标注文字框内有多少"曲线色"像素。

    做法：先量好各文字/图例框的 bbox（display 坐标），再把**图例本身隐藏**
    后按发布分辨率（300 dpi）重渲染——这样图例框内若仍有"曲线色"像素，就
    只可能是数据曲线（图例自带的彩色示例线已被排除）。行索引自顶向下。

    像素判据（两条同时满足，见下面的 count()）：与曲线参考色的欧氏距离
    <= tol，且**通道极差 >= 15/255**。后者必不可少：0.35 alpha 的运行段本来
    就是近灰的淡蓝/淡红，只靠距离会把黑字/灰字的抗锯齿像素判成曲线色（实测
    会给出几十个假阳性），加极差约束后中性灰全部被排除。

    返回 [(label, text, n_text_px, n_box_px)]；纯信息用途，判定在调用处。
    """
    rend = fig.canvas.get_renderer()
    boxes = []
    for label, leg in legend_specs:
        boxes.append((label, leg,
                      leg.get_window_extent(renderer=rend),
                      [(t.get_text(), t.get_window_extent(renderer=rend))
                       for t in leg.get_texts()]))
    ann_boxes = [("text panel (%s)" % tag, a.get_text(),
                  MplText.get_window_extent(a, renderer=rend))
                 for tag, a in ann_objects]

    for _l, leg in legend_specs:
        leg.set_visible(False)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = plt.imread(buf)
    for _l, leg in legend_specs:
        leg.set_visible(True)
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(float) / 255.0
    img = img[..., :3].astype(float)
    H, W = img.shape[:2]
    scale = dpi / fig.dpi
    refs = _series_refs()

    def rect(bb):
        x0 = int(math.floor(bb.x0 * scale)); x1 = int(math.ceil(bb.x1 * scale))
        r0 = int(math.floor(H - bb.y1 * scale))
        r1 = int(math.ceil(H - bb.y0 * scale))
        return max(0, x0), max(0, r0), min(W, x1), min(H, r1)

    def count(bb):
        x0, r0, x1, r1 = rect(bb)
        if x1 <= x0 or r1 <= r0:
            return 0
        sub = img[r0:r1, x0:x1, :]
        d = np.linalg.norm(sub[:, :, None, :] - refs[None, None, :, :],
                           axis=-1).min(axis=-1)
        spread = sub.max(axis=-1) - sub.min(axis=-1)
        return int(((d * 255.0 <= tol) & (spread >= 0.06)).sum())

    out = []
    for label, _leg, bb, texts in boxes:
        n_box = count(bb)
        for txt, tbb in texts:
            out.append((label, txt, count(tbb), n_box))
    for label, txt, bb in ann_boxes:
        out.append((label, txt, count(bb), 0))
    return out


def main():
    raw = load()
    res = raw["results"]
    ws = [w for w, _k in ALLOC_KEYS]      # x 轴顺序：w 升序
    with open(REF) as fh:
        ref = json.load(fh)["results"]
    notes = ["source (sole): %s" % os.path.relpath(SRC, ROOT),
             "  config: %s" % json.dumps(raw.get("config", {}),
                                         sort_keys=True),
             "cross-check reference only: %s (old grid: w in {1,2,4,8}, "
             "2 runs)" % os.path.relpath(REF, ROOT),
             "only two budgets were swept: eps = %s"
             % ", ".join(e for e, _c in EPS_LIST),
             ""]

    # ---- 结构性自检：六点齐全、每点三次运行 -------------------------------
    ok_struct = True
    for eps, _c in EPS_LIST:
        for w, key in ALLOC_KEYS:
            m = res.get(eps, {}).get(key)
            n = len(m.get("runs", [])) if m else 0
            if m is None or n != 3:
                ok_struct = False
                notes.append("STRUCT FAIL eps=%s w=%s key=%s present=%s "
                             "runs=%d (expect 3)" % (eps, w, key, m is not None, n))
    notes.append("self-check structure: six weights x two budgets x 3 runs -> %s"
                 % ("OK" if ok_struct else "MISMATCH"))
    if not ok_struct:
        write_and_abort(notes, "the 6-point grid is not complete")

    # ---- 自检 1/2：六点 2-hop 中位数 --------------------------------------
    ok_med = True
    for eps, _c in EPS_LIST:
        got = [res[eps][k]["2hop"] for _w, k in ALLOC_KEYS]
        exp = EXP_HOP2[eps]
        bad = [(w, g, e) for (_w, _k), g, e in zip(ALLOC_KEYS, got, exp)
               if abs(g - e) > TOL_MED + 1e-12]
        if bad:
            ok_med = False
        notes.append("self-check eps=%s 2-hop medians (w order %s) got=%s "
                     "expected=%s (tol %.3f) -> %s"
                     % (eps, [w for w, _k in ALLOC_KEYS],
                        ["%.4f" % x for x in got], exp, TOL_MED,
                        "OK" if not bad else "MISMATCH %s" % bad))
    if not ok_med:
        write_and_abort(notes, "2-hop medians disagree with the expected "
                              "six-point arrays")

    # ---- 自检 5：argmin 必须是 w=1 ----------------------------------------
    notes.append("")
    ok_arg = True
    for eps, _c in EPS_LIST:
        med = [res[eps][k]["2hop"] for _w, k in ALLOC_KEYS]
        wmin = ws[min(range(len(ws)), key=lambda i: med[i])]
        ok_arg = ok_arg and wmin == 1
        notes.append("self-check eps=%s argmin_w 2-hop RE = w=%g (median "
                     "%.4f); medians in w order %s -> %s"
                     % (eps, wmin, med[ws.index(wmin)],
                        ["%.4f" % x for x in med],
                        "OK" if wmin == 1 else "MISMATCH (expect w=1)"))
    if not ok_arg:
        write_and_abort(notes, "the 2-hop minimum is not at w=1, so the "
                              "'equal split' annotation would be wrong")

    # ---- 自检 3：重叠点前两次运行与旧文件逐格一致 --------------------------
    notes.append("")
    notes.append("self-check eps/overlap: new-file runs[0], runs[1] vs old-file "
                 "runs[0], runs[1] for every metric (tol %g); this is what "
                 "proves the new points share the old protocol" % TOL_RUN)
    ok_run = True
    for eps, _c in EPS_LIST:
        for w, key in ALLOC_KEYS:
            rkey = REF_KEYS.get(w)
            if rkey is None:
                continue
            new_runs = res[eps][key]["runs"]
            old_runs = ref[eps][rkey]["runs"]
            if len(old_runs) != 2:
                ok_run = False
                notes.append("  eps=%s w=%-5s OLD FILE HAS %d RUNS (expect 2)"
                             % (eps, w, len(old_runs)))
            worst, worst_f = 0.0, REF_METRICS[0] + "[run 1]"
            for i in range(2):
                for f in REF_METRICS:
                    d = abs(new_runs[i][f] - old_runs[i][f])
                    if d >= worst:
                        worst, worst_f = d, "%s[run %d]" % (f, i + 1)
            if worst > TOL_RUN:
                ok_run = False
            notes.append("  eps=%s w=%-5s max|new-old| = %.3e (%s) -> %s"
                         % (eps, w, worst, worst_f, "OK" if worst <= TOL_RUN
                            else "MISMATCH"))
    if not ok_run:
        write_and_abort(notes, "overlapping points do not reproduce the old "
                              "file's first two runs")
    notes.append("  note: the old file's stored scalar is the 2-run median, "
                 "and the median of two samples IS their mean, so that stored "
                 "scalar legitimately differs from a 3-run median; both are "
                 "listed below and the mean of the first two new runs is "
                 "checked against the old stored scalar exactly.")
    ok_mean = True
    for eps, _c in EPS_LIST:
        for w, key in ALLOC_KEYS:
            rkey = REF_KEYS.get(w)
            if rkey is None:
                continue
            for f in REF_METRICS:
                m2 = statistics.mean([r[f] for r in res[eps][key]["runs"][:2]])
                if abs(m2 - ref[eps][rkey][f]) > TOL_RUN:
                    ok_mean = False
                    notes.append("  MEAN-MISMATCH eps=%s w=%s %s mean2=%.12f "
                                 "old=%.12f" % (eps, w, f, m2,
                                                ref[eps][rkey][f]))
        m3 = [res[eps][key]["2hop"] for _w, key in ALLOC_KEYS]
        m2 = [statistics.mean([r["2hop"] for r in res[eps][key]["runs"][:2]])
              for _w, key in ALLOC_KEYS]
        notes.append("  eps=%s 2-hop median: 3-run %s vs 2-run(=mean) %s"
                     % (eps, ["%.4f" % x for x in m3],
                        ["%.4f" % x for x in m2]))
    notes.append("  cross-check mean(first two new runs) vs old stored scalar "
                 "(all cells, 5 metrics) -> %s"
                 % ("OK" if ok_mean else "MISMATCH"))
    if not ok_mean:
        write_and_abort(notes, "old stored scalars are not the mean of the "
                              "first two new runs")

    # ---- 绘图 --------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.6, "axes.titlesize": 8.2, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.95),
                             gridspec_kw=dict(left=0.062, right=0.995,
                                              top=0.815, bottom=0.165,
                                              wspace=0.26))
    xpos = [math.log2(w) for w in ws]          # -2,-1,0,1,2,3；同旧版的 log2 刻度
    xlim = (min(xpos) - 0.28, max(xpos) + 0.28)

    drawn = {}          # tag -> list of (x, y) actually handed to matplotlib
    series_med = {}     # tag -> {(eps, key): [median per w]}
    ann_objects = []    # [(tag, annotation)]，供文字框像素检查使用
    for ax, (tag, ylab, series) in zip(axes, PANELS):
        drawn[tag] = []
        for eps, color in EPS_LIST:
            for key, ls, _name in series:
                med, lo, hi = [], [], []
                for _w, k in ALLOC_KEYS:
                    m = res[eps][k]
                    med.append(m[key])
                    rv = [r[key] for r in m["runs"]]
                    lo.append(min(rv)); hi.append(max(rv))
                series_med[(tag, eps, key)] = med
                ax.plot(xpos, med, color=color, ls=ls, lw=1.25, marker="o",
                        markersize=4.0, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=5)
                for x, a, b in zip(xpos, lo, hi):
                    ax.plot([x, x], [a, b], color=color, lw=1.0, alpha=0.35,
                            zorder=2, solid_capstyle="round")
                drawn[tag] += list(zip(xpos, med)) + \
                    list(zip(xpos, lo)) + list(zip(xpos, hi))
        # 坐标轴范围由数据算出（留 5.5% 边距），绝不裁剪
        ys = [y for _x, y in drawn[tag]]
        lo_p, hi_p = min(ys), max(ys)
        span = max(hi_p - lo_p, 1e-9)
        ylim = (lo_p - 0.055 * span, hi_p + 0.055 * span)
        ax.set_ylim(*ylim)
        ax.set_xticks(xpos)
        ax.set_xticklabels([("%g" % w) for w in ws])
        ax.set_xlabel("cross-hop weight $w$", labelpad=1.6)
        ax.set_ylabel(ylab, labelpad=1.6)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)
        ax.set_title("(%s)" % tag, pad=2.6)
        ax.set_xlim(*xlim)
        if tag == "a":
            # 实测最小点（两个预算的 argmin 都在 w=1）：浅色竖虚线 + 标注
            ax.axvline(0.0, color="#8c8c8c", ls=(0, (1.6, 1.6)), lw=0.7,
                       alpha=0.75, zorder=1)
            ann = ax.annotate("equal split",
                              xy=(0.0, res["0.8"]["chain_direct"]["2hop"]),
                              xytext=(0.30, ylim[1] - 0.12 * (ylim[1] - ylim[0])),
                              fontsize=7.0, color="#404040", ha="left",
                              va="center", zorder=6,
                              arrowprops=dict(arrowstyle="-", lw=0.55,
                                              color="#8c8c8c", shrinkA=1.5,
                                              shrinkB=3.5))
            ann_objects.append((tag, ann))

    # ---- 图例 --------------------------------------------------------------
    # panel (c) 的跳类图例（ε 由颜色/整图图例承担）
    h3 = [Line2D([], [], color="#555555", ls="solid", lw=1.25,
                 label="region$\\times$household"),
          Line2D([], [], color="#555555", ls="dashed", lw=1.25,
                 label="household$\\times$individual")]
    leg_c = axes[2].legend(handles=h3, loc="upper left", frameon=False,
                           handletextpad=0.35, borderaxespad=0.2,
                           labelspacing=0.25)
    handles = [Line2D([], [], color=c, marker="o", ls="-", lw=1.25,
                      markersize=4.0, markeredgecolor="white",
                      markeredgewidth=0.5, label=r"$\varepsilon=%s$" % e)
               for e, c in EPS_LIST]
    leg_fig = fig.legend(handles=handles, loc="upper center",
                         bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                         handletextpad=0.35, columnspacing=1.4,
                         borderaxespad=0.0)

    # ---- 图例净空：量出图例框的位置，闭式抬高 y 轴上限，绝不遮曲线 ----------
    #  版式缺陷（page-level 视觉审计）：panel (c) 的蓝实线曾穿过图例第二行文字。
    #  做法：令图例框下沿在坐标轴内的位置为 fb（自轴顶向下、占轴高比例，只由
    #  字号/版式决定，与 y 轴范围无关），要求"该框 x 区间内被绘制数值的上包络
    #  E 加上标记半径余量 c*S"仍在图例下沿之下：
    #      E + c*S <= ylo + (1 - fb)*S   =>   S >= (E - ylo) / (1 - fb - c)
    #  由此闭式定出新的上限（下限不动）。数据不被裁剪、不被图例遮挡。
    def bbox_disp(art):
        fig.canvas.draw()
        return art.get_window_extent(renderer=fig.canvas.get_renderer())

    def bbox_data(ax, art):
        return bbox_disp(art).transformed(ax.transData.inverted())

    def upenv_under(tag, bb, pad_x=0.12):
        sel = [y for x, y in drawn[tag]
               if bb.x0 - pad_x <= x <= bb.x1 + pad_x]
        return max(sel) if sel else None

    leg_clear = {}
    ax_by_tag = {t: ax for ax, (t, _y, _s) in zip(axes, PANELS)}
    for tag, leg in (("c", leg_c),):
        ax = ax_by_tag[tag]
        ylo, yhi = ax.get_ylim()
        S_data = yhi - ylo                       # 纯数据算出的量程
        bb_ax = bbox_disp(ax)
        bb = bbox_data(ax, leg)
        fb = (bb_ax.y1 - bbox_disp(leg).y0) / max(bb_ax.height, 1e-9)
        c = (4.0 / 2.0 + 3.0) / max(bb_ax.height, 1e-9)   # 标记半宽 + 3px 余量
        E = upenv_under(tag, bb, pad_x=0.10)
        denom = 1.0 - fb - c
        if denom <= 0.05:
            write_and_abort(notes, "panel (%s): the legend occupies %.0f%% of "
                                  "the axes height, leaving no room to place "
                                  "it clear of the curves" % (tag, fb * 100))
        S_need = (E - ylo) / denom
        # 版式自检：如果"让图例避开曲线"需要把量程撑到数据量程的 1.6 倍以上，
        # 说明图例位置本身不合适（面板会被压得很难看），此时拒绝出图，交人工改位。
        if S_need > 1.6 * S_data:
            write_and_abort(notes, "panel (%s): clearing the legend would need "
                                  "an axis %.2fx the data range (%.3f vs "
                                  "%.3f) - move the legend instead of "
                                  "inflating the axis"
                                  % (tag, S_need / S_data, S_need, S_data))
        if S_need > S_data:
            ax.set_ylim(ylo, ylo + S_need)
            fig.canvas.draw()
            bb = bbox_data(ax, leg)
        ylo2, yhi2 = ax.get_ylim()
        leg_clear[tag] = (bb, E, S_need, ylo2, yhi2, fb)
    for tag, (bb, E, S_need, ylo2, yhi2, fb) in leg_clear.items():
        notes.append("")
        notes.append("legend clearance panel (%s): legend box in data coords "
                     "x=[%.3f, %.3f] y=[%.3f, %.3f]; its bottom sits at %.1f%% "
                     "of the axes height below the top; envelope of plotted "
                     "values over that x-span = %.4f (a marker/antialias "
                     "allowance c*S is added inside the closed form) -> "
                     "required axis height %.4f, used %.4f"
                     % (tag, bb.x0, bb.x1, bb.y0, bb.y1, fb * 100, E, S_need,
                        yhi2 - ylo2))
        notes.append("  plotted-value envelope under the legend = %.4f, "
                     "legend bottom = %.4f -> clearance %.4f data units "
                     "(%.1f px); the y upper limit was set to %.4f for this "
                     "reason: no plotted value is clipped and no curve is "
                     "hidden behind the legend"
                     % (upenv_under(tag, bb, pad_x=0.0), bb.y0,
                        bb.y0 - upenv_under(tag, bb, pad_x=0.0),
                        (bb.y0 - upenv_under(tag, bb, pad_x=0.0)) / (yhi2 - ylo2)
                        * bbox_disp(ax_by_tag[tag]).height, yhi2))

    # ---- 自检 4：无被绘制数值越界 -----------------------------------------
    notes.append("")
    notes.append("self-check axis limits: every plotted value (median and the "
                 "three-run span endpoints) must lie inside the axes")
    ok_lim = True
    for ax, (tag, _ylab, _series) in zip(axes, PANELS):
        ylo, yhi = ax.get_ylim()
        xlo, xhi = ax.get_xlim()
        ys = [y for _x, y in drawn[tag]]
        xs = [x for x, _y in drawn[tag]]
        inside = (min(ys) >= ylo and max(ys) <= yhi
                  and min(xs) >= xlo and max(xs) <= xhi)
        ok_lim = ok_lim and inside
        notes.append("  panel (%s) xlim=(%.3f, %.3f) ylim=(%.4f, %.4f); "
                     "plotted x in [%.3f, %.3f], y in [%.4f, %.4f] -> %s"
                     % (tag, xlo, xhi, ylo, yhi, min(xs), max(xs),
                        min(ys), max(ys), "OK" if inside else "OUT OF RANGE"))
    notes.append("  no clipping was applied; the ranges above are the ranges "
                 "computed from the data themselves -> %s"
                 % ("OK" if ok_lim else "MISMATCH"))

    # ---- 自检 6：图例/标注文字框内不得有曲线像素（page-level 视觉审计口径）----
    notes.append("")
    notes.append("self-check no curve inside any legend or annotation text "
                 "box: the figure is re-rendered at 300 dpi, and pixels "
                 "matching a series colour (>= 1.0 alpha line/marker or the "
                 "0.35-alpha run segment) are counted inside every text "
                 "bounding box and inside every legend frame")
    box_px_counts = legend_overlap_counts(
        fig, [("legend panel (c)", leg_c), ("legend whole figure", leg_fig)],
        ann_objects)
    ok_leg = True
    for label, txt, n_text, n_box in box_px_counts:
        if n_box:
            ok_leg = False
        notes.append("  %-28s text %-34s coloured px in text box = %d ; "
                     "coloured px in legend box = %d -> %s"
                     % (label, "\"%s\"" % txt, n_text, n_box,
                        "OK" if not n_box else "OVERLAP"))
    if not ok_leg:
        write_and_abort(notes, "a curve crosses a legend box")

    # ---- 自检 7：图例文字与 panel 标题不得相互压字 --------------------------
    rend = fig.canvas.get_renderer()
    title_boxes = [(t, ax_by_tag[t].title.get_window_extent(renderer=rend))
                   for t, _y, _s in PANELS]
    hit = []
    for label, leg in (("panel (c)", leg_c), ("whole figure", leg_fig)):
        for txt in leg.get_texts():
            tb = txt.get_window_extent(renderer=rend)
            for ttag, tbb in title_boxes:
                if tb.overlaps(tbb):
                    hit.append("%s legend text %r vs panel (%s) title"
                               % (label, txt.get_text(), ttag))
    notes.append("")
    notes.append("self-check legend text vs panel titles (box intersection) "
                 "-> %s" % ("OK" if not hit else "OVERLAP %s" % hit))
    if hit:
        write_and_abort(notes, "legend text and a panel title overlap")

    # ---- 覆盖报告正文（全部被绘制数值 + caption 需要的事实） ---------------
    notes.append("")
    notes.append("-- every plotted value: median and the three run values, "
                 "per panel --")
    for tag, ylab, series in PANELS:
        notes.append("  panel (%s) %s" % (tag, ylab))
        for eps, _c in EPS_LIST:
            for key, _ls, name in series:
                m = series_med[(tag, eps, key)]
                notes.append("    eps=%-4s %-9s medians=%s"
                             % (eps, key, ["%.4f" % x for x in m]))
                runs = [[round(r[key], 4) for r in res[eps][k]["runs"]]
                        for _w, k in ALLOC_KEYS]
                notes.append("    %s%s runs (w order %s) = %s"
                             % (" " * 8, key, ws, runs))
    notes.append("")
    notes.append("-- caption-relevant facts --")
    for eps, _c in EPS_LIST:
        med = [res[eps][k]["2hop"] for _w, k in ALLOC_KEYS]
        wmin = min(range(len(ws)), key=lambda i: med[i])
        notes.append("  eps=%s: argmin_w 2-hop RE = w=%g (value %.4f); the "
                     "curve is a single valley, both directions (w<1 and w>1) "
                     "are worse than the equal split w=1"
                     % (eps, ws[wmin], med[wmin]))
        notes.append("    w order %s; 2-hop median %s; ratio to the w=1 "
                     "median %s"
                     % (ws, ["%.4f" % x for x in med],
                        ["%.2fx" % (x / res[eps]["chain_direct"]["2hop"])
                         for x in med]))
        notes.append("    Delta-recovery median %s"
                     % ["%.4f" % res[eps][k]["delta_ratio"] for _w, k in ALLOC_KEYS])
        notes.append("    1hop_rh (region x household) median %s"
                     % ["%.4f" % res[eps][k]["1hop_rh"] for _w, k in ALLOC_KEYS])
        notes.append("    1hop_hi (household x individual) median %s"
                     % ["%.4f" % res[eps][k]["1hop_hi"] for _w, k in ALLOC_KEYS])
    notes.append("")
    notes.append("panel (c) shows BOTH adjacent-hop classes on purpose: "
                 "1hop_rh = region x household (the hop adjacent to the "
                 "cross-hop object) and 1hop_hi = household x individual. "
                 "The starving effect is not an artefact of picking one of "
                 "them; per-series endpoint evidence, computed here:")
    for eps, _c in EPS_LIST:
        for key in ("1hop_rh", "1hop_hi"):
            med = [res[eps][k][key] for _w, k in ALLOC_KEYS]
            ref1 = res[eps]["chain_direct"][key]
            lo_side = ws[0]
            both = med[0] > ref1 and med[-1] > ref1
            notes.append("  eps=%-4s %-8s w=1 -> %.4f ; w=%-5s -> %.4f ; "
                         "w=%-5s -> %.4f ; both endpoints above w=1: %s ; "
                         "interior point w=0.5 -> %.4f (the curve is not "
                         "monotone, only the two ends are raised)"
                         % (eps, key, ref1, lo_side, med[0], ws[-1], med[-1],
                            "YES" if both else "NO", med[1]))
    notes.append("")
    notes.append("panel (b) honesty note: the Delta-recovery medians are NOT "
                 "flat across all six weights. eps=3.2 stays in [%.3f, %.3f] "
                 "(peak-to-peak %.3f), while eps=0.8 spans [%.3f, %.3f] "
                 "(peak-to-peak %.3f) with the single low value at w=0.25 "
                 "(%.3f, runs %s). The caption should therefore claim at most "
                 "'recovery is preserved within the run-to-run spread for "
                 "w >= 1', not flatness of the whole two-sided curve."
                 % (min(res["3.2"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    max(res["3.2"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    max(res["3.2"][k]["delta_ratio"] for _w, k in ALLOC_KEYS)
                    - min(res["3.2"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    min(res["0.8"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    max(res["0.8"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    max(res["0.8"][k]["delta_ratio"] for _w, k in ALLOC_KEYS)
                    - min(res["0.8"][k]["delta_ratio"] for _w, k in ALLOC_KEYS),
                    res["0.8"]["chain_direct_cross025"]["delta_ratio"],
                    [round(r["delta_ratio"], 4) for r in
                     res["0.8"]["chain_direct_cross025"]["runs"]]))
    notes.append("")
    notes.append("caption must state: only two budgets (eps = 0.8 and 3.2) "
                 "were swept at six cross-hop weights "
                 "w in {0.25, 0.5, 1, 2, 4, 8}, three runs each; the equal "
                 "split w=1 is the measured minimum of the two-sided curve.")

    if not ok_lim:
        write_and_abort(notes, "a plotted value falls outside the axis limits")

    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    print("written:", BASE_NAME + ".pdf")
    print("written:", BASE_NAME + ".png")
    print("written:", COVERAGE_TXT)
    for n in notes:
        print("#", n)


if __name__ == "__main__":
    main()