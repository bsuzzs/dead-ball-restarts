"""Step 8 - manuscript figures 1-7 and supplementary figures S1-S5.

House rules applied throughout:
  * every interval drawn is the match-cluster bootstrap interval used in the
    text, never a per-restart Wilson interval, which would look narrower than
    the uncertainty actually claimed;
  * hexagonal maps use normalised coordinates and a display aspect correction, so
    the hexagons are geometrically regular;
  * a colour-blind-safe two-colour scheme separates the two outcome windows.
"""
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import ATTACKING_THIRD_X, FIGURE_DIR, OUTPUT_DIR, RESTART_LABELS
from common import read_restarts

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#5b5b5b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
ORDER = ["Penalty", "Free kick shot", "Corner", "Free kick cross",
         "Free Kick", "Throw in", "Goal kick"]


def save(fig, name, formats=("png", "pdf", "svg", "tiff")):
    for suffix in formats:
        extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(FIGURE_DIR / f"{name}.{suffix}", dpi=600, facecolor="white", **extra)
    plt.close(fig)
    print(f"  wrote {name}")


def dots(ax, values, lo, hi, y, colour, label=None):
    ax.errorbar(values, y, xerr=[values - lo, hi - values], fmt="o", ms=4.5,
                capsize=2.5, lw=1.1, color=colour, label=label)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def figure1(benchmarks):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.2))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.15, top=0.86, wspace=0.28)
    frame = benchmarks.set_index("restart").loc[ORDER]
    for ax, outcome, title in zip(axes, ["shot", "goal"], ["a  Shot rate", "b  Goal rate"]):
        for j, prefix in enumerate(["", "strict_"]):
            dots(ax, frame[f"{prefix}{outcome}_pct"].to_numpy(),
                 frame[f"{prefix}{outcome}_lo"].to_numpy(),
                 frame[f"{prefix}{outcome}_hi"].to_numpy(),
                 np.arange(7) + (j - 0.5) * 0.23, [BLUE, ORANGE][j],
                 ["15-s window", "Opponent-event cutoff"][j])
        ax.set_yticks(range(7), [RESTART_LABELS[k] for k in ORDER] if outcome == "shot" else [])
        ax.invert_yaxis()
        ax.set_xlabel("Restarts resulting in outcome (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
        ax.set_xlim(0.015, 160)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.85, 1.12), ncol=2, frameon=False)
    save(fig, "Figure1")


def figure2(rates):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.1, top=0.92, hspace=0.53, wspace=0.25)
    labels = {"sequence": ["Short-pass sequence", "Other observed sequence"],
              "coordinate": ["Outside-box endpoint", "Inside-box endpoint"]}
    for i, rule in enumerate(["sequence", "coordinate"]):
        for k, outcome in enumerate(["shot", "goal"]):
            ax = axes[i, k]
            for j, prefix in enumerate(["", "strict_"]):
                sub = (rates[(rates.analysis == rule) & (rates.outcome == prefix + outcome)]
                       .set_index("group").loc[[1, 0]])
                dots(ax, sub.pct.to_numpy(), sub.lo.to_numpy(), sub.hi.to_numpy(),
                     np.arange(2) + (j - 0.5) * 0.2, [BLUE, ORANGE][j],
                     ["15-s window", "Opponent-event cutoff"][j])
            ax.set_yticks([0, 1], labels[rule] if k == 0 else [])
            ax.invert_yaxis()
            ax.set_xlabel(f"{outcome.title()} rate (%)")
            ax.set_title(f"{chr(97 + i * 2 + k)}  {rule.title()} rule", loc="left",
                         fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "Figure2")


# --------------------------------------------------------------------------
# Pitch reference lines, in Wyscout normalised coordinates.
PA_X, PA_Y0, PA_Y1 = 83.0, 21.0, 79.0      # penalty area
GA_X, GA_Y0, GA_Y1 = 94.2, 36.8, 63.2      # six-yard area
POST_Y0 = 44.0                              # near goal post
MAP_CMAP, INK, MUTED = "cividis_r", "#172B3A", "#536472"


def hex_map(ax, hx, hy, values, gridsize, mincnt, extent, vmax):
    """Hexagonal binning of an outcome over pitch coordinates."""
    return ax.hexbin(hx, hy, C=values, reduce_C_function=np.mean,
                     gridsize=gridsize, mincnt=mincnt, cmap=MAP_CMAP,
                     vmin=0, vmax=vmax, edgecolors="white", linewidths=0.5,
                     zorder=3, extent=extent)


def regular_hexagons(ax, hb):
    """Force the drawn hexagons to be geometrically regular.

    Equal data units on the two axes do NOT give regular hexagons: hexbin's
    horizontal and vertical grid spacing depend on the extent. A point-up
    regular hexagon has height/width = 2/sqrt(3), so the axis aspect is
    derived from the path matplotlib actually produced.
    """
    vertices = hb.get_paths()[0].vertices
    dx, dy = np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1])
    aspect = (2 / np.sqrt(3)) * dx / dy
    ax.set_aspect(aspect, adjustable="box")
    return aspect


def dashed(ax):
    return dict(color=MUTED, lw=0.75, ls=(0, (4, 3)), zorder=5)


def figure3(restarts):
    """Corner landing map. Mirrored on the side the corner was taken from.

    Note the x >= 60 restriction is a display choice for this map only. The
    zone table in the supplement covers every corner with a recorded endpoint,
    so the two must not be given the same denominator.
    """
    corners = restarts[restarts.restart == "Corner"].dropna(subset=["end_x", "end_y"]).copy()
    corners["end_y_mirrored"] = np.where(corners.y > 50, 100 - corners.end_y, corners.end_y)
    corners = corners[corners.end_x >= 60]

    fig = plt.figure(figsize=(7.1, 6.8))
    ax = fig.add_axes([0.10, 0.11, 0.74, 0.80])
    ax.set_xlim(-6.5, 106.5)
    ax.set_ylim(56, 106)
    hb = hex_map(ax, corners.end_y_mirrored, corners.end_x, corners.goal,
                 gridsize=14, mincnt=10, extent=(-2, 102, 56, 102), vmax=0.10)
    regular_hexagons(ax, hb)
    for y in (PA_Y0, PA_Y1):
        ax.plot([y, y], [PA_X, 100], **dashed(ax))
    for y in (GA_Y0, GA_Y1):
        ax.plot([y, y], [GA_X, 100], **dashed(ax))
    ax.plot([GA_Y0, GA_Y1], [GA_X, GA_X], **dashed(ax))
    ax.plot([PA_Y0, PA_Y1], [PA_X, PA_X], **dashed(ax))
    ax.plot([0, 100], [100, 100], color=MUTED, lw=0.85, zorder=5)
    ax.set_xlabel("Lateral landing coordinate, y (mirrored)")
    ax.set_ylabel("Longitudinal landing coordinate, x")
    ax.set_title(f"{len(corners):,} deliveries · {int(corners.goal.sum())} goals · "
                 f"{corners.goal.mean() * 100:.2f}%", loc="left", fontweight="bold",
                 color=INK)
    bar = fig.colorbar(hb, cax=fig.add_axes([0.855, 0.11, 0.025, 0.80]),
                       ticks=np.linspace(0, 0.10, 6))
    bar.set_label("Goal rate")
    bar.ax.set_yticklabels([f"{v:.0%}" for v in np.linspace(0, 0.10, 6)])
    values = np.asarray(hb.get_array()) * 100
    save(fig, "Figure3")
    return dict(map_subset_n=int(len(corners)), cells=int(values.size),
                median_pct=float(np.median(values)), max_pct=float(values.max()),
                cells_at_or_above_6pct=int((values >= 6).sum()))


def figure6(restarts):
    """Direct free-kick shot map, plus goal rate by distance to the goal line."""
    shots = restarts[restarts.restart == "Free kick shot"].dropna(subset=["x", "y"]).copy()
    shots["y_mirrored"] = np.where(shots.y > 50, 100 - shots.y, shots.y)

    fig = plt.figure(figsize=(7.1, 4.4))
    ax = fig.add_axes([0.08, 0.26, 0.45, 0.64])
    ax.set_xlim(55, 102)
    ax.set_ylim(-2, 54)
    hb = hex_map(ax, shots.x, shots.y_mirrored, shots.goal,
                 gridsize=14, mincnt=5, extent=(55, 102, 0, 55), vmax=0.20)
    regular_hexagons(ax, hb)
    ax.plot([PA_X, 100], [PA_Y0, PA_Y0], **dashed(ax))
    ax.plot([PA_X, PA_X], [PA_Y0, 50], **dashed(ax))
    ax.plot([GA_X, 100], [GA_Y0, GA_Y0], **dashed(ax))
    ax.plot([GA_X, GA_X], [GA_Y0, 50], **dashed(ax))
    ax.plot([100, 100], [0, 50], color=MUTED, lw=0.85, zorder=5)
    ax.plot([100, 101.5, 101.5], [POST_Y0, POST_Y0, 50], color=INK, lw=1.25, zorder=6)
    ax.set_xlabel("Longitudinal coordinate, x")
    ax.set_ylabel("Lateral coordinate, y (mirrored)")
    ax.set_title("a  Shot location", loc="left", fontweight="bold", color=INK)
    # The colorbar sits in its own reserved band; its label needs room beneath
    # it or the tick text is clipped by the figure edge.
    bar = fig.colorbar(hb, cax=fig.add_axes([0.08, 0.115, 0.45, 0.028]),
                       orientation="horizontal", ticks=np.linspace(0, 0.20, 5))
    bar.ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 0.20, 5)], fontsize=7.5)
    bar.set_label("Goal rate", fontsize=8, labelpad=4)

    bands = pd.read_csv(OUTPUT_DIR / "freekick_distance_bands.csv")
    bands = bands[bands.metric == "d_line"].reset_index(drop=True)
    axb = fig.add_axes([0.68, 0.26, 0.29, 0.64])
    y = np.arange(len(bands))
    dots(axb, bands.goal_pct.to_numpy(), bands.lo.to_numpy(), bands.hi.to_numpy(), y, BLUE)
    axb.set_yticks(y, ["<22 m", "22-27 m", "27-32 m", ">=32 m"])
    axb.invert_yaxis()
    axb.set_ylim(len(bands) - 0.4, -0.6)
    axb.set_xlabel("Goal rate (%)")
    axb.set_title("b  By distance to goal line", loc="left", fontweight="bold", color=INK)
    # Counts go beside each estimate, on its own row, with the axis widened to
    # make room. Offsetting them vertically ran them into the panel title.
    lo, hi = axb.get_xlim()
    axb.set_xlim(lo, hi + 0.42 * (hi - lo))
    for i, row in bands.iterrows():
        axb.text(axb.get_xlim()[1], i, f"{int(row.goals)}/{int(row.n)}",
                 ha="right", va="center", fontsize=7, color=MUTED)
    save(fig, "Figure6")


def figure4(contrasts):
    groups = [(["shot", "opp_shot30"], ["Own shot, 15 s", "Opponent shot, 30 s"], "Shot-scale"),
              (["goal", "opp_goal60", "net_goal"],
               ["Own goal, 15 s", "Opponent goal, 60 s", "Net (own − opponent)"], "Goal-scale")]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9))
    fig.subplots_adjust(left=0.265, right=0.975, bottom=0.155, top=0.88,
                        hspace=0.75, wspace=0.60)
    for i, rule in enumerate(["sequence", "coordinate"]):
        sub = contrasts[contrasts.analysis == rule].set_index("outcome")
        for j, (keys, labels, scale) in enumerate(groups):
            ax = axes[i, j]
            frame = sub.loc[keys]
            y = np.arange(len(keys))
            dots(ax, frame.diff_pp.to_numpy(), frame.lo.to_numpy(), frame.hi.to_numpy(),
                 y, [BLUE, ORANGE][j])
            ax.axvline(0, color="0.35", lw=0.8, ls="--")
            ax.set_yticks(y, labels)
            ax.tick_params(labelsize=7.5)
            ax.invert_yaxis()
            ax.set_ylim(len(keys) - 0.45, -0.55)
            ax.set_xlabel("Difference, short minus other (pp)", fontsize=7.5)
            ax.set_title(f"{chr(97 + i * 2 + j)}  {rule.title()} rule, {scale.lower()}",
                         loc="left", fontweight="bold", fontsize=8.2)
    for j in range(2):
        lo = min(axes[i, j].get_xlim()[0] for i in range(2))
        hi = max(axes[i, j].get_xlim()[1] for i in range(2))
        for i in range(2):
            axes[i, j].set_xlim(lo, hi)
    fig.text(0.5, 0.022, "Bars are 95% match-clustered bootstrap intervals; own and "
                         "opponent quantities are differenced inside each resample.",
             ha="center", fontsize=7, color="0.3")
    save(fig, "Figure4")


def figure5(fine, contrasts):
    fig = plt.figure(figsize=(7.1, 5.6))
    grid = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.78, left=0.115, right=0.975,
                            bottom=0.10, top=0.92)
    ax = fig.add_subplot(grid[0, :])
    ax.fill_between(fine["mid"], fine.lo, fine.hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(fine["mid"], fine.shot_pct, "o-", ms=3.5, lw=1.2, color=BLUE)
    ax.axvline(ATTACKING_THIRD_X - 0.3, ls="--", lw=0.9, color="0.4")
    ax.text(ATTACKING_THIRD_X + 1, ax.get_ylim()[1] * 0.92, "Attacking third",
            fontsize=7.5, color="0.3")
    ax.set_xlabel("Throw-in location along pitch length "
                  "(0 = own goal line, 100 = opponent goal line)")
    ax.set_ylabel("15-s shot rate (%)")
    ax.set_title("a  Longitudinal gradient of throw-in shot rate", loc="left",
                 fontweight="bold")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)

    zone = contrasts[contrasts.analysis == "zone"].set_index("outcome")
    rows = [("shot", "15-s shot"), ("goal", "15-s goal"),
            ("strict_shot", "Shot, cutoff"), ("strict_goal", "Goal, cutoff")]
    axb = fig.add_subplot(grid[1, 0])
    frame = zone.loc[[k for k, _ in rows]]
    dots(axb, frame.diff_pp.to_numpy(), frame.lo.to_numpy(), frame.hi.to_numpy(),
         np.arange(4), BLUE)
    axb.axvline(0, color="0.35", lw=0.8, ls="--")
    axb.set_yticks(np.arange(4), [label for _, label in rows])
    axb.invert_yaxis()
    axb.set_xlabel("Attacking third minus elsewhere (pp)")
    axb.set_title("b  Zone contrast (restates location)", loc="left", fontweight="bold")

    pairs = [("attacking_third_forward", "shot", "Forward: shot"),
             ("attacking_third_forward", "goal", "Forward: goal"),
             ("attacking_third_long", "shot", "≥20 m: shot"),
             ("attacking_third_long", "goal", "≥20 m: goal")]
    axc = fig.add_subplot(grid[1, 1])
    sel = pd.DataFrame([contrasts[(contrasts.analysis == a) & (contrasts.outcome == o)].iloc[0]
                        for a, o, _ in pairs])
    dots(axc, sel.diff_pp.to_numpy(), sel.lo.to_numpy(), sel.hi.to_numpy(),
         np.arange(4), ORANGE)
    axc.axvline(0, color="0.35", lw=0.8, ls="--")
    axc.set_yticks(np.arange(4), [label for _, _, label in pairs])
    axc.tick_params(labelsize=7.5)
    axc.invert_yaxis()
    axc.set_xlabel("Difference (pp)")
    axc.set_title("c  Within attacking third", loc="left", fontweight="bold")
    save(fig, "Figure5")


def figure7(results):
    frame = results[(results.source.isin(["setpiece", "openplay"]))
                    & (results.comparison == "more")].set_index("source").loc[
        ["setpiece", "openplay"]]
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.2, top=0.84)
    left = np.zeros(2)
    for column, colour, label in [("wins", BLUE, "Win"), ("draws", "#b2b2b2", "Draw"),
                                  ("losses", ORANGE, "Loss")]:
        values = frame[column].to_numpy() / frame.n.to_numpy() * 100
        ax.barh([0, 1], values, left=left, color=colour, label=label, height=0.5)
        for y, value, base in zip([0, 1], values, left):
            if value < 6:
                ax.text(100, y - 0.36, f"{value:.1f}%", ha="right", va="center",
                        color=colour, fontsize=8)
            else:
                ax.text(base + value / 2, y, f"{value:.1f}%", ha="center", va="center",
                        color="white" if column != "draws" else "black", fontsize=8)
        left = left + values
    ax.set_yticks([0, 1], ["More non-penalty\nrestart goals", "More open-play\ngoals"])
    ax.invert_yaxis()
    ax.set_xlabel("Team-match outcomes (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    save(fig, "Figure7")


# --------------------------------------------------------------------------
def psm_figures(rule, name):
    balance = pd.read_csv(OUTPUT_DIR / f"psm_balance_{rule}.csv")
    scores = pd.read_csv(OUTPUT_DIR / f"psm_scores_{rule}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.8))
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.12, top=0.91, wspace=0.6)
    for column, colour, label in [("smd_before", ORANGE, "Before"), ("smd_after", BLUE, "After")]:
        axes[0].plot(balance[column].abs(), range(len(balance)), "o", ms=3,
                     color=colour, label=label)
    axes[0].set_yticks(range(len(balance)),
                       balance.variable.str.replace("competition_", "")
                       .str.replace("score_state_", ""))
    axes[0].tick_params(labelsize=6.5)
    axes[0].invert_yaxis()
    axes[0].axvline(0.1, ls="--", lw=0.7, color="gray")
    axes[0].set_xlabel("Absolute standardised mean difference")
    axes[0].legend(frameon=False, fontsize=7)
    for value, colour, label in [(1, BLUE, "Short / outside"), (0, ORANGE, "Other / inside")]:
        axes[1].hist(scores.loc[scores.tr == value, "ps"], bins=25, density=True,
                     histtype="step", color=colour, label=label, lw=1.2)
    axes[1].set_xlabel("Propensity score")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{rule.title()} classification: matching diagnostics", fontsize=10)
    save(fig, name)


def shap_figure():
    pretty = {"dist_goal": "Distance to goal", "lateral": "Lateral offset",
              "x": "Pitch length (x)", "y": "Pitch width (y)", "minute": "Match minute",
              "second_half": "Not first half", "home": "Home team",
              "national": "National-team fixture"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.4))
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.14, top=0.88, wspace=0.62)
    for ax, outcome, title in zip(axes, ["shot", "goal"],
                                  ["a  Shot outcome", "b  Goal outcome"]):
        path = OUTPUT_DIR / f"shap_importance_{outcome}.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path).head(10).iloc[::-1]
        names = [pretty.get(f, f.replace("restart_", "Type: ").replace("score_state_", "Score: "))
                 for f in frame.feature]
        ax.barh(range(len(frame)), frame.mean_abs_shap, color=BLUE, height=0.66)
        ax.set_yticks(range(len(frame)), names)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Mean |SHAP| (probability)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    save(fig, "FigureS5")


def schematics(benchmarks, classification):
    def box(ax, x, y, w, h, text, ec=GREY, size=7.6, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    ec=ec, fc="white", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, weight=weight)

    def arrow(ax, a, b, colour=GREY):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                     lw=0.9, color=colour, shrinkA=1, shrinkB=1))

    counts = benchmarks.set_index("restart").n
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.12, 0.900, 0.56, 0.082,
        "Wyscout open event data (Pappalardo et al., 2019)\n"
        "1,941 matches · 7 competitions · ~3.25 M events", ec=BLUE)
    box(ax, 0.12, 0.778, 0.56, 0.072, "Events of type “Free Kick”: 193,273 records", ec=BLUE)
    box(ax, 0.715, 0.688, 0.275, 0.058, "Excluded\n76 penalty-shootout kicks",
        ec=ORANGE, size=6.9)
    box(ax, 0.12, 0.636, 0.56, 0.082,
        f"Analysis sample: {total:,} restarts\nregular and extra time · seven categories",
        ec=BLUE, weight="bold")
    arrow(ax, (0.40, 0.900), (0.40, 0.852))
    arrow(ax, (0.40, 0.778), (0.40, 0.720))
    ax.plot([0.40, 0.685], [0.755, 0.742], lw=0.9, color=ORANGE, solid_capstyle="round")
    arrow(ax, (0.685, 0.742), (0.712, 0.724), ORANGE)
    width, gap = 0.128, 0.0105
    x0 = (1 - (7 * width + 6 * gap)) / 2
    for i, key in enumerate(["Corner", "Free kick cross", "Free kick shot", "Free Kick",
                             "Throw in", "Goal kick", "Penalty"]):
        colour = ORANGE if key == "Free Kick" else BLUE
        xi = x0 + i * (width + gap)
        box(ax, xi, 0.445, width, 0.128,
            f"{RESTART_LABELS[key].replace(' ', chr(10), 1)}\n{counts[key]:,}",
            ec=colour, size=6.5)
        arrow(ax, (0.40, 0.636), (xi + width / 2, 0.577))
    ax.text(0.5, 0.409, "Orange: the residual free-kick pass category, retained here",
            ha="center", fontsize=6.9, color=ORANGE)
    box(ax, 0.055, 0.238, 0.41, 0.130,
        "Outcome A — 15-s window\nfirst own shot after the restart,\n"
        "and whether it was scored\n(same half only)", ec=BLUE, size=7.3)
    box(ax, 0.535, 0.238, 0.41, 0.130,
        "Outcome B — opponent-event cutoff\nsame rule, truncated at the\n"
        "first opponent event\n(not a verified possession window)", ec=BLUE, size=7.3)
    arrow(ax, (0.26, 0.445), (0.26, 0.372))
    arrow(ax, (0.74, 0.445), (0.74, 0.372))
    box(ax, 0.055, 0.062, 0.89, 0.130,
        "Corner classification\n"
        f"primary: sequence rule, first own event within 6 s "
        f"(n = {classification['n']:,}; {classification['unclassified_n']:,} unclassified)\n"
        f"sensitivity: pass-endpoint coordinates (n = {int(counts['Corner']):,})\n"
        "Throw-in geography: attacking third, then direction and recorded displacement",
        size=7.1)
    arrow(ax, (0.5, 0.238), (0.5, 0.196))
    save(fig, "FigureS1")

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.815, 0.94, 0.135,
        "Covariates\ncompetition · pre-restart score state · minute · half · home or away · "
        "restart start x, y\npre-match cumulative points per game for team and for opponent · "
        "matches already played", size=7.2)
    panels = [(0.030, "Descriptive rates\nmatch-clustered\nbootstrap\n2,000 resamples",
               BLUE, "Primary uncertainty"),
              (0.278, "GEE\nbinomial, logit\nexchangeable\nclustered by team",
               BLUE, "Primary adjusted model"),
              (0.526, "GLMM\nmaximum likelihood\nGauss–Hermite\nquadrature",
               GREY, "Corroborating"),
              (0.774, "Propensity-score\nmatching\n1:1, logit caliper\nexact on competition",
               GREY, "Corroborating")]
    for x, text, colour, tag in panels:
        box(ax, x, 0.475, 0.196, 0.235, text, ec=colour, size=6.9)
        ax.text(x + 0.098, 0.437, tag, ha="center", fontsize=6.6, color=colour)
        arrow(ax, (x + 0.098, 0.815), (x + 0.098, 0.722))
    box(ax, 0.030, 0.215, 0.52, 0.175,
        "Supplementary predictive checks — diagnostic only\n"
        "logistic and boosted-tree classifiers, grouped by match\n"
        "SHAP ordering · post-corner GRU benchmark", ec=ORANGE, size=6.9)
    box(ax, 0.585, 0.215, 0.385, 0.175,
        "External description\nIMPECT · StatsBomb · Understat\n"
        "same first-shot rule where the\nsource permits it", ec=ORANGE, size=6.9)
    ax.text(0.5, 0.145, "All estimands are observational associations. Rate differences and "
                        "odds ratios are on different scales and are not interchangeable.",
            ha="center", fontsize=6.9, color=GREY)
    ax.text(0.5, 0.085, "Sequence and endpoint classifications are both measured after "
                        "execution; neither identifies tactical intent.",
            ha="center", fontsize=6.9, color=ORANGE)
    save(fig, "FigureS2")


def main() -> None:
    restarts = read_restarts()
    benchmarks = pd.read_csv(OUTPUT_DIR / "benchmarks.csv")
    corner_rates = pd.read_csv(OUTPUT_DIR / "corner_rates.csv")
    corner_contrasts = pd.read_csv(OUTPUT_DIR / "corner_contrasts.csv")
    throwin_contrasts = pd.read_csv(OUTPUT_DIR / "throwin_contrasts.csv")
    throwin_fine = pd.read_csv(OUTPUT_DIR / "throwin_x_fine.csv")
    match_results = pd.read_csv(OUTPUT_DIR / "match_results.csv")
    classification = json.loads((OUTPUT_DIR / "corner_classification.json").read_text())

    figure1(benchmarks)
    figure2(corner_rates)
    hex_stats = figure3(restarts)
    figure4(corner_contrasts)
    figure5(throwin_fine, throwin_contrasts)
    figure6(restarts)
    figure7(match_results)
    psm_figures("sequence", "FigureS3")
    psm_figures("coordinate", "FigureS4")
    shap_figure()
    schematics(benchmarks, classification)

    # Bin-level spread, so the text can avoid quoting a peak bin as a zone rate.
    (OUTPUT_DIR / "corner_map_bins.json").write_text(json.dumps(hex_stats, indent=2))
    print(f"corner map: {hex_stats['map_subset_n']:,} deliveries shown, "
          f"{hex_stats['cells']} bins, median {hex_stats['median_pct']:.2f}%, "
          f"max {hex_stats['max_pct']:.2f}%, "
          f"{hex_stats['cells_at_or_above_6pct']} bins at or above 6%")


if __name__ == "__main__":
    main()
