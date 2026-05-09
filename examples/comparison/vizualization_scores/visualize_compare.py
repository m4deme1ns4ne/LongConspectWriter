"""
Сравнение оценок двух конспектов.

Использование:
    python visualize_compare.py <scores_a.json> <scores_b.json> [--labels A B] [--out compare.png] [--title "..."]

Пример:
    python visualize_compare.py lecture_01/lcw/lcw_scores.json lecture_01/gemini/gemini_scores.json \
        --labels LCW Gemini --title "Лекция 1 — сравнение"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

MAX_SCORE = 10

LABELS = {
    "p1":  "P1 Структура",
    "p2":  "P2 Деперсон.",
    "p3":  "P3 Строгость",
    "p4a": "P4a Визуал (нам.)",
    "p4b": "P4b Визуал (исп.)",
    "p5":  "P5 Корректность",
    "p6":  "P6 Полнота",
}

RADAR_LABELS = {
    "p1":  "P1\nСтруктура",
    "p2":  "P2\nДеперсонализация",
    "p3":  "P3\nСтрогость",
    "p4a": "P4a\nВизуал (намерение)",
    "p4b": "P4b\nВизуал (исполнение)",
    "p5":  "P5\nКорректность",
    "p6":  "P6\nПолнота",
}

PALETTE = {
    "a":      "#4C72B0",
    "a_fill": "#A8C1E8",
    "b":      "#DD8452",
    "b_fill": "#F5C9A8",
    "grid":   "#DDDDDD",
    "bg":     "#F8F9FA",
    "pos":    "#4CAF50",
    "neg":    "#F44336",
    "neutral":"#9E9E9E",
    "covered":  "#4CAF50",
    "partial":  "#FFC107",
    "missed":   "#F44336",
}


def extract_scores(data: dict) -> dict[str, float | None]:
    def get(path):
        try:
            node = data
            for key in path:
                node = node[key]
            return float(node) if node is not None else None
        except (KeyError, TypeError):
            return None

    return {
        "p1":  get(["p1_structure", "score"]),
        "p2":  get(["p2_depersonalization", "score"]),
        "p3":  get(["p3_formalism", "score"]),
        "p4a": get(["p4_visualization", "p4a_intent", "score"]),
        "p4b": get(["p4_visualization", "p4b_execution", "score"]),
        "p5":  get(["p5_factual_accuracy", "score"]),
        "p6":  get(["p6_coverage", "score"]),
    }


def extract_coverage_counts(data: dict) -> tuple[int, int, int]:
    units = data.get("p6_coverage", {}).get("key_units", [])
    covered = sum(1 for u in units if u.get("status") == "covered")
    partial  = sum(1 for u in units if u.get("status") == "partial")
    missed   = sum(1 for u in units if u.get("status") == "missed")
    return covered, partial, missed


def draw_radar_compare(ax, scores_a, scores_b, label_a, label_b):
    keys = [k for k in scores_a if scores_a[k] is not None or scores_b[k] is not None]
    if not keys:
        ax.set_visible(False)
        return

    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([RADAR_LABELS[k] for k in keys], size=8)
    ax.set_ylim(0, MAX_SCORE)
    ax.set_yticks(range(2, MAX_SCORE + 1, 2))
    ax.set_yticklabels([str(i) for i in range(2, MAX_SCORE + 1, 2)], size=7, color="grey")
    ax.yaxis.grid(True, color=PALETTE["grid"], linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color=PALETTE["grid"], linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_visible(False)

    for scores, color, fill, label in [
        (scores_a, PALETTE["a"], PALETTE["a_fill"], label_a),
        (scores_b, PALETTE["b"], PALETTE["b_fill"], label_b),
    ]:
        vals = [scores[k] if scores[k] is not None else 0 for k in keys]
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, color=color, linewidth=2, label=label)
        ax.fill(angles, vals_plot, color=fill, alpha=0.2)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, frameon=False)


def draw_grouped_bars(ax, scores_a, scores_b, label_a, label_b):
    keys = list(scores_a.keys())
    short_labels = [LABELS[k] for k in keys]
    vals_a = [scores_a[k] if scores_a[k] is not None else 0 for k in keys]
    vals_b = [scores_b[k] if scores_b[k] is not None else 0 for k in keys]

    y = np.arange(len(keys))
    h = 0.35

    ax.barh(y - h / 2, vals_a, height=h, color=PALETTE["a"], label=label_a, zorder=2)
    ax.barh(y + h / 2, vals_b, height=h, color=PALETTE["b"], label=label_b, zorder=2)

    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(va + 0.1, i - h / 2, f"{va:.0f}", va="center", fontsize=8,
                color=PALETTE["a"], fontweight="bold")
        ax.text(vb + 0.1, i + h / 2, f"{vb:.0f}", va="center", fontsize=8,
                color=PALETTE["b"], fontweight="bold")

    ax.set_xlim(0, MAX_SCORE + 1.5)
    ax.set_xticks(range(0, MAX_SCORE + 1, 2))
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Балл", fontsize=9)
    ax.set_title("Сравнение по парадигмам", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.legend(fontsize=9, frameon=False, loc="lower right")


def draw_delta_bars(ax, scores_a, scores_b, label_a, label_b):
    keys = list(scores_a.keys())
    short_labels = [LABELS[k] for k in keys]
    deltas = []
    for k in keys:
        va = scores_a[k] if scores_a[k] is not None else 0
        vb = scores_b[k] if scores_b[k] is not None else 0
        deltas.append(va - vb)

    y = np.arange(len(keys))
    colors = [PALETTE["pos"] if d > 0 else (PALETTE["neg"] if d < 0 else PALETTE["neutral"])
              for d in deltas]

    ax.barh(y, deltas, color=colors, height=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel(f"Δ ({label_a} − {label_b})", fontsize=9)
    ax.set_title("Разница оценок", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    for i, d in enumerate(deltas):
        if d == 0:
            continue
        ax.text(d + (0.1 if d > 0 else -0.1), i,
                f"{d:+.0f}", va="center",
                ha="left" if d > 0 else "right",
                fontsize=8, fontweight="bold",
                color=PALETTE["pos"] if d > 0 else PALETTE["neg"])

    xmax = max(abs(d) for d in deltas) if deltas else 1
    ax.set_xlim(-(xmax + 1.5), xmax + 1.5)


def draw_donut(ax, covered, partial, missed, title, color_center):
    total = covered + partial + missed
    if total == 0:
        ax.text(0.5, 0.5, "Нет\nkey_units", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    sizes  = [covered, partial, missed]
    colors = [PALETTE["covered"], PALETTE["partial"], PALETTE["missed"]]
    ax.pie(sizes, colors=colors, startangle=90,
           wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.5})
    ax.text(0, 0, f"{total}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=color_center)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


def draw_summary_table(ax, scores_a, scores_b, label_a, label_b):
    ax.axis("off")
    valid_a = [v for v in scores_a.values() if v is not None]
    valid_b = [v for v in scores_b.values() if v is not None]
    avg_a = np.mean(valid_a) if valid_a else 0
    avg_b = np.mean(valid_b) if valid_b else 0

    rows = []
    for k, label in LABELS.items():
        va = scores_a[k]
        vb = scores_b[k]
        sa = "N/A" if va is None else f"{va:.0f}"
        sb = "N/A" if vb is None else f"{vb:.0f}"
        if va is not None and vb is not None:
            winner = f"← {label_a}" if va > vb else (f"→ {label_b}" if vb > va else "=")
        else:
            winner = ""
        rows.append([label, sa, sb, winner])

    rows.append(["──────────", "──", "──", ""])
    rows.append(["Среднее", f"{avg_a:.1f}", f"{avg_b:.1f}",
                 f"← {label_a}" if avg_a > avg_b else (f"→ {label_b}" if avg_b > avg_a else "=")])

    col_labels = ["Парадигма", label_a, label_b, "Победитель"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor("#E8EEF4")
            cell.set_text_props(fontweight="bold")
        elif rows[r - 1][0].startswith("──"):
            cell.set_facecolor("#F0F0F0")
        elif r == len(rows):
            cell.set_facecolor("#EAF4EA")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")

    ax.set_title("Итоговая таблица", fontsize=11, fontweight="bold", pad=12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_a", type=Path)
    parser.add_argument("scores_b", type=Path)
    parser.add_argument("--labels", nargs=2, default=None, metavar=("LABEL_A", "LABEL_B"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    data_a = json.loads(args.scores_a.read_text(encoding="utf-8"))
    data_b = json.loads(args.scores_b.read_text(encoding="utf-8"))

    scores_a = extract_scores(data_a)
    scores_b = extract_scores(data_b)
    label_a, label_b = args.labels if args.labels else (args.scores_a.stem, args.scores_b.stem)

    cov_a = extract_coverage_counts(data_a)
    cov_b = extract_coverage_counts(data_b)

    report_title = args.title or f"{label_a} vs {label_b}"
    out_path = args.out or Path(f"compare_{label_a}_vs_{label_b}.png")

    fig = plt.figure(figsize=(16, 11), facecolor=PALETTE["bg"])
    fig.suptitle(report_title, fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97, top=0.94, bottom=0.04,
                           wspace=0.4, hspace=0.5)

    # Радар
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_radar.set_facecolor(PALETTE["bg"])
    draw_radar_compare(ax_radar, scores_a, scores_b, label_a, label_b)
    ax_radar.set_title("Профиль оценок", fontsize=11, fontweight="bold", pad=18)

    # Сгруппированные бары
    ax_bars = fig.add_subplot(gs[0, 1:])
    ax_bars.set_facecolor(PALETTE["bg"])
    draw_grouped_bars(ax_bars, scores_a, scores_b, label_a, label_b)

    # Дельта
    ax_delta = fig.add_subplot(gs[1, :2])
    ax_delta.set_facecolor(PALETTE["bg"])
    draw_delta_bars(ax_delta, scores_a, scores_b, label_a, label_b)

    # Donuts P6
    ax_donut_a = fig.add_subplot(gs[1, 2])
    ax_donut_a.set_facecolor(PALETTE["bg"])
    draw_donut(ax_donut_a, *cov_a, f"P6 охват — {label_a}", PALETTE["a"])

    ax_donut_b = fig.add_subplot(gs[2, 2])
    ax_donut_b.set_facecolor(PALETTE["bg"])
    draw_donut(ax_donut_b, *cov_b, f"P6 охват — {label_b}", PALETTE["b"])

    # Итоговая таблица
    ax_table = fig.add_subplot(gs[2, :2])
    ax_table.set_facecolor(PALETTE["bg"])
    draw_summary_table(ax_table, scores_a, scores_b, label_a, label_b)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
