"""
Визуализация оценки одного конспекта.

Использование:
    python visualize_single.py <scores.json> [--out report.png] [--title "Заголовок"]

Пример:
    python visualize_single.py lecture_01/lcw/lcw_scores.json --title "LCW — Лекция 1"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

MAX_SCORE = 10

LABELS = {
    "p1": "P1\nСтруктура",
    "p2": "P2\nДеперсонализация",
    "p3": "P3\nСтрогость",
    "p4a": "P4a\nВизуал (намерение)",
    "p4b": "P4b\nВизуал (исполнение)",
    "p5": "P5\nКорректность",
    "p6": "P6\nПолнота",
}

COLORS = {
    "bar":    "#4C72B0",
    "radar":  "#4C72B0",
    "fill":   "#A8C1E8",
    "covered":  "#4CAF50",
    "partial":  "#FFC107",
    "missed":   "#F44336",
    "grid":   "#DDDDDD",
    "bg":     "#F8F9FA",
}


def extract_scores(data: dict) -> dict[str, float | None]:
    def get(path):
        try:
            node = data
            for key in path:
                node = node[key]
            v = node
            return float(v) if v is not None else None
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


def draw_radar(ax, scores: dict, color: str, fill_color: str, label: str | None = None):
    keys = [k for k, v in scores.items() if v is not None]
    if not keys:
        ax.set_visible(False)
        return

    values = [scores[k] for k in keys]
    tick_labels = [LABELS[k] for k in keys]
    n = len(keys)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values + values[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels, size=8)
    ax.set_ylim(0, MAX_SCORE)
    ax.set_yticks(range(2, MAX_SCORE + 1, 2))
    ax.set_yticklabels([str(i) for i in range(2, MAX_SCORE + 1, 2)], size=7, color="grey")
    ax.yaxis.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_visible(False)

    ax.plot(angles, values_plot, color=color, linewidth=2, label=label)
    ax.fill(angles, values_plot, color=fill_color, alpha=0.25)

    for angle, value in zip(angles[:-1], values):
        ax.annotate(
            f"{value:.0f}",
            xy=(angle, value),
            xytext=(angle, value + 0.6),
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=color,
        )


def draw_bars(ax, scores: dict, color: str, title: str):
    keys = list(scores.keys())
    values = [scores[k] if scores[k] is not None else 0 for k in keys]
    labels = [LABELS[k] for k in keys]
    nones  = [scores[k] is None for k in keys]

    y = np.arange(len(keys))
    bars = ax.barh(y, values, color=[COLORS["grid"] if n else color for n in nones],
                   height=0.55, zorder=2)
    ax.set_xlim(0, MAX_SCORE)
    ax.set_xticks(range(0, MAX_SCORE + 1, 2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Балл", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axvline(MAX_SCORE, color=COLORS["grid"], linewidth=1, linestyle="--")
    ax.grid(axis="x", color=COLORS["grid"], linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    for bar, val, none in zip(bars, values, nones):
        label = "N/A" if none else f"{val:.0f}"
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9, fontweight="bold",
            color="grey" if none else color,
        )


def draw_coverage_donut(ax, covered: int, partial: int, missed: int, title: str):
    total = covered + partial + missed
    if total == 0:
        ax.text(0.5, 0.5, "Нет данных\nkey_units", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    sizes  = [covered, partial, missed]
    colors = [COLORS["covered"], COLORS["partial"], COLORS["missed"]]
    labels = [f"Покрыто\n{covered}", f"Частично\n{partial}", f"Пропущено\n{missed}"]

    wedges, texts = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              fontsize=8, ncol=3, frameon=False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.text(0, 0, f"{total}\nтем", ha="center", va="center", fontsize=10, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    data = json.loads(args.scores_json.read_text(encoding="utf-8"))
    scores = extract_scores(data)
    covered, partial, missed = extract_coverage_counts(data)

    report_title = args.title or args.scores_json.stem
    out_path = args.out or args.scores_json.with_suffix(".png")

    fig = plt.figure(figsize=(14, 9), facecolor=COLORS["bg"])
    fig.suptitle(report_title, fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97, top=0.92, bottom=0.07,
                           wspace=0.35, hspace=0.45)

    # Радар
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_radar.set_facecolor(COLORS["bg"])
    draw_radar(ax_radar, scores, COLORS["radar"], COLORS["fill"])
    ax_radar.set_title("Профиль оценок", fontsize=11, fontweight="bold", pad=18)

    # Горизонтальные бары
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_bars.set_facecolor(COLORS["bg"])
    draw_bars(ax_bars, scores, COLORS["bar"], "Баллы по парадигмам")

    # Donut P6
    ax_donut = fig.add_subplot(gs[1, 0])
    ax_donut.set_facecolor(COLORS["bg"])
    draw_coverage_donut(ax_donut, covered, partial, missed, "P6 — Охват тем")

    # Текстовая сводка
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.set_facecolor(COLORS["bg"])
    ax_text.axis("off")

    valid = {k: v for k, v in scores.items() if v is not None}
    avg = np.mean(list(valid.values())) if valid else 0

    missed_list = data.get("p6_coverage", {}).get("missed_topics", [])

    lines = [f"Среднее: {avg:.1f} / {MAX_SCORE}"]
    lines.append("")
    for k, label in LABELS.items():
        v = scores[k]
        short = label.replace("\n", " ")
        lines.append(f"{short}: {'N/A' if v is None else f'{v:.0f}'}")
    if missed_list:
        lines.append("")
        lines.append("Пропущенные темы:")
        for t in missed_list[:6]:
            lines.append(f"  • {t[:60]}{'…' if len(t) > 60 else ''}")
        if len(missed_list) > 6:
            lines.append(f"  … ещё {len(missed_list) - 6}")

    ax_text.text(0.05, 0.95, "\n".join(lines), transform=ax_text.transAxes,
                 va="top", fontsize=9, family="monospace",
                 bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": COLORS["grid"]})

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
