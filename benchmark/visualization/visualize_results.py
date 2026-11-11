import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = BASE_DIR / "results" / "idea_foresight"
EXP_FEASIBILITY_ROOT = BASE_DIR / "results" / "protocol_evaluation"
OUTPUT_DIR = BASE_DIR / "results" / "visulized_res"
RASTER_DPI = 450
SINGLE_CHART_FIGSIZE = (13.5, 7.4)
COMPARISON_CHART_FIGSIZE = (14.5, 7.8)
VECTOR_EXTENSIONS = (".svg",)

PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#303030",
    "axes.titlecolor": "#202020",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.color": "#303030",
    "ytick.color": "#303030",
    "grid.color": "#d4d4d4",
    "grid.linestyle": "--",
    "grid.linewidth": 0.75,
    "font.size": 16,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.dpi": 180,
    "savefig.dpi": RASTER_DPI,
}


METRICS = [
    ("strategy_overlap_avg", "Idea-strategy overlap", "#E69F00"),
    ("strategy_combination_avg", "Idea-strategy combination", "#CC79A7"),
]

LLM_BASED_EVALUATION_SERIES = [
    ("strategy_overlap_avg", "Idea-strategy overlap", "#F7A24F"),
    ("strategy_combination_avg", "Idea-strategy combination", "#FBEB66"),
    ("exp_reproducibility_avg", "Experimental reproducibility", "#4EA660"),
    ("exp_feasibility_avg", "Experimental feasibility", "#79CAFB"),
]

MODEL_DISPLAY_NAMES = {
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    "gemini-2.0-flash-thinking-exp": "Gemini-2.0-Flash-thinking-exp",
    "gemini-2.5-pro-thinking": "Gemini-2.5-Pro-Thinking",
    "gpt-5": "GPT-5",
    "gpt_researcher": "GPT-Researcher",
    "grok-4": "Grok-4",
    "supramas": "SupraMAS",
}

MODEL_ORDER = [
    "supramas",
    "gpt_researcher",
    "grok-4",
    "gpt-5",
    "gemini-2.5-pro-thinking",
    "deepseek-v3",
]


def get_experiment_summary_score(row: dict[str, Any]) -> float:
    if "overall_avg" in row:
        return float(row.get("overall_avg", 0.0) or 0.0)
    return float(row.get("avg_score", 0.0) or 0.0)


def load_summary_rows(summary_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}

    for summary_path in summary_root.glob("*/evaluation_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = summary_path.parent.name
        if model_name not in order_map:
            continue
        rows.append(
            {
                "model": model_name,
                "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                **payload,
            }
        )

    rows.sort(key=lambda row: (order_map.get(str(row["model"]), 10**9), str(row["model"])))
    if not rows:
        raise FileNotFoundError(f"No evaluation_summary.json found under {summary_root}")
    return rows


def load_experiment_feasibility_rows(summary_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}

    for summary_path in summary_root.glob("*/evaluation_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = summary_path.parent.name
        if model_name not in order_map:
            continue
        rows.append(
            {
                "model": model_name,
                "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                **payload,
            }
        )

    rows.sort(key=lambda row: (order_map.get(str(row["model"]), 10**9), str(row["model"])))
    if not rows:
        raise FileNotFoundError(f"No evaluation_summary.json found under {summary_root}")
    return rows


def cleanup_old_outputs(output_dir: Path) -> None:
    for pattern in ("*_bar_chart.png", "*_bar_chart.svg", "*_bar_chart.pdf"):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)


def build_output_path(metric_key: str) -> Path:
    return OUTPUT_DIR / f"{metric_key}_bar_chart.png"


def save_figure_bundle(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.08)
    for ext in VECTOR_EXTENSIONS:
        fig.savefig(output_path.with_suffix(ext), bbox_inches="tight", pad_inches=0.08)


def format_score_label(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def annotate_grouped_bars(
    ax: plt.Axes,
    bar_groups: list[tuple[Any, list[float]]],
    x_shift_step: float = 0.0,
    base_y_offset: float = 7.0,
    stacked_y_offset: float = 11.0,
    closeness_threshold: float = 0.04,
    label_color: str = "white",
    inside_bars: bool = True,
) -> None:
    if not bar_groups:
        return

    group_count = len(bar_groups[0][1])
    series_count = len(bar_groups)
    centered_offsets = [((idx - (series_count - 1) / 2) * x_shift_step) for idx in range(series_count)]
    direction = -1 if inside_bars else 1
    vertical_alignment = "top" if inside_bars else "bottom"

    for group_idx in range(group_count):
        items: list[tuple[int, Any, float]] = []
        for series_idx, (bars, values) in enumerate(bar_groups):
            items.append((series_idx, bars[group_idx], values[group_idx]))

        sorted_items = sorted(items, key=lambda item: item[2])
        label_levels: dict[int, int] = {}
        prev_value: float | None = None
        prev_level = 0
        for series_idx, _, value in sorted_items:
            if prev_value is not None and abs(value - prev_value) < closeness_threshold:
                prev_level += 1
            else:
                prev_level = 0
            label_levels[series_idx] = prev_level
            prev_value = value

        for series_idx, bar, value in items:
            ax.annotate(
                format_score_label(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(
                    centered_offsets[series_idx],
                    direction * (base_y_offset + label_levels[series_idx] * stacked_y_offset),
                ),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=10.4,
                fontweight="semibold",
                color=label_color,
                clip_on=False,
                zorder=5,
            )


def render_metric_chart(rows: list[dict[str, Any]], metric_key: str, title: str, color: str) -> Path:
    labels = [str(row["display_name"]) for row in rows]
    values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
    output_path = build_output_path(metric_key)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=SINGLE_CHART_FIGSIZE)
        bars = ax.bar(
            labels,
            values,
            color=color,
            width=0.62,
            edgecolor="#4a4a4a",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_title(title, pad=12, fontweight="normal")
        ax.set_xlabel("Baselines")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.12))
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", width=0.8)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.spines["top"].set_linewidth(0.9)
        ax.spines["right"].set_linewidth(0.9)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=13,
                color="#303030",
            )

        fig.tight_layout()
        save_figure_bundle(fig, output_path)
        plt.close(fig)
    return output_path


def render_experimental_feasibility_chart(rows: list[dict[str, Any]]) -> Path:
    labels = [str(row["display_name"]) for row in rows]
    values = [get_experiment_summary_score(row) for row in rows]
    output_path = OUTPUT_DIR / "experimental_feasibility_evaluation_bar_chart.png"

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=SINGLE_CHART_FIGSIZE)
        bars = ax.bar(
            labels,
            values,
            color="#56B4E9",
            width=0.62,
            edgecolor="#4a4a4a",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_title("Experiment evaluation", pad=12, fontweight="normal")
        ax.set_xlabel("Baselines")
        ax.set_ylabel("Average score")
        ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.12))
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", width=0.8)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=13,
                color="#303030",
            )

        fig.tight_layout()
        save_figure_bundle(fig, output_path)
        plt.close(fig)
    return output_path


def render_experiment_dimension_comparison_chart(rows: list[dict[str, Any]]) -> Path:
    labels = [str(row["display_name"]) for row in rows]
    reproducibility_values = [float(row.get("exp_reproducibility_avg", 0.0) or 0.0) for row in rows]
    feasibility_values = [float(row.get("exp_feasibility_avg", 0.0) or 0.0) for row in rows]
    output_path = OUTPUT_DIR / "experiment_dimension_comparison_bar_chart.png"

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=COMPARISON_CHART_FIGSIZE)
        x = list(range(len(labels)))
        width = 0.38

        bars_reproducibility = ax.bar(
            [v - width / 2 for v in x],
            reproducibility_values,
            width=width,
            label="Reproducibility",
            color="#4C78A8",
            edgecolor="#4a4a4a",
            linewidth=0.7,
            zorder=3,
        )
        bars_feasibility = ax.bar(
            [v + width / 2 for v in x],
            feasibility_values,
            width=width,
            label="Feasibility",
            color="#72B7B2",
            edgecolor="#4a4a4a",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_title("Experiment evaluation dimensions", pad=12, fontweight="normal")
        ax.set_xlabel("Baselines")
        ax.set_ylabel("Average score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, max(1.0, max(reproducibility_values + feasibility_values, default=0.0) * 1.15))
        ax.grid(axis="y", alpha=0.7)
        ax.legend(frameon=False, loc="upper right")
        ax.set_axisbelow(True)

        annotate_grouped_bars(
            ax,
            [
                (bars_reproducibility, reproducibility_values),
                (bars_feasibility, feasibility_values),
            ],
            x_shift_step=0.0,
            base_y_offset=6.0,
            stacked_y_offset=9.0,
            closeness_threshold=0.035,
        )

        fig.tight_layout()
        save_figure_bundle(fig, output_path)
        plt.close(fig)
    return output_path


def render_llm_based_evaluation_chart(
    idea_rows: list[dict[str, Any]], exp_rows: list[dict[str, Any]]
) -> Path:
    idea_map = {str(row["model"]): row for row in idea_rows}
    exp_map = {str(row["model"]): row for row in exp_rows}
    common_models = [m for m in MODEL_ORDER if m in idea_map and m in exp_map]
    if not common_models:
        raise ValueError("No common baselines between Idea_forward and experiment_feasibility summaries.")

    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in common_models]
    output_path = OUTPUT_DIR / "llm_based_evaluation_bar_chart.png"

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=COMPARISON_CHART_FIGSIZE)
        x = list(range(len(labels)))
        width = 0.23
        series_offsets = (-1.5, -0.5, 0.5, 1.5)
        bar_groups: list[tuple[Any, list[float]]] = []
        all_values: list[float] = []

        for (metric_key, legend_label, color), offset in zip(LLM_BASED_EVALUATION_SERIES, series_offsets):
            if metric_key.startswith("exp_"):
                values = [float(exp_map[m].get(metric_key, 0.0) or 0.0) for m in common_models]
            else:
                values = [float(idea_map[m].get(metric_key, 0.0) or 0.0) for m in common_models]
            bars = ax.bar(
                [v + offset * width for v in x],
                values,
                width=width,
                label=legend_label,
                color=color,
                edgecolor="#4a4a4a",
                linewidth=0.7,
                zorder=3,
            )
            bar_groups.append((bars, values))
            all_values.extend(values)

        ax.set_title("LLM-based evaluation", pad=12, fontweight="normal")
        ax.set_xlabel("Baselines")
        ax.set_ylabel("Average score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, max(1.0, max(all_values, default=0.0) * 1.15))
        ax.grid(axis="y", alpha=0.7)
        ax.legend(frameon=False, loc="upper center", ncol=2)
        ax.set_axisbelow(True)
        ax.margins(x=0.06)

        annotate_grouped_bars(
            ax,
            bar_groups,
            x_shift_step=0.0,
            base_y_offset=8.0,
            stacked_y_offset=10.0,
            closeness_threshold=0.04,
            label_color="#202020",
            inside_bars=False,
        )

        fig.tight_layout()
        save_figure_bundle(fig, output_path)
        plt.close(fig)
    return output_path


def main() -> None:
    idea_rows = load_summary_rows(SUMMARY_ROOT)
    exp_rows = load_experiment_feasibility_rows(EXP_FEASIBILITY_ROOT)
    cleanup_old_outputs(OUTPUT_DIR)

    for metric_key, title, color in METRICS:
        output_path = render_metric_chart(idea_rows, metric_key, title, color)
        print(f"[saved] {output_path}")

    exp_dimension_output = render_experiment_dimension_comparison_chart(exp_rows)
    print(f"[saved] {exp_dimension_output}")

    llm_output = render_llm_based_evaluation_chart(idea_rows, exp_rows)
    print(f"[saved] {llm_output}")


if __name__ == "__main__":
    main()
