from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib import rcParams


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["font.size"] = 14
rcParams["axes.linewidth"] = 1.0
rcParams["grid.linewidth"] = 0.5
rcParams["grid.alpha"] = 0.5


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "results" / "human_eval" / "output"
OUTPUT_DIR = REPO_ROOT / "results" / "human_eval"

DIMENSIONS = [
    "Content Completeness",
    "Theme Relevance",
    "Technical Innovation",
    "Technical Feasibility",
    "Application Value",
]

SOURCE_LABELS = {
    "supramas": "SupraMAS",
    "gpt-5": "GPT-5",
    "grok-4": "Grok-4",
    "gpt_researcher": "GPT-Researcher",
    "gemini-2.5-pro-thinking": "Gemini-2.5-Pro-Thinking",
    "gemini-2.0-flash-thinking-exp": "Gemini-2.0-Flash-Thinking-Exp",
    "deepseek-v3": "DeepSeek-V3",
    "deepseek-r1": "DeepSeek-R1",
}

PLOT_SOURCES = [
    "supramas",
    "gpt_researcher",
    "gpt-5",
    "gemini-2.5-pro-thinking",
    "grok-4",
    "deepseek-v3",
]

PLACEHOLDER_SOURCES = {"文件中没有内容", "", None}


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    text = str(value).strip()
    return text if text else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_scores() -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    overall_scores: dict[str, list[float]] = defaultdict(list)
    dim_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    input_files = sorted(INPUT_DIR.glob("result_*.xlsx"))
    if not input_files:
        raise FileNotFoundError(f"No result_*.xlsx files found in: {INPUT_DIR}")

    for path in input_files:

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        worksheet = workbook.active
        rows = worksheet.iter_rows(values_only=True)
        headers = [clean_value(cell) or "" for cell in next(rows)]
        header_index = {header: idx for idx, header in enumerate(headers)}

        required = ["source", "index", "topic", "Evaluation Dimension", "score"]
        missing = [column for column in required if column not in header_index]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")

        current = {"source": None, "index": None, "topic": None}
        for row_number, row in enumerate(rows, start=2):
            record = {
                header: row[idx] if idx < len(row) else None
                for header, idx in header_index.items()
            }

            for key in ("source", "index", "topic"):
                value = clean_value(record.get(key))
                if value is not None:
                    current[key] = value
                else:
                    record[key] = current[key]

            source = clean_value(record.get("source"))
            dimension = clean_value(record.get("Evaluation Dimension"))
            raw_score = clean_value(record.get("score"))

            if source in PLACEHOLDER_SOURCES:
                continue
            if dimension not in DIMENSIONS:
                raise ValueError(
                    f"{path.name} row {row_number} has unexpected dimension: {dimension}"
                )
            if raw_score is None:
                raise ValueError(f"{path.name} row {row_number} has empty score")

            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name} row {row_number} has non-numeric score: {raw_score}"
                ) from exc

            overall_scores[source].append(score)
            dim_scores[source][dimension].append(score)

    return overall_scores, dim_scores


def build_summary(
    overall_scores: dict[str, list[float]],
    dim_scores: dict[str, dict[str, list[float]]],
) -> list[dict[str, object]]:
    summary = []
    for source in PLOT_SOURCES:
        scores = overall_scores.get(source)
        if not scores:
            continue
        item = {
            "source": source,
            "label": SOURCE_LABELS.get(source, source),
            "overall_mean": mean(scores),
            "num_scores": len(scores),
            "dimension_means": {
                dimension: mean(dim_scores[source][dimension]) for dimension in DIMENSIONS
            },
        }
        summary.append(item)

    return summary


def plot_radar(summary: list[dict[str, object]]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    colors = [
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#7E6148",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    for idx, item in enumerate(summary):
        values = [item["dimension_means"][dimension] for dimension in DIMENSIONS]
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2.5 if idx == 0 else 1.8,
            label=item["label"],
            zorder=3,
        )
        ax.fill(angles, values, color=color, alpha=0.08, zorder=2)

    ax.set_thetagrids(np.degrees(angles[:-1]), DIMENSIONS)
    plt.setp(ax.get_xticklabels(), fontsize=16, color="black")
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        label.set_position((1.08, 0))
        if 90 < np.degrees(angle) < 270:
            label.set_ha("right")
        else:
            label.set_ha("left")
        label.set_rotation(0)

    ax.set_rlabel_position(25)
    ax.tick_params(axis="y", colors="black", labelsize=12)
    ax.set_ylim(2.0, 5.0)
    ax.set_yticks([2, 3, 4, 5])
    ax.grid(True)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("#3C5488")
    ax.spines["polar"].set_alpha(0.5)
    ax.spines["polar"].set_linewidth(1.0)

    plt.title("Human Expert Evaluation", size=22, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.10), fontsize=12)
    plt.subplots_adjust(left=0.14, right=0.80, top=0.90, bottom=0.12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"human_eval_radar.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    overall_scores, dim_scores = load_scores()
    summary = build_summary(overall_scores, dim_scores)
    plot_radar(summary)

    print("Overall mean scores by baseline:")
    for item in summary:
        print(
            f"{item['label']}: {item['overall_mean']:.4f} "
            f"(N={item['num_scores']})"
        )

    print("\nDimension means:")
    for item in summary:
        dim_text = ", ".join(
            f"{dimension}={item['dimension_means'][dimension]:.4f}"
            for dimension in DIMENSIONS
        )
        print(f"{item['label']}: {dim_text}")

    summary_path = OUTPUT_DIR / "human_eval_mean_scores.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved radar charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
