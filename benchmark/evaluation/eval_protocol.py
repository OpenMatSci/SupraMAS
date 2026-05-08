'''
python3 /home/cuichaowei/ccw/benchmark_auto/benchmark/eval_exp_hard_mp.py


'''

from benchmark.evaluation.agents import *
import yaml
import os
from benchmark.evaluation.utils import *
from loguru import logger
import multiprocessing as mp
import json
import statistics
import re
import fitz
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark.example.yaml"


def _load_config():
    config_path = Path(os.environ.get("SUPRAMAS_BENCHMARK_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    for key, value in loaded.items():
        if key == "agents" or value == "":
            continue
        os.environ[key] = str(value)
    return loaded


logger.remove()
logger.add(sys.stderr, format="{level} {message}", level="INFO")
config = _load_config()

PROJECT_ROOT = str(BENCHMARK_ROOT)
LABEL_STRATEGIES_PATH = str(BENCHMARK_ROOT / "results" / "idea_foresight" / "label_strategies.json")
RESULT_ROOT = str(BENCHMARK_ROOT / "results" / "protocol_evaluation")
BASELINE_ROOT = str(BENCHMARK_ROOT / "baselines_output" / "protocol_evaluation")
LABEL_EXPERIMENT_PATH = os.path.join(RESULT_ROOT, "label_experiment.json")

EXP_REPRODUCIBILITY = "exp_reproducibility"
EXP_FEASIBILITY = "exp_feasibility"
LEGACY_EVALUATION_FILENAME = "evaluation.jsonl"
EVALUATION_FILENAME_BY_METRIC = {
    EXP_REPRODUCIBILITY: "evaluation_reproducibility.json",
    EXP_FEASIBILITY: "evaluation_feasibility.json",
}

LEGACY_EVALUATOR_TO_METRIC = {
    "EvalExpAgent": EXP_FEASIBILITY,
    "EvalExpAgent_feasibility": EXP_FEASIBILITY,
    "EvalExpAgent_reproducibility": EXP_REPRODUCIBILITY,
    "EvalExpAgent_0410": EXP_REPRODUCIBILITY,
}


def _get_agent_kwargs(primary_key, fallback_key):
    agents_config = config.get("agents", {})
    primary = agents_config.get(primary_key)
    if isinstance(primary, dict):
        return primary
    fallback = agents_config.get(fallback_key, {})
    return fallback if isinstance(fallback, dict) else {}


def _get_eval_agent_kwargs(metric_name):
    if metric_name == EXP_REPRODUCIBILITY:
        return _get_agent_kwargs("eval_exp_reproducibility_agent", "eval_exp_agent")
    if metric_name == EXP_FEASIBILITY:
        return _get_agent_kwargs("eval_exp_feasibility_agent", "eval_exp_agent")
    return _get_agent_kwargs("eval_exp_agent", "eval_agent")


def _normalize_metric_name(value):
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    aliases = {
        "reproducibility": EXP_REPRODUCIBILITY,
        "exp_reproducibility": EXP_REPRODUCIBILITY,
        "experiment_reproducibility": EXP_REPRODUCIBILITY,
        "feasibility": EXP_FEASIBILITY,
        "exp_feasibility": EXP_FEASIBILITY,
        "experiment_feasibility": EXP_FEASIBILITY,
    }
    return aliases.get(normalized)


def _infer_metric_name(row):
    if not isinstance(row, dict):
        return None

    metric_name = _normalize_metric_name(row.get("metric_name"))
    if metric_name:
        return metric_name

    evaluator_name = str(row.get("evaluator", "")).strip()
    return LEGACY_EVALUATOR_TO_METRIC.get(evaluator_name)


def _parse_weight_env(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(str(raw_value).strip())
    except ValueError:
        logger.warning(f"{name}={raw_value} 不是有效浮点数，回退到默认值 {default}")
        return default
    if value < 0:
        logger.warning(f"{name}={raw_value} 小于 0，回退到默认值 {default}")
        return default
    return value


def _get_metric_weights():
    repro_weight = _parse_weight_env("EXP_REPRODUCIBILITY_WEIGHT", 0.5)
    feasibility_weight = _parse_weight_env("EXP_FEASIBILITY_WEIGHT", 0.5)
    total_weight = repro_weight + feasibility_weight
    if total_weight <= 0:
        logger.warning("实验评估权重和 <= 0，回退到默认等权重")
        return {
            EXP_REPRODUCIBILITY: 0.5,
            EXP_FEASIBILITY: 0.5,
        }
    return {
        EXP_REPRODUCIBILITY: repro_weight / total_weight,
        EXP_FEASIBILITY: feasibility_weight / total_weight,
    }


def _resolve_selected_metrics(metric_filter=None):
    if metric_filter is None:
        return [EXP_REPRODUCIBILITY, EXP_FEASIBILITY]

    if isinstance(metric_filter, (list, tuple, set)):
        raw_items = metric_filter
    else:
        raw_items = re.split(r"[,|/]+", str(metric_filter))

    selected_metrics = []
    for item in raw_items:
        normalized = _normalize_metric_name(str(item).strip())
        if normalized and normalized not in selected_metrics:
            selected_metrics.append(normalized)

    if not selected_metrics:
        return [EXP_REPRODUCIBILITY, EXP_FEASIBILITY]

    return selected_metrics


def _normalize_label_idx(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return None


def _sort_rows_by_label_idx(rows):
    return sorted(
        rows,
        key=lambda row: (
            _normalize_label_idx(row.get("label_idx")) is None,
            _normalize_label_idx(row.get("label_idx")) or 0,
        ),
    )


def _save_json_array(save_path, data):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = f"{save_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, save_path)


def _ordered_evaluation_row(row):
    if not isinstance(row, dict):
        return row

    preferred_keys = [
        "label_idx",
        "source_file",
        "baseline_model",
        "metric_name",
        "evaluator",
        "prediction_source",
        "pred_experiment",
        "label_experiment",
        "score",
        "reason",
    ]

    ordered = {}
    for key in preferred_keys:
        if key in row:
            ordered[key] = row[key]

    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def _load_jsonl_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception as e:
                logger.warning(f"跳过无效 JSONL 行: {path}:{line_number}, error: {e}")
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _save_jsonl_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _load_json_array_rows(path):
    if not os.path.exists(path):
        return []

    try:
        rows = _load_json(path)
    except Exception as e:
        logger.warning(f"加载评估 JSON 失败: {path}, error={e}")
        return []

    if not isinstance(rows, list):
        logger.warning(f"评估文件不是 JSON 数组: {path}")
        return []

    return [row for row in rows if isinstance(row, dict)]


def _save_evaluation_rows(path, rows):
    normalized_rows = [_ordered_evaluation_row(row) for row in _dedupe_latest_rows(rows)]
    _save_json_array(path, normalized_rows)


def _evaluation_row_key(row):
    if not isinstance(row, dict):
        return None
    metric_name = _infer_metric_name(row) or "unknown_metric"
    label_idx = _normalize_label_idx(row.get("label_idx"))
    if label_idx is not None:
        return f"label_idx:{label_idx}|metric:{metric_name}"
    source_file = str(row.get("source_file", "")).strip()
    if source_file:
        return f"source_file:{source_file}|metric:{metric_name}"
    prediction_source = str(row.get("prediction_source", "")).strip()
    if prediction_source:
        return f"prediction_source:{prediction_source}|metric:{metric_name}"
    return None


def _dedupe_latest_rows(rows):
    deduped_rows = []
    seen_keys = set()

    for row in reversed(rows):
        key = _evaluation_row_key(row)
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        deduped_rows.append(row)

    deduped_rows.reverse()
    return deduped_rows


def _upsert_evaluation_row(path, new_row):
    new_key = _evaluation_row_key(new_row)
    rows = _load_json_array_rows(path)
    if new_key is not None:
        rows = [row for row in rows if _evaluation_row_key(row) != new_key]
    rows.append(new_row)
    _save_evaluation_rows(path, rows)


def _metric_output_path(output_dir, metric_name):
    filename = EVALUATION_FILENAME_BY_METRIC[metric_name]
    return os.path.join(output_dir, filename)


def _normalize_evaluation_row(row, metric_name=None):
    if not isinstance(row, dict):
        return None

    normalized = dict(row)
    inferred_metric = metric_name or _infer_metric_name(normalized)
    if not inferred_metric:
        return None
    normalized["metric_name"] = inferred_metric
    return _ordered_evaluation_row(normalized)


def _group_rows_by_metric(rows):
    grouped = {
        EXP_REPRODUCIBILITY: [],
        EXP_FEASIBILITY: [],
    }

    for row in _dedupe_latest_rows(rows):
        normalized = _normalize_evaluation_row(row)
        if not normalized:
            continue
        grouped[normalized["metric_name"]].append(normalized)

    return grouped


def _load_existing_evaluation_rows(output_dir):
    paths = {
        metric_name: _metric_output_path(output_dir, metric_name)
        for metric_name in EVALUATION_FILENAME_BY_METRIC
    }
    has_split_files = any(os.path.exists(path) for path in paths.values())

    if has_split_files:
        grouped_rows = {}
        for metric_name, path in paths.items():
            rows = [
                normalized
                for normalized in (
                    _normalize_evaluation_row(row, metric_name=metric_name)
                    for row in _load_json_array_rows(path)
                )
                if normalized
            ]
            grouped_rows[metric_name] = _dedupe_latest_rows(rows)
        return grouped_rows

    legacy_path = os.path.join(output_dir, LEGACY_EVALUATION_FILENAME)
    if not os.path.exists(legacy_path):
        return {
            EXP_REPRODUCIBILITY: [],
            EXP_FEASIBILITY: [],
        }

    legacy_rows = _load_jsonl_rows(legacy_path)
    grouped_rows = _group_rows_by_metric(legacy_rows)
    for metric_name, rows in grouped_rows.items():
        if rows:
            _save_evaluation_rows(paths[metric_name], rows)

    if any(grouped_rows.values()):
        logger.info(f"{output_dir}: 已将旧版 {LEGACY_EVALUATION_FILENAME} 拆分为两个 evaluation_*.json 文件")

    return grouped_rows


def _load_all_split_evaluation_rows(output_dir):
    grouped_rows = _load_existing_evaluation_rows(output_dir)
    merged_rows = []
    for metric_name in (EXP_REPRODUCIBILITY, EXP_FEASIBILITY):
        merged_rows.extend(grouped_rows.get(metric_name, []))
    return _dedupe_latest_rows(merged_rows)


def _read_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    file_content = ""
    try:
        for page in doc:
            file_content += page.get_text("text")
    finally:
        doc.close()
    return file_content


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_label_strategy_records(path):
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON list")

    normalized = {}
    for record_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        label_idx = _normalize_label_idx(row.get("label_idx", record_index))
        source_file = str(row.get("source_file", "")).strip()
        if label_idx is None or not source_file:
            continue
        normalized[label_idx] = {
            "label_idx": label_idx,
            "source_file": source_file,
        }
    return _sort_rows_by_label_idx(list(normalized.values()))


def _load_label_experiments(save_path):
    if not os.path.exists(save_path):
        return []
    try:
        rows = _load_json(save_path)
    except Exception as e:
        logger.warning(f"加载已有 label_experiment.json 失败，将重新抽取: {e}")
        return []
    if not isinstance(rows, list):
        logger.warning(f"{save_path} 不是列表格式，将重新抽取")
        return []

    normalized = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label_idx = _normalize_label_idx(row.get("label_idx"))
        source_file = str(row.get("source_file", "")).strip()
        experiment = str(row.get("experiment", "")).strip()
        if label_idx is None or not source_file:
            continue
        normalized[label_idx] = {
            "label_idx": label_idx,
            "source_file": source_file,
            "experiment": experiment,
        }
    return _sort_rows_by_label_idx(list(normalized.values()))


def extract_label_experiments(label_records, save_path=LABEL_EXPERIMENT_PATH):
    paper_agent = ExtractPaperExpAgent(**_get_agent_kwargs("extract_exp_agent", "extract_agent"))
    saved_rows = _load_label_experiments(save_path)
    saved_map = {
        row["label_idx"]: row
        for row in saved_rows
        if isinstance(row, dict) and row.get("label_idx") is not None
    }

    logger.warning("开始抽取 paper label experiments !!!!!")
    for idx, record in enumerate(label_records, start=1):
        label_idx = record["label_idx"]
        source_file = record["source_file"]

        existing = saved_map.get(label_idx)
        if (
            isinstance(existing, dict)
            and str(existing.get("source_file", "")).strip() == source_file
            and str(existing.get("experiment", "")).strip()
        ):
            logger.info(f"跳过已抽取 label experiment: {idx}/{len(label_records)} - label_{label_idx}")
            continue

        try:
            logger.info(f"抽取 label experiment: {idx}/{len(label_records)} - label_{label_idx}")
            paper_text = _read_pdf_text(source_file)
            response = paper_agent.process({"pdf": paper_text})
            experiment = str(response.get("experiment", "")).strip() if isinstance(response, dict) else ""
            if not experiment:
                raise ValueError("empty experiment extraction")
            saved_map[label_idx] = {
                "label_idx": label_idx,
                "experiment": experiment,
                "source_file": source_file,
            }
            _save_json_array(save_path, _sort_rows_by_label_idx(list(saved_map.values())))
            logger.info(f"label_{label_idx} 抽取完成并已增量保存")
        except Exception as e:
            logger.warning(f"抽取失败: label_{label_idx}, source_file={source_file}, error={e}")

    return _sort_rows_by_label_idx(list(saved_map.values()))


def coalesce_experiment_text(value):
    if isinstance(value, dict):
        for key in ("predicted_experiment", "experiment", "exp", "paper_experiment"):
            if key in value:
                if isinstance(value[key], str):
                    return value[key].strip()
                return json.dumps(value[key], ensure_ascii=False)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "").strip()


def _has_valid_evaluation(row, metric_name=None):
    if not isinstance(row, dict):
        return False

    if metric_name and _infer_metric_name(row) != metric_name:
        return False

    return _extract_score(row) is not None


def _extract_score(row):
    if not isinstance(row, dict):
        return None

    for candidate in (row.get("score"), row.get("exp_evaluation", {}).get("score") if isinstance(row.get("exp_evaluation"), dict) else None):
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate.strip())
            except ValueError:
                continue
    return None


def _compute_metric_average(rows, metric_name):
    scores = []
    for row in rows:
        if not _has_valid_evaluation(row, metric_name=metric_name):
            continue
        score = _extract_score(row)
        if score is not None:
            scores.append(score)
    return round(statistics.mean(scores), 4) if scores else 0.0, len(scores)


def _compute_overall_avg(metric_avgs, metric_counts):
    weights = _get_metric_weights()
    weighted_items = [
        (metric_name, metric_avgs[metric_name], weights.get(metric_name, 0.0))
        for metric_name in (EXP_REPRODUCIBILITY, EXP_FEASIBILITY)
        if metric_counts.get(metric_name, 0) > 0
    ]
    if not weighted_items:
        return 0.0

    total_weight = sum(weight for _, _, weight in weighted_items)
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(avg * weight for _, avg, weight in weighted_items)
    return round(weighted_sum / total_weight, 4)


def _build_summary(label_rows, evaluation_rows, baseline_name, limit):
    target_label_rows = label_rows[:limit] if isinstance(limit, int) and limit > 0 else label_rows
    target_label_indices = {
        row["label_idx"]
        for row in target_label_rows
        if isinstance(row, dict) and row.get("label_idx") is not None
    }

    deduped_rows = []
    for row in _dedupe_latest_rows(evaluation_rows):
        label_idx = _normalize_label_idx(row.get("label_idx"))
        if label_idx in target_label_indices and _has_valid_evaluation(row):
            deduped_rows.append(row)

    metric_avgs = {}
    metric_counts = {}
    for metric_name in (EXP_REPRODUCIBILITY, EXP_FEASIBILITY):
        avg_value, count = _compute_metric_average(deduped_rows, metric_name)
        metric_avgs[metric_name] = avg_value
        metric_counts[metric_name] = count

    return {
        "baseline_model": baseline_name,
        "exp_reproducibility_avg": metric_avgs[EXP_REPRODUCIBILITY],
        "exp_feasibility_avg": metric_avgs[EXP_FEASIBILITY],
        "overall_avg": _compute_overall_avg(metric_avgs, metric_counts),
        "limit": limit,
    }


def evaluate_baseline(agent_framework, label_rows, limit=None, metric_filter=None):
    baseline_dir = os.path.join(BASELINE_ROOT, agent_framework)
    output_dir = os.path.join(RESULT_ROOT, agent_framework)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    evaluation_paths = {
        metric_name: _metric_output_path(output_dir, metric_name)
        for metric_name in EVALUATION_FILENAME_BY_METRIC
    }
    selected_metrics = _resolve_selected_metrics(metric_filter)

    eval_agents = {
        EXP_REPRODUCIBILITY: EvalExpAgent_reproducibility(**_get_eval_agent_kwargs(EXP_REPRODUCIBILITY)),
        EXP_FEASIBILITY: EvalExpAgent_feasibility(**_get_eval_agent_kwargs(EXP_FEASIBILITY)),
    }
    latest_results_by_key = {}

    existing_rows_by_metric = _load_existing_evaluation_rows(output_dir)
    for metric_name, rows in existing_rows_by_metric.items():
        deduped_rows = _dedupe_latest_rows(rows)
        if len(deduped_rows) != len(rows):
            _save_evaluation_rows(evaluation_paths[metric_name], deduped_rows)
            logger.warning(f"{agent_framework}: 检测到重复 {metric_name} 记录，已重写 {os.path.basename(evaluation_paths[metric_name])}")
        for row in deduped_rows:
            label_idx = _normalize_label_idx(row.get("label_idx"))
            if label_idx is not None:
                latest_results_by_key[(label_idx, metric_name)] = row

    target_label_rows = label_rows[:limit] if isinstance(limit, int) and limit > 0 else label_rows
    logger.info(
        f"{agent_framework}: 待评估 label 数量 = {len(target_label_rows)} "
        f"(limit={limit}, metrics={selected_metrics})"
    )

    for record in target_label_rows:
        label_idx = record["label_idx"]
        source_file = record["source_file"]
        label_experiment = str(record.get("experiment", "")).strip()
        prediction_path = os.path.join(baseline_dir, f"label_{label_idx}", "experiment.json")

        if not label_experiment:
            logger.warning(f"{agent_framework}: label_{label_idx} 缺少 label experiment，跳过")
            continue

        if not os.path.exists(prediction_path):
            logger.warning(f"{agent_framework}: 缺少预测实验文件，跳过 label_{label_idx}")
            continue

        try:
            prediction_payload = _load_json(prediction_path)
            pred_experiment = coalesce_experiment_text(prediction_payload)
            if not pred_experiment:
                raise ValueError("empty predicted experiment")

            for metric_name in selected_metrics:
                eval_agent = eval_agents[metric_name]
                existing_row = latest_results_by_key.get((label_idx, metric_name))
                if _has_valid_evaluation(existing_row, metric_name=metric_name):
                    logger.info(f"{agent_framework}: label_{label_idx} 的 {metric_name} 已评估，跳过")
                    continue

                logger.info(f"{agent_framework}: 开始评估 label_{label_idx} 的 {metric_name}")
                exp_evaluation = eval_agent.process({
                    "pred_exp": pred_experiment,
                    "label_exp": label_experiment,
                })
                if not isinstance(exp_evaluation, dict):
                    exp_evaluation = {
                        "score": 0.0,
                        "reason": "LLM evaluation failed.",
                    }

                row = {
                    "label_idx": label_idx,
                    "source_file": source_file,
                    "baseline_model": agent_framework,
                    "metric_name": metric_name,
                    "evaluator": type(eval_agent).__name__,
                    "prediction_source": prediction_path,
                    "pred_experiment": pred_experiment,
                    "label_experiment": label_experiment,
                    "score": float(exp_evaluation.get("score", 0.0)),
                    "reason": str(exp_evaluation.get("reason", "")).strip(),
                }
                row = _ordered_evaluation_row(row)
                _upsert_evaluation_row(evaluation_paths[metric_name], row)
                latest_results_by_key[(label_idx, metric_name)] = row
                logger.info(
                    f"{agent_framework}: label_{label_idx} 的 {metric_name} 评估完成, score={row['score']}"
                )
        except Exception as e:
            logger.warning(f"{agent_framework}: label_{label_idx} 评估失败, error={e}")

    summary = _build_summary(
        label_rows,
        _load_all_split_evaluation_rows(output_dir),
        agent_framework,
        limit,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"{agent_framework}: "
        f"exp_reproducibility_avg={summary['exp_reproducibility_avg']}, "
        f"exp_feasibility_avg={summary['exp_feasibility_avg']}, "
        f"overall_avg={summary['overall_avg']}"
    )


def process_framework(agent_framework, label_rows, limit=None, metric_filter=None):
    selected_metrics = _resolve_selected_metrics(metric_filter)
    logger.warning(f"开始评估 {agent_framework} 的 {selected_metrics} !!!!!")
    evaluate_baseline(agent_framework, label_rows, limit=limit, metric_filter=metric_filter)


def main(limit=None, baseline_filter="", if_mp=True, metric_filter=None):
    os.makedirs(RESULT_ROOT, exist_ok=True)

    label_records = _load_label_strategy_records(LABEL_STRATEGIES_PATH)
    label_rows = extract_label_experiments(label_records, save_path=LABEL_EXPERIMENT_PATH)
    label_rows = [
        row for row in label_rows
        if isinstance(row, dict) and str(row.get("experiment", "")).strip()
    ]

    baseline_ls = sorted(
        [
            name for name in os.listdir(BASELINE_ROOT)
            if os.path.isdir(os.path.join(BASELINE_ROOT, name))
        ]
    )
    if baseline_filter:
        baseline_ls = [name for name in baseline_ls if baseline_filter in name]

    if not baseline_ls:
        logger.warning("未找到可评估的 baseline 目录")
        return

    selected_metrics = _resolve_selected_metrics(metric_filter)

    if if_mp:
        num_processes = min(len(baseline_ls), mp.cpu_count())
        logger.warning(f"使用 {num_processes} 个进程并行评估 {selected_metrics} ...")
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_framework,
                [(agent_framework, label_rows, limit, metric_filter) for agent_framework in baseline_ls],
            )
    else:
        for agent_framework in baseline_ls:
            process_framework(agent_framework, label_rows, limit=limit, metric_filter=metric_filter)

    logger.warning(f"所有 {selected_metrics} 评估完成!")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate protocol feasibility and reproducibility.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N labels.")
    parser.add_argument("--baseline-filter", default="", help="Only evaluate baseline directories containing this string.")
    parser.add_argument("--no-mp", action="store_true", help="Disable multiprocessing.")
    parser.add_argument(
        "--metric",
        default="reproducibility",
        help='Metric filter: "reproducibility", "feasibility", or "reproducibility,feasibility".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        limit=args.limit,
        baseline_filter=args.baseline_filter,
        if_mp=not args.no_mp,
        metric_filter=args.metric,
    )
