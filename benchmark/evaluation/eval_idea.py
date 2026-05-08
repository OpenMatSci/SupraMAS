'''
cd /home/cuichaowei/ccw && python3 /home/cuichaowei/ccw/benchmark_auto/benchmark/eval_idea_hard_mp.py \
  --frameworks supramas deepseek-r1 deepseek-v3 gemini-2.0-flash-thinking-exp gemini-2.5-pro-thinking gpt_researcher grok-4 gpt-5 \
  --recompute-combination \
  --reuse-overlap

--frameworks supramas --recompute-combination --reuse-overlap

cd /home/cuichaowei/ccw && python3 /home/cuichaowei/ccw/benchmark_auto/benchmark/eval_idea_hard_mp.py \
  --frameworks supramas deepseek-v3 gemini-2.5-pro-thinking gpt_researcher grok-4 gpt-5 \
  --recompute-overlap

python /home/cuichaowei/ccw/benchmark_auto/benchmark/eval_idea_hard_mp.py  --frameworks supramas deepseek-v3 gemini-2.5-pro-thinking gpt_researcher grok-4 gpt-5

'''
from benchmark.evaluation.agents import *
import yaml
import os
from benchmark.evaluation.utils import *
from loguru import logger
import multiprocessing as mp
import argparse
import json
import statistics
import traceback
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

TUNING_DIMENSIONS = [
    "Composition tuning",
    "Grain boundary tuning",
    "Interface tuning",
    "Texture tuning",
    "Stress tuning",
]
TUNING_DIMENSION_MAP = {dimension.lower(): dimension for dimension in TUNING_DIMENSIONS}
LABEL_IDEA_GROUP_SIZE = 1

def _canonicalize_tuning_dimension(dimension):
    if not isinstance(dimension, str):
        return ""
    return TUNING_DIMENSION_MAP.get(dimension.strip().lower(), dimension.strip())

def _extract_tunning_method_strategies(strategy_record):
    if not isinstance(strategy_record, dict):
        return []

    tunning_method = strategy_record.get("TunningMethod", {})
    if not isinstance(tunning_method, dict):
        return []

    strategies = tunning_method.get("strategies", [])
    if not isinstance(strategies, list):
        return []

    normalized_strategies = []
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue

        tuning_strategy = str(strategy.get("tuning strategies", "")).strip()
        if not tuning_strategy:
            continue

        normalized_strategies.append({
            "tuning dimensions": _canonicalize_tuning_dimension(strategy.get("tuning dimensions", "")),
            "tuning strategies": tuning_strategy,
            "tuning effects": str(strategy.get("tuning effects", "")).strip(),
        })

    return normalized_strategies


def _flatten_tunning_method_strategies(strategy_records):
    if isinstance(strategy_records, dict):
        strategy_records = [strategy_records]
    if not isinstance(strategy_records, list):
        return []

    flattened_strategies = []
    for strategy_record in strategy_records:
        flattened_strategies.extend(_extract_tunning_method_strategies(strategy_record))

    return flattened_strategies


def _append_dimension_strategy(payload, dimension, strategy_text, tuning_effects=""):
    canonical_dimension = _canonicalize_tuning_dimension(dimension)
    strategy_text = str(strategy_text).strip()
    tuning_effects = str(tuning_effects).strip()
    if canonical_dimension not in payload or not strategy_text:
        return

    strategy_entry = {
        "tuning strategies": strategy_text,
        "tuning effects": tuning_effects,
    }
    if strategy_entry not in payload[canonical_dimension]:
        payload[canonical_dimension].append(strategy_entry)


def _dedupe_preserve_order(items):
    deduped = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _build_pred_dimension_payload(strategy_source):
    payload = {dimension: [] for dimension in TUNING_DIMENSIONS}

    if isinstance(strategy_source, dict) and "TunningMethod" not in strategy_source:
        for dimension in TUNING_DIMENSIONS:
            value = strategy_source.get(dimension, "")
            if isinstance(value, str):
                _append_dimension_strategy(payload, dimension, value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _append_dimension_strategy(
                            payload,
                            dimension,
                            item.get("tuning strategies", ""),
                            item.get("tuning effects", ""),
                        )
                    else:
                        _append_dimension_strategy(payload, dimension, item)
        return payload

    for strategy in _flatten_tunning_method_strategies(strategy_source):
        _append_dimension_strategy(
            payload,
            strategy.get("tuning dimensions", ""),
            strategy.get("tuning strategies", ""),
            strategy.get("tuning effects", ""),
        )

    return payload


def _get_label_record_index(record, record_index):
    label_record_index = record_index + 1
    if isinstance(record, dict):
        raw_label_idx = record.get("label_idx")
        if isinstance(raw_label_idx, int):
            label_record_index = raw_label_idx
        elif isinstance(raw_label_idx, str) and raw_label_idx.strip().isdigit():
            label_record_index = int(raw_label_idx.strip())
    return label_record_index


def _build_label_strategy_library(strategy_source):
    if isinstance(strategy_source, dict):
        strategy_source = [strategy_source]
    if not isinstance(strategy_source, list):
        return []

    normalized_library = []
    entry_by_key = {}

    for record_index, record in enumerate(strategy_source):
        candidate_records = []
        source_file = str(record.get("source_file", "")).strip() if isinstance(record, dict) else ""
        label_record_index = _get_label_record_index(record, record_index)

        if isinstance(record, dict) and "TunningMethod" in record:
            tunning_method = record.get("TunningMethod", {})
            raw_strategies = tunning_method.get("strategies", []) if isinstance(tunning_method, dict) else []
            if isinstance(raw_strategies, list):
                for strategy_index, strategy in enumerate(raw_strategies):
                    if not isinstance(strategy, dict):
                        continue
                    candidate_records.append({
                        "tuning dimensions": _canonicalize_tuning_dimension(strategy.get("tuning dimensions", "")),
                        "tuning strategies": str(strategy.get("tuning strategies", "")).strip(),
                        "tuning effects": str(strategy.get("tuning effects", "")).strip(),
                        "source_ref": {
                            "label_record_index": label_record_index,
                            "strategy_index_in_label": strategy_index + 1,
                            "source_file": source_file,
                        },
                    })
        elif isinstance(record, dict):
            candidate_records.append({
                "tuning dimensions": _canonicalize_tuning_dimension(record.get("tuning dimensions", "")),
                "tuning strategies": str(record.get("tuning strategies", "")).strip(),
                "tuning effects": str(record.get("tuning effects", "")).strip(),
                "source_ref": {
                    "label_record_index": label_record_index,
                    "strategy_index_in_label": 1,
                    "source_file": source_file,
                },
            })

        for strategy in candidate_records:
            dimension = strategy["tuning dimensions"]
            strategy_text = strategy["tuning strategies"]
            tuning_effects = strategy["tuning effects"]
            if dimension not in TUNING_DIMENSIONS or not strategy_text:
                continue

            unique_key = (dimension, strategy_text, tuning_effects)
            source_ref = strategy["source_ref"]
            if unique_key not in entry_by_key:
                entry = {
                    "label_id": (
                        f"label_{source_ref['label_record_index']}"
                        f"_strategy_{source_ref['strategy_index_in_label']}"
                    ),
                    "tuning dimensions": dimension,
                    "tuning strategies": strategy_text,
                    "tuning effects": tuning_effects,
                    "source_refs": [source_ref],
                }
                entry_by_key[unique_key] = entry
                normalized_library.append(entry)
            else:
                entry = entry_by_key[unique_key]
                if source_ref not in entry["source_refs"]:
                    entry["source_refs"].append(source_ref)

    return normalized_library


def _chunk_list(items, chunk_size):
    if chunk_size <= 0:
        return [items] if items else []
    return [items[idx: idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _build_label_idea_groups(label_strategies, label_strategy_library, group_size=LABEL_IDEA_GROUP_SIZE):
    if isinstance(label_strategies, dict):
        label_strategies = [label_strategies]
    if not isinstance(label_strategies, list):
        return []

    label_indices = []
    seen = set()
    for record_index, record in enumerate(label_strategies):
        label_idx = _get_label_record_index(record, record_index)
        if label_idx in seen:
            continue
        seen.add(label_idx)
        label_indices.append(label_idx)

    groups = []
    for group_index, label_idx_group in enumerate(_chunk_list(label_indices, group_size), start=1):
        label_idx_set = set(label_idx_group)
        group_library = []
        for strategy in label_strategy_library:
            source_refs = strategy.get("source_refs", [])
            if any(
                isinstance(source_ref, dict) and source_ref.get("label_record_index") in label_idx_set
                for source_ref in source_refs
            ):
                group_library.append(strategy)

        groups.append({
            "group_index": group_index,
            "label_indices": label_idx_group,
            "label_strategies": group_library,
        })

    return groups


def _build_label_idea_strategy_stats(label_strategies):
    if isinstance(label_strategies, dict):
        label_strategies = [label_strategies]
    if not isinstance(label_strategies, list):
        return {}

    label_idea_stats = {}
    for record_index, record in enumerate(label_strategies):
        if not isinstance(record, dict):
            continue

        label_idx = _get_label_record_index(record, record_index)
        source_file = str(record.get("source_file", "")).strip()
        tunning_method = record.get("TunningMethod", {})
        raw_strategies = tunning_method.get("strategies", []) if isinstance(tunning_method, dict) else []
        normalized_strategies = []

        if isinstance(raw_strategies, list):
            for strategy_index, strategy in enumerate(raw_strategies):
                if not isinstance(strategy, dict):
                    continue
                dimension = _canonicalize_tuning_dimension(strategy.get("tuning dimensions", ""))
                strategy_text = str(strategy.get("tuning strategies", "")).strip()
                if dimension not in TUNING_DIMENSIONS or not strategy_text:
                    continue

                normalized_strategies.append({
                    "strategy_index_in_label": strategy_index + 1,
                    "tuning dimensions": dimension,
                    "tuning strategies": strategy_text,
                    "tuning effects": str(strategy.get("tuning effects", "")).strip(),
                })

        label_idea_stats[label_idx] = {
            "label_idx": label_idx,
            "source_file": source_file,
            "total_strategy_num": len(normalized_strategies),
            "strategies": normalized_strategies,
        }

    return label_idea_stats


def _build_label_idea_stat_groups(label_idea_stats, group_size=LABEL_IDEA_GROUP_SIZE):
    if not isinstance(label_idea_stats, dict):
        return []

    valid_label_indices = [
        label_idx
        for label_idx, stats in sorted(label_idea_stats.items())
        if isinstance(stats, dict) and stats.get("total_strategy_num", 0) > 0
    ]
    groups = []
    for group_index, label_idx_group in enumerate(_chunk_list(valid_label_indices, group_size), start=1):
        groups.append({
            "group_index": group_index,
            "label_indices": label_idx_group,
            "label_ideas": [
                {
                    "label_idx": label_idx,
                    "source_file": label_idea_stats[label_idx]["source_file"],
                    "label_strategy_num": label_idea_stats[label_idx]["total_strategy_num"],
                    "strategies": label_idea_stats[label_idx]["strategies"],
                }
                for label_idx in label_idx_group
            ],
        })

    return groups


def _build_label_strategy_lookup(label_strategy_library):
    return {
        strategy["label_id"]: {
            "label_id": strategy["label_id"],
            "tuning dimensions": strategy["tuning dimensions"],
            "tuning strategies": strategy["tuning strategies"],
            "tuning effects": strategy["tuning effects"],
            "source_refs": strategy.get("source_refs", []),
        }
        for strategy in label_strategy_library
    }


def _collect_label_strategy_details(matched_label_ids, label_strategy_lookup):
    matched_details = []
    for label_id in matched_label_ids:
        detail = label_strategy_lookup.get(label_id)
        if detail:
            matched_details.append(detail)
    return matched_details


def _merge_overlap_chunk_results(pred_dimension_payload, chunk_results):
    merged_dimension_details = {}
    merged_matched_label_ids = []
    merged_reason_summaries = []

    for dimension in TUNING_DIMENSIONS:
        matched_ids = []
        positive_analyses = []
        fallback_analysis = ""
        score = 0

        for chunk_result in chunk_results:
            if not isinstance(chunk_result, dict):
                continue

            detail = chunk_result.get("dimension_details", {}).get(dimension, {})
            detail = detail if isinstance(detail, dict) else {}

            matched_ids.extend(detail.get("matched_label_ids", []))

            analysis = str(detail.get("analysis", "")).strip()
            if detail.get("score", 0) == 1:
                score = 1
                if analysis and analysis not in positive_analyses:
                    positive_analyses.append(analysis)
            elif not fallback_analysis and analysis:
                fallback_analysis = analysis

            merged_matched_label_ids.extend(chunk_result.get("matched_label_ids", []))
            reason_summary = str(chunk_result.get("reason_summary", "")).strip()
            if reason_summary and reason_summary not in merged_reason_summaries:
                merged_reason_summaries.append(reason_summary)

        merged_dimension_details[dimension] = {
            "score": score,
            "matched_label_ids": _dedupe_preserve_order(matched_ids),
            "analysis": " | ".join(positive_analyses) if positive_analyses else fallback_analysis,
        }

    merged_matched_label_ids = _dedupe_preserve_order(
        merged_matched_label_ids + [
            matched_id
            for detail in merged_dimension_details.values()
            for matched_id in detail["matched_label_ids"]
        ]
    )
    return {
        "label_side_score": len(merged_matched_label_ids),
        "dimension_details": merged_dimension_details,
        "matched_label_ids": merged_matched_label_ids,
        "reason_summary": "\n".join(merged_reason_summaries),
    }


def _build_empty_overlap_result(pred_dimension_payload, label_strategy_library):
    label_strategy_lookup = _build_label_strategy_lookup(label_strategy_library)
    return {
        "label_side_score": 0,
        "pred_strategy_num": sum(len(strategies) for strategies in pred_dimension_payload.values()),
        "pred_dimension_num": sum(1 for strategies in pred_dimension_payload.values() if strategies),
        "label_strategy_num": len(label_strategy_library),
        "label_strategy_lookup": label_strategy_lookup,
        "dimension_details": {
            dimension: {
                "score": 0,
                "matched_label_ids": [],
                "analysis": "No predicted strategy provided for this dimension."
                if not pred_dimension_payload.get(dimension)
                else "No similar label strategy was found.",
            }
            for dimension in TUNING_DIMENSIONS
        },
        "matched_label_ids": [],
        "matched_label_strategies": [],
        "reason_summary": "",
    }


def _calculate_strategy_overlap_score(overlap_result):
    if not isinstance(overlap_result, dict):
        return 0.0
    label_side_score = overlap_result.get("label_side_score", 0)
    label_strategy_num = overlap_result.get("label_strategy_num", 0)
    if not isinstance(label_side_score, (int, float)):
        return 0.0
    if not isinstance(label_strategy_num, (int, float)) or float(label_strategy_num) <= 0:
        return 0.0
    return round(float(label_side_score) / float(label_strategy_num), 4)


def _extract_strategy_combination_score(combination_result):
    if not isinstance(combination_result, dict):
        return 0.0

    raw_score = combination_result.get("strategy_combination_score")
    if isinstance(raw_score, (int, float)):
        return round(float(raw_score), 4)

    raw_score = combination_result.get("score", 0.0)
    if isinstance(raw_score, (int, float)):
        return round(float(raw_score), 4)

    return 0.0


def _calculate_overall_score(overlap_result, combination_result):
    overlap_score = _calculate_strategy_overlap_score(overlap_result)
    combination_score = _extract_strategy_combination_score(combination_result)
    return round((overlap_score + combination_score) / 2, 4)


def _build_evaluation_summary(rows):
    overlap_scores = []
    combination_scores = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        strategy_overlap = row.get("strategy_overlap", {})
        strategy_combination = row.get("strategy_combination", {})

        overlap_score = strategy_overlap.get("score")
        if not isinstance(overlap_score, (int, float)):
            overlap_score = _calculate_strategy_overlap_score(strategy_overlap)
        combination_score = _extract_strategy_combination_score(strategy_combination)

        overlap_scores.append(float(overlap_score))
        if isinstance(combination_score, (int, float)):
            combination_scores.append(float(combination_score))

    strategy_overlap_avg = round(statistics.mean(overlap_scores), 4) if overlap_scores else 0.0
    strategy_combination_avg = round(statistics.mean(combination_scores), 4) if combination_scores else 0.0
    overall_avg = round((strategy_overlap_avg + strategy_combination_avg) / 2, 4)

    return {
        "topic_num": len(rows),
        "strategy_overlap_avg": strategy_overlap_avg,
        "strategy_combination_avg": strategy_combination_avg,
        "overall_avg": overall_avg,
    }


def _save_json_array(save_path, data):
    tmp_path = f"{save_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, save_path)


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
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _dedupe_latest_topic_rows(rows):
    deduped_rows = []
    seen_topics = set()

    for row in reversed(rows):
        topic = row.get("topic") if isinstance(row, dict) else None
        if isinstance(topic, str) and topic:
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
        deduped_rows.append(row)

    deduped_rows.reverse()
    return deduped_rows


def _upsert_evaluation_row(path, new_row):
    topic = new_row.get("topic") if isinstance(new_row, dict) else None
    rows = _load_jsonl_rows(path)
    if isinstance(topic, str) and topic:
        rows = [row for row in rows if row.get("topic") != topic]
    rows.append(new_row)
    _save_jsonl_rows(path, _dedupe_latest_topic_rows(rows))


def extract_labels(pdf_ls, save_path=None):
    paper_agent = ExtractPaperAgent(**config['agents']['extract_agent'])
    labels = []
    processed_files = set()

    if save_path and os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                saved_labels = json.load(f)
            if isinstance(saved_labels, list):
                labels = saved_labels
                processed_files = {
                    label.get("source_file")
                    for label in labels
                    if isinstance(label, dict) and label.get("source_file")
                }
                logger.warning(
                    f"检测到已有 label_strategies.json，已加载 {len(labels)} 条记录，将按 source_file 增量续跑"
                )
            else:
                logger.warning(f"{save_path} 不是列表格式，将重新提取标签")
        except Exception as e:
            logger.warning(f"加载已有 label_strategies.json 失败，将重新提取标签: {e}")

    logger.warning(f"处理论文文件，提取调控策略标签 !!!!!")
    # 提取每个标签的相关信息
    for idx, pdf in enumerate(pdf_ls):
        if pdf in processed_files:
            logger.info(f"跳过已保存论文文件: {idx+1}/{len(pdf_ls)} - {pdf}")
            continue
        try:
            logger.info(f"处理论文文件: {idx+1}/{len(pdf_ls)} - {pdf}")
            doc = fitz.open(pdf)
            file_content = ""
            for page in doc:
                file_content += page.get_text("text")
            doc.close()
            template = paper_agent.process({"pdf":file_content})["template"]
            label = json.loads(template)
            label["source_file"] = pdf
            labels.append(label)
            processed_files.add(pdf)
            if save_path:
                _save_json_array(save_path, labels)
                logger.info(f"增量保存完成: {save_path}")
            logger.info(f"处理成功!")
        except Exception as e:
            logger.warning(f"处理失败: {pdf}, error: {e}")
    return labels


def idea2strategy(idea):
    
    parser_agent = Idea2StrategyAgent(**config['agents']['idea2strategy_agent'])
    idea_ours = []

    # 解析idea的调控策略
    input_dict = {'idea': idea}
    pred_strategy = parser_agent.process(input_dict)['templates']
    # idea_ours.append(idea_strategy)
        
    return pred_strategy


def evaluate_strategy_overlap(pred_strategies,label_strategies):
    pred_dimension_payload = _build_pred_dimension_payload(pred_strategies)
    label_strategy_library = _build_label_strategy_library(label_strategies)
    label_strategy_lookup = _build_label_strategy_lookup(label_strategy_library)
    label_idea_groups = _build_label_idea_groups(
        label_strategies,
        label_strategy_library,
        group_size=LABEL_IDEA_GROUP_SIZE,
    )

    if not label_strategy_library:
        logger.warning("label 策略库为空，返回空结果")
        empty_result = _build_empty_overlap_result(pred_dimension_payload, label_strategy_library)
        empty_result.update({
            "pred_strategy_num": sum(len(strategies) for strategies in pred_dimension_payload.values()),
            "pred_dimension_num": sum(1 for strategies in pred_dimension_payload.values() if strategies),
            "label_strategy_num": len(label_strategy_library),
            "label_group_size": LABEL_IDEA_GROUP_SIZE,
            "label_group_num": 0,
            "label_strategy_lookup": label_strategy_lookup,
            "matched_label_strategies": [],
        })
        return empty_result

    logger.info(
        f"strategy_overlap 分组串行评估: group_size={LABEL_IDEA_GROUP_SIZE}, "
        f"group_num={len(label_idea_groups)}"
    )

    def _run_overlap_group(group):
        group_agent = EvalOverlapAgent(**config['agents']['eval_agent'])
        group_result = group_agent.process({
            "input_strategies": pred_dimension_payload,
            "label_strategies": group["label_strategies"],
        })
        if not group_result:
            logger.warning(
                f"strategy_overlap group {group['group_index']} 评估失败，返回空 chunk 结果"
            )
            return _build_empty_overlap_result(pred_dimension_payload, group["label_strategies"])
        return group_result

    chunk_results = []
    for group in label_idea_groups:
        try:
            chunk_results.append(_run_overlap_group(group))
        except Exception as e:
            logger.warning(
                f"strategy_overlap group {group['group_index']} 串行评估异常: {e}"
            )
            chunk_results.append(
                _build_empty_overlap_result(pred_dimension_payload, group["label_strategies"])
            )

    result = _merge_overlap_chunk_results(pred_dimension_payload, chunk_results)
    if not result:
        logger.warning("strategy_overlap chunk 合并失败，返回空结果")
        empty_result = _build_empty_overlap_result(pred_dimension_payload, label_strategy_library)
        empty_result.update({
            "pred_strategy_num": sum(len(strategies) for strategies in pred_dimension_payload.values()),
            "pred_dimension_num": sum(1 for strategies in pred_dimension_payload.values() if strategies),
            "label_strategy_num": len(label_strategy_library),
            "label_group_size": LABEL_IDEA_GROUP_SIZE,
            "label_group_num": len(label_idea_groups),
            "label_strategy_lookup": label_strategy_lookup,
            "matched_label_strategies": [],
        })
        return empty_result

    matched_label_strategies = _collect_label_strategy_details(
        result["matched_label_ids"],
        label_strategy_lookup,
    )
    return {
        "label_side_score": result["label_side_score"],
        "pred_strategy_num": sum(len(strategies) for strategies in pred_dimension_payload.values()),
        "pred_dimension_num": sum(1 for strategies in pred_dimension_payload.values() if strategies),
        "label_strategy_num": len(label_strategy_library),
        "label_group_size": LABEL_IDEA_GROUP_SIZE,
        "label_group_num": len(label_idea_groups),
        "label_strategy_lookup": label_strategy_lookup,
        "dimension_details": result["dimension_details"],
        "matched_label_ids": result["matched_label_ids"],
        "matched_label_strategies": matched_label_strategies,
        "reason_summary": result["reason_summary"],
    }

def evaluate_strategy_combination(pred_strategies, overlap_result, label_strategies):
    label_idea_stats = _build_label_idea_strategy_stats(label_strategies)
    label_idea_groups = _build_label_idea_stat_groups(
        label_idea_stats,
        group_size=LABEL_IDEA_GROUP_SIZE,
    )
    matched_by_label = {}
    per_label_scores = []

    matched_label_strategies = overlap_result.get("matched_label_strategies", []) if isinstance(overlap_result, dict) else []
    for matched_strategy in matched_label_strategies:
        if not isinstance(matched_strategy, dict):
            continue

        label_id = str(matched_strategy.get("label_id", "")).strip()
        for source_ref in matched_strategy.get("source_refs", []):
            if not isinstance(source_ref, dict):
                continue

            label_idx = source_ref.get("label_record_index")
            strategy_index_in_label = source_ref.get("strategy_index_in_label")
            if not isinstance(label_idx, int) or not isinstance(strategy_index_in_label, int):
                continue
            if label_idx not in label_idea_stats:
                continue

            label_bucket = matched_by_label.setdefault(label_idx, {
                "matched_strategy_indices": set(),
                "matched_label_ids": set(),
            })
            label_bucket["matched_strategy_indices"].add(strategy_index_in_label)
            if label_id:
                label_bucket["matched_label_ids"].add(label_id)

    if not label_idea_groups:
        return {
            "score": 0.0,
            "coverage_score": 0.0,
            "strategy_combination_score": 0.0,
            "best_label_idx": None,
            "matched_strategy_num": 0,
            "label_strategy_num": 0,
            "best_label_source_file": "",
            "best_label_strategies": [],
            "best_label_matched_strategy_indices": [],
            "best_label_matched_label_ids": [],
            "best_coverage_label_idx": None,
            "best_coverage_source_file": "",
            "best_coverage_label_strategies": [],
            "best_coverage_matched_strategy_indices": [],
            "best_coverage_matched_label_ids": [],
            "best_combination_label_idx": None,
            "best_combination_source_file": "",
            "best_combination_label_strategies": [],
            "best_combination_matched_strategy_indices": [],
            "best_combination_matched_label_ids": [],
            "per_label_scores": [],
            "coverage_reason": "No valid label idea with non-empty strategies was found.",
            "reason": "No valid label idea with non-empty strategies was found.",
        }

    best_coverage_result = None
    for label_idx, stats in sorted(label_idea_stats.items()):
        total_strategy_num = stats["total_strategy_num"]
        if total_strategy_num <= 0:
            continue

        label_bucket = matched_by_label.get(label_idx, {
            "matched_strategy_indices": set(),
            "matched_label_ids": set(),
        })
        matched_strategy_indices = sorted(label_bucket["matched_strategy_indices"])
        matched_strategy_num = len(matched_strategy_indices)
        score = matched_strategy_num / total_strategy_num

        label_result = {
            "label_idx": label_idx,
            "matched_strategy_num": matched_strategy_num,
            "label_strategy_num": total_strategy_num,
            "coverage_score": round(score, 4),
            "strategy_combination_score": 0.0,
        }
        per_label_scores.append(label_result)

        ranking_key = (score, matched_strategy_num, -label_idx)
        if matched_strategy_num > 0 and (
            best_coverage_result is None or ranking_key > best_coverage_result["ranking_key"]
        ):
            best_coverage_result = {
                "ranking_key": ranking_key,
                "label_idx": label_idx,
                "coverage_score": round(score, 4),
                "matched_strategy_num": matched_strategy_num,
                "label_strategy_num": total_strategy_num,
                "source_file": stats["source_file"],
                "strategies": stats["strategies"],
                "matched_strategy_indices": matched_strategy_indices,
                "matched_label_ids": sorted(label_bucket["matched_label_ids"]),
            }

    if best_coverage_result is None:
        coverage_score = 0.0
        coverage_reason = "No matched label strategies were found in strategy_overlap, so coverage_score is 0."
    else:
        coverage_score = best_coverage_result["coverage_score"]
        coverage_reason = (
            f"coverage_score = matched_strategy_num / label_strategy_num = "
            f"{best_coverage_result['matched_strategy_num']} / {best_coverage_result['label_strategy_num']} = "
            f"{best_coverage_result['coverage_score']}"
        )

    logger.info(
        f"strategy_combination 分组串行评估: group_size={LABEL_IDEA_GROUP_SIZE}, "
        f"group_num={len(label_idea_groups)}"
    )

    def _run_combination_group(group):
        combination_agent = EvalCombinationAgent(**config['agents']['eval_agent'])
        group_result = combination_agent.process({
            "input_strategies": pred_strategies,
            "label_idea_groups": group["label_ideas"],
        })
        if not group_result:
            logger.warning(
                f"strategy_combination group {group['group_index']} 评估失败，返回空 chunk 结果"
            )
            return {"label_results": []}
        return group_result

    def _print_combination_group_result(group, group_output, status="done"):
        printable_output = group_output if isinstance(group_output, dict) else {"label_results": []}
        print(
            f"[strategy_combination][group {group['group_index']}/{len(label_idea_groups)}]"
            f"[label_indices={group['label_indices']}][{status}] "
            f"{json.dumps(printable_output, ensure_ascii=False)}",
            flush=True,
        )

    group_outputs = []
    for group in label_idea_groups:
        try:
            group_output = _run_combination_group(group)
            group_outputs.append(group_output)
            _print_combination_group_result(group, group_output)
        except Exception as e:
            logger.warning(
                f"strategy_combination group {group['group_index']} 串行评估异常: {e}"
            )
            group_output = {"label_results": []}
            group_outputs.append(group_output)
            _print_combination_group_result(group, group_output, status="error")

    combination_results_by_label = {
        label_idx: {
            "strategy_combination_score": 0.0,
            "reason": (
                "LLM-based strategy combination evaluation failed for this label idea, "
                "so strategy_combination_score defaults to 0."
            ),
        }
        for label_idx in label_idea_stats
        if label_idea_stats[label_idx]["total_strategy_num"] > 0
    }

    for group_output in group_outputs:
        if not isinstance(group_output, dict):
            continue
        label_results = group_output.get("label_results", [])
        if not isinstance(label_results, list):
            continue

        for item in label_results:
            if not isinstance(item, dict):
                continue
            label_idx = item.get("label_idx")
            if not isinstance(label_idx, int) or label_idx not in combination_results_by_label:
                continue

            raw_score = item.get("strategy_combination_score")
            score = round(float(raw_score), 4) if isinstance(raw_score, (int, float)) else 0.0
            score = max(0.0, min(1.0, score))
            reason_text = str(item.get("reason", "")).strip()
            combination_results_by_label[label_idx] = {
                "strategy_combination_score": score,
                "reason": reason_text or combination_results_by_label[label_idx]["reason"],
            }

    best_combination_result = None
    for label_result in per_label_scores:
        label_idx = label_result["label_idx"]
        stats = label_idea_stats[label_idx]
        label_bucket = matched_by_label.get(label_idx, {
            "matched_strategy_indices": set(),
            "matched_label_ids": set(),
        })

        combination_eval = combination_results_by_label.get(label_idx, {})
        combination_score = combination_eval.get("strategy_combination_score", 0.0)
        combination_reason = combination_eval.get(
            "reason",
            (
                "LLM-based strategy combination evaluation failed for this label idea, "
                "so strategy_combination_score defaults to 0."
            ),
        )

        label_result["strategy_combination_score"] = combination_score

        ranking_key = (
            combination_score,
            label_result["coverage_score"],
            label_result["matched_strategy_num"],
            -label_idx,
        )
        if best_combination_result is None or ranking_key > best_combination_result["ranking_key"]:
            best_combination_result = {
                "ranking_key": ranking_key,
                "label_idx": label_idx,
                "strategy_combination_score": combination_score,
                "reason": combination_reason,
                "matched_strategy_num": label_result["matched_strategy_num"],
                "label_strategy_num": label_result["label_strategy_num"],
                "coverage_score": label_result["coverage_score"],
                "source_file": stats["source_file"],
                "strategies": stats["strategies"],
                "matched_strategy_indices": sorted(label_bucket["matched_strategy_indices"]),
                "matched_label_ids": sorted(label_bucket["matched_label_ids"]),
            }

    if best_combination_result is None:
        best_combination_result = {
            "label_idx": None,
            "strategy_combination_score": 0.0,
            "reason": "No valid label idea with non-empty strategies was found.",
            "matched_strategy_num": 0,
            "label_strategy_num": 0,
            "coverage_score": 0.0,
            "source_file": "",
            "strategies": [],
            "matched_strategy_indices": [],
            "matched_label_ids": [],
        }

    return {
        "score": best_combination_result["strategy_combination_score"],
        "coverage_score": coverage_score,
        "strategy_combination_score": best_combination_result["strategy_combination_score"],
        "best_label_idx": best_combination_result["label_idx"],
        "matched_strategy_num": (
            best_coverage_result["matched_strategy_num"] if best_coverage_result is not None else 0
        ),
        "label_strategy_num": (
            best_coverage_result["label_strategy_num"] if best_coverage_result is not None else 0
        ),
        "best_label_source_file": best_combination_result["source_file"],
        "best_label_strategies": best_combination_result["strategies"],
        "best_label_matched_strategy_indices": best_combination_result["matched_strategy_indices"],
        "best_label_matched_label_ids": best_combination_result["matched_label_ids"],
        "best_coverage_label_idx": (
            best_coverage_result["label_idx"] if best_coverage_result is not None else None
        ),
        "best_coverage_source_file": (
            best_coverage_result["source_file"] if best_coverage_result is not None else ""
        ),
        "best_coverage_label_strategies": (
            best_coverage_result["strategies"] if best_coverage_result is not None else []
        ),
        "best_coverage_matched_strategy_indices": (
            best_coverage_result["matched_strategy_indices"] if best_coverage_result is not None else []
        ),
        "best_coverage_matched_label_ids": (
            best_coverage_result["matched_label_ids"] if best_coverage_result is not None else []
        ),
        "best_combination_label_idx": best_combination_result["label_idx"],
        "best_combination_source_file": best_combination_result["source_file"],
        "best_combination_label_strategies": best_combination_result["strategies"],
        "best_combination_matched_strategy_indices": best_combination_result["matched_strategy_indices"],
        "best_combination_matched_label_ids": best_combination_result["matched_label_ids"],
        "per_label_scores": per_label_scores,
        "coverage_reason": coverage_reason,
        "reason": best_combination_result["reason"],
    }



def get_idea_labels(header_dir,topic_header,agent_framework):
    
    logger.warning(f"开始评估的idea前瞻性 !!!!!")
    topic_name = os.path.basename(os.path.normpath(topic_header))
    extract_dir = os.path.join(header_dir, agent_framework, "extract_strategies")
    strategy_output_path = os.path.join(extract_dir, "pred_strategies.json")
    cache_payload = {}
    pred_strategies = None
    strategy_section_keys = {"MaterialSystem", "TunningMethod", "ExperimentTech", "Application"}

    # 加载idea
    if os.path.exists(strategy_output_path):
        with open(strategy_output_path,"r") as f:
            loaded_payload = json.load(f)

        if isinstance(loaded_payload, dict) and topic_name in loaded_payload:
            logger.warning(f"已存在 {topic_name} 的评估结果，直接加载评估结果 !!!!!")
            pred_strategies = loaded_payload[topic_name]
            cache_payload = loaded_payload
        elif isinstance(loaded_payload, dict) and strategy_section_keys.issubset(loaded_payload.keys()):
            logger.warning("检测到旧版单-topic 缓存格式，将按当前 topic 重新写入新格式 !!!!!")
        elif isinstance(loaded_payload, dict):
            cache_payload = loaded_payload

    if pred_strategies is None:
        idea_path = topic_header + "/idea.json"
        with open(idea_path,"r") as f:
            idea = json.load(f)["idea"]
        pred_strategies = idea2strategy(idea)
        cache_payload[topic_name] = pred_strategies
        os.makedirs(extract_dir, exist_ok=True)
        with open(strategy_output_path,"w") as f:
            json.dump(cache_payload, f, ensure_ascii=False, indent=2)


    # 加载labels
    label_path = header_dir + "label_strategies.json"
    with open(label_path,"r") as f:
        label_strategies = json.load(f)
    
    return pred_strategies,label_strategies


def _has_new_combination_schema(row):
    if not isinstance(row, dict):
        return False

    strategy_combination = row.get("strategy_combination", {})
    if not isinstance(strategy_combination, dict):
        return False

    if "coverage_score" not in strategy_combination:
        return False

    return (
        "strategy_combination_score" in strategy_combination
        or isinstance(strategy_combination.get("score"), (int, float))
    )


def _has_reusable_overlap_schema(row):
    if not isinstance(row, dict):
        return False

    strategy_overlap = row.get("strategy_overlap", {})
    if not isinstance(strategy_overlap, dict):
        return False

    required_keys = {
        "score",
        "label_side_score",
        "matched_label_ids",
        "matched_label_strategies",
        "dimension_details",
    }
    return required_keys.issubset(strategy_overlap.keys())


def main(agent_framework, limit=None, recompute_combination=False, reuse_overlap=False, recompute_overlap=False):
    header_dir = str(BENCHMARK_ROOT / "results" / "idea_foresight") + "/"
    topic_dir = str(BENCHMARK_ROOT / "baselines_output" / "idea_foresight" / agent_framework) + "/"

    topic_ls = sorted(
        [i for i in os.listdir(topic_dir) if os.path.isdir(os.path.join(topic_dir, i))]
    )
    if isinstance(limit, int) and limit > 0:
        topic_ls = topic_ls[:limit]
    evaluation_path = os.path.join(header_dir, f"{agent_framework}/evaluation.jsonl")
    summary_path = os.path.join(header_dir, f"{agent_framework}/evaluation_summary.json")
    latest_results_by_topic = {}

    if os.path.exists(evaluation_path):
        existing_rows = _load_jsonl_rows(evaluation_path)
        deduped_rows = _dedupe_latest_topic_rows(existing_rows)
        if len(deduped_rows) != len(existing_rows):
            _save_jsonl_rows(evaluation_path, deduped_rows)
            logger.warning("检测到重复 topic，已按最新记录去重重写 evaluation.jsonl")

        for row in deduped_rows:
            topic_name = row.get("topic")
            if isinstance(topic_name, str) and topic_name:
                latest_results_by_topic[topic_name] = row
        if recompute_combination or recompute_overlap:
            topic_evaluated_ls = []
        else:
            topic_evaluated_ls = [
                topic_name
                for topic_name, row in latest_results_by_topic.items()
                if _has_new_combination_schema(row)
            ]
    else:
        topic_evaluated_ls = []

    logger.info(f"当前评估 topic 数量: {len(topic_ls)} (limit={limit})")

    # 评估每个topic的策略一致性
    for topic_value in topic_ls:
        try:
            # agent + topic 的header路径
            topic_header = os.path.join(topic_dir, topic_value)
            existing_row = latest_results_by_topic.get(topic_value, {})

            if topic_value in topic_evaluated_ls:
                logger.info(f"Topic “{topic_value}” has been evaluated, skipping...")
                continue
            if recompute_combination and _has_new_combination_schema(existing_row):
                logger.info(
                    f"Topic “{topic_value}” already has strategy_combination, "
                    f"but --recompute-combination is enabled, so it will be re-evaluated."
                )
            if recompute_overlap and _has_reusable_overlap_schema(existing_row):
                logger.info(
                    f"Topic “{topic_value}” already has strategy_overlap, "
                    f"but --recompute-overlap is enabled, so it will be re-evaluated."
                )

            logger.warning(f"开始评估 {topic_value} 的idea和exp !!!!!")

            # 加载 idea 和 labels
            pred_strategies,label_strategies = get_idea_labels(header_dir,topic_header,agent_framework)

            ##################################### idea 策略一致性评估 ######################################
            if recompute_overlap:
                overlap_result = evaluate_strategy_overlap(pred_strategies,label_strategies)
                logger.info(
                    f"strategy overlap label_side_score: {overlap_result['label_side_score']}"
                )
            elif reuse_overlap and _has_reusable_overlap_schema(existing_row):
                overlap_result = existing_row["strategy_overlap"]
                logger.info(
                    f"复用已有 strategy_overlap: label_side_score={overlap_result['label_side_score']}"
                )
            else:
                if reuse_overlap:
                    logger.info("未找到可复用的 strategy_overlap，将重新评估 overlap。")
                overlap_result = evaluate_strategy_overlap(pred_strategies,label_strategies)
                logger.info(
                    f"strategy overlap label_side_score: {overlap_result['label_side_score']}"
                )


            ##################################### idea 组合性评估 ######################################
            if recompute_combination:
                combination_result = evaluate_strategy_combination(pred_strategies, overlap_result, label_strategies)
                logger.info(
                    f"coverage score: {combination_result['coverage_score']}, "
                    f"strategy combination score: {combination_result['strategy_combination_score']}, "
                    f"best_combination_label_idx: {combination_result.get('best_combination_label_idx')}"
                )
            elif recompute_overlap and _has_new_combination_schema(existing_row):
                combination_result = existing_row["strategy_combination"]
                logger.info(
                    f"复用已有 strategy_combination: coverage score={combination_result.get('coverage_score')}, "
                    f"strategy combination score={combination_result.get('strategy_combination_score', combination_result.get('score'))}, "
                    f"best_combination_label_idx={combination_result.get('best_combination_label_idx')}"
                )
            else:
                if recompute_overlap:
                    logger.info("未找到可复用的 strategy_combination，将基于新的 overlap 重新评估 combination。")
                combination_result = evaluate_strategy_combination(pred_strategies, overlap_result, label_strategies)
                logger.info(
                    f"coverage score: {combination_result['coverage_score']}, "
                    f"strategy combination score: {combination_result['strategy_combination_score']}, "
                    f"best_combination_label_idx: {combination_result.get('best_combination_label_idx')}"
                )

            overlap_score = _calculate_strategy_overlap_score(overlap_result)
            overall_score = _calculate_overall_score(overlap_result, combination_result)
            logger.info(
                f"strategy overlap score: {overlap_score}, "
                f"overall score: {overall_score}"
            )

            _upsert_evaluation_row(evaluation_path, {
                "topic": topic_value,
                "strategy_overlap": {
                    "score": overlap_score,
                    "label_side_score": overlap_result["label_side_score"],
                    "pred_strategy_num": overlap_result["pred_strategy_num"],
                    "pred_dimension_num": overlap_result["pred_dimension_num"],
                    "label_strategy_num": overlap_result["label_strategy_num"],
                    "label_group_size": overlap_result["label_group_size"],
                    "label_group_num": overlap_result["label_group_num"],
                    "matched_label_ids": overlap_result["matched_label_ids"],
                    "matched_label_strategies": overlap_result["matched_label_strategies"],
                    "dimension_details": overlap_result["dimension_details"],
                    "reason_summary": overlap_result["reason_summary"],
                },
                "strategy_combination": combination_result,
                "overall_score": overall_score,
            })
        except Exception as e:
            logger.error(
                f"评估 topic {topic_value} 失败，跳过该 topic 继续执行: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            continue

    summary_rows_by_topic = {}
    for row in _dedupe_latest_topic_rows(_load_jsonl_rows(evaluation_path)):
        topic_name = row.get("topic")
        if topic_name in topic_ls:
            summary_rows_by_topic[topic_name] = row
    summary_rows = list(summary_rows_by_topic.values())
    summary = _build_evaluation_summary(summary_rows)
    summary["limit"] = limit
    summary["evaluated_topics"] = len(summary_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(
        f"evaluation summary - overlap_avg: {summary['strategy_overlap_avg']}, "
        f"combination_avg: {summary['strategy_combination_avg']}, "
        f"overall_avg: {summary['overall_avg']}"
    )

    logger.warning(f"所有评估均已完成 !!!!!")    


def process_framework(agent_framework, limit=None, recompute_combination=False, reuse_overlap=False, recompute_overlap=False):
    logger.warning(f"开始评估 {agent_framework} 的创新性和idea前瞻性 !!!!!")
    main(
        agent_framework=agent_framework,
        limit=limit,
        recompute_combination=recompute_combination,
        reuse_overlap=reuse_overlap,
        recompute_overlap=recompute_overlap,
    )


def process_framework_safe(agent_framework, limit=None, recompute_combination=False, reuse_overlap=False, recompute_overlap=False):
    try:
        process_framework(
            agent_framework,
            limit=limit,
            recompute_combination=recompute_combination,
            reuse_overlap=reuse_overlap,
            recompute_overlap=recompute_overlap,
        )
        return {"framework": agent_framework, "status": "success"}
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(
            f"框架 {agent_framework} 评估失败，已转成可序列化错误结果返回主进程: "
            f"{type(e).__name__}: {e}\n{error_traceback}"
        )
        return {
            "framework": agent_framework,
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_traceback,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate idea strategy overlap and combination scores for one or more agent frameworks."
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        required=True,
        help="One or more agent framework names, for example: grok-4 gpt-5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N topics after sorting.",
    )
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable multiprocessing and run frameworks sequentially.",
    )
    parser.add_argument(
        "--recompute-combination",
        action="store_true",
        help="Recompute strategy_combination even if evaluation.jsonl already has a new combination schema.",
    )
    parser.add_argument(
        "--reuse-overlap",
        action="store_true",
        help="Reuse existing strategy_overlap from evaluation.jsonl when available.",
    )
    parser.add_argument(
        "--recompute-overlap",
        action="store_true",
        help="Recompute strategy_overlap while reusing existing strategy_combination when available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent_framework_ls = list(dict.fromkeys(args.frameworks))
    if_mp = not args.no_mp

    if if_mp and len(agent_framework_ls) > 1:
        # 创建进程池
        num_processes = min(len(agent_framework_ls), mp.cpu_count())
        logger.warning(f"使用 {num_processes} 个进程并行评估...")
        
        # 创建进程池并运行多进程评估
        with mp.Pool(processes=num_processes) as pool:
            framework_results = pool.starmap(
                process_framework_safe,
                [
                    (
                        af,
                        args.limit,
                        args.recompute_combination,
                        args.reuse_overlap,
                        args.recompute_overlap,
                    )
                    for af in agent_framework_ls
                ],
            )
        failed_frameworks = [result for result in framework_results if result.get("status") != "success"]
        for result in failed_frameworks:
            logger.error(
                f"框架 {result.get('framework')} 评估失败: "
                f"{result.get('error_type')}: {result.get('error_message')}"
            )
        
        logger.warning(f"所有框架评估完成!")
    else:
        for agent_framework in agent_framework_ls:
            framework_result = process_framework_safe(
                agent_framework,
                limit=args.limit,
                recompute_combination=args.recompute_combination,
                reuse_overlap=args.reuse_overlap,
                recompute_overlap=args.recompute_overlap,
            )
            if framework_result.get("status") != "success":
                logger.error(
                    f"框架 {framework_result.get('framework')} 评估失败: "
                    f"{framework_result.get('error_type')}: {framework_result.get('error_message')}"
                )
        logger.warning(f"所有框架评估完成!")
