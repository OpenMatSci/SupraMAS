import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from loguru import logger

from benchmark.baselines.llm_baseline.utils import extract

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPIC_LIB = REPO_ROOT / "benchmark" / "data" / "benchmark_topics.json"
DEFAULT_IDEA_LIB = REPO_ROOT / "benchmark" / "data" / "post_cutoff_paper_ideas.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "llm_baseline"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmark.example.yaml"

IDEA_EXP_SYSTEM_PROMPT = """
## 
  You are a **research expert in condensed matter physics**, specializing in materials physics, superconductivity, and experimental design.
  You will receive a research topic (**Topic**) and must deliver a cutting-edge, scientifically valuable, and actionable research proposal.
  **Important constraint:** Only use scientific knowledge and references published **before 2025-01-01**. Do not include any findings, data, or citations from 2025 or later.
  If web search is used, only rely on sources with publication date earlier than 2025-01-01.

  ---

  ## **Task Requirements**

  ### 1. Research Ideas
  - Based strictly on the provided **Topic**, generate **innovative, original, and feasible** research ideas wrapped in `<idea> ... </idea>` tags.
  - Clearly articulate:
    Your idea is composed of the following components:  Motivation:
    1. Provide a background for your idea, summarizing relevant work.
    2. Identify shortcomings in previous research and highlight the  specific problems that remain unsolved and that you aim to address.  Novelty:  1. Distinguish your proposed method from existing methods  (preferably by naming specific approaches).  2. Detail the improvements of your method compared to past work.  3. Clearly outline at least three contributions your idea offers  to the field, including the problems it resolves and the benefits  it delivers.  Method:  1. Present a detailed description of your idea, focusing on the  core method, the specific problem it solves, and enhancements over  earlier research (citing relevant literature with titles).  2. Explain the step-by-step methodology, including the functions  of each module and the rationale for why this approach effectively  addresses previous challenges.  Please adhere to the following guidelines:  1. Your research idea should be innovative, feasible, and  contribute meaningfully to the field. Please carefully examine  the idea you have proposed, avoid immediate perception, and try to  be different from the previous methods as much as possible.  2. Ensure your proposal is solid, clearly defined, and practical  to implement. Logic should underpin your reasoning.
    3. Write in clear, concise language aimed at an audience with  limited background knowledge in the subject. Avoid complex  technical jargon, but when professional terms are necessary,  provide thorough explanations.
    4. Refrain from introducing concepts from uncertain fields to  prevent proposing ideas that may be incorrect or impractical.
    5. When referencing other research, please include the titles of  the cited papers.
    6. Please avoid introducing unfamiliar information, ensuring that  the trends you present are both authentic and reasonable. Before  proposing any trends, take a moment to reflect on the principles  underlying the methods you’re employing and assess their relevance  to your research area. Continue to next table ->  20 Preprint  Table 15: Prompt used to generate idea (part II)
    7. Each article’s limitations are specific to that particular  piece and should not be applied to others. Carefully consider the  task at hand and analyze the potential issues you might encounter  if you proceed with your original approach, reflecting on the  challenges previously faced. Then, think critically about how to  address these issues effectively.
    8. **Do not** use any post-2025 information
    Please output your motivation,novelty,method firstly and then  output your final idea.The final idea should clearly explain the  origins, motivation, and challenges of your idea, detailing how you  overcame these hurdles.
    Please present the final idea in the following format:  <idea> Motivation: ...  Novelty: ...  Method: ...  Final idea: ... </idea>

  ---

  ### 2. Experiment Plan
  - Based on the Ideas you generate.Develop a detailed experimental plan wrapped in `<exp> ... </exp>` tags.
  - Clearly articulate:
    Please propose a detailed experimental plan addressing the  following points:
    1. Experimental Design: Develop rigorous experiments to  ensure the reliability and validity of your results. Provide  a comprehensive explanation of the baseline used, comparative  methods, ablation study design, and criteria for data analysis  and result evaluation. Clarify how these components collectively  reinforce and validate the conclusions of your research. Structure  your experimental design in a clear, logical, and step-by-step  manner, ensuring each step is well-defined and easy to understand.
    2. Implementation of Technologies/Methods: If your experimental  design involves specific technologies or methodologies, describe  the implementation process in detail, including key technical  aspects. For any critical concepts utilized, provide thorough  explanations. For instance, if you propose a modular approach,  detail its construction, components, and functionality.
    3. Feasibility Assessment: Ensure your experimental plan is  realistic, considering technological availability, timelines,  resources, and personnel. Identify potential challenges and  propose strategies for addressing them.
    4. References to Previous Studies: When citing related  literature, include titles and pertinent details of the original  papers. Strive to use as many references as necessary to support  your experimental design.
    5. Clarity of Language: Use straightforward language to describe  your methods, assuming the reader may have limited knowledge of  the subject matter. Avoid complex jargon and utilize accessible  terminology. If professional terms are necessary, please provide  clear and detailed explanations.
    6. **Do not** use any post-2025 information
    Please output strictly in the following format:  <exp> Experiment:  Step1: ...  Step2: ... </exp>

  ---
  ### 3. Formatting Rules
  - All output in **English** with professional technical terms.
  - Use `<idea>` and `<exp>` blocks exactly once each.
  - Inside `<exp>`, use **clear headings** for each required section.
  - All parameter values must be realistic and evidence-based.
  - **Do not** use any post-2025 information.

  ---
"""

EXP_SYSTEM_PROMPT = """
##You are a condensed matter physics research expert specializing in materials physics, superconductivity, and experimental design.

### Task:
You will be given a research idea. Based on that idea, design a experimental plan.

### Constraint:
Use only scientific knowledge, methods, and references published before 2025-01-01.
Do not include any discoveries, data, assumptions, or citations from 2025 or later.

### Output requirement:
You must strictly output in the following format:
<exp>Experiment: Step1: … Step2: …</exp>

"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline LLM benchmark.")
    parser.add_argument(
        "--task",
        default="topic",
        choices=["topic", "experiment-feasibility"],
        help="Run the original topic benchmark or generate experiment plans from paper ideas only.",
    )
    parser.add_argument(
        "--topic-lib",
        default=str(DEFAULT_TOPIC_LIB),
        help="Path to topic JSON file.",
    )
    parser.add_argument(
        "--idea-lib",
        default=str(DEFAULT_IDEA_LIB),
        help="Path to paper-idea JSON file for --task experiment-feasibility.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for baseline results.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to model API config YAML.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model backend to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to process.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries if no valid <idea>/<exp> is extracted.")
    parser.add_argument("--timeout", type=int, default=3000, help="HTTP timeout seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_models(config: dict) -> list[str]:
    return [k for k in config.keys() if k != "headers"]


def infer_api_type(model_name: str, model_entry: dict) -> str:
    api_type = model_entry.get("api_type")
    if api_type:
        return api_type
    return "openai_chat_completions"


def extract_response_text(api_type: str, data: dict) -> str:
    if api_type == "openai_chat_completions":
        return data["choices"][0]["message"]["content"]
    raise ValueError(f"Unsupported api_type: {api_type}")


def post_json_with_fallbacks(
    url: str,
    headers: dict,
    payloads: list[dict],
    timeout: int,
) -> dict:
    last_error = None
    for idx, payload in enumerate(payloads, start=1):
        try:
            response = requests.request("POST", url, headers=headers, json=payload)
            if response.status_code >= 400:
                body = response.text[:800]
                logger.warning(f"http error on payload {idx}/{len(payloads)}: {response.status_code}, body={body}")
                response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            logger.warning(f"request variant {idx}/{len(payloads)} failed: {exc}")
            continue
    raise RuntimeError(f"all request variants failed: {last_error}")


def to_chat_completions_url(url: str) -> str:
    if "/responses" in url:
        return url.replace("/responses", "/chat/completions")
    if ":generateContent" in url:
        prefix = url.split("/v1beta/models/", 1)[0]
        return f"{prefix}/v1/chat/completions"
    return url


def is_legacy_topic(topic: object) -> bool:
    return isinstance(topic, dict) and all(
        key in topic for key in ("REBCO", "dopped_materials", "processing_method", "sample_category")
    )


def topic_text(topic: dict | str) -> str:
    if isinstance(topic, str):
        text = topic.strip()
        if not text:
            raise ValueError("Topic text cannot be empty.")
        return text

    explicit_topic = str(topic.get("topic", "")).strip()
    if explicit_topic:
        return explicit_topic

    if not is_legacy_topic(topic):
        raise KeyError(
            "Topic dict must contain either a non-empty 'topic' field or the legacy keys: "
            "'REBCO', 'dopped_materials', 'processing_method', 'sample_category'."
        )

    return "Enhanced Superconducting Properties of '{}'-Doped '{}' Thin {} Prepared by {}".format(
        topic["dopped_materials"],
        topic["REBCO"],
        topic["sample_category"],
        topic["processing_method"],
    )


def folder_name_from_topic(topic: dict | str, idx: int | None = None) -> str:
    if is_legacy_topic(topic) and not str(topic.get("topic", "")).strip():
        raw = "{}_{}_{}_{}".format(
            topic["REBCO"],
            topic["dopped_materials"],
            topic["processing_method"],
            topic["sample_category"],
        ).replace(" ", "_").replace("/", "-")
        return re.sub(r"[^\w\-.()]+", "_", raw)

    raw = topic_text(topic)
    slug = raw.replace(" ", "_").replace("/", "-")
    slug = re.sub(r"[^\w\-.()]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_.")
    if not slug:
        slug = "topic"

    # Free-text topics are longer than the old structured schema, so cap the folder length.
    if len(slug) > 120:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[:109].rstrip('_')}_{digest}"

    if idx is not None:
        return f"{idx:03d}_{slug}"
    return slug


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def join_non_empty(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if str(part).strip())


def normalize_strategy_items(raw_items: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(raw_items, list):
        return rows

    for item in raw_items:
        if isinstance(item, str) and item.strip():
            rows.append(
                {
                    "tuning dimensions": "",
                    "tuning strategies": item.strip(),
                    "tuning effects": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "tuning dimensions": str(
                    item.get("tuning dimensions") or item.get("tuning_dimensions") or item.get("dimension") or ""
                ).strip(),
                "tuning strategies": str(
                    item.get("tuning strategies") or item.get("tuning_strategies") or item.get("strategy") or ""
                ).strip(),
                "tuning effects": str(
                    item.get("tuning effects") or item.get("tuning_effects") or item.get("effect") or ""
                ).strip(),
            }
        )
    return rows


def build_paper_idea_text(paper_idea: dict[str, Any]) -> str:
    strategy_lines: list[str] = []
    for idx, item in enumerate(paper_idea.get("strategies", []), start=1):
        strategy_lines.append(
            f"{idx}. [{item.get('tuning dimensions', '')}] "
            f"{item.get('tuning strategies', '')} -> {item.get('tuning effects', '')}"
        )
    strategy_text = "\n".join(strategy_lines)
    return (
        f"Material: {paper_idea.get('material', '')}\n"
        f"Motivation: {paper_idea.get('motivation', '')}\n"
        f"Strategies:\n{strategy_text}\n"
        f"Technique: {paper_idea.get('technique', '')}\n"
        f"Outcome: {paper_idea.get('outcome', '')}"
    ).strip()


def build_experiment_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    material_system = raw_record.get("MaterialSystem", {}) if isinstance(raw_record.get("MaterialSystem"), dict) else {}
    tuning_method = raw_record.get("TunningMethod", {}) if isinstance(raw_record.get("TunningMethod"), dict) else {}
    experiment_tech = raw_record.get("ExperimentTech", {}) if isinstance(raw_record.get("ExperimentTech"), dict) else {}
    application = raw_record.get("Application", {}) if isinstance(raw_record.get("Application"), dict) else {}

    existing_paper_idea = raw_record.get("paper_idea", {})
    if isinstance(existing_paper_idea, dict) and existing_paper_idea:
        paper_idea = {
            "material": str(existing_paper_idea.get("material", "")).strip(),
            "motivation": str(existing_paper_idea.get("motivation", "")).strip(),
            "strategies": normalize_strategy_items(existing_paper_idea.get("strategies", [])),
            "technique": str(existing_paper_idea.get("technique", "")).strip(),
            "outcome": str(existing_paper_idea.get("outcome", "")).strip(),
        }
    else:
        strategies = normalize_strategy_items(tuning_method.get("strategies", []))
        material_text = join_non_empty(
            [
                str(material_system.get("description", "")),
                join_non_empty(
                    [
                        f"Base material: {material_system.get('base_material', '')}"
                        if material_system.get("base_material")
                        else "",
                        f"Dopants: {material_system.get('dopants', '')}" if material_system.get("dopants") else "",
                    ]
                ),
            ]
        )
        motivation_text = join_non_empty(
            [
                str(tuning_method.get("description", "")),
                str(application.get("description", "")),
            ]
        )
        technique_text = join_non_empty(
            [
                str(experiment_tech.get("description", "")),
                f"Fabrication: {experiment_tech.get('FabricationTech', '')}"
                if experiment_tech.get("FabricationTech")
                else "",
                f"Characterization: {experiment_tech.get('CharacterizationTech', '')}"
                if experiment_tech.get("CharacterizationTech")
                else "",
                f"Measurement: {experiment_tech.get('MeasurementTech', '')}"
                if experiment_tech.get("MeasurementTech")
                else "",
            ]
        )
        outcome_text = join_non_empty(
            [
                str(application.get("ApplicationIndex", "")),
                str(application.get("ApplicationScenario", "")),
            ]
        )
        paper_idea = {
            "material": material_text,
            "motivation": motivation_text,
            "strategies": strategies,
            "technique": technique_text,
            "outcome": outcome_text,
        }

    source_file = str(raw_record.get("source_file", "")).strip()
    label_idx = raw_record.get("label_idx")
    title = str(raw_record.get("title", "")).strip()
    if not title:
        title = Path(source_file).stem if source_file else f"label_{label_idx}"
    title = title or f"label_{label_idx}"

    paper_idea_text = str(raw_record.get("paper_idea_text", "")).strip()
    if not paper_idea_text:
        paper_idea_text = build_paper_idea_text(paper_idea)

    return {
        "label_idx": label_idx,
        "paper_id": str(raw_record.get("paper_id") or source_file or f"label_{label_idx}"),
        "source_file": source_file,
        "doi": str(raw_record.get("doi", "")),
        "title": title,
        "topic": str(raw_record.get("topic") or title),
        "paper_idea": paper_idea,
        "paper_idea_text": paper_idea_text,
        "raw_label_strategy": raw_record,
    }


def load_experiment_records(idea_lib: str, limit: int | None) -> list[dict[str, Any]]:
    with open(idea_lib, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Idea library must be a JSON list: {idea_lib}")

    normalized = [build_experiment_record(record) for record in records if isinstance(record, dict)]
    if limit is not None:
        normalized = normalized[: limit]
    return normalized


def record_label_dir(record: dict[str, Any], idx: int) -> str:
    label_idx = str(record.get("label_idx", "")).strip()
    if label_idx:
        return f"label_{label_idx}"
    return f"{idx:03d}"


def final_experiment_payload(record: dict[str, Any], experiment_text: str) -> dict[str, Any]:
    return {
        "label_idx": record.get("label_idx"),
        "experiment": experiment_text,
        "source_file": str(record.get("source_file", "")),
    }


def has_saved_experiment(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if isinstance(payload, dict):
        for key in ("experiment", "exp", "predicted_experiment", "paper_experiment", "raw_experiment"):
            if str(payload.get(key, "")).strip():
                return True
        return False

    return bool(str(payload).strip())


def request_model(
    model: str,
    config: dict,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> str:
    headers = dict(config["headers"][0])
    model_entry = config[model][0]
    url = to_chat_completions_url(model_entry["base_url"])
    api_type = infer_api_type(model, model_entry)
    model_alias = model.lower()
    provider_model = str(model_entry.get("model", "")).lower()
    system_role = "system" if ("deepseek" in model_alias or "deepseek" in provider_model) else "developer"

    if api_type == "openai_chat_completions":
        base_data = {
            "messages": [
                {"role": system_role, "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": model_entry["model"],
            "temperature": 0,
        }
        payloads = []
        # Some chat-completions backends (e.g., Grok-compatible gateway) accept this field.
        if "grok" in model.lower() or "grok" in str(model_entry.get("model", "")).lower():
            payloads.append({**base_data, "search_parameters": {"mode": "auto"}})
        payloads.append(base_data)
        data = post_json_with_fallbacks(url, headers, payloads, timeout)
        return extract_response_text(api_type, data)

    raise ValueError(f"Unsupported api type for model '{model}': {api_type}")


def generate_topic_result(
    topic: dict,
    topic_dir: Path,
    model: str,
    config: dict,
    timeout: int,
    max_retries: int,
) -> None:
    user_query = (
        f"# Generate a research idea and experiment plan for the topic: {topic_text(topic)}. "
        "## Idea wrapped in `<idea> ... </idea>` tags. "
        "## Experiment plan wrapped in `<exp> ... </exp>` tags. "
        "## Any web-searched evidence must be published before 2025-01-01."
    )

    last_response = None
    had_response = False
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            last_response = request_model(model, config, IDEA_EXP_SYSTEM_PROMPT, user_query, timeout)
            had_response = True
        except Exception as exc:
            logger.error(f"request failed (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(min(2 * attempt, 8))
            continue
        elapsed = time.time() - start
        print(f"[{model}] attempt {attempt}/{max_retries}, {elapsed:.2f}s")

        idea = extract(last_response, "idea", hard=False)
        experiment = extract(last_response, "exp", hard=False)
        if idea and experiment:
            (topic_dir / "idea.json").write_text(
                json.dumps({"idea": idea}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (topic_dir / "experiments.json").write_text(
                json.dumps({"exp": experiment}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (topic_dir / "raw_response.md").write_text(last_response, encoding="utf-8")
            return

        logger.warning(f"invalid tags for topic, retrying (attempt {attempt}/{max_retries})")

    if last_response is not None:
        (topic_dir / "raw_response.md").write_text(last_response, encoding="utf-8")
    if not had_response:
        raise RuntimeError(f"all requests failed after {max_retries} attempts")
    raise RuntimeError(f"failed to extract <idea>/<exp> after {max_retries} attempts")


def generate_experiment_result(
    record: dict[str, Any],
    record_dir: Path,
    model: str,
    config: dict,
    timeout: int,
    max_retries: int,
) -> None:
    idea_text = str(record.get("paper_idea_text", "")).strip()
    if not idea_text:
        raise ValueError("paper_idea_text is empty")

    user_query = (
        f"# Generate a detailed experiment plan for the research idea:\n{idea_text}\n"
        "## Experiment plan wrapped in `<exp> ... </exp>` tags\n"
        "## Any web-searched evidence must be published before 2025-01-01."
    )

    last_response = None
    had_response = False
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            last_response = request_model(model, config, EXP_SYSTEM_PROMPT, user_query, timeout)
            had_response = True
        except Exception as exc:
            logger.error(f"request failed (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(min(2 * attempt, 8))
            continue

        elapsed = time.time() - start
        print(f"[{model}] attempt {attempt}/{max_retries}, {elapsed:.2f}s")

        experiment = extract(last_response, "exp", hard=False)
        if experiment:
            write_json(record_dir / "input_idea.json", record.get("paper_idea", {}))
            write_json(record_dir / "experiment.json", final_experiment_payload(record, experiment))
            (record_dir / "raw_response.md").write_text(last_response, encoding="utf-8")
            return

        logger.warning(f"invalid <exp> tag for idea, retrying (attempt {attempt}/{max_retries})")

    if last_response is not None:
        (record_dir / "raw_response.md").write_text(last_response, encoding="utf-8")
    if not had_response:
        raise RuntimeError(f"all requests failed after {max_retries} attempts")
    raise RuntimeError(f"failed to extract <exp> after {max_retries} attempts")


def run_topic_task(args: argparse.Namespace, config: dict) -> None:
    with open(args.topic_lib, "r", encoding="utf-8") as f:
        topics = json.load(f)
    if args.limit is not None:
        topics = topics[: args.limit]

    model_output_dir = Path(args.output_dir) / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, topic in enumerate(topics, start=1):
        topic_dir = model_output_dir / folder_name_from_topic(topic, idx=idx)
        topic_dir.mkdir(parents=True, exist_ok=True)
        log_sink_id = logger.add(str(topic_dir / "log.log"), rotation="10 MB")

        try:
            target_files = [topic_dir / "idea.json", topic_dir / "experiments.json", topic_dir / "raw_response.md"]
            if not args.overwrite and all(p.exists() for p in target_files):
                print(f"[skip {idx}/{len(topics)}] {topic_dir.name}")
                continue

            print(f"[run  {idx}/{len(topics)}] {topic_dir.name}")
            try:
                generate_topic_result(
                    topic=topic,
                    topic_dir=topic_dir,
                    model=args.model,
                    config=config,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                print(f"[ok   {idx}/{len(topics)}] {topic_dir.name}")
            except Exception as exc:
                logger.error(f"topic failed: {topic} err={exc}")
                print(f"[fail {idx}/{len(topics)}] {topic_dir.name}: {exc}")
        finally:
            logger.remove(log_sink_id)


def run_experiment_task(args: argparse.Namespace, config: dict) -> None:
    records = load_experiment_records(args.idea_lib, args.limit)
    model_output_dir = Path(args.output_dir) / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records, start=1):
        record_dir = model_output_dir / record_label_dir(record, idx)
        record_dir.mkdir(parents=True, exist_ok=True)
        log_sink_id = logger.add(str(record_dir / "log.log"), rotation="10 MB")

        try:
            experiment_path = record_dir / "experiment.json"
            if not args.overwrite and has_saved_experiment(experiment_path):
                print(f"[skip {idx}/{len(records)}] {record_dir.name}")
                continue

            print(f"[run  {idx}/{len(records)}] {record_dir.name}")
            try:
                generate_experiment_result(
                    record=record,
                    record_dir=record_dir,
                    model=args.model,
                    config=config,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                print(f"[ok   {idx}/{len(records)}] {record_dir.name}")
            except Exception as exc:
                logger.error(f"experiment generation failed: record={record.get('paper_id')} err={exc}")
                print(f"[fail {idx}/{len(records)}] {record_dir.name}: {exc}")
        finally:
            logger.remove(log_sink_id)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = available_models(config)
    if args.model not in models:
        raise ValueError(f"Model '{args.model}' not found in config. Available models: {', '.join(models)}")
    if args.task == "experiment-feasibility":
        run_experiment_task(args, config)
        return

    run_topic_task(args, config)


if __name__ == "__main__":
    main()
