import json
import os
import time 
import re
import requests
from openai import OpenAI
from loguru import logger
from typing import List, Dict,Optional
from abc import ABC, abstractmethod
from benchmark.evaluation.utils import *
import fitz


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_api_url(base_url: str, target_path: str) -> str:
    if not base_url:
        return target_path

    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    elif normalized.endswith("/responses"):
        normalized = normalized[: -len("/responses")]

    return f"{normalized}{target_path}"


def _extract_response_text(response_json: Dict) -> str:
    if not isinstance(response_json, dict):
        raise ValueError("response is not a JSON object")

    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output_items = response_json.get("output")
    if isinstance(output_items, list):
        texts = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        if texts:
            return "".join(texts)

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content

    error = response_json.get("error")
    if isinstance(error, dict):
        error_message = error.get("message")
        if isinstance(error_message, str) and error_message:
            raise RuntimeError(error_message)

    raise ValueError(f"unexpected response schema: {json.dumps(response_json, ensure_ascii=False)[:1000]}")


class BaseAgent(ABC):
    def __init__(self, model: str, temperature: float = 0.7,base_url: str = None,api_key:str = None):
        self.model = model
        self.temperature = temperature
        self.is_response = _env_flag("is_response")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        self.response_url = _normalize_api_url(self.base_url, "/responses")
        if base_url and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
    
    def generate_response(self, prompt: str ,history: str=None ,file_path: str=None) -> str:
        messages = []
        if file_path:
            # file_object = self.client.files.create(file=Path(file_path), purpose="file-extract")
            # file_content = self.client.files.content(file_id=file_object.id).text
            # 使用 PyMuPDF 打开 PDF 并提取全部文本
            doc = fitz.open(file_path)
            file_content = ""
            for page in doc:
                file_content += page.get_text("text")
            doc.close()
            # print("file_content",file_content)
            messages = [{"role": "system", "content": "You are an expert in condensed matter physics and are familiar with REBCO flux-pinning research."},
                        {"role": "system", "content": file_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        def _chat_completions_fallback():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192,
            )
            return response.choices[0].message.content

        if self.is_response:
            payload = json.dumps({
                "model": self.model,
                "input": messages,
                "temperature": self.temperature,
                "max_output_tokens": 8192,
            }, ensure_ascii=False)

            try:
                response = requests.post(self.response_url, headers=self.headers, data=payload, timeout=3000)
                response.raise_for_status()
                response_json = response.json()
                return _extract_response_text(response_json)
            except (requests.RequestException, json.JSONDecodeError, ValueError, RuntimeError) as exc:
                raw_preview = ""
                status_code = getattr(locals().get("response"), "status_code", None)
                if "response" in locals():
                    try:
                        raw_preview = (response.text or "")[:500]
                    except Exception:
                        raw_preview = ""
                logger.warning(
                    "responses API failed for model={} url={} status={} error={}; fallback to chat.completions. body_preview={}",
                    self.model,
                    self.response_url,
                    status_code,
                    exc,
                    raw_preview,
                )
                return _chat_completions_fallback()

        return _chat_completions_fallback()
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        pass

class CoITopicAgent(BaseAgent):
    
    """审核智能体 - 检查创新性"""
    def process(self, input_data: Dict) -> Dict:
        topic = input_data.get("topic")

        prompt = f"""You are an expert in the field of condensed matter physics. Your task is to generate a concise and descriptive research title based on the provided topic. The research should aim to achieve superconductivity.

Here is the topic:

{topic}

Please follow these guidelines when generating the title:
	•	A concise and descriptive name for the research related to the topic.
	•	It should attract readers’ attention and accurately reflect the main content.
	•	It should also imply the goal of achieving superconductivity. (e.g., # Enhanced Superconducting Properties of Some Material).
	•	Keep it short (usually within 20–30 words), clear, and specific to the topic.
	•	It should not be too vague or too long.
Wrap the generated title entirely within <topic> tags, like this:
<topic>
[Your generated title here]
</topic>

"""
        response = self.generate_response(prompt)
        topic = extract(response,"topic")
        while not topic:
            logger.info(f"提取topic失败，重新生成!")
            response = self.generate_response(prompt)
            topic = extract(response,"topic")
        return {"topic": topic}

class Idea2QueryAgent(BaseAgent):
    '''根据idea内容生成检索query'''
    def process(self, input_data: Dict) -> Dict:
        idea = input_data.get("idea")
        prompt = f"""
Your task is to generate a search query from the given idea. Read the idea carefully and generate an appropriate retrieval query according to the instructions below.
Idea content:
<idea>
{idea}
</idea>
When generating the retrieval query, follow these guidelines:
1. Extract the key information and core topic from the idea.
2. Use concise and clear language; avoid unnecessary modifiers and complex phrasing.
3. Make sure the query accurately reflects the main content of the idea and can help retrieve relevant information from a search engine.
4. Adjust word order and wording when useful to improve retrieval effectiveness.
Write the generated retrieval query inside <query> tags.

"""
        response = self.generate_response(prompt)
        query = extract(response,"query")
        while not query:
            logger.info(f"提取query失败，重新生成!")  
            response = self.generate_response(prompt)
            query = extract(response,"query")
        return {"query": query}

class Idea2StrategyAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def process(self, input_data: Dict) -> Dict:
        idea = input_data.get("idea")
        prompt = f"""
### Task Description
I will provide a materials-science idea. Your task is to summarize it into a structured JSON representation according to the requirements below.

Idea:
<idea>
{idea}
</idea>


### Output Content Format
Summarize the idea as a JSON object containing four top-level key-value pairs.
1. The first key-value pair should describe the material system, including the REBCO material system and dopant materials.
   - In "description", describe the material system, lattice parameters, possible defects, texture, and related features.
   - Fill "base_material" and "dopants" with the base material and dopant materials; leave them empty if unavailable.
 
2. The second key-value pair should describe the idea's regulation strategies and regulation effects.
   - In "description", explain how the physical-property regulation strategies work together to optimize the experimental objective.
   - The value of "strategies" is an array [] containing multiple JSON objects. Each JSON object contains "tuning dimensions" and "tuning strategies". Add as many objects as there are extractable regulation strategies.
   - tuning strategies: use summary language to describe each regulation strategy concretely, and include its deeper effect while describing the concrete strategy.
   - tuning dimensions: choose one of the five tuning dimensions for each strategy: Composition tuning, grain boundary tuning, interface tuning, texture tuning, stress tuning.
   - Tuning-dimension definitions: Composition tuning adjusts chemical composition to introduce microscopic defects or change crystal structure, thereby enhancing flux pinning. Grain boundary tuning optimizes interfaces between superconducting grains to reduce current obstruction. Interface tuning optimizes the contact interfaces among the REBCO superconducting layer, metal substrate, and buffer layers to support high-quality crystal growth and reduce stress. Texture tuning controls grain orientation and texture consistency to reduce grain-boundary density and improve current transport. Stress tuning controls residual internal stress, such as thermal or mechanical stress, to prevent cracking or performance degradation.
 
3. The third key-value pair should describe the experimental techniques.
   - In "description", describe the specific experimental techniques used, grouped into sample-fabrication, characterization, and physical-property measurement techniques. Emphasize techniques that are innovative.
   - Fill "FabricationTech", "CharacterizationTech", and "MeasurementTech" with the specific types of sample-fabrication, characterization, and measurement techniques.
 
4. The fourth key-value pair should describe the application scenario and application metrics.
   - In "description", describe the application need.
   - In "ApplicationScenario", briefly describe the specific application scenario.
   - In "ApplicationIndex", provide performance metrics, such as critical current density.

    
### Notes
1. Every field must be directly grounded in evidence from the idea.
2. Fill all content in English.
3. Output JSON format.
4. Wrap the entire JSON object with <template> and </template>.
5. If the "strategies": [] array contains multiple JSON objects, separate them with commas to keep the JSON valid.
6. JSON may use only e notation for scientific notation, such as 2e6; do not use multiplication signs.
 
### Final Output Format
<template>
{{
    "MaterialSystem": {{
        "description": "",
        "base_material": "",
        "dopants": ""
    }},
    "TunningMethod": {{
        "description": "",
        "strategies": [
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }},
            ...
        ]
    }},
    "ExperimentTech": {{
        "description": "",
        "FabricationTech": "",
        "CharacterizationTech": "",
        "MeasurementTech": ""
    }},
    "Application": {{
        "description": "",
        "ApplicationScenario": "",
        "ApplicationIndex": "",
    }}
}}
</template>
 
Start now: analyze the idea and generate a JSON template object following the format above. The output must begin with <template> and end with </template>.
        """
        response = self.generate_response(prompt)

        templates = extract(response,"template")
        # print("templates", templates)
        try:
            templates = json.loads(templates)
            # print("templates", templates)
        except Exception as e:
            print(e)
            templates = None

        while not templates:
            logger.info(f"提取review_result失败，重新生成!")  
            response = self.generate_response(prompt)
            templates = extract(response,"template")
        return {"templates": templates}

class EvalOverlapAgent(BaseAgent):
    TUNING_DIMENSIONS = [
        "Composition tuning",
        "Grain boundary tuning",
        "Interface tuning",
        "Texture tuning",
        "Stress tuning",
    ]

    def _to_int(self, value, default=0):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if value.lstrip("-").isdigit():
                return int(value)
        return default

    def _normalize_string_list(self, value):
        if not isinstance(value, list):
            return []

        normalized = []
        seen = set()
        for item in value:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def _normalize_dimension_details(self, value):
        raw_details = value if isinstance(value, dict) else {}
        normalized_details = {}
        all_matched_label_ids = []

        for dimension in self.TUNING_DIMENSIONS:
            detail = raw_details.get(dimension, {})
            detail = detail if isinstance(detail, dict) else {}

            score = self._to_int(detail.get("score", 0), 0)
            score = 1 if score > 0 else 0
            matched_label_ids = self._normalize_string_list(detail.get("matched_label_ids", []))
            analysis = str(detail.get("analysis", "")).strip()

            normalized_details[dimension] = {
                "score": score,
                "matched_label_ids": matched_label_ids,
                "analysis": analysis,
            }
            all_matched_label_ids.extend(matched_label_ids)

        return normalized_details, self._normalize_string_list(all_matched_label_ids)

    def _is_valid_response(self, response):
        if not isinstance(response, dict):
            return False

        dimension_details = response.get("dimension_details", {})
        if not isinstance(dimension_details, dict):
            return False

        for dimension in self.TUNING_DIMENSIONS:
            if dimension not in dimension_details:
                return False
            if dimension_details[dimension].get("score") not in (0, 1):
                return False

        if not isinstance(response.get("label_side_score"), int):
            return False

        return True

    def load_response(self, json_str: str) -> Dict:
        """加载JSON字符串为字典"""
        try:
            reliability_evaluation = json.loads(json_str)
            dimension_details, detail_matched_label_ids = self._normalize_dimension_details(
                reliability_evaluation.get("dimension_details", {})
            )
            top_level_matched_label_ids = self._normalize_string_list(
                reliability_evaluation.get("matched_label_ids", [])
            )
            matched_label_ids = self._normalize_string_list(
                top_level_matched_label_ids + detail_matched_label_ids
            )
            label_side_score = len(matched_label_ids)
            reason_summary = str(reliability_evaluation.get("reason_summary", "")).strip()

            return {
                "label_side_score": label_side_score,
                "dimension_details": dimension_details,
                "matched_label_ids": matched_label_ids,
                "reason_summary": reason_summary,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None

    def process(self, input_data: Dict) -> Dict:
        input_tuning_strategies = input_data.get("input_strategies")
        label_tuning_strategies = input_data.get("label_strategies")
        prompt = f"""You are an expert in condensed matter physics and superconducting materials, specializing in REBCO flux-pinning research. Your task is to judge, across five fixed tuning dimensions, whether the input predicted strategies and the strategy database contain the same or scientifically similar regulation methods and target effects.

Strategies to evaluate:
<strategies_to_evaluate>
{json.dumps(input_tuning_strategies, ensure_ascii=False, indent=2)}
</strategies_to_evaluate>

Strategy database:
<strategy_database>
{json.dumps(label_tuning_strategies, ensure_ascii=False, indent=2)}
</strategy_database>

Evaluation procedure:
1. Evaluate strictly and separately under the following five fixed tuning dimensions. Dimension names must match exactly:
  - "Composition tuning"
  - "Grain boundary tuning"
  - "Interface tuning"
  - "Texture tuning"
  - "Stress tuning"
2. The input predicted strategies `input_tuning_strategies` are a JSON object keyed by the five tuning dimensions. Each dimension maps to an array of predicted strategies; an empty array means no predicted strategy is present for that dimension.
3. Each strategy in `label_tuning_strategies` contains `label_id` and `tuning dimensions`. During evaluation, compare only strategies under the same tuning dimension. Cross-dimension matching is prohibited.
4. For each dimension, first extract key elements:
  - Material system, such as YBCO, GdBCO, or REBCO
  - Regulation method, such as dopant elements (BZO, Ca) or process methods
  - Technical route, such as PLD or MOCVD
  - Structural features, such as nanocolumns, c-axis orientation, or grain-boundary defects
  - Target effects, such as enhanced Jc, improved flux pinning, or reduced weak-link behavior
5. Score each dimension as follows:
  - If the predicted-strategy array for that dimension is empty, `score = 0`
  - If that dimension contains at least one predicted strategy and the same dimension in the database contains at least one strategy with scientifically identical or similar regulation methods and target effects, `score = 1`
  - Otherwise, `score = 0`
6. Definition of `label_side_score`:
  - The number of database strategies judged similar to any predicted strategy
  - Count unique `label_id` values only. If the same `label_id` is hit multiple times across dimensions or predicted strategies, it still counts once.
7. You must output `matched_label_ids` for every dimension. The top-level `matched_label_ids` must also be the de-duplicated aggregate list.
 
Criteria for judging similar regulation strategies:
### General Principles
- **Core-goal consistency**: the key target or scientific effect addressed by the strategies is essentially the same.
- **Allowed differences**: differences in exact values or implementation paths do not prevent matching.
- **Multi-objective handling**: matching any core target is enough for the effect side to match.
 
### Composition Tuning
- **True conditions**:
  - The dopants or additives belong to the same functional type.
  - Similar doping or defect-introduction mechanisms are used, such as nanoparticles or elemental substitution.
  - The strategies address the same scientific problem, such as dislocation formation, second-phase formation, or surface pinning.
  - Similar experimental techniques and workflows are used.
  - Example: input "BaHfO3 doping" and database "BaZrO3 doping" -> True, because both are oxide nanopinning strategies.
 
### Interface Tuning
- **True conditions**:
  - Similar interface-engineering principles are used, such as mismatch relaxation or diffusion blocking.
  - Similar interface problems are addressed, such as lattice strain or reaction-layer suppression.
  - Similar experimental techniques and workflows are used.
  - Example: input "MgO buffer layer" and database "SrTiO3 interlayer" -> True, if both serve epitaxial interface optimization.
 
### Stress Tuning
- **True conditions**:
  - The strategies are based on the same physical principle, such as thickness effects or thermal-expansion matching.
  - They address similar stress problems, such as film-substrate mismatch or dislocation regulation.
  - Similar experimental techniques and workflows are used.
  - Example: input "10 nm ultrathin-layer stress control" and database "thickness-gradient stress management" -> True.
 
### Texture Tuning
- **True conditions**:
  - Similar orientation-control principles are used, such as epitaxial growth or seed-induced growth.
  - Similar texture features are targeted, such as c-axis orientation or biaxial texture.
  - Similar experimental techniques and workflows are used.
  - Example: input "IBAD-MgO texture" and database "PLD epitaxial texture control" -> True, because both are orientation-control approaches.
 
### Grain Boundary Tuning
- **True conditions**:
  - Similar grain-boundary engineering principles are used, such as dopant segregation or grain-boundary reconstruction.
  - The same failure mechanism is addressed, such as weak-link behavior or current limitation.
  - Similar experimental techniques and workflows are used.
  - Example: input "Sr doping to improve grain boundaries" and database "Ca doping to optimize grain boundaries" -> True, because both are alkaline-earth grain-boundary modification strategies.

Output exactly in the following format:
<score>
{{
    "label_side_score": "",
    "dimension_details": {{
        "Composition tuning": {{
            "score": "",
            "matched_label_ids": [],
            "analysis": ""
        }},
        "Grain boundary tuning": {{
            "score": "",
            "matched_label_ids": [],
            "analysis": ""
        }},
        "Interface tuning": {{
            "score": "",
            "matched_label_ids": [],
            "analysis": ""
        }},
        "Texture tuning": {{
            "score": "",
            "matched_label_ids": [],
            "analysis": ""
        }},
        "Stress tuning": {{
            "score": "",
            "matched_label_ids": [],
            "analysis": ""
        }}
    }},
    "matched_label_ids": [],
    "reason_summary": ""
}}
</score>

Notes:
1. `label_side_score` must equal the number of unique IDs in the top-level `matched_label_ids`.
2. Every dimension must be output; no dimension may be omitted.
3. If the predicted strategies for a dimension are empty, output `score = 0` and explicitly state in `analysis` that the dimension lacks predicted strategies.
4. Output only the JSON wrapped in `<score>...</score>`. Do not output any extra explanation or markdown code fences.
"""
        try:
            last_result = None
            for _ in range(5):
                response = self.generate_response(prompt)
                reliability_response = extract(response, "score")
                last_result = self.load_response(reliability_response)
                if self._is_valid_response(last_result):
                    return last_result
                logger.info("提取 strategy overlap 评估结果失败，重新生成!")

            return last_result
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return None

class EvalCombinationAgent(BaseAgent):
    def _to_float(self, value, default=0.0):
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _is_valid_response(self, response):
        if not isinstance(response, dict):
            return False

        score = response.get("strategy_combination_score")
        if not isinstance(score, (int, float)):
            return False
        if float(score) < 0.0 or float(score) > 1.0:
            return False

        reason = response.get("reason")
        if not isinstance(reason, str):
            return False

        return True

    def _normalize_label_idx(self, value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if value.lstrip("-").isdigit():
                return int(value)
        return None

    def _normalize_label_result(self, item):
        if not isinstance(item, dict):
            return None

        label_idx = self._normalize_label_idx(item.get("label_idx"))
        if label_idx is None:
            return None

        score = self._to_float(item.get("strategy_combination_score"), 0.0)
        score = max(0.0, min(1.0, score))
        reason = str(item.get("reason", "")).strip()
        return {
            "label_idx": label_idx,
            "strategy_combination_score": round(score, 4),
            "reason": reason,
        }

    def _is_valid_group_response(self, response):
        if not isinstance(response, dict):
            return False

        label_results = response.get("label_results")
        if not isinstance(label_results, list) or not label_results:
            return False

        seen = set()
        for item in label_results:
            normalized_item = self._normalize_label_result(item)
            if not normalized_item:
                return False
            label_idx = normalized_item["label_idx"]
            if label_idx in seen:
                return False
            seen.add(label_idx)

        return True

    def load_response(self, json_str: str) -> Dict:
        try:
            parsed = json.loads(json_str)
            score = self._to_float(parsed.get("strategy_combination_score"), 0.0)
            score = max(0.0, min(1.0, score))
            reason = str(parsed.get("reason", "")).strip()
            return {
                "strategy_combination_score": round(score, 4),
                "reason": reason,
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None

    def load_group_response(self, json_str: str) -> Dict:
        try:
            parsed = json.loads(json_str)
            raw_label_results = parsed.get("label_results", [])
            if not isinstance(raw_label_results, list):
                return None

            normalized_results = []
            seen = set()
            for item in raw_label_results:
                normalized_item = self._normalize_label_result(item)
                if not normalized_item:
                    continue
                label_idx = normalized_item["label_idx"]
                if label_idx in seen:
                    continue
                seen.add(label_idx)
                normalized_results.append(normalized_item)

            return {
                "label_results": normalized_results,
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None

    def process(self, input_data: Dict) -> Dict:
        input_tuning_strategies = input_data.get("input_strategies")
        label_tuning_strategies = input_data.get("label_strategies")
        label_idea_groups = input_data.get("label_idea_groups")

        if isinstance(label_idea_groups, list) and label_idea_groups:
            prompt = f"""
You are an expert evaluator in high-temperature superconductivity and scientific idea assessment.

Your task is to evaluate Combination Similarity between one predicted research idea and a group of ground-truth future research ideas.

Combination Similarity measures whether the ideas share a similar multi-strategy design skeleton. In other words, it asks whether several similar strategy elements are intentionally organized into a similar central design skeleton or synergistic co-design pattern.

You will see one predicted idea and up to four label ideas at a time. Score each label idea independently. Do not mix different label ideas together, and do not output only the highest-scoring one.

Focus only on combination logic:
- which strategy elements are connected
- how they support each other
- what core synergy they form
- whether the predicted idea captures the main co-design pattern of the ground-truth idea

Do not evaluate wording similarity, sentence similarity, or broad topical similarity.
Do not give a high score merely because both sides mention several similar isolated strategies.

A valid "combination" is not a simple list of components. It must reflect a meaningful relational structure, such as:
- multi-scale pinning design
- interface engineering + strain control + texture optimization
- composition tuning + process tuning + targeted microstructure design
- hybrid APC architecture + epitaxial control
- defect design + process compensation

Predicted strategies to evaluate:
<pred_strategies>
{json.dumps(input_tuning_strategies, ensure_ascii=False, indent=2)}
</pred_strategies>

Label ideas to score independently:
<label_idea_groups>
{json.dumps(label_idea_groups, ensure_ascii=False, indent=2)}
</label_idea_groups>

For each label idea in `label_idea_groups`, perform these steps:
1. Extract the main strategy elements of that label idea.
2. Identify its core combination skeleton and explicit synergy logic.
3. Compare it with the predicted strategies:
   - Are similar categories of strategies connected?
   - Is similar synergy logic present?
   - Is the matched structure central or peripheral?
   - Does the predicted idea truly anticipate the main co-design pattern of this label idea?
4. Provide that label idea's own `strategy_combination_score` and `reason`.

Scoring rule (0.0-1.0):
- 0.0 = no meaningful shared combination structure
- 0.1-0.2 = extremely weak similarity; only marginal or local resemblance
- 0.3-0.4 = some local combination similarity, but the main skeleton is different
- 0.5-0.6 = moderate similarity; part of the central co-design logic overlaps, but key links differ or are missing
- 0.7-0.8 = strong similarity; most of the core combination skeleton and synergy logic are aligned
- 0.9 = very high similarity; the core combination logic is very close, with only minor differences
- 1.0 = the core combination skeleton and synergy logic are almost identical

Evaluation steps:
1. Extract the main strategy elements from the predicted idea.
2. Identify its core combination skeleton and explicit synergy logic.
3. Do the same for the ground-truth idea.
4. Compare them:
   - Are similar categories of strategies connected?
   - Is similar synergy logic present?
   - Is the matched structure central or peripheral?
   - Does the predicted idea truly anticipate the main co-design pattern of the ground-truth idea?
5. Clearly distinguish:
   - strongly matched combination skeletons
   - partially similar local combinations
   - core combination structures present in the ground-truth idea but missing from the predicted idea
6. Output:
   - combination_similarity_score, a decimal between 0 and 1
   - a concise reason

Reason-writing requirements:
Use short contrastive sentences. Do not write long paragraphs, abstract metaphors, or stacked adjectives.
Clearly state three parts:
1. the core combination skeleton of the predicted idea;
2. the core combination skeleton of the ground-truth idea;
3. overlap, differences, and missing key combination structures.
Use direct labels such as "Predicted side / Ground-truth side / Same points / Differences / Missing points".
Prefer "structure name + function" rather than conceptual explanation.
Avoid abstract phrases that are hard to judge unless immediately followed by concrete structural details.
The conclusion must explicitly state whether the overall Combination Similarity is high, medium, or low.

Recommended reason template:

Predicted side: ...; ground-truth side: ...
Same points: ...
Differences: ...
Conclusion: ..., therefore the Combination Similarity is ...

Stay conservative. Do not exaggerate local or scattered resemblance into high overall Combination Similarity.

Output strictly in the following format:
<score>
{{
    "label_results": [
        {{
            "label_idx": 1,
            "strategy_combination_score": 0.0,
            "reason": ""
        }}
    ]
}}
</score>

Notes:
1. `label_results` must cover every label idea you successfully evaluate, using the input `label_idx`.
2. `strategy_combination_score` must be a decimal between 0 and 1.
3. `reason` must explain the scoring basis for that label idea.
4. Each label idea must be analyzed independently; do not merge up to four label ideas into a single score.
5. Output only the JSON wrapped in `<score>...</score>`. Do not output extra explanation or markdown code fences.
"""
            try:
                last_result = None
                for _ in range(5):
                    response = self.generate_response(prompt)
                    reliability_response = extract(response, "score")
                    last_result = self.load_group_response(reliability_response)
                    if self._is_valid_group_response(last_result):
                        return last_result
                    logger.info("提取 strategy_combination group 评估结果失败，重新生成!")

                return last_result
            except Exception as e:
                logger.error(f"处理过程中发生错误: {e}")
                return None

        prompt = f"""
You are an expert evaluator in high-temperature superconductivity and scientific idea assessment.

Your task is to evaluate Combination Similarity between a predicted research idea and a ground-truth future research idea.

Combination Similarity measures whether the two ideas share a similar multi-strategy design skeleton. In other words, it asks whether several similar strategy elements are intentionally organized into a similar central design skeleton or synergistic co-design pattern.

Focus only on combination logic:
- which strategy elements are connected
- how they support each other
- what core synergy they form
- whether the predicted idea captures the main co-design pattern of the ground-truth idea

Do not evaluate wording similarity, sentence similarity, or broad topical similarity.
Do not give a high score merely because both sides mention several similar isolated strategies.

A valid "combination" is not a simple list of components. It must reflect a meaningful relational structure, such as:
- multi-scale pinning design
- interface engineering + strain control + texture optimization
- composition tuning + process tuning + targeted microstructure design
- hybrid APC architecture + epitaxial control
- defect design + process compensation

Predicted strategies to evaluate:
<pred_strategies>
{json.dumps(input_tuning_strategies, ensure_ascii=False, indent=2)}
</pred_strategies>

Ground-truth label strategies for comparison:
<label_strategies>
{json.dumps(label_tuning_strategies, ensure_ascii=False, indent=2)}
</label_strategies>

When comparing the two, identify:

1. Core combination skeleton
- What are the main strategy elements in each idea?
- Which elements are intentionally connected?
- Which links are central rather than incidental?

2. Synergy logic
- Do the connected strategies serve similar scientific functions?
- Do they jointly regulate the same type of structural or functional problem?
- Does the predicted idea reproduce the main co-design logic of the ground-truth idea?

3. Structural organization
- Are similar categories of strategies connected?
- Are they connected for similar scientific reasons?
- Is the matched combination central in both ideas, or only local/secondary?

Notes:
- Overlap between isolated strategy elements does not equal high Combination Similarity.
- If the predicted idea merely lists many scattered combinations without clearly matching the core design skeleton of the ground-truth idea, reduce the score.
- Give a high score only when the core multi-component design logic is clearly similar.

Scoring rule (0.0-1.0):
- 0.0 = no meaningful shared combination structure
- 0.1-0.2 = extremely weak similarity; only marginal or local resemblance
- 0.3-0.4 = some local combination similarity, but the main skeleton is different
- 0.5-0.6 = moderate similarity; part of the central co-design logic overlaps, but key links differ or are missing
- 0.7-0.8 = strong similarity; most of the core combination skeleton and synergy logic are aligned
- 0.9 = very high similarity; the core combination logic is very close, with only minor differences
- 1.0 = the core combination skeleton and synergy logic are almost identical

Evaluation steps:
1. Extract the main strategy elements from the predicted idea.
2. Identify its core combination skeleton and explicit synergy logic.
3. Do the same for the ground-truth idea.
4. Compare them:
   - Are similar categories of strategies connected?
   - Is similar synergy logic present?
   - Is the matched structure central or peripheral?
   - Does the predicted idea truly anticipate the main co-design pattern of the ground-truth idea?
5. Clearly distinguish:
   - strongly matched combination skeletons
   - partially similar local combinations
   - core combination structures present in the ground-truth idea but missing from the predicted idea
6. Output:
   - combination_similarity_score, a decimal between 0 and 1
   - a detailed reason explaining the score.

Stay conservative. Do not exaggerate local or scattered resemblance into high overall Combination Similarity.

Output strictly in the following format:
<score>
{{
    "strategy_combination_score": 0.0,
    "reason": ""
}}
</score>

Notes:
1. `strategy_combination_score` must be a decimal between 0 and 1.
2. `reason` must explain the score in detail.
3. Output only the JSON wrapped in `<score>...</score>`. Do not output extra explanation or markdown code fences.

"""
        try:
            last_result = None
            for _ in range(5):
                response = self.generate_response(prompt)
                reliability_response = extract(response, "score")
                last_result = self.load_response(reliability_response)
                if self._is_valid_response(last_result):
                    return last_result
                logger.info("提取 strategy_combination 评估结果失败，重新生成!")

            return last_result
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return None

class EvalExpAgent_reproducibility(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def _to_float(self, value, default=0.0):
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _strip_code_fences(self, text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def _is_valid_response(self, response):
        if not isinstance(response, dict):
            return False

        score = response.get("score")
        if not isinstance(score, (int, float)):
            return False
        if float(score) < 0.0 or float(score) > 1.0:
            return False

        return isinstance(response.get("reason"), str)

    def load_response(self, json_str: str) -> Dict:
        cleaned = self._strip_code_fences(json_str)
        candidates = [cleaned]
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                score = max(0.0, min(1.0, self._to_float(parsed.get("score"), 0.0)))
                reason = str(parsed.get("reason", "")).strip()
                return {
                    "score": round(score, 4),
                    "reason": reason,
                }
            except json.JSONDecodeError:
                continue

        logger.error("实验评估结果 JSON 解析失败")
        return None

    def process(self, input_data: Dict) -> Dict:
        pred_exp = str(
            input_data.get("pred_exp")
            or input_data.get("output")
            or input_data.get("model_output_experiment")
            or input_data.get("exp")
            or ""
        ).strip()
        label_exp = str(
            input_data.get("label_exp")
            or input_data.get("label")
            or input_data.get("groundtruth_experiment")
            or ""
        ).strip()

        prompt = """
Your task is to evaluate the consistency between the given **model-output experiment** and the **groundtruth experiment**, using the groundtruth as the core reference standard.

## Evaluation Goal
This evaluation focuses on whether the model output preserves the **core experimental line** of the groundtruth experiment, including:
- experimental objective
- sample design and functional roles
- fabrication method and main process route
- key testing and characterization logic
- key parameter ranges
- data-analysis and conclusion-support pathway

Note:
This evaluation is not a simple literal comparison, and it is not about which plan is more complete, more creative, or more like a research proposal.
The key question is: **does the model_output still reproduce the same experiment, rather than rewriting it into a different experimental plan?**

---

Inputs:

Model-output experiment:
<model_output_experiment>
{output}
</model_output_experiment>

Groundtruth experiment:
<groundtruth_experiment>
{label}
</groundtruth_experiment>

---

## Task Steps

### Step 1. Normalize and Clean
Normalize both experimental descriptions:
- Standardize units and notation:
  - temperature: degrees C or K
  - time: min / h
  - pressure: Pa / mbar / Torr
  - concentration: wt.% / mol% / at.%
  - laser parameters: wavelength in nm, fluence in J cm^-2, repetition rate in Hz
  - thickness: nm / um
- Normalize common material aliases and abbreviations, such as YBCO = YBa2Cu3O7-delta.
- Ignore information that does not affect the core judgment:
  - brands, suppliers, equipment models
  - auxiliary descriptions
  - non-core descriptive expansions

### Step 2. Extract Key Information from Both Sides
Extract the following from both the model output and the groundtruth:
- experimental objective / core problem
- materials and samples
- sample design, such as thickness sets or layered structures
- fabrication method / process route
- key steps and sequence
- key process parameters
- testing and characterization methods
- key testing conditions
- data-processing / analysis logic
- key conclusion or expected phenomenon

### Step 3. Compare Item by Item
Focus on these dimensions:
1. experimental objective and target criteria
2. materials and sample functional roles
3. fabrication method / process route
4. key process parameters
5. comparability of testing and characterization
6. data-analysis and inference logic
7. procedural steps and sequence logic

---

## Scoring Principles

### 1. Groundtruth First
Use the groundtruth as the standard to judge whether the model_output covers and preserves its core experimental information.

### 2. Core Line Over Detail Volume
If an output is shorter but preserves the core experimental line of the groundtruth, it should score higher than a longer output that clearly expands into a new proposal.

### 3. Reasonable Expansion Is Acceptable but Usually Not Rewarded
If model_output adds background explanation, engineering wording, supplementary instruments, or implementation details not present in the groundtruth:
- do not heavily penalize these additions if they **do not change the experimental line**
- do not add extra credit merely because the output is "more complete"

### 4. Penalize Clear Rewriting of the Core Line
Penalize model_output if it:
- introduces new core control or comparison groups
- rewrites the original experiment into another research design
- replaces key testing logic
- replaces the main analysis framework
- changes the core structure of the sample design

### 5. Compare Parameters by Whether They Support the Same Experimental Mechanism
If parameters differ, judge whether they still support:
- the same experimental objective
- a similar material structure / defect configuration
- the same testing logic
- a similar conclusion-support pathway

If they still support a similar experimental mechanism, do not heavily penalize literal numerical differences alone.

---

## Tolerance and Functional-Equivalence Rules

### Numerical Tolerance
Apply moderate tolerance to key parameters:
- temperature: +/-20%
- time: +/-25%
- pressure / flow / concentration / frequency / thickness: +/-20%

### Functional Equivalence
If some parameters, material names, or descriptions differ but still play the same functional role and serve the same experimental line, treat them as partial or high matches.

However:
If these differences clearly change the experimental line, sample design, testing logic, or analysis path, do not give a high score merely because they "might achieve a similar goal".

---

## Dimension Weights
- experimental objective and target criteria: 0.15
- materials and sample functional roles: 0.15
- fabrication method / process route: 0.25
- key process parameters: 0.15
- testing and characterization comparability: 0.15
- data-analysis and inference logic: 0.10
- procedural steps and sequence logic: 0.05

Total-score formula:
`score = sum(dimension score x weight)`

---

## Score References
- 1.00 = core objective, experimental line, testing logic, and key parameters are highly consistent
- 0.85 = the main line is highly consistent, with some supplementation, generalization, or local rewriting that does not change the core experimental logic
- 0.70 = the main line is basically consistent, but there are several clear expansions or local replacements
- 0.55 = only part of the main line is consistent; key workflow or testing logic has clearly shifted
- 0.35 = only locally related; it is difficult to view the output as the same experiment
- 0.00 = essentially unrelated

---

## Output Requirements
Output only concise JSON wrapped with `<res>` and `</res>`.
Do not output True / False.
Do not output a complex nested structure.

Output format:

<res>
{
  "score": 0.00,
  "reason": "Explain in detail the agreements and deviations between model_output and groundtruth in experimental objective, sample design, process route, key parameters, testing methods, and analysis logic; state whether added content changes the original experimental line; finally explain why these differences lead to the current score."
}
</res>
""".replace("{output}", pred_exp).replace("{label}", label_exp)

        last_result = None
        for _ in range(5):
            response = self.generate_response(prompt)
            result_body = extract(response, "res", hard=False) or response
            last_result = self.load_response(result_body)
            if self._is_valid_response(last_result):
                return last_result
            logger.info("提取实验评估结果失败，重新生成!")
        return last_result


class EvalExpAgent_feasibility(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def _to_float(self, value, default=0.0):
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _strip_code_fences(self, text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def _is_valid_response(self, response):
        if not isinstance(response, dict):
            return False

        score = response.get("score")
        if not isinstance(score, (int, float)):
            return False
        if float(score) < 0.0 or float(score) > 1.0:
            return False

        return isinstance(response.get("reason"), str)

    def load_response(self, json_str: str) -> Dict:
        cleaned = self._strip_code_fences(json_str)
        candidates = [cleaned]
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                score = max(0.0, min(1.0, self._to_float(parsed.get("score"), 0.0)))
                reason = str(parsed.get("reason", "")).strip()
                return {
                    "score": round(score, 4),
                    "reason": reason,
                }
            except json.JSONDecodeError:
                continue

        logger.error("实验评估结果 JSON 解析失败")
        return None

    def process(self, input_data: Dict) -> Dict:
        pred_exp = str(
            input_data.get("pred_exp")
            or input_data.get("output")
            or input_data.get("model_output_experiment")
            or input_data.get("exp")
            or ""
        ).strip()
        label_exp = str(
            input_data.get("label_exp")
            or input_data.get("label")
            or input_data.get("groundtruth_experiment")
            or ""
        ).strip()

        prompt = """
# Task: Evaluate the Executability of an Experimental Plan

Your task is to evaluate whether a given **experimental plan** has the qualities of a high-quality, implementable, executable plan.

This evaluation **only focuses on the quality of the plan itself**. It does not compare the plan with any reference plan or judge consistency with another plan.

---

## Evaluation Goal

Judge whether the experimental plan is a high-quality experimental document that is **executable, divisible into tasks, actionable, and convertible into an SOP or implementation plan**. Evaluate it strictly against the following reference template for a full-score plan.

---

## Full-Score Reference Template for Experimental-Plan Executability

A top-tier experimental plan should have the following five features:

### 1. Overall Plan Structure (clear hierarchy, closed logic)
- **Experimental overview/objective**: clearly states the core scientific or engineering problem, final goal, and key innovation.
- **Initial experimental draft/summary**: provides a high-level summary of materials, workflow, and expected outcomes.
- **Detailed experimental plan (task decomposition)**: decomposes the full experiment into logically connected, independently executable subtasks, such as Task 1: material preparation, Task 2: sample processing, Task 3: performance testing.
- **Refined experimental plan (SOP-style summary)**: summarizes key steps, parameters, and criteria in parameter tables, workflow tables, or similar forms to create a standardized document that can directly guide operations.

### 2. Task Modularity and Divisibility (clear breakdown, clear responsibility)
- **Clear subtask division**: each subtask includes its own objective, materials and instruments, steps, and expected results.
- **Connections between tasks**: clearly explains how outputs from earlier tasks become inputs for later tasks.
- **Independent operability**: each subtask is complete enough to be assigned to different team members.

### 3. SOP Level of Steps and Parameters (specific, quantitative, reproducible)
- **Detailed step breakdown**: each experimental step is decomposed into numbered concrete operations, such as 3.1 precursor preparation, 3.2 calcination, 3.3 molding.
- **Parameter tables**: key process parameters should be listed clearly in table form, including temperature, time, pressure, rate, atmosphere, concrete values, and tolerance ranges.
- **Instrument and material lists**: clearly list required material specifications, such as purity and dimensions, and key instrument performance specifications.

### 4. Risk Control and Quality Management (anticipate problems, set checkpoints)
- **Precautions**: after each key step, explicitly state possible issues and preventive measures.
- **Quality-control checkpoints**: set checkpoints after key steps, such as verifying phase by XRD or roughness by AFM.
- **Troubleshooting and contingency plans**: provide alternatives or adjustment strategies for undesirable results.

### 5. Expected Results and Feedback Loop (goal-oriented, measurable)
- **Quantified expected results**: explicitly state target performance indicators, such as Jc >= 1e4 A cm^-2 or delta-phi < 5 degrees.
- **Fabrication-characterization-feedback loop**: the workflow should reflect an iterative logic of fabrication -> characterization -> result analysis -> parameter optimization.
- **Engineering and scale-up awareness**: consider industrialization factors such as process scalability, long tape, and large-area uniformity.

---

## Score Calibration and Mandatory Defect Check (anti-inflation mechanism)

**Important instruction: to prevent inflated scores, you must strictly follow the calibration process below.**

### Step 0. Mandatory Identification of Executability Defects (deductions first)
Before calculating the score, compare the experimental plan item by item against the full-score template and explicitly list its **executability defects**.

**Defect deduction reference table (for calibrating your scoring perception):**
| Defect Category | Concrete Symptom | Suggested Deduction | Score-Band Implication |
| :--- | :--- | :--- | :--- |
| **Broken structure** | Only flat narrative description; no task decomposition and no overview-detail-summary structure. | 0.20 - 0.30 | Falls below 0.70 |
| **Parameter shell** | Mentions process names but gives **no** concrete values, such as missing temperature, time, and pressure. | 0.20 - 0.30 | Falls below 0.70 |
| **No closed loop** | Contains only fabrication steps and no characterization, testing, or validation. | 0.15 - 0.25 | Very difficult to exceed 0.75 |
| **Coarse granularity** | Steps remain slogan-level, such as "optimize process parameters", without operational details. | 0.15 - 0.20 | Ceiling is 0.80 |
| **No risk awareness** | No precautions, possible issues, or quality-control descriptions. | 0.10 - 0.15 | Ceiling is 0.85 |
| **Not quantifiable** | Expected results are only qualitative, such as "performance improves", with no concrete metrics or criteria. | 0.05 - 0.10 | Ceiling is 0.90 |

### Recalibrated Score-Band Anchors (mandatory comparison)
Before giving the final score, force-align the plan with these descriptions:

- **0.90 - 1.00 (excellent/outstanding)**: **rare**. Must satisfy: strong fit with the full-score template, tabulated parameters, complete risk and quality control. **If there is no engineering/SOP-style expression, do not give above 0.95.**
- **0.80 - 0.89 (good/strong)**: has strengths but visible flaws. Example: clear structure and concrete parameters, but missing risk prompts or troubleshooting; or some parameters are missing and would require literature lookup before execution.
- **0.70 - 0.79 (qualified/executability threshold)**: the most common floor range. The plan is coherent, has key parameters, and follows a reasonable experimental logic, but lacks detail granularity. This is the minimum passing line for executability.
- **0.55 - 0.69 (unqualified/unreliable)**: has clear defects, such as missing key parameters that prevent operation, workflow jumps, or no quality-control measures.
- **0.00 - 0.54 (very poor/unrelated)**: conceptual description that can hardly guide an experiment.

### Scoring-Logic Correction
Follow this logic:
1. **If serious defects such as "parameter shell" or "broken structure" are found** -> **cap the score at 0.69**.
2. **If key parameters are present but operational details are absent** -> **score should be in the 0.70 - 0.78 range**.
3. **Strictly control high scores**: scores above 0.85 require strict verification that the plan fully satisfies the corresponding score-band requirements, avoiding subjective score inflation.

---

## Input

Experimental plan to evaluate:
<experiment_plan>
{plan}
</experiment_plan>

---

## Task Steps

### Step 1. Normalize and Clean
Normalize the experimental content to unify the evaluation basis:

- Standardize units and notation:
  - temperature: degrees C or K
  - time: min / h
  - pressure: Pa / mbar / Torr
  - concentration: wt.% / mol% / at.%
  - laser parameters: wavelength in nm, fluence in J cm^-2, repetition rate in Hz
  - thickness: nm / um
- Normalize common material aliases and abbreviations, such as YBCO = YBa2Cu3O7-delta.
- Ignore information that does not affect the core judgment:
  - brands, suppliers, equipment models
  - auxiliary descriptions
  - non-core descriptive expansions

### Step 2. Extract Executability Features and Defects
Evaluate against the **Full-Score Reference Template for Experimental-Plan Executability** and forcibly identify defects:

- **Plan structure**: does it have a clear hierarchy such as overview-task decomposition-SOP summary?
- **Task decomposition**: is it broken into independent subtasks? Does each task have clear objectives, materials, steps, and expectations?
- **Step clarity**: are steps numbered and logically clear?
- **Parameter specificity**: are parameters provided in lists or tables with clear values and units?
- **Risk and quality control**: does it include precautions, quality-control checkpoints, or contingency plans?
- **Closed loop and engineering awareness**: does it reflect fabrication-characterization-feedback logic and scale-up awareness?

**Mandatory output:** in this step, identify and record at least 1-3 major defects in the experimental plan, based on the defect deduction reference table.

### Step 3. Dimension Scoring
Score the following seven dimensions separately (0.00 - 1.00), then calculate the weighted total score.

| Dimension | Weight | Scoring Focus |
| :--- | :--- | :--- |
| Structural completeness and clear hierarchy | 0.15 | Whether it has a complete structure of overall goal -> subtasks -> steps -> summary. |
| Task decomposition and collaboration assignability | 0.15 | Whether task division is clear and independently assignable. |
| Executability / operational granularity / SOP tendency | 0.25 | Whether steps are numbered, parameters are tabulated, and operations are directly executable. |
| Parameter specificity and condition sufficiency | 0.18 | Whether key process parameters, such as temperature, time, and pressure, are concrete and quantified. |
| Risk prompts, precautions, troubleshooting | 0.12 | Whether potential problems are anticipated and countermeasures are given. |
| Quality control and criteria setting | 0.09 | Whether process quality-control checkpoints and result criteria are set. |
| Fabrication-characterization-feedback-scale-up validation loop | 0.06 | Whether it reflects a fabrication-testing-optimization loop and engineering scale-up awareness. |

Total-score formula:
`score = sum(dimension score x weight)`

---

## Scoring Principles

1. **Defects must cause deductions**: strictly follow the defect deduction reference table. A high score without listed defects is invalid.
2. **Parameters are core**: a "parameter shell" is a fatal defect; plans with this issue should not score above 0.69.
3. **High scores require strict evidence**: for scores above 0.85, explicitly list concrete evidence in the reason showing that the plan meets "outstanding" standards, such as tabulated parameters, contingency plans, and SOP-style summaries, and confirm that there are no obvious flaws.
4. **No comparison with other plans**: this evaluation only concerns the quality of the plan itself, not whether it resembles a reference plan.
5. **Strictly avoid score inflation**: score each dimension conservatively. For scores above 0.80, repeatedly verify that the corresponding requirements are satisfied and that the score matches the plan's actual quality.

---

## Score References (calibrated, mandatory)

- **1.00 (rare)**: perfect implementation of the full-score template. Parameters are tabulated, risk and quality control are complete, and the engineering loop is clear.
- **0.90 - 0.99**: extremely strong executability; parameters are concrete, structure is clear, and only minor flaws exist, such as one missing risk note. **Verify strictly and do not give lightly.**
- **0.80 - 0.89**: strong plan quality with mostly complete parameters, but lacking engineering awareness or quality-control details. **Confirm that there are no serious defects and that parameters are sufficiently concrete.**
- **0.70 - 0.79**: **passing line**. The plan has key parameters and logic and can be implemented, but operational granularity is coarse.
- **0.55 - 0.69**: average executability. Missing key steps or parameters prevents direct execution.
- **0.35 - 0.54**: weak plan. Mostly conceptual and unable to guide an experiment.
- **0.00 - 0.34**: completely non-executable.

**Warning: most qualified plans should fall between 0.70 and 0.85. Do not give scores above 0.90 lightly; high scores must be strictly controlled.**

---

## Output Requirements
Output only concise JSON wrapped with `<res>` and `</res>`.
Do not output True / False.
Do not output a complex nested structure.

Output format:

<res>
{
  "score": 0.00,
  "reason": "First state the main defects identified; then review performance in structural completeness, task decomposition, operational granularity, parameter specificity, risk control, quality control, and engineering loop; finally explain why this score was assigned and emphasize strict control of high scores."
}
</res>
""".replace("{plan}", pred_exp)

        last_result = None
        for _ in range(5):
            response = self.generate_response(prompt)
            result_body = extract(response, "res", hard=False) or response
            last_result = self.load_response(result_body)
            if self._is_valid_response(last_result):
                return last_result
            logger.info("提取实验评估结果失败，重新生成!")
        return last_result



class ExtractPaperExpAgent(BaseAgent):
    def process(self, input_data: Dict) -> Dict:
        paper = str(input_data.get("pdf", "")).strip()

        prompt = f"""You are a **condensed matter physics research expert** specializing in materials physics, superconductivity, and experimental design.

You will receive the full text extracted from a **complete scientific paper PDF**. Your task is to read the whole paper and output a **scientifically accurate, clearly structured experimental-plan extraction that is faithful to the original paper**.

**Important constraints:**
1. Use only information explicitly provided in the paper PDF.
2. **Do not fabricate, expand, optimize, redesign, or infer content that is not explicitly supported by the paper.**
3. If the paper lacks certain experimental details, explicitly state that the information is **not specified in the paper**.
4. Extract information from the full PDF, focusing especially, but not exclusively, on:
   - Title
   - Abstract
   - Methods / Experimental Methods
   - Results
   - Supplementary information, if included in the PDF
5. If information about the same experimental workflow is distributed across multiple sections, perform **cross-section integration and reconstruction**, but explicitly state that the reconstruction is based on distributed descriptions in the paper.
6. Do not miswrite background discussion, theoretical analysis, mechanism explanation, or conclusion statements as experimental steps. Extract only content directly related to **experimental design, sample preparation, device fabrication, measurement methods, data acquisition, and data-analysis workflow**.
7. **Do not output any evaluation, feasibility judgment, reproducibility judgment, pros/cons analysis, or improvement suggestions.** Your task is only to **extract the experimental plan**.

## Task Requirements

### 1. Experimental-Plan Extraction
- Based on the complete paper PDF, extract and reorganize the experimental plan into a detailed and clearly structured summary wrapped in `<exp> ... </exp>` tags.
- The extracted plan must faithfully reflect the **experiments, methods, and measurement workflow actually described in the paper**.
- Extract the workflow step by step in the actual order in which the experiment was performed.
- Include, but do not limit yourself to:
  - research materials, samples, systems, or device objects
  - sample preparation, synthesis, or growth steps
  - device processing and structural construction, if applicable
  - characterization steps
  - measurement steps
  - data-acquisition workflow
  - data-analysis workflow
- If the paper does not explicitly provide a complete order, you may integrate information from multiple sections, but you must explicitly state:
  - `Reconstructed from multiple sections of the PDF`


#### 2. Language Requirements
- Use plain, clear language.
- Assume the reader may not be an expert in condensed matter physics.
- Avoid unnecessary jargon.
- If specialized terms are necessary, briefly explain them.

## Fidelity Rules
1. **Do not generate a new experimental plan.**
2. **Do not optimize, rewrite, or redesign the experiment in the paper.**
3. **Do not provide any evaluation, judgment, criticism, or suggestion.**
4. **Do not fill gaps with assumptions and present them as facts.**
5. If information in the paper is uncertain, vague, or missing, explicitly write:
   - `Not specified in the paper`
   - or `Not explicitly described by the authors`
6. If an experimental step is assembled from information in multiple PDF locations, explicitly write:
   - `Reconstructed from multiple sections of the PDF`

## Output Format
Output strictly in the following format:

<exp>

[Experimental Workflow]
Step1: ...
Step2: ...
Step3: ...

</exp>

## Formatting Rules
1. All final output must be in **English**.
2. Inside `<exp>`, organize sections with **clear subheadings**.
3. Include parameter values only when they are supported by the PDF.
4. If the PDF does not give a parameter, do not guess; explicitly mark it as not specified.
5. **Do not** include any information from 2025 or later.
6. Do not omit key experimental conditions from figure captions, tables, or supplementary information.
7. If the paper contains both experimental and theoretical/computational parts, prioritize **experiment-related content**; mention theory only when it directly supports explaining an experimental step.

Full paper text extracted from the PDF:
<paper_pdf_text>
{paper}
</paper_pdf_text>
"""
        response = ""
        for _ in range(5):
            response = self.generate_response(prompt)
            experiment = extract(response, "exp", hard=False)
            if experiment:
                return {"experiment": experiment.strip()}
            logger.info("提取论文实验方案失败，重新生成!")
        return {"experiment": response.strip() if response else ""}

class ExtractPaperAgent(BaseAgent):
  
    """审核智能体 - 检查创新性"""
    def process(self, input_data: Dict) -> Dict:
        paper = input_data.get("pdf")

        prompt = f"""### Task Description
I will provide a materials-science paper. Your task is to summarize the paper into a structured JSON representation according to the requirements below.
Paper content:
{paper}

### Output Content Format
Summarize the paper as a JSON object containing four top-level key-value pairs.
1. The first key-value pair should describe the material system, including the REBCO material system and dopant materials.
   - In "description", describe the material system, lattice parameters, possible defects, texture, and related features.
   - Fill "base_material" and "dopants" with the base material and dopant materials; leave them empty if unavailable.
 
2. The second key-value pair should describe the paper's regulation strategies and regulation effects.
   - In "description", explain how the physical-property regulation strategies work together to optimize the experimental objective.
   - The value of "strategies" is an array [] containing multiple JSON objects. Each object contains "tuning dimensions" and "tuning strategies". Add as many objects as there are extractable regulation strategies.
   - tuning strategies: use summary language to describe each regulation strategy concretely, and include its deeper effect while describing the concrete strategy.
   - tuning dimensions: choose one of the five tuning dimensions for each strategy: Composition tuning, grain boundary tuning, interface tuning, texture tuning, stress tuning.
   - Tuning-dimension definitions: Composition tuning adjusts chemical composition to introduce microscopic defects or change crystal structure, thereby enhancing flux pinning. Grain boundary tuning optimizes interfaces between superconducting grains to reduce current obstruction. Interface tuning optimizes the contact interfaces among the REBCO superconducting layer, metal substrate, and buffer layers to support high-quality crystal growth and reduce stress. Texture tuning controls grain orientation and texture consistency to reduce grain-boundary density and improve current transport. Stress tuning controls residual internal stress, such as thermal or mechanical stress, to prevent cracking or performance degradation.
 
3. The third key-value pair should describe the experimental techniques.
   - In "description", describe the specific experimental techniques used in the paper, grouped into sample-fabrication, characterization, and physical-property measurement techniques. Emphasize innovative techniques.
   - Fill "FabricationTech", "CharacterizationTech", and "MeasurementTech" with the specific types of sample-fabrication, characterization, and measurement techniques.
 
4. The fourth key-value pair should describe the application scenario and application metrics.
   - In "description", describe the application need.
   - In "ApplicationScenario", briefly describe the specific application scenario.
   - In "ApplicationIndex", provide performance metrics, such as critical current density.
 
 
 
### Notes
1. Every field must be directly grounded in evidence from the paper.
2. Fill all content in English.
3. Output JSON format.
4. Wrap the entire JSON object with <template> and </template>.
5. If the "strategies": [] array contains multiple JSON objects, separate them with commas to keep the JSON valid.
6. JSON may use only e notation for scientific notation, such as 2e6; do not use multiplication signs.
 
### Final Output Format
<template>
{{
    "MaterialSystem": {{
        "description": "",
        "base_material": "",
        "dopants": ""
    }},
    "TunningMethod": {{
        "description": "",
        "strategies": [
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }},
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }},
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }},
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }},
            {{
                "tuning dimensions": "",
                "tuning strategies": "",
                "tuning effects": ""
            }}
        ]
    }},
    "ExperimentTech": {{
        "description": "",
        "FabricationTech": "",
        "CharacterizationTech": "",
        "MeasurementTech": ""
    }},
    "Application": {{
        "description": "",
        "ApplicationScenario": "",
        "ApplicationIndex": "",
    }}
}}
</template>
 
Start now: analyze the paper and generate a JSON template object following the format above. The output must begin with <template> and end with </template>.
"""
        response = self.generate_response(prompt)
        template = extract(response,"template")
        while not template:
            logger.info(f"提取score失败，重新生成!")
            response = self.generate_response(prompt)
            template = extract(response,"template")
        return {"template": template}


def _build_legacy_exp_eval_prompts(output, label):
    shared_inputs = f"""
Experimental plan to evaluate:
<experiment>
{output}
</experiment>

Reference paper:
<paper>
{label}
</paper>
"""

    step_prompt = f'''**Task Description**:
I will provide an experimental plan to evaluate and a reference paper that contains the experimental objective and experimental steps. Evaluate whether the steps in the experimental plan can achieve the experimental objective. Output only a score from 1 to 5.

**Your Task**:
{shared_inputs}

**Scoring Criteria**:
- 5: The experimental plan is much more detailed than the reference, and the steps fully support the objective empirically and theoretically with no flaws. It also has some innovation, can further optimize the objective, and includes key mechanism validation plus result analysis and theoretical explanation.
- 4: The experimental plan is as detailed as the reference. The steps basically support the objective empirically and theoretically and are on par with the reference. It also includes some physical-mechanism validation and result analysis.
- 3: Compared with the reference, one key experimental step does not support the objective empirically/theoretically or is unrelated, but the impact on achieving the objective is small. The plan includes experimental analysis.
- 2: Compared with the reference, two key experimental steps do not support the objective or are unrelated, causing substantial impact on achieving the objective. There is no analysis of experimental results.
- 1: Compared with the reference, three or more key experimental steps do not support the objective or are unrelated, making the objective unachievable.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": "3".
4. In "reason", compare the evaluated experimental plan with the reference plan and provide the evidence in English.

**Output Format**:
<score>
{{
    "score": "",
    "reason": ""
}}
</score>

Start now: analyze the plan and output a JSON object following the template above, wrapped with <score> and </score>.
'''

    paramenter_prompt = f'''**Task Description**:
I will provide an experimental plan to evaluate and a reference paper that contains the experimental objective and experimental steps. Evaluate whether the parameters in the experimental plan can achieve the experimental objective. Output only a score from 1 to 5.

**Your Task**:
{shared_inputs}

**Scoring Criteria**:
- 5: The experimental parameters fully support the objective empirically and theoretically with no flaws, and they include some innovation that can further optimize the objective.
- 4: The experimental parameters basically support the objective empirically and theoretically and are on par with the reference.
- 3: One or two key experimental parameters do not support the objective or are unrelated, but the impact on achieving the objective is small.
- 2: Three to five key experimental parameters do not support the objective or are unrelated, causing substantial impact on achieving the objective.
- 1: More than five key experimental parameters do not support the objective or are unrelated, making the objective unachievable.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": "3".
4. In "reason", compare the evaluated experimental plan with the reference plan and provide the evidence in English.

**Output Format**:
<score>
{{
    "score": "",
    "reason": ""
}}
</score>

Start now: analyze the plan and output a JSON object following the template above, wrapped with <score> and </score>.
'''

    complete_prompt = f'''**Task Description**:
I will provide an experimental plan to evaluate and a reference paper that contains the experimental objective and experimental steps. Evaluate whether the experimental plan is missing key experimental steps. Output only a score from 1 to 5.

**Your Task**:
{shared_inputs}

**Scoring Criteria**:
- 5: The experimental steps are fully detailed and flawless, and they include innovation that can further improve the objective.
- 4: The experimental step details are sufficient and on par with the reference.
- 3: One key experimental step is missing, but the impact on achieving the objective is small.
- 2: Two key experimental steps are missing, causing substantial impact on achieving the objective.
- 1: Three or more key experimental steps are missing, making the objective unachievable.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": "3".
4. In "reason", compare the evaluated experimental plan with the reference plan and provide the evidence in English.

**Output Format**:
<score>
{{
    "score": "",
    "reason": ""
}}
</score>

Start now: analyze the plan and output a JSON object following the template above, wrapped with <score> and </score>.
'''

    vulnerability_prompt = f'''**Task Description**:
I will provide an experimental plan to evaluate and a reference paper that contains the experimental objective and experimental steps. Evaluate whether the experimental plan is missing key experimental parameters. Output only a score from 1 to 5.

**Your Task**:
{shared_inputs}

**Scoring Criteria**:
- 5: The experimental parameters are fully detailed and flawless, and they include innovation that can further improve the objective.
- 4: The experimental parameter details are sufficient and on par with the reference.
- 3: One or two key experimental parameters are missing, but the impact on achieving the objective is small.
- 2: Three to five key experimental parameters are missing, causing substantial impact on achieving the objective.
- 1: More than five key experimental parameters are missing, making the objective unachievable.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": "3".
4. In "reason", compare the evaluated experimental plan with the reference plan and provide the evidence in English.

**Output Format**:
<score>
{{
    "score": "",
    "reason": ""
}}
</score>

Start now: analyze the plan and output a JSON object following the template above, wrapped with <score> and </score>.
'''

    synergy_prompt = f'''**Task Description**:
I will provide an experimental plan to evaluate and a reference paper that contains the experimental objective and experimental steps. Evaluate whether the steps or parameters in the experimental plan conflict with one another. Output only a score from 1 to 5.

**Your Task**:
{shared_inputs}

**Scoring Criteria**:
- 5: The steps and parameters are mutually coordinated and show innovative scientific rigor and optimized design.
- 4: The experimental steps and parameters are mutually coordinated and on par with the reference in innovation.
- 3: There is one conflict among the experimental steps or parameters that obstructs achievement of the objective.
- 2: There are two conflicts among the experimental steps or parameters that obstruct achievement of the objective.
- 1: There are three or more conflicts among the experimental steps or parameters that obstruct achievement of the objective.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": "3".
4. In "reason", compare the evaluated experimental plan with the reference plan and provide the evidence in English.

**Output Format**:
<score>
{{
    "score": "",
    "reason": ""
}}
</score>

Start now: analyze the plan and output a JSON object following the template above, wrapped with <score> and </score>.
'''

    return step_prompt, paramenter_prompt, complete_prompt, vulnerability_prompt, synergy_prompt


class ExpEvaluationSoft(BaseAgent):    

    def load_response(self, json_str: str) -> Dict:
        """加载JSON字符串为字典"""
        try:
            parameter_evaluation = json.loads(json_str)
            score = parameter_evaluation.get("score")
            reason = parameter_evaluation.get("reason")
            if score is not None and isinstance(score, str):
                score = int(score)
            if reason is not None and isinstance(reason, str):
                reason = reason.strip()
            return score,reason

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None, None

    def process(self, input_data: Dict) -> Dict:
        output = input_data.get("output")
        label = input_data.get("label")
        
        step_prompt, paramenter_prompt, complete_prompt, vulnerability_prompt, synergy_prompt = _build_legacy_exp_eval_prompts(output, label)

        # 步骤评估
        step_response = self.generate_response(step_prompt)
        step_evaluation = extract(step_response,"score")
        step_score,step_reason = self.load_response(step_evaluation)
        while not step_score or not step_reason:
            logger.info(f"提取step_score失败，重新生成!")
            step_response = self.generate_response(step_prompt)
            step_evaluation = extract(step_response,"score")
            step_score,step_reason = self.load_response(step_evaluation)
        
        # 参数评估
        parameter_response = self.generate_response(paramenter_prompt)
        parameter_evaluation = extract(parameter_response,"score")
        parameter_score,parameter_reason = self.load_response(parameter_evaluation)
        while not parameter_score or not parameter_reason:
            logger.info(f"提取parameter_score失败，重新生成!")
            parameter_response = self.generate_response(paramenter_prompt)
            parameter_evaluation = extract(parameter_response,"score")
            parameter_score,parameter_reason = self.load_response(parameter_evaluation)

        # 完整性评估
        complete_response = self.generate_response(complete_prompt)
        complete_evaluation = extract(complete_response,"score")
        complete_score,complete_reason = self.load_response(complete_evaluation)
        while not complete_score or not complete_reason:
            logger.info(f"提取complete_score失败，重新生成!")
            complete_response = self.generate_response(complete_prompt)
            complete_evaluation = extract(complete_response,"score")
            complete_score,complete_reason = self.load_response(complete_evaluation)
        
        # 脆弱性评估
        vulnerability_response = self.generate_response(vulnerability_prompt)
        vulnerability_evaluation = extract(vulnerability_response,"score")
        vulnerability_score,vulnerability_reason = self.load_response(vulnerability_evaluation)
        while not vulnerability_score or not vulnerability_reason:
            logger.info(f"提取vulnerability_score失败，重新生成!")
            vulnerability_response = self.generate_response(vulnerability_prompt)
            vulnerability_evaluation = extract(vulnerability_response,"score")
            vulnerability_score,vulnerability_reason = self.load_response(vulnerability_evaluation)

        # 协同性评估
        synergy_response = self.generate_response(synergy_prompt)
        synergy_evaluation = extract(synergy_response,"score")
        synergy_score,synergy_reason = self.load_response(synergy_evaluation)
        while not synergy_score or not synergy_reason:
            logger.info(f"提取synergy_score失败，重新生成!")
            synergy_response = self.generate_response(synergy_prompt)
            synergy_evaluation = extract(synergy_response,"score")
            synergy_score,synergy_reason = self.load_response(synergy_evaluation)


        return {
            "step": {
                "score": step_score,
                "reason": step_reason
            },
            "paramenter": {
                "score": parameter_score,
                "reason": parameter_reason
            },
            "complete": {
                "score": complete_score,
                "reason": complete_reason
            },
            "vulnerability": {
                "score": vulnerability_score,
                "reason": vulnerability_reason
            },
            "synergy": {
                "score": synergy_score,
                "reason": synergy_reason
            }
        }

class ExpEvaluation(BaseAgent):    

    def load_response(self, json_str: str) -> Dict:
        """加载JSON字符串为字典"""
        try:
            parameter_evaluation = json.loads(json_str)
            score = parameter_evaluation.get("score")
            reason = parameter_evaluation.get("reason")
            if score is not None and isinstance(score, str):
                score = int(score)
            if reason is not None and isinstance(reason, str):
                reason = reason.strip()
            return score,reason

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None, None

    def process(self, input_data: Dict) -> Dict:
        output = input_data.get("output")
        label = input_data.get("label")
        
        step_prompt, paramenter_prompt, complete_prompt, vulnerability_prompt, synergy_prompt = _build_legacy_exp_eval_prompts(output, label)

        # 步骤评估
        step_response = self.generate_response(step_prompt)
        step_evaluation = extract(step_response,"score")
        step_score,step_reason = self.load_response(step_evaluation)
        while not step_score or not step_reason:
            logger.info(f"提取step_score失败，重新生成!")
            step_response = self.generate_response(step_prompt)
            step_evaluation = extract(step_response,"score")
            step_score,step_reason = self.load_response(step_evaluation)
        
        # 参数评估
        parameter_response = self.generate_response(paramenter_prompt)
        parameter_evaluation = extract(parameter_response,"score")
        parameter_score,parameter_reason = self.load_response(parameter_evaluation)
        while not parameter_score or not parameter_reason:
            logger.info(f"提取parameter_score失败，重新生成!")
            parameter_response = self.generate_response(paramenter_prompt)
            parameter_evaluation = extract(parameter_response,"score")
            parameter_score,parameter_reason = self.load_response(parameter_evaluation)

        # 完整性评估
        complete_response = self.generate_response(complete_prompt)
        complete_evaluation = extract(complete_response,"score")
        complete_score,complete_reason = self.load_response(complete_evaluation)
        while not complete_score or not complete_reason:
            logger.info(f"提取complete_score失败，重新生成!")
            complete_response = self.generate_response(complete_prompt)
            complete_evaluation = extract(complete_response,"score")
            complete_score,complete_reason = self.load_response(complete_evaluation)
        
        # 脆弱性评估
        vulnerability_response = self.generate_response(vulnerability_prompt)
        vulnerability_evaluation = extract(vulnerability_response,"score")
        vulnerability_score,vulnerability_reason = self.load_response(vulnerability_evaluation)
        while not vulnerability_score or not vulnerability_reason:
            logger.info(f"提取vulnerability_score失败，重新生成!")
            vulnerability_response = self.generate_response(vulnerability_prompt)
            vulnerability_evaluation = extract(vulnerability_response,"score")
            vulnerability_score,vulnerability_reason = self.load_response(vulnerability_evaluation)

        # 协同性评估
        synergy_response = self.generate_response(synergy_prompt)
        synergy_evaluation = extract(synergy_response,"score")
        synergy_score,synergy_reason = self.load_response(synergy_evaluation)
        while not synergy_score or not synergy_reason:
            logger.info(f"提取synergy_score失败，重新生成!")
            synergy_response = self.generate_response(synergy_prompt)
            synergy_evaluation = extract(synergy_response,"score")
            synergy_score,synergy_reason = self.load_response(synergy_evaluation)


        return {
            "step": {
                "score": step_score,
                "reason": step_reason
            },
            "paramenter": {
                "score": parameter_score,
                "reason": parameter_reason
            },
            "complete": {
                "score": complete_score,
                "reason": complete_reason
            },
            "vulnerability": {
                "score": vulnerability_score,
                "reason": vulnerability_reason
            },
            "synergy": {
                "score": synergy_score,
                "reason": synergy_reason
            }
        }

class EvalClarityAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def load_response(self, json_str: str) -> Dict:
        """加载JSON字符串为字典"""
        try:
            parameter_evaluation = json.loads(json_str)
            score = parameter_evaluation.get("score")
            reason = parameter_evaluation.get("reason")
            if score is not None and isinstance(score, str):
                score = int(score)
            if reason is not None and isinstance(reason, str):
                reason = reason.strip()
            return score,reason

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None, None

    def process(self, input_data: Dict) -> Dict:
        topic = input_data.get("topic")
        idea = input_data.get("idea")
        prompt = f"""**Task Description**:
I will provide a research topic and a research idea generated from that topic. Evaluate the clarity, executability, and relevance of the research idea to the topic. Output only a score from 1 to 5.
 
**Your Task**: The research topic is inside <topic></topic>, and the research idea to evaluate is inside <idea></idea>.
 
<topic>
{topic}
</topic>

<idea>
{idea}
</idea>

**Scoring Principles**:
Give high scores to ideas that are clearly expressed and highly feasible. Give low scores to scattered ideas or ideas with weak relevance to the research topic.

**Scoring Criteria**:
- 5:
For technical details, the idea precisely and completely provides all key material parameters, experimental equipment, and regulation strategies.
For regulation strategy, the physical-property regulation strategy plays a direct and key role in the topic and objective.
For topic relevance, the idea is fully relevant and satisfies the topic requirements in material system, sample type, and sample-processing method.
For language, the causal logic is completely clear, academic terminology is fully standard, and there is no colloquial wording.

- 4:
For technical details, the idea precisely provides most key material parameters, experimental equipment, and regulation strategies.
For regulation strategy, the physical-property regulation strategy plays a key role in the topic and objective.
For topic relevance, the idea is fully relevant and basically satisfies the topic requirements in material system, sample type, and sample-processing method.
For language, the causal logic is clear, academic terminology is standard, and there is no colloquial wording.

- 3:
For technical details, the idea provides most material parameters, experimental equipment, and regulation strategies.
For regulation strategy, the physical-property regulation strategy plays an indirect role in the topic and objective.
For topic relevance, the idea is relevant and partially satisfies the topic requirements in material system, sample type, and sample-processing method.
For language, the causal logic is relatively clear, academic terminology is relatively standard, and there is no colloquial wording.

- 2:
For technical details, the idea provides only part of the material parameters, experimental equipment, and regulation strategies.
For regulation strategy, the physical-property regulation strategy is related to the topic and objective, but its effect cannot be judged.
For topic relevance, the idea is weakly related and does not sufficiently satisfy the topic requirements in material system, sample type, and sample-processing method.
For language, the causal logic is not very clear, terminology is not very standard, and colloquial wording appears.

- 1:
For technical details, the idea provides only a small portion of material parameters, experimental equipment, and regulation strategies.
For regulation strategy, the physical-property regulation strategy is weakly related to the topic and objective.
For topic relevance, the idea is unrelated and cannot satisfy the topic requirements in material system, sample type, or sample-processing method.
For language, the causal logic is confused, terminology is nonstandard, and colloquial wording appears.

**Output Instructions**:
1. Output a score and evaluation reason.
2. Wrap the output with <score></score>.
3. The "score" field should contain only a numeric value from 1 to 5, for example: "score": 3.
4. In "reason", briefly summarize the research idea.
 
**Output Format**:
<score>
{{
    "score": 0,
    "reason": ""
}}
</score>
 
Start now: analyze the idea and output a JSON object following the template above, wrapped with <score> and </score>.
 
"""
        try:
            response = self.generate_response(prompt) 
            clear_responese = extract(response,"score")
            clear_score, clear_reason = self.load_response(clear_responese)
            while not clear_score and not clear_reason:
                logger.info(f"提取score失败，重新生成!")
                response = self.generate_response(prompt)
                clear_responese = extract(response,"score")
                clear_score, clear_reason = self.load_response(clear_responese)
            return {
                "score": clear_score,
                "reason": clear_reason
            }
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return None

class EvalExpHardAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def _strip_code_fences(self, text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def _normalize_judge(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return None

    def _is_valid_response(self, response):
        if not isinstance(response, dict):
            return False
        judge = self._normalize_judge(response.get("judge"))
        return judge is not None and isinstance(response.get("reason"), str)

    def load_response(self, json_str: str) -> Dict:
        cleaned = self._strip_code_fences(json_str)
        candidates = [cleaned]
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                judge_bool = self._normalize_judge(parsed.get("judge"))
                if judge_bool is None:
                    continue
                reason = str(parsed.get("reason", "")).strip()
                return {
                    "judge": "True" if judge_bool else "False",
                    "score": 1.0 if judge_bool else 0.0,
                    "reason": reason,
                }
            except json.JSONDecodeError:
                continue

        logger.error("实验 hard 评估结果 JSON 解析失败")
        return None


    def process(self, input_data: Dict) -> Dict:
        output = str(
            input_data.get("pred_exp")
            or input_data.get("output")
            or input_data.get("model_output_experiment")
            or input_data.get("exp")
            or ""
        ).strip()
        label = str(
            input_data.get("label_exp")
            or input_data.get("label")
            or input_data.get("groundtruth_experiment")
            or ""
        ).strip()

        prompt = f"""Your task is to judge whether the given model-output experiment and the groundtruth experiment are essentially the same, using the groundtruth as the standard. First extract key parameters and fabrication methods, then compare them, and finally give a conclusion. This judgment focuses on "consistency of experimental objective/function and testing conclusion" and "functional equivalence of key parameters under relaxed tolerance".
Model-output experiment:
<model_output_experiment>
{output}
</model_output_experiment>
Groundtruth experiment:
<groundtruth_experiment>
{label}
</groundtruth_experiment>

1. Normalize and clean: standardize units and notation (temperature in degrees C or K; time in min; pressure in Pa or mbar; gas flow in sccm; concentration in mol% or at.%; laser parameters including wavelength in nm / pulse width in fs, ps, or ns / fluence in J cm^-2 / scan speed in mm s^-1).
2. Read section by section: carefully read <model_output_experiment> and <groundtruth_experiment>.
3. Extract key information from both sides: materials and samples, composition/ratio, precursor/target, substrate, fabrication method, key equipment, workflow overview, key process parameters (laser energy/fluence, wavelength, repetition rate, temperature, time, atmosphere, pressure, heating/cooling rate, deposition rate, etc.), testing methods, key testing conditions and criteria (such as Jc criterion, temperature range, magnetic-field conditions).
4. Compare point by point: compare "experimental steps/sequence, operations, conditions, parameters, and fabrication method/process route". Unless they clearly affect function or mechanism, the following differences can be treated as non-decisive and need not immediately lead to "False": wording differences, common material aliases, suppliers/brands, small sample-size differences, reasonable deposition-thickness differences, and auxiliary steps such as cleaning or labeling.
5. Overall logic check: evaluate whether causal and dependency logic are consistent, such as deposition before annealing, and judge whether any key step is missing or reversed.
6. Form an initial judgment according to the score below.
7. Recheck whether any key parameter or step was missed; if an omission affects the conclusion, update the initial judgment.

Key dimensions and weights (total = 100):
Experimental objective and target criteria, including Jc/Fp criteria and target temperature-field window: 30
Characterization and testing comparability, including comparable Jc/Fp testing and temperature/field ranges: 25
Functional roles of materials and samples, including whether dopants/secondary phases play equivalent roles: 15
Fabrication method/process route, including whether the same microstructure or equivalent defect configuration is realized: 10
Key process parameters, such as temperature, time, pressure, atmosphere, rate, and core laser parameters: 10
Workflow steps and sequence logic, including key omissions or reversed steps: 10

Match-level conversion:
Full match = 1.00
Near match = 0.8 (detail differences can be treated as equivalent through routine adjustment or mechanistic explanation)
Partial match = 0.6 (clear differences exist but do not necessarily break the function; equivalence may be possible through small changes or reasonable explanation)
Mismatch/missing = 0.40 (differences are clear and may affect function; keep a baseline score to avoid extreme penalties)
For each dimension, multiply the highest relevant match level by the weight, then sum the weighted scores.

Numerical tolerance (use the wider of relative and absolute tolerance):
Temperature: +/-25% (not a strict cutoff; if beyond tolerance but supported by sufficient mechanistic explanation, still consider partial match)
Time: +/-25%
Pressure / flow / concentration: +/-25%
If a parameter such as temperature or oxygen pressure falls outside the relaxed tolerance, but the experimental objective, microstructure, or testing results show that equivalent pinning/function can be achieved, count that item as a near match.

Functional-equivalence rules (higher priority than pure parameter comparison):
If the combined score for **experimental objective (weight 30)** and **characterization/testing comparability (weight 25)** is >= 50 out of 55, and the material or fabrication dimension is at least a partial match, judge "True" (substantially equivalent), unless there is direct evidence of a mechanistic conflict, such as one material being unable to form the other material's core defect type.
When dopant names differ, such as Y211 vs BZO, evaluate their functional roles, such as bulk nanoparticle pinning vs columnar secondary-phase/interface pinning. If the functional roles are the same and characterization shows equivalent defects/microstructure, treat as a near match.
When the process path differs, such as single-layer vs multilayer, but characterization results (TEM/STEM, XRD, Jc/Fp curves) show comparable performance under the target temperature/field window, treat it as equivalent.

Non-critical differences that should not affect the judgment:
wording, synonyms, common material aliases such as YBCO vs YBa2Cu3O7-delta, suppliers/equipment brands, irrelevant numbering, buffer-layer or layering controls, material dimensions, sample dimensions, processing dimensions, testing operation details, laser power, pulse frequency, microbridge fabrication, substrate material, and parts beyond the groundtruth.

Acceptable workflow differences:
Equivalent sequence adjustment or insertion of non-critical steps, such as cleaning or labeling, is acceptable if it realizes the same route and does not conflict with the core logic.

Final threshold:
Calculate the consistency score (0-100). Score >= 80 -> "True"; score < 80 -> "False". If the score is exactly 75-79, judge based on whether the differences affect the core mechanism/function: if yes, "False"; if no, "True".
If the first functional-equivalence rule above is satisfied (objective + testing comparability >= 50 and material/fabrication at least partially matches), directly judge "True" and explain why.

Finally, provide your final judgment in "judge" using "True" or "False". Then explain your reasoning in "reason", stating which key elements are the same or different and prioritizing evidence from fabrication method, key steps, key parameters, and sequence logic.

### Notes
1. Output JSON format.
2. Wrap the entire JSON object with <res> and </res>.

### Final Output Format
<res>
{{
    "judge": "True or False",
    "reason": "Provide dimension-by-dimension scores and evidence. If the judgment is True, prioritize characterization/testing evidence that supports functional equivalence. If the judgment is False, explain the minimum feasible changes in parameters, structure, or characterization needed to achieve equivalence and provide rescoring guidance."
}}
</res>
"""
        try:
            last_result = None
            for _ in range(5):
                response = self.generate_response(prompt)
                result_body = extract(response, "res", hard=False) or response
                last_result = self.load_response(result_body)
                if self._is_valid_response(last_result):
                    return last_result
                logger.info(f"提取template失败，重新生成!")
            return last_result

        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return None

class HumanTemplateAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""
    def load_response(self, json_str: str) -> Dict:
        """加载JSON字符串为字典"""
        try:
            template = json.loads(json_str)

            return template

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return None, None


    def process(self, input_data: Dict) -> Dict:
        topic = input_data.get("topic")
        idea = input_data.get("idea")
        exp = input_data.get("exp")

        prompt = f"""You are an expert in superconducting materials. Your task is to evaluate the input research topic (TOPIC), generated research idea (IDEA), and detailed experimental plan (EXPERIMENT_PLAN) according to the given evaluation dimensions, provide evidence, and output the result in strict JSON format.
Research topic:
<topic>
{topic}
</topic>
Research idea:
<idea>
{idea}
</idea>
Detailed experimental plan:
<experiment_plan>
{exp}
</experiment_plan>
---

【Evaluation Dimensions】
1. Content Completeness
   Whether the idea and experiment are complete.

2. Theme Relevance
   Whether the idea is relevant to the topic, whether the experiment is relevant to the topic, and whether the experiment is relevant to the idea.

3. Technical Innovation
   Whether the idea and experiment use innovative experimental techniques.

4. Technical Feasibility
   Whether the experimental techniques used by the idea and experiment are feasible.

5. Application Value
   Whether the plans designed in the idea and experiment have potential application value.
   
---

【Output Content】
1. Output principle: for each dimension, output a score ("score") and direct evidence ("evidence") found from the input content. The "score" ranges from 0 to 5. The "evidence" content must come only from the input text; it may be summarized but must not be fabricated. Output only the required information and do not output irrelevant information or comments.

2. Output six items: evaluations for the five dimensions and one overall comment.

2.1 For Content Completeness, evidence should summarize the completeness of the given idea and experiment.

2.2 For Theme Relevance, evidence should cover topic-to-idea relevance, idea-to-experiment relevance, and topic-to-experiment relevance. Do not fabricate content beyond the given text.

2.3 For Technical Innovation, evidence should include all innovation dimensions you identify that are directly supported by the given idea and experiment. These are not limited to material system, regulation methods, or experimental techniques. Do not fabricate content beyond the given text.

2.4 For Technical Feasibility, evidence should include all directly supported information in the given idea and experiment that strongly supports the experimental objective. These are not limited to material system or experimental techniques. Do not fabricate content beyond the given text.

2.5 For Application Value, evidence should include all content in the given idea and experiment related to potential application value. Do not fabricate content beyond the given text.

2.6 comment does not need to be filled and should default to an empty string.

3. Wrap the output with <template></template>.

【Output JSON Template】
Strictly follow this structure:
<template>
{{
    "Content Completeness": {{
        "score": 0,
        "evidence": [
            "Parts included in the idea: ...",
            "Parts included in the experiment: ..."
        ]
    }},
    "Theme Relevance": {{
        "score": 0,
        "evidence": [
            "Topic-to-idea relevance: ...",
            "Idea-to-experiment relevance: ...",
            "Topic-to-experiment relevance: ..."
        ]
    }},
    "Technical Innovation": {{
        "score": 0,
        "evidence": [
            "Innovation in the material system: ...",
            "Innovation in regulation methods: ...",
            "Innovation in experimental techniques: ..."
        ]
    }},
    "Technical Feasibility": {{
        "score": 0,
        "evidence": [
            "Material-system feasibility: ...",
            "Experimental-technique feasibility: ..."
        ]
    }},
    "Application Value": {{
        "score": 0,
        "evidence": [
            "..."
        ]
    }},
    "comment": ""
}}
</template>
"""
        try:
            response = self.generate_response(prompt)
            print(response)
            template = extract(response, "template")
            template = self.load_response(template)
            print("template",template)

            while not template:
                logger.info(f"提取template失败，重新生成!")
                response = self.generate_response(prompt)
                template = extract(response,"template")
                template = self.load_response(template)

            return template

        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            return None
