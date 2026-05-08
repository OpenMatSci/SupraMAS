"""
可回溯思维链多智能体框架 (Traceback Pipeline) - 重构版
从机理分析 -> 实验方案 -> Idea -> 策略树节点
每一层回溯都整合模板库(template/knowledge)知识引用
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from loguru import logger
from abc import ABC, abstractmethod
import sys
import os
import yaml

from supramas.agents import get_llms
from supramas.conversation_logger import ConversationLogger
from supramas.utils import extract


def cprint(text: str, color: Optional[str] = None, attrs: Optional[List[str]] = None):
    """Simple colored print fallback"""
    try:
        from termcolor import cprint as tc_cprint  # type: ignore
        tc_cprint(text, color, attrs=attrs)
    except ImportError:
        print(text)


# ==============================================================================
# 数据模型
# ==============================================================================

@dataclass
class TemplateRef:
    """模板库引用"""
    template_id: str
    source_field: str  # 来自 idea 的哪个字段 (material, motivation, strategy_Composition等)
    description: str
    role_in_traceback: str = ""  # 在回溯中的作用


@dataclass
class StrategyTreeRef:
    """策略树引用"""
    node_id: str
    source_field: str
    description: str
    role_in_traceback: str = ""


@dataclass
class TracebackLayer:
    """回溯层 — 每一层回溯的结构"""
    layer_name: str  # mechanism / experiment / idea / strategy_tree
    content: Dict[str, Any]
    template_refs: List[Dict[str, Any]] = field(default_factory=list)
    strategy_tree_refs: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class TracebackChain:
    """完整的回溯链"""
    mechanism_finding: Dict[str, Any]
    
    # 回溯到实验方案
    experiment_layer: TracebackLayer = field(default_factory=lambda: TracebackLayer(layer_name="experiment", content={}))
    
    # 回溯到 Idea
    idea_layer: TracebackLayer = field(default_factory=lambda: TracebackLayer(layer_name="idea", content={}))
    
    # 回溯到策略树节点
    strategy_layer: TracebackLayer = field(default_factory=lambda: TracebackLayer(layer_name="strategy_tree", content={}))
    
    # 完整推理链
    reasoning_chain: List[str] = field(default_factory=list)
    
    # 可信度评估
    credibility: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# BaseAgent
# ==============================================================================

class BaseAgent(ABC):
    def __init__(self, llm=None, model: str = None, temperature: float = 0.7, base_url: str = None, api_key: str = None):
        if llm is not None:
            self.llm = llm
            self.model = getattr(llm, 'model', model)
            self.temperature = temperature
        else:
            self.model = model
            self.temperature = temperature
            if base_url and api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL")
                )

    def generate_response(self, prompt: str, history: str = None, file_path: str = None,
                          extract_key: str = None, load_json=False, **kwargs) -> str:
        messages = []
        if file_path:
            import fitz
            doc = fitz.open(file_path)
            file_content = ""
            for page in doc:
                file_content += page.get_text("text")
            doc.close()
            messages = [
                {"role": "system", "content": "你是凝聚态物理领域的专家，熟悉 REBCO 磁通钉扎的相关研究。"},
                {"role": "system", "content": file_content},
            ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        if hasattr(self, 'llm'):
            if extract_key:
                return self.llm.response(messages, extract_key=extract_key,
                                         temperature=self.temperature, max_tokens=8192,
                                         load_json=load_json, **kwargs)
            else:
                return self.llm.response(messages, temperature=self.temperature,
                                         max_tokens=8192, **kwargs)
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=8192,
            )
            if extract_key:
                return extract(response.choices[0].message.content, extract_key)
            else:
                return response.choices[0].message.content


# ==============================================================================
# 智能体 1: 机理分析解析
# ==============================================================================

class MechanismAnalyzer(BaseAgent):
    """机理分析解析智能体"""
    
    def __init__(self, llm):
        super().__init__(llm)
        
    def extract_key_findings(self, mechanism_text: str) -> List[Dict[str, Any]]:
        """从机理分析中提取关键发现"""
        
        prompt = f"""
你是一个材料科学机理分析专家。请从以下机理分析报告中提取关键发现。

# 机理分析报告:
{mechanism_text}

# 任务要求
请提取以下信息：
1. 核心发现（如性能差异、关键现象）
2. 关键参数（如温度、气压、成分等）
3. 物理机制解释
4. 参数之间的关联关系

# 输出格式
结果用<response>...</response>标签包裹。

以JSON格式返回，格式如下：
<response>
{{
    "findings": [
        {{
            "name": "发现名称",
            "description": "详细描述",
            "key_parameters": ["参数1", "参数2"],
            "mechanism": "物理机制解释",
            "importance": "high/medium/low"
        }}
    ]
}}
</response>
"""
        response = self.generate_response(prompt, load_json=True, extract_key="response")
        findings = response.get("findings", [])
        return findings


# ==============================================================================
# 智能体 2: 实验方案匹配 + 知识溯源
# ==============================================================================

class ExperimentMatcher(BaseAgent):
    """实验方案匹配智能体 — 同时回溯模板库知识在实验方案中的体现"""
    
    def __init__(self, llm):
        super().__init__(llm)
        
    def match_to_experiment(
        self,
        finding: Dict[str, Any],
        experiment_plan: str,
        step_knowledge_map: Dict[int, Dict] = None
    ) -> TracebackLayer:
        """将机理发现匹配到实验步骤，并溯源模板库知识"""
        
        # 构造知识上下文
        knowledge_context = ""
        if step_knowledge_map:
            knowledge_context = "\n# 各实验步骤引用的模板库知识 (knowledge):\n"
            for step_id, step_info in step_knowledge_map.items():
                knowledge_context += f"\n## 实验步骤 {step_id}:\n"
                for kid, desc in step_info.get("knowledge", {}).items():
                    knowledge_context += f"  - 知识 [{kid}]: {desc}\n"
        
        prompt = f"""
你是一个实验设计专家。请分析以下机理发现与实验方案的对应关系，并追溯模板库知识的使用。

# 机理发现:
{json.dumps(finding, ensure_ascii=False, indent=2)}

# 实验方案:
{experiment_plan}
{knowledge_context}

# 任务要求
1. 找到与该机理发现直接相关的实验步骤
2. 分析每个步骤中引用的模板库知识(knowledge)对实验设计的指导作用
3. 建立 "机理发现 ← 实验操作 ← 模板库知识" 的回溯链

# 输出格式
结果用<response>...</response>标签包裹。

以JSON格式返回：
<response>
{{
    "matched_steps": [
        {{
            "step_id": "步骤编号",
            "step_name": "步骤名称",
            "key_operations": ["操作1", "操作2"],
            "parameters": {{"参数名": "参数值"}},
            "connection_to_finding": "与机理发现的联系说明",
            "confidence": 0.0-1.0
        }}
    ],
    "knowledge_traces": [
        {{
            "knowledge_id": "知识ID",
            "from_step": "来源步骤ID",
            "core_method": "知识的核心方法",
            "guided_aspects": ["受该知识指导的实验方案部分"],
            "reasoning_chain": "从知识到方案到发现的推理链",
            "contribution_to_finding": "该知识对机理发现的贡献",
            "confidence": 0.0-1.0
        }}
    ],
    "overall_reasoning": "整体回溯推理"
}}
</response>
"""
        
        response = self.generate_response(prompt, load_json=True, extract_key="response")
        
        layer = TracebackLayer(
            layer_name="experiment",
            content={
                "matched_steps": response.get("matched_steps", []),
            },
            template_refs=response.get("knowledge_traces", []),
            reasoning=response.get("overall_reasoning", ""),
            confidence=self._avg_confidence(response.get("matched_steps", []))
        )
        return layer
    
    def _avg_confidence(self, items: List[Dict]) -> float:
        confs = [item.get("confidence", 0) for item in items if isinstance(item.get("confidence"), (int, float))]
        return sum(confs) / len(confs) if confs else 0.0


# ==============================================================================
# 智能体 3: Idea追溯（利用 _source 字段精确回溯）
# ==============================================================================

class SourceAwareIdeaTracer(BaseAgent):
    """
    Idea追溯智能体 — 利用 all_protocol.json 中 idea 的 _source 字段
    (strategy_tree + template) 进行精确的知识溯源
    """
    
    def __init__(self, llm):
        super().__init__(llm)
    
    def trace_to_idea(
        self,
        experiment_steps: List[Dict[str, Any]],
        idea_raw: Dict[str, Any],
        idea_text: str,
        finding: Dict[str, Any]
    ) -> TracebackLayer:
        """将实验步骤追溯到 Idea，提取嵌入的 strategy_tree 和 template 引用"""
        
        # 1. 先从 idea_raw 中提取所有显式的 _source 引用
        explicit_template_refs = self._extract_all_template_refs(idea_raw)
        explicit_strategy_refs = self._extract_all_strategy_refs(idea_raw)
        
        # 2. 使用 LLM 分析实验步骤如何实现 Idea，并验证知识来源
        prompt = f"""
你是一个科研创新分析专家。请分析实验步骤背后的创新思想来源。

# 机理发现:
{json.dumps(finding, ensure_ascii=False, indent=2)}

# 实验步骤:
{json.dumps(experiment_steps, ensure_ascii=False, indent=2)}

# Idea完整信息:
{idea_text}

# Idea中已声明的模板库知识引用:
{json.dumps(explicit_template_refs, ensure_ascii=False, indent=2)}

# Idea中已声明的策略树引用:
{json.dumps(explicit_strategy_refs, ensure_ascii=False, indent=2)}

# 任务要求
1. 找出实验步骤源自Idea中的哪些创新点（material、motivation、strategies、technique、outcome）
2. 对Idea中声明的每条模板库引用(template)，分析其如何指导了该创新点的形成
3. 对Idea中声明的每条策略树引用(strategy_tree)，分析其如何演化为创新点
4. 这些Idea创新点如何通过实验步骤演化为机理发现
5. 生成完整的回溯推理链

# 输出格式
结果用<response>...</response>标签包裹。

以JSON格式返回：
<response>
{{
    "idea_components": [
        {{
            "idea_field": "来自 idea 的字段（material/motivation/strategy_XX/technique/outcome）",
            "idea_content": "具体内容",
            "innovation_point": "创新点",
            "implementation_in_experiment": "实验中如何实现",
            "connection_to_finding": "如何关联到机理发现",
            "confidence": 0.0-1.0
        }}
    ],
    "template_knowledge_analysis": [
        {{
            "template_id": "模板ID",
            "source_field": "来自 idea 的哪个字段",
            "original_description": "引用的原始描述",
            "how_it_guided_idea": "如何指导了 idea 的形成",
            "contribution_to_finding": "对机理发现的贡献",
            "confidence": 0.0-1.0
        }}
    ],
    "strategy_tree_analysis": [
        {{
            "node_id": "策略树节点ID",
            "source_field": "来自 idea 的哪个字段",
            "original_description": "引用的原始描述",
            "how_it_evolved_to_idea": "如何演化为 idea 中的创新点",
            "confidence": 0.0-1.0
        }}
    ],
    "reasoning": "整体追溯推理链"
}}
</response>
"""
        response = self.generate_response(prompt, load_json=True, extract_key="response")
        
        layer = TracebackLayer(
            layer_name="idea",
            content={
                "idea_components": response.get("idea_components", []),
            },
            template_refs=response.get("template_knowledge_analysis", []),
            strategy_tree_refs=response.get("strategy_tree_analysis", []),
            reasoning=response.get("reasoning", ""),
            confidence=self._avg_confidence(response.get("idea_components", []))
        )
        return layer
    
    def _extract_all_template_refs(self, idea: Dict) -> List[Dict[str, str]]:
        """从 idea 的所有 _source 字段提取 template 引用"""
        refs = []
        seen = set()
        
        # 顶级字段
        for key in ["material_source", "motivation_source", "technique_source", "outcome_source"]:
            source = idea.get(key, {})
            if isinstance(source, dict) and "template" in source:
                field_name = key.replace("_source", "")
                for kid, desc in source["template"].items():
                    ref_key = f"{field_name}_{kid}"
                    if ref_key not in seen:
                        seen.add(ref_key)
                        refs.append({
                            "template_id": kid,
                            "source_field": field_name,
                            "description": desc
                        })
        
        # strategies 中的引用
        for strat in idea.get("strategies", []):
            if isinstance(strat, dict):
                ss = strat.get("strategy_source", {})
                if isinstance(ss, dict) and "template" in ss:
                    dim = strat.get("tuning_dimensions", "unknown")
                    for kid, desc in ss["template"].items():
                        ref_key = f"strategy_{dim}_{kid}"
                        if ref_key not in seen:
                            seen.add(ref_key)
                            refs.append({
                                "template_id": kid,
                                "source_field": f"strategy ({dim})",
                                "description": desc
                            })
        return refs
    
    def _extract_all_strategy_refs(self, idea: Dict) -> List[Dict[str, str]]:
        """从 idea 的所有 _source 字段提取策略树引用"""
        refs = []
        seen = set()
        
        # 顶级字段
        for key in ["material_source", "motivation_source", "technique_source", "outcome_source"]:
            source = idea.get(key, {})
            if isinstance(source, dict) and "strategy_tree" in source:
                field_name = key.replace("_source", "")
                for node_id, desc in source["strategy_tree"].items():
                    ref_key = f"{field_name}_{node_id}"
                    if ref_key not in seen:
                        seen.add(ref_key)
                        refs.append({
                            "node_id": node_id,
                            "source_field": field_name,
                            "description": desc
                        })
        
        # strategies 中的引用
        for strat in idea.get("strategies", []):
            if isinstance(strat, dict):
                ss = strat.get("strategy_source", {})
                if isinstance(ss, dict) and "strategy_tree" in ss:
                    dim = strat.get("tuning_dimensions", "unknown")
                    for node_id, desc in ss["strategy_tree"].items():
                        ref_key = f"strategy_{dim}_{node_id}"
                        if ref_key not in seen:
                            seen.add(ref_key)
                            refs.append({
                                "node_id": node_id,
                                "source_field": f"strategy ({dim})",
                                "description": desc
                            })
        return refs
    
    def _avg_confidence(self, items: List[Dict]) -> float:
        confs = [item.get("confidence", 0) for item in items if isinstance(item.get("confidence"), (int, float))]
        return sum(confs) / len(confs) if confs else 0.0


# ==============================================================================
# 智能体 4: 策略树映射
# ==============================================================================

class StrategyMapper(BaseAgent):
    """策略树映射智能体 — 将 Idea 精确映射到策略树节点"""
    
    def __init__(self, llm):
        super().__init__(llm)
        
    def map_to_strategy_tree(
        self,
        idea_components: List[Dict[str, Any]],
        strategy_tree_refs: List[Dict[str, Any]],
        strategy_tree: Dict[str, Any],
        finding: Dict[str, Any]
    ) -> TracebackLayer:
        """将 Idea 映射到策略树节点，利用显式引用验证"""
        
        flat_strategies = self._flatten_strategy_tree(strategy_tree)
        
        prompt = f"""
你是一个知识图谱专家。请将Idea组件精确映射到策略树节点。

# 机理发现:
{json.dumps(finding, ensure_ascii=False, indent=2)}

# Idea组件:
{json.dumps(idea_components, ensure_ascii=False, indent=2)}

# Idea中已声明的策略树引用:
{json.dumps(strategy_tree_refs, ensure_ascii=False, indent=2)}

# 策略树（扁平化）:
{json.dumps(flat_strategies, ensure_ascii=False, indent=2)}

# 任务要求
1. 验证Idea中声明的策略树引用是否与策略树节点内容一致
2. 找出每个Idea组件源自策略树的哪些节点
3. 分析策略树节点的 tuning_strategies 和 tuning_effects 如何演化为 Idea
4. 分析策略树节点的 draw_backs 是否是 Idea 创新的出发点
5. 评估映射的可信度

# 输出格式
结果用<response>...</response>标签包裹。

以JSON格式返回：
<response>
{{
    "strategy_mappings": [
        {{
            "strategy_level": "节点level",
            "tuning_dimension": "调控维度",
            "tuning_strategy": "具体策略",
            "tuning_effect": "调控效果",
            "drawback_addressed": "该策略的不足如何被Idea克服（如有）",
            "idea_evolution": "如何演化为Idea",
            "connection_to_finding": "与机理发现的联系",
            "match_score": 0.0-1.0,
            "verified_by_explicit_ref": true/false
        }}
    ],
    "reasoning": "映射推理链",
    "coverage_assessment": "策略树覆盖度评估"
}}
</response>
"""
        
        response = self.generate_response(prompt, load_json=True, extract_key="response")
        
        layer = TracebackLayer(
            layer_name="strategy_tree",
            content={
                "strategy_mappings": response.get("strategy_mappings", []),
                "coverage_assessment": response.get("coverage_assessment", "")
            },
            reasoning=response.get("reasoning", ""),
            confidence=self._avg_match_score(response.get("strategy_mappings", []))
        )
        return layer
    
    def _flatten_strategy_tree(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """递归扁平化策略树"""
        flat = []
        if "level" in node:
            node_info = {
                "level": node.get("level", ""),
                "tuning_strategies": node.get("tuning_strategies", []),
                "draw_backs": node.get("draw_backs", {})
            }
            flat.append(node_info)
        if "children" in node and node["children"]:
            for child in node["children"]:
                flat.extend(self._flatten_strategy_tree(child))
        return flat
    
    def _avg_match_score(self, items: List[Dict]) -> float:
        scores = [item.get("match_score", 0) for item in items if isinstance(item.get("match_score"), (int, float))]
        return sum(scores) / len(scores) if scores else 0.0


# ==============================================================================
# 智能体 5: 可信度评估 + 思维链整合
# ==============================================================================

class CredibilityEvaluator(BaseAgent):
    """可信度评估与思维链整合智能体"""
    
    def __init__(self, llm):
        super().__init__(llm)
    
    def evaluate_and_integrate(
        self,
        finding: Dict[str, Any],
        experiment_layer: TracebackLayer,
        idea_layer: TracebackLayer,
        strategy_layer: TracebackLayer
    ) -> Tuple[List[str], Dict[str, Any]]:
        """评估整条回溯链的可信度并整合为连贯的推理链"""
        
        prompt = f"""
你是一个科研逻辑分析专家。请评估以下回溯链的可信度并整合为连贯的推理链。

# 机理发现:
{json.dumps(finding, ensure_ascii=False, indent=2)}

# 回溯层1: 实验方案
## 匹配的实验步骤:
{json.dumps(experiment_layer.content, ensure_ascii=False, indent=2)}
## 模板库知识溯源:
{json.dumps(experiment_layer.template_refs[:10], ensure_ascii=False, indent=2)}
## 推理: {experiment_layer.reasoning}

# 回溯层2: Idea
## Idea组件:
{json.dumps(idea_layer.content, ensure_ascii=False, indent=2)}
## 模板库引用:
{json.dumps(idea_layer.template_refs[:10], ensure_ascii=False, indent=2)}
## 策略树引用:
{json.dumps(idea_layer.strategy_tree_refs[:10], ensure_ascii=False, indent=2)}
## 推理: {idea_layer.reasoning}

# 回溯层3: 策略树
## 策略映射:
{json.dumps(strategy_layer.content, ensure_ascii=False, indent=2)}
## 推理: {strategy_layer.reasoning}

# 任务要求
1. 生成清晰的回溯思维链，从机理发现逐层追溯到策略树
2. 评估每一层回溯的可信度（证据是否充分、推理是否连贯）
3. 标注模板库知识在每一层中的支撑作用
4. 评估整体回溯链的可信度
5. 指出可信度较弱的环节和需要补充的证据

# 输出格式
结果用<response>...</response>标签包裹。

以JSON格式返回：
<response>
{{
    "reasoning_chain": [
        "推理步骤1: ...",
        "推理步骤2: ...",
        "..."
    ],
    "credibility": {{
        "overall_score": 0.0-1.0,
        "experiment_layer_score": 0.0-1.0,
        "idea_layer_score": 0.0-1.0,
        "strategy_layer_score": 0.0-1.0,
        "template_support_score": 0.0-1.0,
        "weak_links": ["薄弱环节1", "薄弱环节2"],
        "strong_evidence": ["强证据1", "强证据2"],
        "suggestions": ["改进建议1", "改进建议2"]
    }},
    "knowledge_flow_summary": "模板库知识在整条回溯链中的流动描述"
}}
</response>
"""
        response = self.generate_response(prompt, load_json=True, extract_key="response")
        
        reasoning_chain = response.get("reasoning_chain", [])
        credibility = response.get("credibility", {})
        credibility["knowledge_flow_summary"] = response.get("knowledge_flow_summary", "")
        
        return reasoning_chain, credibility


# ==============================================================================
# 主流程 Pipeline
# ==============================================================================

class TracebackPipeline(BaseAgent):
    """可回溯思维链主流程"""
    
    def __init__(self, llm, data_dir: Path, protocol_path: Path = None):
        super().__init__(llm)
        self.data_dir = Path(data_dir)
        self.protocol_path = Path(protocol_path) if protocol_path else None
        
        # 初始化各个智能体
        self.mechanism_analyzer = MechanismAnalyzer(llm)
        self.experiment_matcher = ExperimentMatcher(llm)
        self.idea_tracer = SourceAwareIdeaTracer(llm)
        self.strategy_mapper = StrategyMapper(llm)
        self.credibility_evaluator = CredibilityEvaluator(llm)
        
    def load_data(self) -> Tuple[str, str, str, Dict, Optional[Dict]]:
        """加载所有输入数据"""
        
        protocol_data = None
        protocol_file = self.protocol_path or self.data_dir / "all_protocol.json"
        
        if protocol_file.exists():
            logger.info(f"从 all_protocol.json 加载结构化数据: {protocol_file}")
            with open(protocol_file, 'r', encoding='utf-8') as f:
                protocol_data = json.load(f)
            
            # 提取 idea 文本
            idea_raw = protocol_data.get("idea", "")
            if isinstance(idea_raw, dict):
                idea_text = self._format_idea_text(idea_raw)
            else:
                idea_text = str(idea_raw)
            
            # 提取实验方案
            steps = protocol_data.get("steps", [])
            plan_parts = [f"# 实验初稿\n{protocol_data.get('draft', '')}\n"]
            for step in steps:
                step_id = step.get("id", "?")
                desc = step.get("description", {})
                if isinstance(desc, dict):
                    desc_text = json.dumps(desc, ensure_ascii=False, indent=2)
                else:
                    desc_text = str(desc)
                plan_parts.append(f"## 实验任务 {step_id + 1}\n{desc_text}\n")
                plan_parts.append(step.get("detailed_plan", ""))
            plan_text = "\n".join(plan_parts)
        else:
            logger.info("all_protocol.json 不存在，回退到独立文件模式")
            plan_path = self.data_dir / "plan.md"
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_text = f.read()
            idea_path = self.data_dir / "idea.md"
            with open(idea_path, 'r', encoding='utf-8') as f:
                idea_text = f.read()
        
        # 加载机理分析
        mechanism_path = self.data_dir / "mechanism_analysis.md"
        with open(mechanism_path, 'r', encoding='utf-8') as f:
            mechanism_text = f.read()
        
        # 加载策略树
        tree_path = self.data_dir / "tree_formated.json"
        with open(tree_path, 'r', encoding='utf-8') as f:
            strategy_tree = json.load(f)
        
        return mechanism_text, plan_text, idea_text, strategy_tree, protocol_data
    
    def _format_idea_text(self, idea: Dict) -> str:
        """将 idea 字典格式化为可读文本（支持 strategy_tree + template 嵌套格式）"""
        parts = []
        main_fields = [
            ("material", "材料体系"),
            ("motivation", "研究动机"),
            ("technique", "实验技术"),
            ("outcome", "预期结果"),
        ]
        for key, label in main_fields:
            if key in idea:
                parts.append(f"{label} ({key}): {idea[key]}")
                source_key = f"{key}_source"
                if source_key in idea and isinstance(idea[source_key], dict):
                    source_val = idea[source_key]
                    if "strategy_tree" in source_val or "template" in source_val:
                        if "strategy_tree" in source_val:
                            for node_id, text in source_val["strategy_tree"].items():
                                parts.append(f"  策略树来源 [{node_id}]: {text}")
                        if "template" in source_val:
                            for kid, text in source_val["template"].items():
                                parts.append(f"  模板库来源 [{kid}]: {text}")
                    else:
                        for node_id, source_text in source_val.items():
                            parts.append(f"  来源 [{node_id}]: {source_text}")
        
        if "strategies" in idea and isinstance(idea["strategies"], list):
            parts.append("\n创新策略:")
            for i, strat in enumerate(idea["strategies"], 1):
                if isinstance(strat, dict):
                    dim = strat.get('tuning_dimensions', '')
                    strat_text = strat.get('tuning_strategies', '')
                    parts.append(f"  {i}. [{dim}] {strat_text}")
                    ss = strat.get('strategy_source', {})
                    if "strategy_tree" in ss:
                        for node_id, text in ss["strategy_tree"].items():
                            parts.append(f"     策略树来源 [{node_id}]: {text}")
                    if "template" in ss:
                        for kid, text in ss["template"].items():
                            parts.append(f"     模板库来源 [{kid}]: {text}")
                else:
                    parts.append(f"  {i}. {strat}")
        
        if "node_mapping" in idea and isinstance(idea["node_mapping"], dict):
            nm = idea["node_mapping"]
            parts.append(f"\n节点映射: 使用 {nm.get('total_nodes_used', 'N/A')} 个节点")
            parts.append(f"  主要来源: {nm.get('primary_source_stages', '')}")
            parts.append(f"  覆盖摘要: {nm.get('coverage_summary', '')}")
        
        return "\n".join(parts) if parts else str(idea)
    
    def run(self, output_dir: Optional[Path] = None) -> List[TracebackChain]:
        """运行完整的回溯流程"""
        
        cprint("\n" + "="*80, "cyan")
        cprint("🔄 开始可回溯思维链生成流程", "cyan", attrs=["bold"])
        cprint("  模式: 机理分析 → 实验方案 → Idea → 策略树 (每层含模板库知识溯源)", "cyan")
        cprint("="*80 + "\n", "cyan")
        
        # ========== 步骤 1: 加载数据 ==========
        cprint("[步骤 1/5] 加载数据...", "yellow")
        mechanism_text, plan_text, idea_text, strategy_tree, protocol_data = self.load_data()
        
        # 提取各步骤的知识引用
        step_knowledge_map = {}
        idea_raw = {}
        if protocol_data:
            for step in protocol_data.get("steps", []):
                knowledge = step.get("knowledge")
                if knowledge and isinstance(knowledge, dict):
                    step_knowledge_map[step.get("id", -1)] = {
                        "step_id": step.get("id"),
                        "step_description": step.get("description", {}),
                        "knowledge": knowledge,
                        "detailed_plan": step.get("detailed_plan", "")
                    }
            idea_raw = protocol_data.get("idea", {})
            if not isinstance(idea_raw, dict):
                idea_raw = {}
            cprint("✓ 数据加载完成（含 all_protocol.json 结构化数据）\n", "green")
        else:
            cprint("✓ 数据加载完成（独立文件模式）\n", "green")
        
        # ========== 步骤 2: 提取机理发现 ==========
        cprint("[步骤 2/5] 分析机理发现...", "yellow")
        findings = self.mechanism_analyzer.extract_key_findings(mechanism_text)
        cprint(f"✓ 提取到 {len(findings)} 个关键发现\n", "green")
        
        # ========== 步骤 3: 对每个发现生成回溯链 ==========
        all_chains = []
        
        for idx, finding in enumerate(findings):
            cprint("\n" + "="*80, "cyan")
            cprint(f"📍 处理发现 {idx+1}/{len(findings)}: {finding.get('name', 'Unknown')}", "cyan", attrs=["bold"])
            cprint("="*80 + "\n", "cyan")
            
            # ----- 3.1 回溯到实验方案 (含模板库知识溯源) -----
            cprint("  [层1] 回溯到实验方案 + 模板库知识溯源...", "yellow")
            experiment_layer = self.experiment_matcher.match_to_experiment(
                finding, plan_text, step_knowledge_map
            )
            n_steps = len(experiment_layer.content.get("matched_steps", []))
            n_knowledge = len(experiment_layer.template_refs)
            cprint(f"  ✓ 匹配到 {n_steps} 个实验步骤, {n_knowledge} 条知识引用\n", "green")
            
            # ----- 3.2 回溯到 Idea (含嵌入的 template + strategy_tree 引用) -----
            cprint("  [层2] 回溯到Idea + 模板库/策略树知识溯源...", "yellow")
            exp_steps_for_idea = experiment_layer.content.get("matched_steps", [])
            idea_layer = self.idea_tracer.trace_to_idea(
                exp_steps_for_idea, idea_raw, idea_text, finding
            )
            n_comps = len(idea_layer.content.get("idea_components", []))
            n_tpl = len(idea_layer.template_refs)
            n_st = len(idea_layer.strategy_tree_refs)
            cprint(f"  ✓ 追溯到 {n_comps} 个Idea组件, {n_tpl} 条模板引用, {n_st} 条策略树引用\n", "green")
            
            # ----- 3.3 回溯到策略树节点 -----
            cprint("  [层3] 回溯到策略树节点...", "yellow")
            idea_comps = idea_layer.content.get("idea_components", [])
            strategy_tree_refs = idea_layer.strategy_tree_refs
            strategy_layer = self.strategy_mapper.map_to_strategy_tree(
                idea_comps, strategy_tree_refs, strategy_tree, finding
            )
            n_mappings = len(strategy_layer.content.get("strategy_mappings", []))
            cprint(f"  ✓ 映射到 {n_mappings} 个策略节点\n", "green")
            
            # ----- 3.4 可信度评估 + 思维链整合 -----
            cprint("  [整合] 可信度评估 + 思维链整合...", "yellow")
            reasoning_chain, credibility = self.credibility_evaluator.evaluate_and_integrate(
                finding, experiment_layer, idea_layer, strategy_layer
            )
            overall_score = credibility.get("overall_score", 0)
            cprint(f"  ✓ 回溯链整合完成 | 可信度评分: {overall_score:.2f}\n", "green")
            
            # 组装完整的回溯链
            chain = TracebackChain(
                mechanism_finding=finding,
                experiment_layer=experiment_layer,
                idea_layer=idea_layer,
                strategy_layer=strategy_layer,
                reasoning_chain=reasoning_chain,
                credibility=credibility
            )
            
            # ----- 3.5 注入 ID 和层间引用关系 -----
            self._inject_ids(chain, idx)
            
            all_chains.append(chain)
        
        # ========== 步骤 4: 保存结果 ==========
        if output_dir:
            cprint("[步骤 4/5] 保存结果...", "yellow")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # JSON 格式
            chains_dict = [asdict(chain) for chain in all_chains]
            json_path = output_dir / "traceback_chains.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chains_dict, f, ensure_ascii=False, indent=2)
            
            # Markdown 报告
            self._generate_report(all_chains, output_dir / "traceback_report.md")
            
            cprint(f"✓ 结果已保存到: {output_dir}\n", "green")
        
        # ========== 步骤 5: 总结 ==========
        cprint("\n" + "="*80, "cyan")
        cprint("✅ 回溯思维链生成完成!", "cyan", attrs=["bold"])
        avg_cred = sum(c.credibility.get("overall_score", 0) for c in all_chains) / len(all_chains) if all_chains else 0
        cprint(f"  共 {len(all_chains)} 条回溯链 | 平均可信度: {avg_cred:.2f}", "cyan")
        cprint("="*80 + "\n", "cyan")
        
        return all_chains
    
    def _inject_ids(self, chain: TracebackChain, chain_idx: int):
        """
        为回溯链中的每一层元素注入唯一 ID，并建立层间引用关系。
        
        ID 格式:
          - 机理发现: M{chain_idx+1}
          - 实验步骤: E{chain_idx+1}.{step_idx+1}
          - Idea组件: I{chain_idx+1}.{comp_idx+1}
          - 策略节点: S{chain_idx+1}.{strat_idx+1}
        
        引用关系:
          - experiment_step.mechanism_id → 对应的机理发现 ID
          - idea_component.experiment_step_ids → 对应的实验步骤 ID 列表
          - strategy_mapping.idea_component_ids → 对应的 Idea 组件 ID 列表
        """
        prefix = chain_idx + 1
        mechanism_id = f"M{prefix}"
        
        # 1. 给机理发现注入 ID
        if isinstance(chain.mechanism_finding, dict):
            chain.mechanism_finding["id"] = mechanism_id
        
        # 2. 给实验步骤注入 ID + 关联 mechanism_id
        matched_steps = chain.experiment_layer.content.get("matched_steps", [])
        for i, step in enumerate(matched_steps):
            step["id"] = f"E{prefix}.{i+1}"
            step["mechanism_id"] = mechanism_id
        
        # 3. 给 Idea 组件注入 ID + 关联 experiment_step_ids
        idea_components = chain.idea_layer.content.get("idea_components", [])
        exp_step_ids = [s["id"] for s in matched_steps]
        for i, comp in enumerate(idea_components):
            comp["id"] = f"I{prefix}.{i+1}"
            # 匹配关联：通过文本匹配找出该 idea 组件对应哪些实验步骤
            comp["experiment_step_ids"] = self._match_idea_to_steps(
                comp, matched_steps, exp_step_ids
            )
        
        # 4. 给策略节点注入 ID + 关联 idea_component_ids
        strategy_mappings = chain.strategy_layer.content.get("strategy_mappings", [])
        idea_comp_ids = [c["id"] for c in idea_components]
        for i, strat in enumerate(strategy_mappings):
            strat["id"] = f"S{prefix}.{i+1}"
            # 匹配关联：通过 tuning_dimension 等信息找出对应的 Idea 组件
            strat["idea_component_ids"] = self._match_strategy_to_ideas(
                strat, idea_components, idea_comp_ids
            )
    
    def _match_idea_to_steps(
        self,
        idea_comp: Dict[str, Any],
        steps: List[Dict[str, Any]],
        step_ids: List[str]
    ) -> List[str]:
        """
        通过文本匹配将 Idea 组件关联到对应的实验步骤。
        匹配策略：检查 idea 的 implementation_in_experiment 是否提及某个 step。
        如果无法匹配，则返回所有 step_ids（全关联）。
        """
        impl = idea_comp.get("implementation_in_experiment", "").lower()
        idea_field = idea_comp.get("idea_field", "").lower()
        matched = []
        
        for step, sid in zip(steps, step_ids):
            step_id_str = str(step.get("step_id", "")).lower()
            step_name = step.get("step_name", "").lower()
            # 检查 idea 的实现描述中是否提及该步骤的 ID 或名称
            if step_id_str and step_id_str in impl:
                matched.append(sid)
            elif step_name and any(w in impl for w in step_name.split()[:3] if len(w) > 2):
                matched.append(sid)
            # 也检查关键词匹配
            elif self._keyword_overlap(idea_field, step_name, step.get("connection_to_finding", "")):
                matched.append(sid)
        
        return matched if matched else step_ids  # 无法匹配则全关联
    
    def _match_strategy_to_ideas(
        self,
        strat: Dict[str, Any],
        idea_comps: List[Dict[str, Any]],
        idea_ids: List[str]
    ) -> List[str]:
        """
        通过 tuning_dimension 和文本匹配将策略树节点关联到对应的 Idea 组件。
        """
        strat_dim = strat.get("tuning_dimension", "").lower()
        strat_evolution = strat.get("idea_evolution", "").lower()
        matched = []
        
        for comp, cid in zip(idea_comps, idea_ids):
            idea_field = comp.get("idea_field", "").lower()
            idea_content = comp.get("idea_content", "").lower()
            # 直接匹配 tuning_dimension
            if strat_dim and strat_dim in idea_field:
                matched.append(cid)
            # 检查 idea_evolution 是否与 idea_content 有关键词重叠
            elif self._keyword_overlap(strat_evolution, idea_content, idea_field):
                matched.append(cid)
        
        return matched if matched else idea_ids[:1]  # 无法匹配则关联第一个
    
    def _keyword_overlap(self, text1: str, text2: str, text3: str = "") -> bool:
        """检查两段文本是否有有意义的关键词重叠"""
        combined = (text2 + " " + text3).lower()
        keywords = [w for w in text1.lower().split() if len(w) > 3]
        if not keywords:
            return False
        overlap = sum(1 for kw in keywords if kw in combined)
        return overlap >= 2
    
    def _generate_report(self, chains: List[TracebackChain], output_path: Path):
        """生成结构化的可读 Markdown 报告"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 可回溯思维链分析报告\n\n")
            f.write("> **回溯路径**: 机理分析 → 实验方案 → Idea → 策略树节点\n")
            f.write("> **每层回溯均含**: 模板库(template/knowledge)知识溯源\n\n")
            f.write("---\n\n")
            
            # 总览
            f.write("## 总览\n\n")
            f.write(f"| 回溯链 | 机理发现 | 重要性 | 可信度 |\n")
            f.write(f"|--------|---------|-------|-------|\n")
            for idx, chain in enumerate(chains):
                name = chain.mechanism_finding.get("name", "N/A") if isinstance(chain.mechanism_finding, dict) else str(chain.mechanism_finding)
                importance = chain.mechanism_finding.get("importance", "N/A") if isinstance(chain.mechanism_finding, dict) else "N/A"
                score = chain.credibility.get("overall_score", 0)
                f.write(f"| {idx+1} | {name} | {importance} | {score:.2f} |\n")
            f.write("\n---\n\n")
            
            # 每个回溯链的详情
            for idx, chain in enumerate(chains):
                finding_name = chain.mechanism_finding.get('name', str(chain.mechanism_finding)) if isinstance(chain.mechanism_finding, dict) else str(chain.mechanism_finding)
                f.write(f"## 回溯链 {idx+1}: {finding_name}\n\n")
                
                # === 机理发现 ===
                f.write("### 🔬 层0: 机理发现\n\n")
                if isinstance(chain.mechanism_finding, dict):
                    for k, v in chain.mechanism_finding.items():
                        f.write(f"- **{k}**: {v}\n")
                else:
                    f.write(f"**发现**: {chain.mechanism_finding}\n")
                f.write("\n")
                
                # === 回溯到实验方案 ===
                f.write("### 🧪 层1: 回溯到实验方案\n\n")
                f.write(f"**置信度**: {chain.experiment_layer.confidence:.2f}\n\n")
                
                f.write("#### 匹配的实验步骤\n\n")
                for step in chain.experiment_layer.content.get("matched_steps", []):
                    step_eid = step.get('id', '')
                    mid = step.get('mechanism_id', '')
                    id_tag = f" `[{step_eid}]`" if step_eid else ""
                    ref_tag = f" → 关联机理 `{mid}`" if mid else ""
                    f.write(f"- **步骤 {step.get('step_id', 'N/A')}**{id_tag}: {step.get('step_name', 'N/A')}{ref_tag}\n")
                    f.write(f"  - 关键操作: {', '.join(step.get('key_operations', []))}\n")
                    params = step.get('parameters', {})
                    if params:
                        f.write(f"  - 参数: {json.dumps(params, ensure_ascii=False)}\n")
                    f.write(f"  - 与机理发现的联系: {step.get('connection_to_finding', 'N/A')}\n")
                    f.write(f"  - 置信度: {step.get('confidence', 0):.2f}\n\n")
                
                f.write("#### 模板库知识溯源 (实验步骤层)\n\n")
                if chain.experiment_layer.template_refs:
                    for kr in chain.experiment_layer.template_refs:
                        kid = kr.get('knowledge_id', 'N/A')
                        step_id = kr.get('from_step', 'N/A')
                        f.write(f"- **知识 [{kid}]** ← 实验步骤 {step_id}\n")
                        f.write(f"  - 核心方法: {kr.get('core_method', 'N/A')}\n")
                        guided = kr.get('guided_aspects', [])
                        if guided:
                            f.write(f"  - 指导方面: {', '.join(guided)}\n")
                        f.write(f"  - 推理链: {kr.get('reasoning_chain', 'N/A')}\n")
                        f.write(f"  - 对发现的贡献: {kr.get('contribution_to_finding', 'N/A')}\n")
                        conf = kr.get('confidence', 0)
                        if isinstance(conf, (int, float)):
                            f.write(f"  - 置信度: {conf:.2f}\n")
                        f.write("\n")
                else:
                    f.write("_无模板库知识引用_\n\n")
                
                f.write(f"**推理**: {chain.experiment_layer.reasoning}\n\n")
                
                # === 回溯到 Idea ===
                f.write("### 💡 层2: 回溯到 Idea\n\n")
                f.write(f"**置信度**: {chain.idea_layer.confidence:.2f}\n\n")
                
                f.write("#### Idea 组件\n\n")
                for comp in chain.idea_layer.content.get("idea_components", []):
                    comp_id = comp.get('id', '')
                    exp_refs = comp.get('experiment_step_ids', [])
                    id_tag = f" `[{comp_id}]`" if comp_id else ""
                    ref_tag = f" → 关联实验步骤 {', '.join(f'`{r}`' for r in exp_refs)}" if exp_refs else ""
                    f.write(f"- **{comp.get('idea_field', 'N/A')}**{id_tag}: {comp.get('innovation_point', 'N/A')}{ref_tag}\n")
                    f.write(f"  - 内容: {comp.get('idea_content', 'N/A')}\n")
                    f.write(f"  - 实验实现: {comp.get('implementation_in_experiment', 'N/A')}\n")
                    f.write(f"  - 与发现关联: {comp.get('connection_to_finding', 'N/A')}\n")
                    f.write(f"  - 置信度: {comp.get('confidence', 0):.2f}\n\n")
                
                f.write("#### 模板库知识溯源 (Idea层)\n\n")
                if chain.idea_layer.template_refs:
                    for ref in chain.idea_layer.template_refs:
                        tid = ref.get('template_id', 'N/A')
                        sf = ref.get('source_field', 'N/A')
                        f.write(f"- **模板 [{tid}]** → {sf}\n")
                        f.write(f"  - 原始描述: {ref.get('original_description', ref.get('description', 'N/A'))}\n")
                        f.write(f"  - 对 Idea 的指导: {ref.get('how_it_guided_idea', 'N/A')}\n")
                        f.write(f"  - 对发现的贡献: {ref.get('contribution_to_finding', 'N/A')}\n\n")
                else:
                    f.write("_无Idea层面模板库引用_\n\n")
                
                f.write("#### 策略树引用 (Idea层)\n\n")
                if chain.idea_layer.strategy_tree_refs:
                    for ref in chain.idea_layer.strategy_tree_refs:
                        nid = ref.get('node_id', 'N/A')
                        sf = ref.get('source_field', 'N/A')
                        f.write(f"- **节点 [{nid}]** → {sf}\n")
                        f.write(f"  - 原始描述: {ref.get('original_description', ref.get('description', 'N/A'))}\n")
                        f.write(f"  - 演化为 Idea: {ref.get('how_it_evolved_to_idea', 'N/A')}\n\n")
                else:
                    f.write("_无Idea层面策略树引用_\n\n")
                
                f.write(f"**推理**: {chain.idea_layer.reasoning}\n\n")
                
                # === 回溯到策略树 ===
                f.write("### 🌳 层3: 回溯到策略树\n\n")
                f.write(f"**置信度**: {chain.strategy_layer.confidence:.2f}\n\n")
                
                f.write("#### 策略节点映射\n\n")
                for strat in chain.strategy_layer.content.get("strategy_mappings", []):
                    verified = "✅ 已验证" if strat.get('verified_by_explicit_ref') else "🔍 推断"
                    strat_id = strat.get('id', '')
                    idea_refs = strat.get('idea_component_ids', [])
                    id_tag = f" `[{strat_id}]`" if strat_id else ""
                    ref_tag = f" → 关联Idea {', '.join(f'`{r}`' for r in idea_refs)}" if idea_refs else ""
                    f.write(f"- **Level {strat.get('strategy_level', 'N/A')}**{id_tag} [{verified}]{ref_tag}\n")
                    f.write(f"  - 调控维度: {strat.get('tuning_dimension', 'N/A')}\n")
                    f.write(f"  - 策略: {strat.get('tuning_strategy', 'N/A')}\n")
                    f.write(f"  - 调控效果: {strat.get('tuning_effect', 'N/A')}\n")
                    drawback = strat.get('drawback_addressed', '')
                    if drawback:
                        f.write(f"  - 克服的不足: {drawback}\n")
                    f.write(f"  - 演化为 Idea: {strat.get('idea_evolution', 'N/A')}\n")
                    f.write(f"  - 与发现联系: {strat.get('connection_to_finding', 'N/A')}\n")
                    f.write(f"  - 匹配分数: {strat.get('match_score', 0):.2f}\n\n")
                
                coverage = chain.strategy_layer.content.get("coverage_assessment", "")
                if coverage:
                    f.write(f"**策略树覆盖度**: {coverage}\n\n")
                
                f.write(f"**推理**: {chain.strategy_layer.reasoning}\n\n")
                
                # === 完整推理链 ===
                f.write("### 🔗 完整推理链\n\n")
                for i, reasoning in enumerate(chain.reasoning_chain, 1):
                    f.write(f"{i}. {reasoning}\n")
                f.write("\n")
                
                # === 可信度评估 ===
                f.write("### 📊 可信度评估\n\n")
                cred = chain.credibility
                f.write(f"| 评估维度 | 分数 |\n")
                f.write(f"|---------|------|\n")
                f.write(f"| 整体可信度 | **{cred.get('overall_score', 0):.2f}** |\n")
                f.write(f"| 实验方案回溯 | {cred.get('experiment_layer_score', 0):.2f} |\n")
                f.write(f"| Idea回溯 | {cred.get('idea_layer_score', 0):.2f} |\n")
                f.write(f"| 策略树回溯 | {cred.get('strategy_layer_score', 0):.2f} |\n")
                f.write(f"| 模板库支撑 | {cred.get('template_support_score', 0):.2f} |\n\n")
                
                weak_links = cred.get('weak_links', [])
                if weak_links:
                    f.write("**⚠ 薄弱环节:**\n")
                    for wl in weak_links:
                        f.write(f"- {wl}\n")
                    f.write("\n")
                
                strong_evidence = cred.get('strong_evidence', [])
                if strong_evidence:
                    f.write("**✅ 强证据:**\n")
                    for se in strong_evidence:
                        f.write(f"- {se}\n")
                    f.write("\n")
                
                suggestions = cred.get('suggestions', [])
                if suggestions:
                    f.write("**💡 改进建议:**\n")
                    for sg in suggestions:
                        f.write(f"- {sg}\n")
                    f.write("\n")
                
                kf = cred.get('knowledge_flow_summary', '')
                if kf:
                    f.write(f"**知识流动**: {kf}\n\n")
                
                f.write("---\n\n")


# ==============================================================================
# 入口
# ==============================================================================

def main():
    """主函数示例"""
    sys.path.append(str(Path(__file__).parent.parent))
    from supramas.agents import get_llms
    from supramas.conversation_logger import ConversationLogger

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    for key, value in config.items():
        if value == "":
            continue
        else:
            os.environ[key] = str(value)
    
    data_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversation_logger = ConversationLogger(log_file=output_dir / "traceback_conversation.json")
    llm = get_llms(conversation_logger)
    
    pipeline = TracebackPipeline(llm=llm, data_dir=data_dir)
    chains = pipeline.run(output_dir=output_dir)
    
    print(f"\n生成了 {len(chains)} 条回溯链")


if __name__ == "__main__":
    main()
