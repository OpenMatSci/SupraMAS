import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import fitz
from loguru import logger
from openai import OpenAI

from supramas.utils import *


class BaseAgent(ABC):
    def __init__(self, llm=None, model: str = None, temperature: float = 0.7, base_url: str = None, api_key: str = None):
        if llm is not None:
            # 使用外部传入的LLM实例
            self.llm = llm
            self.model = getattr(llm, 'model', model)
            self.temperature = temperature
        else:
            # 保持原有的创建方式作为后备
            self.model = model
            self.temperature = temperature
            if base_url and api_key:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))

    def generate_response(self, prompt: str, history: str = None, file_path: str = None, extract_key: str = None, load_json=False, **kwargs) -> str:
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
            messages = [
                {"role": "system", "content": "你是材料科学领域的专家，熟悉磁通钉扎的相关研究。阅读下面这篇文章内容，并根据文章内容完成任务。"},
                {"role": "user", "content": f"""文章如下:{file_content}"""},
            ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # 使用外部LLM的response_方法
        if hasattr(self, 'llm'):
            if extract_key:
                return self.llm.response(messages, extract_key=extract_key, temperature=self.temperature, max_tokens=8192, load_json=load_json, **kwargs)
            else:
                return self.llm.response(messages, temperature=self.temperature, max_tokens=8192, **kwargs)
        else:
            # 后备方案：使用原有的client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192,
            )
            if extract_key:
                from supramas.utils import extract
                return extract(response.choices[0].message.content, extract_key)
            else:
                return response.choices[0].message.content

    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        pass


class RootQuery(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict) -> Dict:
        if isinstance(input_data, str):
            topic = input_data
            retrive_queries = topic

            prompt = f"""
            You are an expert in REBCO superconductor research, specializing in flux pinning.  
            I will give you a research topic or description. Please generate a concise, effective **English search query** suitable for academic databases (e.g., Web of Science, Scopus).

            Guidelines:
            - Use standard abbreviations commonly seen in the literature (e.g., YBCO, GdBCO, BZO, PLD, MOCVD).
            - Include any relevant terms from the topic: REBCO type, dopants, fabrication methods, sample forms, etc.—but only if they appear or are clearly implied.
            - **Always end the query with the word "pinning"**.
            - Do not add quotes, Boolean operators, or extra symbols—just space-separated keywords.
            - If the topic is vague, generate the most reasonable query based on typical REBCO pinning studies.

            Example:  
            Input: "GdBCO films with BaZrO3 nanorods made by PLD"  
            Output: GdBCO BaZrO3 PLD films pinning

            Now process this topic:

            {topic}

            Output only the query inside <query> tags:
            <query>
            [generated search query]
            </query>
            """
            conversation_kwargs = {
                "stage": "strategy_mining",
                "role": "root_query_agent",
                "meta": {
                    "input_data": input_data,
                    "prompt": prompt,
                }
            }
            retrive_queries = self.generate_response(prompt, extract_key="query", **conversation_kwargs)
            
                    
        elif isinstance(input_data, dict):
            rebco = input_data["REBCO"]
            dopped_materials = input_data["dopped_materials"]
            processing_method = input_data["processing_method"]
            sample_category = input_data["sample_category"]

            prompt = f"""你是一名超导材料文献检索专家，专门从事REBCO超导体磁通钉扎研究。我会给你提供REBCO材料、掺杂材料、加工方式、样品类型。你需要将其组合成检索query。严格遵循以下步骤：
            ### 您的任务
            <REBCO>
            {rebco}
            </REBCO>
            <掺杂材料>
            {dopped_materials}
            </掺杂材料>
            <加工方式>
            {processing_method}
            </加工方式>
            <样品类型>
            {sample_category}
            </样品类型>
            
            ### 内容说明
            1. 提供的REBCO材料可能是化学式也可能是缩写，如YBa2Cu3O7 (YBCO)，请按照缩写YBCO的形式组合成检索query。
            2. 提供的掺杂材料可能是化学式也可能是缩写，如BaZrO3 (BZO)，只保留化学式或者缩写中的一种组合成检索query。可能会提供多种掺杂材料，请依次提取，组合到检索query中。
            
            ### 生成说明
            1.最终生成的检索query是英文
            2.任何情况下，检索query的最后都应该加上pinning 
            3.组合成检索query只需要顺序排列各个字段，不需要添加其它符号。
            
            ### 示例参考
            1.REBCO：REBa2Cu3O7-δ (RE= Eu and Gd) 
            提取：(Eu,Gd)BCO
            2.掺杂材料：BaZrO3, RE3TaO7 (e.g., Yb3TaO7)
            提取：BaZrO3 Yb3TaO7
            3.加工方式：PLD
            提取：PLD
            4.样品类型：films
            提取：films
            
            组合成最终query：
            (Eu,Gd)BCO BaZrO3 Yb3TaO7 PLD films pinning
            
            请按以下格式输出：
            <query>
            最终检索query（直接可用的完整字符串）
            </query>
            """
            kwargs = {
                "stage": "strategy_mining",
                "role": "root_query_agent",
                "meta": {
                    "input_data": input_data,
                    "prompt": prompt,
                }
            }
            retrive_queries = self.generate_response(prompt, extract_key="query", **kwargs)
            print("query", retrive_queries)
            while not retrive_queries:
                logger.info(f"提取review_result失败，重新生成!")
                retrive_queries = self.generate_response(prompt, extract_key="query", **kwargs)

        # 审核计划
        return {"root_queries": retrive_queries}


class Strategy2Query(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict) -> Dict:
        tuning_strategies = input_data['tuning_strategies']
        prompt = f"""
            你是一名超导材料文献检索专家，专门从事REBCO超导体磁通钉扎研究。请将以下调控策略转化为专业文献检索query，严格遵循以下步骤：
            
            ### 任务流程
            1. **核心要素提取**：
            - 识别策略中的关键实体：
                • 材料体系（如：REBCO, YBCO, SmBCO）
                • 掺杂元素/化合物（如：BZO, BSO, CeO₂）
                • 工艺方法（如：PLD, MOCVD, 热处理）
                • 结构特征（如：纳米柱、c轴取向、多层结构）
                • 性能指标（如：Jc, Fp, 钉扎力）
            
            2. **核心策略提取**：
            - 将研究策略转化为关键词：
                • 成分掺杂策略：如梯度掺杂、多元掺杂
                • 晶界优化策略：如晶界钉扎
                • 界面优化策略：如晶格失配
                • 织构优化策略：如取向调控
                • 应力优化策略：如热膨胀系数匹配
            
            3. **学术术语映射**：
            - 将通俗表述转为学术用语：
                • "增强" → "enhancement" OR "improvement"
                • "纳米颗粒" → "nanoparticles" OR "nanodots"
                • "磁通钉扎" → "flux pinning"
                • "临界电流" → "critical current density" OR "Jc"
            
            3. **query构建规范**：
            - query由2-4个关键词组成
            - 组成query的关键词应该是特殊的工艺、结构性能、核心策略等富有创新性的方法
            - 将关键词按顺序排列成短句
            - 关键词之间不含任何连接词
            
            ### 示例参考
            [调控策略]
            通过BaZrO₃纳米柱掺杂和低温热处理(350℃)增强REBCO薄膜的c轴磁通钉扎强度
            
            [生成过程]
            1. 要素提取：
            - 材料：REBCO, YBCO
            - 掺杂：BaZrO₃ (BZO)
            - 结构：纳米柱(nanorods), c轴取向
            - 性能：磁通钉扎(flux pinning), Jc
            
            2. 策略提取：
            - 低温热处理(low-temperature heat treatment）
            - c轴钉扎（c-axis pinning）
            
            3. 术语映射：
            - "增强" → "enhancement"
            - "纳米柱" → "nanorods" OR "nanocolumns"
            - "低温" → "low temperature"
            
            4. 最终query：
            low-temperature heat treatment c-axis pinning
            
            ### 您的任务
            <调控策略>
            {tuning_strategies}
            </调控策略>
            
            ### 生成说明
            1.提供的调控策略只会包含部分实体与策略
            2.最终生成的检索query是英文
            3.任何情况下，关键词中都应包含pinning或者pinning的衍生词
            
            请按以下格式输出：
            <query>
            最终检索query（直接可用的完整字符串）
            </query>


        """

        conversation_kwargs = {
            "stage": "strategy_mining",
            "role": "strategy2query_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        retrive_queries = self.generate_response(prompt, extract_key="query", **conversation_kwargs)

        # 审核计划
        return {"retrive_queries": retrive_queries}


class ReviewAgent(BaseAgent):
    def process(self, input_data: Dict) -> Dict:
        file_path = input_data.get("file_path")
        drawbacks_prompt = f"""
##### 任务一说明
请你关注论文中的“结果与讨论”部分，完成以下任务：
###### 提取“研究中存在的不足”，包括两方面内容：
1. 实验方法与技术上的不足：
   - 如：成分调控不充分、工艺方法受限、沉积温度控制不佳、外延生长质量不足等
2. 实验结果上的不足：
   - 如：薄膜与缺陷结构形貌不佳、钉扎强度有限、临界电流密度提升有限等

如果文献未直接描述实验结果上的不足，可以从方法不足中合理推断出可能导致的结果不足。
###### 输出格式要求：
用如下JSON格式输出 1 条不足内容（可根据文献情况裁剪），输出内容须用英文填写，并确保源自文献内容或合理推断，请将内容添加到在下面的<shortcomings>内：
<shortcomings>
{{
  "shortcoming1":"...",
  "shortcoming2":"...",
  "shortcoming3":"...",
  "shortcoming4":"..."
}}
</shortcomings>
        """
        history = []

        conversation_kwargs = {
            "stage": "strategy_mining",
            "role": "drawback_agent",
            "meta": {
                "input_data": input_data,
                "prompt": drawbacks_prompt,
            }
        }
        drawbacks = self.generate_response(drawbacks_prompt, file_path=file_path, extract_key="shortcomings", load_json=True, **conversation_kwargs)

        history = [{"role": "user", "content": drawbacks_prompt}, {"role": "assistant", "content": json.dumps(drawbacks)}]

        exception_prompt = f"""
        ##### 任务二说明
请总结的“实验结果上的不足”（<shortcomings>内），将这些不足直接转化为“期望的研究效果”，以指导后续研究目标的设定。
<shortcomings>
{json.dumps(drawbacks)}
</shortcomings>


###### 转化内容示例：
- 如果不足是：“缺陷尺寸分布不均匀”，期望可以是：“获得均匀分布的纳米尺度缺陷”
- 如果不足是：“磁通钉扎强度有限”，期望可以是：“显著提升磁通钉扎强度以增强临界电流密度”

###### 输出格式要求：
用如下 JSON 格式输出 1 条期望内容（必须紧扣任务一结果不足转化而来），字段内容须用英文填写,请将内容添加到在下面的<expectation>内：
<expectation>
{{
  "expectation1": "...",
  "expectation2": "...",
  "expectation3": "...",
  "expectation4": "...",
}}
</expectation>
        """

        conversation_kwargs = {
            "stage": "strategy_mining",
            "role": "expectation_agent",
            "meta": {
                "input_data": input_data,
                "prompt": exception_prompt,
            }
        }
        expectations = self.generate_response(exception_prompt, history=history, file_path=file_path, extract_key="expectation", load_json=True, **conversation_kwargs)

        return drawbacks, expectations


class ParserAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict) -> Dict:
        file_path = input_data.get("file_path")
        prompt = f"""
请重点阅读文献中的“摘要、引言、结果分析”部分，从中提取关于以下五个维度的调控策略与实验效果，并用结构化模板表达出来：
五个调控维度包括：
1. Composition tuning（成分调控）
2. Grain boundary tuning（晶界调控）
3. Interface tuning（界面调控）
4. Texture tuning（织构调控）
5. Stress tuning（应力调控）

每一个模板应包括以下字段：
- tuning dimensions: 从上述五个调控维度中选一个
- tuning strategies: 针对该调控维度所采用的调控手段
- tuning effects: 该策略对缺陷形貌、钉扎性能或临界电流的提升效果（包括材料、掺杂物、工艺参数等）

输出格式要求：
以如下JSON数组格式输出5个调控策略模板，分别对应上面的五个维度，每个调控策略严格遵循三字段结构,输出内容需是英文,请将内容添加到在下面的<template>内：
<template>
[{{
  "tuning dimensions": "...",
  "tuning strategies": "...",
  "tuning effects": "..."
}},
{{
  "tuning dimensions": "...",
  "tuning strategies": "...",
  "tuning effects": "..."
}},
and so on
......
]
</template>
        """
        conversation_kwargs = {
            "stage": "strategy_mining",
            "role": "strategy_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        templates = self.generate_response(prompt, file_path=file_path, extract_key="template", load_json=True, **conversation_kwargs)

        return {"templates": templates}


class PaperNode:
    def __init__(self, title: str, article: str, tuning_strategy: str, draw_backs: str, desired_effects: str, url: str):
        self.title = title
        self.article = article
        self.tuning_strategy = tuning_strategy
        self.draw_backs = draw_backs
        self.desired_effects = desired_effects
        self.children = []
        self.children_num = 0
        self.url = url

    def add_child(self, child_node):
        self.children.append(child_node)
        self.children_num += 1


class PaperTree:
    def __init__(self, topic, root_paper: Optional[PaperNode] = None):
        self.topic = topic
        self.root = root_paper

    def set_root(self, root_paper: PaperNode):
        self.root = root_paper

    def find_node(self, title: str, start_node: Optional[PaperNode] = None) -> Optional[PaperNode]:
        """根据论文标题查找节点"""
        if not start_node:
            if not self.root:
                return None
            start_node = self.root

        if start_node.title == title:
            return start_node

        for child in start_node.children:
            result = self.find_node(title, child)
            if result:
                return result

        return None

    def add_paper(self, paper: PaperNode, parent_title: Optional[str] = None) -> bool:
        """添加论文到树中"""
        if not self.root:
            self.root = paper

            return True

        if not parent_title:
            # 如果没有指定父节点，添加到根节点
            self.root.add_child(paper)
            return True

        parent_node = self.find_node(parent_title)
        if parent_node:
            parent_node.add_child(paper)
            return True
        else:
            return False

    def get_deepth(self, node: Optional[PaperNode] = None) -> int:
        """获取树的深度"""
        if not node:
            if not self.root:
                return 0
            node = self.root

        if not node.children:
            return 1

        max_depth = 0
        for child in node.children:
            depth = self.get_deepth(child)
            if depth > max_depth:
                max_depth = depth

        return max_depth + 1

    # 将树导出为结构化的数据
    def export_mardown(self, saved_path="paper_tree.md") -> str:
        """导出为Markdown格式"""
        if not self.root:
            return ""

        def _export(node: PaperNode, level: int, step: str) -> str:
            indent = "  " * level
            if level == 0:
                result = f"{indent}- **{self.topic}**\n"
            else:
                result = f"{indent}- **{node.title}**\n"
            result += f"{indent} - level: {step}\n"
            # result += f"{indent}  - 文章: {node.article}\n"
            result += f"{indent}  - stagites and effects: {node.tuning_strategy}\n"
            result += f"{indent}  - Drawbacks: {node.draw_backs}\n"
            result += f"{indent}  - expections: {node.desired_effects}\n"
            children = []
            for i in range(node.children_num):
                child = node.children[i]
                children.append(step + f".{i}")

                result += _export(child, level + 1, step + f".{i}")
            result += f"{indent} - childern: {children}"
            # 统一保存到topic_path目录
            if hasattr(self, 'topic_path') and self.topic_path:
                saved_path = os.path.join(self.topic_path, saved_path)
            with open(saved_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"paper tree已保存：{saved_path}")
            return result

        return _export(self.root, 0, '1')

    def export_json(self, topic, saved_path="paper_tree.json") -> str:
        """导出为Markdown格式"""
        if not self.root:
            return ""

        def _export_json(node, level, step):
            node_data = {
                "level": step,
                "title" if level != 0 else "topic": node.title if level != 0 else topic,
                "abstract": node.article,  # 是str
                "url": node.url,
                "tuning_strategies": node.tuning_strategy,  # 要求是list[dict]
                "draw_backs": node.draw_backs,  # 是dict
                "expections": node.desired_effects,  # 是dict
                "children": [],
            }
            logger.info(f"children_num{node.children_num}")
            for i in range(node.children_num):
                child = node.children[i]
                node_data["children"].append(_export_json(child, level + 1, step + f".{i+1}"))
            return node_data

        tree_dict = _export_json(self.root, 0, "1")

        # 统一保存到topic_path目录
        if hasattr(self, 'topic_path') and self.topic_path:
            saved_path = os.path.join(self.topic_path, saved_path)
        # 保存成 .json 文件
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)

        print(f"Paper tree已保存：{saved_path}")
        return tree_dict
