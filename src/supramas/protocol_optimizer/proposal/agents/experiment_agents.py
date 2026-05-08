import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List
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

    def generate_response(self, prompt: str, history: str = None, file_path: str = None, extract_key: str = None,load_json=False, **kwargs) -> str:
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
                {"role": "system", "content": "你是凝聚态物理领域的专家，熟悉 REBCO 磁通钉扎的相关研究。"},
                {"role": "system", "content": file_content},
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


class DraftGenerationAgent(BaseAgent):
    """实验生成智能体 - 负责生成实验初稿"""

    def process(self, input_data: Dict) -> Dict:
        prompt = f"""
##### Task Description
Your task is to generate a first draft of an experimental plan for enhancing the current-carrying capacity of artificial-coated superconductors through flux pinning, based on the scientific research idea provided by the user. Describe the experimental procedures required and the factors to be considered in each step of the process.
Please carefully study the following experimental concept and strictly follow the instructions to complete the writing of the first draft.

##### User's Idea
{input_data['idea']}

##### Requirements for the Content of the First Draft
1. The first draft must comprehensively cover all the basic elements of the experiment, such as the experimental purpose, experimental materials, experimental equipment, experimental procedures, precautions, expected results, etc., and the description of each element should be detailed.
2. For the experimental procedure part, it needs to cover the defect introduction strategy and preparation of the target, the selection and preparation of the texture strategy of the metal substrate and buffer layer, the thin film preparation process of the superconducting layer, the characterization of the thin film and defects, the testing of superconducting and current-carrying properties, and the analysis of experimental results (the specific experimental procedure is determined according to the idea provided by the user, and creative content can be generated beyond the process paradigm).
3. For the precautions part, it is necessary to explain the factors to be considered in each link of the experimental design and the impact of each link on the final experimental result.
3.1: Explain what physical and material science factors need to be considered in the defect introduction strategy and preparation process of the target, how to analyze and test, and what experimental methods are needed to achieve this. At the same time, how will the target and doping affect the thin film deposition and the final experimental result.
3.2: Explain what material science and crystallography factors need to be considered in the preparation of the superconducting substrate, isolation layer, buffer layer, etc., how to analyze and test, and what experimental methods are needed to achieve this. At the same time, how will the texture of the superconducting substrate and isolation layer affect the thin film deposition and the final experimental result.
3.3: Explain what material science, kinetic and thermodynamic factors need to be considered in the entire experimental process of thin film deposition, how to analyze and test, and what experimental methods are needed to achieve this. At the same time, how will the experimental process and the texture of the superconducting layer affect the experimental result.
3.4: Explain which indicators can characterize the morphology of the thin film and defects. In order to obtain the required artificial pinning centers, what factors should be considered in the target doping and thin film deposition process, how to analyze and test the defect morphology, and what experimental methods are needed. At the same time, how will these different types of thin film defect morphologies affect the pinning strength and the final experimental result in different temperature and magnetic field environments.
3.5: Explain which indicators can characterize the pinning strength and current-carrying performance. Analyze from the results: in order to improve the current-carrying performance in different temperature and magnetic field environments, how to use the pinning mechanism to design artificial pinning centers.
3.6: Comprehensively analyze the electrical, thermal, mechanical properties, uniformity and isotropy of the sample. How to improve these properties by optimizing each process.
4. The language expression should be clear and accurate, and ambiguous or unclear expressions should be firmly avoided to ensure that readers can accurately understand the experimental content.
5. The content of the first draft must be completely in accordance with the requirements, and the overall logic is rigorous and coherent, with natural transitions between each part and no logical loopholes.
6. The first draft should present a complete experimental design framework, enabling readers to have a relatively clear and comprehensive understanding of the entire experiment by reading the first draft.

##### Output Format Requirements
Please write your first draft of the experiment in the following standard area in the following format:
1.Experimental Purpose: Elaborate on the experimental purpose in detail here
2.Experimental Materials: List the required raw materials, materials that need to be synthesized, doping materials, etc.
  and experimental instruments: List the experimental equipment and experimental techniques required for each step of the experiment
3.Experimental Procedures: Plan the overall experimental process, including the key points mentioned in the requirements for the content of the first draft
  and precautions: Explain the factors to be considered in each link of the experimental design and the impact of each link on the final experimental result
4.Expected Results: Describe the expected experimental results

The maximum heading level is five, using "##### "

##### Please write your first draft of the experiment in the following standard area in the following format.

<draft>
Write the first draft of the experiment here according to the above format.The maximum heading level is five, using "#####".
</draft>
"""
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "draft_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        # 调用 LLM 生成实验初稿
        draft = self.generate_response(prompt, extract_key="draft",**conversation_kwargs)
        while not draft:
            logger.info(f"提取draft失败，重新生成!")
            draft = self.generate_response(prompt, extract_key="draft",**conversation_kwargs)
            logger.info(f"实验初稿: {draft}")
        return {"draft":draft}


class ExperimentDecompositionAgent(BaseAgent):
    """实验拆解智能体 - 将实验初稿拆分为多个实验计划"""

    def process(self, input_data: Dict) -> Dict:
        prompt = f"""
        You are a senior experimental design assistant. Your task is to break down the provided first draft of an experiment into multiple **executable, sequential experimental plans**.

        First, carefully read the experiment draft:
        <experiment_draft>
        {input_data['draft']}
        </experiment_draft>

        ##### Decomposition Guidelines:
        1. Carefully analyze each step in the first draft of the experiment and the corresponding precautions, and strictly break down the experiment according to the content of the first draft.
        2. Analyze each independently executable part and take it as a separate experimental step.
        3. Clearly define the ultimate goal of the experiment, establish the corresponding goal for each experimental step, and analyze the existing experimental basis and precautions for each experimental step.
        4. Ensure that each experimental step has a clear goal, procedure, and expected result.
        5. There should be a logical sequence and coherence among the experimental plans.
        6. Avoid omitting important information in the first draft of the experiment.

        ##### Output Requirements:
        - Output **only one JSON object**, wrapped strictly in `<subtask>` tags.
        - Use keys `"exp1"`, `"exp2"`, `"exp3"`, ... in sequential order (no skipping).
        - For each broken - down experimental step, it should include the following parts:
            1. Existing experimental basis
            2. Experimental goal
            3. Experimental materials and instruments
            4. Factors to be considered in the experiment

        **Strictly follow this format**:

        <subtask>
        {{
        "exp1": [Experimental Plan 1],
        "exp2": [Experimental Plan 2],
        ...
        }}
        </subtask>

        Do not include any additional text, XML tags, markdown, or explanations outside the <subtask> tags.
        """
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "subtask_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }

        subtasks = self.generate_response(prompt,extract_key="subtask",load_json=True,**conversation_kwargs)

        return {"steps":subtasks}


class RetrievalAgent(BaseAgent):
    """检索智能体 - 生成相关概念"""

    def process(self, input_data: Dict) -> Dict:
        plan = input_data['plan']
        prompt = f"""
##### You are an expert in the field of high-temperature superconductivity and flux pinning, and you need to accurately extract the core research objectives from the experimental plan document. Please process the provided experimental plan according to the following steps:
<Experimental Plan>
{plan}
</Experimental Plan>
##### Processing Steps:
1. Read through the entire experimental plan, paying attention to the existing experimental basis, experimental objectives, experimental materials and instruments, and factors to be considered in the experiment.
2. Convert the experimental plan into a concise declarative sentence, stating what material system is used, what means are employed, what effect is achieved, and explain the reason for achieving this effect.

##### Output Requirements:
1. Arrange the generated queries in the <queries> tag in numerical order.
2. Control the length of the query statement between 25 and 50 words.
3. The maximum heading level is five, using "##### ".  

##### Example Format:
<queries>
{{
    "query1": "",
    "query2": "",
    "query3": "",
    ...
}}
</queries>
"""
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "knowlege_query_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        queries = self.generate_response(prompt, extract_key="queries",load_json=True,**conversation_kwargs)

        # 根据计划生成相关概念
        return queries


class KnowledgeBaseAgent(BaseAgent):
    """知识库智能体 - 检索相关知识"""

    def process(self, input_data: Dict) -> Dict:
        plan = input_data['plan']
        templates = input_data['templates']
        # 检索知识库
        prompt = f"""
#####You are an expert in the field of high - temperature superconductivity and flux pinning. Your task is to find the knowledge used in the experimental plan from the given structured template library. Please carefully read the following content and operate as required.
Experimental idea:
<idea_plan>
{plan}
</idea_plan>

#####Retrieved structured template library:
<kb_knowledge>
{templates}
</kb_knowledge>

#####When analyzing, please follow these steps:
1.Carefully study the entire idea, paying attention to the motivation, novelty, and required knowledge.
2.The structured template library is a reference for the experimental process and lists the factors to be considered during the experiment. In combination with the experimental plan, focus on the information in the structured template library that is helpful for achieving the research objective of the idea.
3.For the knowledge and factors that need to be considered in the idea, generate a guiding knowledge chain based on the template content.

###Output Format:
1. The maximum heading level is five, using "##### ".  
2. Please generate the knowledge chain within the <knowledge> tag.
Each knowledge item should be presented in the following format:
[
    {{
        "knowledge_id": "<specific knowledge ID>",
        "knowledge": "<useful knowledge corresponding to this ID>"
    }},
    ...
]
Multiple knowledge items should be separated by commas. The overall output must be accurate, concise, and closely aligned with the knowledge actually used in the experiment plan.

<knowledge>
Enter the generated knowledge chain here.
</knowledge>
            """
        conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "knowledge_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        knowledge = self.generate_response(prompt, extract_key="knowledge",load_json=True,**conversation_kwargs)

        knowledge_dict = {item["knowledge_id"]: item["knowledge"] for item in knowledge}

        return knowledge_dict


class ExpertAgent(BaseAgent):
    """专家智能体 - 制定详细实验计划"""

    def process(self, input_data: Dict) -> Dict:
        plan = input_data['plan']
        templates = input_data['templates']
        # knowledge = input_data['knowledge']
        history = input_data['history']
        feedback = input_data['feedback']
        prompt = f"""
##### You are an expert in the field of high - temperature superconductivity and flux pinning. Your task is to generate a complete experimental plan using the given preliminary research plan and the reference structured template library.
Meanwhile, the previous experimental plan and feedback will also be provided for your reference.
First, please carefully read the following preliminary experimental plan:
<Experimental Plan>
{plan}
</Experimental Plan>

Now, please carefully read the following retrieved structured template library:
<Relevant Knowledge>
{templates}
</Relevant Knowledge>

The following are the optimization suggestions. By default, it is None.
<Optimization Suggestions>
{feedback}
</Optimization Suggestions>

The following is the previous experimental plan. By default, it is None.
<Previous Experimental Plan>
{history}
(The format is: Sub-experiment 1: ..., Sub-experiment 2: ..., ...)
</Previous Experimental Plan>

###When analyzing, please follow these steps:
1. If there are experiment generation suggestions, prioritize the suggestions as the criteria for generating the complete experimental plan.
2. Carefully study the experimental plan, paying attention to the experimental objectives, experimental materials and instruments, and experiment precautions.
3. If there is a previous experimental steps, the experimental plan to be generated must consider it as the existing experimental basis.
   Ensure the logicality and relevance between the experimental plan generation and previous experimental steps.
4. The structured template library is a reference for the experimental process and provides the factors to be considered during the experiment. 
   Combine with the experimental plan and focus on the information in the structured template library that is helpful for completing the experiment.
   Especially in the target material preparation process, the reduction of the carbon content must be taken into account. Multiple means such as repeated grinding and calcination should be comprehensively applied to minimize the carbon content as much as possible.
5. For the factors to be considered in the experimental plan and experiment precautions, you need to design the experimental plan based on these factors.
6. When conducting experimental design, the experimental steps should be very finely subdivided and no experimental details should be missing. In particular, the experimental process parameters must be complete, such as PLD parameters. 
   If there are uncertainties regarding the process parameters, you should not make them up randomly. Instead, you should at least specify a range for them.
7. The expected goals of the experiment should be detailed. For instance, the expected value of the critical current density should be specified under a specific temperature and electric field strength.
8. Finally, the experimental plan to be generated in is a part of a whole experiment. 
   The final content does not need to over-analyze the results of this experimental plan, nor does it need to include the steps of previous experiments.

##### Output Format
1. The maximum heading level is five, using "##### ".  

<detail_exp> 
[Provide the optimized experimental plan here]
</detail_exp>
"""
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "plan_expert_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        # 制定详细计划
        detailed_plan = self.generate_response(prompt, extract_key="detail_exp",**conversation_kwargs)
        while not detailed_plan:
            logger.info(f"提取detailed_plan失败，重新生成!")
            detailed_plan = self.generate_response(prompt, extract_key="detail_exp",**conversation_kwargs)
        return detailed_plan


class ReviewAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict) -> Dict:
        detailed_plan = input_data['detailed_plan']
        knowledge = input_data['knowledge']
        prompt = f"""
##### Your task is to check whether an experimental plan is reasonable based on the given external knowledge.
First, please carefully read the following experimental plan:
<Experimental Plan>
{detailed_plan}
</Experimental Plan>
 
Then, please carefully read the following external knowledge:
<External Knowledge>
{knowledge}
</External Knowledge>

##### When checking, please follow these methods:
1. Compare and analyze each step of the experimental plan with the external knowledge.
2. Consider whether the steps in the experimental plan violate the principles, rules, etc. in the external knowledge.
3. Comprehensively consider the overall logic and feasibility of the experimental plan.

Give the judgment result in the <result> tag. Return "True" if you think the experimental plan is reasonable, and "False" if it is not. If the judgment is "False", clearly point out the unreasonable parts and the judgment basis based on the external knowledge in the <reason> tag.
<result>
{{
    "judgment": "True",
    "reason": "The experimental plan is reasonable."
}}
</result>
"""
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "plan_critic_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }

        review_result = self.generate_response(prompt, extract_key="result",load_json=True,**conversation_kwargs)

        judge = review_result["judgment"]
        reason = review_result["reason"]        
        # 审核计划
        return {"review_result": judge, "feedback": reason}



class PlanWritingAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict) -> Dict:
        draft = input_data['draft']
        plans = input_data['plans']
        prompt = f"""
You will assist in perfecting the experimental plan by integrating the initial experimental draft and experimental steps into a coherent and complete experimental plan. Please follow the following process:
First, receive the initial experimental draft:
<Initial experimental draft>
{draft}
</Initial experimental draft>

Then, review the original steps to be integrated:
<Original steps>
{plans}
</Original steps>

##### Please execute the following optimization and integration process:
Structural analysis phase
Identify the missing links in the initial draft and the logical gaps between steps.
Review the expressions that may cause ambiguity.

Step optimization phase
First, reorganize the experimental steps:
a) Ensure that each step has clear pre - conditions.
b) Add necessary transitional explanations.
c) Include safety warnings and key checkpoints.
d) Standardize measurement units and technical terms.

Content integration phase
Incorporate the optimized steps into the framework of the initial draft.
Add the following elements:
• A list of materials and equipment preparation (indicating quantities and specifications).
• Descriptions of expected phenomena at each stage.
• Experimental discussion: Discuss experimental phenomena and experimental data.
• Data Recording Standards

Verification phase
1.Check the coherence of each step. Ensure that each step builds on the phased results of the previous step and that the expected results of each step are consistent with the overall goal.
2.Check and supplement the missing process parameters for each experimental step.

##### Finally, output the complete experimental plan within the <final_plan> tag. Requirements:
1.Divide the experiment into stages (preparation/implementation process/analysis).
2.Summarize key process parameters in a table.
3.Maintain objectivity in technical descriptions and avoid subjective speculation.
4.Mark all modifications with [Revised] and state the reasons.

##### Formatting Requirements:  
When formatting content, please follow these guidelines:  
1. The maximum heading level is five, using "##### ".  
2. Ensure compliance with Markdown syntax standards.  
3. Maintain the original meaning of the content.  

##### Notes:  
• Use the ⚠️ symbol to mark steps involving hazardous operations.  
• Maintain objectivity in technical descriptions and avoid subjective speculation.  
• All modifications must be marked with **[Revision]** and include an explanation.  
• Time estimation errors must not exceed ±15%.

##### Output Format Requirements
<final_plan>
output the complete experimental plan there
</final_plan>
        """
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "refine_experiment_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        final_plan = self.generate_response(prompt, extract_key="final_plan",**conversation_kwargs)
        while not final_plan:
            logger.info(f"提取final_plan失败,重新生成!")
            final_plan = self.generate_response(prompt, extract_key="final_plan",**conversation_kwargs)
        # 审核计划
        return {"final_plan":final_plan}


class ExperimentPlan:
    def __init__(self, idea=None, draft=None):
        """初始化实验计划"""
        self.idea = idea
        self.draft = draft
        self.final_plan = None
        self.steps = []  # 存储实验步骤
        self.expected_results = []
        self.references = []


    def add_step(self, step_id, description, detailed_plan=None, knowledge=None):
        """添加实验步骤"""
        self.steps.append({"id": step_id, "description": description, "detailed_plan": detailed_plan, "knowledge": knowledge})


    def update_goal(self, new_goal):
        """更新实验目标"""
        self.draft = new_goal

    def update_step(self, step_id, description=None, detailed_plan=None, knowledge=None):
        """修改某个实验步骤的描述"""
        for step in self.steps:
            if step["id"] == step_id:
                step["description"] = description if description else step["description"]
                step["detailed_plan"] = detailed_plan 
                step["knowledge"] = knowledge
                return True
        return False

    def remove_step(self, step_id):
        """删除某个实验步骤"""
        self.steps = [step for step in self.steps if step["id"] != step_id]

    def add_expected_result(self, result):
        """添加预期实验结果"""
        self.expected_results.append(result)

    def add_reference(self, reference):
        """添加文献参考"""
        self.references.append(reference)

    def get_step(self, step_id):
        """获取某个实验步骤的描述"""
        for step in self.steps:
            if step["id"] == step_id:
                return step["description"]
        return None

    def get_steps(self, step_id):
        """获取第id前所有的实验步骤"""
        steps = []
        for step in self.steps:
            if step["id"] != step_id:
                steps.append(step)
            if step["id"] == step_id:
                break
        if steps != []:
            return steps
        return None

    def to_dict(self):
        """将实验计划转换为字典"""
        return {
            "idea": self.idea,
            "draft": self.draft,
            "steps": self.steps,
            "expected_results": self.expected_results,
            "references": self.references,
            "final_plan": self.final_plan,
            # 删除sub_exp属性，因为steps已经存储了相同的数据
        }


    @staticmethod
    def _format_idea_markdown(idea):
        """将 idea dict 格式化为可读的 Markdown"""
        if isinstance(idea, str):
            return idea

        if not isinstance(idea, dict):
            return str(idea)

        md = ""
        # 定义主要字段及其中文标签
        main_fields = [
            ("material", "材料体系"),
            ("motivation", "研究动机"),
            ("technique", "实验技术"),
            ("outcome", "预期结果"),
        ]

        for key, label in main_fields:
            if key in idea:
                md += f"**{label}** ({key}): {idea[key]}\n\n"
                # 输出对应的来源信息
                source_key = f"{key}_source"
                if source_key in idea and isinstance(idea[source_key], dict):
                    for node_id, source_text in idea[source_key].items():
                        md += f"> **来源 [{node_id}]**: {source_text}\n"
                    md += "\n"

        # 格式化 strategies
        if "strategies" in idea and isinstance(idea["strategies"], list):
            md += "**调控策略** (strategies):\n\n"
            for idx, strategy in enumerate(idea["strategies"], 1):
                dim = strategy.get("tuning_dimensions", "")
                strat = strategy.get("tuning_strategies", "")
                md += f"{idx}. **{dim}**: {strat}\n"
                # 输出策略来源
                source_node = strategy.get("source_node", {})
                if isinstance(source_node, dict):
                    for node_id, source_text in source_node.items():
                        md += f"   > **来源 [{node_id}]**: {source_text}\n"
                md += "\n"

        # 格式化 node_mapping
        if "node_mapping" in idea and isinstance(idea["node_mapping"], dict):
            nm = idea["node_mapping"]
            md += "**节点映射** (node_mapping):\n\n"
            for k, v in nm.items():
                md += f"- **{k}**: {v}\n"
            md += "\n"
            # 输出 node_mapping_source
            if "node_mapping_source" in idea and isinstance(idea["node_mapping_source"], dict):
                for k, v in idea["node_mapping_source"].items():
                    md += f"> **{k}**: {v}\n"
                md += "\n"

        return md

    @staticmethod
    def _format_knowledge_markdown(knowledge):
        """将 knowledge dict 格式化为可读的 Markdown"""
        if isinstance(knowledge, str):
            return knowledge

        if not isinstance(knowledge, dict):
            return str(knowledge) if knowledge else ""

        md = ""
        for kid, desc in knowledge.items():
            md += f"- **知识 [{kid}]**: {desc}\n"
        return md

    def export_markdown(self):
        """导出实验计划为 Markdown"""
        md = "# Idea 创新 \n"
        md += self._format_idea_markdown(self.idea)
        md += "\n"

        md += f"# 实验计划 \n ## 实验初稿\n{self.draft}\n\n"

        md += "## 详细实验方案\n"
        for step in self.steps:
            md += f"\n### 实验任务{step['id']+1}\n"
            if isinstance(step["description"], dict):
                for k,v in step["description"].items():
                    md += f"\n- **{k}**: {v}\n"
            elif isinstance(step["description"], list):
                md += f"\n{step['description'][0]}\n"

            md += f"\n#### 具体内容 \n{step['detailed_plan']}\n"
            md += f"\n#### 相关知识 \n{self._format_knowledge_markdown(step['knowledge'])}\n"

        md += "\n# 精炼的实验计划\n"
        md += f"{self.final_plan}\n"

        return md

    # def export_markdown(self):
    #     """导出实验计划为 Markdown"""
    #     md = f"# Idea 创新 \n{self.idea}\n\n"
    #     md += f"# 实验计划 \n ## 实验初稿\n{self.draft}\n\n"

    #     md += "## 详细实验方案\n"
    #     for step in self.steps:
    #         md += f"\n### 实验任务{step['id']} \n{step['description']}\n"
    #         md += f"\n#### 详细实验计划 \n{step['detailed_plan']}\n"
    #         md += f"\n#### 相关知识 \n{step["knowledge"]}"

    #     md += "\n# 精炼的实验计划\n"
    #     md += f"{self.final_plan}\n"

    #     # md += "\n## 预期结果\n"
    #     # for result in self.expected_results:
    #     #     md += f"- {result}\n"

    #     # md += "\n## 参考文献\n"
    #     # for ref in self.references:
    #     #     md += f"- {ref}\n"

    #     return md

    def save_dict(self, file_path=None):
        """保存 Markdown 文件"""
        md_dict = self.to_dict()

        # 统一保存到topic_path目录
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(md_dict, f, ensure_ascii=False, indent=4)

        print(f"实验计划 Markdown 已保存：{file_path}")



    def save_markdown(self, file_path=None):
        """保存 Markdown 文件"""
        md_content = self.export_markdown()
        # 统一保存到topic_path目录
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"实验计划 Markdown 已保存：{file_path}")

