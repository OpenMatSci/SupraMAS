from abc import ABC, abstractmethod
import json
import json
import os
from typing import Dict, List

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

class RetrievalAgent(BaseAgent):
    """检索智能体 - 生成相关概念"""

    def process(self, input_data: Dict) -> Dict:
        plan = input_data['plan']
        prompt = f"""
You are an expert in the field of high-temperature superconductivity and flux pinning. Your task is to convert a hierarchical strategy tree—which includes regulation dimensions, strategies, effects, drawbacks, and expectations—into a series of concise and technically sound research objective sentences:
Each query should be framed as a declarative research objective describing:
•	the material system,
•	the proposed strategy (can come from the current node or be traced from expectation to knowledge base),
•	the expected performance enhancement (e.g., Jc improvement, isotropic pinning),
•	and the underlying reasoning (e.g., defect landscape, lattice strain, orientation mismatch, etc.).
<PolicyTree>
{plan}
</PolicyTree>
🎯 Input Format:
The input is a strategy tree, where each node contains:
    •    "topic": includes REBCO composition, doped material, method, and sample type
    •    "tuning_strategies": each includes tuning dimension, tuning strategy, tuning effect
    •    "draw_backs": limitations of the current method
    •    "expections": target performance improvements or desired mechanism
    •    "children": nested subnodes with the same structure

🔄 Your Task:
For each node (starting from the root and moving recursively to children):
    1.    Extract the most important expectation(s) and map them to suitable known tuning strategies from the node or from other papers in the tree.
    2.    Formulate a query sentence that:
    •    specifies the material system,
    •    proposes the matched or novel strategy,
    •    describes the target improvement,
    •    and justifies the strategy based on microstructural or physical mechanisms.
    3.    If a child node satisfies an expectation of its parent node, generate the child node query accordingly, tracing its origin expectation and linked strategy.
    
Processing Steps:
1.Read through the entire experimental idea, focusing on the motivation, novelty, and required knowledge.
2.Convert the idea into a concise declarative sentence that specifies the material system, methods used, desired effect, and the reason for achieving this effect.

✏️ Output Requirements:
    •    List all queries in numerical order under the <queries> tag.
    •    Each query must be between 25–50 words.
    •    Focus on clarity, technical accuracy, and completeness.
    •    You may use child nodes to refine or satisfy parent expectations.
    •    Avoid vague terms like “better performance”—instead specify “improved Jc at 77K under 5T” or “enhanced isotropic pinning”.

Output Format:
<queries>
A method to enhance the in-field critical current density of YBCO-coated conductors by incorporating BaZrO₃ nanoparticles via PLD. The lattice mismatch between YBCO and BaZrO₃ introduces columnar dislocations and nanoparticles, improving flux pinning. This scalable approach achieves up to 5-fold J_c enhancement at 7 T.
</queries>
            """
        conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "knowlege_query_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        query = self.generate_response(prompt, extract_key="queries",**conversation_kwargs)

        # 根据计划生成相关概念
        return query


class KnowledgeBaseAgent(BaseAgent):
    """知识库智能体 - 检索相关知识"""

    def process(self, input_data: Dict) -> Dict:
        plan = input_data['plan']
        templates = input_data['templates']
        # 检索知识库
        prompt = f"""
You are an expert in the field of high - temperature superconductivity and flux pinning. Your task is to find the knowledge used in the experimental plan from the given structured template library. Please carefully read the following content and operate as required.
Experimental idea:
<idea_plan>
{plan}
</idea_plan>

Retrieved structured template library:
<kb_knowledge>
{templates}
</kb_knowledge>

When analyzing, please follow these steps:
1.Carefully study the entire idea, paying attention to the motivation, novelty, and required knowledge.
2.The structured template library is a reference for the experimental process and lists the factors to be considered during the experiment. In combination with the experimental plan, focus on the information in the structured template library that is helpful for achieving the research objective of the idea.
3.For the knowledge and factors that need to be considered in the idea, generate a guiding knowledge chain based on the template content.

Please generate the knowledge chain within the <knowledge> tag.
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

        return knowledge


class IdeaAgent(BaseAgent):
    """专家智能体 - 制定详细实验计划"""

    def process(self, input_data: Dict) -> Dict:
        topic = input_data['topic']
        plan = input_data['plan']
        templates = input_data['templates']
        history = input_data['history']
        feedback = input_data['feedback']

        prompt = f"""
#####Task description
You are an expert in the field of high-temperature superconductivity and flux pinning. Your task is to generate a complete experimental idea using the topic, idea tree and the available structured template library. Every piece of content must be traceable to specific nodes in the strategy tree with citations and clear reasoning explanations.
The topic involves the REBCO, dopped materials and outcomes. You should follow the topic strictly.
The idea tree includes the goals for achieving the topic and the ideas for solving experimental problems.
The structured template library is content that can be referred to when generating ideas.
Meanwhile, the previous experimental plan and feedback will also be provided for your reference (defaulting to None).

#####Your task
<topic> {topic} </topic>

<idea tree> {plan} </idea tree>

Now, please carefully read the following retrieved structured template library. Each template has a unique knowledge ID — you must cite these IDs when they inform your idea:
<Related Knowledge>
{templates}
</Related Knowledge>

The following are the optimization suggestions, defaulting to None:
<Optimization Suggestions>
{feedback}
</Optimization Suggestions>

The following is the previous experimental idea, defaulting to None:
<Previous Experimental Idea>
{history}
(The format is: Sub-experiment 1: ..., Sub-experiment 2: ..., ...)
</Previous Experimental Idea>

#####Contents of idea tree
1.the topic, which shows the motivation of the research.
2.the first-stage papers. The first-stage papers are retrieved according to the motivation presented by the topic. The methods in the first-stage papers may be able to meet the goals of the topic.
3.the second-stage papers. The second-stage papers are retrieved according to the drawbacks try to solve in first stage papers. The methods in the second-stage papers may be able to solve the deficiencies in the first-stage papers.
4.the third-stage papers. The third-stage papers are retrieved according to the drawbacks try to solve in second stage papers. The methods in the third-stage papers may be able to solve the deficiencies in the second-stage papers.
5.Each paper introduces the strategy and the possible effects that the strategy may achieve. The strategy is developed from five potential strategy dimensions: Composition tuning/ Grain boundary tuning/ Interface tuning/ Texture tuning/ Stress tuning.

#####Output content
The constructed idea needs to comprehensively draw on the strategies of the five dimensions for innovation to meet the motivation of the topic as much as possible.
The final output content needs to be described separately from the five dimensions.
The description of the idea does not require the provision of specific parameters; only general strategies need to be presented. It is not acceptable to simply make permutations and combinations from the tuning strategies in the idea tree. Instead, both innovation and scientificity should be fully considered.
Comprehensively apply the strategies of each dimension to generate five different ideas (especially the composition tuning strategy should be different from each idea), and illustrate the core innovation points in each idea.

The output must be a JSON object with complete explanatory traceability.
For each content field (material, motivation, strategies, technique, outcome, node_mapping), you must provide:
1. Content field: The actual experimental content
2. Source field: Contains TWO parts — "strategy_tree" (node ID mapping with quotes and reasoning) AND "template" (template library knowledge ID mapping showing which knowledge items informed this part)
Here are some of the total key-value pair：
- First key-value pair: Material system including REBCO and doped materials             - Example: "material": "YBCO, doped material: BHO"                          
- Third key-value pair: Research purpose             - Example: "motivation": "The primary goal of this experiment is to maximize flux pinning in multi-element-doped REBCO films through mixed phase additions, enhancing critical current density (Jc) across broad temperature and magnetic field ranges."                          
- Fifth key-value pair: Multiple tuning strategies and effects             - Key "strategies" should contain an array[] of 4-6 JSON objects             - Each object contains:                 • tuning dimensions: Select one from (Composition tuning, Grain boundary tuning, Interface tuning, Texture tuning, Stress tuning)                 - Example: "tuning dimensions": "Composition tuning"                 • tuning strategies: Summarize specific strategies while explaining deeper impacts                 - Example: "tuning strategies": "Introducing Y₂BaCuO₅ nanoparticles via alternating multilayer deposition to create intrinsic defects"                • strategy_source: Contains "strategy_tree" and "template" sub-objects
- sixth key-value pair: Key experimental techniques             - Example: "technique": "The fabrication of BHO-doped EuBCO CCs utilizes hot-wall pulsed laser deposition (PLD), an innovative technique for high-rate deposition of thick superconducting films. Sample characterization employs transmission electron microscopy (TEM) to observe nanorod and nanoparticle structures. In-field critical current (Ic) and Jc measurements are conducted using a four-probe transport method with high-field superconducting magnets, providing precise data on flux-pinning properties. The use of photolithography and wet etching to form microbridges is a standard but critical technique for accurate Ic measurements."                         
- eighth key-value pair: Expected outcomes with numerical values             - Example: "outcome": "For high-field magnets use (e.g., NMR, cryogen-free superconducting magnets). Critical current density (Jc) 1.5-3 times higher than Pure GdBCO at 30 K or less and ≥ 3 T; Pinning force density (Fp) > 1.0 TN/m3 (FAST) and > 1.4 TN/m3 (SLOW) at 4.2 K; Critical current (Ic) up to 464 A for FAST sample at 77.3 K, self-field."   



#####Source field explanation format requirements:
Each source field contains two parts:
1. "strategy_tree": Node ID mapping with quotes and reasoning
   - "Node-ID": "Based on this node's research findings/analysis/data regarding [specific information], we select/adopt/infer [specific decision] because [scientific principle/logical reasoning]"
2. "template": Template library knowledge ID mapping
   - "knowledge_id": "How this template knowledge guided this specific aspect of the idea"
Example explanatory formats for strategy_tree:
- "Based on this node's research findings about ..., we choose ... because ..."
- "This node reveals ..., therefore we infer/determine ..."
- "According to this node's analysis of ..., we judge that this solution can ..."
- "The data provided by this node indicates that we should ..."
- Must quote the original text from the node and enclose the quoted text in quotation marks
Requirements:
1. Mandatory explanatory traceability: Every source field must contain BOTH strategy_tree node IDs AND template knowledge IDs with detailed reasoning
2. Scientific logic chain: Each explanation must show clear "information → reasoning → decision" logic
3. Complete coverage: All major content must be traceable to specific strategy tree nodes and template knowledge items
4. Use English for all content
5. Output in valid JSON format
6. Wrap entire JSON object in <final_idea></final_idea> tags
7. Use scientific notation (e.g., 3.9e6) instead of multiplication symbols

#####Final output format
<final_idea>
{{
"material": "Material description derived from strategy tree and template knowledge",
"material_source": {{
    "strategy_tree": {{
        "1.1": "Based on this node's research findings regarding YBCO substrate optimization \"xxxxxx (quoted original text)\", we select this material system because it provides excellent current-carrying capacity foundation",
        "2.3.1": "This node reveals the critical threshold for BHO doping concentration \"xxxxxx (quoted original text)\", therefore we adopt this doping scheme to achieve optimal pinning effects"
    }},
    "template": {{
        "52": "Template knowledge 52 on YBCO composition informed the material selection by providing optimal doping concentration ranges",
        "27": "Template knowledge 27 guided the choice of dopant materials based on lattice mismatch criteria"
    }}
}},

"motivation": "Research purpose derived from strategy tree",
"motivation_source": {{
    "strategy_tree": {{
        "1.1": "Based on this node's analysis of high-field application requirements \"xxxxxx (quoted original text)\", we determine this research direction because current technology limitations need to be addressed",
        "1.2": "This node identifies existing performance bottlenecks \"xxxxxx (quoted original text)\", thus we focus on low-temperature high-field enhancement"
    }},
    "template": {{
        "15": "Template knowledge 15 on flux pinning enhancement targets confirmed the research motivation and performance goals"
    }}
}},

"strategies": [
    {{
        "tuning_dimensions": "Composition tuning",
        "tuning_strategies": "Strategy description derived from strategy tree",
        "strategy_source": {{
            "strategy_tree": {{
                "1.3.2": "Based on this node's in-depth analysis of Y₂BaCuO₅ nanoparticle pinning mechanisms \"xxxxxx (quoted original text)\", we infer that controlling particle size and distribution can effectively enhance flux pinning strength",
                "2.1.1": "According to the particle size optimization data provided in this node \"xxxxxx (quoted original text)\", we should adopt specific size ranges to maximize pinning efficiency"
            }},
            "template": {{
                "33": "Template knowledge 33 on nanoparticle deposition procedures guided the specific composition tuning approach"
            }}
        }}
    }},
    {{
        "tuning_dimensions": "Interface tuning",
        "tuning_strategies": "Strategy description derived from strategy tree",
        "strategy_source": {{
            "strategy_tree": {{
                "2.2.3": "According to this node's interface engineering analysis \"xxxxxx (quoted original text)\", we judge this approach can precisely control interfacial pinning effects"
            }},
            "template": {{
                "41": "Template knowledge 41 on interface characterization methods informed the interface tuning strategy"
            }}
        }}
    }}
],

"technique": "Technique description derived from strategy tree",
"technique_source": {{
    "strategy_tree": {{
        "2.2.1": "Based on this node's PLD process optimization scheme \"xxxxxx (quoted original text)\", we determine this method can precisely control film microstructure",
        "3.1.3": "This node emphasizes key TEM characterization indicators guiding us to focus on interface structure and defect distribution \"xxxxxx (quoted original text)\""
    }},
    "template": {{
        "19": "Template knowledge 19 on PLD deposition parameters guided the technique selection and optimization approach"
    }}
}},

"outcome": "Expected outcome derived from strategy tree",
"outcome_source": {{
    "strategy_tree": {{
        "1.4.1": "According to this node's reported performance enhancement data \"xxxxxx (quoted original text)\", we predict the feasibility of achieving targeted improvements",
        "2.3.2": "This node's theoretical predictions provide the improvement range basis for our expected outcomes \"xxxxxx (quoted original text)\""
    }},
    "template": {{
        "8": "Template knowledge 8 on Jc measurement standards informed the expected outcome benchmarks"
    }}
}},

"node_mapping": {{
    "total_nodes_used": 8,
    "primary_source_stages": "First-stage: 3 nodes, Second-stage: 3 nodes, Third-stage: 2 nodes",
    "coverage_summary": "Node selection and combination explanation"
}}
}}
</final_idea>

            """
        # 制定详细计划
        conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "idea_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        detailed_plan = self.generate_response(prompt, extract_key="final_idea",load_json=True,**conversation_kwargs)

        return detailed_plan



class ExpertAgent(BaseAgent):
    """专家智能体 - 制定详细实验计划"""

    def process(self, input_data: Dict) -> Dict:
        topic = input_data['topic']
        plan = input_data['plan']
        templates = input_data['templates']
        # knowledge = input_data['knowledge']
        history = input_data['history']
        feedback = input_data['feedback']

        prompt = f"""
            #####Task description
            You are an expert in the field of high-temperature superconductivity and flux pinning. Your task is to generate a complete experimental idea using the topic, idea tree and the available structured template library.
            The topic involves the REBCO, dopped materials and outcomes. You should follow the topic strictly.
            The idea tree includes the goals for achieving the topic and the ideas for solving experimental problems.
            The structured template library is content that can be referred to when generating ideas.
            Meanwhile, the previous experimental plan and feedback will also be provided for your reference (defaulting to None).
            
            #####Your task
            <topic>
            {topic}
            </topic>
            
            <idea tree>
            {plan}
            </idea tree>
            
            Now, please carefully read the following retrieved structured template library:
            <Related Knowledge>
            {templates}
            </Related Knowledge>
            
            The following are the optimization suggestions, defaulting to None:
            <Optimization Suggestions>
            {feedback}
            </Optimization Suggestions>
            
            The following is the previous experimental idea, defaulting to None:
            <Previous Experimental Idea>
            {history}
            (The format is: Sub-experiment 1: ..., Sub-experiment 2: ..., ...)
            </Previous Experimental Idea>
            
            #####Contents of idea tree
            1.the topic, which shows the motivation of the research.
            2.the first-stage papers. The first-stage papers are retrieved according to the motivation presented by the topic. The methods in the first-stage papers may be able to meet the goals of the topic.
            3.the second-stage papers. The second-stage papers are retrieved according to the drawbacks try to solve in first stage papers. The methods in the second-stage papers may be able to solve the deficiencies in the first-stage papers.
            4.the third-stage papers. The third-stage papers are retrieved according to the drawbacks try to solve in second stage papers. The methods in the third-stage papers may be able to solve the deficiencies in the second-stage papers.
            5.Each paper introduces the strategy and the possible effects that the strategy may achieve. The strategy is developed from five potential strategy dimensions: Composition tuning/ Grain boundary tuning/ Interface tuning/ Texture tuning/ Stress tuning.
            
            #####Output content
            1. The output must be a JSON object containing 5 key-value pairs
            2. First key-value pair: Material system including REBCO and doped materials
            - Example: "material": "YBCO, doped material: BHO"
            
            3. Second key-value pair: Research purpose
            - Example: "motivation": "The primary goal of this experiment is to maximize flux pinning in multi-element-doped REBCO films through mixed phase additions, enhancing critical current density (Jc) across broad temperature and magnetic field ranges."
            
            4. Third key-value pair: Multiple tuning strategies and effects
            - Key "strategies" should contain an array[] of 4-6 JSON objects
            - Each object contains two key-value pairs:
                • tuning dimensions: Select one from (Composition tuning, Grain boundary tuning, Interface tuning, Texture tuning, Stress tuning)
                - Example: "tuning dimensions": "Composition tuning"
                • tuning strategies: Summarize specific strategies while explaining deeper impacts
                - Example: "tuning strategies": "Introducing Y₂BaCuO₅ nanoparticles via alternating multilayer deposition to create intrinsic defects"
            
            5. Fourth key-value pair: Key experimental techniques
            - Example: "technique": "The fabrication of BHO-doped EuBCO CCs utilizes hot-wall pulsed laser deposition (PLD), an innovative technique for high-rate deposition of thick superconducting films. Sample characterization employs transmission electron microscopy (TEM) to observe nanorod and nanoparticle structures. In-field critical current (Ic) and Jc measurements are conducted using a four-probe transport method with high-field superconducting magnets, providing precise data on flux-pinning properties. The use of photolithography and wet etching to form microbridges is a standard but critical technique for accurate Ic measurements."
            
            6. Fifth key-value pair: Expected outcomes with numerical values
            - Example: "outcome": "For high-field magnets use (e.g., NMR, cryogen-free superconducting magnets). Critical current density (Jc) 1.5-3 times higher than Pure GdBCO at 30 K or less and ≥ 3 T; Pinning force density (Fp) > 1.0 TN/m3 (FAST) and > 1.4 TN/m3 (SLOW) at 4.2 K; Critical current (Ic) up to 464 A for FAST sample at 77.3 K, self-field."
            
            ##### Requirements:
            1. All content must be directly sourced from literature evidence
            2. Use English for all content
            3. Output in valid JSON format
            4. Wrap entire JSON object in <final_idea></final_idea> tags
            5. Separate objects in "strategies" array with commas
            6. Use scientific notation (e.g., 3.9e6) instead of multiplication symbols


            #####Final output format 
            The maximum heading level is five, using "##### "
            <final_idea>
            {{
                "material": "",
                "motivation": "",
                "strategies": [
                    {{
                        "tuning dimensions": "",
                        "tuning strategies": ""
                    }},
                    {{
                        "tuning dimensions": "",
                        "tuning strategies": ""
                    }},
                    {{
                        "tuning dimensions": "",
                        "tuning strategies": ""
                    }}
                ],
                "technique": "",
                "outcome": ""
            }}
            </final_idea>
            """
        # 制定详细计划
        conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "idea_expert_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }
        detailed_plan = self.generate_response(prompt, extract_key="final_idea",load_json=True,**conversation_kwargs)

        return detailed_plan
       

class ReviewAgent(BaseAgent):
    """审核智能体 - 检查实验计划"""

    def process(self, input_data: Dict):
        detailed_plan = input_data['detailed_plan']
        knowledge = input_data['knowledge']
        prompt = f"""
Your task is to check the reasonableness of an experimental idea based on the given external knowledge.
First, please carefully read the following experimental plan:
<Experimental Idea>
{detailed_plan}
</Experimental Idea>

Then, please carefully read the following external knowledge:
<External Knowledge>
{knowledge}
</External Knowledge>

When checking, please follow these methods:
1.Compare and analyze each step of the experimental plan with the external knowledge.
2.Consider whether the steps in the experimental plan violate the principles, rules, etc. in the external knowledge.
3.Comprehensively consider the overall logic and feasibility of the experimental plan.

Give the judgment result in the <result> tag. Return "True" if you think the experimental plan is reasonable, and "False" if it is not. If the judgment is "False", clearly point out the unreasonable parts and the judgment basis based on the external knowledge in the <reason> tag.
<result>
{{
    "judge": "True",
    "reason": "The experimental plan is reasonable."
}}
</result>

        """
        conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "idea_critic_agent",
            "meta": {
                "input_data": input_data,
                "prompt": prompt,
            }
        }

        review_result = self.generate_response(prompt, extract_key="result",load_json=True,**conversation_kwargs)

        judge = review_result["judge"]
        reason = review_result["reason"]        
        # 审核计划
        return {"review_result": judge, "feedback": reason}


