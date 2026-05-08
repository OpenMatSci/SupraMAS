import os
import sys
import sys


# 获取当前文件的父目录（即 mymodule 的父目录，也就是项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 添加到 sys.path 中

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# from dotenv import load_dotenv
import yaml
from dotenv import load_dotenv
from loguru import logger

from supramas.protocol_optimizer.evaluator import EvaluatorPipeline as EvaluatorPipeline
from supramas.protocol_optimizer.proposal import ProposalPipeline as ProposalPipeline

from supramas.protocol_optimizer.proposal.agents.experiment_agents import (
    ExperimentPlan, PlanWritingAgent)

from supramas.utils import *
from supramas.protocol_optimizer.vector_retrieval import VectorRetrieval

load_dotenv()

# 删除EXP类，不再需要

class AGDPipline:
    def __init__(self, type, config, topic_path=None, topic=None, llm=None):
        self.exp_plan = ExperimentPlan()
        self.retrieval = VectorRetrieval()
        # 删除self.exp，不再使用EXP类
        self.proposal = ProposalPipeline(
            config=config, retrival=self.retrieval, type=type,exp_plan=self.exp_plan, topic=topic, llm=llm
        )
        self.evaluator = EvaluatorPipeline(retrival=self.retrieval, exp_plan = self.exp_plan ,llm=llm)
        self.topic_path = topic_path

        self.plan_writing = PlanWritingAgent(**config['agents']['plan_writing'])
        self.topic = topic
        self.llm = llm
        
    # def save_experiment_markdown(self, expsummary, save_path: str) -> str:
    #     """
    #     将指定用户的实验方案导出为 Markdown 并保存为 .md 文件。

    #     参数:
    #         save_path: 保存文件的完整路径（包括文件名，如 /tmp/plan.md）

    #     返回:
    #         保存成功后的文件路径
    #     """

    #     def parse_and_convert_to_markdown(data):
    #         markdown = ""
    #         if isinstance(data, str):
    #             try:
    #                 data_list = json.loads(data)
    #             except Exception as e:
    #                 print("[Error] final_idea JSON 解析失败:", e)
    #                 return ""
    #         elif isinstance(data, list):
    #             data_list = data
    #         else:
    #             print("[Error] 不支持的类型:", type(data))
    #             return ""

    #         for idx, json_str in enumerate(data_list, 1):
    #             try:
    #                 if idx==1:
    #                     idea_list = json.loads(json_str.strip(), strict=False)
    #                     for idea in idea_list:
    #                         markdown += f"##### Idea {idx}\n"
    #                         markdown += f"- **Material**: {idea.get('material', '')}\n"
    #                         markdown += f"- **Motivation**: {idea.get('motivation', '')}\n"
    #                         markdown += f"- **Strategies**:\n"
    #                         for strategy in idea.get("strategies", []):
    #                             dim = strategy.get("tuning dimensions", "")
    #                             strat = strategy.get("tuning strategies", "")
    #                             markdown += f"  - **tuning dimensions**: {dim}\n"
    #                             markdown += f"  - **tuning strategies**: {strat}\n"
    #                         markdown += f"- **Technique**: {idea.get('technique', '')}\n"
    #                         markdown += f"- **Outcome**: {idea.get('outcome', '')}\n\n"
    #             except Exception as e:
    #                 print("[Error] idea 内容解析失败:", e)
    #                 continue
    #         return markdown

    #     # 获取数据
    #     exp_summary = expsummary

    #     idea = exp_summary.get("idea", "")
    #     draft_experiment = exp_summary.get("draft_experiment", "")
    #     decomposed_steps = exp_summary.get("decomposed_steps", "")
    #     detailed_sub_experiments = exp_summary.get("detailed_sub_experiments", {})
    #     final_plan = exp_summary.get("improved_experiment", "")

    #     idea_markdown = parse_and_convert_to_markdown(idea)

    #     # 组装内容
    #     lines = [
    #         f"##### Final Idea:\n\n{idea_markdown}",
    #         f"\n\n{draft_experiment}",
    #         f"\n\n{decomposed_steps}",
    #     ]
    #     for title, detail in detailed_sub_experiments.items():
    #         lines.append(f"\n\n{title}\n\n{detail}")
    #     lines.append(f"\n\n{final_plan}")

    #     markdown_text = "\n\n".join(lines)

    #     with open(save_path, "w", encoding="utf-8") as f:
    #         f.write(markdown_text)

    #     print(f"[Export] 成功保存至: {save_path}")
    #     return save_path

    # def parse_and_convert_to_markdown(self, data_list):
    #     markdown = ""
    #     for idx, json_str in enumerate(data_list,1):
    #         try:
    #             if idx==1:
    #                 # 去除可能的换行符与空格
    #                 clean_str = json_str.strip()

    #                 # 每个字符串其实是一个数组形式 [ {...} ]
    #                 idea_list = json.loads(clean_str)

    #                 for idea in idea_list:
    #                     markdown += f"##### Idea {idx}\n"
    #                     markdown += f"- **Material**: {idea.get('material', '')}\n"
    #                     markdown += f"- **Motivation**: {idea.get('motivation', '')}\n"
    #                     markdown += f"- **Strategies**:\n"
    #                     for strategy in idea.get("strategies", []):
    #                         dim = strategy.get("tuning dimensions", "")
    #                         strat = strategy.get("tuning strategies", "")
    #                         markdown += f"  - **tuning dimensions**: {dim}\n"
    #                         markdown += f"  - **tuning strategies**: {strat}\n"
    #                     markdown += f"- **Technique**: {idea.get('technique', '')}\n"
    #                     markdown += f"- **Outcome**: {idea.get('outcome', '')}\n\n"

    #         except Exception as e:
    #             print(f"[Error] JSON 解析失败: {e}")
    #             continue

    #     return markdown

    # def process_and_optimize_idea(self, args):
    #     """处理并优化单个 idea"""
    #     idea_str, evaluator, proposal = args
    #     try:
    #         idea = json.loads(idea_str)
    #         feedback = evaluator.forward(idea)
    #         updated_idea = proposal.generate(idea_str, feedback)
    #         return updated_idea
    #     except Exception as e:
    #         return f"[Error processing idea] {str(e)}"


    def forward(self,idea, type, policy_tree=None):

        if type == "experiment":
            # 保存 idea 到实验计划中
            self.exp_plan.idea = idea

            # 返回一个精炼后的实验+详细方案
            final_plan = self.proposal.experiment_generate(idea)
            
            # # evaluator 评估精炼后的实验 + 详细的实验方案； 返回一个反馈列表
            # feedbacks = self.evaluator.forward(final_plan,type=type)  

            # print("feedbacks",feedbacks)
            # logger.info("experiment外部循环\n")

            # for feedback in feedbacks:
            #     idx = int(feedback["idx"])
            #     content = self.exp_plan.steps[idx]
            #     content_feedback = feedback["feedback"]
                
            #     # 改进后的子实验
            #     detailed_plan,knowledage = self.proposal.sub_task_generation(idx,plan = content, feedback=content_feedback)

            #     self.exp_plan.update_step(idx, detailed_plan = detailed_plan)
            #     logger.info(f"改进后的详细计划: {detailed_plan}")

            # self.exp_plan.final_plan = self.plan_writing.process(
            #     {"draft": self.exp_plan.draft, "plans": self.exp_plan.steps}
            # )
            #  保存实验方案 json
            self.exp_plan.save_dict(file_path=os.path.join(self.topic_path, "all_protocol.json"))
            #  保存实验方案 md
            self.exp_plan.save_markdown(file_path=os.path.join(self.topic_path, "all_protocol.md"))

            return self.exp_plan.final_plan
          
           
        elif type == "idea":
            logger.info("成功")

            initial_idea, final_idea = self.proposal.idea_generate(policy_tree, topic=self.topic)

            return initial_idea, final_idea

                




# if __name__ == "__main__":
#     idea = """Title: "Machine Learning-Driven Optimization of Artificial Pinning Centers in REBCO Superconducting Strips for Enhanced Bearing Current Capacity"
# Origins and Motivation:
# The foundational work in "Improvement of Critical Current Properties in REBCO Thin Films by Introduction of Artificial Pinning Centers" (Paper 0) demonstrated that introducing nanorods and nanoparticles as artificial pinning centers can significantly enhance the critical current density (Jc) and upper critical field (Bc2) in REBCO coated conductors. However, this study also highlighted challenges, such as the complexity of controlling the microstructure of pinning centers and the need for precise experimental conditions. These limitations have hindered the practical application of these findings in real-world superconducting systems. To address these challenges, this research proposes a novel approach that integrates advanced computational modeling, machine learning, and in-situ characterization techniques to systematically optimize the microstructure of pinning centers, thereby improving the bearing current capacity of superconducting strips.
# Novelty and Difference from Previous Works:
# The novelty of this research lies in its use of machine learning algorithms to predict and optimize the size, number density, and morphology of artificial pinning centers, combined with in-situ characterization techniques for real-time feedback during fabrication. Unlike previous methods that relied on trial-and-error experimental approaches, this method systematically explores the parameter space to identify the optimal combination of pinning center properties. The integration of machine learning allows for the identification of complex patterns and relationships that may not be evident through traditional analysis, enabling more precise control over the microstructure. Additionally, the use of in-situ characterization techniques ensures that the desired microstructure is achieved with high accuracy, addressing the limitations of previous research.
# Challenges and How They Are Overcome:
# 1. Complexity of Microstructure Control: Controlling the microstructure of pinning centers is inherently complex due to the interplay of multiple parameters (size, number density, morphology). This challenge is overcome by leveraging advanced computational modeling to simulate the effects of these parameters on Jc and Bc2, providing a theoretical foundation for optimization.
# 2. Need for Precise Experimental Conditions: Achieving the desired microstructure requires precise experimental conditions, which can be difficult to maintain. This is addressed by incorporating in-situ characterization techniques, such as real-time X-ray diffraction and scanning electron microscopy, to monitor the formation of pinning centers and provide immediate feedback during fabrication.
# 3. Efficient Exploration of Parameter Space: Exploring the vast parameter space for optimal pinning center properties is time-consuming and resource-intensive. Machine learning algorithms are employed to analyze simulation data and identify the optimal combination of parameters, significantly reducing the time and effort required for optimization.
# Methodology:
# The research employs a three-step approach:
# 1. Computational Modeling: Simulate the effects of pinning center parameters on Jc and Bc2 using the condensation energy pinning interaction mechanism and electron scattering theory.
# 2. Machine Learning Optimization: Train machine learning algorithms on simulation data to predict the optimal combination of pinning center properties.
# 3. In-Situ Characterization: Use real-time X-ray diffraction and scanning electron microscopy during fabrication to monitor and adjust the microstructure of pinning centers.
# By combining these components, this research provides a comprehensive and systematic approach to optimizing pinning centers, ultimately enhancing the bearing current capacity of superconducting strips."""
#     ########### 测试idea部分 ##############
#     # agd = AGDPipline(type="idea")
#     # final_idea = agd.forward(idea,type="idea")
#     # logger.warning(f"final idea生成如下{final_idea}")
#     # input("Press Enter to continue...")
#     ########### 测试实验部分 ##############
    #     idea_path = "examples/quickstart/topic.json"
#     # 打开并读取 JSON 文件
#     with open(idea_path, "r", encoding="utf-8") as f:
#         idea = json.load(f)
#     # idea = json.loads(idea_path)
#     print(idea)
#     print(len(idea))
#     user_coi = coi_utils.get_cois()
#     # agd = AGDPipline(type="experiment")
#     # generate_content = agd.forward(final_idea,type="experiment")
#     # print(generate_content)
