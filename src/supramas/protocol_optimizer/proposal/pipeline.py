import json
import sys
import os
import sys
from typing import Dict, List
from supramas.utils import translate_api
from loguru import logger


class ProposalPipeline:
    def __init__(self, config, retrival, type, exp_plan = None,topic=None,llm=None):  # exp参数设为可选，默认为None
        self.config = config
        self.exp_plan = exp_plan
        self.type = type
        self.topic = topic
        self.llm = llm
        if type == "experiment":
            from .agents.experiment_agents import (DraftGenerationAgent,
                                            ExperimentDecompositionAgent,
                                            ExperimentPlan, ExpertAgent,
                                            KnowledgeBaseAgent,
                                            PlanWritingAgent, RetrievalAgent,
                                            ReviewAgent)
            self.expert_agent = ExpertAgent(llm=self.llm ,**self.config['agents']['expert'])
            self.retrieval_agent = RetrievalAgent(llm=self.llm,**self.config['agents']['retrieval'])
            self.kb_agent = KnowledgeBaseAgent(llm=self.llm,**self.config['agents']['knowledge_base'])
            self.draft_agent = DraftGenerationAgent(llm=self.llm,**self.config['agents']['draft_generation'])
            self.decomp_agent = ExperimentDecompositionAgent(llm=self.llm,**self.config['agents']['experiment_decomposition'])
            self.review_agent = ReviewAgent(llm=self.llm,**self.config['agents']['review'])
            self.plan_writing = PlanWritingAgent(llm=self.llm,**self.config['agents']['plan_writing'])

        elif type == "idea":
            from .agents.idea_agents import (
                                                ExpertAgent,
                                                KnowledgeBaseAgent,
                                                RetrievalAgent,
                                                ReviewAgent,
                                                IdeaAgent
                                            )
            self.idea_agent = IdeaAgent(llm=self.llm,**self.config['agents']['expert'])
            self.expert_agent = ExpertAgent(llm=self.llm,**self.config['agents']['expert'])
            self.retrieval_agent = RetrievalAgent(llm=self.llm,**self.config['agents']['retrieval'])
            self.kb_agent = KnowledgeBaseAgent(llm=self.llm,**self.config['agents']['knowledge_base'])
            self.review_agent = ReviewAgent(llm=self.llm,**self.config['agents']['review'])

        self.retrival = retrival

    def sub_task_generation(self, idx, plan, feedback=None):

        queries = self.retrieval_agent.process({"plan": plan})
        query = queries["query1"]
        # 翻译
        query_tran = translate_api(query)

        logger.info(f"query:\n\n{query}")

        # 检索milvs向量库
        
        knowledge_templates = self.retrival([query_tran])

        logger.info(f"templates模板:\n\n{knowledge_templates}")

        # 生成有用知识
        knowledge = self.kb_agent.process({"plan": plan, "templates": knowledge_templates})

        # 制定详细计划
        detailed_plan = self.expert_agent.process(
            {
                "topic": self.topic,
                "plan": plan,
                "templates": knowledge_templates,
                "feedback": feedback,
                "history": self.exp_plan.get_steps(idx) if idx > 1 else None,
            }
        )

        logger.info(f"详细计划: {detailed_plan}\n参考知识库:{knowledge}")

        # 审核循环
        is_pass = False
        t = 0

        while not is_pass and t < 3:
            t += 1
            # 获取相关概念
            queries = self.retrieval_agent.process({"plan": plan})
            query = queries["query1"]
            # 翻译
            query_tran = translate_api(query)

            logger.info(f"query:\n\n{query}")
            # 检索milvs向量库
            knowledge_templates = self.retrival([query_tran])

            knowledge = self.kb_agent.process({"plan": detailed_plan, "templates": knowledge_templates})

            review_result = self.review_agent.process(
                {
                    "detailed_plan": detailed_plan,
                    "knowledge": knowledge,
                }
            )

            is_pass = True if review_result['review_result'] == "True" else False
            if is_pass:
                continue

            logger.info(f"审核不通过\n\n")
            # 如果审核不通过，更新详细计划
            feedback = review_result["feedback"]
            detailed_plan = self.expert_agent.process(
                {
                    "topic": self.topic,
                    "plan": detailed_plan,
                    "templates": knowledge,
                    "feedback": feedback,
                    "history": self.exp_plan.get_steps(idx) if idx > 1 else None,
                }
            )

            logger.info(f"详细计划expert:\n\n{detailed_plan}")

        return detailed_plan, knowledge


    def idea_generate(self, policy_tree, topic=None):

        query = self.retrieval_agent.process({"plan": policy_tree})
        # 翻译
        query_tran = translate_api(query)
        # 检索milvs向量库
        knowledge_templates = self.retrival([query_tran])

        # 生成有用知识
        knowledge = self.kb_agent.process({"plan": policy_tree, "templates": knowledge_templates})

        # 制定详细计划
        initial_idea = self.idea_agent.process(
            {
                "topic": self.topic,
                "plan": policy_tree,
                "templates": knowledge_templates,
                "feedback": None,
                "history": [],
            }
        )
        logger.info(f"initial_idea 生成完成!")
        # 审核循环
        is_pass = False
        t = 0
        refine_idea=initial_idea

        while not is_pass and t < 2:
            t += 1
            logger.info(f"开始第{t}次审核!")
            # 获取相关概念
            query = self.retrieval_agent.process({"plan":initial_idea})
            # 翻译
            query_tran = translate_api(query)

            # 检索milvs向量库
            knowledge_templates = self.retrival([query_tran])

            knowledge = self.kb_agent.process({"plan": initial_idea, "templates": knowledge_templates})

            review_result = self.review_agent.process(
                {
                    "detailed_plan": initial_idea,
                    "knowledge": knowledge,
                }
            )

            logger.info(f"审核结果: {review_result['review_result']}\n理由: {review_result['feedback']}")

            is_pass = True if review_result['review_result'] == "True" else False
            if is_pass:
                continue

            logger.info(f"审核不通过\n")
            # 如果审核不通过，更新详细计划
            feedback = review_result["feedback"]
            refine_idea = self.expert_agent.process(
                {
                    "topic": topic,
                    "plan": initial_idea,
                    #####################################################################
                    "templates": knowledge_templates,
                    # "knowledge": knowledge,
                    "feedback": feedback,
                    #####################################################################
                    "history": [],
                }
            )

        return initial_idea,refine_idea


    def experiment_generate(self, idea):
        """生成实验方案"""

        # 步骤1：生成实验草案
        draft = self.draft_agent.process({"idea": idea})
        self.exp_plan.draft = draft["draft"]  # 存储到exp_plan而不是exp

        # 步骤2：分解实验步骤
        steps = self.decomp_agent.process({"draft": draft["draft"]})

        # 步骤3：详细化每个子实验 - 详细内容
        for idx,step in enumerate(steps["steps"].values()):

            description = step
            
            # 添加子实验的描述
            self.exp_plan.add_step(idx, description=description)

            # 迭代每一步详细实验计划
            detailed_plan, knowledge = self.sub_task_generation(idx, plan=description)

            self.exp_plan.update_step(idx, detailed_plan=detailed_plan,knowledge=knowledge)
            logger.info(f"详细计划: {detailed_plan}")


        # 步骤4：生成最终实验计划
        final_plan = self.plan_writing.process({
            "draft": draft["draft"],
            "plans": self.exp_plan.steps,
        })

        self.exp_plan.final_plan = final_plan["final_plan"]  # 存储到exp_plan

        logger.info(f"##### 最终实验计划:\n {final_plan['final_plan']}")

        # logger.warning(f"Proposal Professor: {self.exp_plan.final_plan}")
        logger.info("实验方案生成完成")

        return final_plan["final_plan"]
