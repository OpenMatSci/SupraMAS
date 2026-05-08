import json
import os
import sys
import time
from typing import List

import asyncio

import scipdf
from dotenv import load_dotenv
from loguru import logger

from supramas.protocol_optimizer.pipeline import AGDPipline
from supramas.llm import openai_llm
from supramas.strategy_tree.pipeline import PolicyTreePipline
from supramas.prompts import *
from supramas.strategy_tree.searcher import Result, SementicSearcher
from supramas.utils import *
from termcolor import cprint


def get_llms(conversation_logger):
    load_dotenv()
    if "MAIN_LLM_MODEL" not in os.environ or os.environ["MAIN_LLM_MODEL"] == "":
        raise ValueError("MAIN_LLM_MODEL is not set")
    model = os.environ.get("MAIN_LLM_MODEL", "gpt4o-0513")
    main_llm = openai_llm(model,conversation_logger)
    return main_llm


class NeedDownloadException(Exception):
    def __init__(self, title, normalized_title, tree_depth, expectations, parent_title):
        self.title = title
        self.normalized_title = normalized_title
        self.tree_depth = tree_depth
        self.expectations = expectations
        self.parent_title = parent_title
        super().__init__(f"Need to download: {title}")

class DeepResearchAgent:
    def __init__(
        self,
        topic,
        topic_path="saves/",
        llm=None,
        **kwargs,
    ) -> None:
        # 统一使用topic_path作为基础路径
        self.topic_path = topic_path
        self.begin_time = time.time()
        self.llm = llm
        self.search_qeuries = []
        self.reader = SementicSearcher(topic_path, header_dir=kwargs.get("header_dir"))
        self.config = kwargs.get("config")
        self.topic = topic
        

    def wrap_messages(self, prompt):
        return [{"role": "user", "content": prompt}]


    def idea_innovation(self, policy_tree):

        self.agd = AGDPipline(
            type="idea",  # 默认类型，可以在使用时修改
            config=self.config,
            topic=self.topic,
            topic_path=self.topic_path,
            llm=self.llm
        )

        initial_idea, final_idea = self.agd.forward(idea=[], type="idea", policy_tree=policy_tree)

        return initial_idea, final_idea


    def experiment_construction(self, idea):

        self.agd = AGDPipline(
            type="experiment",  # 默认类型，可以在使用时修改
            config=self.config,
            topic=self.topic,
            topic_path=self.topic_path,
            llm=self.llm
        )

        final_exp = self.agd.forward(idea=idea, type="experiment")

        return final_exp


    def regenerate_idea(self,  idea, human_feedback):

        ###########################################################################
        logger.warning("重新生成idea!")
        ###########################################################################

        final_idea = None
        while not final_idea:
            input_data = {
                "idea": idea,
                "human_feedback": human_feedback,
            }
            prompt = get_deep_regenerate_idea_prompt(input_data)
            messages = self.wrap_messages(prompt)
            
            conversation_kwargs = {
            "stage": "idea_innovation",
            "role": "knowledge_agent",
            "meta": {
                "input_data": input_data,
                "prompt": messages,
            }
        }
            final_idea = self.llm.response(messages, extract_key="final_idea", **conversation_kwargs)

        return final_idea
    
    def regenerate_experiment(self,  experiment, human_feedback):

        ###########################################################################
        logger.warning("重新生成实验!")
        ###########################################################################
        input_data = {
            "experiment": experiment,
            "human_feedback": human_feedback,
        }
        prompt = get_deep_refine_experiment_prompt(input_data)
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "experiment_construction",
            "role": "human_interaction_agent",
            "meta": {
                "input_data": input_data,
                "prompt": messages,
            }
        }
        refined_experiment = None
        while not refined_experiment:
            
            refined_experiment = self.llm.response(messages, extract_key="experiment",load_json=True,**conversation_kwargs)

        return refined_experiment


    def human_intervention(self,plan,type):

        human_feedback = input("请您提供对plan的改进建议,如果有改进建议,请您提供改进建议,如果没有,请您直接输入回车: ")
        refined_plan = plan
        while human_feedback != "":
            if type == "idea":   
                refined_plan = self.regenerate_idea(idea=plan, human_feedback=human_feedback)
                cprint(f"final_idea: {refined_plan}", "blue")
                human_feedback = input("请您提供对idea的改进建议,如果有改进建议,请您提供改进建议,如果没有,请您直接输入回车")

            else:
                refined_plan = self.regenerate_experiment(experiment=plan, human_feedback=human_feedback)
                cprint(f"final_experiment: {refined_plan}", "blue")
                human_feedback = input("请您提供对idea的改进建议,如果有改进建议,请您提供改进建议,如果没有,请您直接输入回车")
                
        return refined_plan
        

    def human_intervention_auto(self,plan,human_feedback,type):

        refined_plan = plan

        if type == "idea":   
            refined_plan = self.regenerate_idea(idea=plan, human_feedback=human_feedback)
            reasons = None
            cprint(f"final_idea: {refined_plan}", "blue")

        else:
            refined_plan_with_feedback = self.regenerate_experiment(experiment=plan, human_feedback=human_feedback)
            refined_plan = refined_plan_with_feedback["modified_scheme"]
            reasons = refined_plan_with_feedback["reasons"]
            
        return refined_plan, reasons