import os
import sys

import yaml

from loguru import logger

from supramas.strategy_tree.agents import *

from supramas.strategy_tree.searcher import SementicSearcher
from supramas.utils import extract
from supramas.strategy_tree.vector_retrieval import VectorRetrieval
from supramas.strategy_tree.searcher.semantic_search import Result
import json


class PolicyTreePipline:
    def __init__(
        self,
        args,
        main_llm,
    ):
        self.topic = args.topic
        self.paper_tree = PaperTree(args.topic)
        self.retrieval = VectorRetrieval()
        self.max_tree_depth = getattr(args, "max_tree_depth", 3)
        self.grobid_url = args.config.get("GROBID_URL", "http://127.0.0.1:8070")
        self.config = args.config
        self.topic_path = args.topic_path
        self.llm = main_llm

        self.reader = SementicSearcher(
            ban_paper=[],
            grobid_url=self.grobid_url,
            google_search_api_key=self.config['GOOGLE_SEARCH_API_KEY'],
            semantic_search_api_key=self.config.get("SEMENTIC_SEARCH_API_KEY") or self.config.get("SEMANTIC_SEARCH_API_KEY"),
            header_dir=getattr(args, 'header_dir', None),
        )

        # 创建代理时传入main_llm
        agent_config = self.config.get('agents', {})
        self.paper_query_agent = Strategy2Query(
            llm=self.llm,
            **agent_config.get('query_agent', {})
        )
        self.root_query_agent = RootQuery(
            llm=self.llm,
            **agent_config.get('root_agent', {})
        )
        self.review_agent = ReviewAgent(
            llm=self.llm,
            **agent_config.get('review', {})
        )
        self.tuning_strategy_agent = ParserAgent(
            llm=self.llm,
            **agent_config.get('parser', {})
        )

        self.avoid_papers = []
        self.label_papers = []

    def build(self, expectations, parent_title=None, tree_depth=0):
        
        tree_depth += 1
        if tree_depth > self.max_tree_depth:
            print(f"正在构建策略树,当前深度: {tree_depth},大于最大深度,跳过该节点构建!")
            return False
        
        logger.info(f"正在构建策略树,当前深度: {tree_depth}")
        queries = []
        if tree_depth == 1:
            root_query = self.root_query_agent.process(expectations)
            queries.append(root_query["root_queries"])

        else:
            strageties = self.retrieval.search_milvus_queries([expectations], collection_name="tuning_strategy")
            strageties = strageties["data"]
            for stragety in strageties:
                paper_query = self.paper_query_agent.process({"tuning_strategies": stragety["tuning_strategies"]})
                queries.append(paper_query["retrive_queries"])

        # 2. 检索文献 - 相似度最高一篇
        current_papers = []
        for query in queries:
            print(f"开始检索文献,query: {query}")
            if tree_depth == 1:
                print("开始搜索根节点")
            else:
                print("开始搜索子节点")

            results = self.reader.search_(
                query, max_results=5, paper_list=None, llm=self.llm, rerank_query=None, publicationDate=None
            )
            current_paper = results[0] if len(results) > 0 else None

            if results and len(results) > 0 and results[0]:
                if current_paper.title in self.avoid_papers:
                    logger.info(f"Paper {current_paper.title} is in avoid_papers, skipping and trying next query...")
                    continue
                self.avoid_papers.append(current_paper.title)
                current_papers.append(current_paper)
                # 命中后不再继续尝试后续 query
                break
            else:
                # 当前 query 未命中，自动尝试下一个 query
                logger.info(f"检索文献失败,query:{query}, tree_depth:{tree_depth}, 自动尝试下一个query")
        if len(current_papers) == 0:
            logger.info(f"所有query均未检索到文献, 自动放弃该节点构建")
            return False

        root_paper = current_papers[0]
        logger.info(f"检索到的文献: {root_paper.title}")
        # 3. 结构化数据
        pdf_url = root_paper.pdf_url
        shortcomings, expectations = self.review_agent.process({"file_path": pdf_url})
        logger.info(f"不足和期望 提取完成!")

        tuning_strategy = self.tuning_strategy_agent.process({"file_path": pdf_url})
        logger.info(f"调控策略 提取完成!")

        # 4. 构建树节点
        paper_node = PaperNode(
            title=root_paper.title,
            article=root_paper.abstract,
            tuning_strategy=tuning_strategy["templates"],
            draw_backs=shortcomings,
            desired_effects=expectations,
            url=root_paper.pdf_url,
        )

        # 5. 构建树
        self.paper_tree.add_paper(paper_node, parent_title=parent_title)
        logger.info(f"构建树节点完成,title: {paper_node.title}")

        for idx,effect in expectations.items():
            is_build = self.build(expectations=effect, parent_title=paper_node.title, tree_depth=tree_depth)
            if is_build:
                print(f"构建树节点完成,title: {paper_node.title}!")
            else:
                print(f"构建树节点失败!")

        return True

    def forward(self):

        logger.info(f"开始构建策略树")
        
        is_build = self.build(expectations=self.topic)
        if not is_build:
            logger.info(f"策略树构建失败")
            return None

        policy_tree = self.paper_tree.export_json(topic=self.topic,saved_path=self.topic_path / "policy_tree.json")
        logger.info(f"策略树构建完成,保存到{self.topic_path / 'policy_tree.json'}")

        return policy_tree


# if __name__ == "__main__":
#     topic = {
#         "REBCO": "(Eu,Er)BCO",
#         "dopped_materials": "BaHfO₃(BHO)",
#         "processing_method": "PLD",
#         "sample_category": "films",
#     }
#     header_dir = "./results"

#     topic_values = "_".join([value for value in topic.values()])

#     topic_path = f"{header_dir}/{topic_values}"
#     os.makedirs(topic_path, exist_ok=True)

#     ptp = PolicyTreePipline(
#         config, topic, topic_path=topic_path
#     )
#     ptp.forward()
