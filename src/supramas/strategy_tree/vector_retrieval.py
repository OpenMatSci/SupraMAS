import json
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv
from pymilvus import (AnnSearchRequest, Collection, CollectionSchema, DataType,
                      FieldSchema, MilvusClient, RRFRanker, WeightedRanker,
                      connections, db, utility)
# from transformers import AutoModel, AutoTokenizer  # 已弃用，改用硅基流动 API

# import torch


class VectorRetrieval:
    def __init__(self):
        self.milvus_client = MilvusClient(
            uri=os.environ["MILVUS_URI"],
            user=os.environ["MILVUS_USER"],
            password=os.environ["MILVUS_PASSWORD"],
            db_name=os.environ["MILVUS_DATABASE"],
        )

    def encode_data(self, text):
        """使用硅基流动 API (Qwen/Qwen3-Embedding-8B) 获取文本编码向量，维度4096"""
        url = os.getenv("EMBEDDING_API_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings")
        headers = {
            "Authorization": f"Bearer {os.getenv('EMBEDDING_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("EMBEDDING_NAME", "Qwen/Qwen3-Embedding-8B"),
            "input": [text]
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        embbeding = result["data"][0]["embedding"]
        return embbeding

    def search_milvus_papers(self, queries, limit=5, year=None, collection_name=None):
        if not collection_name:
            collection_name = os.getenv("SEAECH_NAME")
        all_results = []
        for query in [queries]:
            # embedding = self.encode_data(query)

            search_res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="title_abstract_dense",
                data=[
                    self.encode_data(query)
                ],  # Use the `emb_text` function to convert the question to an embedding vector
                limit=5,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["title", "abstract", "pdf_path"],
            )

            for hit in search_res[0]:
                entity = hit["entity"]
                # pub_year = entity.get("year")
                # if year and pub_year != year:
                #     continue
                all_results.append(
                    {
                        "title": entity.get("title"),
                        "abstract": entity.get("abstract"),
                        # "authors": entity.get("authors"),
                        # "year": entity.get("year"),
                        # "source": entity.get("source"),
                        # "doi": entity.get("doi"),
                        "url": entity.get("pdf_path"),
                    }
                )
                if len(all_results) >= limit:
                    break

        return {"data": all_results, "total": len(all_results), "offset": 0}

    def search_milvus_queries(self, queries, limit=3, year=None, collection_name=None):
        collection_name = os.getenv("TUNING_NAME")
        all_results = []
        for query in queries:
            # embedding = self.encode_data(query)

            search_res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="tuning_effects_dense",
                data=[
                    self.encode_data(query)
                ],  # Use the `emb_text` function to convert the question to an embedding vector
                limit=5,  # Return top 5 results
                search_params={"metric_type": "IP", "params": {}},  # Inner product distance
                output_fields=[
                    "id",
                    "tuning_dimensions",
                    "tuning_strategies",
                    "tuning_effects",
                ],  # Return the text field
            )
            for hit in search_res[0]:
                entity = hit["entity"]
                # pub_year = entity.get("year")
                # if year and pub_year != year:
                #     continue
                all_results.append(
                    {
                        "tuning_dimensions": entity.get("tuning_dimensions"),
                        "tuning_strategies": entity.get("tuning_strategies"),
                        # "authors": entity.get("authors"),
                        # "year": entity.get("year"),
                        # "source": entity.get("source"),
                        # "doi": entity.get("doi"),
                        "tuning_effects": entity.get("tuning_effects"),
                    }
                )
                if len(all_results) >= limit:
                    break

        return {"data": all_results, "total": len(all_results), "offset": 0}

    def forward(self, queries, collection_name=None):
        '''
        输入: queries 是一个包含查询文本的list
            milvus_client 是一个 Milvus 客户端实例
        输出: json.dumps(retrieved_contents, indent=4) 一定是str

        collection_name:要查询的集合名称。
        anns_field:要搜索的向量字段名。
        data:查询向量。
        limit:返回的结果数量。
        search_params:搜索参数,这里使用内积(IP)作为距离度量。
        output_fields:要返回的字段。
        '''
        # 加载Milvus
        if not collection_name:
            collection_name = os.getenv("COLLECTION_NAME")

        for query in queries:

            """
            # 筛选出template_type_dense 为 First promblem solving method 的对象
            query_vector = encode_data(query)
            expr = "template_type_dense == 'First promblem solving method'
            """
            search_res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="template_name_dense",
                data=[
                    self.encode_data(query)
                ],  # Use the `emb_text` function to convert the question to an embedding vector
                limit=5,  # Return top 5 results
                search_params={"metric_type": "IP", "params": {}},  # Inner product distance
                output_fields=[
                    "id",
                    "tuning_dimensions",
                    "tuning_strategies",
                    "tuning_effects",
                ],  # Return the text field
                # 筛选出template_type_dense 为 First promblem solving method 的对象
                # expr=expr
            )
        retrieved_contents = [
            (
                res["entity"]["id"],
                res["entity"]["template_name"],
                res["entity"]["application_scenarios"],
                res["entity"]["reason_flow"],
                res["entity"]["example_application"],
            )
            for res in search_res[0]
        ]
        # print(json.dumps(retrieved_contents, indent=4))
        return retrieved_contents

    def __call__(self, queries):
        """使实例可以像函数一样调用，等同于调用 forward"""
        return self.forward(queries)
