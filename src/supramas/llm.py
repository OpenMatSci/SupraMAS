import base64
import logging
import os
from abc import abstractmethod

import httpx
import numpy as np
import openai
import requests
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from supramas.utils import *


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        logging.info(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_openai_url(img_pth):
    end = img_pth.split(".")[-1]
    if end == "jpg":
        end = "jpeg"
    base64_image = encode_image(img_pth)
    return f"data:image/{end};base64,{base64_image}"


class base_llm:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, messages, **kwargs):
        pass


class openai_llm(base_llm):
    def __init__(self, model="gpt4o-0513",conversation_logger=None) -> None:
        super().__init__()

        self.model = model
        self.conversation_logger = conversation_logger

        if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
            raise ValueError("OPENAI_API_KEY is not set")

        api_key = os.environ.get("OPENAI_API_KEY", None)
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        self.use_request = os.environ.get("USE_REQUEST", False)=="True"

        if self.use_request:
            self.headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            self.url = base_url

        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)



    def cal_cosine_similarity(self, vec1, vec2):
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chat_completion(self, messages, load_json=False,**kwargs):
        try:
            if self.use_request:
                payload = json.dumps({
                    "model": kwargs.get("model", self.model),
                    "messages": messages,
                    "n": kwargs.get("n", 1),
                    "temperature": kwargs.get("temperature", 0.7),
                    "timeout": kwargs.get("timeout", 180),
                })

                response = requests.post(self.url, headers=self.headers, data=payload, timeout=3000)
                # response.raise_for_status()
                output = response.json()["choices"][0]["message"]["content"]

            else:
                response = self.client.chat.completions.create(
                    model=kwargs.get("model", self.model),
                    messages=messages,
                    n=kwargs.get("n", 1),
                    temperature=kwargs.get("temperature", 0.7),
                    timeout=kwargs.get("timeout", 180),
                )
                output = response.choices[0].message.content

            data = extract(output,kwargs.get("extract_key"))
            if load_json:   
                data = json.loads(data.replace("\\", "\\\\")) 
                if isinstance(data,str):
                    return None
            return data
        except Exception as e:
            print(f"Error in response: {e},retry!")
            return None

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    def response(self, messages, load_json=False,**kwargs):
        
        data = self.chat_completion(messages, load_json=load_json, **kwargs)

        while not data:
            data = self.chat_completion(messages, load_json=load_json, **kwargs)

        self.conversation_logger.add_entry_logger(kwargs.get("stage"),kwargs.get("role"),data,kwargs.get("meta",{}))

        return data
        

        
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    def get_embbeding(self, text):
        """使用硅基流动 API (Qwen/Qwen3-Embedding-8B) 获取文本编码向量，维度4096"""
        try:
            url = os.getenv("EMBEDDING_API_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings")
            headers = {
                "Authorization": f"Bearer {os.getenv('EMBEDDING_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": os.getenv("EMBEDDING_NAME", "Qwen/Qwen3-Embedding-8B"),
                "input": text
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            embbeding = result["data"]

            if len(embbeding) == 0:
                return None
            elif len(embbeding) == 1:
                return embbeding[0]["embedding"]
            else:
                return [e["embedding"] for e in embbeding]
        except Exception as e:
            print(f"get embbeding failed: {e}")
            print(e)
            logging.info(e)

            return
