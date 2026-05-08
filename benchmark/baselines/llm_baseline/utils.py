import json
import redis
from loguru import logger
import os
import re
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

def extract_sub_experiments(text):
        pattern = re.compile(r"<sub_experiment_(\d+)>\s*(.*?)\s*</sub_experiment_\1>", re.DOTALL)
        experiments = []
        for match in pattern.findall(text):
            id, content = match
            experiments.append({"id": id, "content": content.strip()})
        # print("experiments", experiments)
        return experiments
def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

def extract_exp(text):
    pattern = re.compile(r"<exp(\d+)>\s*(.*?)\s*</exp\1>", re.DOTALL)
    # **解析实验内容**
    experiments = []
    for match in pattern.findall(text):
        id, content = match
        experiments.append({"idx": id, "content": content.strip()})
    return experiments

def extract_json(text):
    if "```json" in text:
        target_str = get_content_between_a_b("```json", "```", text)
        return target_str
    else:
        return text



def return_data_json(data):
    return json.dumps(data, ensure_ascii=False)


def set_redis_stage(stage,session_key):
    redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"),password=os.getenv("REDIS_PASSWORD"))
    redis_client.set(f"{session_key}:{stage}",stage)

def get_redis_stage(session_key,stage):
    redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"),password=os.getenv("REDIS_PASSWORD"))
    return redis_client.get(f"{session_key}:{stage}")

def delete_redis_stage(session_key,stage):
    redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"),password=os.getenv("REDIS_PASSWORD"))
    redis_client.delete(f"{session_key}:{stage}")

def check_stage_redis(session_key,stage):
    value = get_redis_stage(session_key,stage)
    if value and value.decode('utf-8') == stage:
        return True
    return False


def translate_api(text):
    import json

    from volcengine.ApiInfo import ApiInfo
    from volcengine.Credentials import Credentials
    from volcengine.ServiceInfo import ServiceInfo
    from volcengine.base.Service import Service

    k_access_key = os.environ.get("VOLCENGINE_ACCESS_KEY")
    k_secret_key = os.environ.get("VOLCENGINE_SECRET_KEY")
    if not k_access_key or not k_secret_key:
        logger.warning("Volcengine translation credentials are not set; returning untranslated text.")
        return text
    k_service_info = \
        ServiceInfo('translate.volcengineapi.com',
                    {'Content-Type': 'application/json'},
                    Credentials(k_access_key, k_secret_key, 'translate', 'cn-north-1'),
                    5,
                    5)
    k_query = {
        'Action': 'TranslateText',
        'Version': '2020-06-01'
    }
    k_api_info = {
        'translate': ApiInfo('POST', '/', k_query, {}, {})
    }
    service = Service(k_service_info, k_api_info)
    body = {
        'TargetLanguage': 'en',
        'TextList': [text],
    }
    res = service.json('translate', {}, json.dumps(body))
    target_text = json.loads(res)['TranslationList'][0]['Translation']
    return target_text
    
    
