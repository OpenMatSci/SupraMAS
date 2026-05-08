import json
import os
import re

from loguru import logger


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


def extract(text, type, hard=True):
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


def extract_json(text):
    if "```json" in text:
        target_str = get_content_between_a_b("```json", "```", text)
        return target_str
    else:
        return text


def return_data_json(data):
    return json.dumps(data, ensure_ascii=False)


def translate_api(text, target_lang='zh'):
    """Translate text with Volcengine when credentials are configured.

    Credentials are intentionally read from environment variables so the public
    repository never contains access keys.
    """
    access_key = os.environ.get("VOLCENGINE_ACCESS_KEY")
    secret_key = os.environ.get("VOLCENGINE_SECRET_KEY")
    if not access_key or not secret_key:
        logger.warning("VOLCENGINE_ACCESS_KEY/VOLCENGINE_SECRET_KEY are not set; returning untranslated text.")
        return text

    from volcengine.ApiInfo import ApiInfo
    from volcengine.base.Service import Service
    from volcengine.Credentials import Credentials
    from volcengine.ServiceInfo import ServiceInfo

    service_info = ServiceInfo(
        'translate.volcengineapi.com',
        {'Content-Type': 'application/json'},
        Credentials(access_key, secret_key, 'translate', 'cn-north-1'),
        5,
        5,
    )
    query = {'Action': 'TranslateText', 'Version': '2020-06-01'}
    api_info = {'translate': ApiInfo('POST', '/', query, {}, {})}
    service = Service(service_info, api_info)

    max_chunk_size = 5000
    results = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i : i + max_chunk_size]
        body = {'TargetLanguage': target_lang, 'TextList': [chunk]}
        res = service.json('translate', {}, json.dumps(body))
        data = json.loads(res)
        if 'ResponseMetadata' in data and 'Error' in data['ResponseMetadata']:
            err = data['ResponseMetadata']['Error']
            raise RuntimeError(f"Translation failed: {err.get('Code')} - {err.get('Message')}")
        translation_list = data.get("TranslationList")
        if not translation_list:
            raise ValueError(f"Translation response missing TranslationList: {res}")
        results.append(translation_list[0]['Translation'])
    return ''.join(results)


# 从根目录utils.py保留的函数
def extract_ideas(response_text):
    """
    Extract all <idea>...</idea> blocks from the model response.

    Args:
        response_text (str): The text response from the LLM.

    Returns:
        list of dict: Each item is a dict with keys 'motivation', 'Composition tuning',
                      'Grain boundary tuning', 'Interface tuning', 'Texture tuning', and 'Stress tuning'.
    """
    # 提取所有<idea>...</idea>块
    idea_blocks = re.findall(r"<idea(\d+)>\s*(.*?)\s*</idea\1>", response_text, re.DOTALL)
    ideas = []
    for block in idea_blocks:
        idx, data = block
        idea = {
            "motivation": extract_tag_content(data, "motivation"),
            "Composition tuning": extract_tag_content(data, "Composition tuning"),
            "Grain boundary tuning": extract_tag_content(data, "Grain boundary tuning"),
            "Interface tuning": extract_tag_content(data, "Interface tuning"),
            "Texture tuning": extract_tag_content(data, "Texture tuning"),
            "Stress tuning": extract_tag_content(data, "Stress tuning"),
        }
        ideas.append(idea)

    return ideas


def extract_tag_content(text, tag):
    """
    Helper function to extract content inside a specific tag.

    Args:
        text (str): A block of text.
        tag (str): The tag name to extract.

    Returns:
        str: The content inside the tag or an empty string if not found.
    """
    pattern = fr"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


# 从ProtocolOptimizer/protocol_utils.py添加的函数
def extract_sub_experiments(text):
    """提取子实验"""
    pattern = re.compile(r"<sub_experiment_(\d+)>\s*(.*?)\s*</sub_experiment_\1>", re.DOTALL)
    experiments = []
    for match in pattern.findall(text):
        id, content = match
        experiments.append({"id": id, "content": content.strip()})
    return experiments


def extract_final_idea(response):
    """提取最终想法"""
    return re.findall(r"<template\d+>(.*?)</template\d+>", response, re.DOTALL)


def extract_exp(text):
    """提取实验"""
    pattern = re.compile(r"<exp(\d+)>\s*(.*?)\s*</exp\1>", re.DOTALL)
    experiments = []
    for match in pattern.findall(text):
        id, content = match
        experiments.append({"idx": id, "content": content.strip()})
    return experiments
