import json
# import markdown

topic = "Enhanced Superconducting Properties of 'EuBa2Cu3O7'-Doped 'Ba2YTaO6' Thin Films Prepared by PLD"

# Optional defaults for local conversion helpers.
protocol_file = "all_protocol.json"
saved_path = "all_protocol.md"

def format_idea_markdown(idea):
    """将 idea dict 格式化为可读的 Markdown"""
    if isinstance(idea, str):
        return idea
    if not isinstance(idea, dict):
        return str(idea)

    md = ""
    main_fields = [
        ("material", "材料体系"),
        ("motivation", "研究动机"),
        ("technique", "实验技术"),
        ("outcome", "预期结果"),
    ]
    for key, label in main_fields:
        if key in idea:
            md += f"**{label}** ({key}): {idea[key]}\n\n"
            source_key = f"{key}_source"
            if source_key in idea and isinstance(idea[source_key], dict):
                for node_id, source_text in idea[source_key].items():
                    md += f"> **来源 [{node_id}]**: {source_text}\n"
                md += "\n"

    if "strategies" in idea and isinstance(idea["strategies"], list):
        md += "**调控策略** (strategies):\n\n"
        for idx, strategy in enumerate(idea["strategies"], 1):
            dim = strategy.get("tuning_dimensions", "")
            strat = strategy.get("tuning_strategies", "")
            md += f"{idx}. **{dim}**: {strat}\n"
            source_node = strategy.get("source_node", {})
            if isinstance(source_node, dict):
                for node_id, source_text in source_node.items():
                    md += f"   > **来源 [{node_id}]**: {source_text}\n"
            md += "\n"

    if "node_mapping" in idea and isinstance(idea["node_mapping"], dict):
        nm = idea["node_mapping"]
        md += "**节点映射** (node_mapping):\n\n"
        for k, v in nm.items():
            md += f"- **{k}**: {v}\n"
        md += "\n"
        if "node_mapping_source" in idea and isinstance(idea["node_mapping_source"], dict):
            for k, v in idea["node_mapping_source"].items():
                md += f"> **{k}**: {v}\n"
            md += "\n"

    return md


def format_knowledge_markdown(knowledge):
    """将 knowledge dict 格式化为可读的 Markdown"""
    if isinstance(knowledge, str):
        return knowledge
    if not isinstance(knowledge, dict):
        return str(knowledge) if knowledge else ""
    md = ""
    for kid, desc in knowledge.items():
        md += f"- **知识 [{kid}]**: {desc}\n"
    return md


def export_markdown(exp_plan):
    """导出实验计划为 Markdown"""
    md = "# Idea 创新 \n"
    md += format_idea_markdown(exp_plan['idea'])
    md += "\n"

    md += f"# 实验计划 \n ## 实验初稿\n{exp_plan['draft']}\n\n"

    md += "## 详细实验方案\n"
    for step in exp_plan['steps']:
        md += f"\n### 实验任务{step['id']+1}\n"
        if isinstance(step["description"], dict):
            for k,v in step["description"].items():
                md += f"\n- **{k}**: {v}\n"
        elif isinstance(step["description"], list):
            md += f"\n{step['description'][0]}\n"

        md += f"\n#### 具体内容 \n{step['detailed_plan']}\n"
        md += f"\n#### 相关知识 \n{format_knowledge_markdown(step['knowledge'])}\n"

    md += "\n# 精炼的实验计划\n"
    md += f"{exp_plan['final_plan']}\n"

    return md



if __name__ == "__main__":
    raise SystemExit("Use supramas.utils.results_format_cli for command-line conversion.")
