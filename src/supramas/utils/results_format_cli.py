"""
results_format.py
将 final_protocol.json / all_protocol.json 转换为 Markdown 文件。

用法:
    # 转换单个文件
    python results_format.py path/to/final_protocol.json

    # 自动扫描某目录下所有 JSON 文件并批量转换
    python results_format.py path/to/results_test/

    # 不传参数则使用脚本内 DEFAULT_PATH
    python results_format.py
"""

import json
import sys
import os
from pathlib import Path

# ──────────────────────────────────────────────
# 默认路径（不传命令行参数时使用）
# ──────────────────────────────────────────────
DEFAULT_PATH = "outputs"

# 需要转换的 JSON 文件名（支持多种命名）
TARGET_FILENAMES = {"final_protocol.json", "all_protocol.json", "final_protocol.json"}


# ══════════════════════════════════════════════
#  辅助：source 字典 → Markdown 引用块
# ══════════════════════════════════════════════

def _render_source(source: dict, indent: str = "> ") -> str:
    """将 _source dict（含 strategy_tree / template 两个子键）渲染为 Markdown 引用块。"""
    if not source or not isinstance(source, dict):
        return ""
    lines = []
    strategy_tree = source.get("strategy_tree", {})
    template = source.get("template", {})
    if strategy_tree:
        lines.append(f"{indent}**Strategy Tree**")
        for node_id, text in strategy_tree.items():
            lines.append(f"{indent}- Node {node_id}: {text}")
    if template:
        lines.append(f"{indent}**Template**")
        for tmpl_id, text in template.items():
            lines.append(f"{indent}- Template {tmpl_id}: {text}")
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════
#  Section 1: Idea
# ══════════════════════════════════════════════

def format_idea(idea: dict) -> str:
    if not isinstance(idea, dict):
        return str(idea)

    md = "## 创新 Idea\n\n"

    # 主字段：字段名 → 中文标签
    main_fields = [
        ("material",   "材料体系"),
        ("motivation", "研究动机"),
        ("technique",  "实验技术"),
        ("outcome",    "预期结果"),
    ]
    for key, label in main_fields:
        if key not in idea:
            continue
        md += f"### {label}\n\n{idea[key]}\n\n"
        source = idea.get(f"{key}_source")
        if source:
            md += _render_source(source)
            md += "\n"

    # strategies
    strategies = idea.get("strategies", [])
    if strategies:
        md += "### 调控策略\n\n"
        for idx, s in enumerate(strategies, 1):
            dim   = s.get("tuning_dimensions", "")
            strat = s.get("tuning_strategies", "")
            md += f"**{idx}. {dim}**\n\n{strat}\n\n"
            # 新版：strategy_source；旧版：source_node
            source = s.get("strategy_source") or s.get("source_node")
            if source:
                md += _render_source(source, indent="> ")
                md += "\n"

    # node_mapping
    nm = idea.get("node_mapping")
    if nm and isinstance(nm, dict):
        md += "### 节点映射\n\n"
        for k, v in nm.items():
            md += f"- **{k}**: {v}\n"
        md += "\n"

    return md


# ══════════════════════════════════════════════
#  Section 2: 实验初稿 (draft)
# ══════════════════════════════════════════════

def format_draft(draft: str) -> str:
    if not draft:
        return ""
    return f"## 实验初稿\n\n{draft}\n\n"


# ══════════════════════════════════════════════
#  Section 3: 详细实验步骤 (steps)
# ══════════════════════════════════════════════

def format_steps(steps: list) -> str:
    if not steps:
        return ""
    md = "## 详细实验方案\n\n"
    for step in steps:
        step_id = step.get("id", "?")
        md += f"### 实验步骤 {step_id + 1}\n\n"

        # description
        desc = step.get("description", {})
        if isinstance(desc, dict):
            for k, v in desc.items():
                md += f"- **{k}**: {v}\n"
            md += "\n"
        elif isinstance(desc, str):
            md += desc + "\n\n"

        # detailed_plan
        detailed = step.get("detailed_plan", "")
        if detailed:
            md += f"#### 具体操作\n\n{detailed}\n\n"

        # knowledge
        knowledge = step.get("knowledge", {})
        if knowledge:
            md += "#### 相关知识\n\n"
            if isinstance(knowledge, dict):
                for kid, kdesc in knowledge.items():
                    md += f"- **[{kid}]** {kdesc}\n"
            else:
                md += str(knowledge)
            md += "\n"

    return md


# ══════════════════════════════════════════════
#  Section 4: 精炼实验计划 (final_plan)
# ══════════════════════════════════════════════

def format_final_plan(final_plan: str) -> str:
    if not final_plan or not final_plan.strip():
        return ""
    return f"## 精炼实验计划\n\n{final_plan}\n\n"


# ══════════════════════════════════════════════
#  主入口：JSON → Markdown 字符串
# ══════════════════════════════════════════════

def json_to_markdown(data: dict, title: str = "") -> str:
    md = ""
    if title:
        md += f"# {title}\n\n---\n\n"

    if "idea" in data:
        md += format_idea(data["idea"])
        md += "\n---\n\n"

    if "draft" in data:
        md += format_draft(data["draft"])
        md += "---\n\n"

    if "steps" in data:
        md += format_steps(data["steps"])
        md += "---\n\n"

    if "final_plan" in data:
        md += format_final_plan(data["final_plan"])

    return md


# ══════════════════════════════════════════════
#  文件转换
# ══════════════════════════════════════════════

def convert_file(json_path: Path) -> Path:
    """读取 JSON，写入同目录同名 .md 文件，返回输出路径。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 用父目录名作为标题（通常是课题名）
    title = json_path.parent.name

    md_content = json_to_markdown(data, title=title)

    out_path = json_path.with_suffix(".md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return out_path


def discover_and_convert(root: Path) -> list[Path]:
    """递归搜索 root 下所有目标 JSON 文件并转换，返回输出文件列表。"""
    outputs = []
    for json_file in root.rglob("*.json"):
        if json_file.name in TARGET_FILENAMES:
            try:
                out = convert_file(json_file)
                print(f"✅ {json_file.relative_to(root)}  →  {out.name}")
                outputs.append(out)
            except Exception as e:
                print(f"❌ {json_file}: {e}")
    return outputs


# ══════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    target = Path(arg)

    if not target.exists():
        print(f"路径不存在: {target}")
        sys.exit(1)

    if target.is_file():
        if target.suffix.lower() != ".json":
            print(f"请传入 .json 文件，收到: {target}")
            sys.exit(1)
        out = convert_file(target)
        print(f"✅ Markdown 已保存: {out}")
    else:
        # 目录：批量转换
        outputs = discover_and_convert(target)
        if outputs:
            print(f"\n共转换 {len(outputs)} 个文件。")
        else:
            print(f"在 {target} 下未找到目标 JSON 文件 {TARGET_FILENAMES}。")


if __name__ == "__main__":
    main()
