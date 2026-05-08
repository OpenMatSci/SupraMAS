import json
import os

import markdown

# 使用topic_path作为基础路径
topic_path = os.environ.get('TOPIC_PATH', './')
os.chdir(topic_path)

# 读取 Markdown 文件
input_filename = "experiment_plan.md"
with open(input_filename, "r", encoding="utf-8") as file:
    content = file.read()
# 将 Markdown 转换为 HTML
html_output = markdown.markdown(content)

# 保存 HTML 到文件
output_filename = "result.html"

with open(output_filename, "w", encoding="utf-8") as file:
    file.write(html_output)

print(f"HTML 文件已保存为 {output_filename}")
