import html2text
from pathlib import Path

def html_to_markdown(html_path, md_path):
    html_path = Path(html_path)
    md_path = Path(md_path)

    html_content = html_path.read_text(encoding="utf-8")

    h = html2text.HTML2Text()
    h.ignore_links = False        # 保留链接
    h.ignore_images = False       # 保留图片
    h.body_width = 0              # 不自动换行
    h.protect_links = True

    markdown = h.handle(html_content)

    md_path.write_text(markdown, encoding="utf-8")
    print(f"转换完成：{md_path}")

if __name__ == "__main__":
    html_file = "3级_ex1_迭代第4次.html"
    md_file = "3级_ex1_迭代第4次.md"
    html_to_markdown(html_file, md_file)
