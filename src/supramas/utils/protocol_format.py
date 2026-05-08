from pathlib import Path
import markdown
import html2text


def md_to_html(md_path, html_path):
    md_path = Path(md_path)
    html_path = Path(html_path)

    md_text = md_path.read_text(encoding="utf-8")

    html_body = markdown.markdown(md_text)

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_body)
    print(f"[MD → HTML] 完成：{html_path}")


def html_to_md(html_path, md_path):
    html_path = Path(html_path)
    md_path = Path(md_path)

    html_text = html_path.read_text(encoding="utf-8")

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0
    h.protect_links = True

    md_text = h.handle(html_text)

    md_path.write_text(md_text, encoding="utf-8")
    print(f"[HTML → MD] 完成：{md_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Markdown ⇄ HTML 双向转换工具"
    )
    parser.add_argument(
        "--mode",
        choices=["md2html", "html2md"],
        default="html2md",
        help="转换模式（默认：md2html）"
    )
    parser.add_argument(
        "--input",
        help="输入文件路径",
        default="idea.html"
    )
    parser.add_argument(
        "--output",
        help="输出文件路径",
        default="idea.md"
    )

    args = parser.parse_args()

    if args.mode == "md2html":
        md_to_html(args.input, args.output)
    else:
        html_to_md(args.input, args.output)
