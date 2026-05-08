"""
运行可回溯思维链分析的示例脚本
"""

import sys
import os
from pathlib import Path
import yaml
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入需要的模块
from supramas.agents import get_llms  # type: ignore
from supramas.conversation_logger import ConversationLogger  # type: ignore

# 导入traceback pipeline
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from supramas.explainability.pipeline import TracebackPipeline  # type: ignore


def main():
    """主函数
    
    用法:
        python run_traceback.py                         # 使用 input/ 目录中的独立文件
        python run_traceback.py /path/to/all_protocol.json  # 使用 all_protocol.json
    """
    
    # 加载配置文件
    config_path = project_root / "config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # 设置环境变量
    for key, value in config.items():
        if value == "":
            continue
        else:
            os.environ[key] = str(value)
    
    # 配置路径
    data_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理命令行参数：可选的 all_protocol.json 路径
    protocol_path = None
    if len(sys.argv) > 1:
        protocol_path = Path(sys.argv[1])
        if not protocol_path.exists():
            logger.error(f"指定的 protocol 文件不存在: {protocol_path}")
            sys.exit(1)
        logger.info(f"使用 protocol 文件: {protocol_path}")
    
    # 初始化日志记录器
    conversation_logger = ConversationLogger(
        log_file=output_dir / "traceback_conversation.json"
    )
    
    # 获取LLM实例
    logger.info("初始化LLM...")
    llm = get_llms(conversation_logger)
    
    # 创建回溯pipeline
    logger.info("创建回溯Pipeline...")
    pipeline = TracebackPipeline(
        llm=llm, 
        data_dir=data_dir,
        protocol_path=protocol_path
    )
    
    # 运行回溯分析
    logger.info("开始运行回溯分析...")
    chains = pipeline.run(output_dir=output_dir)
    
    # 输出总结
    logger.info("\n" + "="*80)
    logger.info("回溯分析完成!")
    logger.info(f"生成了 {len(chains)} 条回溯链")
    
    # 显示可信度摘要
    for i, chain in enumerate(chains):
        name = chain.mechanism_finding.get("name", "N/A") if isinstance(chain.mechanism_finding, dict) else str(chain.mechanism_finding)
        score = chain.credibility.get("overall_score", 0)
        logger.info(f"  回溯链 {i+1}: {name} | 可信度: {score:.2f}")
    
    logger.info(f"结果保存在: {output_dir}")
    logger.info("  - traceback_chains.json: JSON格式的完整数据")
    logger.info("  - traceback_report.md: 可读的Markdown报告")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
