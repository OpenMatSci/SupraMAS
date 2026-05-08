import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
class ConversationLogger:
    def __init__(self,log_file):
        """初始化对话记录"""
        self.log_file = log_file
        self._get_latest_log()


    def _get_latest_log(self):
        if os.path.exists(self.log_file):
            self.logs = json.load(open(self.log_file,"r",encoding="utf-8"))
        else:
            self.logs = []
        

    def add_entry_logger(self, stage: str, role: str, message: str, meta: Optional[Dict] = None) -> None:
        """
        添加对话日志条目
        
        Args:
            stage: 记录所处的阶段
            role: 智能体名称
            message: 智能体输出内容
            meta: 额外信息（可选）
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'role': role,
            'message': message,
            'meta': meta if meta is not None else {}
        }
        self.logs.append(entry)
        with open(self.log_file,"w",encoding="utf-8") as f:
            json.dump(self.logs,f,indent=4,ensure_ascii=False)


    def get_all_logs(self) -> List[Dict[str, Any]]:
        """获取所有日志条目"""
        return self.logs

    def get_logs_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """按阶段查询日志"""
        return [log for log in self.logs if log['stage'] == stage]

    def get_logs_by_role(self, role: str) -> List[Dict[str, Any]]:
        """按角色查询日志"""
        return [log for log in self.logs if log['role'] == role]

    def get_logs_by_time_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        按时间范围查询日志
        时间格式: ISO 8601 (e.g., '2023-06-01T00:00:00')
        """
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return [log for log in self.logs 
                   if start <= datetime.fromisoformat(log['timestamp']) <= end]
        except ValueError as e:
            raise ValueError("Invalid datetime format. Use ISO 8601 format.") from e

    def get_latest_log(self) -> Optional[Dict[str, Any]]:
        """获取最新日志条目"""
        return self.logs[-1] if self.logs else None

    def get_logs_with_meta_key(self, meta_key: str) -> List[Dict[str, Any]]:
        """查询包含特定meta键的日志"""
        return [log for log in self.logs if meta_key in log['meta']]

    def clear_logs(self) -> None:
        """清空所有日志"""
        self.logs.clear()