# -*- coding: utf-8 -*-
"""
露尼西亚AI助手 - 主程序入口
重构后的模块化版本
"""

import sys
import os

# 在导入任何其他模块之前，先设置警告过滤器
import warnings
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore")

import atexit
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import QObject, pyqtSignal
from src.core.async_resource_manager import cleanup_on_exit

# 设置Windows控制台编码为UTF-8，避免emoji显示错误
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
        # 同时设置Python的标准输出编码
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"Warning: Failed to set console encoding: {e}")

# 额外的警告抑制
import asyncio
import logging

# 设置asyncio日志级别为WARNING，避免过多调试信息
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)

# 导入自定义模块
from config.config import load_config
from src.agents.ai_agent import AIAgent
from src.main_window import AIAgentApp

def main():
    """主程序入口"""
    # 设置异常处理，避免程序退出时的异常输出
    import signal
    import atexit
    
    def signal_handler(signum, frame):
        """信号处理器，静默退出"""
        cleanup_on_exit()
        sys.exit(0)
    
    # 注册信号处理器和退出清理函数
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup_on_exit)
    except Exception:
        pass
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))  # 深蓝色背景
    palette.setColor(QPalette.WindowText, QColor(205, 214, 244))  # 浅蓝色文本
    palette.setColor(QPalette.Base, QColor(24, 24, 37))  # 更深的基础色
    palette.setColor(QPalette.AlternateBase, QColor(49, 50, 68))  # 交替基础色
    palette.setColor(QPalette.ToolTipBase, QColor(180, 190, 254))  # 工具提示基础色
    palette.setColor(QPalette.ToolTipText, QColor(24, 24, 37))  # 工具提示文本
    palette.setColor(QPalette.Text, QColor(205, 214, 244))  # 文本颜色
    palette.setColor(QPalette.Button, QColor(49, 50, 68))  # 按钮颜色
    palette.setColor(QPalette.ButtonText, QColor(205, 214, 244))  # 按钮文本
    palette.setColor(QPalette.BrightText, QColor(245, 224, 220))  # 亮色文本
    palette.setColor(QPalette.Highlight, QColor(137, 180, 250))  # 高亮色
    palette.setColor(QPalette.HighlightedText, QColor(24, 24, 37))  # 高亮文本
    
    app.setPalette(palette)
    
    # 设置字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    
    # 加载配置
    config = load_config()
    
    # 创建主窗口
    window = AIAgentApp(config)
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


