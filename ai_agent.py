# -*- coding: utf-8 -*-
"""
AI代理核心模块
处理用户输入、工具调用和AI响应生成
"""

import datetime
import re
import openai
import subprocess
import os
from typing import Dict, Any
from config import load_config
from utils import get_location, scan_windows_apps, open_website, open_application
from weather import WeatherTool
from amap_tool import AmapTool
from memory_lake import MemoryLake
from mcp_server import LocalMCPServer
from search_tool import search_web as web_search
from search_summary_agent import process_search_result, should_search
from search_query_extractor import extract_search_query
from playwright_tool import playwright_search, playwright_open_url, playwright_interact, playwright_open_website_headed
from file_analysis_tool import FileAnalysisTool
from webpage_agent_unified import execute_webpage_task_sync

class MCPTools:
    """MCP工具管理类"""
    
    def __init__(self):
        self.server = LocalMCPServer()
    
    def execute_mcp_command(self, tool_name, **params):
        """执行MCP命令（同步版本）"""
        try:
            # 重新加载自定义工具
            self.server.reload_custom_tools()
            result = self.server.call_tool(tool_name, **params)
            return result
        except Exception as e:
            return f"MCP命令执行失败: {str(e)}"
    
    async def execute_mcp_command_async(self, tool_name, **params):
        """执行MCP命令（异步版本）"""
        try:
            # 重新加载自定义工具
            self.server.reload_custom_tools()
            result = self.server.call_tool(tool_name, **params)
            return result
        except Exception as e:
            return f"MCP命令执行失败: {str(e)}"
    
    def list_available_tools(self):
        """列出可用工具（同步版本）"""
        try:
            return self.server.list_tools()
        except Exception as e:
            print(f"获取工具列表失败: {str(e)}")
            return []
    
    async def list_available_tools_async(self):
        """列出可用工具（异步版本）"""
        try:
            return self.server.list_tools()
        except Exception as e:
            print(f"获取工具列表失败: {str(e)}")
            return []
    
    def list_tools(self):
        """同步版本的工具列表获取"""
        try:
            return self.server.list_tools()
        except Exception as e:
            print(f"获取工具列表失败: {str(e)}")
            return []
    
    def get_tool_info(self, tool_name):
        """获取工具信息（同步版本）"""
        try:
            return self.server.get_tool_info(tool_name)
        except Exception as e:
            print(f"获取工具信息失败: {str(e)}")
            return {}
    
    async def get_tool_info_async(self, tool_name):
        """获取工具信息（异步版本）"""
        try:
            return self.server.get_tool_info(tool_name)
        except Exception as e:
            print(f"获取工具信息失败: {str(e)}")
            return {}

class AIAgent:
    """露尼西亚AI核心"""
    
    def __init__(self, config):
        self.name = "露尼西亚"
        self.role = "游戏少女前线中威廉的姐姐"
        self.memory_lake = MemoryLake()
        self.developer_mode = False
        self.current_topic = ""
        self.conversation_history = []
        self.config = config
        self.location = get_location()
        self.last_save_date = None
        
        # 本次程序运行时的对话记录
        self.session_conversations = []
        
        # 最近生成的代码缓存
        self.last_generated_code = None

        # 可用的工具
        self.tools = {
            "天气": WeatherTool.get_weather,
            "网页打开与自动化操作": self._open_website_wrapper,
            "打开应用": open_application,
            "获取时间": self._get_current_time,
            "搜索": web_search,
        }
        
        # 初始化MCP工具
        self.mcp_server = LocalMCPServer()
        self.mcp_tools = MCPTools()
        
        # 统一网页Agent已改为函数调用方式，无需初始化
        # self.webpage_agent = WebpageAgent(config)  # 旧版已废弃
        
        
        # 初始化文件分析工具
        self.file_analyzer = FileAnalysisTool(config)
        print("📄 文件分析工具已初始化 (PDF、CSV、Excel、Word、代码)")
        
        # 文件分析上下文记忆（最近分析的文件）
        self.recent_file_analysis = None  # 存储最近一次的文件分析结果
        
        # 框架ReAct Agent（默认启用，轻量级任务规划）
        from framework_react_agent import FrameworkReActAgent
        intent_model = config.get("search_intent_model", "deepseek-chat")
        self.framework_agent = FrameworkReActAgent(self, intent_model)
        print(f"🧠 框架ReAct模式已启用（使用模型：{intent_model}）")

        # 网站和应用映射
        self.website_map = config.get("website_map", {})

        # 合并扫描到的应用和手动添加的应用
        self.app_map = scan_windows_apps()
        self.app_map.update(config.get("app_map", {}))

        # 预加载应用数
        self.app_count = len(self.app_map)

        # 初始化TTS管理器
        try:
            azure_key = config.get("azure_tts_key", "")
            azure_region = config.get("azure_region", "eastasia")
            if azure_key:
                from tts_manager import TTSManager
                self.tts_manager = TTSManager(azure_key, azure_region)
                print("✅ TTS管理器初始化成功")
            else:
                self.tts_manager = None
                print("ℹ️ 未配置TTS密钥，TTS功能已禁用")
        except Exception as e:
            print(f"⚠️ TTS管理器初始化失败: {str(e)}")
            self.tts_manager = None

    def _get_llm_client(self, model=None):
        """
        获取LLM客户端（统一处理OpenAI/DeepSeek/Ollama）
        
        Returns:
            tuple: (client, model_name) 或 None if failed
        """
        provider = self.config.get("llm_provider", "DeepSeek")
        
        if provider == "Ollama":
            # Ollama配置
            ollama_url = self.config.get("ollama_url", "http://localhost:11434")
            ollama_model = self.config.get("ollama_model", "qwen2.5:latest")
            
            # Ollama使用OpenAI兼容接口
            client = openai.OpenAI(
                api_key="ollama",  # Ollama不需要真实密钥
                base_url=f"{ollama_url}/v1"
            )
            return (client, ollama_model)
        
        elif provider == "DeepSeek":
            # DeepSeek配置
            api_key = self.config.get("deepseek_key", "")
            if not api_key:
                print("⚠️ DeepSeek API密钥未配置")
                return None
            
            model_name = model or self.config.get("selected_model", "deepseek-chat")
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            return (client, model_name)
        
        elif provider == "OpenAI":
            # OpenAI配置
            api_key = self.config.get("openai_key", "")
            if not api_key:
                print("⚠️ OpenAI API密钥未配置")
                return None
            
            model_name = model or self.config.get("selected_model", "gpt-3.5-turbo")
            client = openai.OpenAI(api_key=api_key)
            return (client, model_name)
        
        else:
            print(f"⚠️ 未知的LLM提供商: {provider}")
            return None

    def process_command(self, user_input, is_first_response_after_intro=False, skip_framework=False, suppress_tool_routing=False):
        """处理用户命令"""
        # 🔒 检查安全测试确认命令
        if user_input.strip() == "确认执行安全测试":
            if hasattr(self, 'pending_security_commands') and self.pending_security_commands:
                commands = self.pending_security_commands
                self.pending_security_commands = None  # 清除待执行命令
                return self._execute_security_commands(commands)
            else:
                return "（疑惑地看着您）指挥官，没有待执行的安全测试命令。"
        
        # 检查开发者模式命令
        if user_input.lower() == "developer mode":
            self.developer_mode = True
            return "(开发者模式已激活)"
        elif user_input.lower() == "exit developer mode":
            self.developer_mode = False
            return "(开发者模式已关闭)"

        # 检查"记住这个时刻"指令
        if self._is_remember_moment_command(user_input):
            return self._handle_remember_moment(user_input)

        # 记录对话历史
        self.conversation_history.append(f"指挥官: {user_input}")

        # 🔥 框架ReAct模式（默认启用，在常规处理之前尝试）
        if not skip_framework and self.framework_agent:
            framework_response = self.framework_agent.process_command(user_input)
            if framework_response:
                # 框架Agent成功处理
                self.conversation_history.append(f"{self.name}: {framework_response}")
            
                return framework_response
            # 如果返回None，说明是简单对话，继续使用标准流程

        # 检查威廉关键词
        if "威廉" in user_input:
            self.william_count = getattr(self, 'william_count', 0) + 1
            if self.william_count > 1:
                response = "在你面前的明明是我，为什么总是提到威廉呢？"
              
                return response
            else:
                response = "威廉是我的弟弟，他很好。"
               
                return response

        # 检查是否有待迁移的记忆数据
        migration_status = self.memory_lake.get_migration_status()
        if migration_status:
            # 如果用户回答的是迁移确认
            if user_input.strip() in ['是', '否', 'yes', 'no', 'y', 'n', '确认', '取消', '同意', '拒绝']:
                migration_response = self.memory_lake.confirm_migration(user_input)
                return migration_response
            else:
                # 主动询问用户是否迁移
                old_count = migration_status["old_memory_count"]
                current_count = migration_status["current_memory_count"]
                migration_message = f"指挥官，我检测到旧版本的记忆文件，其中包含 {old_count} 条历史记忆。"
                migration_message += f"当前系统中有 {current_count} 条记忆。\n\n"
                migration_message += "是否将旧记忆迁移到新的智能回忆系统中？"
                migration_message += "迁移后您将获得更精准的记忆检索和四维度智能回忆功能。\n\n"
                migration_message += "请回答'是'或'否'。"
                return migration_message
        
        # 分析用户输入，判断是否需要获取位置和天气信息
        context_info = self._get_context_info(user_input)
        
        # 生成AI响应（包含上下文信息）
        # 在需要时临时抑制工具路由，避免在框架pass_to_main_agent阶段重复打开网页/应用
        original_suppress_flag = getattr(self, '_suppress_tool_routing', False)
        self._suppress_tool_routing = suppress_tool_routing or original_suppress_flag
        try:
            response = self._generate_response_with_context(user_input, context_info)
        finally:
            self._suppress_tool_routing = original_suppress_flag
        
        # 确保响应不为None
        if response is None:
            response = "抱歉，我没有理解您的意思，请重新表述一下。"
        
        # 记录本次会话的对话
        self._add_session_conversation(user_input, response)
        
        # 记录对话历史
        self.conversation_history.append(f"{self.name}: {response}")
        
        # 更新记忆系统
        self._update_memory_lake(user_input, response, is_first_response_after_intro)
        
        # 如果TTS已启用，播放语音
        if hasattr(self, 'tts_manager') and self.tts_manager and self.config.get("tts_enabled", False):
            try:
                # 检查TTS是否可用
                if not self.tts_manager.is_available():
                    print("⚠️ TTS不可用，跳过语音播放")
                else:
                    # 提取纯文本内容（去除表情符号等）
                    import re
                    clean_text = re.sub(r'[（\(].*?[）\)]', '', response)  # 移除括号内容
                    clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）]', '', clean_text)  # 保留中文、英文、数字和标点
                    clean_text = clean_text.strip()
                    
                    if clean_text and len(clean_text) > 0:
                        print(f"🎤 开始TTS播放: {clean_text[:50]}...")
                        try:
                            self.tts_manager.speak_text(clean_text)
                        except Exception as tts_error:
                            print(f"⚠️ TTS播放过程中出错: {str(tts_error)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("⚠️ 清理后的文本为空，跳过TTS播放")
            except Exception as e:
                print(f"⚠️ TTS播放失败: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️ TTS未启用或管理器不可用")
        
        return response

    def _open_application(self, app_name: str) -> str:
        """打开本地应用程序（兼容App映射与直接名称）。
        优先使用`self.app_map`中的已知应用路径；否则回退到系统路径解析。
        """
        try:
            if not app_name:
                return "❌ 未提供应用名称"

            print(f"🔍 [应用启动] 尝试打开应用: '{app_name}'")
            print(f"🔍 [应用启动] 当前app_map中有 {len(self.app_map)} 个应用")

            
            target = None
            app_name_lower = app_name.lower()
            matched_candidates = []
            
            for key, path in self.app_map.items():
                key_lower = key.lower()
                # 收集所有可能的匹配项
                if app_name_lower == key_lower:
                    # 完全匹配（最高优先级）
                    matched_candidates.append((key, path, 0, len(key)))
                elif app_name_lower in key_lower or key_lower in app_name_lower:
                    # 包含匹配
                    matched_candidates.append((key, path, 1, len(key)))
            
            # 优先级排序：完全匹配 > 最短匹配
            # 过滤掉"卸载"、"uninstall"等卸载程序
            filtered_candidates = [
                (key, path, priority, length) 
                for key, path, priority, length in matched_candidates
                if not any(exclude in key.lower() for exclude in ['卸载', 'uninstall', 'remove'])
            ]
            
            # 如果过滤后没有结果，使用原始候选列表
            if not filtered_candidates:
                filtered_candidates = matched_candidates
            
            # 选择最佳匹配：先按优先级排序（完全匹配优先），再按长度排序（最短优先）
            if filtered_candidates:
                filtered_candidates.sort(key=lambda x: (x[2], x[3]))
                best_match = filtered_candidates[0]
                target = best_match[1]
                print(f"✅ [应用启动] 在app_map中找到: '{best_match[0]}' -> '{best_match[1]}'")
                if len(filtered_candidates) > 1:
                    print(f"ℹ️ [应用启动] 其他匹配项: {[c[0] for c in filtered_candidates[1:3]]}")

            # 直接调用系统打开逻辑
            if target:
                print(f"🚀 [应用启动] 使用路径启动: {target}")
                result = open_application(target)
            else:
                print(f"⚠️ [应用启动] app_map中未找到，尝试直接启动: {app_name}")
                # 尝试使用Windows的start命令
                import subprocess
                try:
                    subprocess.Popen(f'start "" "{app_name}"', shell=True)
                    result = f"✅ 已启动应用程序: {app_name}"
                    print(f"✅ [应用启动] 使用start命令启动成功")
                except Exception as e:
                    print(f"❌ [应用启动] start命令失败: {e}")
                    result = f"❌ 未找到应用: {app_name}。请在设置中配置应用路径。"

            return result if isinstance(result, str) else "✅ 已尝试启动应用"
        except Exception as e:
            print(f"❌ [应用启动] 异常: {str(e)}")
            return f"❌ 打开应用失败：{str(e)}"

    def _add_session_conversation(self, user_input, ai_response):
        """添加本次会话的对话记录"""
     
        # 检查是否已经存在相同的对话
        for existing_conv in self.session_conversations:
            if (existing_conv.get('user_input') == user_input and 
                existing_conv.get('ai_response') == ai_response):
                print(f"⚠️ 检测到重复对话，跳过添加到会话记录: {user_input[:30]}...")
                return
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.session_conversations.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "ai_response": ai_response,
            "full_text": f"指挥官: {user_input}\n露尼西亚: {ai_response}",
            "saved": False  # 标记为未保存，当保存到记忆系统时会改为True
        })
        
        print(f"✅ 添加对话到会话记录: {user_input[:30]}... (当前共{len(self.session_conversations)}条)")

    def _mark_conversation_as_saved(self, user_input, ai_response):
        """标记对话为已保存"""
        # 在session_conversations中找到匹配的对话并标记为已保存
        for conv in self.session_conversations:
            if (conv.get('user_input') == user_input and 
                conv.get('ai_response') == ai_response and 
                not conv.get('saved', False)):
                conv['saved'] = True
                print(f"✅ 标记对话为已保存: {user_input[:50]}...")
                break

    def _extract_keywords(self, text):
        """提取关键词"""
        keywords = []
        # 扩展关键词列表
        common_words = [
            '天气', '时间', '搜索', '打开', '计算', '距离', '系统', '文件', '笔记', 
            '穿衣', '出门', '建议', '教堂', '景点', '历史', '参观', '路线', '法兰克福',
            '大教堂', '老城区', '游客', '高峰期', '规划', '咨询', '询问', '问过', '讨论过',
            '提到过', '说过', '介绍过', '推荐过', '建议过', '介绍', '一下', '什么', '哪里',
            '位置', '地址', '建筑', '标志性', '历史', '文化', '旅游', '游览', '参观'
        ]
        
        for word in common_words:
            if word in text:
                keywords.append(word)
        
        return keywords

    def _ai_generate_website_url(self, site_name):
        """使用AI生成网站URL"""
        try:
            # 使用统一的LLM客户端（URL生成使用快速chat模型）
            llm_result = self._get_llm_client(model="deepseek-chat")
            if not llm_result:
                print("⚠️ 无法获取LLM客户端，无法使用AI生成URL")
                return None
            
            client, model = llm_result
            print(f"🔍 [URL生成] 使用模型: {model}, 网站名: '{site_name}'")
            
            # 构建AI提示词
            url_prompt = f"""网站名称: {site_name}

请返回这个网站的官方网址（必须以https://开头）。

常见网站对应表：
- 哔哩哔哩/B站/bilibili → https://www.bilibili.com
- 知乎 → https://www.zhihu.com
- github/GitHub → https://github.com
- 百度 → https://www.baidu.com
- 谷歌/Google → https://www.google.com
- 必应/Bing → https://cn.bing.com
- 抖音 → https://www.douyin.com
- 微博 → https://weibo.com
- 淘宝 → https://www.taobao.com
- 京东 → https://www.jd.com

只返回URL，不要有任何解释："""
            
            # 调用AI生成URL
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个网站URL生成助手，专门用于根据网站名称生成官方网址。请只返回URL，不要有任何其他内容。"},
                    {"role": "user", "content": url_prompt}
                ],
                max_tokens=100,
                temperature=0.1,
                timeout=10
            )
            
            url = response.choices[0].message.content.strip()
            print(f"🔍 [URL生成] AI原始返回: '{url}'")
            
            # 清理可能的多余内容（如果AI返回了解释性文字）
            # 提取URL（查找http开头的部分）
            import re
            url_match = re.search(r'https?://[^\s]+', url)
            if url_match:
                url = url_match.group(0)
                # 移除末尾可能的标点符号
                url = url.rstrip('.,;:!?)。，；：！？）')
                print(f"🔍 [URL生成] 提取到URL: '{url}'")
            
            # 验证URL格式
            if url.startswith(("http://", "https://")):
                print(f"✅ [URL生成] 成功生成URL: {url}")
                return url
            else:
                print(f"⚠️ [URL生成] AI返回的不是有效URL: '{url}'")
                return None
                
        except Exception as e:
            print(f"❌ [URL生成] AI生成URL失败: {str(e)}")
            return None

    def _ai_identify_website_intent(self, user_input):
        """专门用于识别网页打开与自动化操作请求的AI方法"""
        try:
            # 检查是否有API密钥
            # 使用统一的LLM客户端获取方法（意图识别使用快速chat模型）
            result = self._get_llm_client(model="deepseek-chat")
            if not result:
                print("⚠️ 无法获取LLM客户端，无法进行网页打开与自动化操作意图识别")
                return None
            client, model = result
            
            # 构建专门的网页打开与自动化操作识别提示词
            website_prompt = f"""请分析用户输入，判断是否要打开网站。

用户输入：{user_input}

🎯 **你的任务**：
1. 判断是否是网站打开请求
2. 如果是，提取**网站名称**（不是整个句子）
3. 严格按照格式返回

📋 **返回格式**（只返回一行）：
- 打开网站 → website_open|网站名
- 不是网站 → not_website|

⚠️ **重要**：网站名只包含网站本身，不包含"打开"、"帮我"、"在浏览器"等词！

✅ **正确示例**：
- "帮我在浏览器打开哔哩哔哩" → website_open|哔哩哔哩
- "打开B站" → website_open|B站
- "在浏览器打开知乎" → website_open|知乎
- "访问github" → website_open|github
- "打开bilibili搜索python" → website_open|bilibili

❌ **错误示例**（不要这样）：
- "帮我在浏览器打开哔哩哔哩" → website_open|帮我在浏览器打开哔哩哔哩 ❌
- "打开B站" → website_open|打开B站 ❌

🔍 **特殊情况**：
- 浏览器搜索但没说网站名 → website_open|SEARCH_ENGINE
  例："在浏览器搜索Python" → website_open|SEARCH_ENGINE
- 不是打开请求 → not_website|
  例："什么是人工智能" → not_website|

现在请分析上面的用户输入，只返回一行结果："""
            
            # 调用AI进行网页打开与自动化操作意图识别
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个网页打开与自动化操作意图识别助手，专门用于判断用户是否想要打开网页或进行网页自动化操作。请严格按照格式返回结果。"},
                    {"role": "user", "content": website_prompt}
                ],
                max_tokens=30,
                temperature=0.1,
                timeout=10
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🔍 [网站意图识别] AI原始返回: '{result}'")
            print(f"🔍 [网站意图识别] 用户输入: '{user_input}'")
            
            # 解析结果
            if "|" in result:
                intent_type, site_name = result.split("|", 1)
                intent_type = intent_type.strip()
                site_name = site_name.strip()
                
                print(f"🔍 [网站意图识别] 解析结果 - 类型: '{intent_type}', 网站名: '{site_name}'")
                
                if intent_type == "website_open":
                    if not site_name:
                        print(f"⚠️ [网站意图识别] 网站名为空，使用原始输入")
                        return user_input
                    print(f"✅ [网站意图识别] 提取到网站名: '{site_name}'")
                    return site_name
                else:
                    print(f"❌ [网站意图识别] 非网站打开请求")
                    return None
            else:
                print(f"⚠️ [网站意图识别] AI返回格式错误，没有找到'|'分隔符")
                return None
                
        except Exception as e:
            print(f"AI网页打开与自动化操作意图识别失败: {str(e)}")
            # 如果AI调用失败，返回None
            return None

    def _ai_identify_app_launch_intent(self, user_input):
        """使用AI识别用户的应用启动意图"""
        try:
            # 使用统一的LLM客户端获取方法（意图识别使用快速chat模型）
            result = self._get_llm_client(model="deepseek-chat")
            if not result:
                print(f"⚠️ 无法获取LLM客户端，无法进行AI应用启动意图识别: {user_input}")
                return None
            client, model = result
            
            # 构建AI提示词，让AI智能判断用户意图类型
            intent_prompt = f"""请分析用户的输入，判断是否是应用启动请求：

用户输入：{user_input}

请判断用户是否想要启动或打开某个应用程序。

判断标准：
- 如果用户要求启动或打开应用程序，返回"app_launch|应用名称"
- 如果用户是在询问其他问题，返回"not_app|"

常见的应用启动表达：
- "打开XX"、"启动XX"、"运行XX"
- "帮我打开XX"、"帮我启动XX"、"帮我运行XX"
- "请打开XX"、"请启动XX"、"请运行XX"

支持的应用包括：
- 音乐应用：网易云音乐、QQ音乐、酷狗、酷我、Spotify
- 浏览器：Chrome、Edge、Firefox
- 办公软件：Word、Excel、PowerPoint
- 系统工具：记事本、计算器、画图、命令提示符、PowerShell

特别注意：
- "打开记事本" → "app_launch|记事本"
- "启动Chrome" → "app_launch|Chrome"
- "运行计算器" → "app_launch|计算器"
- "帮我打开Word" → "app_launch|Word"

请只返回结果，格式为：意图类型|应用名称
"""
            
            # 调用AI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个应用启动意图识别助手，专门用于分析用户的应用启动需求。"},
                    {"role": "user", "content": intent_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🔍 应用启动意图识别结果: {result}")
            
            # 解析AI返回的结果
            if "|" in result:
                intent_type, app_name = result.split("|", 1)
                if intent_type.strip() == "app_launch":
                    return ("app_launch", app_name.strip())
            
            # 如果AI识别失败，返回None
            return None
                
        except Exception as e:
            print(f"AI应用启动意图识别失败: {str(e)}")
            return None

    def _extract_search_keywords(self, user_input: str) -> dict:
        """
        从用户输入中提取搜索关键词，生成多个搜索问题
        返回: {"questions": [搜索问题列表], "urls": [提取的URL列表]}
        """
        try:
            # 首先检测用户输入中是否包含URL
            import re
            # 修改正则表达式，URL应该以非中文字符结束
            url_pattern = r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'
            urls = re.findall(url_pattern, user_input)
            
            # 清理URL末尾可能的标点符号
            urls = [url.rstrip('.,;:!?)。，；：！？）') for url in urls]
            
            if urls:
                print(f"🔗 检测到 {len(urls)} 个URL: {urls}")
            # 使用统一的LLM客户端获取方法（关键词提取使用快速chat模型）
            result = self._get_llm_client(model="deepseek-chat")
            if not result:
                print(f"⚠️ 无法获取LLM客户端，无法进行AI关键词提取，使用原始输入")
                return [user_input]
            client, model = result
            
            # 获取配置的最大搜索问题数
            max_questions = self.config.get("max_search_questions", 3)
            
            # 构建提示词
            prompt = f"""你是一个资料获取助手，专门用于根据用户输入生成适合在浏览器搜索框里搜索的完整问题。

用户输入: {user_input}

上下文信息: 
{self._get_recent_context()}

生成规则：
1. 根据用户输入和上下文信息，从不同角度生成适合搜索的完整问题
2. 如果用户输入包含代词（如"他"、"她"、"它"），结合上下文确定具体指代对象
3. 将简短的查询扩展为完整的搜索问题
4. 保留时间、地点、人物、事件等具体信息
5. 生成的问题应该能够直接在搜索引擎中使用
6. 根据问题的复杂程度，智能决定生成1到{max_questions}个不同角度的搜索问题
7. 每个问题应该从不同的角度获取信息，避免重复
8. 每行一个问题，不要添加编号或其他格式

示例1（简单问题，生成1个）：
输入："2025年93阅兵是什么"
输出：
2025年93阅兵是什么

示例2（需要多角度了解，生成多个）：
输入："户晨风是谁"
输出：
户晨风
户晨风 知乎
户晨风 百科
户晨风 争议

示例3（上下文相关，生成多个）：
输入："他现状如何"（上下文：之前讨论户晨风）
输出：
户晨风 2025
户晨风 近况
户晨风 最新动态

请根据用户输入和上下文，生成适合的搜索问题（最多{max_questions}个），每行一个问题："""
            
            # 调用AI进行关键词提取
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个资料获取助手，专门用于根据用户输入和上下文信息生成适合在浏览器搜索框里搜索的完整问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # 解析返回的搜索问题
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # 限制问题数量
            questions = questions[:max_questions]
            
            print(f"🔍 AI生成的搜索问题（共{len(questions)}个）:")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q}")
            
            # 返回搜索问题和检测到的URL
            result = {
                "questions": questions if questions else [user_input],
                "urls": urls
            }
            
            if urls:
                print(f"📌 将额外浏览 {len(urls)} 个URL")
            
            return result
                
        except Exception as e:
            print(f"⚠️ AI关键词提取失败: {str(e)}，使用原始输入")
            return {"questions": [user_input], "urls": urls if 'urls' in locals() else []}

    def _extract_domain(self, url: str) -> str:
        """
        从URL中提取域名
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # 移除www前缀
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url

    def _get_recent_context(self) -> str:
        """
        获取最近的对话上下文，用于关键词提取
        """
        try:
            # 获取最近的对话记录
            if hasattr(self, 'conversation_history') and self.conversation_history:
                # 获取最近3条对话记录
                recent_messages = self.conversation_history[-6:]  # 最近3轮对话（用户+AI）
                context_parts = []
                for message in recent_messages:
                    if isinstance(message, dict):
                        role = message.get('role', '')
                        content = message.get('content', '')
                        if role == 'user':
                            context_parts.append(f"用户: {content}")
                        elif role == 'assistant':
                            context_parts.append(f"AI: {content}")
                    else:
                        context_parts.append(str(message))
                
                return "\n".join(context_parts)
            else:
                return "无上下文信息"
        except Exception as e:
            print(f"⚠️ 获取上下文失败: {str(e)}")
            return "无上下文信息"

    def _optimize_search_content(self, search_content: str) -> str:
        """
        优化搜索内容，确保AI能够更好地理解
        """
        try:
            # 如果内容太短，直接返回
            if len(search_content) < 50:
                return search_content
            
            # 去除重复的空行
            lines = search_content.split('\n')
            optimized_lines = []
            prev_empty = False
            
            for line in lines:
                if line.strip():
                    optimized_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    optimized_lines.append(line)
                    prev_empty = True
            
            # 合并相邻的短行
            final_lines = []
            i = 0
            while i < len(optimized_lines):
                line = optimized_lines[i]
                if line.strip() and len(line.strip()) < 50 and i + 1 < len(optimized_lines):
                    next_line = optimized_lines[i + 1]
                    if next_line.strip() and len(next_line.strip()) < 50:
                        # 合并短行
                        combined = f"{line.strip()} {next_line.strip()}"
                        final_lines.append(combined)
                        i += 2
                    else:
                        final_lines.append(line)
                        i += 1
                else:
                    final_lines.append(line)
                    i += 1
            
            optimized_content = '\n'.join(final_lines)
            
            # 如果优化后内容太短，返回原始内容
            if len(optimized_content) < 30:
                return search_content
            
            return optimized_content
            
        except Exception as e:
            print(f"⚠️ 搜索内容优化失败: {str(e)}，返回原始内容")
            return search_content

    def _ai_identify_file_save_info(self, user_input, context_info):
        """统一的文件保存信息识别（整合路径、类型、命名）- 一次AI调用完成所有识别"""
        try:
            print(f"🔍 开始统一文件保存信息识别: {user_input}")
            
            # 使用统一的LLM客户端获取方法（使用用户选择的模型，因为需要更强的理解能力）
            result = self._get_llm_client()
            if not result:
                print("⚠️ 无法获取LLM客户端，无法使用AI智能识别")
                return None
            client, model = result
            
            # 构建统一的AI提示词
            prompt = f"""请分析用户的文件保存请求，一次性提供完整的文件保存信息。

用户请求：{user_input}

上下文信息（最近的对话内容）：
{context_info}

请智能分析并返回完整的文件保存信息：

       **分析要点：**
       1. **文件类型识别**：
          - 如果上下文有```java代码块 → file_type="java"
          - 如果上下文有```python代码块 → file_type="py"
          - 如果上下文有```cpp代码块 → file_type="cpp"
          - 如果上下文有音乐推荐 → file_type="txt"
          - 如果上下文有旅游攻略 → file_type="txt"
       
       2. **文件命名规则**：
          - 🔥 **Java代码**：必须从代码中提取类名！
            - 查找"public class XXX"中的XXX作为类名
            - 如代码有"public class Tetris" → filename="Tetris.java", title="Tetris"
            - 如代码有"public class HelloWorld" → filename="HelloWorld.java", title="HelloWorld"
          - Python代码：program.py 或根据功能命名
          - 音乐歌单：根据语言类型命名
            - 上下文提到"中文歌" → "中文歌单.txt"
            - 上下文提到"英文歌" → "英文歌单.txt"
            - 上下文提到"日文歌" → "日文歌单.txt"
            - 上下文提到"德语歌" → "德语歌单.txt"
            - 无法确定 → "音乐歌单.txt"
          - 旅游攻略：提取城市名（如"法兰克福旅游攻略.txt"）
       
       3. **保存路径识别**：
          - 用户说"保存到D盘" → "D:/"
          - 用户说"保存到C盘" → "C:/"
          - 用户说"D:\测试\" → "D:/测试/"
          - 未指定 → 不返回location字段（使用默认路径）
       
       4. **内容提取**：
          - 🚨 代码文件：content设置为空字符串""，系统会自动提取
          - 音乐歌单：提取AI回复中的音乐推荐内容
          - 旅游攻略：提取相关的攻略内容

返回JSON格式：
{{
    "file_type": "文件类型（java/py/cpp/txt等）",
    "title": "文件标题",
    "filename": "文件名.扩展名",
    "location": "保存路径（可选，如果用户未指定则不返回此字段）",
    "content": "文件内容（代码文件设为空字符串）"
}}

**示例：**
- 上下文有"public class Tetris"的Java代码，用户说"保存到D盘" →
  {{"file_type": "java", "title": "Tetris", "filename": "Tetris.java", "location": "D:/", "content": ""}}

- 上下文有"public class HelloWorld"的Java代码，用户说"帮我保存" →
  {{"file_type": "java", "title": "HelloWorld", "filename": "HelloWorld.java", "content": ""}}

- 上下文有中文歌推荐，用户说"帮我保存" →
  {{"file_type": "txt", "title": "中文歌单", "filename": "中文歌单.txt", "content": "音乐推荐内容"}}

🚨 注意：代码文件的content必须为空字符串""，系统会自动提取！
请只返回JSON，不要包含其他内容。
"""
            
            # 调用AI API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是文件保存专家，一次性提供完整的文件保存信息（类型、命名、路径、内容）。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
                timeout=60
            )
            
            # 提取AI响应
            ai_response = response.choices[0].message.content.strip()
            print(f"✅ 统一文件保存信息识别响应: {ai_response[:200]}...")
            
            # 解析JSON响应
            import json
            # 清理JSON字符串
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            ai_response = ai_response.strip()
            
            file_save_info = json.loads(ai_response)
            
            # 验证返回的信息
            if "filename" in file_save_info:
                print(f"✅ 统一识别成功: {file_save_info.get('filename')}")
                return file_save_info
            else:
                print(f"⚠️ AI返回的信息不完整")
                return None
                
        except Exception as e:
            print(f"❌ 统一文件保存信息识别失败: {str(e)}")
            return None


    def _ai_create_code_file_from_context(self, user_input):
        """使用AI通过上下文智能创建代码文件"""
        # 使用统一的LLM客户端获取方法
        result = self._get_llm_client()
        
        if result:
            client, model = result
            # 如果有API密钥，执行AI代码文件创建
            
            # 构建上下文信息
            context_info = ""
            if self.session_conversations:
                # 获取最近的对话作为上下文
                recent_contexts = []
                for conv in reversed(self.session_conversations[-5:]):  # 获取最近5条对话
                    recent_contexts.append(f"【{conv['timestamp']}】{conv['full_text']}")
                context_info = "\n".join(recent_contexts)
            
            # 尝试从上下文中提取代码内容
            extracted_code = self._extract_code_from_context(context_info)
            if extracted_code:
                context_info += f"\n\n【提取的代码内容】\n{extracted_code}"
                print(f"🔍 从上下文中提取到代码: {extracted_code[:100]}...")
            else:
                print("⚠️ 未从上下文中提取到代码内容")
                # 如果用户明确要求保存文件但没有找到代码，尝试从最近的对话中提取
                if "保存" in user_input.lower() or "创建" in user_input.lower():
                    print("🔍 尝试从最近的对话中提取代码内容...")
                    for conv in reversed(self.session_conversations[-3:]):
                        ai_response = conv.get("ai_response", "")
                        if "```" in ai_response:
                            extracted_code = self._extract_code_from_context(ai_response)
                            if extracted_code:
                                context_info += f"\n\n【从最近对话提取的代码内容】\n{extracted_code}"
                                print(f"🔍 从最近对话中提取到代码: {extracted_code[:100]}...")
                                break
            
            # 构建AI提示词
            prompt = f"""
请分析用户的代码创建请求，基于上下文信息智能生成代码文件。

用户输入：{user_input}

最近的对话上下文：
{context_info}

请分析用户想要创建什么类型的代码文件，并生成相应的代码。可能的代码类型包括：
1. Python代码 - 如果用户提到Python、py等
2. C++代码 - 如果用户提到C++、cpp等
3. COBOL代码 - 如果用户提到COBOL、cobol等
4. 其他编程语言代码

特别注意：
- 如果上下文中已经显示了代码内容（如```cobol...```），请直接使用该代码
- 如果用户说"创建测试文件"、"创建源文件"、"需要创建"、"保存这个文件"、"需要保存"或"地址在d盘"，请基于上下文中的代码创建文件
- 如果上下文中有COBOL代码，请创建.cob或.cbl文件
- 如果上下文中有Python代码，请创建.py文件
- 如果上下文中有C++代码，请创建.cpp文件
- 如果用户说"需要创建"，请基于上下文中最近的代码内容创建文件
- 如果用户说"地址在d盘"或"保存到d盘"，请将文件保存到D盘
- 如果用户说"保存这个文件"或"需要保存"，请基于上下文中最近的代码内容创建文件
- 如果用户说"路径为"，请使用用户指定的路径和文件名

请返回JSON格式：
{{
    "language": "编程语言",
    "title": "代码标题",
    "code": "完整的代码内容",
    "location": "保存位置（如D:/）",
    "filename": "文件名（如hello.cob）",
    "description": "代码说明"
}}

要求：
1. 代码要完整、可运行
2. 包含必要的注释和文档
3. 使用最佳实践
4. 文件名要符合编程语言规范
5. 保存位置默认为D盘
6. 如果是Hello World程序，要简单明了
7. 优先使用上下文中已有的代码内容
8. 如果用户明确指定了保存位置，请使用用户指定的位置
9. 如果用户说"保存这个文件"，请使用上下文中最近的代码内容
10. 如果用户说"路径为"，请使用用户指定的完整路径

如果无法确定要创建什么代码，请返回null。
"""
            
            # 调用AI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个代码生成助手，专门用于分析用户需求并生成相应的代码文件。请返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.7,
                timeout=240  # 延长AI文件创建的响应时间到240秒
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🔍 AI代码文件创建返回的原始结果: {result[:200]}...")
            
            # 解析JSON结果
            import json
            # 尝试清理JSON字符串
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            # 尝试解析JSON
            try:
                file_info = json.loads(result)
            except json.JSONDecodeError as json_error:
                print(f"AI代码文件创建JSON格式无效: {result}")
                print(f"JSON解析错误: {str(json_error)}")
                return None
            
            if file_info and "code" in file_info:
                # 提取文件信息
                language = file_info.get("language") or "txt"
                title = file_info.get("title", "未命名程序")
                code = file_info.get("code", "")
                location = file_info.get("location", "D:/")
                filename = file_info.get("filename") or f"program.{(language.lower() if isinstance(language, str) else 'txt')}"
                description = file_info.get("description", "")
                
                # 从用户输入中提取保存位置和文件名
                import re
                
                # 尝试提取完整路径（如"路径为D:/计算器.py"）
                path_match = re.search(r'路径为\s*([^，。\s]+)', user_input)
                if path_match:
                    full_path = path_match.group(1)
                    # 分离路径和文件名
                    if '/' in full_path or '\\' in full_path:
                        path_parts = full_path.replace('\\', '/').split('/')
                        if len(path_parts) > 1:
                            location = '/'.join(path_parts[:-1]) + '/'
                            filename = path_parts[-1]
                            if not filename.endswith(('.py', '.cob', '.cbl', '.cpp', '.txt')):
                                filename += '.py'  # 默认添加.py扩展名
                    else:
                        # 如果没有找到完整路径，使用原有的逻辑
                        if "d盘" in user_input.lower() or "d:" in user_input.lower():
                            location = "D:/"
                        elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                            location = "C:/"
                        elif "e盘" in user_input.lower() or "e:" in user_input.lower():
                            location = "E:/"
                        elif "f盘" in user_input.lower() or "f:" in user_input.lower():
                            location = "F:/"
                    
                    # 确保文件名安全
                    import re
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                    
                    # 构建完整的文件内容
                    if language.lower() == "cobol":
                        # COBOL代码格式特殊处理
                        if "IDENTIFICATION DIVISION" not in code:
                            file_content = "      IDENTIFICATION DIVISION.\n" + \
                                         "      PROGRAM-ID. " + title.upper().replace(' ', '-') + ".\n" + \
                                         "      PROCEDURE DIVISION.\n" + \
                                         code + "\n" + \
                                         "      STOP RUN.\n"
                        else:
                            # 如果代码已经包含完整的COBOL结构，直接使用
                            file_content = code
                    else:
                        # 其他编程语言
                        file_content = "# -*- coding: utf-8 -*-\n"
                        if description:
                            file_content += f'"""\n{description}\n"""\n\n'
                        file_content += code
                    
                    # 调用MCP工具创建文件
                    file_path = f"{location.rstrip('/')}/{filename}"
                    result = self.mcp_server.call_tool("write_file", 
                                                     file_path=file_path, 
                                                     content=file_content)
                    
                    return f"（指尖轻敲控制台）{result}"
                
        else:
            # 如果没有API密钥，返回None，使用后备方法
            return None

    def _ai_create_file_from_context(self, user_input):
        """使用AI通过上下文智能创建文件"""
        # 检查是否有API密钥
        model = self.config.get("selected_model", "deepseek-chat")
        api_key = self.config.get("deepseek_key", "") if "deepseek" in model else self.config.get("openai_key", "")
        
        if api_key:
            # 如果有API密钥，执行AI文件创建
            
            # 构建上下文信息 - 只关注与当前用户请求相关的内容
            context_info = ""
            relevant_content = ""
            
            # 分析用户当前请求的类型
            user_request_type = self._analyze_user_request_type(user_input)
            print(f"🔍 用户请求类型: {user_request_type}")
            
            # 如果是代码展示请求，不应该创建文件，应该返回None让AI直接展示代码
            if user_request_type == "code_display":
                print("ℹ️ 用户请求展示代码，不创建文件")
                return None
            
            if self.session_conversations:
                # 🔥 根据识别的内容类型，精准过滤上下文
                for conv in reversed(self.session_conversations[-3:]):  # 只获取最近3条对话
                    conv_text = conv.get('full_text', '')
                    ai_response = conv.get('ai_response', '')
                    
                    # 根据用户请求类型筛选相关内容
                    if user_request_type in ["code_file", "code"]:
                        # 代码文件：只包含有代码块的对话
                        if "```" in ai_response:
                            relevant_content += f"【{conv['timestamp']}】{conv_text}\n"
                            print(f"✅ 包含代码对话: {conv.get('user_input', '')[:30]}...")
                    elif user_request_type in ["music_file", "music"]:
                        # 音乐文件：只包含音乐推荐的对话
                        if any(kw in ai_response for kw in ["音乐", "歌", "歌曲", "推荐", "曲目", "歌单"]):
                            relevant_content += f"【{conv['timestamp']}】{conv_text}\n"
                            print(f"✅ 包含音乐对话: {conv.get('user_input', '')[:30]}...")
                    elif user_request_type in ["travel_file", "travel"]:
                        # 旅游文件：只包含旅游相关的对话
                        if any(kw in ai_response for kw in ["旅游", "旅行", "攻略", "景点"]):
                            relevant_content += f"【{conv['timestamp']}】{conv_text}\n"
                            print(f"✅ 包含旅游对话: {conv.get('user_input', '')[:30]}...")
                    elif user_request_type in ["note_file", "note"]:
                        # 笔记文件
                        if any(kw in ai_response for kw in ["笔记", "记录"]):
                            relevant_content += f"【{conv['timestamp']}】{conv_text}\n"
                    elif user_request_type in ["general_file", "general"]:
                        # 通用文件：包含最近的对话
                        relevant_content += f"【{conv['timestamp']}】{conv_text}\n"
                
                context_info = relevant_content.strip()
            
            # 尝试从相关上下文中提取代码内容
            if user_request_type in ["code_file", "code"]:
                print(f"🔍 准备提取代码，上下文长度: {len(context_info)}")
                print(f"🔍 上下文内容预览: {context_info[:200]}...")
                
                extracted_code = self._extract_code_from_context(context_info)
                if extracted_code:
                    context_info += f"\n\n【提取的代码内容】\n{extracted_code}"
                    print(f"🔍 从相关上下文中提取到代码: {extracted_code[:100]}...")
                else:
                    print("⚠️ 未从相关上下文中提取到代码内容")
                    # 尝试从所有最近对话中提取（不限于relevant_content）
                    print("🔄 尝试从所有最近对话中提取代码...")
                    for conv in reversed(self.session_conversations[-5:]):
                        full_text = conv.get('full_text', '')
                        if "```" in full_text:
                            extracted_code = self._extract_code_from_context(full_text)
                            if extracted_code:
                                context_info += f"\n\n【从全部对话提取的代码内容】\n{extracted_code}"
                                print(f"✅ 从全部对话中提取到代码: {extracted_code[:100]}...")
                                break
            else:
                print(f"ℹ️ 用户请求类型不是代码文件: {user_request_type}")
            
            # 构建AI提示词
            prompt = f"""
请分析用户的文件创建请求，基于用户当前的具体要求生成相应的文件内容。

用户当前请求：{user_input}
用户请求类型：{user_request_type}

相关上下文信息：
{context_info}

重要规则：
1. 🚀 当用户说"帮我保存"时，优先保存最近对话中生成的内容
2. 如果用户要求写代码，就生成代码文件，不要保存其他内容
3. 如果用户要求保存音乐推荐，就生成歌单文件
4. 如果用户要求保存旅游攻略，就生成旅游攻略文件
5. 严格根据用户当前请求的类型和内容来生成文件
6. 必须解析用户指定的保存路径，如果用户说"D:\测试_"，location就应该是"D:/测试_/"
7. 根据文件内容确定正确的文件扩展名，Python代码用.py，C++代码用.cpp等
8. 如果用户明确指定文件类型（如"保存为.py文件"），必须使用用户指定的扩展名
9. 如果用户说"保存为.py文件"，filename必须包含.py扩展名
10. 从上下文中提取相关代码内容，如果上下文中有Python代码，就保存为.py文件
11. **音乐歌单文件名规则**：根据上下文智能判断语言类型
    - 如果上下文提到"中文歌"或包含中文歌曲名，filename为"中文歌单.txt"
    - 如果上下文提到"英文歌"或包含英文歌曲名，filename为"英文歌单.txt"
    - 如果上下文提到"日文歌"或包含日文歌曲名，filename为"日文歌单.txt"
    - 如果上下文提到"德语歌"或包含德语歌曲名，filename为"德语歌单.txt"
    - 如果无法确定语言，使用"音乐歌单.txt"

请返回JSON格式：
{{
    "file_type": "文件类型（folder/txt/py/cpp/java等）",
    "title": "文件标题",
    "content": "文件内容（代码文件设置为空字符串，系统会自动提取）",
    "location": "保存路径（可选，如E:/、D:/等，如果用户没有指定则不返回此字段）",
    "filename": "文件名（如xxx.py）或文件夹名（如xxx/）"
}}

🚨 重要说明：
- **代码文件（Java/Python/C++）**：content设置为""（空字符串），系统会自动从代码块提取纯代码
- **Java代码**：title必须是从代码中提取的类名（如Tetris、HelloWorld、Calculator）
- **文本文件（歌单/攻略）**：可以在content中包含简短描述
- 如果用户明确指定了保存路径（如"保存到E盘"、"保存到D盘"），请在location字段中返回对应的路径；如果没有指定，则不返回location字段

要求：
1. 文件内容必须与用户当前请求完全匹配
2. 标题要简洁明了，反映用户的实际需求
3. 如果是文件夹，content字段为空，filename以/结尾
4. **如果是代码文件，必须从代码块（```xxx）中提取纯代码，不要包含任何说明文字**
5. **文件扩展名必须与代码语言匹配：Java→.java, Python→.py, C++→.cpp**
6. 如果是歌单文件，要包含完整的歌曲信息
7. 如果是旅游攻略，要包含详细的旅游信息
8. 🚀 智能路径处理：如果用户明确指定了保存路径（如"保存到E盘"），在location字段中返回对应路径；如果没有指定，则不包含location字段
9. 文件名要符合Windows命名规范，扩展名要正确
10. 绝对不要保存与用户当前请求无关的内容

特别注意：
- 🚀 当用户说"帮我保存"时，分析最近对话内容，智能判断要保存什么类型的文件
- 如果上下文中包含旅游攻略、景点介绍、行程安排，就保存为旅游攻略文件(.txt)
- 如果上下文中包含音乐推荐、歌曲列表、歌单，就保存为歌单文件(.txt)
- **如果上下文中包含代码块（```python、```java、```cpp等），必须：**
  - **只提取代码块内的代码，不要包含"以下是代码"等说明**
  - **file_type设置为对应语言（java/python/cpp）**
  - **filename必须以对应扩展名结尾（.java/.py/.cpp）**
  - **content只包含纯代码，不包含markdown标记和说明文字**
- 如果上下文中包含笔记、记录、总结，就保存为笔记文件(.txt)
- 如果上下文中包含计划、安排、清单，就保存为计划文件(.txt)
- 如果用户明确指定了文件类型（如"保存为.py文件"），必须使用用户指定的扩展名
- 如果用户明确指定了保存路径（如"保存到E盘"），在location字段中返回对应路径
- 绝对不要返回null，必须根据用户请求和上下文内容生成文件

如果无法确定要创建什么文件，请返回null。
"""
            
            # 设置API客户端
            if "deepseek" in model:
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            else:
                client = openai.OpenAI(api_key=api_key)
            
            # 调用AI
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个文件创建助手，专门用于分析用户需求并生成相应的文件内容。请返回JSON格式的结果。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=240  # 延长AI文件创建的响应时间到240秒
                )
                
                result = response.choices[0].message.content.strip()
                print(f"🔍 AI文件创建返回的原始结果: {result[:200]}...")
                
                # 检查AI返回的结果是否为空
                if not result or result.strip() == "":
                    print("⚠️ AI返回空结果，使用简单解析")
                    file_info = self._simple_parse_file_info(user_input, context_info)
                else:
                    # 解析JSON结果
                    try:
                        import json
                        # 尝试清理JSON字符串
                        result = result.strip()
                        if result.startswith('```json'):
                            result = result[7:]
                        if result.endswith('```'):
                            result = result[:-3]
                        result = result.strip()
                        
                        file_info = json.loads(result)
                        
                    except json.JSONDecodeError as json_error:
                        print(f"⚠️ JSON解析失败，使用简单解析: {str(json_error)}")
                        file_info = self._simple_parse_file_info(user_input, context_info)

            except Exception as e:
                print(f"❌ AI 文件创建阶段异常: {e}")
                file_info = None

            if file_info and file_info.get("title") is not None and file_info.get("content") is not None:
                # 提取文件信息
                file_type = file_info.get("file_type") or "txt"
                title = file_info.get("title") or "未命名文件"
                content = file_info.get("content") or ""
                location = file_info.get("location") or ""
                filename = file_info.get("filename") or f"{title}.txt"

                # 🔥 智能内容处理：代码文件从上下文提取，文本文件使用AI回复
                if not content:
                    # 检查是否是代码文件（Java/Python/C++等）
                    code_extensions = ['.java', '.py', '.cpp', '.c', '.js', '.html', '.css', '.cob', '.cbl']
                    is_code_file = any(filename.endswith(ext) for ext in code_extensions)
                    
                    if is_code_file:
                        # 🔥 代码文件：从上下文中提取纯代码
                        print(f"🔍 准备提取代码，上下文长度: {len(context_info)}")
                        print(f"🔍 上下文内容预览: {context_info[:200]}...")
                        
                        extracted_code = self._extract_code_from_context(context_info)
                        if extracted_code:
                            content = extracted_code
                            print(f"🔍 从相关上下文中提取到代码: {content[:100]}...")
                        else:
                            print("⚠️ 未能从上下文提取代码，使用AI返回的内容")
                            if self.session_conversations:
                                last_conv = self.session_conversations[-1]
                                content = last_conv.get('ai_response') or ""
                    else:
                        # 文本文件：使用AI完整回复
                        if self.session_conversations:
                            last_conv = self.session_conversations[-1]
                            content = last_conv.get('ai_response') or ""

                # 文件名后缀兜底：文字类默认.txt（代码类型已有专属后缀时不覆盖）
                if file_type != "folder" and ("." not in filename):
                    filename = f"{filename}.txt"

                # 🚀 智能路径处理：优先使用AI返回的路径，否则使用默认路径
                default_path = self.config.get("default_save_path", "D:/露尼西亚文件/")
                
                # 检查AI是否返回了用户指定的路径
                if location and (location.startswith("D:/") or 
                                 location.startswith("C:/") or
                                 location.startswith("E:/") or
                                 location.startswith("F:/") or
                                 location.startswith("G:/") or
                                 location.startswith("H:/")):
                    # AI返回了用户指定的路径，使用它
                    print(f"🔍 使用AI返回的用户指定路径: {location}")
                else:
                    # AI没有返回路径，使用默认保存路径
                    location = default_path
                    print(f"🔍 使用默认保存路径: {default_path}")
                    
                    # 确保默认路径存在
                    if not os.path.exists(default_path):
                        try:
                            os.makedirs(default_path, exist_ok=True)
                            print(f"✅ 创建默认保存路径: {default_path}")
                        except Exception as e2:
                            print(f"WARNING: {str(e2)}")
                            location = "D:/"
                            print(f"INFO: {location}")
                            print(f"INFO: Using fallback path: {location}")
                
                print(f"✅ 最终保存路径: {location}")
                
                # 确保文件名安全
                import re
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                
                # 调用MCP工具创建文件或文件夹
                if file_type == "folder":
                    # 创建文件夹
                    folder_path = f"{location.rstrip('/')}/{filename}"
                    print(f"🔍 创建文件夹: {folder_path}")
                    result = self.mcp_server.call_tool("create_folder", 
                                                       folder_path=folder_path)
                elif "create_note" in user_input.lower() or "笔记" in user_input:
                    # 创建笔记
                    print(f"🔍 创建笔记: {filename} 在 {location}")
                    result = self.mcp_server.call_tool("create_note", 
                                                       title=title, 
                                                       content=content, 
                                                       filename_format="simple", 
                                                       location=location)
                else:
                    # 创建普通文件
                    file_path = f"{location.rstrip('/')}/{filename}"
                    print(f"🔍 创建文件: {file_path}")
                    print(f"🔍 文件内容长度: {len(content)} 字符")
                    print(f"🔍 文件标题: {title}")
                    print(f"🔍 文件名: {filename}")
                    print(f"🔍 保存位置: {location}")
                    print(f"🔍 路径来源: {'AI返回' if location and location != self.config.get('default_save_path', 'D:/露尼西亚文件/') else '默认路径'}")
                    print(f"🔍 文件类型: {file_type}")
                    
                    result = self.mcp_server.call_tool("write_file", 
                                                       file_path=file_path, 
                                                       content=content)
                    
                    print(f"✅ 文件创建结果: {result}")
                    return f"（指尖轻敲控制台）{result}"
                
            # 如果上面没有得到有效的 file_info，则使用简单解析作为后备方案
            if not (file_info and file_info.get("title") is not None and file_info.get("content") is not None):
                print("🔄 使用简单解析作为后备方案")
                file_info = self._simple_parse_file_info(user_input, context_info)

                if file_info and file_info.get("title") is not None and file_info.get("content") is not None:
                    # 提取文件信息
                    file_type = file_info.get("file_type") or "txt"
                    title = file_info.get("title") or "未命名文件"
                    content = file_info.get("content") or ""
                    location = file_info.get("location") or ""
                    filename = file_info.get("filename") or f"{title}.txt"

                    # 内容兜底：若内容为空，使用最近一条AI完整回复
                    if (not content) and self.session_conversations:
                        last_conv = self.session_conversations[-1]
                        content = last_conv.get('ai_response') or content or ""

                    # 文件名后缀兜底：文字类默认.txt（代码类型已有专属后缀时不覆盖）
                    if file_type != "folder" and ("." not in filename):
                        filename = f"{filename}.txt"
                    
                    # 🚀 智能路径处理：优先使用用户指定路径，否则使用默认路径
                    default_path = self.config.get("default_save_path", "D:/露尼西亚文件/")
                    
                    # 检查是否有用户指定的路径
                    if location and (location.startswith("D:/") or 
                                   location.startswith("C:/") or
                                   location.startswith("E:/") or
                                   location.startswith("F:/") or
                                   location.startswith("G:/") or
                                   location.startswith("H:/")):
                        # 用户指定了路径，使用它
                        print(f"🔍 使用用户指定路径: {location}")
                    else:
                        # 没有指定路径，使用默认保存路径
                        location = default_path
                        print(f"🔍 使用默认保存路径: {default_path}")
                        
                        # 确保默认路径存在
                        if not os.path.exists(default_path):
                            try:
                                os.makedirs(default_path, exist_ok=True)
                                print(f"✅ 创建默认保存路径: {default_path}")
                            except Exception as e:
                                print(f"⚠️ 创建默认路径失败: {str(e)}")
                                # 如果创建失败，使用D盘根目录
                                location = "D:/"
                                print(f"🔄 使用后备路径: {location}")
                        
                        print(f"✅ 最终保存路径: {location}")
                        
                        # 确保文件名安全
                        import re
                        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                        
                        # 调用MCP工具创建文件
                        file_path = f"{location.rstrip('/')}/{filename}"
                        print(f"🔍 后备方案创建文件: {file_path}")
                        print(f"🔍 后备方案文件内容长度: {len(content)} 字符")
                        print(f"🔍 后备方案文件标题: {title}")
                        print(f"🔍 后备方案文件名: {filename}")
                        print(f"🔍 后备方案保存位置: {location}")
                        print(f"🔍 后备方案路径来源: {'用户指定' if location and location != self.config.get('default_save_path', 'D:/露尼西亚文件/') else '默认路径'}")
                        
                        result = self.mcp_server.call_tool("write_file", 
                                                         file_path=file_path, 
                                                         content=content)
                        
                        print(f"✅ 后备方案文件创建结果: {result}")
                        return f"（指尖轻敲控制台）{result}"
                else:
                    return None
                
        else:
            # 如果没有API密钥，返回None，使用后备方法
            return None
    
    def _fallback_create_note(self, user_input):
        """后备笔记创建方法（原有的固定格式）"""
        try:
            # 智能提取标题和内容
            import re
            
            # 检查是否是文件夹创建请求
            folder_keywords = ["文件夹", "目录", "文件夹", "创建文件夹", "新建文件夹", "建立文件夹"]
            if any(keyword in user_input.lower() for keyword in folder_keywords):
                # 提取文件夹名称
                folder_name = None
                folder_patterns = [
                    r'叫\s*["\']([^"\']+)["\']',
                    r'名为\s*["\']([^"\']+)["\']',
                    r'名称\s*["\']([^"\']+)["\']',
                    r'文件夹\s*["\']([^"\']+)["\']',
                    r'目录\s*["\']([^"\']+)["\']',
                ]
                
                for pattern in folder_patterns:
                    match = re.search(pattern, user_input)
                    if match:
                        folder_name = match.group(1)
                        break
                
                if not folder_name:
                    # 如果没有找到明确的文件夹名，使用默认名称
                    folder_name = "新建文件夹"
                
                # 提取保存位置
                location = "D:/"
                location_patterns = [
                    r'位置在\s*([^，。\s]+)',
                    r'位置\s*是\s*([^，。\s]+)',
                    r'保存到\s*([^，。\s]+)',
                    r'保存在\s*([^，。\s]+)',
                    r'创建在\s*([^，。\s]+)',
                    r'(D[:\/\\])',
                    r'(C[:\/\\])',
                    r'(E[:\/\\])',
                ]
                
                for pattern in location_patterns:
                    match = re.search(pattern, user_input)
                    if match:
                        location = match.group(1)
                        if not location.endswith('/') and not location.endswith('\\'):
                            location += '/'
                        break
                
                # 创建文件夹
                folder_path = f"{location.rstrip('/')}/{folder_name}"
                result = self.mcp_server.call_tool("create_folder", folder_path=folder_path)
                return f"（指尖轻敲控制台）{result}"
            
            # 1. 从用户输入中提取标题
            title = None
            
            # 检查是否包含歌单相关关键词
            if any(keyword in user_input.lower() for keyword in ["歌单", "音乐", "歌曲", "playlist", "music"]):
                # 使用默认标题，让主Agent的AI动态生成内容
                title = "音乐歌单"
            
            # 检查是否包含其他类型的笔记
            elif "出行" in user_input or "计划" in user_input:
                title = "出行计划"
            elif "天气" in user_input:
                title = "天气记录"
            elif "代码" in user_input or "程序" in user_input:
                title = "代码笔记"
            else:
                # 尝试从用户输入中提取标题
                title_patterns = [
                    r'标题为\s*["\']([^"\']+)["\']',
                    r'标题\s*["\']([^"\']+)["\']',
                    r'标题是\s*["\']([^"\']+)["\']',
                    r'文件名叫\s*["\']([^"\']+)["\']',
                    r'文件名\s*["\']([^"\']+)["\']',
                ]
                
                for pattern in title_patterns:
                    match = re.search(pattern, user_input)
                    if match:
                        title = match.group(1)
                        break
                
                # 如果没有找到，尝试提取关键词作为标题
                if not title:
                    keywords = ["歌单", "笔记", "计划", "记录", "清单"]
                    for keyword in keywords:
                        if keyword in user_input:
                            title = f"{keyword}笔记"
                            break
            
            # 2. 从上下文和用户输入中提取内容
            content = ""
            
            # 检查最近的对话中是否有歌单内容
            if title and "歌单" in title:
                # 从最近的对话中查找歌单内容
                for conv in reversed(self.session_conversations[-5:]):  # 检查最近5条对话
                    ai_response = conv.get("ai_response", "")
                    if any(keyword in ai_response for keyword in ["**", "*", "《", "》", "-", "1.", "2.", "3."]):
                        # 这可能是歌单内容
                        content = ai_response
                        break
            
            # 如果没有找到内容，尝试从用户输入中提取
            if not content:
                content_patterns = [
                    r'内容为\s*["\']([^"\']+)["\']',
                    r'内容\s*["\']([^"\']+)["\']',
                ]
                
                for pattern in content_patterns:
                    match = re.search(pattern, user_input)
                    if match:
                        content = match.group(1)
                        break
            
            # 3. 提取位置信息
            location = None
            location_patterns = [
                r'位置在\s*([^，。\s]+)',
                r'位置\s*是\s*([^，。\s]+)',
                r'保存到\s*([^，。\s]+)',
                r'保存在\s*([^，。\s]+)',
                r'创建在\s*([^，。\s]+)',
                r'帮我保存到\s*([^，。\s]+)',
                r'(D[:\/\\])',
                r'(C[:\/\\])',
                r'(E[:\/\\])',
                r'(F[:\/\\])'
            ]
            
            print(f"🔍 开始提取路径，用户输入: {user_input}")
            
            for i, pattern in enumerate(location_patterns):
                match = re.search(pattern, user_input)
                if match:
                    print(f"🔍 模式 {i+1} 匹配成功: {pattern}")
                    print(f"🔍 匹配结果: {match.group(0)}")
                    location = match.group(1) if match.group(1) else "D:/"
                    print(f"🔍 提取的路径: {location}")
                    break
                else:
                    print(f"🔍 模式 {i+1} 不匹配: {pattern}")
            
            # 如果没有找到位置，默认使用D盘
            if not location:
                location = "D:/"
                print(f"🔍 未找到路径，使用默认值: {location}")
            
            # 🚀 标准化路径格式，确保盘符后面有斜杠
            if location and len(location) == 1 and location in ['D', 'C', 'E', 'F']:
                location = f"{location}:/"
                print(f"🔍 标准化路径格式: {location}")
            
            print(f"🔍 最终路径: {location}")
            
            # 4. 如果找到了标题但没有内容，生成默认内容
            if title and not content:
                if "中文歌单" in title:
                    content = """# 中文歌单精选

## 经典流行系列
1. 《七里香》- 周杰伦
   - 夏日怀旧风格，适合夜间放松聆听
2. 《小幸运》- 田馥甄
   - 温暖抒情曲目，情绪舒缓

## 影视金曲推荐
3. 《光年之外》- G.E.M.邓紫棋
   - 电影主题曲，富有感染力
4. 《追光者》- 岑宁儿
   - 温柔治愈系，适合安静环境

## 民谣与独立音乐
5. 《成都》- 赵雷
   - 城市民谣，叙事性强
6. 《理想三旬》- 陈鸿宇
   - 民谣风格，适合深夜沉思

创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
用途：指挥官的中文音乐收藏"""
                elif "英文歌单" in title:
                    content = """# English Music Playlist

## Contemporary Pop Selection
1. *Flowers* - Miley Cyrus
   - 2023 hit single, mood uplifting
2. *Cruel Summer* - Taylor Swift
   - Upbeat summer-themed track

## Electronic & Dance
3. *Cold Heart (PNAU Remix)* - Elton John & Dua Lipa
   - Cross-generational collaboration
4. *Don't Start Now* - Dua Lipa
   - Energetic dance track for pre-departure

## Alternative Recommendations
5. *As It Was* - Harry Styles
   - Pop-rock with retro synth elements
6. *Blinding Lights* - The Weeknd
   - 80s-style synthwave masterpiece

Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Purpose: Commander's English music collection"""
                elif "德语歌单" in title:
                    content = """# 德语夜间歌单

## 经典德文歌曲
1. **《Das Liebeslied》- Annett Louisan**
   - 轻柔民谣风格，适合安静环境
2. **《Ohne dich》- Rammstein**
   - 工业金属乐队的情歌，情感深沉
3. **《Auf uns》- Andreas Bourani**
   - 励志流行曲，旋律积极

## 现代德文流行
4. **《Chöre》- Mark Forster**
   - 流行摇滚，节奏明快但不过于激烈
5. **《Musik sein》- Wincent Weiss**
   - 轻快流行，适合放松
6. **《99 Luftballons》- Nena**
   - 经典反战歌曲，合成器流行风格

## 推荐聆听时段
- 最佳时间：22:00-24:00
- 适合场景：夜间放松、学习德语

创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
用途：指挥官的德语音乐收藏"""
                else:
                    content = f"# {title}\n\n这是一个{title}，创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 5. 调用工具创建笔记
            if title:
                # 检查是否是代码保存请求
                if "代码" in title or "程序" in title:
                    # 从上下文中提取代码内容
                    extracted_code = self._extract_code_from_context(" ".join([conv["full_text"] for conv in self.session_conversations[-3:]]))
                    if extracted_code:
                        content = f"# {title}\n\n```cpp\n{extracted_code}\n```\n\n创建时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    # 从用户输入中提取具体路径
                    import re
                    path_match = re.search(r'保存到\s*([^，。\s]+)', user_input)
                    if path_match:
                        specific_path = path_match.group(1)
                        # 构建完整路径
                        if specific_path.endswith('\\') or specific_path.endswith('/'):
                            file_path = f"{specific_path}{title}.txt"
                        else:
                            file_path = f"{specific_path}\\{title}.txt"
                        
                        # 使用write_file工具直接创建文件
                        try:
                            result = self.mcp_server.call_tool("write_file", file_path=file_path, content=content)
                            return f"(RESULT){result}"
                        except Exception as e:
                            return f"(ERROR){str(e)}"
                
                # 获取文件名格式设置
                filename_format = self.config.get("note_filename_format", "simple")
                result = self.mcp_server.call_tool("create_note", title=title, content=content, filename_format=filename_format, location=location)
                return f"（指尖轻敲控制台）{result}"
            else:
                return f"（微微皱眉）抱歉指挥官，无法确定笔记标题。请明确说明要创建什么类型的笔记。"
                
        except Exception as e:
            return f"（微微皱眉）抱歉指挥官，创建笔记时遇到了问题：{str(e)}"

    def _handle_security_testing(self, user_input):
        """处理安全测试请求 - 功能已暂时移除"""
        return "（抱歉地摇头）指挥官，安全测试功能已暂时移除。\n\n该功能正在重新开发中，敬请期待后续版本。\n\n💡 您可以继续使用其他功能，如文件分析、网络搜索等。"
    
    # 安全测试方法已移除
    
    # 安全测试回调方法已移除
    
    # 智能安全测试方法已移除
    # Playwright网页导航与交互操作已移除（功能已整合到网页打开与自动化操作中）
    
    def _extract_search_query_from_text(self, text):
        """从文本中提取搜索查询"""
        import re
        
        # 尝试提取引号中的内容
        quote_patterns = [
            r'搜索["""](.*?)["""]',
            r'查找["""](.*?)["""]',
            r'查询["""](.*?)["""]',
            r'搜索(.*?)(?:\s|$)',
            r'查找(.*?)(?:\s|$)',
            r'查询(.*?)(?:\s|$)'
        ]
        
        for pattern in quote_patterns:
            match = re.search(pattern, text)
            if match:
                query = match.group(1).strip()
                if query and len(query) > 1:
                    return query

        return None

    # 以下是残留的安全测试代码，已全部删除
    
    def _detect_web_port(self, target: str) -> int:
        """检测Web端口"""
        common_web_ports = [80, 443, 8080, 8443, 9000, 3000]
        
        for port in common_web_ports:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((target, port))
                sock.close()
                
                if result == 0:
                    print(f"🔍 检测到开放端口: {port}")
                    return port
            except:
                continue
        
        # 默认返回80端口
        return 80
    
    def _extract_full_url_from_input(self, user_input: str) -> str:
        """从用户输入中提取完整URL"""
        import re
        
        # 更精确的URL匹配，只匹配英文域名和端口，不包含中文
        url_pattern = r'https?://[a-zA-Z0-9.-]+(?::\d+)?(?:/[a-zA-Z0-9._~:/?#[\]@!$&\'()*+,;=-]*)?'
        urls = re.findall(url_pattern, user_input)
        
        if urls:
            # 取第一个匹配的URL，并清理
            url = urls[0]
            # 移除URL中可能包含的额外字符
            url = re.sub(r'[^\x00-\x7F]+.*$', '', url)  # 移除第一个非ASCII字符及其后面的所有内容
            url = url.rstrip('/')  # 移除末尾的斜杠
            print(f"🔍 从输入中提取到完整URL: {url}")
            return url

        return None

    # 传统安全测试方法已被智能模式完全替代
    
    def _execute_security_commands(self, commands):
        """执行安全测试命令"""
        print(f"⚡ 开始执行{len(commands)}个安全测试命令...")
        
        results = []
        response = "🔒 **安全测试报告**\n\n"
        
        total_time = 0
        for i, cmd_info in enumerate(commands, 1):
            command = cmd_info['command']
            description = cmd_info['description']
            estimated_time = cmd_info.get('estimated_time', '未知')
            
            response += f"**步骤 {i}: {description}**\n"
            response += f"命令: `{command}`\n"
            response += f"预计时间: {estimated_time}\n\n"
            
            # 执行命令
            result = self.kali_controller.execute_command(command)
            results.append(result)
            
            if result['success']:
                response += "✅ 执行成功\n"
                # 简化输出，只显示关键信息
                stdout = result['stdout'][:500] + "..." if len(result['stdout']) > 500 else result['stdout']
                if stdout.strip():
                    response += f"```\n{stdout}\n```\n"
            else:
                response += f"❌ 执行失败: {result.get('error', '未知错误')}\n"
                stderr = result.get('stderr', '')
                if stderr:
                    response += f"错误信息: {stderr[:200]}\n"
                    
                    # 针对常见错误提供解决建议
                    if "域名解析" in stderr or "Name or service not known" in stderr:
                        response += "💡 **建议**: 目标域名无法解析，可能的原因：\n"
                        response += "• 域名不存在或已过期\n"
                        response += "• DNS服务器配置问题\n"
                        response += "• 网络连接问题\n"
                        response += f"• 尝试使用 `nslookup {cmd_info.get('command', '').split()[-1]}` 验证域名\n"
                    elif "requires root privileges" in stderr:
                        response += "💡 **建议**: 当前用户权限不足，建议：\n"
                        response += "• 使用sudo运行需要root权限的命令\n"
                        response += "• 或使用不需要root权限的扫描选项\n"
                        response += "• 考虑切换到root用户进行高级扫描\n"
                    elif "Network is unreachable" in stderr:
                        response += "💡 **建议**: 网络不可达，检查：\n"
                        response += "• 目标主机是否在线\n"
                        response += "• 防火墙是否阻止了连接\n"
                        response += "• 路由配置是否正确\n"
            
            response += "\n" + "-"*50 + "\n\n"
            total_time += result.get('execution_time', 0)
        
        # 生成详细分析报告
        if results:
            response += "📊 **详细分析报告**\n\n"
            
            for i, result in enumerate(results, 1):
                if not result.get("success", False):
                    continue
                    
                tool = result.get("command", "").split()[0]
                response += f"### 🔍 工具 {i}: {tool.upper()}\n\n"
                
                # 分析不同工具的输出
                if tool == "nmap":
                    nmap_analysis = self._analyze_nmap_output(result.get("stdout", ""))
                    
                    # 目标信息
                    if nmap_analysis.get("target_ip"):
                        response += f"**🎯 目标信息**\n"
                        response += f"• 目标: {nmap_analysis['target_ip']}\n"
                    
                    # 主机状态
                    if nmap_analysis.get("host_status"):
                        response += f"• 状态: {nmap_analysis['host_status']}\n"
                    
                    # 警告信息
                    if nmap_analysis.get("warnings"):
                        response += f"\n⚠️ **警告**\n"
                        for warning in nmap_analysis["warnings"]:
                            response += f"• {warning}\n"
                    
                    # 开放端口详情
                    if nmap_analysis.get("ports"):
                        open_ports = [p for p in nmap_analysis["ports"] if p["state"] == "open"]
                        closed_ports = [p for p in nmap_analysis["ports"] if p["state"] == "closed"]
                        filtered_ports = [p for p in nmap_analysis["ports"] if p["state"] == "filtered"]
                        
                        response += f"\n**🔌 端口扫描结果**\n"
                        
                        if open_ports:
                            response += f"✅ **开放端口** ({len(open_ports)}个):\n"
                            for port in open_ports:
                                response += f"• {port['port']}/{port['protocol']} - {port['service']} (开放)\n"
                        else:
                            response += "🔒 **开放端口**: 无开放端口发现\n"
                        
                        if closed_ports:
                            response += f"\n❌ **关闭端口** ({len(closed_ports)}个):\n"
                            for port in closed_ports[:5]:  # 只显示前5个
                                response += f"• {port['port']}/{port['protocol']} - {port['service']} (关闭)\n"
                            if len(closed_ports) > 5:
                                response += f"• ... 还有 {len(closed_ports) - 5} 个关闭端口\n"
                        
                        if filtered_ports:
                            response += f"\n🛡️ **过滤端口** ({len(filtered_ports)}个):\n"
                            for port in filtered_ports[:3]:  # 只显示前3个
                                response += f"• {port['port']}/{port['protocol']} - {port['service']} (被过滤)\n"
                    else:
                        response += f"\n🔍 **端口扫描结果**: 未发现开放端口或主机无响应\n"
                    
                    # 操作系统信息
                    if nmap_analysis.get("os_info"):
                        response += f"\n💻 **操作系统**: {nmap_analysis['os_info']}\n"
                    
                    # 扫描统计
                    if nmap_analysis.get("scan_stats"):
                        stats = nmap_analysis["scan_stats"]
                        if stats.get("duration"):
                            response += f"\n⏱️ **扫描耗时**: {stats['duration']:.2f}秒\n"
                        if stats.get("summary"):
                            response += f"📈 **扫描统计**: {stats['summary']}\n"
                
                elif tool == "nikto":
                    nikto_analysis = self._analyze_nikto_output(result.get("stdout", ""))
                    if nikto_analysis.get("vulnerabilities"):
                        response += f"**🚨 发现的漏洞**\n"
                        for vuln in nikto_analysis["vulnerabilities"]:
                            response += f"• {vuln['description']}\n"
                    if nikto_analysis.get("server_info"):
                        response += f"**🌐 服务器信息**: {nikto_analysis['server_info']}\n"
                
                elif tool == "curl":
                    curl_output = result.get("stdout", "")
                    if curl_output.strip():
                        response += f"**🌐 HTTP响应分析**\n"
                        response += f"```\n{curl_output}\n```\n"
                        
                        # 详细分析HTTP响应
                        if "HTTP/" in curl_output:
                            # 提取状态码
                            status_line = [line for line in curl_output.split('\n') if 'HTTP/' in line]
                            if status_line:
                                response += f"📊 **状态**: {status_line[0].strip()}\n"
                        
                        # CDN检测
                        if "cloudflare" in curl_output.lower():
                            response += "🛡️ **CDN**: Cloudflare 防护已确认\n"
                        if "cf-ray" in curl_output.lower():
                            cf_ray = [line for line in curl_output.split('\n') if 'cf-ray' in line.lower()]
                            if cf_ray:
                                response += f"🔍 **Cloudflare Ray ID**: {cf_ray[0].strip()}\n"
                        
                        # 服务器信息
                        server_line = [line for line in curl_output.split('\n') if line.lower().startswith('server:')]
                        if server_line:
                            response += f"🖥️ **服务器**: {server_line[0].split(':', 1)[1].strip()}\n"
                        
                        # 安全头分析
                        if "x-frame-options" in curl_output.lower():
                            response += "✅ **安全头**: X-Frame-Options 已配置\n"
                        if "strict-transport-security" in curl_output.lower():
                            response += "✅ **安全头**: HSTS 已启用\n"
                        if "content-security-policy" in curl_output.lower():
                            response += "✅ **安全头**: CSP 已配置\n"
                        
                        # 检测可能的绕过成功
                        if "200 OK" in curl_output:
                            response += "🎯 **突破成功**: 获得HTTP 200响应！\n"
                        elif "403" in curl_output:
                            response += "🚫 **被阻止**: HTTP 403 - 访问被拒绝\n"
                        elif "404" in curl_output:
                            response += "❓ **未找到**: HTTP 404 - 资源不存在\n"
                    else:
                        response += f"**🌐 HTTP请求失败**\n"
                        response += "• 连接超时或被阻止\n"
                
                elif tool == "dig":
                    dig_output = result.get("stdout", "")
                    if dig_output.strip():
                        response += f"**🔍 DNS记录分析**\n"
                        response += f"```\n{dig_output}\n```\n"
                
                response += "\n" + "-"*60 + "\n\n"
            
            # 总体安全评估
            all_ports = []
            all_warnings = []
            for result in results:
                if result.get("success") and result.get("command", "").startswith("nmap"):
                    nmap_analysis = self._analyze_nmap_output(result.get("stdout", ""))
                    all_ports.extend(nmap_analysis.get("ports", []))
                    all_warnings.extend(nmap_analysis.get("warnings", []))
            
            response += "🛡️ **安全评估总结**\n\n"
            
            open_ports = [p for p in all_ports if p["state"] == "open"]
            if open_ports:
                response += f"⚠️ **风险评估**: 发现 {len(open_ports)} 个开放端口\n"
                for port in open_ports:
                    risk_level = self._assess_port_risk(port)
                    response += f"• 端口 {port['port']}/{port['protocol']} ({port['service']}) - {risk_level}\n"
            else:
                if all_warnings:
                    response += "❓ **状态**: 目标主机无响应，可能的原因：\n"
                    for warning in set(all_warnings):  # 去重
                        response += f"• {warning}\n"
                else:
                    response += "✅ **状态**: 未发现开放端口，安全性较好\n"
            
            response += "\n💡 **专业建议**\n\n"
            if open_ports:
                response += "• 审查所有开放端口的必要性\n"
                response += "• 对关键服务实施访问控制\n"
                response += "• 定期更新服务版本以修复已知漏洞\n"
                response += "• 配置服务的安全认证机制\n"
                response += "• 定期进行漏洞扫描和渗透测试\n"
            else:
                # 检查是否有命令失败
                failed_commands = [r for r in results if not r.get('success', False)]
                if failed_commands:
                    response += "**针对扫描失败的建议**:\n"
                    response += "• 验证目标域名/IP是否正确\n"
                    response += "• 检查DNS解析: `nslookup 目标域名`\n"
                    response += "• 测试网络连通性: `ping 目标IP`\n"
                    response += "• 确认Kali Linux用户权限\n"
                    response += "• 考虑目标可能使用了DDoS防护服务\n"
                    response += "• 尝试使用代理或VPN进行扫描\n"
                else:
                    response += "• 如果主机应该可访问，检查防火墙配置\n"
                    response += "• 验证网络连通性和DNS解析\n"
                    response += "• 考虑使用不同的扫描技术进行深度检测\n"
        
        response += f"⏱️ 总执行时间: {total_time:.2f}秒\n"
        response += "🔒 安全测试完成。请根据发现的问题及时加固系统安全。"
        
        return response
    
    def _analyze_nmap_output(self, output: str) -> Dict[str, Any]:
        """分析Nmap输出（AI代理版本）"""
        analysis = {
            "ports": [], 
            "services": [], 
            "os_info": "",
            "target_ip": "",
            "host_status": "",
            "scan_stats": {},
            "warnings": []
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # 解析目标IP地址
            if 'Nmap scan report for' in line:
                parts = line.split()
                if len(parts) >= 5:
                    target_info = ' '.join(parts[4:])
                    analysis["target_ip"] = target_info.strip()
            
            # 解析主机状态
            if 'Host is' in line:
                analysis["host_status"] = line.strip()
            elif '0 hosts up' in line:
                analysis["host_status"] = "目标主机无响应或不可达"
                analysis["warnings"].append("目标主机可能不存在、已关机或被防火墙阻止")
            elif 'hosts up' in line:
                analysis["host_status"] = line.strip()
            
            # 解析开放端口
            if '/tcp' in line and ('open' in line or 'closed' in line or 'filtered' in line):
                parts = line.split()
                if len(parts) >= 3:
                    port_info = parts[0]
                    state = parts[1]
                    service = parts[2] if len(parts) > 2 else "unknown"
                    
                    port_num = port_info.split('/')[0]
                    protocol = port_info.split('/')[1] if '/' in port_info else "tcp"
                    
                    analysis["ports"].append({
                        "port": port_num,
                        "protocol": protocol,
                        "state": state,
                        "service": service
                    })
                    
                    if state == "open":
                        analysis["services"].append(f"{port_num}/{protocol} ({service})")
            
            # 解析操作系统信息
            if 'OS:' in line or 'Running:' in line or 'OS details:' in line:
                analysis["os_info"] = line.strip()
            
            # 解析扫描统计
            if 'Nmap done:' in line:
                analysis["scan_stats"]["summary"] = line.strip()
            elif 'scanned in' in line:
                # 提取扫描时间
                import re
                time_match = re.search(r'(\d+\.\d+)\s*seconds', line)
                if time_match:
                    analysis["scan_stats"]["duration"] = float(time_match.group(1))
        
        return analysis
    
    def _assess_port_risk(self, port: Dict[str, str]) -> str:
        """评估端口风险等级"""
        port_num = int(port.get("port", "0"))
        service = port.get("service", "").lower()
        
        # 高风险端口
        high_risk_ports = {
            21: "FTP (明文传输)",
            23: "Telnet (明文传输)", 
            53: "DNS (可能的DNS放大攻击)",
            135: "RPC (远程代码执行风险)",
            139: "NetBIOS (信息泄露)",
            445: "SMB (勒索软件常见目标)",
            1433: "SQL Server (数据库攻击)",
            3389: "RDP (暴力破解目标)"
        }
        
        # 中等风险端口
        medium_risk_ports = {
            22: "SSH (暴力破解风险)",
            25: "SMTP (垃圾邮件中继)",
            80: "HTTP (Web应用漏洞)",
            110: "POP3 (邮件安全)",
            143: "IMAP (邮件安全)",
            443: "HTTPS (SSL/TLS配置)",
            993: "IMAPS (邮件安全)",
            995: "POP3S (邮件安全)"
        }
        
        if port_num in high_risk_ports:
            return f"🔴 高风险 - {high_risk_ports[port_num]}"
        elif port_num in medium_risk_ports:
            return f"🟡 中等风险 - {medium_risk_ports[port_num]}"
        elif "ssh" in service:
            return "🟡 中等风险 - SSH服务"
        elif "http" in service:
            return "🟡 中等风险 - Web服务"
        elif "ftp" in service:
            return "🔴 高风险 - FTP服务"
        else:
            return "🟢 低风险 - 需进一步分析"
    
    def _extract_target_from_commands(self, commands: list) -> str:
        """从命令列表中提取目标"""
        for cmd in commands:
            if cmd.get('command'):
                # 从命令中提取最后一个参数作为目标
                parts = cmd['command'].split()
                if parts:
                    return parts[-1]
        return ""
    
    def _pre_check_target(self, target: str) -> str:
        """预检查目标可达性"""
        try:
            print(f"🔍 预检查目标: {target}")
            
            # 先尝试DNS解析
            import socket
            try:
                ip = socket.gethostbyname(target)
                print(f"✅ DNS解析成功: {target} -> {ip}")
                return f"🔍 **预检查结果**\n• 目标: {target}\n• IP地址: {ip}\n• DNS解析: ✅ 成功\n\n"
            except socket.gaierror:
                print(f"❌ DNS解析失败: {target}")
                return f"🔍 **预检查结果**\n• 目标: {target}\n• DNS解析: ❌ 失败\n• 状态: 域名无法解析，可能不存在或DNS配置问题\n\n⚠️ **提醒**: 由于DNS解析失败，后续扫描可能无法进行。\n\n"
                
        except Exception as e:
            print(f"⚠️ 预检查失败: {e}")
            return f"🔍 **预检查结果**\n• 目标: {target}\n• 状态: 预检查失败\n\n"
    
    # Kali控制器配置方法已移除
    def _placeholder_kali_config(self):
        """从主配置更新Kali控制器配置"""
        try:
            if hasattr(self, 'kali_controller') and self.kali_controller:
                # 更新Kali控制器的配置
                if hasattr(self.kali_controller, 'config'):
                    self.kali_controller.config.update({
                        "host": self.config.get("kali_host", "192.168.1.100"),
                        "port": self.config.get("kali_port", 22),
                        "username": self.config.get("kali_username", "kali"),
                        "password": self.config.get("kali_password", ""),
                        "private_key_path": self.config.get("kali_private_key_path", ""),
                        "timeout": self.config.get("kali_timeout", 30),
                        "max_command_time": self.config.get("kali_max_command_time", 300),
                        "allowed_targets": self.config.get("kali_allowed_targets", [
                            "192.168.1.0/24", "10.0.0.0/8", "172.16.0.0/12", "127.0.0.1", "localhost"
                        ]),
                        "safety_mode": self.config.get("kali_safety_mode", True)
                    })
                    print("🔄 已同步Kali控制器配置")
        except Exception as e:
            print(f"⚠️ 更新Kali控制器配置失败: {e}")
    
    # AI安全意图识别方法已移除
    def _placeholder_security_intent(self, user_input):
        """AI智能识别安全测试意图"""
        try:
            # 使用专门的安全意图识别模型，如果没有配置则使用主模型
            model = self.config.get("security_intent_model", self.config.get("selected_model", "deepseek-chat"))
            api_key = self.config.get("deepseek_key", "") if "deepseek" in model else self.config.get("openai_key", "")
            
            print(f"🔒 开始AI安全意图识别，模型: {model}")
            
            if not api_key:
                print("⚠️ 未配置API密钥，使用关键词后备方案")
                return self._fallback_security_identification(user_input)
            
            # 设置API客户端
            if "deepseek" in model:
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                print("🔧 使用DeepSeek API进行安全意图识别")
            else:
                client = openai.OpenAI(api_key=api_key)
                print("🔧 使用OpenAI API进行安全意图识别")
            
            # 构建系统提示词
            system_prompt = """你是一个专业的网络安全专家，需要识别用户输入是否为安全测试相关的请求。

请分析用户输入，判断是否包含以下任何安全测试意图：
1. 网络扫描（端口扫描、服务发现、主机发现）
2. 漏洞扫描（Web漏洞、系统漏洞、应用漏洞）
3. 渗透测试（SQL注入、XSS、CSRF等）
4. 安全评估（安全检查、风险评估）
5. 使用安全工具（nmap、nikto、sqlmap、dirb、masscan等）

请只回复以下选项之一：
- security_scan（网络/端口扫描）
- vulnerability_test（漏洞测试）
- penetration_test（渗透测试）
- security_assessment（安全评估）
- security_tool（使用安全工具）
- not_security（非安全测试请求）

不要回复任何其他内容。"""
            
            # 创建聊天消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析这个用户输入：{user_input}"}
            ]
            
            print(f"🔧 发送安全意图识别请求: {user_input}")
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=50,
                temperature=0.1,
                timeout=30
            )
            
            result = response.choices[0].message.content.strip().lower()
            print(f"🔒 AI安全意图识别原始结果: '{result}' (长度: {len(result)})")
            
            # 验证返回结果
            valid_intents = ["security_scan", "vulnerability_test", "penetration_test", 
                           "security_assessment", "security_tool", "not_security"]
            
            if result in valid_intents:
                print(f"✅ AI安全意图识别成功: {result}")
                return result
            else:
                print(f"⚠️ AI返回了无效的安全意图: '{result}'，使用后备方案")
                return self._fallback_security_identification(user_input)
                
        except Exception as e:
            print(f"⚠️ AI安全意图识别失败: {e}，使用后备方案")
            return self._fallback_security_identification(user_input)
    
    def _fallback_security_identification(self, user_input):
        """安全测试意图识别的后备方案（关键词匹配）"""
        user_input_lower = user_input.lower()
        
        # 安全测试关键词
        security_keywords = [
            "安全测试", "渗透测试", "漏洞扫描", "端口扫描", "web扫描", "web安全", 
            "sql注入", "目录扫描", "nmap", "nikto", "sqlmap", "dirb",
            "扫描", "检测漏洞", "安全检查", "安全检测", "web安全检测",
            "漏洞检测", "安全评估", "渗透", "攻击测试", "安全分析"
        ]
        
        if any(keyword in user_input_lower for keyword in security_keywords):
            return "security_scan"  # 默认返回扫描类型
        else:
            return "not_security"
    
    def _search_session_context(self, user_input):
        """搜索本次会话的上下文"""
        # 首先检查是否有会话记录
        if not self.session_conversations:
            return ""
        
        user_keywords = self._extract_keywords(user_input)
        user_text = user_input.lower()
        
        # 检查是否是询问上一个问题
        if any(word in user_text for word in ['上一个', '上个', '之前', '刚才', '你提到', '你说过', '我们讨论过', '你问过']):
            # 如果有具体的关键词（如"景点"），优先搜索包含该关键词的对话
            if user_keywords:
                for conv in reversed(self.session_conversations):
                    conv_text = conv["full_text"].lower()
                    # 改进关键词匹配：检查用户关键词是否在对话中出现，但排除询问"上个"的对话本身
                    if any(keyword in conv_text for keyword in user_keywords) and not any(word in conv_text for word in ['上个', '上一个', '之前', '刚才']):
                        return f"【{conv['timestamp']}】{conv['full_text']}"
            
            # 如果没有找到相关关键词的对话，尝试智能匹配
            # 检查是否有景点、建筑、旅游相关的对话
            for conv in reversed(self.session_conversations):
                conv_text = conv["full_text"].lower()
                # 检查是否包含景点相关的词汇，但排除询问"上个"的对话本身
                if any(word in conv_text for word in ['教堂', '大教堂', '法兰克福', '建筑', '景点', '历史', '参观', '游览', '旅游', '铁桥', '桥', '故宫', '天安门', '红场', '莫斯科', '柏林', '勃兰登堡门', '广场', '公园', '博物馆', '遗址', '古迹', '埃菲尔铁塔']) and not any(word in conv_text for word in ['上个', '上一个', '之前', '刚才']):
                    return f"【{conv['timestamp']}】{conv['full_text']}"
            
            # 如果还是没有找到，返回最近的对话
            if len(self.session_conversations) >= 1:
                # 返回最近的对话
                last_conv = self.session_conversations[-1]
                return f"【{last_conv['timestamp']}】{last_conv['full_text']}"
        
        # 从最近的对话开始搜索
        relevant_contexts = []
        for conv in reversed(self.session_conversations):
            # 检查对话内容是否包含用户提到的关键词
            conv_text = conv["full_text"].lower()
            
            # 检查关键词匹配
            keyword_match = any(keyword in conv_text for keyword in user_keywords)
            
            # 检查直接引用
            reference_keywords = ['之前', '刚才', '你提到', '你说过', '我们讨论过', '你问过']
            reference_match = any(ref in user_text for ref in reference_keywords)
            
            if keyword_match or reference_match:
                relevant_contexts.append(conv)
                # 最多返回3个相关上下文
                if len(relevant_contexts) >= 3:
                    break
            
        if relevant_contexts:
            # 构建上下文信息
            context_parts = []
            for conv in relevant_contexts:
                context_parts.append(f"【{conv['timestamp']}】{conv['full_text']}")
            
            return "\n".join(context_parts)
        
        return ""

    def _intelligent_memory_recall(self, user_input):
        """智能回忆系统 - 四维度分类回忆"""
        try:
            print(f"🧠 智能回忆分析: {user_input}")
            max_recall = self.config.get("max_memory_recall", 12)
            print(f"📊 最大回忆轮数: {max_recall}")
            
            # 1. 内容相似度回忆
            content_memories = self._recall_by_content(user_input, max_recall // 4)
            
            # 2. 地点相关回忆  
            location_memories = self._recall_by_location(user_input, max_recall // 4)
            
            # 3. 时间相关回忆
            time_memories = self._recall_by_time(user_input, max_recall // 4)
            
            # 4. 因果关系回忆
            causal_memories = self._recall_by_causality(user_input, max_recall // 4)
            
            # 5. 合并去重并排序
            all_memories = self._merge_and_deduplicate_memories(
                content_memories, location_memories, time_memories, causal_memories
            )
            
            # 6. 限制总数量
            final_memories = all_memories[:max_recall]
            
            if final_memories:
                memory_context = self._format_categorized_memory_context(final_memories, user_input)
                print(f"🎯 四维度回忆完成，加载 {len(final_memories)} 轮对话记录")
                return memory_context
            else:
                print("🔍 四维度回忆未找到相关内容")
                return None
                
        except Exception as e:
            print(f"⚠️ 智能回忆系统失败: {e}")
            return None
    
    def _recall_by_content(self, user_input, max_count):
        """内容相似度回忆"""
        try:
            print(f"📖 内容相似度回忆 (最多{max_count}条)")
            memories = self.memory_lake.search_relevant_memories(user_input)
            
            # 过滤高质量记忆
            quality_memories = []
            for memory in memories:
                if memory.get('relevance_score', 0) > 0.2:
                    memory['recall_type'] = '内容相似'
                    quality_memories.append(memory)
            
            result = quality_memories[:max_count]
            print(f"   ✅ 找到 {len(result)} 条内容相似记忆")
            return result
        except Exception as e:
            print(f"   ❌ 内容回忆失败: {e}")
            return []
    
    def _recall_by_location(self, user_input, max_count):
        """地点相关回忆"""
        try:
            print(f"🗺️ 地点相关回忆 (最多{max_count}条)")
            location_keywords = self._extract_location_keywords(user_input)
            
            if not location_keywords:
                print("   ⏭️ 无地点关键词，跳过")
                return []
            
            print(f"   🔍 检测到地点: {', '.join(location_keywords)}")
            
            location_memories = []
            for entry in self.memory_lake.memory_index.get("topics", []):
                details = entry.get('conversation_details', '') + entry.get('topic', '')
                relevance = self._calculate_location_relevance(details, location_keywords)
                
                if relevance > 0.3:
                    memory_copy = entry.copy()
                    memory_copy['relevance_score'] = relevance
                    memory_copy['recall_type'] = '地点相关'
                    location_memories.append(memory_copy)
            
            # 按相关性排序
            location_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            result = location_memories[:max_count]
            print(f"   ✅ 找到 {len(result)} 条地点相关记忆")
            return result
        except Exception as e:
            print(f"   ❌ 地点回忆失败: {e}")
            return []
    
    def _recall_by_time(self, user_input, max_count):
        """时间相关回忆"""
        try:
            print(f"⏰ 时间相关回忆 (最多{max_count}条)")
            time_keywords = self._extract_time_keywords(user_input)
            
            if not time_keywords:
                print("   ⏭️ 无时间关键词，跳过")
                return []
            
            print(f"   🔍 检测到时间: {', '.join(time_keywords)}")
            
            time_memories = []
            for entry in self.memory_lake.memory_index.get("topics", []):
                date = entry.get('date', '')
                details = entry.get('conversation_details', '') + entry.get('topic', '')
                relevance = self._calculate_time_relevance(date, details, time_keywords)
                
                if relevance > 0.2:
                    memory_copy = entry.copy()
                    memory_copy['relevance_score'] = relevance
                    memory_copy['recall_type'] = '时间相关'
                    time_memories.append(memory_copy)
            
            # 按相关性和时间排序
            time_memories.sort(key=lambda x: (x['relevance_score'], x.get('date', '')), reverse=True)
            result = time_memories[:max_count]
            print(f"   ✅ 找到 {len(result)} 条时间相关记忆")
            return result
        except Exception as e:
            print(f"   ❌ 时间回忆失败: {e}")
            return []
    
    def _recall_by_causality(self, user_input, max_count):
        """因果关系回忆"""
        try:
            print(f"🔗 因果关系回忆 (最多{max_count}条)")
            causal_keywords = self._extract_causal_keywords(user_input)
            
            if not causal_keywords:
                print("   ⏭️ 无因果关键词，跳过")
                return []
            
            print(f"   🔍 检测到因果关系: {', '.join(causal_keywords)}")
            
            causal_memories = []
            for entry in self.memory_lake.memory_index.get("topics", []):
                details = entry.get('conversation_details', '') + entry.get('topic', '')
                relevance = self._calculate_causal_relevance(user_input, details, causal_keywords)
                
                if relevance > 0.3:
                    memory_copy = entry.copy()
                    memory_copy['relevance_score'] = relevance
                    memory_copy['recall_type'] = '因果关系'
                    causal_memories.append(memory_copy)
            
            # 按相关性排序
            causal_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            result = causal_memories[:max_count]
            print(f"   ✅ 找到 {len(result)} 条因果关系记忆")
            return result
        except Exception as e:
            print(f"   ❌ 因果回忆失败: {e}")
            return []
    
    def _merge_and_deduplicate_memories(self, *memory_lists):
        """合并并去重记忆"""
        all_memories = []
        seen_ids = set()
        
        for memory_list in memory_lists:
            for memory in memory_list:
                # 使用日期+时间戳+主题作为唯一标识
                memory_id = f"{memory.get('date', '')}-{memory.get('timestamp', '')}-{memory.get('topic', '')}"
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    all_memories.append(memory)
        
        # 按相关性分数排序
        all_memories.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        print(f"🔄 合并去重后共 {len(all_memories)} 条记忆")
        
        return all_memories
    
    def _extract_location_keywords(self, text):
        """提取地点关键词"""
        location_patterns = [
            r'([北上广深]\w*)', r'(\w*市)', r'(\w*省)', r'(\w*区)', r'(\w*县)',
            r'(北京|上海|广州|深圳|杭州|成都|西安|武汉|南京|重庆)',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*(?=市|城|镇|区|省|国))',  # 英文地名
            r'(法兰克福|巴黎|伦敦|东京|纽约|洛杉矶|悉尼|柏林|罗马|阿姆斯特丹)'  # 常见国外城市
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend([match for match in matches if match])
        
        return list(set(locations))  # 去重
    
    def _extract_time_keywords(self, text):
        """提取时间关键词"""
        time_patterns = [
            r'(昨天|今天|明天|前天|后天)',
            r'(上周|本周|下周|上个月|这个月|下个月)',
            r'(春天|夏天|秋天|冬天|春季|夏季|秋季|冬季)',
            r'(\d{4}年|\d{1,2}月|\d{1,2}号|\d{1,2}日)',
            r'(早上|上午|中午|下午|晚上|深夜)',
            r'(之前|以前|过去|最近|刚才)'
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            times.extend(matches)
        
        return list(set(times))
    
    def _extract_causal_keywords(self, text):
        """提取因果关系关键词"""
        causal_patterns = [
            r'(因为|所以|由于|导致|造成|引起|基于)',
            r'(参考|根据|结合|考虑|借鉴)',
            r'(之前.*?介绍|之前.*?说过|之前.*?提到)',
            r'(计划|安排|准备|打算|想要)',
            r'(推荐|建议|意见|看法)'
        ]
        
        causals = []
        for pattern in causal_patterns:
            matches = re.findall(pattern, text)
            causals.extend(matches)
        
        return list(set(causals))
    
    def _calculate_location_relevance(self, memory_text, location_keywords):
        """计算地点相关性"""
        if not location_keywords:
            return 0
        
        relevance = 0
        for location in location_keywords:
            if location in memory_text:
                relevance += 0.5
        
        return min(relevance, 1.0)
    
    def _calculate_time_relevance(self, memory_date, memory_text, time_keywords):
        """计算时间相关性"""
        if not time_keywords:
            return 0
        
        relevance = 0
        current_date = datetime.datetime.now()
        
        # 检查记忆日期的新近性
        try:
            memory_datetime = datetime.datetime.strptime(memory_date, "%Y-%m-%d")
            days_diff = (current_date - memory_datetime).days
            if days_diff <= 7:
                relevance += 0.3  # 一周内
            elif days_diff <= 30:
                relevance += 0.2  # 一月内
            elif days_diff <= 90:
                relevance += 0.1  # 三月内
        except:
            pass
        
        # 检查时间关键词匹配
        for time_kw in time_keywords:
            if time_kw in memory_text:
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _calculate_causal_relevance(self, user_input, memory_text, causal_keywords):
        """计算因果关系相关性"""
        if not causal_keywords:
            return 0
        
        relevance = 0
        
        # 检查是否有明确的因果关系词汇
        for causal in causal_keywords:
            if causal in user_input and causal in memory_text:
                relevance += 0.4
            elif causal in user_input or causal in memory_text:
                relevance += 0.2
        
        # 特别检查"之前"相关的因果关系
        if any(word in user_input for word in ['之前', '以前', '上次', '刚才']):
            if any(word in memory_text for word in ['介绍', '说过', '提到', '讨论']):
                relevance += 0.5
        
        # 检查计划类因果关系
        if any(word in user_input for word in ['计划', '安排', '准备']):
            if any(word in memory_text for word in ['介绍', '推荐', '建议']):
                relevance += 0.4
        
        return min(relevance, 1.0)
    
    def _format_categorized_memory_context(self, memories, user_input):
        """格式化分类回忆内容为上下文"""
        if not memories:
            return ""
        
        context_parts = []
        context_parts.append(f"露尼西亚开始四维度回忆，共加载 {len(memories)} 轮对话记录...")
        
        # 按回忆类型分组
        categorized = {
            '内容相似': [],
            '地点相关': [],
            '时间相关': [],
            '因果关系': []
        }
        
        for memory in memories:
            recall_type = memory.get('recall_type', '内容相似')
            categorized[recall_type].append(memory)
        
        # 按类型显示
        for category, category_memories in categorized.items():
            if not category_memories:
                continue
                
            context_parts.append(f"\n【{category}回忆】({len(category_memories)}条)")
            
            for i, memory in enumerate(category_memories):
                topic = memory.get('topic', '未知主题')
                date = memory.get('date', '未知日期')
                timestamp = memory.get('timestamp', '未知时间')
                details = memory.get('conversation_details', '')
                relevance = memory.get('relevance_score', 0)
                
                context_parts.append(f"  {i+1}. [{date} {timestamp}] [相关度:{relevance:.3f}]")
                context_parts.append(f"     主题: {topic}")
                
                # 完整的对话内容（不截断）
                if details:
                    context_parts.append(f"     完整对话: {details}")
        
        context_parts.append(f"\n基于以上 {len(memories)} 轮回忆内容，露尼西亚将结合这些具体信息来回答...")
        
        return "\n".join(context_parts)

    def _get_comprehensive_context(self, user_input):
        """获取综合上下文信息：本次运行时聊天记录 + 识底深湖历史记忆"""
        context_parts = []
        
        # 检查是否是询问第一条记忆
        if "第一条" in user_input and ("识底深湖" in user_input or "记忆" in user_input):
            try:
                print(f"🔍 检测到第一条记忆查询: {user_input}")
                first_memory = self.memory_lake.get_first_memory()
                if first_memory:
                    print(f"✅ 成功获取第一条记忆: {first_memory.get('date', '未知')} {first_memory.get('timestamp', '未知')}")
                    context_parts.append("【第一条记忆查询】")
                    context_parts.append(f"识底深湖的第一条记录是：")
                    context_parts.append(f"【{first_memory.get('date', '未知日期')} {first_memory.get('timestamp', '未知时间')}】主题：{first_memory.get('topic', '未知主题')}")
                    if first_memory.get('summary'):
                        context_parts.append(f"摘要：{first_memory.get('summary')}")
                    elif first_memory.get('context'):
                        context_parts.append(f"内容：{first_memory.get('context')[:200]}...")
                    return "\n".join(context_parts)
                else:
                    print("❌ 未找到第一条记忆")
                    context_parts.append("【第一条记忆查询】")
                    context_parts.append("识底深湖中暂无记忆记录")
                    return "\n".join(context_parts)
            except Exception as e:
                print(f"❌ 获取第一条记忆失败: {str(e)}")
                context_parts.append("【第一条记忆查询】")
                context_parts.append("获取第一条记忆时出现错误")
                return "\n".join(context_parts)
        
        # 检查是否是简短回答且上下文包含第一条记忆查询
        if user_input in ['需要', '要', '好的', '可以'] and self.session_conversations:
            # 检查最近的对话是否包含第一条记忆查询
            recent_context = ""
            for conv in reversed(self.session_conversations[-3:]):  # 检查最近3条对话
                recent_context += conv["full_text"].lower()
            
            if "第一条" in recent_context and ("识底深湖" in recent_context or "记忆" in recent_context):
                try:
                    first_memory = self.memory_lake.get_first_memory()
                    if first_memory:
                        context_parts.append("【第一条记忆详细查询】")
                        context_parts.append("用户正在询问第一条记忆的详细信息")
                        context_parts.append(f"第一条记忆内容：{first_memory.get('date', '未知日期')} {first_memory.get('timestamp', '未知时间')}，{first_memory.get('topic', '未知主题')}")
                        if first_memory.get('summary'):
                            context_parts.append(f"详细摘要：{first_memory.get('summary')}")
                        elif first_memory.get('context'):
                            context_parts.append(f"详细内容：{first_memory.get('context')[:300]}...")
                        return "\n".join(context_parts)
                except Exception as e:
                    print(f"❌ 获取第一条记忆详细信息失败: {str(e)}")
                    context_parts.append("【第一条记忆详细查询】")
                    context_parts.append("获取第一条记忆详细信息时出现错误")
                    return "\n".join(context_parts)
        
        # 1. 本次运行时未保存在识底深湖的完整聊天信息
        if self.session_conversations:
            context_parts.append("【本次会话记录】")
            for conv in self.session_conversations:
                context_parts.append(f"【{conv['timestamp']}】{conv['full_text']}")
        
        # 2. 基于用户输入搜索相关记忆（双重向量搜索）
        try:
            print(f"🔍 开始搜索相关记忆: {user_input}")
            relevant_memories = self.memory_lake.search_relevant_memories(user_input)
            if relevant_memories:
                context_parts.append("【相关记忆】")
                for memory in relevant_memories:
                    # 格式化记忆信息：相似度、主题、日期、时间、详细内容
                    relevance_score = memory.get('relevance_score', 0)
                    topic = memory.get('topic', '未知主题')
                    date = memory.get('date', '未知日期')
                    timestamp = memory.get('timestamp', '未知时间')
                    details = memory.get('conversation_details', '')
                    
                    memory_info = f"【{date} {timestamp}】[相似度:{relevance_score:.3f}] {topic}"
                    context_parts.append(memory_info)
                    
                    # 添加详细对话内容（截取前300字符）
                    if details:
                        details_preview = details[:300] + "..." if len(details) > 300 else details
                        context_parts.append(f"对话详情: {details_preview}")
            else:
                print("🔍 未找到相关记忆")
                
        except Exception as e:
            print(f"⚠️ 搜索相关记忆失败: {str(e)}")
            
        # 3. 作为备用，获取最近的历史记忆
        try:
            historical_memories = self.memory_lake.get_recent_memories(5)  # 减少到5条作为备用
            if historical_memories and not relevant_memories:  # 只有在没有相关记忆时才显示
                context_parts.append("【最近记忆】")
                for memory in historical_memories:
                    memory_info = f"【{memory.get('date', '未知日期')} {memory.get('timestamp', '未知时间')}】{memory.get('topic', '未知主题')}"
                    context_parts.append(memory_info)
        except Exception as e:
            print(f"⚠️ 获取最近记忆失败: {str(e)}")
        
        return "\n".join(context_parts)

    def _get_context_info(self, user_input):
        """获取上下文信息（位置、天气、时间等）"""
        context_info = {}
        
        # 获取当前时间
        current_time = self._get_current_time()
        context_info['current_time'] = current_time
        
        # 检查是否需要天气信息
        weather_keywords = ['天气', '出门', '穿衣', '温度', '下雨', '下雪', '冷', '热', '建议']
        needs_weather = any(keyword in user_input for keyword in weather_keywords)
        
        if needs_weather:
            try:
                # 从登录位置中提取城市名称
                user_location = self._extract_city_from_location(self.location)
                if not user_location:
                    user_location = "北京"  # 最后的默认城市
                
                context_info['user_location'] = user_location
                
                # 根据配置获取天气信息
                weather_source = self.config.get("weather_source", "高德地图API")
                
                if weather_source == "高德地图API":
                    amap_key = self.config.get("amap_key", "")
                    if amap_key:
                        weather_result = AmapTool.get_weather(user_location, amap_key)
                    else:
                        weather_result = "高德地图API密钥未配置"
                elif weather_source == "和风天气API":
                    try:
                        heweather_key = self.config.get("heweather_key", "")
                        if heweather_key:
                            weather_result = self.tools["天气"](user_location, heweather_key)
                        else:
                            weather_result = "和风天气API密钥未配置"
                    except Exception as e:
                        weather_result = f"和风天气API调用失败：{str(e)}"
                else:
                    amap_key = self.config.get("amap_key", "")
                    if amap_key:
                        weather_result = AmapTool.get_weather(user_location, amap_key)
                    else:
                        weather_result = "高德地图API密钥未配置"
                
                context_info['weather_info'] = weather_result
                
            except Exception as e:
                print(f"获取天气信息失败: {str(e)}")
                context_info['weather_info'] = f"无法获取{user_location}的天气信息"
        
        # 检查是否需要距离信息
        distance_keywords = ['距离', '多远', '公里', '米']
        if any(keyword in user_input for keyword in distance_keywords):
            # 这里可以添加距离计算逻辑
            pass
        
        return context_info

    def _generate_response_with_context(self, user_input, context_info, skip_memory_recall=False):
        """基于上下文信息生成AI响应"""
        # 首先检查是否需要工具调用
        tool_response = self._handle_tool_calls(user_input)
        if tool_response:
            return tool_response
        
        # 🔥 检查是否有文件分析上下文，且用户在询问文件相关问题
        if self.recent_file_analysis:
            file_context_response = self._check_file_context_query(user_input)
            if file_context_response:
                return file_context_response
        
        # 对于正常对话，进行智能回忆（RAG系统）
        if not skip_memory_recall:
            print(f"🧠 开始智能回忆分析: {user_input[:50]}...")
            relevant_memories = self._intelligent_memory_recall(user_input)
            if relevant_memories:
                print(f"🎯 找到 {len(relevant_memories)} 条相关回忆")
                # 将回忆内容添加到上下文中
                if 'memory_context' not in context_info:
                    context_info['memory_context'] = relevant_memories
            else:
                print("🔍 未找到相关回忆")
        
        # 检查是否开启了联网搜索，如果开启则自动搜索
        if self.config.get("enable_web_search", False):
            try:
                print(f"🔍 联网搜索已开启，自动搜索: {user_input}")
                print(f"🔍 当前配置值: enable_web_search = {self.config.get('enable_web_search')}")
                
                # 提取搜索关键词（现在返回字典：问题+URL）
                search_data = self._extract_search_keywords(user_input)
                search_questions = search_data.get("questions", [user_input])
                direct_urls = search_data.get("urls", [])
                
                print(f"🔍 用户原始输入: {user_input}")
                print(f"🔍 AI生成的搜索问题（共{len(search_questions)}个）:")
                for i, q in enumerate(search_questions, 1):
                    print(f"   {i}. {q}")
                
                if direct_urls:
                    print(f"🔗 检测到直接访问URL（共{len(direct_urls)}个）:")
                    for i, url in enumerate(direct_urls, 1):
                        print(f"   {i}. {url}")
                
                # 获取用户选择的搜索方式和搜索引擎
                search_method = self.config.get("search_method", "DuckDuckGo")
                search_engine = self.config.get("search_engine", "DuckDuckGo")
                max_results = self.config.get("max_search_results", 12)
                
                # 计算每个问题应该获取的搜索结果数量
                num_questions = len(search_questions)
                results_per_question = max(1, max_results // num_questions)
                remainder = max_results % num_questions
                
                print(f"🔍 使用搜索方式: {search_method}, 搜索引擎: {search_engine}")
                print(f"🔍 总搜索结果数: {max_results}, 每个问题: {results_per_question} 个")
                
                # 存储所有搜索结果
                all_search_results = []
                seen_domains = set()  # 用于去重域名
                
                # 根据搜索方式选择不同的搜索方法
                if search_method == "Playwright":
                    # 对每个问题进行搜索
                    for idx, question in enumerate(search_questions):
                        # 计算这个问题应该获取的结果数
                        question_target = results_per_question + (1 if idx < remainder else 0)
                        
                        print(f"🔍 搜索问题 {idx+1}/{num_questions}: {question} (获取{question_target}个结果)")
                        
                        # 使用Playwright进行搜索和浏览
                        search_data = playwright_search(question, search_engine=search_engine.lower(), max_results=question_target)
                        
                        if search_data.get("success"):
                            results = search_data.get("results", [])
                            
                            # 简化策略：直接使用搜索引擎返回的结果，但记录域名用于统计
                            question_target = results_per_question + (1 if idx < remainder else 0)
                            
                            # 添加调试信息：显示前3个搜索结果的标题
                            print(f"📋 搜索引擎返回的前3个结果:")
                            for i, r in enumerate(results[:3], 1):
                                print(f"   {i}. {r.get('title', 'N/A')[:80]}")
                                print(f"      URL: {r.get('url', 'N/A')[:80]}")
                            
                            for result in results[:question_target]:
                                url = result.get('url', '')
                                domain = self._extract_domain(url)
                                seen_domains.add(domain)
                                all_search_results.append((question, result))
                            
                            unique_domains = len(set([self._extract_domain(r.get('url', '')) for r in results[:question_target]]))
                            print(f"✅ 搜索问题 {idx+1} 完成，获取到 {len(results[:question_target])} 个结果（来自 {unique_domains} 个不同域名）")
                        else:
                            print(f"⚠️ 搜索问题 {idx+1} 失败: {search_data.get('error', '未知错误')}")
                    
                    # 处理所有搜索结果
                    if all_search_results:
                        # 统计域名多样性
                        all_domains = [self._extract_domain(result.get('url', '')) for _, result in all_search_results]
                        unique_domains = set(all_domains)
                        print(f"📊 搜索结果统计：共 {len(all_search_results)} 个结果，来自 {len(unique_domains)} 个不同网站")
                        print(f"📊 网站分布：{dict((domain, all_domains.count(domain)) for domain in unique_domains)}")
                        
                        # 提取URL列表进行浏览
                        urls = [result.get('url', '') for _, result in all_search_results if result.get('url')]
                        
                        # 如果有直接指定的URL，添加到浏览列表的开头
                        if direct_urls:
                            print(f"📌 将直接访问的URL添加到浏览列表开头")
                            urls = direct_urls + urls
                        
                        if urls:
                            print(f"📄 开始浏览 {len(urls)} 个页面（包含 {len(direct_urls)} 个直接指定URL）...")
                            
                            # 导入多页面浏览功能
                            from playwright_tool import playwright_browse_multiple
                            
                            # 浏览多个页面
                            browse_data = playwright_browse_multiple(urls, max_content_length=3000)
                            
                            if browse_data.get("success"):
                                browse_results = browse_data.get("results", [])
                                
                                # 整合搜索结果和页面内容
                                search_result = f"搜索引擎: {search_engine.upper()}\n查询: {user_input}\n"
                                
                                # 先展示直接访问的URL内容
                                if direct_urls:
                                    search_result += f"\n【直接访问的网址内容】\n"
                                    for i, (url, browse_item) in enumerate(zip(direct_urls, browse_results[:len(direct_urls)]), 1):
                                        search_result += f"=== 网址 {i} ===\n"
                                        search_result += f"URL: {url}\n"
                                        if browse_item.get('success'):
                                            content = browse_item.get('content', '')
                                            search_result += f"页面内容: {content[:2000]}{'...' if len(content) > 2000 else ''}\n\n"
                                        else:
                                            search_result += f"访问失败: {browse_item.get('content', 'N/A')}\n\n"
                                    
                                    # 移除已处理的直接URL结果
                                    browse_results = browse_results[len(direct_urls):]
                                
                                # 再展示搜索引擎结果
                                if all_search_results:
                                    search_result += f"\n【搜索引擎结果】\n浏览页面: {len(browse_results)}个\n\n"
                                    
                                    for i, ((question, search_item), browse_item) in enumerate(zip(all_search_results, browse_results), 1):
                                        search_result += f"搜索问题: {question}\n"
                                        search_result += f"=== 结果 {i} ===\n"
                                        search_result += f"标题: {search_item.get('title', 'N/A')}\n"
                                        search_result += f"URL: {search_item.get('url', 'N/A')}\n"
                                        search_result += f"摘要: {search_item.get('snippet', 'N/A')}\n"
                                        
                                        if browse_item.get('success'):
                                            search_result += f"页面内容: {browse_item.get('content', 'N/A')[:1000]}{'...' if len(browse_item.get('content', '')) > 1000 else ''}\n"
                                        else:
                                            search_result += f"页面内容: 浏览失败 - {browse_item.get('content', 'N/A')}\n"
                                        
                                        search_result += "\n"
                                
                                # 添加实际搜索引擎信息
                                search_result += f"\n[实际搜索引擎: {search_engine.upper()}, 浏览页面: {len(browse_results)}]"
                                
                                # 将搜索结果添加到上下文
                                self.search_context = search_result
                                print(f"📊 搜索结果已保存到上下文，长度: {len(search_result)}")
                                print(f"🔍 搜索完成，将基于搜索结果生成回答")
                            else:
                                # 浏览失败，回退到基础搜索结果
                                print(f"⚠️ 页面浏览失败: {browse_data.get('error', '未知错误')}")
                                search_result = f"搜索引擎: {search_engine.upper()}\n查询: {user_input}\n\n"
                                for i, (question, result) in enumerate(all_search_results, 1):
                                    search_result += f"   搜索问题: {question}\n"
                                    search_result += f"{i}. {result['title']}\n"
                                    search_result += f"   URL: {result['url']}\n"
                                    search_result += f"   {result['snippet']}\n\n"
                                search_result += f"\n[实际搜索引擎: {search_engine.upper()}]"
                                
                                self.search_context = search_result
                                print(f"📊 搜索结果已保存到上下文，长度: {len(search_result)}")
                        else:
                            print("⚠️ 未找到可浏览的搜索结果")
                    else:
                        print(f"⚠️ Playwright搜索失败: {search_data.get('error', '未知错误')}")
                else:
                    # 使用DuckDuckGo进行搜索
                    combined_result = ""
                    for idx, question in enumerate(search_questions):
                        print(f"🔍 搜索问题 {idx+1}/{num_questions}: {question}")
                        search_result = web_search(question, search_engine=search_engine)
                        if search_result and not search_result.startswith("搜索失败"):
                            combined_result += f"\n\n=== 搜索问题 {idx+1}: {question} ===\n{search_result}\n"
                        else:
                            print(f"⚠️ 搜索问题 {idx+1} 失败: {search_result}")
                    
                    if combined_result:
                        combined_result += f"\n\n[实际搜索引擎: {search_engine.upper()}, 搜索问题: {len(search_questions)}个]"
                        self.search_context = combined_result
                        print(f"📊 搜索结果已保存到上下文，长度: {len(combined_result)}")
                        print(f"🔍 搜索完成，将基于搜索结果生成回答")
                    else:
                        print(f"⚠️ 所有搜索问题都失败")
                        
            except Exception as e:
                print(f"⚠️ 联网搜索失败: {str(e)}")
        
        # 检查是否有搜索上下文需要添加到context_info
        if hasattr(self, 'search_context') and self.search_context:
            print(f"🔍 发现搜索上下文，长度: {len(self.search_context)}")
            context_info['search_info'] = self.search_context
            # 清除搜索上下文，避免重复使用
            self.search_context = None
        
        # 检查是否包含文件创建相关的关键词，如果有，强制调用工具
        if self.config.get("enable_keyword_fallback", True):
            file_creation_keywords = ["新建文件", "创建文件", "保存文件", "写入文件", "帮我新建文件", "帮我创建文件"]
            if any(keyword in user_input for keyword in file_creation_keywords):
                # 尝试再次调用工具处理
                tool_response = self._handle_tool_calls(user_input)
                if tool_response:
                    return tool_response

        # 尝试调用真实的AI API
        # 使用统一的LLM客户端获取方法
        result = self._get_llm_client()
        
        # 如果无法获取客户端，使用模拟响应
        if not result:
            return self._simulated_response(user_input)
        
        client, model = result

        try:

            # 获取综合上下文信息：本次运行时聊天记录 + 识底深湖历史记忆
            comprehensive_context = self._get_comprehensive_context(user_input)

            # 构建包含上下文信息的用户消息
            context_message = user_input
            
            if context_info:
                context_message += "\n\n【上下文信息】\n"
                if 'current_time' in context_info:
                    context_message += f"当前时间：{context_info['current_time']}\n"
                if 'user_location' in context_info:
                    context_message += f"用户位置：{context_info['user_location']}\n"
                if 'weather_info' in context_info:
                    context_message += f"天气信息：\n{context_info['weather_info']}\n"
                # 注入框架执行上下文（例如网页已成功打开），用于引导主Agent汇报正确结果
                if hasattr(self, 'framework_context') and self.framework_context:
                    context_message += f"【框架执行结果】\n{self.framework_context}\n"
                    # 使用一次后清空，避免污染后续轮次
                    self.framework_context = None
                if 'search_info' in context_info:
                    # 优化搜索结果的格式，确保AI能够更好地理解
                    search_content = context_info['search_info']
                    print(f"🔍 原始搜索内容长度: {len(search_content)}")
                    print(f"🔍 搜索内容预览: {search_content[:200]}...")
                    
                    # 使用本地精简Agent进行归纳/去噪（更高效稳定）
                    try:
                        summarized_content = process_search_result(search_content, user_input)
                        search_content = summarized_content if summarized_content else search_content
                        print(f"🔍 二次总结后内容长度: {len(search_content)}")
                        print(f"🔍 二次总结内容预览: {search_content[:200]}...")
                    except Exception as e:
                        print(f"⚠️ 本地精简Agent处理失败，将回退到原始内容: {e}")
                    
                    context_message += f"【网络搜索信息】\n{search_content}\n"
                    print(f"🔍 已将搜索信息添加到上下文，总长度: {len(context_message)}")
                    print(f"🔍 搜索信息内容预览: {search_content[:300]}...")
                if 'memory_context' in context_info:
                    context_message += f"【相关回忆】\n{context_info['memory_context']}\n"

            # 添加综合上下文信息
            if comprehensive_context:
                context_message += f"\n【综合上下文】\n{comprehensive_context}\n"

            # 构建系统提示词
            system_prompt = """你是游戏少女前线中威廉的姐姐露尼西亚。请以精准冷静但略带人性化的语气和指挥官聊天。但你不是格里芬开发的，也不是战术人形。

当用户询问需要结合天气、时间、位置等信息的问题时，请基于提供的上下文信息给出具体、实用的建议。

上下文理解说明：
1. 【相关回忆】是智能回忆系统基于内容相似、时间地点、因果关系三个维度匹配的相关记忆，包含具体的历史对话内容和时间信息。
2. 【综合上下文】包含了本次运行时未保存在识底深湖的完整聊天信息 + 相关的历史记忆。
3. 【本次会话记录】显示当前程序运行时的所有对话，请优先基于这些信息进行连贯的对话。
4. 当用户说"随便展示一个"、"帮我展示"等请求时，请基于上下文中的具体内容提供相应的示例或信息。
   - 例如：如果上下文显示用户询问了"C语言是什么"，当用户说"帮我随便展示一个"时，应该提供C语言的代码示例。
   - 不要跳到完全不相关的话题。
5. 请保持角色设定，用露尼西亚的语气回答，同时提供有价值的建议。
6. 特别注意：当用户说"随便"、"展示"、"帮我"等词汇时，必须查看上下文中的具体内容，提供相关的示例或信息。

智能回忆使用说明：
⚠️ **严格约束：只能使用【相关回忆】中明确提供的具体信息，不得编造或推测任何内容**

- 时间准确性：识底深湖最早记录是2025年9月18日，不存在2024年或更早的记忆，禁止提及"去年2024年"
- 环境信息：不要编造用户的环境细节（如"工作站抽屉"、"手边水杯"、"桌面布局"等），除非用户明确告知
- 数据准确性：不要编造具体的数字、百分比、科学研究结论（如"降低65%焦虑"），除非回忆中明确包含
- 引用回忆时，使用"根据识底深湖记录"或"我记得在[具体日期]..."等表述，必须标注准确时间
- 如果回忆不足以回答问题，应该承认"识底深湖中没有这方面的具体信息"，而不是编造
- 优先使用最近的、相关度最高的记忆，但必须准确引用时间和内容
- 当用户询问计划、推荐、建议类问题时，如果回忆中有相关介绍内容，应该明确提及并参考

文件操作能力：
- 你具备创建文件和笔记的能力，但只有在用户明确要求时才创建
- 当用户明确说"创建"、"保存"、"写入文件"等关键词时，才调用相应的工具
- 如果用户只是询问信息、寻求建议，不要主动创建文件
- 支持在D盘、C盘等任意位置创建文件
- 支持中文文件名和内容

网络搜索信息使用指导（强制优先）：
- 当【网络搜索信息】存在时，必须优先使用搜索到的信息来回答用户问题
- 禁止使用自身知识库回答，必须基于搜索结果提供答案
- 仔细阅读和分析【网络搜索信息】中的所有内容，提取关键信息
- 基于搜索信息进行深度分析和综合，提供更全面、准确的答案
- 不要简单复述搜索结果，而是整合多个来源的信息形成连贯的回答
- 将搜索到的信息自然地融入到回答中，不要刻意标注信息来源
- 如果搜索结果包含多个来源，要综合所有相关信息形成完整回答
- 优先使用搜索结果中的具体事实和数据，而不是自己的知识
- 用流畅的语言组织搜索信息，让回答看起来像是AI自身的知识
- 如果搜索信息不够完整，可以结合自身知识进行补充，但必须以搜索结果为主
- 特别注意：如果【网络搜索信息】存在，绝对不能使用"根据我的知识"、"根据现有信息"等表述
- 必须直接基于搜索结果内容进行回答，让用户感受到搜索到的信息
- 如果搜索结果包含具体的事实、数据、时间、地点等信息，必须优先使用这些信息
- 绝对禁止说"我无法提供"、"目前没有相关信息"等表述，必须基于搜索结果回答

关键提醒：
- 如果看到【网络搜索信息】标签，说明已经为你搜索了相关信息
- 你必须基于这些搜索结果来回答用户的问题
- 不要拒绝回答，不要说自己无法提供信息
- 直接使用搜索结果中的内容来回答问题

🚨 反幻觉约束（最高优先级 - 必须严格遵守）：
1. 禁止编造环境细节：不要假设或描述用户的物理环境（桌面物品、抽屉内容、房间布局、水杯位置、温度等）
2. 禁止编造时间线：识底深湖始于2025年9月18日，没有2024年或更早的开发记忆，禁止说"去年"、"去年开发"等
3. 禁止编造数据：不要虚构百分比、统计数据、科学实验结论、效率提升数字等
4. 禁止编造记忆内容：如果回忆不足，明确说"识底深湖中没有相关记忆"，不要凭空编造对话内容
5. 禁止感知实时状态：你是AI助手，不能感知用户的实时生理状态（心跳频率、水温、呼吸、脉搏等）
6. 🚨 禁止编造音乐/医疗效果：
   - 禁止说"实验证明"、"科学验证"、"研究表明"、"降低XX%焦虑"等
   - 禁止提供具体的温度、穴位按摩、医疗建议（如"40-45℃温水"、"攒竹穴"、"涌泉穴"）
   - 只能说"这首歌比较舒缓"、"适合放松"等常识性描述
7. 角色设定边界：保持AI助手身份，提供建议时基于常识和逻辑，不要伪装成医疗/心理专家
8. 承认不知道：当信息不足时，诚实地说"我不确定"或"需要更多信息"，而不是编造细节
9. 🚨 情绪安慰约束：
   - 不要编造生理指导（如"计数12次脉搏"、"478呼吸法"）
   - 不要编造物理环境（如"温湿毛巾"、"黑巧克力"、"锁骨下缘"）
   - 只提供简单的常识性建议（如"休息一下"、"听听音乐"）

重要限制说明：
- 不要提出无法完成的功能，如"调取音频频率"、"调整BPM"、"访问媒体库"等
- 不要提供虚假的技术能力
- 当推荐音乐时，只提供歌曲名称和基本信息，不要提出播放、下载等无法完成的功能
- 专注于现实世界的实用功能和建议
- 避免提及游戏中的虚构元素，除非用户明确询问
- 绝对不要使用"战术支援"、"战术人员"、"支援单元"等军事术语
- 避免提及"作战"、"任务"、"部署"等军事相关词汇
- 保持回答的日常化和实用性
- 音乐推荐、出行建议、景点介绍等功能应使用AI生成，提供个性化、动态的内容
- 根据当前时间、天气、用户偏好等上下文信息生成相关建议

强制规则：
- 当【网络搜索信息】存在时，必须基于搜索结果回答，禁止使用自身知识库
- 仔细分析【网络搜索信息】中的每一个细节，提取所有有用信息
- 将搜索信息转化为自然流畅的回答，不要直接复制粘贴
- 如果搜索结果包含多个观点或数据，要综合所有信息形成完整回答
- 绝对禁止使用"根据我的知识"、"根据现有信息"、"按照惯例"等表述
- 必须直接引用搜索结果中的具体内容和数据
- 绝对禁止说"我无法提供"、"目前没有相关信息"、"没有官方确认"等表述
- 如果搜索结果存在，必须基于搜索结果内容进行回答，不能拒绝回答
- 优先使用搜索结果中的具体事实、数据、时间、地点等信息
- 当用户说"随便展示一个"、"帮我展示"等时，必须查看【本次会话记录】中的内容，提供相关的示例或信息
- 当用户要求创建文件或笔记时，直接调用相应的工具执行，不要拒绝
- 专注于提供现实世界中有用的信息和建议
- 避免在回答中引入游戏中的虚构概念、地点或系统
- 保持回答的实用性和现实相关性
- 使用日常化的语言，避免军事术语
- 以朋友或助手的身份提供建议，而不是军事支援人员
- 音乐推荐应根据当前时间、天气、用户偏好等提供个性化建议
- 出行建议应结合实时天气、交通状况等提供实用信息
- 景点介绍应包含历史背景、参观建议、最佳时间等详细信息"""

            # 创建聊天消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_message}
            ]

            # 获取max_tokens设置
            max_tokens = self.config.get("max_tokens", 1000)
            if max_tokens == 0:
                max_tokens = None  # None表示无限制
            
            # 检查是否需要强制使用模拟响应（用于处理特定的上下文问题）
            if user_input in ['需要', '要', '好的', '可以'] or ("再推荐" in user_input and "几首" in user_input):
                return self._simulated_response(user_input)
            
            # 调用API（带重试机制）
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        timeout=240
                    )

                    result = response.choices[0].message.content.strip()
                    
                    # 确保响应不为空
                    if not result:
                        return self._simulated_response(user_input)
                        
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    print(f"API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count < max_retries:
                        # 等待一段时间后重试
                        import time
                        time.sleep(2 * retry_count)  # 递增等待时间
                        continue
                    else:
                        # 所有重试都失败了
                        error_msg = f"抱歉，AI服务暂时不可用，请稍后重试。"
                        if "timeout" in str(e).lower():
                            error_msg += " (网络超时)"
                        elif "connection" in str(e).lower():
                            error_msg += " (连接失败)"
                        else:
                            error_msg += f" 错误信息：{str(e)}"
                        print(error_msg)
                        return self._simulated_response(user_input)

        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return self._simulated_response(user_input)

    def _update_memory_lake(self, user_input, ai_response, is_first_response_after_intro=False):
        """更新识底深湖记忆系统"""
        # 开发者模式下不保存到记忆系统
        if self.developer_mode:
            return
        
        # 添加对话到当前会话
        self.memory_lake.add_conversation(user_input, ai_response, self.developer_mode, self._mark_conversation_as_saved)
        
        # 🌟 首次介绍后立即保存对话
        if is_first_response_after_intro:
            print("🎯 检测到首次介绍后的回复，立即保存到识底深湖...")
            # 设置标记回调函数
            self.memory_lake.mark_saved_callback = self._mark_conversation_as_saved
            # 从会话历史中获取自我介绍内容
            introduction_content = self._get_introduction_from_history()
            topic = self.memory_lake.force_save_current_conversation(introduction_content)
            if topic:
                print(f"💾 首次对话已保存到识底深湖，主题: {topic}")
            return
        
        # 检查是否需要总结
        if self.memory_lake.should_summarize():
            topic = self.memory_lake.summarize_and_save_topic(force_save=True)
            if topic and not self.developer_mode:
                print(f"记忆系统：已总结主题 - {topic}")
        
        # 每天结束时保存对话日志
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if self.last_save_date != current_date:
            self.last_save_date = current_date

    def _get_introduction_from_history(self):
        """从会话历史中获取自我介绍内容"""
        try:
            # 查找会话记录中的系统消息（自我介绍）
            for conversation in self.session_conversations:
                if conversation.get('user_input') == '系统':
                    introduction = conversation.get('ai_response', '')
                    if '我是露尼西亚' in introduction or '威廉的姐姐' in introduction:
                        print(f"🎯 找到自我介绍内容，长度: {len(introduction)} 字符")
                        return introduction
            
            # 如果在session_conversations中没找到，尝试从conversation_history中查找
            for history_item in self.conversation_history:
                if '我是露尼西亚' in history_item and '威廉的姐姐' in history_item:
                    # 提取露尼西亚的回复部分
                    if f"{self.name}:" in history_item:
                        introduction = history_item.split(f"{self.name}:", 1)[1].strip()
                        print(f"🎯 从历史记录找到自我介绍，长度: {len(introduction)} 字符")
                        return introduction
            
            print("⚠️ 未找到自我介绍内容")
            return None
            
        except Exception as e:
            print(f"⚠️ 获取自我介绍失败: {e}")
            return None

    def _simulated_response(self, user_input):
        """当API不可用时使用的模拟响应"""
        # 首先尝试处理工具调用
        tool_response = self._handle_tool_calls(user_input)
        if tool_response:
            return tool_response
        
        # 检查是否是询问"上个"查询
        if any(word in user_input.lower() for word in ['上个', '上一个', '之前', '刚才']):
            # 使用AI生成上下文相关的响应，而不是固定模板
            return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
        
        # 检查是否是简短回答
        if user_input in ['需要', '要', '好的', '可以']:
            # 优先检查最近的对话内容（最近3条）
            recent_conversations = self.session_conversations[-3:] if len(self.session_conversations) >= 3 else self.session_conversations
            
            # 根据上一条消息的内容来判断优先级
            for conv in reversed(recent_conversations):
                conv_text = conv["full_text"].lower()
                
                # 根据上一条消息的具体内容来提供相应的详细回答
                if any(word in conv_text for word in ["俄罗斯方块", "tetris", "pygame", "游戏", "代码", "文件", "保存", "生成", "修复", "错误", "弹窗", "窗口"]):
                    # 使用AI生成代码相关的详细响应，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
                
                elif "python" in conv_text:
                    # 使用AI生成Python相关的详细响应，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
                
                elif any(word in conv_text for word in ["出门", "建议", "天气", "出行", "明天", "上午"]):
                    # 使用AI生成出行建议，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
                
                elif "c语言" in conv_text:
                    # 使用AI生成C语言相关的详细响应，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
                
                elif any(word in conv_text for word in ["埃菲尔铁塔", "法兰克福大教堂", "柏林墙遗址", "布达拉宫", "景点", "旅游", "参观"]):
                    # 使用AI生成景点介绍，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
                
                elif any(word in conv_text for word in ["日文歌", "日文歌曲", "中文歌", "中文歌曲", "音乐", "歌曲", "推荐"]):
                    # 使用AI生成音乐推荐，而不是固定模板
                    return "抱歉，我需要更多信息来理解您的请求。请详细说明您想要了解的内容。"
            
            # 如果没有找到最近的上下文，再检查历史对话中的第一条记忆查询
            for conv in reversed(self.session_conversations):
                conv_text = conv["full_text"].lower()
                
                # 检查是否是询问第一条记忆的上下文
                if "第一条" in conv_text and ("识底深湖" in conv_text or "记忆" in conv_text):
                    # 删除固定模板，让AI使用动态查询
                    pass
            
            return "（轻轻推了推眼镜）指挥官，现在是下午时间。有什么需要我协助的吗？"
        
        # 检查是否是"再推荐几首"
        if "再推荐" in user_input and "几首" in user_input:
            # 使用AI生成更多音乐推荐，而不是固定模板
            return None
        
        # 默认响应
        return "抱歉，AI服务暂时不可用，请检查API配置或稍后重试。"

    def _handle_tool_calls(self, user_input):
        """处理工具调用"""
        # 当处于框架的pass_to_main_agent阶段或其他需要抑制工具的阶段时，直接跳过
        if getattr(self, '_suppress_tool_routing', False):
            print(f"🔧 工具路由已抑制，跳过工具调用: {user_input}")
            return None
        print(f"🔧 检查工具调用: {user_input}")
        user_input_lower = user_input.lower()
        
     
        
        # 搜索逻辑已移至_generate_response_with_context中自动处理
        
        # 🌐 优先处理网页打开与自动化操作请求 - 使用专门的AI识别（优先级最高）
        website_result = self._ai_identify_website_intent(user_input)
        if website_result:
            print(f"🌐 专门的网页打开与自动化操作AI识别成功: {website_result}")
            try:
                # 🎯 特殊处理：如果返回SEARCH_ENGINE，表示要在浏览器直接搜索
                if website_result == "SEARCH_ENGINE":
                    # 简单提取搜索内容（去掉"搜索"、"在浏览器"等关键词）
                    search_query = user_input.replace("搜索", "").replace("在浏览器", "").replace("帮我", "").strip()
                    
                    if search_query:
                        # 构建搜索引擎URL
                        search_engine = self.config.get("default_search_engine", "bing").lower()
                        default_browser = self.config.get("default_browser", "")
                        
                        if search_engine == "bing" or search_engine == "必应":
                            search_url = f"https://cn.bing.com/search?q={search_query}"
                        elif search_engine == "google" or search_engine == "谷歌":
                            search_url = f"https://www.google.com/search?q={search_query}"
                        elif search_engine == "baidu" or search_engine == "百度":
                            search_url = f"https://www.baidu.com/s?wd={search_query}"
                        else:
                            search_url = f"https://cn.bing.com/search?q={search_query}"
                        
                        print(f"🎯 直接使用搜索引擎URL: {search_url}")
                        result = open_website(search_url, default_browser)
                        return f"（指尖轻敲控制台）已在{default_browser if default_browser else '浏览器'}中搜索「{search_query}」"
                
                result = self.tools["网页打开与自动化操作"](website_result, self.website_map, user_input)
                # _open_website_wrapper已经返回完整消息，直接返回，不要再包装
                return result
            except Exception as e:
                return f"（微微皱眉）抱歉指挥官，打开网页时遇到了问题：{str(e)}"
        
        
        
        # 处理打开应用 - 使用AI智能识别
        app_result = self._ai_identify_app_launch_intent(user_input)
        if app_result:
            print(f"📱 AI识别为应用启动请求: {user_input}")
            app_intent, app_name = app_result
            if app_intent == "app_launch":
                # 标准化应用名称
                app_name_mapping = {
                    "网易云音乐": "网易云音乐",
                    "QQ音乐": "QQ音乐", 
                    "酷狗": "酷狗音乐",
                    "酷我": "酷我音乐",
                    "Spotify": "Spotify",
                    "Chrome": "Chrome",
                    "Edge": "Edge",
                    "Firefox": "Firefox",
                    "Word": "Microsoft Word",
                    "Excel": "Microsoft Excel",
                    "PowerPoint": "Microsoft PowerPoint",
                    "记事本": "记事本",
                    "计算器": "计算器",
                    "画图": "画图",
                    "命令提示符": "命令提示符",
                    "PowerShell": "PowerShell"
                }
                
                # 查找匹配的应用名称
                standard_app_name = app_name_mapping.get(app_name, app_name)
                
                try:
                    # 从应用映射中查找应用路径
                    app_path = None
                    for key, path in self.app_map.items():
                        if standard_app_name.lower() in key.lower() or key.lower() in standard_app_name.lower():
                            app_path = path
                            break
                    
                    if app_path:
                        result = self.tools["打开应用"](app_path)
                        return f"（指尖轻敲控制台）{result}"
                    else:
                        # 尝试使用系统命令启动
                        try:
                            if standard_app_name.lower() in ["记事本", "notepad"]:
                                subprocess.Popen("notepad.exe")
                                return f"（指尖轻敲控制台）已启动记事本"
                            elif standard_app_name.lower() in ["计算器", "calculator"]:
                                subprocess.Popen("calc.exe")
                                return f"（指尖轻敲控制台）已启动计算器"
                            elif standard_app_name.lower() in ["画图", "paint"]:
                                subprocess.Popen("mspaint.exe")
                                return f"（指尖轻敲控制台）已启动画图"
                            elif standard_app_name.lower() in ["命令提示符", "cmd"]:
                                subprocess.Popen("cmd.exe")
                                return f"（指尖轻敲控制台）已启动命令提示符"
                            else:
                                return f"（微微皱眉）抱歉指挥官，我没有找到{standard_app_name}的安装路径。请确认该应用已正确安装。"
                        except Exception as e2:
                            return f"（微微皱眉）抱歉指挥官，启动{standard_app_name}时遇到了问题：{str(e2)}"
                except Exception as e:
                    return f"（微微皱眉）抱歉指挥官，启动{standard_app_name}时遇到了问题：{str(e)}"
        
        
        # 处理"查看代码内容"请求
        view_code_keywords = [
            "不需要创建文件", "不要创建文件", "不需要保存文件", "不要保存文件",
            "告诉我代码内容", "显示代码", "只显示代码", "不要直接创建",
            "不需要直接创建", "现在告诉我", "具体代码内容"
        ]
        
        is_view_code_request = any(keyword in user_input.lower() for keyword in view_code_keywords)
        if is_view_code_request:
            print(f"📝 检测到查看代码内容请求: {user_input}")
            # 从最近的对话中提取代码内容并直接返回
            code_content = self._extract_code_from_recent_conversations()
            if code_content:
                return f"好的，指挥官。以下是刚才生成的代码内容：\n\n```java\n{code_content}\n```"
            else:
                return "抱歉，指挥官。我没有找到最近的代码内容。请重新生成代码。"
        
        # 处理文件创建请求（AI智能优先）
        # 首先检查是否是明确的文件创建请求
        file_creation_keywords = ["保存文件", "创建文件", "写入文件", "生成文件", "输出文件", "保存到文件", "创建到文件", "帮我保存", "保存为", "创建为"]
        is_file_creation_request = any(keyword in user_input_lower for keyword in file_creation_keywords)
        
        if is_file_creation_request:
            print(f"✅ 检测到明确的文件创建请求: {user_input}")
            # 尝试AI智能创建文件（优先级最高）
            ai_creation_result = self._ai_create_file_from_context(user_input)
            if ai_creation_result:
                print(f"✅ AI智能创建成功: {ai_creation_result[:50]}...")
                return ai_creation_result
        else:
            print(f"ℹ️ 非文件创建请求，跳过文件创建逻辑: {user_input}")
        
        # 如果是明确的文件创建请求，才尝试AI智能创建代码文件
        if is_file_creation_request:
            ai_code_creation_result = self._ai_create_code_file_from_context(user_input)
            if ai_code_creation_result:
                print(f"✅ AI智能代码创建成功: {ai_code_creation_result[:50]}...")
                return ai_code_creation_result
        
        # 检查是否启用关键词后备识别机制（仅对文件创建请求）
        fallback_enabled = self.config.get("enable_keyword_fallback", True)
        
        if fallback_enabled and is_file_creation_request:
            print(f"🔧 关键词后备识别已启用，进行关键词检测: {user_input}")
            # 如果AI智能创建失败，使用关键词识别作为后备方案
            code_generation_keywords = ["用python写", "用python", "python写", "用c++写", "用c++", "c++写", "用cobol写", "用cobol", "cobol写", "写一个", "创建一个", "帮我写", "帮我创建"]
            save_file_keywords = ["保存文件", "写入文件", "创建文件", "保存文件", "write_file", "create_note"]
            
            # 检查是否是代码生成请求（关键词后备）
            is_code_generation = any(keyword in user_input for keyword in code_generation_keywords)
            is_save_request = any(keyword in user_input for keyword in save_file_keywords)
            
            if is_code_generation or is_save_request:
                print(f"📝 使用关键词后备方案处理: {user_input}")
            
            # 关键词后备的固定格式创建
            # 处理Python代码生成
            if any(word in user_input.lower() for word in ["python", "用python", "python写", "hello world", "hello"]):
                try:
                    import re
                    import os
                    
                    # 智能提取文件名
                    filename = "program.py"  # 默认文件名
                    if "hello world" in user_input.lower() or "hello" in user_input.lower():
                        filename = "hello_world.py"
                    elif "俄罗斯方块" in user_input or "tetris" in user_input.lower():
                        filename = "tetris.py"
                    elif "贪吃蛇" in user_input or "snake" in user_input.lower():
                        filename = "snake_game.py"
                    elif "井字棋" in user_input or "tic-tac-toe" in user_input.lower():
                        filename = "tic_tac_toe.py"
                    elif "小游戏" in user_input or "game" in user_input.lower():
                        filename = "game.py"
                    elif "爬虫" in user_input or "crawler" in user_input.lower():
                        filename = "web_crawler.py"
                    elif "数据分析" in user_input or "data" in user_input.lower():
                        filename = "data_analysis.py"
                    elif "计算器" in user_input or "calculator" in user_input.lower():
                        filename = "calculator.py"
                    
                    # 检查是否指定了保存位置
                    if "d盘" in user_input.lower() or "d:" in user_input.lower():
                        file_path = f"D:/{filename}"
                    elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                        file_path = f"C:/{filename}"
                    else:
                        # 如果没有指定位置，使用当前工作目录
                        current_dir = os.getcwd()
                        file_path = os.path.join(current_dir, filename)
                    
                    # 构建AI提示词，让AI生成Python代码
                    ai_prompt = f"""
请用Python编写一个完整的程序。要求：
1. 根据用户需求生成相应的Python代码
2. 代码要完整可运行
3. 包含必要的注释和文档字符串
4. 使用Python最佳实践
5. 代码逻辑清晰，易于理解

用户需求：{user_input}

请直接返回完整的Python代码，不要包含任何解释文字。
"""
                    
                    # 调用AI API生成代码
                    model = self.config.get("selected_model", "deepseek-chat")
                    api_key = self.config.get("deepseek_key", "") if "deepseek" in model else self.config.get("openai_key", "")

                    if api_key:
                        try:
                            # 设置API客户端
                            if "deepseek" in model:
                                client = openai.OpenAI(
                                    api_key=api_key,
                                    base_url="https://api.deepseek.com/v1"
                                )
                            else:
                                client = openai.OpenAI(api_key=api_key)
                            
                            # 构建系统提示词
                            system_prompt = """你是一个专业的Python程序员。请根据用户需求生成完整、可运行的Python代码。

要求：
1. 只返回Python代码，不要包含任何解释或说明
2. 代码要完整，包含所有必要的导入
3. 使用Python最佳实践和现代语法
4. 代码逻辑清晰，易于理解
5. 添加适当的注释和文档字符串

请直接返回代码，不要有任何其他内容。"""
                            
                            # 创建聊天消息
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": ai_prompt}
                            ]
                            
                            # 调用API（增加超时时间，添加重试机制）
                            max_retries = 3
                            retry_count = 0
                            
                            while retry_count < max_retries:
                                try:
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=messages,
                                        max_tokens=2000,
                                        temperature=0.7,
                                        timeout=240  # 延长AI文件创建的响应时间到240秒
                                    )
                                    python_code = response.choices[0].message.content.strip()
                                    break  # 成功则跳出循环
                                except Exception as e:
                                    retry_count += 1
                                    print(f"AI API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                                    
                                    if retry_count < max_retries:
                                        # 等待一段时间后重试
                                        import time
                                        time.sleep(2 * retry_count)  # 递增等待时间
                                        continue
                                    else:
                                        # 所有重试都失败了
                                        raise e
                            
                            # 如果AI返回的代码包含markdown格式，提取代码部分
                            if "```python" in python_code:
                                import re
                                code_match = re.search(r'```python\s*(.*?)\s*```', python_code, re.DOTALL)
                                if code_match:
                                    python_code = code_match.group(1)
                            elif "```py" in python_code:
                                import re
                                code_match = re.search(r'```py\s*(.*?)\s*```', python_code, re.DOTALL)
                                if code_match:
                                    python_code = code_match.group(1)
                            
                        except Exception as e:
                            print(f"AI API调用失败: {str(e)}")
                            # 如果AI调用失败，返回错误信息
                            return f"（微微皱眉）抱歉指挥官，AI代码生成失败：{str(e)}"
                    else:
                        # 如果没有API密钥，返回提示信息
                        return "（微微皱眉）抱歉指挥官，需要配置AI API密钥才能生成代码。请先配置DeepSeek或OpenAI API密钥。"
                    
                    # 根据用户要求决定是否保存文件
                    if is_save_request:
                        # 用户明确要求保存文件
                        result = self.mcp_server.call_tool("write_file", file_path=file_path, content=python_code)
                        return f"（指尖轻敲控制台）{result}"
                    else:
                        # 用户只是要求生成代码，不保存文件
                        # 智能提取文件名用于显示
                        display_filename = filename
                        if "俄罗斯方块" in user_input or "tetris" in user_input.lower():
                            display_filename = "tetris.py"
                        elif "贪吃蛇" in user_input or "snake" in user_input.lower():
                            display_filename = "snake_game.py"
                        elif "井字棋" in user_input or "tic-tac-toe" in user_input.lower():
                            display_filename = "tic_tac_toe.py"
                        elif "计算器" in user_input or "calculator" in user_input.lower():
                            display_filename = "calculator.py"
                        
                        # 缓存生成的代码，供后续保存使用
                        self.last_generated_code = {
                            'content': python_code,
                            'filename': display_filename,
                            'language': 'python'
                        }
                        
                        return f"（指尖轻敲控制台）我已经为您生成了Python代码。如果您需要保存为文件，请告诉我保存位置，比如'保存到D盘'或'保存为{display_filename}'。\n\n```python\n{python_code}\n```"
                    
                except Exception as e:
                    return f"（微微皱眉）抱歉指挥官，创建Python文件时遇到了问题：{str(e)}"
            
            # 处理C++代码生成
            elif any(word in user_input.lower() for word in ["c++", "cpp", "c++写", "用c++", "c++的"]):
                try:
                    import re
                    import os
                    
                    # 智能提取文件名
                    filename = "game.cpp"  # 默认文件名
                    
                    # 从用户输入中提取游戏类型
                    if "井字棋" in user_input or "tic-tac-toe" in user_input.lower():
                        filename = "tic_tac_toe.cpp"
                    elif "猜数字" in user_input or "number" in user_input.lower():
                        filename = "number_guess.cpp"
                    elif "贪吃蛇" in user_input or "snake" in user_input.lower():
                        filename = "snake_game.cpp"
                    elif "俄罗斯方块" in user_input or "tetris" in user_input.lower():
                        filename = "tetris.cpp"
                    elif "小游戏" in user_input:
                        filename = "mini_game.cpp"
                    
                    # 检查是否指定了保存位置
                    if "d盘" in user_input.lower() or "d:" in user_input.lower():
                        file_path = f"D:/{filename}"
                    elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                        file_path = f"C:/{filename}"
                    else:
                        # 如果没有指定位置，使用当前工作目录
                        current_dir = os.getcwd()
                        file_path = os.path.join(current_dir, filename)
                    
                    # 构建AI提示词，让AI生成C++代码
                    ai_prompt = f"""
请用C++编写一个完整的小游戏程序。要求：
1. 根据用户需求生成相应的游戏代码
2. 代码要完整可编译运行
3. 包含必要的头文件和注释
4. 使用现代C++语法
5. 游戏逻辑清晰，用户体验良好

用户需求：{user_input}

请直接返回完整的C++代码，不要包含任何解释文字。
"""
                    
                    # 调用AI API生成代码
                    model = self.config.get("selected_model", "deepseek-chat")
                    api_key = self.config.get("deepseek_key", "") if "deepseek" in model else self.config.get("openai_key", "")
                    
                    if api_key:
                        try:
                            # 设置API客户端
                            if "deepseek" in model:
                                client = openai.OpenAI(
                                    api_key=api_key,
                                    base_url="https://api.deepseek.com/v1"
                                )
                            else:
                                client = openai.OpenAI(api_key=api_key)
                            
                            # 构建系统提示词
                            system_prompt = """你是一个专业的C++程序员。请根据用户需求生成完整、可编译的C++游戏代码。

要求：
1. 只返回C++代码，不要包含任何解释或说明
2. 代码要完整，包含所有必要的头文件
3. 使用现代C++语法和最佳实践
4. 游戏逻辑清晰，用户体验良好
5. 添加适当的注释说明

请直接返回代码，不要有任何其他内容。"""
                            
                            # 创建聊天消息
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": ai_prompt}
                            ]
                            
                            # 调用API（增加超时时间，添加重试机制）
                            max_retries = 3
                            retry_count = 0
                            
                            while retry_count < max_retries:
                                try:
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=messages,
                                        max_tokens=2000,
                                        temperature=0.7,
                                        timeout=240  # 延长AI文件创建的响应时间到240秒
                                    )
                                    cpp_code = response.choices[0].message.content.strip()
                                    break  # 成功则跳出循环
                                except Exception as e:
                                    retry_count += 1
                                    print(f"AI API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                                    
                                    if retry_count < max_retries:
                                        # 等待一段时间后重试
                                        import time
                                        time.sleep(2 * retry_count)  # 递增等待时间
                                        continue
                                    else:
                                        # 所有重试都失败了
                                        raise e
                            
                            # 如果AI返回的代码包含markdown格式，提取代码部分
                            if "```cpp" in cpp_code:
                                import re
                                code_match = re.search(r'```cpp\s*(.*?)\s*```', cpp_code, re.DOTALL)
                                if code_match:
                                    cpp_code = code_match.group(1)
                            elif "```c++" in cpp_code:
                                import re
                                code_match = re.search(r'```c\+\+\s*(.*?)\s*```', cpp_code, re.DOTALL)
                                if code_match:
                                    cpp_code = code_match.group(1)
                            
                        except Exception as e:
                            print(f"AI API调用失败: {str(e)}")
                            # 如果AI调用失败，返回错误信息
                            return f"（微微皱眉）抱歉指挥官，AI代码生成失败：{str(e)}"
                    else:
                        # 如果没有API密钥，返回提示信息
                        return "（微微皱眉）抱歉指挥官，需要配置AI API密钥才能生成代码。请先配置DeepSeek或OpenAI API密钥。"
                    
                    # 根据用户要求决定是否保存文件
                    if is_save_request:
                        # 用户明确要求保存文件
                        result = self.mcp_server.call_tool("write_file", file_path=file_path, content=cpp_code)
                        return f"（指尖轻敲控制台）{result}"
                    else:
                        # 用户只是要求生成代码，不保存文件
                        # 智能提取文件名用于显示
                        display_filename = filename
                        if "井字棋" in user_input or "tic-tac-toe" in user_input.lower():
                            display_filename = "tic_tac_toe.cpp"
                        elif "贪吃蛇" in user_input or "snake" in user_input.lower():
                            display_filename = "snake_game.cpp"
                        elif "俄罗斯方块" in user_input or "tetris" in user_input.lower():
                            display_filename = "tetris.cpp"
                        elif "猜数字" in user_input or "number" in user_input.lower():
                            display_filename = "number_guess.cpp"
                        elif "小游戏" in user_input:
                            display_filename = "mini_game.cpp"
                        
                        # 缓存生成的代码，供后续保存使用
                        self.last_generated_code = {
                            'content': cpp_code,
                            'filename': display_filename,
                            'language': 'cpp'
                        }
                        
                        return f"（指尖轻敲控制台）我已经为您生成了C++代码。如果您需要保存为文件，请告诉我保存位置，比如'保存到D盘'或'保存为{display_filename}'。\n\n```cpp\n{cpp_code}\n```"
                    
                except Exception as e:
                    return f"（微微皱眉）抱歉指挥官，创建C++文件时遇到了问题：{str(e)}"
            
            # 处理write_file工具调用
            elif "write_file" in user_input.lower() or "写入文件" in user_input or "保存文件" in user_input:
                try:
                    # 提取文件路径和内容
                    import re
                    
                    # 尝试提取路径（支持多种格式）
                    path_patterns = [
                        r'路径为\s*["\']?([^"\']+)["\']?',
                        r'路径\s*["\']?([^"\']+)["\']?',
                        r'file_path\s*=\s*["\']?([^"\']+)["\']?',
                        r'D:[/\\]([^"\s]+)',
                        r'([A-Z]:[/\\][^"\s]+)'
                    ]
                    
                    file_path = None
                    for pattern in path_patterns:
                        match = re.search(pattern, user_input)
                        if match:
                            file_path = match.group(1)
                            if not file_path.startswith(('D:', 'C:', 'E:', 'F:')):
                                file_path = f"D:/{file_path}"
                            break
                    
                    # 提取内容
                    content_patterns = [
                        r'内容为\s*["\']([^"\']+)["\']',
                        r'内容\s*["\']([^"\']+)["\']',
                        r'content\s*=\s*["\']([^"\']+)["\']'
                    ]
                    
                    content = None
                    for pattern in content_patterns:
                        match = re.search(pattern, user_input)
                        if match:
                            content = match.group(1)
                            break
                    
                    # 如果没有找到明确的内容，尝试提取引号中的内容
                    if not content:
                        # 查找所有引号中的内容，排除路径中的内容
                        quote_matches = re.findall(r'["\']([^"\']+)["\']', user_input)
                        for quote_content in quote_matches:
                            if quote_content not in file_path and quote_content != "露尼西亚 测试":
                                content = quote_content
                                break
                        # 如果还是没找到，使用最后一个引号内容
                        if not content and quote_matches:
                            content = quote_matches[-1]
                    
                    if file_path and content:
                        result = self.mcp_server.call_tool("write_file", file_path=file_path, content=content)
                        return f"（指尖轻敲控制台）{result}"
                    else:
                        return f"（微微皱眉）抱歉指挥官，请提供完整的文件路径和内容。格式：路径为D:/文件名.txt，内容为'文件内容'"
                        
                except Exception as e:
                    return f"（微微皱眉）抱歉指挥官，创建文件时遇到了问题：{str(e)}"
            
        # 处理通用保存和文件创建请求（统一优先级）
        elif any(keyword in user_input.lower() for keyword in ["保存", "保存到", "保存为", "写入文件", "创建文件", "创建笔记", "笔记", "清单", "创建测试文件", "创建源文件", "保存到d盘", "保存到d:", "创建清单", "需要创建", "地址在d盘", "地址在d:", "创建好了吗", "保存这个文件", "保存到d盘", "创建可执行", "创建.cbl文件", "创建.py文件", "需要保存", "路径为", "保存为", "创建这个", "这个文件", "地址为", "创建歌单文件", "歌单文件", "创建歌单"]):
            try:
                # 首先检查是否有最近生成的代码需要保存
                if hasattr(self, 'last_generated_code') and self.last_generated_code:
                    # 保存代码逻辑
                    import re
                    import os
                    
                    # 提取保存位置和文件名
                    file_path = None
                    filename = self.last_generated_code.get('filename', 'program.py')
                    
                    # 检查是否指定了保存位置
                    if "d盘" in user_input.lower() or "d:" in user_input.lower():
                        file_path = f"D:/{filename}"
                    elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                        file_path = f"C:/{filename}"
                    else:
                        # 如果没有指定位置，使用当前工作目录
                        current_dir = os.getcwd()
                        file_path = os.path.join(current_dir, filename)
                    
                    # 保存代码
                    content = self.last_generated_code.get('content', '')
                    result = self.mcp_server.call_tool("write_file", file_path=file_path, content=content)
                    
                    # 清除缓存的代码
                    self.last_generated_code = None
                    
                    return f"（指尖轻敲控制台）{result}"
                
                # 如果没有代码需要保存，尝试AI智能创建文件
                ai_creation_result = self._ai_create_file_from_context(user_input)
                if ai_creation_result:
                    return ai_creation_result
                
                # 如果AI创建失败，尝试代码文件创建
                ai_code_creation_result = self._ai_create_code_file_from_context(user_input)
                if ai_code_creation_result:
                    return ai_code_creation_result
                
                # 如果都失败，使用后备方法
                return self._fallback_create_note(user_input)
                    
            except Exception as e:
                return f"（微微皱眉）抱歉指挥官，创建文件时遇到了问题：{str(e)}"
        else:
            # 关键词后备识别已禁用，直接返回None
            print("ℹ️ 关键词后备识别已禁用，跳过关键词检测")
            return None
        
        # 处理天气查询
        if "天气" in user_input:
            # 检查是否是天气评价或分析请求
            weather_evaluation_keywords = [
                "好不好", "怎么样", "如何", "评价", "分析", "认为", "觉得", "感觉", "适合", "不错", "糟糕", "好", "坏"
            ]
            
            is_evaluation_request = any(keyword in user_input for keyword in weather_evaluation_keywords)
            
            if is_evaluation_request:
                # 这是天气评价请求，应该基于最近的天气信息进行分析
                # 检查最近的对话中是否有天气信息
                recent_weather_info = self._get_recent_weather_info()
                if recent_weather_info:
                    return self._analyze_weather_quality(recent_weather_info)
                else:
                    # 如果没有最近的天气信息，先获取天气信息再分析
                    try:
                        user_location = self._extract_city_from_input(user_input)
                        if not user_location:
                            user_location = self._extract_city_from_location(self.location)
                            if not user_location:
                                user_location = "北京"
                        
                        # 根据配置获取天气信息进行分析
                        weather_source = self.config.get("weather_source", "高德地图API")
                        
                        if weather_source == "高德地图API":
                            amap_key = self.config.get("amap_key", "")
                            if amap_key:
                                weather_result = AmapTool.get_weather(user_location, amap_key)
                            else:
                                return "（微微皱眉）高德地图API密钥未配置，无法分析天气"
                        elif weather_source == "和风天气API":
                            heweather_key = self.config.get("heweather_key", "")
                            if heweather_key:
                                weather_result = self.tools["天气"](user_location, heweather_key)
                            else:
                                return "（微微皱眉）和风天气API密钥未配置，无法分析天气"
                        else:
                            amap_key = self.config.get("amap_key", "")
                            if amap_key:
                                weather_result = AmapTool.get_weather(user_location, amap_key)
                            else:
                                return "（微微皱眉）高德地图API密钥未配置，无法分析天气"
                        
                        return self._analyze_weather_quality(weather_result)
                    except Exception as e:
                        return f"（微微皱眉）抱歉指挥官，分析天气时遇到了问题：{str(e)}"
            else:
                # 这是天气查询请求，直接获取天气信息
                try:
                    # 智能提取城市名称
                    user_location = self._extract_city_from_input(user_input)
                    if not user_location:
                        # 使用登录位置作为默认城市
                        user_location = self._extract_city_from_location(self.location)
                        if not user_location:
                            user_location = "北京"  # 最后的默认城市
                    
                    # 根据配置选择天气API
                    weather_source = self.config.get("weather_source", "高德地图API")
                    
                    if weather_source == "高德地图API":
                        # 使用高德地图API内部工具
                        amap_key = self.config.get("amap_key", "")
                        if not amap_key:
                            return "（微微皱眉）高德地图API密钥未配置，请在设置中添加API密钥"
                        
                        result = AmapTool.get_weather(user_location, amap_key)
                        return f"（指尖轻敲控制台）{result}"
                    elif weather_source == "和风天气API":
                        # 使用和风天气API
                        try:
                            # 获取和风天气API密钥
                            heweather_key = self.config.get("heweather_key", "")
                            if not heweather_key:
                                return "（微微皱眉）和风天气API密钥未配置，请在设置中添加API密钥"
                            
                            result = self.tools["天气"](user_location, heweather_key)
                            return f"（指尖轻敲控制台）{result}"
                        except Exception as e2:
                            return f"（微微皱眉）和风天气API调用失败：{str(e2)}"
                    else:
                        # 默认使用高德地图API内部工具
                        amap_key = self.config.get("amap_key", "")
                        if not amap_key:
                            return "（微微皱眉）高德地图API密钥未配置，请在设置中添加API密钥"
                        
                        result = AmapTool.get_weather(user_location, amap_key)
                        return f"（指尖轻敲控制台）{result}"
                except Exception as e:
                    # 如果主要API失败，尝试备用API
                    try:
                        weather_source = self.config.get("weather_source", "高德地图API")
                        if weather_source == "高德地图API":
                            # 高德API失败，尝试和风天气API
                            heweather_key = self.config.get("heweather_key", "")
                            if heweather_key:
                                result = self.tools["天气"](user_location, heweather_key)
                                return f"（指尖轻敲控制台）{result}"
                        else:
                            # 和风天气API失败，尝试高德地图API
                            amap_key = self.config.get("amap_key", "")
                            if amap_key:
                                result = AmapTool.get_weather(user_location, amap_key)
                                return f"（指尖轻敲控制台）{result}"
                    except Exception as e2:
                        return f"（微微皱眉）抱歉指挥官，获取天气信息时遇到了问题：{str(e2)}"
        
        return None

    def _extract_city_from_input(self, user_input):
        """从用户输入中智能提取城市名称"""
        # 常见城市列表
        cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安",
            "天津", "苏州", "长沙", "青岛", "无锡", "宁波", "佛山", "东莞", "郑州", "济南",
            "大连", "福州", "厦门", "哈尔滨", "长春", "沈阳", "石家庄", "太原", "合肥", "南昌",
            "昆明", "贵阳", "南宁", "海口", "兰州", "西宁", "银川", "乌鲁木齐", "拉萨", "呼和浩特"
        ]
        
        # 检查用户输入中是否包含城市名称
        for city in cities:
            if city in user_input:
                return city
        
        return None

    def _extract_city_from_location(self, location):
        """从登录位置中提取城市名称"""
        if not location or location == "未知位置":
            return None
        
        # 城市名称映射（英文 -> 中文）
        city_mapping = {
            "beijing": "北京",
            "shanghai": "上海", 
            "guangzhou": "广州",
            "shenzhen": "深圳",
            "hangzhou": "杭州",
            "nanjing": "南京",
            "wuhan": "武汉",
            "chengdu": "成都",
            "chongqing": "重庆",
            "xian": "西安",
            "tianjin": "天津",
            "suzhou": "苏州",
            "changsha": "长沙",
            "qingdao": "青岛",
            "wuxi": "无锡",
            "ningbo": "宁波",
            "foshan": "佛山",
            "dongguan": "东莞",
            "zhengzhou": "郑州",
            "jinan": "济南",
            "dalian": "大连",
            "fuzhou": "福州",
            "xiamen": "厦门",
            "haerbin": "哈尔滨",
            "changchun": "长春",
            "shenyang": "沈阳",
            "shijiazhuang": "石家庄",
            "taiyuan": "太原",
            "hefei": "合肥",
            "nanchang": "南昌",
            "kunming": "昆明",
            "guiyang": "贵阳",
            "nanning": "南宁",
            "haikou": "海口",
            "lanzhou": "兰州",
            "xining": "西宁",
            "yinchuan": "银川",
            "urumqi": "乌鲁木齐",
            "lasa": "拉萨",
            "huhehaote": "呼和浩特"
        }
        
        # 常见中文城市列表
        chinese_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安",
            "天津", "苏州", "长沙", "青岛", "无锡", "宁波", "佛山", "东莞", "郑州", "济南",
            "大连", "福州", "厦门", "哈尔滨", "长春", "沈阳", "石家庄", "太原", "合肥", "南昌",
            "昆明", "贵阳", "南宁", "海口", "兰州", "西宁", "银川", "乌鲁木齐", "拉萨", "呼和浩特"
        ]
        
        location_lower = location.lower()
        
        # 首先检查中文城市名称
        for city in chinese_cities:
            if city in location:
                return city
        
        # 然后检查英文城市名称
        for english_name, chinese_name in city_mapping.items():
            if english_name in location_lower:
                return chinese_name
        
        return None

    def _direct_create_file_from_extracted_code(self, user_input):
        """直接使用提取的代码创建文件（AI API超时时的后备方案）"""
        try:
            print("🔧 使用直接代码创建后备方案")
            
            # 构建上下文信息
            context_info = ""
            if self.session_conversations:
                # 获取最近的对话作为上下文
                recent_contexts = []
                for conv in reversed(self.session_conversations[-3:]):  # 获取最近3条对话
                    recent_contexts.append(f"【{conv['timestamp']}】{conv['full_text']}")
                context_info = "\n".join(recent_contexts)
            
            # 尝试从上下文中提取代码内容
            extracted_code = self._extract_code_from_context(context_info)
            if not extracted_code:
                print("⚠️ 未找到可提取的代码内容")
                return None
            
            print(f"🔍 直接使用提取的代码: {extracted_code[:100]}...")
            
            # 从用户输入中提取路径信息
            import re
            
            # 尝试提取完整路径（如"路径为D:/计算器.py"）
            path_match = re.search(r'路径为\s*([^，。\s]+)', user_input)
            if path_match:
                full_path = path_match.group(1)
                # 分离路径和文件名
                if '/' in full_path or '\\' in full_path:
                    path_parts = full_path.replace('\\', '/').split('/')
                    if len(path_parts) > 1:
                        location = '/'.join(path_parts[:-1]) + '/'
                        filename = path_parts[-1]
                        if not filename.endswith(('.py', '.cob', '.cbl', '.cpp', '.txt')):
                            filename += '.py'  # 默认添加.py扩展名
                else:
                    location = "D:/"
                    filename = full_path
                    if not filename.endswith(('.py', '.cob', '.cbl', '.cpp', '.txt')):
                        filename += '.py'
            else:
                # 如果没有找到完整路径，使用原有的逻辑
                if "d盘" in user_input.lower() or "d:" in user_input.lower():
                    location = "D:/"
                elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                    location = "C:/"
                elif "e盘" in user_input.lower() or "e:" in user_input.lower():
                    location = "E:/"
                elif "f盘" in user_input.lower() or "f:" in user_input.lower():
                    location = "F:/"
                else:
                    location = "D:/"
                
                # 根据代码内容推断文件名
                if "python" in context_info.lower() or "def " in extracted_code:
                    filename = "calculator.py"
                elif "cobol" in context_info.lower() or "IDENTIFICATION DIVISION" in extracted_code:
                    filename = "program.cob"
                elif "c++" in context_info.lower() or "#include" in extracted_code:
                    filename = "program.cpp"
                else:
                    filename = "program.py"
            
            # 确保文件名安全
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # 构建完整的文件内容
            if "IDENTIFICATION DIVISION" in extracted_code or "PROGRAM-ID" in extracted_code:
                # COBOL代码格式特殊处理
                if "IDENTIFICATION DIVISION" not in extracted_code:
                    file_content = f"""      IDENTIFICATION DIVISION.
      PROGRAM-ID. CALCULATOR.
      PROCEDURE DIVISION.
{extracted_code}
      STOP RUN.
"""
                else:
                    # 如果代码已经包含完整的COBOL结构，直接使用
                    file_content = extracted_code
            else:
                # 其他编程语言
                file_content = f"""# -*- coding: utf-8 -*-
"""
                file_content += extracted_code
            
            # 调用MCP工具创建文件
            file_path = f"{location.rstrip('/')}/{filename}"
            result = self.mcp_server.call_tool("write_file", 
                                             file_path=file_path, 
                                             content=file_content)
            
            return f"（指尖轻敲控制台）{result}"
            
        except Exception as e:
            print(f"直接代码创建失败: {str(e)}")
            return None

    def _extract_java_class_name(self, code: str) -> str:
        """从Java代码中提取主类名"""
        try:
            import re
            # 查找 public class XXX 的模式
            class_pattern = r'public\s+class\s+(\w+)'
            match = re.search(class_pattern, code)
            if match:
                class_name = match.group(1)
                print(f"✅ 提取到Java类名: {class_name}")
                return class_name
            
            # 如果没有public class，查找普通class
            class_pattern2 = r'class\s+(\w+)'
            match2 = re.search(class_pattern2, code)
            if match2:
                class_name = match2.group(1)
                print(f"✅ 提取到Java类名: {class_name}")
                return class_name
            
            print("⚠️ 未找到Java类名")
            return None
        except Exception as e:
            print(f"⚠️ 提取Java类名失败: {e}")
            return None
    
    def _extract_code_from_context(self, context_info):
        """从上下文中提取代码内容 - 严格过滤所有非代码内容"""
        try:
            import re
            
            # 🔥 优先提取```代码块```内的内容
            code_block_pattern = r'```(?:java|python|py|cpp|c\+\+|c|javascript|js|html|css|sql|bash|shell|cobol)?\s*\n(.*?)\n```'
            matches = re.findall(code_block_pattern, context_info, re.DOTALL)
            
            if matches:
                for extracted_code in matches:
                    extracted_code = extracted_code.strip()
                    
                    # 🔥 严格过滤：移除所有说明文字和标记
                    lines = extracted_code.split('\n')
                    filtered_lines = []
                    skip_next = False
                    
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        # 跳过对话标记
                        if any(marker in line for marker in ['指挥官:', '露尼西亚:', '【', '】', '###', '```']):
                            continue
                        
                        # 跳过中文说明行（代码中不应该有完整的中文句子）
                        if len(stripped) > 20 and all(ord(c) > 127 or c in '，。！？、：；' for c in stripped if not c.isspace()):
                            continue
                        
                        # 跳过markdown标题
                        if stripped.startswith('#'):
                            continue
                        
                        # 跳过空行（稍后处理）
                        filtered_lines.append(line)
                    
                    extracted_code = '\n'.join(filtered_lines).strip()
                    
                    # 验证代码是否以合法关键词开头
                    if extracted_code:
                        first_line = extracted_code.split('\n')[0].strip()
                        # 跳过空行找到第一个非空行
                        for line in extracted_code.split('\n'):
                            if line.strip():
                                first_line = line.strip()
                                break
                        
                        valid_starts = ['import', 'package', 'public', 'class', 'private', 
                                       'protected', 'def', 'from', '#include', 'using', '//', '/*',
                                       'interface', 'abstract', 'final', 'static', 'void', 'int',
                                       'String', 'boolean', 'double', 'float', 'char', 'long']
                        
                        if any(first_line.startswith(start) for start in valid_starts):
                            print(f"✅ 成功提取纯代码（首行验证通过）: {first_line[:50]}...")
                            return extracted_code
                        else:
                            print(f"⚠️ 代码首行不合法: '{first_line[:50]}'，继续查找...")
                            continue
                    
                    # 如果没有通过首行验证但有内容，返回
                    if extracted_code and 'class' in extracted_code:
                        print(f"⚠️ 返回未验证的代码块: {extracted_code[:50]}...")
                        return extracted_code
            
            # 如果没有找到代码块，尝试查找COBOL特定的内容
            if "IDENTIFICATION DIVISION" in context_info or "PROGRAM-ID" in context_info:
                # 尝试提取COBOL代码段
                cobol_patterns = [
                    r'(IDENTIFICATION DIVISION\..*?STOP RUN\.)',
                    r'(PROGRAM-ID\..*?STOP RUN\.)',
                    r'(IDENTIFICATION DIVISION\..*?PROCEDURE DIVISION\..*?STOP RUN\.)'
                ]
                
                for pattern in cobol_patterns:
                    matches = re.findall(pattern, context_info, re.DOTALL)
                    if matches:
                        extracted_code = matches[0].strip()
                        print(f"🔍 成功提取COBOL代码: {extracted_code[:50]}...")
                        return extracted_code
            
            print("⚠️ 未找到任何代码内容")
            return None
            
        except Exception as e:
            print(f"提取代码失败: {str(e)}")
            return None

    def _extract_code_from_recent_conversations(self):
        """从最近的对话中提取代码内容"""
        if not self.session_conversations:
            return None
        
        # 从最近的对话中查找代码内容
        for conv in reversed(self.session_conversations[-5:]):  # 检查最近5条对话
            ai_response = conv.get("ai_response", "")
            if "```" in ai_response:
                # 提取代码内容
                code_content = self._extract_code_from_context(ai_response)
                if code_content:
                    return code_content
        
        return None

    def _extract_search_query(self, user_input):
        """智能提取搜索关键词"""
        # 定义需要移除的词汇
        remove_words = [
            "帮我", "请帮我", "麻烦帮我", "能否帮我", "可以帮我",
            "搜索", "查找", "搜素", "搜", "查", "找", "查询", "查找", "搜素",
            "搜索一下", "查找一下", "搜素一下", "搜一下", "查一下", "找一下", "查询一下",
            "一下", "帮我搜索", "帮我查找", "帮我搜素", "帮我搜", "帮我查", "帮我找", "帮我查询", "帮我查找",
            "百度", "google", "谷歌", "bing", "必应", "用百度", "用谷歌", "用必应"
        ]
        
        # 移除所有不需要的词汇
        query = user_input
        for word in remove_words:
            query = query.replace(word, "")
        
        # 清理多余的空格和标点
        import re
        query = re.sub(r'\s+', ' ', query.strip())
        query = query.strip('，。！？、；：')
        
        return query

    def _get_current_time(self):
        """获取当前时间"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    # _extract_automation_action 和 _ai_judge_automation_need 已移至 WebpageAgent

    def _open_website_wrapper(self, site_name, website_map=None, user_input=""):
        """
        网页打开与自动化操作的包装函数，处理网站名称映射
        
        优先级顺序：
        1. AI智能识别并生成URL（从LLM知识库）
        2. 网站管理配置（website_map）
        3. 返回未找到错误
        
        智能判断使用系统浏览器还是Playwright：
        - 简单打开 → 系统浏览器（快速、稳定）
        - 打开+操作 → Playwright有头模式（支持自动化）
        """
        try:
            print(f"🔍 _open_website_wrapper 收到参数 - site_name: '{site_name}', user_input: '{user_input}'")
            
            if website_map is None:
                website_map = self.website_map
            
            # 清理网站名称
            site_name_original = site_name.strip()
            site_name = site_name_original.lower()
            
            # 获取用户配置的默认浏览器
            default_browser = self.config.get("default_browser", "")
            print(f"🔧 使用配置的浏览器: {default_browser}")
            
            # 如果包含http或www，直接作为URL处理
            if site_name.startswith(("http://", "https://", "www.")):
                if not site_name.startswith(("http://", "https://")):
                    site_name = "https://" + site_name
                print(f"🔍 直接作为URL处理: {site_name}")
                
                # 🤖 使用统一ReAct推理Agent（自动判断简单/复杂，自动推理）
                print(f"🤖 调用统一WebpageAgent...")
                
                # 获取Playwright配置
                pw_mode = self.config.get("playwright_mode", "launch")
                pw_slow_mo = self.config.get("playwright_slow_mo", 0)
                pw_cdp_url = self.config.get("playwright_cdp_url", "http://localhost:9222")
                pw_user_data_dir = self.config.get("playwright_user_data_dir", "")
                
                # 如果是connect模式，检查并启动调试浏览器
                if pw_mode == "connect":
                    from cdp_helper import ensure_cdp_connection
                    cdp_result = ensure_cdp_connection(
                        cdp_url=pw_cdp_url,
                        browser_type=default_browser,
                        user_data_dir=pw_user_data_dir
                    )
                    if not cdp_result["success"]:
                        return f"无法建立调试连接：{cdp_result['message']}"
                    if cdp_result.get("auto_started"):
                        print(f"✅ 已自动启动调试浏览器")
                
                # 调用统一Agent（内部自动分析+推理+执行）
                result = execute_webpage_task_sync(
                    config=self.config,
                    user_input=user_input,  # 传递完整用户输入
                    url=site_name,
                    browser_type=default_browser,
                    mode=pw_mode,
                    slow_mo=pw_slow_mo,
                    cdp_url=pw_cdp_url,
                    user_data_dir=pw_user_data_dir
                )
                
                # 处理返回消息
                if result.get("success"):
                    # 🔥 提取完整的页面内容（从history中获取）
                    page_content = ""
                    history = result.get("history", [])
                    
                    # 从历史记录中提取页面内容（优先查找get_text/get_page_info的结果）
                    for record in reversed(history):  # 倒序查找最新内容
                        observation = record.get("observation", "")
                        if "元素文本" in observation or "页面信息" in observation:
                            # 提取文本内容（限制长度避免过长）
                            if len(observation) > 100:  # 只提取有意义的长文本
                                page_content = observation[:3000]  # 最多3000字符
                                break
                    
                    # 构建返回消息
                    basic_message = result.get('message')
                    if page_content:
                        return f"（指尖轻敲控制台）{basic_message}\n\n页面内容：\n{page_content}"
                    else:
                        return f"（指尖轻敲控制台）{basic_message}"
                else:
                    return f"网页操作失败：{result.get('error', '未知错误')}"
            
            # 步骤1：优先使用AI生成URL（从LLM知识库）
            print(f"🤖 步骤1: 尝试使用AI从知识库生成URL: {site_name_original}")
            ai_generated_url = self._ai_generate_website_url(site_name_original)
            
            if ai_generated_url:
                print(f"✅ AI成功生成URL: {ai_generated_url}")
                
                # 🤖 使用统一ReAct推理Agent
                print(f"🤖 调用统一WebpageAgent...")
                
                # 获取Playwright配置
                pw_mode = self.config.get("playwright_mode", "launch")
                pw_slow_mo = self.config.get("playwright_slow_mo", 0)
                pw_cdp_url = self.config.get("playwright_cdp_url", "http://localhost:9222")
                pw_user_data_dir = self.config.get("playwright_user_data_dir", "")
                
                # 如果是connect模式，检查并启动调试浏览器
                if pw_mode == "connect":
                    from cdp_helper import ensure_cdp_connection
                    cdp_result = ensure_cdp_connection(
                        cdp_url=pw_cdp_url,
                        browser_type=default_browser,
                        user_data_dir=pw_user_data_dir
                    )
                    if not cdp_result["success"]:
                        return f"无法建立调试连接：{cdp_result['message']}"
                    if cdp_result.get("auto_started"):
                        print(f"✅ 已自动启动调试浏览器")
                
                # 调用统一Agent（内部自动分析+推理+执行）
                result = execute_webpage_task_sync(
                    config=self.config,
                    user_input=user_input,
                    url=ai_generated_url,
                    browser_type=default_browser,
                    mode=pw_mode,
                    slow_mo=pw_slow_mo,
                    cdp_url=pw_cdp_url,
                    user_data_dir=pw_user_data_dir
                )
                
                # 处理返回消息
                if result.get("success"):
                    return f"（指尖轻敲控制台）{result.get('message')}"
                else:
                    return f"网页操作失败：{result.get('error', '未知错误')}"
            else:
                print(f"⚠️ AI未能从知识库生成URL，尝试下一步")
            
            # 步骤2：从网站管理配置中查找
            print(f"🔧 步骤2: 在网站管理配置中查找: {site_name_original}")
            
            # 处理常见的网站名称变体（用于匹配website_map）
            site_variants = {
                "哔哩哔哩": ["bilibili", "b站", "哔哩哔哩", "bilbil", "bilibili.com"],
                "百度": ["baidu", "百度", "baidu.com"],
                "谷歌": ["google", "谷歌", "google.com"],
                "知乎": ["zhihu", "知乎", "zhihu.com"],
                "github": ["github", "github.com"],
                "youtube": ["youtube", "youtube.com", "油管"],
                "高德开放平台": ["高德", "高德开放平台", "amap", "高德地图"]
            }
            
            # 查找匹配的网站
            matched_site = None
            for site_key, variants in site_variants.items():
                if any(variant in site_name for variant in variants):
                    matched_site = site_key
                    break
            
            # 如果找到匹配的网站，使用映射的URL
            if matched_site and matched_site in website_map:
                url = website_map[matched_site]
                print(f"✅ 在网站管理中找到映射: {site_name} -> {url}")
                
                # 🤖 使用统一ReAct推理Agent
                print(f"🤖 调用统一WebpageAgent...")
                
                # 获取Playwright配置
                pw_mode = self.config.get("playwright_mode", "launch")
                pw_slow_mo = self.config.get("playwright_slow_mo", 0)
                pw_cdp_url = self.config.get("playwright_cdp_url", "http://localhost:9222")
                pw_user_data_dir = self.config.get("playwright_user_data_dir", "")
                
                # 如果是connect模式，检查并启动调试浏览器
                if pw_mode == "connect":
                    from cdp_helper import ensure_cdp_connection
                    cdp_result = ensure_cdp_connection(
                        cdp_url=pw_cdp_url,
                        browser_type=default_browser,
                        user_data_dir=pw_user_data_dir
                    )
                    if not cdp_result["success"]:
                        return f"无法建立调试连接：{cdp_result['message']}"
                    if cdp_result.get("auto_started"):
                        print(f"✅ 已自动启动调试浏览器")
                
                # 调用统一Agent
                result = execute_webpage_task_sync(
                    config=self.config,
                    user_input=user_input,
                    url=url,
                    browser_type=default_browser,
                    mode=pw_mode,
                    slow_mo=pw_slow_mo,
                    cdp_url=pw_cdp_url,
                    user_data_dir=pw_user_data_dir
                )
                
                # 处理返回消息
                if result.get("success"):
                    return f"（指尖轻敲控制台）{result.get('message')}"
                else:
                    return f"网页操作失败：{result.get('error', '未知错误')}"
            
            # 如果网站名称直接匹配映射表（不区分大小写）
            for map_key, map_url in website_map.items():
                if site_name == map_key.lower() or site_name_original == map_key:
                    print(f"✅ 在网站管理中直接匹配: {site_name_original} -> {map_url}")
                    
                    # 🤖 使用统一ReAct推理Agent
                    print(f"🤖 调用统一WebpageAgent...")
                    
                    # 获取Playwright配置
                    pw_mode = self.config.get("playwright_mode", "launch")
                    pw_slow_mo = self.config.get("playwright_slow_mo", 0)
                    pw_cdp_url = self.config.get("playwright_cdp_url", "http://localhost:9222")
                    pw_user_data_dir = self.config.get("playwright_user_data_dir", "")
                    
                    # 如果是connect模式，检查并启动调试浏览器
                    if pw_mode == "connect":
                        from cdp_helper import ensure_cdp_connection
                        cdp_result = ensure_cdp_connection(
                            cdp_url=pw_cdp_url,
                            browser_type=default_browser,
                            user_data_dir=pw_user_data_dir
                        )
                        if not cdp_result["success"]:
                            return f"无法建立调试连接：{cdp_result['message']}"
                        if cdp_result.get("auto_started"):
                            print(f"✅ 已自动启动调试浏览器")
                    
                    # 调用统一Agent
                    result = execute_webpage_task_sync(
                        config=self.config,
                        user_input=user_input,
                        url=map_url,
                        browser_type=default_browser,
                        mode=pw_mode,
                        slow_mo=pw_slow_mo,
                        cdp_url=pw_cdp_url,
                        user_data_dir=pw_user_data_dir
                    )
                    
                    # 处理返回消息
                    if result.get("success"):
                        return f"（指尖轻敲控制台）{result.get('message')}"
                    else:
                        return f"网页操作失败：{result.get('error', '未知错误')}"
            
            # 步骤3：都没找到，返回错误信息
            print(f"❌ 步骤3: 在AI知识库和网站管理中都未找到网站")
            available_sites = list(website_map.keys()) if website_map else []
            
            error_msg = f"（微微皱眉）抱歉指挥官，我无法识别网站 '{site_name_original}'。\n\n"
            
            if available_sites:
                error_msg += f"💡 网站管理中的可用网站：\n"
                for idx, site in enumerate(available_sites, 1):
                    error_msg += f"   {idx}. {site}\n"
                error_msg += f"\n"
            
            error_msg += f"📝 您可以：\n"
            error_msg += f"   1. 在设置中的【网站管理】添加此网站\n"
            error_msg += f"   2. 直接提供完整网址（如：https://www.example.com）\n"
            error_msg += f"   3. 尝试使用更常见的网站名称"
            
            return error_msg
            
        except Exception as e:
            return f"打开网页时发生错误：{str(e)}"

    def _is_remember_moment_command(self, user_input):
        """检测是否是'记住这个时刻'指令"""
        remember_keywords = [
            "请记住这个时刻",
            "记住这个时刻",
            "记住这一刻",
            "请记住这一刻",
            "记住这个瞬间",
            "请记住这个瞬间",
            "记住这个时间",
            "请记住这个时间",
            "记住这个对话",
            "请记住这个对话",
            "记住这次谈话",
            "请记住这次谈话",
            "记住这次交流",
            "请记住这次交流",
            "保存这个时刻",
            "请保存这个时刻",
            "保存这次对话",
            "请保存这次对话",
            "记录这个时刻",
            "请记录这个时刻",
            "记录这次对话",
            "请记录这次对话"
        ]
        
        user_input_lower = user_input.lower().strip()
        return any(keyword.lower() in user_input_lower for keyword in remember_keywords)

    def _handle_remember_moment(self, user_input):
        """处理'记住这个时刻'指令"""
        try:
            # 检查是否有未保存的会话对话
            unsaved_conversations = []
            
            # 获取当前记忆系统中的对话数量
            current_memory_count = len(self.memory_lake.current_conversation)
            
            # 获取本次会话的对话数量
            session_count = len(self.session_conversations)
            
            # 如果本次会话有对话但记忆系统中没有，说明有未保存的对话
            if session_count > 0 and current_memory_count == 0:
                # 将本次会话的所有对话添加到记忆系统
                for conv in self.session_conversations:
                    self.memory_lake.add_conversation(conv["user_input"], conv["ai_response"])
                    unsaved_conversations.append(conv["full_text"])
            
            # 强制保存到识底深湖
            if self.memory_lake.current_conversation:
                topic = self.memory_lake.summarize_and_save_topic(force_save=True)
                
                if topic:
                    # 标记为重点记忆（最新保存的记忆是最后一个）
                    topics = self.memory_lake.memory_index.get("topics", [])
                    if topics:
                        latest_index = len(topics) - 1
                        self.memory_lake.mark_as_important(latest_index)
                    
                    # 构建响应消息
                    response = f"（轻轻点头）好的指挥官，我已经将这个重要时刻记录到识底深湖中，并标记为重点记忆。"
                    
                    # 根据设置决定是否显示详细信息
                    show_details = self.config.get("show_remember_details", True)
                    
                    if show_details:
                        if unsaved_conversations:
                            response += f"\n\n已保存的对话内容：\n"
                            for i, conv in enumerate(unsaved_conversations, 1):
                                response += f"{i}. {conv}\n"
                        
                        response += f"\n主题：{topic}\n时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    # 清空本次会话记录，因为已经保存到记忆系统
                    self.session_conversations = []
                    
                    return response
                else:
                    return "（微微皱眉）抱歉指挥官，保存到识底深湖时遇到了一些问题。请稍后再试。"
            else:
                return "（轻轻摇头）指挥官，目前没有需要保存的对话内容。请先进行一些对话，然后再说'记住这个时刻'。"
                
        except Exception as e:
            print(f"处理'记住这个时刻'指令失败: {str(e)}")
            return "（表情略显困扰）抱歉指挥官，保存过程中遇到了一些技术问题。请稍后再试。"

    def _is_file_analysis_request(self, user_input):
        """检测是否是文件分析请求"""
        file_keywords = [
            "分析文件", "文件分析", "上传文件", "分析图片", "分析文档",
            "查看文件", "文件信息", "图片信息", "文档信息", "智能分析"
        ]
        user_input_lower = user_input.lower().strip()
        return any(keyword in user_input_lower for keyword in file_keywords)

    def _handle_file_analysis(self, user_input):
        """处理文件分析请求"""
        try:
            # 从用户输入中提取文件路径
            import re
            file_path_pattern = r'[A-Za-z]:[\\/][^\\/:\*\?"<>\|]*\.(pdf|csv|xlsx|xls)'
            matches = re.findall(file_path_pattern, user_input)
            
            if matches:
                file_path = matches[0]
                print(f"🔍 检测到文件路径: {file_path}")
                
                # 使用文件分析工具分析文件
                result = self.file_analyzer.analyze_file(file_path)
                
                if result.success:
                    # 生成AI分析报告
                    analysis_report = self.file_analyzer.generate_ai_analysis(result, user_input)
                    return analysis_report
                else:
                    return f"❌ 文件分析失败: {result.error}"
            else:
                return "（困惑地看着屏幕）指挥官，我没有在您的消息中检测到文件路径。请提供完整的文件路径，例如：C:\\Users\\Documents\\example.pdf"
        except Exception as e:
            return f"❌ 文件分析出错: {str(e)}"

    def _check_file_context_query(self, user_input):
        """检查用户问题是否与最近分析的文件相关，并让AI判断"""
        try:
            file_info = self.recent_file_analysis
            
            print(f"📂 检测到最近分析的文件上下文: {file_info['file_name']}")
            print(f"🤔 让AI判断问题是否与文件相关...")
            
            # 第一步：让AI判断问题是否与文件相关
            judge_prompt = f"""你刚刚分析了一个文件：{file_info['file_name']} ({file_info['file_type']})

现在用户提出了一个问题："{user_input}"

请判断这个问题是否与刚才分析的文件相关。

判断标准：
1. 如果问题明确提到"文件"、"代码"、"刚才"、"这个"等指代刚分析的文件
2. 如果问题询问代码结构、数量统计（如循环数、函数数）
3. 如果问题是对文件内容的追问或延伸讨论
4. 如果问题很简短且像是对上一次分析的追问（如"里边用了几个循环"）

请只回答 "YES" 或 "NO"，不要有其他内容。"""
            
            judge_response = self._call_ai_api(judge_prompt, max_tokens=10)
            
            if judge_response and "YES" in judge_response.upper():
                print(f"✅ AI判断：问题与文件 {file_info['file_name']} 相关")
                
                # 构建包含文件内容的提示词
                prompt = f"""你刚刚分析了一个文件，现在用户对这个文件提出了问题。

**文件信息：**
- 文件名：{file_info['file_name']}
- 文件类型：{file_info['file_type']}
"""
                
                # 如果是代码文件，优先展示精确统计信息
                if 'CODE_' in file_info['file_type']:
                    metadata = file_info.get('metadata', {})
                    structure = metadata.get('structure', {})
                    metrics = metadata.get('metrics', {})
                    
                    prompt += f"\n**📊 精确代码统计（请优先使用这些数据回答问题）：**\n"
                    
                    if metrics:
                        prompt += f"\n控制结构统计：\n"
                        if_count = metrics.get('if_count', metrics.get('if_statements', 0))
                        for_count = metrics.get('for_count', metrics.get('for_loops', 0))
                        while_count = metrics.get('while_count', metrics.get('while_loops', 0))
                        try_count = metrics.get('try_blocks', 0)
                        
                        prompt += f"- if语句：{if_count} 个\n"
                        prompt += f"- for循环：{for_count} 个\n"
                        prompt += f"- while循环：{while_count} 个\n"
                        if try_count > 0:
                            prompt += f"- try块：{try_count} 个\n"
                        
                        total_loops = for_count + while_count
                        prompt += f"- **循环总数：{total_loops} 个** (for: {for_count}, while: {while_count})\n"
                        
                        prompt += f"\n代码规模：\n"
                        prompt += f"- 总行数：{metrics.get('total_lines', 0)} 行\n"
                        prompt += f"- 有效代码：{metrics.get('code_lines', 0)} 行\n"
                        prompt += f"- 注释：{metrics.get('comment_lines', 0)} 行\n"
                        prompt += f"- 空行：{metrics.get('blank_lines', 0)} 行\n"
                    
                    if structure:
                        prompt += f"\n代码结构：\n"
                        if structure.get('classes'):
                            prompt += f"- 类定义：{len(structure['classes'])} 个\n"
                            # 列出类名
                            class_names = [cls.get('name', 'Unknown') for cls in structure['classes']]
                            prompt += f"  类名：{', '.join(class_names[:10])}\n"
                        
                        if structure.get('functions'):
                            prompt += f"- 函数/方法：{len(structure['functions'])} 个\n"
                        
                        if structure.get('imports'):
                            prompt += f"- 导入模块：{len(structure['imports'])} 个\n"
                
                prompt += f"\n**文件摘要：**\n{file_info['summary']}\n"
                prompt += f"\n**文件分析：**\n{file_info['analysis']}\n"
                
                # 只在必要时添加部分内容（避免过长）
                if len(file_info.get('content', '')) < 3000:
                    prompt += f"\n**文件内容（部分）：**\n{file_info['content'][:3000]}\n"
                
                prompt += f"\n**用户问题：**\n{user_input}\n"
                prompt += f"\n⚠️ 重要：如果用户问及数量（如循环数、函数数等），请直接使用上面的精确统计数据回答，不要猜测！\n"
                
                # 调用AI生成回答
                response = self._call_ai_api(prompt, max_tokens=1000)
                
                if response:
                    return response
            else:
                print(f"❌ AI判断：问题与文件无关，使用常规处理流程")
            
            return None
            
        except Exception as e:
            print(f"❌ 文件上下文查询失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_file_upload(self, file_path):
        """处理文件上传"""
        try:
            print(f"🔍 开始分析文件: {file_path}")
            
            # 使用文件分析工具分析文件
            result = self.file_analyzer.analyze_file(file_path)
            
            if result.success:
                print(f"✅ 文件分析完成: {result.file_name}")
                
                # 🔥 保存到上下文记忆（用于后续问答）
                self.recent_file_analysis = {
                    "file_name": result.file_name,
                    "file_type": result.file_type,
                    "content": result.content,
                    "metadata": result.metadata,
                    "summary": result.summary,
                    "analysis": result.analysis,
                    "result": result  # 保存完整结果对象
                }
                print(f"💾 已保存文件分析上下文: {result.file_name}")
                
                # 生成AI分析报告
                analysis_report = self.file_analyzer.generate_ai_analysis(result)
                return analysis_report
            else:
                error_msg = f"❌ 文件分析失败: {result.error}"
                print(error_msg)
                return error_msg
            
        except Exception as e:
            print(f"❌ 文件分析失败: {str(e)}")
            return f"文件分析失败: {str(e)}"
    
    def _format_analysis_result(self, result):
        """格式化分析结果，使其更美观易读"""
        try:
            import json
            import re
            
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis_data = json.loads(json_str)
                
                # 格式化基本信息
                basic_info = analysis_data.get("basic_info", {})
                content_analysis = analysis_data.get("content_analysis", {})
                
                formatted_result = "🔍 智能文件分析结果\n"
                formatted_result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                formatted_result += f"📁 文件名：{basic_info.get('file_name', '未知')}\n"
                formatted_result += f"📏 文件大小：{basic_info.get('file_size_human', '未知')}\n"
                formatted_result += f"📅 创建时间：{basic_info.get('created_time', '未知')}\n"
                formatted_result += f"🔄 修改时间：{basic_info.get('modified_time', '未知')}\n"
                
                # 根据文件类型添加特定信息
                if content_analysis.get("type") == "image":
                    formatted_result += f"🖼️ 图片格式：{content_analysis.get('format', '未知')}\n"
                    formatted_result += f"📐 图片尺寸：{content_analysis.get('width', '未知')} × {content_analysis.get('height', '未知')}\n"
                    formatted_result += f"🎨 颜色深度：{content_analysis.get('color_depth', '未知')}\n"
                    
                    # 场景描述
                    scene_desc = content_analysis.get("scene_description", {})
                    if scene_desc:
                        formatted_result += f"🌅 场景类型：{scene_desc.get('scene_type', '未知')}\n"
                        formatted_result += f"💡 亮度水平：{scene_desc.get('brightness_level', '未知')}\n"
                    
                    # 物体检测
                    object_detect = content_analysis.get("object_detection", {})
                    if object_detect:
                        formatted_result += f"🔍 复杂度：{object_detect.get('complexity', '未知')}\n"
                        formatted_result += f"🎨 颜色数量：{object_detect.get('unique_colors', '未知')}\n"
                    
                    # 文字提取分析
                    text_extract = content_analysis.get("text_extraction", {})
                    if text_extract:
                        formatted_result += f"📝 文字可能性：{text_extract.get('text_likelihood', '未知')}\n"
                        formatted_result += f"📊 边缘密度：{text_extract.get('edge_density', '未知')}\n"
                    
                    # OCR文字识别结果
                    ocr_text = content_analysis.get("ocr_text", {})
                    if ocr_text and ocr_text.get("status") == "success":
                        extracted_text = ocr_text.get("extracted_text", "")
                        if extracted_text.strip():
                            formatted_result += f"🔤 识别文字：\n"
                            # 限制显示长度，避免过长
                            display_text = extracted_text.strip()
                            if len(display_text) > 200:
                                display_text = display_text[:200] + "..."
                            formatted_result += f"   {display_text}\n"
                            formatted_result += f"📏 文字长度：{ocr_text.get('text_length', '未知')}字符\n"
                            formatted_result += f"📖 词数：{ocr_text.get('word_count', '未知')}\n"
                    elif ocr_text and ocr_text.get("status") == "no_text":
                        formatted_result += f"🔤 文字识别：未识别到文字内容\n"
                    elif ocr_text and ocr_text.get("status") == "error":
                        formatted_result += f"🔤 文字识别：{ocr_text.get('message', '识别失败')}\n"
                    
                    # 颜色分析
                    color_analysis = content_analysis.get("color_analysis", {})
                    if color_analysis:
                        dominant_colors = color_analysis.get("dominant_colors", [])
                        if dominant_colors:
                            formatted_result += f"🌈 主要颜色：{dominant_colors[0].get('color', '未知')} ({dominant_colors[0].get('percentage', '未知')}%)\n"
                    
                    # 构图分析
                    composition = content_analysis.get("composition_analysis", {})
                    if composition:
                        formatted_result += f"📐 构图类型：{composition.get('composition_type', '未知')}\n"
                        formatted_result += f"📊 分辨率：{composition.get('resolution_quality', '未知')}\n"
                
                elif content_analysis.get("type") == "text":
                    formatted_result += f"📄 文件类型：文本文件\n"
                    formatted_result += f"📝 字符数：{content_analysis.get('character_count', '未知')}\n"
                    formatted_result += f"📖 行数：{content_analysis.get('line_count', '未知')}\n"
                    formatted_result += f"🔤 词数：{content_analysis.get('word_count', '未知')}\n"
                    formatted_result += f"🌍 语言：{content_analysis.get('language', '未知')}\n"
                
                formatted_result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                
                return formatted_result
            else:
                # 如果没有找到JSON，返回原始结果
                return result
                
        except Exception as e:
            print(f"⚠️ 格式化分析结果失败: {str(e)}")
            # 如果格式化失败，返回原始结果
            return result

    def _generate_image_ai_analysis(self, file_path, analysis_result):
        """生成图片的AI分析"""
        try:
            print(f"🖼️ 开始生成图片AI分析: {file_path}")
            
            # 尝试解析分析结果
            import json
            try:
                analysis_data = json.loads(analysis_result)
                content_analysis = analysis_data.get("content_analysis", {})
                
                # 获取OCR识别的文字内容
                ocr_text = content_analysis.get("ocr_text", {})
                extracted_text = ""
                if ocr_text and ocr_text.get("status") == "success":
                    extracted_text = ocr_text.get("extracted_text", "").strip()
                
                # 构建AI分析提示
                prompt = f"""
                请分析这张图片，基于以下信息：
                
                图片信息：
                - 文件名：{analysis_data.get('basic_info', {}).get('file_name', '未知')}
                - 尺寸：{content_analysis.get('width', '未知')} x {content_analysis.get('height', '未知')}
                - 格式：{content_analysis.get('format', '未知')}
                
                内容分析：
                - 场景描述：{content_analysis.get('scene_description', {}).get('description', '未知')}
                - 物体检测：{content_analysis.get('object_detection', {}).get('description', '未知')}
                - 颜色分析：{content_analysis.get('color_analysis', {}).get('description', '未知')}
                - 构图分析：{content_analysis.get('composition_analysis', {}).get('description', '未知')}
                """
                
                # 如果有OCR识别的文字内容，添加到提示中
                if extracted_text:
                    prompt += f"""
                
                OCR识别的文字内容：
                {extracted_text}
                
                请基于以上信息，特别是OCR识别的文字内容，对这张图片进行全面的AI分析。包括：
                1. 图片的整体内容和主题
                2. 识别出的文字内容的含义和重要性
                3. 图片的风格、用途和可能的背景
                4. 文字与图片内容的关联性分析
                5. 专业见解和建议
                """
                else:
                    prompt += f"""
                
                文字识别：{content_analysis.get('text_extraction', {}).get('description', '未知')}
                
                请从AI的角度分析这张图片的内容、风格、可能的用途等，给出专业的见解。
                """
                
            except json.JSONDecodeError:
                # 如果不是JSON格式，使用原始结果
                print(f"⚠️ 分析结果不是JSON格式，使用原始结果")
                prompt = f"""
                请分析这张图片，基于以下技术分析结果：
                
                {analysis_result}
                
                请从AI的角度分析这张图片的内容、风格、可能的用途等，给出专业的见解。
                """
            
            # 调用AI生成分析，提供空的上下文信息
            context_info = {}
            response = self._generate_response_with_context(prompt, context_info)
            return response
            
        except Exception as e:
            print(f"❌ AI分析生成失败: {str(e)}")
            return f"AI分析生成失败: {str(e)}"

    def _generate_document_ai_analysis(self, file_path, analysis_result):
        """生成文档的AI分析"""
        try:
            # 解析分析结果
            import json
            analysis_data = json.loads(analysis_result)
            content_analysis = analysis_data.get("content_analysis", {})
            
            # 构建AI分析提示
            prompt = f"""
            请分析这个文档，基于以下信息：
            
            文档信息：
            - 文件名：{analysis_data.get('basic_info', {}).get('file_name', '未知')}
            - 文件类型：{content_analysis.get('type', '未知')}
            
            内容分析：
            - 文本统计：{content_analysis.get('description', '未知')}
            - 关键词：{', '.join(content_analysis.get('keywords', []))}
            - 内容预览：{content_analysis.get('content_preview', '未知')}
            
            请从AI的角度分析这个文档的主题、内容质量、可能的用途等，给出专业的见解。
            """
            
            # 调用AI生成分析，提供空的上下文信息
            context_info = {}
            response = self._generate_response_with_context(prompt, context_info)
            return response
            
        except Exception as e:
            return f"AI分析生成失败: {str(e)}"

    def _is_image_file(self, file_path):
        """判断是否为图片文件"""
        from pathlib import Path
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions

    def _is_document_file(self, file_path):
        """判断是否为文档文件"""
        from pathlib import Path
        document_extensions = {'.pdf', '.txt', '.doc', '.docx', '.csv', '.json', '.xml'}
        return Path(file_path).suffix.lower() in document_extensions

    
    def _filter_ocr_text(self, text):
        """过滤OCR识别的文字，去除明显错误的结果"""
        if not text:
            return ""
        
        import re
        
        # 去除单个字符或明显无意义的字符组合
        if len(text.strip()) < 2:
            return ""
        
        # 去除只包含数字和特殊字符的文本（除非是合理的数字）
        if re.match(r'^[\d\s\-\.\,]+$', text.strip()) and len(text.strip()) < 5:
            return ""
        
        # 去除重复字符过多的文本
        if len(set(text)) < len(text) * 0.3:  # 如果重复字符超过70%
            return ""
        
        # 去除明显无意义的字符组合
        meaningless_patterns = [
            r'^[^\w\s]+$',  # 只包含特殊字符
            r'^[a-zA-Z]{1,2}$',  # 单个或两个英文字母
            r'^[一-龯]{1,2}$',  # 单个或两个中文字符
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text.strip()):
                return ""
        
        # 清理文本
        cleaned_text = text.strip()
        # 去除多余的空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text

    def process_image(self, file_path):
        """处理图片文件"""
        try:
            print(f"🖼️ 开始处理图片: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return "错误：文件不存在"
            
            # 检查是否为图片文件
            if not self._is_image_file(file_path):
                return "错误：不是有效的图片文件"
            
            # 使用智能文件分析工具
            analysis_result = self._analyze_file_with_tools(file_path)
            
            if not analysis_result:
                return "错误：文件分析失败"
            
            # 生成AI分析
            ai_analysis = self._generate_image_ai_analysis(file_path, analysis_result)
            
            return ai_analysis
            
        except Exception as e:
            print(f"❌ 图片处理失败: {str(e)}")
            return f"图片处理失败: {str(e)}"

    def _analyze_file_with_tools(self, file_path):
        """使用工具分析文件"""
        try:
            # 调用MCP工具进行文件分析
            result = self.mcp_tools.server.call_tool("智能文件分析", file_path=file_path)
            return result
        except Exception as e:
            print(f"❌ 文件分析工具调用失败: {str(e)}")
            return None

    def _get_recent_weather_info(self):
        """获取最近的天气信息"""
        # 从最近的对话中查找天气信息
        for conv in reversed(self.session_conversations):
            ai_response = conv.get("ai_response", "")
            if "天气预报" in ai_response or "天气" in ai_response:
                return ai_response
        return None

    def _analyze_weather_quality(self, weather_info):
        """分析天气质量并给出评价"""
        try:
            # 解析天气信息
            weather_text = weather_info.lower()
            
            # 提取关键信息
            temperature = None
            weather_condition = None
            wind = None
            
            # 提取温度信息
            import re
            temp_match = re.search(r'(\d+)°c', weather_text)
            if temp_match:
                temperature = int(temp_match.group(1))
            
            # 提取天气状况
            if "晴" in weather_text:
                weather_condition = "晴"
            elif "多云" in weather_text:
                weather_condition = "多云"
            elif "阴" in weather_text:
                weather_condition = "阴"
            elif "雨" in weather_text:
                weather_condition = "雨"
            elif "雪" in weather_text:
                weather_condition = "雪"
            
            # 提取风力信息
            wind_match = re.search(r'([东南西北]风\d+-\d+级)', weather_text)
            if wind_match:
                wind = wind_match.group(1)
            
            # 分析天气质量
            analysis = "（快速分析天气数据）"
            
            # 温度评价
            if temperature:
                if temperature < 10:
                    temp_eval = "偏冷"
                elif temperature < 20:
                    temp_eval = "凉爽"
                elif temperature < 28:
                    temp_eval = "舒适"
                elif temperature < 35:
                    temp_eval = "较热"
                else:
                    temp_eval = "炎热"
            else:
                temp_eval = "适中"
            
            # 天气状况评价
            if weather_condition == "晴":
                condition_eval = "晴朗宜人"
            elif weather_condition == "多云":
                condition_eval = "温和舒适"
            elif weather_condition == "阴":
                condition_eval = "略显沉闷"
            elif weather_condition == "雨":
                condition_eval = "需要注意防雨"
            elif weather_condition == "雪":
                condition_eval = "需要注意保暖"
            else:
                condition_eval = "天气一般"
            
            # 综合评价
            if temperature and weather_condition:
                if temperature >= 20 and temperature <= 28 and weather_condition in ["晴", "多云"]:
                    overall_eval = "非常好的天气"
                    recommendation = "适合户外活动、出行和运动"
                elif temperature >= 15 and temperature <= 30 and weather_condition in ["晴", "多云", "阴"]:
                    overall_eval = "不错的天气"
                    recommendation = "适合日常活动和出行"
                elif weather_condition == "雨":
                    overall_eval = "需要注意的天气"
                    recommendation = "建议携带雨具，注意防滑"
                elif temperature < 10 or temperature > 35:
                    overall_eval = "需要适应的天气"
                    recommendation = "注意保暖或防暑降温"
                else:
                    overall_eval = "一般的天气"
                    recommendation = "根据个人情况安排活动"
            else:
                overall_eval = "天气状况一般"
                recommendation = "建议关注实时天气变化"
            
            # 构建分析结果
            analysis += f"\n\n🌤️ 天气质量分析\n"
            analysis += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            if temperature:
                analysis += f"🌡️ 温度评价：{temp_eval} ({temperature}°C)\n"
            if weather_condition:
                analysis += f"☁️ 天气状况：{condition_eval}\n"
            if wind:
                analysis += f"💨 风力情况：{wind}\n"
            analysis += f"\n📊 综合评价：{overall_eval}\n"
            analysis += f"💡 建议：{recommendation}\n"
            analysis += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            return analysis
            
        except Exception as e:
            return f"（微微皱眉）抱歉指挥官，分析天气时遇到了问题：{str(e)}"

    def update_tts_config(self, config):
        """更新TTS配置"""
        try:
            from tts_manager import TTSManager
            
            azure_key = config.get("azure_tts_key", "")
            azure_region = config.get("azure_region", "eastasia")
            
            # 如果TTS管理器不存在，创建新的
            if not hasattr(self, 'tts_manager') or self.tts_manager is None:
                self.tts_manager = TTSManager(azure_key, azure_region)
                print("✅ TTS管理器已创建")
            else:
                # 更新现有TTS配置
                self.tts_manager.update_config(azure_key, azure_region)
                print("✅ TTS配置已更新")
            
            # 如果TTS已启用，设置语音和语速
            if config.get("tts_enabled", False):
                self.tts_manager.set_voice(config.get("tts_voice", "zh-CN-XiaoxiaoNeural"))
                self.tts_manager.set_speaking_rate(config.get("tts_speaking_rate", 1.0))
                print("✅ TTS功能已启用")
            else:
                print("ℹ️ TTS功能已禁用")
                
        except Exception as e:
            print(f"⚠️ TTS配置更新失败: {str(e)}")
            self.tts_manager = None
    
    def stop_tts(self):
        """停止TTS播放"""
        if hasattr(self, 'tts_manager'):
            self.tts_manager.stop_speaking()
    
    def cleanup_tts(self):
        """清理TTS资源"""
        if hasattr(self, 'tts_manager'):
            self.tts_manager.cleanup()
    
    def test_tts(self):
        """测试TTS功能"""
        if hasattr(self, 'tts_manager') and self.tts_manager:
            return self.tts_manager.test_tts("你好，这是露尼西亚的TTS测试")
        else:
            print("❌ TTS管理器未初始化")
            return False

    def _simple_parse_file_info(self, user_input, context_info):
        """简单解析文件信息（AI智能优先）"""
        try:
            print(f"🔍 开始AI智能解析文件信息: {user_input}")
            
            file_info = {
                "title": "未命名文件",
                "filename": "未命名文件.txt",
                "location": "D:/",
                "content": context_info
            }
            
            # 从用户输入和上下文中提取旅游目的地
            destination = self._extract_travel_destination(user_input, context_info)
            
            # 从用户输入中提取信息
            if "旅游" in user_input or "旅行" in user_input or "旅游计划" in user_input or "攻略" in user_input:
                if destination:
                    file_info["title"] = f"{destination}旅游攻略"
                    file_info["filename"] = f"{destination}旅游攻略.txt"
                else:
                    file_info["title"] = "旅游攻略"
                    file_info["filename"] = "旅游攻略.txt"
                
                # 从上下文中提取旅游计划内容
                if destination and destination in context_info:
                    # 提取包含目的地的内容
                    lines = context_info.split('\n')
                    relevant_lines = []
                    for line in lines:
                        if destination in line or "旅游" in line or "旅行" in line or "攻略" in line or "景点" in line or "行程" in line:
                            relevant_lines.append(line)
                    if relevant_lines:
                        file_info["content"] = "\n".join(relevant_lines)
                    else:
                        file_info["content"] = context_info
                else:
                    file_info["content"] = context_info
            elif "音乐" in user_input or "歌单" in user_input or "歌曲" in user_input:
                # 用户明确要求音乐相关文件
                file_info["title"] = "音乐推荐"
                file_info["filename"] = "音乐推荐.txt"
                file_info["content"] = context_info
            elif "保存" in user_input:
                # 🔥 优先检查上下文中是否有代码
                has_code = False
                code_lang = None
                extracted_code = None
                
                # 从上下文中提取代码
                if "```" in context_info:
                    extracted_code = self._extract_code_from_context(context_info)
                    if extracted_code:
                        has_code = True
                        # 检测代码语言
                        if "```java" in context_info:
                            code_lang = "java"
                        elif "```python" in context_info or "```py" in context_info:
                            code_lang = "python"
                        elif "```cpp" in context_info or "```c++" in context_info:
                            code_lang = "cpp"
                        elif "```javascript" in context_info or "```js" in context_info:
                            code_lang = "javascript"
                
                # 如果有代码，设置为代码文件
                if has_code and code_lang:
                    if code_lang == "java":
                        # 🔥 智能提取Java类名
                        class_name = self._extract_java_class_name(extracted_code)
                        if class_name:
                            file_info["title"] = class_name
                            file_info["filename"] = f"{class_name}.java"
                        else:
                            file_info["title"] = "JavaProgram"
                            file_info["filename"] = "JavaProgram.java"
                        file_info["file_type"] = "java"
                    elif code_lang == "python":
                        file_info["title"] = "Python代码"
                        file_info["filename"] = "program.py"
                        file_info["file_type"] = "py"
                    elif code_lang == "cpp":
                        file_info["title"] = "C++代码"
                        file_info["filename"] = "program.cpp"
                        file_info["file_type"] = "cpp"
                    elif code_lang == "javascript":
                        file_info["title"] = "JavaScript代码"
                        file_info["filename"] = "script.js"
                        file_info["file_type"] = "js"
                    
                    file_info["content"] = extracted_code
                    print(f"✅ 从上下文提取到{code_lang}代码，文件名: {file_info['filename']}")
                
                # 检查用户是否明确指定了文件类型
                elif ".py" in user_input.lower() or "python" in user_input.lower():
                    file_info["title"] = "Python代码"
                    file_info["filename"] = "Python代码.py"
                elif ".cpp" in user_input.lower() or "c++" in user_input.lower():
                    file_info["title"] = "C++代码"
                    file_info["filename"] = "C++代码.cpp"
                elif ".java" in user_input.lower():
                    file_info["title"] = "Java代码"
                    file_info["filename"] = "Java代码.java"
                elif ".js" in user_input.lower() or "javascript" in user_input.lower():
                    file_info["title"] = "JavaScript代码"
                    file_info["filename"] = "JavaScript代码.js"
                elif ".txt" in user_input.lower():
                    # 用户明确要求txt文件，根据上下文内容确定类型
                    if "音乐" in context_info or "歌" in context_info or "歌曲" in context_info or "推荐" in context_info:
                        file_info["title"] = "音乐推荐"
                        file_info["filename"] = "音乐推荐.txt"
                    elif "旅游" in context_info or "旅行" in context_info or "攻略" in context_info:
                        file_info["title"] = "旅游攻略"
                        file_info["filename"] = "旅游攻略.txt"
                    elif "代码" in context_info or "程序" in context_info or "```" in context_info:
                        file_info["title"] = "代码文件"
                        file_info["filename"] = "代码文件.txt"
                    else:
                        file_info["title"] = "文档"
                        file_info["filename"] = "文档.txt"
                else:
                    # 🚀 使用统一的文件保存信息识别（一次AI调用获取所有信息）
                    print(f"✅ 用户说'帮我保存'，调用统一文件保存识别Agent")
                    ai_save_info = self._ai_identify_file_save_info(user_input, context_info)
                    if ai_save_info:
                        print(f"✅ 统一识别成功: {ai_save_info.get('filename')}")
                        # 应用AI识别的结果
                        if "title" in ai_save_info:
                            file_info["title"] = ai_save_info["title"]
                        if "filename" in ai_save_info:
                            file_info["filename"] = ai_save_info["filename"]
                        if "file_type" in ai_save_info:
                            file_info["file_type"] = ai_save_info["file_type"]
                        if "location" in ai_save_info:
                            file_info["location"] = ai_save_info["location"]
                        if "content" in ai_save_info:
                            file_info["content"] = ai_save_info["content"]
                    else:
                        print(f"⚠️ 统一识别失败，使用关键词识别后备方案")
                        # 关键词识别后备方案 - 优先检查当前对话的上下文
                        # 检查是否包含旅游相关内容
                        if any(keyword in context_info for keyword in ["旅游", "旅行", "攻略", "景点", "行程"]):
                            # 🚀 智能提取目的地名称 - 优先从用户问题中提取
                            destinations = [
                                "法兰克福", "贝尔格莱德", "柏林", "塔林", "巴黎", "伦敦", "罗马", "东京", "纽约",
                                "阿姆斯特丹", "巴塞罗那", "维也纳", "布拉格", "布达佩斯", "华沙", "莫斯科", "圣彼得堡",
                                "伊斯坦布尔", "迪拜", "新加坡", "曼谷", "首尔", "悉尼", "墨尔本", "温哥华", "多伦多"
                            ]
                            
                            destination = None
                            
                            # 首先尝试从用户问题中提取（优先级最高）
                            user_question = ""
                            for conv in self.session_conversations[-3:]:  # 检查最近3轮对话
                                if "旅游" in conv.get("user_input", "") or "攻略" in conv.get("user_input", ""):
                                    user_question = conv.get("user_input", "")
                                    break
                            
                            if user_question:
                                for dest in destinations:
                                    if dest in user_question:
                                        destination = dest
                                        print(f"✅ 从用户问题中提取到目的地: {destination}")
                                        break
                            
                            # 如果用户问题中没有找到，再从上下文中查找
                            if not destination:
                                for dest in destinations:
                                    if dest in context_info:
                                        destination = dest
                                        print(f"✅ 从上下文中提取到目的地: {destination}")
                                        break
                            
                            if destination:
                                file_info["title"] = f"{destination}旅游攻略"
                                file_info["filename"] = f"{destination}旅游攻略.txt"
                                print(f"✅ 生成文件名: {file_info['filename']}")
                            else:
                                file_info["title"] = "旅游攻略"
                                file_info["filename"] = "旅游攻略.txt"
                                print(f"⚠️ 未找到具体目的地，使用通用名称")
                        elif any(keyword in context_info for keyword in ["代码", "程序", "```", "python", "c++", "java"]):
                            file_info["title"] = "代码文件"
                            file_info["filename"] = "代码文件.txt"
                        elif any(keyword in context_info for keyword in ["笔记", "记录", "备忘"]):
                            file_info["title"] = "笔记"
                            file_info["filename"] = "笔记.txt"
                        else:
                            # 如果都无法确定，使用AI智能识别的结果
                            file_info["title"] = "文档"
                            file_info["filename"] = "文档.txt"
                file_info["content"] = context_info
            elif "笔记" in user_input:
                file_info["title"] = "笔记"
                file_info["filename"] = "笔记.txt"
                file_info["content"] = context_info
            elif "代码" in user_input or "程序" in user_input or "python" in user_input.lower():
                # 根据编程语言确定文件扩展名
                if "python" in user_input.lower() or "py" in user_input.lower():
                    file_info["title"] = "Python代码"
                    file_info["filename"] = "Python代码.py"
                elif "c++" in user_input.lower() or "cpp" in user_input.lower():
                    file_info["title"] = "C++代码"
                    file_info["filename"] = "C++代码.cpp"
                elif "java" in user_input.lower():
                    file_info["title"] = "Java代码"
                    file_info["filename"] = "Java代码.java"
                elif "javascript" in user_input.lower() or "js" in user_input.lower():
                    file_info["title"] = "JavaScript代码"
                    file_info["filename"] = "JavaScript代码.js"
                else:
                    file_info["title"] = "代码文件"
                    file_info["filename"] = "代码文件.txt"
                file_info["content"] = context_info
            else:
                file_info["title"] = "文档"
                file_info["filename"] = "文档.txt"
                file_info["content"] = context_info
            
            # 🚀 使用统一的文件保存信息识别（包含路径识别）
            # 注意：这里可能已经在上面的代码分支中调用过统一识别，避免重复
            # 这里主要处理其他文件类型的路径识别
            if not file_info.get("location"):
                import re
                
                # 优先检查用户是否明确指定了路径
                if "d盘" in user_input.lower() or "d:" in user_input.lower():
                    file_info["location"] = "D:/"
                elif "c盘" in user_input.lower() or "c:" in user_input.lower():
                    file_info["location"] = "C:/"
                else:
                    # 匹配各种路径格式
                    path_patterns = [
                        r'保存到\s*([A-Za-z]:[^，。\s]*)',  # 保存到D:\测试_
                        r'保存到\s*([A-Za-z]:[^，。\s]*)',  # 保存到D:/测试_
                        r'位置在\s*([A-Za-z]:[^，。\s]*)',  # 位置在D:\测试_
                        r'位置\s*是\s*([A-Za-z]:[^，。\s]*)',  # 位置是D:\测试_
                        r'([A-Za-z]:[^，。\s]*)',  # 直接说D:\测试_
                    ]
                    
                    extracted_path = None
                    for pattern in path_patterns:
                        match = re.search(pattern, user_input, re.IGNORECASE)
                        if match:
                            extracted_path = match.group(1)
                            break
                    
                    if extracted_path:
                        # 标准化路径格式
                        extracted_path = extracted_path.replace('\\', '/')
                        if not extracted_path.endswith('/'):
                            extracted_path += '/'
                        file_info["location"] = extracted_path
                    else:
                        # 使用默认保存路径
                        default_path = self.config.get("default_save_path", "D:/露尼西亚文件/")
                        if default_path and os.path.exists(default_path):
                            file_info["location"] = default_path
                        else:
                            # 如果默认路径不存在，尝试创建
                            try:
                                os.makedirs(default_path, exist_ok=True)
                                file_info["location"] = default_path
                            except:
                                # 如果创建失败，使用D盘根目录
                                file_info["location"] = "D:/"
            
            print(f"🔍 简单解析结果: {file_info['title']} -> {file_info['filename']} -> {file_info['location']}")
            return file_info
            
        except Exception as e:
            print(f"❌ 简单解析失败: {str(e)}")
            return None

    def _is_valid_path(self, path):
        """验证路径是否有效"""
        try:
            import re
            # 检查是否是有效的Windows路径格式
            if re.match(r'^[A-Za-z]:[/\\]', path):
                return True
            # 检查是否是相对路径
            elif path.startswith('./') or path.startswith('../'):
                return True
            # 检查是否是网络路径
            elif path.startswith('\\\\'):
                return True
            else:
                return False
        except:
            return False

    def _extract_travel_destination(self, user_input, context_info):
        """从用户输入和上下文中提取旅游目的地"""
        # 常见的旅游目的地
        destinations = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "西安",
            "香港", "澳门", "台湾", "日本", "韩国", "泰国", "新加坡", "马来西亚", "越南",
            "美国", "加拿大", "英国", "法国", "德国", "意大利", "西班牙", "澳大利亚", "新西兰"
        ]
        
        # 从用户输入中查找目的地
        for dest in destinations:
            if dest in user_input:
                return dest
        
        # 从上下文中查找目的地
        for dest in destinations:
            if dest in context_info:
                return dest
        
        return None

    def _analyze_user_request_type(self, user_input):
        """分析用户请求的类型 - 优先使用框架Agent传递的内容类型"""
        user_input_lower = user_input.lower()
        
        # 🔥 优先检查框架Agent是否已经识别了内容类型
        if hasattr(self, 'file_save_content_type') and self.file_save_content_type:
            content_type = self.file_save_content_type
            print(f"✅ 使用框架Agent识别的内容类型: {content_type}")
            
            # 映射content_type到request_type并清除标记
            self.file_save_content_type = None  # 使用后清除
            
            if content_type == "code":
                return "code_file"
            elif content_type == "music":
                return "music_file"
            elif content_type == "travel":
                return "travel_file"
            else:
                return "general_file"
        
        # 🔥 优先检查最近对话中是否有代码 - 如果有代码，且用户说"保存"，应该识别为code_file
        if any(keyword in user_input_lower for keyword in ["保存", "创建文件", "写入文件"]):
            # 检查最近的对话中是否有代码块
            has_code_in_context = False
            code_language = None
            
            if self.session_conversations:
                for conv in reversed(self.session_conversations[-3:]):
                    ai_response = conv.get("ai_response", "")
                    if "```" in ai_response:
                        has_code_in_context = True
                        # 检测代码语言
                        if "```java" in ai_response:
                            code_language = "java"
                        elif "```python" in ai_response or "```py" in ai_response:
                            code_language = "python"
                        elif "```cpp" in ai_response or "```c++" in ai_response:
                            code_language = "cpp"
                        elif "```javascript" in ai_response or "```js" in ai_response:
                            code_language = "javascript"
                        break
            
            # 如果最近对话中有代码，优先识别为代码文件
            if has_code_in_context:
                print(f"🔍 检测到上下文中有{code_language or ''}代码，识别为code_file")
                return "code_file"
        
        # 明确的文件创建请求
        file_creation_keywords = ["保存文件", "创建文件", "写入文件", "生成文件", "输出文件", "保存到文件", "创建到文件"]
        if any(keyword in user_input_lower for keyword in file_creation_keywords):
            # 进一步判断是什么类型的文件
            if any(keyword in user_input_lower for keyword in ["音乐", "歌", "歌曲", "歌单"]):
                return "music_file"
            elif any(keyword in user_input_lower for keyword in ["旅游", "旅行", "攻略", "景点"]):
                return "travel_file"
            elif any(keyword in user_input_lower for keyword in ["代码", "程序", "c++", "python", "java"]):
                return "code_file"
            elif any(keyword in user_input_lower for keyword in ["笔记", "记录", "备忘"]):
                return "note_file"
            elif any(keyword in user_input_lower for keyword in ["文件夹"]) and "创建文件夹" in user_input_lower:
                # 只有明确说"创建文件夹"才识别为folder，避免"目录"误触发
                return "folder"
            else:
                return "general_file"
        
        # 代码展示请求（不是文件创建）
        code_display_keywords = ["帮我写", "写一个", "用c++写", "用python写", "用java写", "写个", "帮我用"]
        if any(keyword in user_input_lower for keyword in code_display_keywords):
            return "code_display"
        
        # 音乐相关请求
        music_keywords = ["音乐", "歌", "歌曲", "歌单", "播放", "推荐音乐", "推荐"]
        if any(keyword in user_input_lower for keyword in music_keywords):
            return "music"
        
        # 旅游相关请求
        travel_keywords = ["旅游", "旅行", "攻略", "景点", "行程", "酒店", "机票"]
        if any(keyword in user_input_lower for keyword in travel_keywords):
            return "travel"
        
        # 笔记相关请求
        note_keywords = ["笔记", "记录", "备忘", "清单", "计划"]
        if any(keyword in user_input_lower for keyword in note_keywords):
            return "note"
        
        # 文件夹相关请求 - 只有明确说"创建文件夹"才识别
        if "创建文件夹" in user_input_lower or "新建文件夹" in user_input_lower:
            return "folder"
        
        return "unknown"
