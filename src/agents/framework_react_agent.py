#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
框架ReAct Agent - 轻量级任务规划协调器
只负责制定执行框架，具体操作由主Agent完成
"""

import json
from typing import Dict, Any, List, Optional

class FrameworkReActAgent:
    """框架ReAct Agent - 任务分解和协调"""
    
    def __init__(self, base_agent, intent_model: str = "deepseek-chat"):
        """
        初始化框架Agent
        
        Args:
            base_agent: 基础AIAgent实例
            intent_model: 意图识别使用的模型
        """
        self.base_agent = base_agent
        self.intent_model = intent_model
        self.max_steps = 15  # 最大步数
        self.current_framework = []  # 当前框架
        self.completed_steps = []  # 已完成的步骤
        
    def _ai_identify_file_creation_intent(self, user_input: str) -> tuple:
        """
        使用AI快速识别用户是否想要创建/保存文件，并识别保存内容类型
        
        Args:
            user_input: 用户输入
            
        Returns:
            (bool, str): (是否文件创建请求, 内容类型:code/music/travel/general)
        """
        try:
            import openai
            
            # 使用轻量级chat模型
            model = "deepseek-chat"
            api_key = self.base_agent.config.get("deepseek_key", "")
            
            if not api_key:
                print("⚠️ 无API密钥，无法识别文件创建意图，继续框架流程")
                return (False, "")  # 返回False让框架继续处理
            
            # 🔥 获取最近对话上下文，判断要保存什么内容
            recent_context = ""
            if self.base_agent.session_conversations:
                for conv in self.base_agent.session_conversations[-2:]:
                    user_msg = conv.get('user_input', '')
                    ai_resp = conv.get('ai_response', '')
                    has_code = "```" in ai_resp
                    has_music = any(kw in ai_resp for kw in ["推荐", "音乐", "歌曲", "歌单"])
                    recent_context += f"用户: {user_msg}\nAI回复特征: [包含代码={has_code}, 包含音乐={has_music}]\n\n"
            
            prompt = f"""判断用户是否想要创建或保存文件，并识别要保存的内容类型。

用户输入：{user_input}

最近对话上下文：
{recent_context}

判断标准：
1. **是否是文件创建请求**：
   - 明确的保存操作："保存"、"创建文件"、"写入文件"、"保存到" → YES
   - 只请求内容不说保存："推荐音乐"、"写代码"（不说保存） → NO

2. **识别保存内容类型**（如果是保存请求）：
   - 用户明确指出："保存代码" → content_type="code"
   - 用户明确指出："保存歌单"/"保存音乐" → content_type="music"
   - 用户没明确指出，默认使用**最近的**内容：
     * 上文AI回复包含音乐 → content_type="music"
     * 上文AI回复包含代码 → content_type="code"
     * 否则 → content_type="general"

返回JSON：
{{
    "is_file_creation": true/false,
    "content_type": "code/music/travel/general"
}}

只返回JSON，不要其他内容。"""

            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是文件创建意图识别助手。只返回JSON。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1,
                timeout=15
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🔍 [文件创建意图识别] AI返回: {result}")
            
            # 解析JSON
            import json
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            result_dict = json.loads(result)
            is_file_creation = result_dict.get("is_file_creation", False)
            content_type = result_dict.get("content_type", "general")
            
            print(f"🔍 [文件创建意图识别] 判断: 创建={is_file_creation}, 类型={content_type}")
            return (is_file_creation, content_type)
            
        except Exception as e:
            print(f"⚠️ AI文件创建意图识别失败: {e}，返回False继续框架流程")
            return (False, "")
    
    def _fast_path_open_website(self, user_input: str) -> Optional[List[Dict[str, Any]]]:
        """检测是否属于纯"打开网站/网页"的简单请求，返回最小执行框架。
        触发条件示例：
        - 打开哔哩哔哩 / 打开bilibili / 打开 bilibili.com
        - 去知乎 / 打开百度
        - open youtube / go to github

        若检测到简单导航意图，则仅规划：
        1) get_url_from_website_map（提取URL）
        2) call_playwright_react（执行打开，仅传入原始user_input防止额外动作）
        3) pass_to_main_agent（总结）
        """
        text = user_input.strip().lower()
        # 关键词启发式（尽量保守，减少误判）
        trigger_keywords = ["打开", "open", "go to", "进入", "去", "上", "访问"]
        site_indicators = [".com", ".cn", ".net", ".org", "bilibili", "哔哩", "b站", "baidu", "google", "知乎", "zhihu", "github", "youtube", "优酷", "youku"]

        is_simple = any(k in user_input for k in trigger_keywords) and any(s in user_input for s in site_indicators)
        # 明确排除包含搜索、登录、点击、播放量等操作词的复杂场景
        complex_indicators = ["搜索", "search", "登录", "login", "点击", "click", "播放量", "highest", "排序", "sort", "下载", "download"]
        if any(c in user_input for c in complex_indicators):
            return None

        # 如果是纯域名或带http的也视为简单
        if ("http://" in text) or ("https://" in text) or text.startswith("www."):
            is_simple = True

        if not is_simple:
            return None

        # 使用AI提取网站名称（不是整个用户输入）
        site_name = self.base_agent._ai_identify_website_intent(user_input)
        
        if not site_name:
            print(f"⚠️ 无法从'{user_input}'中提取网站名称")
            return None
        
        print(f"✅ [网站名称提取] 从'{user_input}' 提取到 '{site_name}'")

        return [
            {"description": f"获取『{site_name}』的URL", "action": "get_url_from_website_map", "params": {"website_name": site_name}},
            {"description": "在浏览器中打开该网站", "action": "call_playwright_react", "params": {"url": "从上一步获取的URL"}},
            {"description": "总结并回复用户", "action": "pass_to_main_agent", "params": {}}
        ]

    def process_command(self, user_input: str) -> str:
        """
        使用框架ReAct模式处理命令
        
        工作流程：
        1. 制定执行框架
        2. 逐步执行框架
        3. 动态调整框架（如果需要）
        4. 返回最终结果
        """
        print("\n" + "="*60)
        print("🧠 [框架ReAct] 启动任务规划引擎")
        print("="*60)
        
        # 0. 使用AI识别文件创建意图和内容类型
        is_file_creation, content_type = self._ai_identify_file_creation_intent(user_input)
        if is_file_creation:
            print(f"ℹ️ AI识别为文件创建请求（内容类型: {content_type}），直接交回主Agent处理")
            # 将识别的内容类型传递给主Agent
            self.base_agent.file_save_content_type = content_type
            return self.base_agent.process_command(user_input, skip_framework=True)

        # 第一步：针对简单“只打开网站”的需求，走快速通道，避免多余动作
        simple_framework = self._fast_path_open_website(user_input)
        if simple_framework:
            framework = simple_framework
        else:
            # 常规：调用规划模型制定执行框架
            framework = self._plan_framework(user_input)
        
        if not framework:
            print("❌ 无法制定执行框架，使用标准模式")
            return None
        
        self.current_framework = framework
        total_steps = len(framework)
        
        print(f"\n📋 [执行框架] 共 {total_steps} 步")
        for i, step in enumerate(framework, 1):
            print(f"  [{i}] {step.get('description', 'N/A')} (action: {step.get('action', 'None')})")
        print("")
        
        # 逐步执行框架
        collected_info = {}  # 收集的信息
        
        for step_idx, step in enumerate(framework, 1):
            print(f"\n{'='*60}")
            print(f"🎯 [第 {step_idx}/{total_steps} 步] {step['description']}")
            print(f"{'='*60}")
            
            # 执行这一步
            result = self._execute_step(step, user_input, collected_info)
            
            print(f"✅ [完成] {result[:200]}{'...' if len(result) > 200 else ''}")
            
            # 保存结果
            collected_info[f"step_{step_idx}"] = result
            self.completed_steps.append({
                "step": step_idx,
                "description": step['description'],
                "action": step.get('action', ''),  # 🔥 保存action字段，用于后续判断
                "result": result
            })
            
            # 检查是否需要调整框架
            if step_idx < total_steps:
                should_adjust = self._should_adjust_framework(user_input, collected_info, framework[step_idx:])
                if should_adjust:
                    print(f"\n🔄 [框架调整] 根据当前进展重新规划后续步骤...")
                    new_framework = self._adjust_framework(user_input, collected_info, framework[step_idx:])
                    if new_framework:
                        # 更新框架
                        framework = framework[:step_idx] + new_framework
                        total_steps = len(framework)
                        print(f"📋 [新框架] 更新为 {total_steps} 步")
                        for i, s in enumerate(framework[step_idx:], step_idx + 1):
                            print(f"  [{i}] {s['description']}")
        
        # 生成最终回答
        print(f"\n{'='*60}")
        print(f"✅ [框架执行完成] 共完成 {len(self.completed_steps)} 步")
        print(f"{'='*60}\n")
        
        final_answer = self._generate_final_answer(user_input, collected_info)
        return final_answer
    
    def _plan_framework(self, user_input: str) -> List[Dict[str, Any]]:
        """
        制定执行框架
        
        Args:
            user_input: 用户输入
            
        Returns:
            框架列表 [{"description": "步骤描述", "action": "action_type", "params": {...}}]
        """
        prompt = f"""你是一个任务规划专家，需要为用户的请求制定执行框架。

用户请求：{user_input}

请分析用户的请求，制定执行框架。

**可用的操作类型：**
1. get_weather - 获取天气信息（直接调用天气API）
2. get_location - 获取位置信息
3. search_web - 搜索网络信息
4. analyze_file - 分析最近上传的文件
5. open_application - 打开应用程序
6. get_url_from_website_map - 从网站管理或AI知识库获取网站URL
7. call_playwright_react - 调用Playwright ReAct Agent执行网页自动化
8. use_mcp_tool - 使用MCP工具
9. pass_to_main_agent - 将信息传递给主Agent（用于最终回答）

**规划原则：**
1. **步数完全自由**：根据任务复杂度自主决定，可以是1步、3步、8步或任意数量
2. **工具选择智能**：
   - 简单对话 → pass_to_main_agent（1步即可）
   - 天气查询 → get_weather + pass_to_main_agent
   - 网页操作 → get_url_from_website_map + call_playwright_react + pass_to_main_agent
   - 文件追问 → analyze_file + pass_to_main_agent
   - 信息查询 → search_web + pass_to_main_agent（最多2步，避免重复搜索）
   - 复杂任务 → 多个工具组合，但避免重复相同类型的搜索
   - **代码生成类任务** → 直接返回null（让主Agent处理）
3. **最后一步必须是pass_to_main_agent**：将收集的信息传给主Agent生成回答
4. **避免重复**：不要规划多个相同类型的search_web步骤，一次搜索即可

**特别注意 - 直接交给主Agent的任务类型：**
如果用户请求属于以下类型，请直接返回 null（表示不需要框架规划，交给主Agent处理）：
- **代码生成**：写代码、用Python写、用Java写、用C++写、生成代码、编写程序等
- **文件创建**：保存文件、创建文件、写入文件等（已在前置检查中处理）
- **纯AI对话**：不需要任何工具调用的简单对话
- **音乐/电影/书籍推荐**：推荐音乐、推荐电影、推荐书籍等（主Agent可以直接生成）
- **创意内容生成**：写诗、写故事、写文章等（主Agent的创作能力）

识别标准：
- 包含"写"、"写个"、"写一个"、"帮我写"、"生成代码"、"编写"等词
- 明确提到编程语言：Python、Java、C++、JavaScript、Go等
- 要求HelloWorld、计算器、游戏等代码示例
- **推荐类请求**："推荐音乐"、"推荐歌曲"、"推荐电影"、"推荐书籍"
- **创作类请求**："写首诗"、"写个故事"、"帮我想个文案"

示例（应返回null）：
- "帮我用Java写个helloworld" → null
- "用Python写一个计算器" → null  
- "写个Python爬虫" → null
- "生成一个C++程序" → null
- **"推荐几首音乐" → null**
- **"推荐一些好听的歌" → null**
- **"帮我推荐几本书" → null**

**返回格式要求（严格遵守）：**
返回一个JSON数组，每个元素必须包含：
- "description": 步骤描述
- "action": 操作类型（从上面9种操作中选择）
- "params": 参数对象（可为空{{}}）

⚠️ 注意：字段名必须是"action"，不是"operation"或其他！

用户问题：{user_input}

请规划执行框架（JSON数组格式，只返回JSON，不要有其他内容）：
"""
        
        # 直接调用OpenAI API（因为base_agent没有统一的_call_ai_api方法）
        try:
            import openai
            
            # 获取API密钥
            if "deepseek" in self.intent_model:
                api_key = self.base_agent.config.get("deepseek_key", "")
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            else:
                api_key = self.base_agent.config.get("openai_key", "")
                client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": "你是任务规划专家，擅长将复杂任务分解为清晰的执行步骤。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                timeout=15
            )
            
            response = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return None
        
        try:
            # 清理响应
            response = response.strip()
            
            # 检查AI是否返回null（表示应该交给主Agent处理）
            if response.lower() in ["null", "none", "空"]:
                print("ℹ️ AI规划模型建议直接交给主Agent处理")
                return None
            
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # 再次检查是否为null
            if response.lower() in ["null", "none", "空"]:
                print("ℹ️ AI规划模型建议直接交给主Agent处理")
                return None
            
            framework = json.loads(response)
            
            # 调试：打印解析后的框架
            print(f"🔍 [调试] AI规划的框架: {json.dumps(framework, ensure_ascii=False, indent=2)}")
            
            # 如果返回空数组，说明不需要框架
            if not framework or len(framework) == 0:
                return None
            
            return framework
            
        except json.JSONDecodeError as e:
            print(f"❌ 框架解析失败: {e}")
            print(f"原始响应: {response[:200]}")
            return None
    
    def _execute_step(self, step: Dict, user_input: str, collected_info: Dict) -> str:
        """
        执行框架中的一步
        
        Args:
            step: 步骤定义
            user_input: 原始用户输入
            collected_info: 已收集的信息
            
        Returns:
            执行结果
        """
        action = step.get("action")
        params = step.get("params", {})
        
        try:
            if action == "get_location":
                location = self.base_agent.location
                return f"位置：{location}"
            
            elif action == "get_url_from_website_map":
                # 从网站管理或AI知识库获取URL
                # 支持多种可能的参数名：name, website, website_name
                site_name = (
                    params.get("name") or 
                    params.get("website") or 
                    params.get("website_name") or 
                    ""
                )
                print(f"    🔍 查找网站URL: {site_name}")
                print(f"    🔍 params内容: {params}")
                
                # 🔥 优先检查：如果用户输入或site_name中已包含完整URL，直接提取返回
                import re
                # 检查用户输入
                url_pattern = r'https?://[^\s\u4e00-\u9fff]+'  # 匹配http(s)://开头到中文或空格前的URL
                url_match = re.search(url_pattern, user_input)
                if url_match:
                    extracted_url = url_match.group(0)
                    # 移除末尾可能的中文字符
                    extracted_url = re.sub(r'[\u4e00-\u9fff]+$', '', extracted_url)
                    print(f"    ✅ 从用户输入中直接提取URL: {extracted_url}")
                    return f"获取到URL: {extracted_url}"
                
                # 检查site_name参数
                url_match = re.search(url_pattern, site_name)
                if url_match:
                    extracted_url = url_match.group(0)
                    extracted_url = re.sub(r'[\u4e00-\u9fff]+$', '', extracted_url)
                    print(f"    ✅ 从site_name参数中直接提取URL: {extracted_url}")
                    return f"获取到URL: {extracted_url}"

                # 占位/泛化词过滤：避免将"相关社交媒体平台"等占位词当成真实网站
                placeholder_indicators = [
                    "相关社交媒体平台", "相关平台", "相关网站", "某平台", "某网站", "社交平台", "社交媒体平台"
                ]
                if any(ind in site_name for ind in placeholder_indicators):
                    return "❌ 未提供明确网站名称，已跳过获取URL"
                
                # 优先从网站管理中查找
                website_map = self.base_agent.website_map
                url = website_map.get(site_name)
                
                # 如果没有，尝试AI生成
                if not url:
                    print(f"    🤖 网站管理中未找到，尝试AI生成URL...")
                    url = self.base_agent._ai_generate_website_url(site_name)
                    if url:
                        print(f"    ✅ AI成功生成URL: {url}")
                
                if url:
                    return f"获取到URL: {url}"
                else:
                    return f"❌ 无法找到网站 {site_name} 的URL"
            
            elif action == "call_playwright_react":
                # 调用Playwright ReAct Agent执行网页自动化
                url = params.get("url", "")
                # 如果用户是一般信息查询，不需要打开浏览器，直接跳过
                intent_open_keywords = ["打开", "浏览器", "登录", "点击", "网页", "在\n浏览器", "在浏览器", "搜索并打开", "访问", "进入"]
                informational_keywords = ["是谁", "现状", "状态", "被封", "是否", "怎么", "简介", "情况", "了吗", "吗", "介绍", "详细"]
                if any(k in user_input for k in informational_keywords) and not any(k in user_input for k in intent_open_keywords):
                    return "ℹ️ 这是信息查询任务，无需打开网页；已基于搜索给出答案"
                
                print(f"    🔍 原始URL参数: {url}")
                print(f"    🔍 已收集信息: {list(collected_info.keys())}")
                
                # 🔍 智能URL提取（从params或collected_info）
                # 检测占位符：previous、步骤、获取、{{、}}等
                is_placeholder = (
                    not url or 
                    "previous" in url.lower() or
                    "步骤" in url or 
                    "获取" in url or
                    "{{" in url or
                    "}}" in url or
                    not url.startswith("http")
                )
                
                if is_placeholder:
                    # URL是占位符，从已收集信息中提取实际URL
                    print(f"    🔄 检测到占位符，从已收集信息中提取URL...")
                    for key, value in collected_info.items():
                        if "获取到URL:" in str(value):
                            url = value.split("获取到URL:")[1].strip()
                            print(f"    ✅ 从{key}中提取URL: {url}")
                            break
                
                if not url or not url.startswith("http"):
                    print(f"    ❌ 最终URL无效: {url}")
                    return "❌ 未找到有效的网站URL，无法执行"
                
                print(f"    🤖 调用网页打开功能: {url}")
                print(f"    📝 用户任务: {user_input}")
                
                # 直接调用主Agent的网页打开功能（明确传递user_input参数）
                result = self.base_agent._open_website_wrapper(
                    site_name=url,
                    website_map=None,
                    user_input=user_input
                )
                return result
            
            elif action == "get_weather":
                # 从已收集信息中获取位置
                location_info = collected_info.get("step_1", "")
                city = self.base_agent._extract_city_from_location(location_info)
                if not city:
                    city = self.base_agent._extract_city_from_location(self.base_agent.location)
                
                weather_source = self.base_agent.config.get("weather_source", "高德地图API")
                if weather_source == "高德地图API":
                    from src.tools.amap_tool import AmapTool
                    amap_key = self.base_agent.config.get("amap_key", "")
                    weather = AmapTool.get_weather(city, amap_key)
                else:
                    heweather_key = self.base_agent.config.get("heweather_key", "")
                    weather = self.base_agent.tools["天气"](city, heweather_key)
                
                return f"天气：{weather}"
            
            elif action == "search_web":
                query = params.get("query", user_input)
                # 临时开启搜索
                original = self.base_agent.config.get("enable_web_search", False)
                print(f"🔍 [框架search_web] 保存原始值: enable_web_search = {original}")
                
                try:
                    self.base_agent.config["enable_web_search"] = True
                    print(f"🔍 [框架search_web] 临时开启搜索")
                    
                    # 仅执行搜索与注入，不重复触发内层回忆（避免与主Agent重复）
                    result = self.base_agent._generate_response_with_context(query, {}, skip_memory_recall=True)
                    
                    return result
                finally:
                    # 确保无论如何都会恢复原始值
                    self.base_agent.config["enable_web_search"] = original
                    print(f"🔍 [框架search_web] 恢复原始值: enable_web_search = {original}")
            
            elif action == "analyze_file":
                if self.base_agent.recent_file_analysis:
                    info = self.base_agent.recent_file_analysis
                    return f"文件分析：{info['summary']}\n{info['analysis']}"
                return "无文件上下文"
            
            elif action == "open_application":
                # 兼容多种参数名：name, application_name, app, app_name
                app_name = (
                    params.get("name") or
                    params.get("application_name") or
                    params.get("app") or
                    params.get("app_name") or
                    ""
                )
                return self.base_agent._open_application(app_name)
            
            elif action == "open_website":
                site_name = params.get("name", "")
                return self.base_agent._open_website_wrapper(site_name, user_input)
            
            elif action == "pass_to_main_agent":
                # 将结果交给主Agent生成最终回答：复用主Agent系统提示与流程
                print(f"    🔄 将框架执行结果传递给主Agent总结...")

                # 为避免重复联网搜索：临时关闭联网搜索，但保留之前已写入的 search_context
                original_search_flag = self.base_agent.config.get("enable_web_search", False)
                self.base_agent.config["enable_web_search"] = False
                try:
                    # 直接调用主Agent的对话处理流程，并显式跳过框架以避免死循环；
                    # 同时抑制工具路由，避免重复打开浏览器/应用
                    # 🔥 将所有框架执行步骤的完整结果传递给主Agent
                    if collected_info:
                        # 将所有步骤结果汇总，每步最多2000字符
                        context_parts = []
                        for idx, key in enumerate(sorted(collected_info.keys())):
                            step_result = collected_info[key]
                            # 限制每步长度，避免上下文过长
                            max_length = 2000 if len(collected_info) > 1 else 5000  # 单步任务可以更长
                            if len(step_result) > max_length:
                                step_result = step_result[:max_length] + "..."
                            context_parts.append(f"【步骤 {idx+1}】\n{step_result}")
                        
                        full_context = "\n\n".join(context_parts)
                        self.base_agent.framework_context = f"框架执行结果：\n{full_context}"
                        print(f"📋 [传递上下文] 已将 {len(collected_info)} 步结果传递给主Agent（总长度: {len(full_context)} 字符）")
                    
                    final_answer = self.base_agent.process_command(user_input, skip_framework=True, suppress_tool_routing=True)
                    return final_answer
                finally:
                    self.base_agent.config["enable_web_search"] = original_search_flag
            
            else:
                return f"未知操作：{action}"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"执行失败：{str(e)}"
    
    def _should_adjust_framework(self, user_input: str, collected_info: Dict, remaining_steps: List) -> bool:
        """
        判断是否需要调整框架
        
        Args:
            user_input: 用户输入
            collected_info: 已收集的信息
            remaining_steps: 剩余步骤
            
        Returns:
            是否需要调整
        """
        # 简单策略：如果已完成步骤超过5步，检查一次
        if len(self.completed_steps) == 5:
            return True
        return False
    
    def _adjust_framework(self, user_input: str, collected_info: Dict, remaining_steps: List) -> List[Dict]:
        """
        调整执行框架
        
        Args:
            user_input: 用户输入
            collected_info: 已收集的信息
            remaining_steps: 原剩余步骤
            
        Returns:
            新的步骤列表
        """
        prompt = f"""你是任务规划专家，需要根据当前进展调整执行框架。

原始用户请求：{user_input}

已完成的步骤：
{self._format_completed_steps()}

已收集的信息：
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

原计划的剩余步骤：
{json.dumps(remaining_steps, ensure_ascii=False, indent=2)}

请根据当前进展，重新规划后续步骤。返回JSON数组格式，例如：
[
    {{"description": "整合信息并回答", "action": "answer_question", "params": {{}}}}
]

如果不需要调整，返回原剩余步骤。
"""
        
        # 直接调用OpenAI API
        try:
            import openai
            
            if "deepseek" in self.intent_model:
                api_key = self.base_agent.config.get("deepseek_key", "")
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            else:
                api_key = self.base_agent.config.get("openai_key", "")
                client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": "你是任务规划专家，根据进展调整执行计划。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                timeout=15
            )
            
            response = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return remaining_steps
        
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except:
            return remaining_steps
    
    def _format_completed_steps(self) -> str:
        """格式化已完成的步骤"""
        if not self.completed_steps:
            return "（暂无）"
        
        lines = []
        for step in self.completed_steps:
            lines.append(f"[第 {step['step']} 步] {step['description']}")
        return "\n".join(lines)
    
    def _generate_final_answer(self, user_input: str, collected_info: Dict) -> str:
        """
        生成最终答案 - 框架Agent只负责协调，不负责回答
        
        Args:
            user_input: 用户输入
            collected_info: 收集的所有信息
            
        Returns:
            最终回答
        """
        # 检查最后一步是否已经是回答或传递给主Agent
        if self.completed_steps:
            last_step = self.completed_steps[-1]
            last_action = last_step.get("action", "")  # 🔥 改为检查action而非description
            
            # 🔥 如果最后一步的action是pass_to_main_agent，说明已经调用过主Agent
            if last_action == "pass_to_main_agent":
                # 最后一步已经完成回答，直接返回
                print("✅ 最后一步已是pass_to_main_agent，直接返回结果，不再重复调用")
                return last_step["result"]
            
            # 兼容旧的检查方式
            last_description = last_step.get("description", "").lower()
            if any(keyword in last_description for keyword in ["answer", "回答", "主agent", "传递"]):
                print("✅ 最后一步包含回答关键词，直接返回结果")
                return last_step["result"]
        
        # 如果最后一步不是pass_to_main_agent，强制调用主Agent处理
        print("⚠️ 框架未以pass_to_main_agent结束，强制调用主Agent处理")
        
        # 将框架执行结果注入到主Agent的上下文中
        context_summary = "\n\n".join([
            f"【步骤 {step['step']}】{step['description']}\n{step['result'][:500]}" 
            for step in self.completed_steps
        ])
        
        self.base_agent.framework_context = f"框架执行结果：\n{context_summary}"
        
        # 调用主Agent，让它基于框架执行结果生成回答
        return self.base_agent.process_command(user_input, skip_framework=True, suppress_tool_routing=True)


# 测试代码
if __name__ == "__main__":
    print("框架ReAct Agent模块加载成功")

