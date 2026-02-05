#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网页操作Agent
专门处理网页打开和自动化操作
"""

import openai
import json
from typing import Dict, Any, Optional, List
from playwright_tool import playwright_open_website_headed
from utils import open_website


class WebpageAgent:
    """网页操作专属Agent"""
    
    def __init__(self, config: dict):
        """
        初始化网页操作Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    def analyze_webpage_operation(self, user_input: str) -> Dict[str, Any]:
        """
        分析用户的网页操作需求
        
        Args:
            user_input: 用户输入
            
        Returns:
            {
                "operation_type": "search" / "click" / "fill" / "scroll" / "navigate" / "none",
                "target": "操作目标内容",
                "selector": "CSS选择器（如果需要）",
                "description": "操作描述"
            }
        """
        try:
            model = "deepseek-chat" if "deepseek" in self.config.get("selected_model", "deepseek-chat") else "gpt-3.5-turbo"
            api_key = self.config.get("deepseek_key", "") if "deepseek" in model else self.config.get("openai_key", "")
            
            if not api_key:
                print("⚠️ 没有API密钥，无法分析网页操作")
                return {"operation_type": "none", "target": "", "selector": "", "description": ""}
            
            # AI分析网页操作的提示词
            prompt = f"""分析用户想要在网页上执行什么操作。

用户说：{user_input}

请判断用户想要执行的操作类型，并提取精确的操作目标。

**操作类型优先级**（从高到低识别）：

1. **click** - 点击元素（关键词：点、点击、点进、点开、打开XX界面、选择）
   **重要：target应该是网页上实际显示的文本，去掉"界面"、"页面"等后缀**
   **特别注意："打开XX界面" = 点击XX按钮，不是navigate！**
   
   示例：
   - "打开B站并点开第一个视频" → {{"operation_type": "click", "target": "第一个视频"}}
   - "点击登录按钮" → {{"operation_type": "click", "target": "登录"}}
   - "点开登录界面" → {{"operation_type": "click", "target": "登录"}}
   - "打开登录界面" → {{"operation_type": "click", "target": "登录"}}  ← 重要！
   - "进入设置页面" → {{"operation_type": "click", "target": "设置"}}
   - "打开设置" → {{"operation_type": "click", "target": "设置"}}  ← 重要！
   - "点击注册" → {{"operation_type": "click", "target": "注册"}}
   
2. **fill** - 填写内容（关键词：输入、填写、填入）
   **target是要填写的内容，不是输入框名称**
   
   示例：
   - "在搜索框输入python" → {{"operation_type": "fill", "target": "python"}}
   - "填写用户名为admin" → {{"operation_type": "fill", "target": "admin"}}
   
3. **scroll** - 滚动页面（关键词：滚动、下拉、翻页）
   示例：
   - "向下滚动" → {{"operation_type": "scroll", "target": "down"}}
   - "滚动到底部" → {{"operation_type": "scroll", "target": "bottom"}}
   
4. **search** - 在网站内搜索（关键词：搜索+在XX上/XX内）
   **target是搜索关键词**
   
   示例：
   - "打开B站搜索python" → {{"operation_type": "search", "target": "python"}}
   - "在知乎上找教程" → {{"operation_type": "search", "target": "教程"}}
   
5. **navigate** - 只是打开网页（无其他操作，只有打开网站时才用）
   **注意："打开XX"如果XX是界面/页面名称（如登录界面），是click不是navigate！**
   
   示例：
   - "打开B站" → {{"operation_type": "navigate", "target": ""}}  ← B站是网站
   - "搜索并打开bilibili" → {{"operation_type": "navigate", "target": ""}}  ← bilibili是网站
   - "访问知乎" → {{"operation_type": "navigate", "target": ""}}  ← 知乎是网站
   
   反例（这些不是navigate）：
   - "打开登录界面" → {{"operation_type": "click", "target": "登录"}}  ← 登录是按钮
   - "打开设置" → {{"operation_type": "click", "target": "设置"}}  ← 设置是按钮

**核心规则**：
1. 判断"打开XX"的类型：
   - 如果XX是网站名称（B站、知乎等）→ navigate
   - 如果XX是界面/页面/按钮（登录界面、设置等）→ click
   
2. 对于click操作：
   - 去掉"按钮"、"界面"、"页面"、"选项"等后缀
   - 只保留网页上实际可见的文本
   - "登录按钮" → "登录"
   - "登录界面" → "登录"
   - "设置页面" → "设置"
   - "打开登录" → "登录"
   - "打开设置" → "设置"
   
3. 对于fill操作：
   - target是要填写的内容，不是输入框的名称
   
4. 对于search操作：
   - target是搜索关键词

**重要：只返回一个最核心的操作，不要返回多个操作的列表！**

**如何判断"最核心"操作**：
- 如果包含"打开XX网站 + 其他操作" → 返回"其他操作"（打开网站会自动完成）
- 如果只有"打开XX网站" → 返回navigate

返回JSON格式：
{{"operation_type": "类型", "target": "精确的目标文本", "description": "操作描述"}}

示例（单个操作）：
- "打开B站并搜索java" → {{"operation_type": "search", "target": "java"}}  ← 返回搜索（打开会自动完成）
- "打开B站并打开登录界面" → {{"operation_type": "click", "target": "登录"}}  ← 返回点击（打开会自动完成）
- "打开B站并点击第一个视频" → {{"operation_type": "click", "target": "第一个视频"}}  ← 返回点击
- "打开B站" → {{"operation_type": "navigate", "target": ""}}  ← 只有打开，返回navigate

**核心原则**：
- "打开网站"是基础操作，如果后面还有其他操作，就返回后面的操作
- 只有当用户只要求打开网站时，才返回navigate

只返回一个JSON对象，不要返回数组！
"""
            
            print(f"🔍 [WebpageAgent] 分析网页操作需求...")
            
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1") if "deepseek" in model else openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个网页操作分析助手，专门识别用户的网页自动化操作需求。请严格返回JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1,
                timeout=10
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"🤖 [WebpageAgent] AI分析结果: {result_text}")
            
            # 解析JSON
            # 清理可能的markdown代码块
            if "```" in result_text:
                result_text = result_text.split("```")[1] if result_text.count("```") >= 2 else result_text
                result_text = result_text.replace("json", "").strip()
            
            operation = json.loads(result_text)
            
            # 🤖 如果AI返回了多步操作（列表），说明需要ReAct推理
            if isinstance(operation, list):
                print(f"⚠️ [WebpageAgent] AI识别出 {len(operation)} 步操作，但当前架构只支持单步")
                print(f"💡 [WebpageAgent] 建议使用ReAct推理模式处理复杂多步操作")
                
                # 临时方案：只取最后一个操作（通常是用户最关心的）
                if operation:
                    last_operation = operation[-1]
                    print(f"📌 [WebpageAgent] 临时使用最后一步操作: {last_operation.get('description', '')}")
                    return last_operation
                else:
                    return {"operation_type": "none", "target": "", "description": "无有效操作"}
            
            return operation
            
        except Exception as e:
            print(f"❌ [WebpageAgent] 分析失败: {str(e)}")
            return {"operation_type": "navigate", "target": "", "description": "打开网页"}
    
    def execute_webpage_operation(
        self,
        url: str,
        operation: Dict[str, Any],
        browser_type: str = "edge",
        mode: str = "launch",
        slow_mo: int = 0,
        cdp_url: str = "http://localhost:9222",
        user_data_dir: str = ""
    ) -> Dict[str, Any]:
        """
        执行网页操作
        
        Args:
            url: 目标URL
            operation: 操作信息（从analyze_webpage_operation获取）
            browser_type: 浏览器类型
            mode: Playwright模式
            slow_mo: 慢速延迟
            cdp_url: CDP地址
            user_data_dir: 用户数据目录
            
        Returns:
            {"success": bool, "message": str, "title": str}
        """
        operation_type = operation.get("operation_type", "navigate")
        target = operation.get("target", "")
        
        print(f"🎯 [WebpageAgent] 执行操作: {operation_type}, 目标: {target}")
        
        # 根据操作类型决定使用哪种方式
        if operation_type == "navigate":
            # 只是打开，使用系统浏览器（快速）
            print(f"🌐 [WebpageAgent] 简单打开，使用系统浏览器")
            result = open_website(url, browser_type)
            return {
                "success": True,
                "message": result,
                "title": url
            }
        
        elif operation_type == "search":
            # 搜索操作，使用Playwright
            print(f"🔍 [WebpageAgent] 搜索操作，使用Playwright: {target}")
            result = playwright_open_website_headed(
                url,
                browser_type=browser_type,
                search_query=target,
                mode=mode,
                slow_mo=slow_mo,
                cdp_url=cdp_url,
                user_data_dir=user_data_dir
            )
            return result
        
        elif operation_type == "click":
            # 点击操作 - 使用Playwright打开网页并执行点击
            print(f"🖱️ [WebpageAgent] 点击操作: {target}")
            actions = [{"type": "click_text", "text": target}]
            result = playwright_open_website_headed(
                url,
                browser_type=browser_type,
                search_query="",  # 不搜索
                mode=mode,
                slow_mo=slow_mo,
                cdp_url=cdp_url,
                user_data_dir=user_data_dir,
                actions=actions
            )
            
            if result.get("success"):
                # 检查点击操作是否成功
                actions_performed = result.get("actions_performed", [])
                if actions_performed and actions_performed[0].get("success"):
                    return {
                        "success": True,
                        "message": f"已在网页上点击「{target}」",
                        "title": result.get("title", url),
                        "url": result.get("url", url),
                        "action_success": True
                    }
                else:
                    # 网页已打开，但点击失败 - 仍返回success=True避免降级
                    return {
                        "success": True,
                        "message": f"已打开网页，但未找到可点击的元素「{target}」",
                        "title": result.get("title", url),
                        "url": result.get("url", url),
                        "action_success": False,
                        "action_error": f"未找到可点击的元素「{target}」"
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "打开网页失败"),
                    "title": url,
                    "url": url
                }
        
        elif operation_type == "fill":
            # 填写操作
            print(f"✏️ [WebpageAgent] 填写操作: {target}")
            actions = [{"type": "fill", "selector": "input, textarea", "text": target}]
            result = playwright_open_website_headed(
                url,
                browser_type=browser_type,
                search_query="",
                mode=mode,
                slow_mo=slow_mo,
                cdp_url=cdp_url,
                user_data_dir=user_data_dir,
                actions=actions
            )
            
            if result.get("success"):
                actions_performed = result.get("actions_performed", [])
                if actions_performed and actions_performed[0].get("success"):
                    return {
                        "success": True,
                        "message": f"已填写「{target}」",
                        "title": result.get("title", url),
                        "url": result.get("url", url),
                        "action_success": True
                    }
                else:
                    # 网页已打开，但填写失败 - 仍返回success=True避免降级
                    return {
                        "success": True,
                        "message": f"已打开网页，但未找到可填写的输入框",
                        "title": result.get("title", url),
                        "url": result.get("url", url),
                        "action_success": False,
                        "action_error": f"未找到可填写的输入框"
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "打开网页失败"),
                    "title": url,
                    "url": url
                }
        
        elif operation_type == "scroll":
            # 滚动操作
            print(f"📜 [WebpageAgent] 滚动操作: {target}")
            direction = "down"
            if "上" in target or "顶部" in target:
                direction = "top"
            elif "下" in target or "底部" in target:
                direction = "bottom"
            
            actions = [{"type": "scroll", "direction": direction}]
            result = playwright_open_website_headed(
                url,
                browser_type=browser_type,
                search_query="",
                mode=mode,
                slow_mo=slow_mo,
                cdp_url=cdp_url,
                user_data_dir=user_data_dir,
                actions=actions
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"已滚动页面",
                    "title": result.get("title", url),
                    "url": result.get("url", url)
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "滚动操作失败"),
                    "title": url,
                    "url": url
                }
        
        else:
            # 默认打开
            result = open_website(url, browser_type)
            return {
                "success": True,
                "message": result,
                "title": url
            }


if __name__ == "__main__":
    # 测试
    import json
    
    config = {
        "deepseek_key": "your_key",
        "selected_model": "deepseek-chat",
        "default_browser": "edge"
    }
    
    agent = WebpageAgent(config)
    
    test_cases = [
        "打开B站搜索python",
        "搜索并打开bilibili",
        "点进第一个视频",
        "打开B站"
    ]
    
    for case in test_cases:
        print(f"\n测试: {case}")
        result = agent.analyze_webpage_operation(case)
        print(f"结果: {json.dumps(result, ensure_ascii=False)}")

