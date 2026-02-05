# -*- coding: utf-8 -*-
"""
记忆系统模块 - 识底深湖
处理对话记忆、主题总结和上下文回忆
"""

import json
import os
import datetime
import re
import openai
import numpy as np
from config.config import load_config
from src.memory.memory_summary_agent import MemorySummaryAgent
from src.tools.simple_vector_encoder import get_vector_encoder

class MemoryLake:
    """记忆系统 - 识底深湖"""
    
    def __init__(self, memory_file="memory_lake.json", chat_logs_dir="chat_logs"):
        self.memory_file = memory_file
        self.chat_logs_dir = chat_logs_dir
        
        # 新的文件结构
        self.memory_index_file = os.path.join(chat_logs_dir, "memory_index.json")
        self.memories_dir = os.path.join(chat_logs_dir, "memories")
        self.vectors_dir = os.path.join(chat_logs_dir, "vectors")
        self.daily_logs_dir = os.path.join(chat_logs_dir, "daily_logs")
        
        # 初始化目录结构
        self._init_directory_structure()
        
        # 迁移旧数据（如果存在）
        self._migrate_old_data()
        
        self.memory_index = self.load_memory()
        self.current_conversation = []
        self.last_save_date = None
        self.config = load_config()
        
        # 迁移相关标记
        self.pending_migration = None
        
        # 初始化记忆总结AI代理
        self.summary_agent = MemorySummaryAgent(self.config)
        
        # 初始化向量编码器（识底深湖向量数据库）
        try:
            print("🗄️ 识底深湖向量数据库已启用")
        except UnicodeEncodeError:
            print("[INFO] 识底深湖向量数据库已启用")
        self.vector_encoder = get_vector_encoder()
        
        # 为向量编码器更新词汇表（使用现有记忆的主题）
        self._update_vector_vocab()
        
        # 🚀 修复：初始化mark_saved_callback属性
        self.mark_saved_callback = None
        
        # 确保目录存在
        if not os.path.exists(self.chat_logs_dir):
            os.makedirs(self.chat_logs_dir)

        # 确保第一条记忆是重点记忆
        self.ensure_first_memory_important()
        
        # 为现有记忆生成向量（如果缺失）
        self._generate_missing_vectors()
        
        # 检查是否有未迁移的记忆文件
        self._check_unmigrated_files()

    def _init_directory_structure(self):
        """初始化新的目录结构"""
        directories = [
            self.chat_logs_dir,
            self.memories_dir,
            self.vectors_dir,
            self.daily_logs_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 创建目录: {directory}")

    def _migrate_old_data(self):
        """检测旧数据，不自动迁移"""
        # 这个方法现在只做检测，不执行迁移
        pass

    def _migrate_vector_files(self):
        """迁移向量相关文件"""
        try:
            # 迁移词汇表文件
            old_vocab_file = "topic_vocab.json"
            new_vocab_file = os.path.join(self.vectors_dir, "topic_vocab.json")
            
            if os.path.exists(old_vocab_file) and not os.path.exists(new_vocab_file):
                os.rename(old_vocab_file, new_vocab_file)
                print(f"📦 迁移词汇表: {old_vocab_file} -> {new_vocab_file}")
            
            # 迁移嵌入缓存文件
            old_embedding_file = "embedding_cache.json"
            new_embedding_file = os.path.join(self.vectors_dir, "embedding_cache.json")
            
            if os.path.exists(old_embedding_file) and not os.path.exists(new_embedding_file):
                os.rename(old_embedding_file, new_embedding_file)
                print(f"📦 迁移嵌入缓存: {old_embedding_file} -> {new_embedding_file}")
                
        except Exception as e:
            print(f"⚠️ 向量文件迁移失败: {e}")

    def _convert_old_format(self, old_topic):
        """转换老格式记忆到新格式"""
        # 创建新格式的记忆对象
        new_topic = {
            "topic": old_topic.get("topic", "未知主题"),
            "timestamp": old_topic.get("timestamp", "00:00:00"),
            "date": old_topic.get("date", "unknown"),
            "conversation_count": 1,  # 老格式默认为1轮对话
            "keywords": old_topic.get("keywords", []),
            "conversation_details": old_topic.get("context", old_topic.get("conversation_details", "")),
            "is_important": old_topic.get("is_important", False)
        }
        
        # 保留其他可能的字段
        for key, value in old_topic.items():
            if key not in new_topic:
                new_topic[key] = value
        
        return new_topic
    
    def _sanitize_filename(self, filename):
        """清理文件名，移除不合法字符"""
        import re
        # 移除或替换不合法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized

    def _check_unmigrated_files(self):
        """检查是否有未迁移的记忆文件"""
        try:
            if os.path.exists(self.memory_file):
                print(f"🔍 发现记忆文件: {self.memory_file}")
                
                # 读取文件内容检查
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                
                # 获取记忆数量
                if isinstance(old_data, dict) and "topics" in old_data:
                    old_memory_count = len(old_data["topics"])
                elif isinstance(old_data, list):
                    old_memory_count = len(old_data)
                else:
                    old_memory_count = 0
                
                # 获取当前记忆数量
                current_memory_count = len(self.memory_index.get("topics", []))
                
                print(f"📊 旧文件记忆数: {old_memory_count}, 当前记忆数: {current_memory_count}")
                
                if old_memory_count > 0:
                    print("💡 检测到未迁移的记忆数据，将通过对话询问用户...")
                    # 设置迁移标记，让AI主动询问
                    self.pending_migration = {
                        "old_memory_count": old_memory_count,
                        "current_memory_count": current_memory_count,
                        "old_data": old_data
                    }
                else:
                    print("ℹ️ 旧文件为空，删除...")
                    os.remove(self.memory_file)
                    
        except Exception as e:
            print(f"⚠️ 检查未迁移文件失败: {e}")
            self.pending_migration = None

    def _force_migrate_old_data(self):
        """强制迁移旧数据（无论时间戳如何）"""
        try:
            print("🔄 强制迁移模式启动...")
            
            # 读取旧数据
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            
            # 处理旧数据格式
            if isinstance(old_data, dict) and "topics" in old_data:
                new_topics = old_data["topics"]
            elif isinstance(old_data, list):
                new_topics = old_data
            else:
                print("⚠️ 无法识别的数据格式")
                return
            
            if not new_topics:
                print("⚠️ 旧文件中没有找到记忆数据")
                return
            
            # 读取现有索引
            existing_memory_index = {"memories": []}
            if os.path.exists(self.memory_index_file):
                try:
                    with open(self.memory_index_file, 'r', encoding='utf-8') as f:
                        existing_memory_index = json.load(f)
                except Exception as e:
                    print(f"⚠️ 读取现有索引失败: {e}")
            
            # 获取现有记忆ID集合
            existing_ids = set()
            for memory_info in existing_memory_index.get("memories", []):
                existing_ids.add(memory_info.get("id", ""))
            
            # 迁移新记忆
            new_count = 0
            duplicate_count = 0
            
            for i, topic in enumerate(new_topics):
                date = topic.get("date", "unknown")
                timestamp = topic.get("timestamp", "00-00-00")
                memory_id = f"{date}_{timestamp.replace(':', '-')}"
                
                # 检查重复
                if memory_id in existing_ids:
                    duplicate_count += 1
                    continue
                
                # 转换格式并保存
                converted_topic = self._convert_old_format(topic)
                
                timestamp_clean = timestamp.replace(":", "-")
                topic_name = self._sanitize_filename(topic.get("topic", f"imported_{i}"))
                memory_filename = f"{date}_{timestamp_clean}_{topic_name}.json"
                memory_filepath = os.path.join(self.memories_dir, memory_filename)
                
                with open(memory_filepath, 'w', encoding='utf-8') as f:
                    json.dump(converted_topic, f, ensure_ascii=False, indent=2)
                
                # 更新索引
                existing_memory_index["memories"].append({
                    "id": memory_id,
                    "filename": memory_filename,
                    "topic": topic.get("topic", ""),
                    "date": date,
                    "timestamp": timestamp,
                    "is_important": topic.get("is_important", False)
                })
                
                new_count += 1
            
            # 保存索引
            with open(self.memory_index_file, 'w', encoding='utf-8') as f:
                json.dump(existing_memory_index, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 强制迁移完成: 新增 {new_count} 条记忆，跳过 {duplicate_count} 条重复")
            
            # 备份原文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.memory_file}.migrated_{timestamp}"
            os.rename(self.memory_file, backup_file)
            print(f"📦 已备份并移除旧文件: {backup_file}")
            
        except Exception as e:
            print(f"⚠️ 强制迁移失败: {e}")

    def get_migration_status(self):
        """获取迁移状态"""
        return self.pending_migration
    
    def confirm_migration(self, user_response):
        """用户确认迁移"""
        if not self.pending_migration:
            return "没有待迁移的数据。"
        
        if user_response.strip().lower() in ['是', 'yes', 'y', '确认', '同意']:
            try:
                print("✅ 用户确认迁移，开始执行...")
                old_data = self.pending_migration["old_data"]
                
                # 执行迁移
                success = self._execute_migration(old_data)
                
                if success:
                    self.pending_migration = None
                    return "✅ 记忆迁移完成！您的所有历史对话记录已成功转移到新的文件结构中，现在可以享受更智能的回忆功能了。"
                else:
                    return "❌ 迁移过程中出现错误，请检查系统日志。原始数据已安全保留。"
                    
            except Exception as e:
                print(f"⚠️ 迁移执行失败: {e}")
                return f"❌ 迁移失败：{str(e)}"
                
        elif user_response.strip().lower() in ['否', 'no', 'n', '取消', '拒绝']:
            print("❌ 用户取消迁移")
            self.pending_migration = None
            return "好的，已取消迁移。您的旧记忆文件将保持原样，但无法使用新的智能回忆功能。如需迁移，请重新启动露尼西亚。"
        else:
            return "请回答'是'或'否'来确认是否进行记忆迁移。"
    
    def _execute_migration(self, old_data):
        """执行记忆迁移"""
        try:
            # 处理旧数据格式
            if isinstance(old_data, dict) and "topics" in old_data:
                new_topics = old_data["topics"]
            elif isinstance(old_data, list):
                new_topics = old_data
            else:
                print("⚠️ 无法识别的数据格式")
                return False
            
            if not new_topics:
                print("⚠️ 旧文件中没有找到记忆数据")
                return False
            
            # 读取现有索引
            existing_memory_index = {"memories": []}
            if os.path.exists(self.memory_index_file):
                try:
                    with open(self.memory_index_file, 'r', encoding='utf-8') as f:
                        existing_memory_index = json.load(f)
                except Exception as e:
                    print(f"⚠️ 读取现有索引失败: {e}")
            
            # 获取现有记忆ID集合
            existing_ids = set()
            for memory_info in existing_memory_index.get("memories", []):
                existing_ids.add(memory_info.get("id", ""))
            
            # 迁移新记忆
            new_count = 0
            duplicate_count = 0
            
            for i, topic in enumerate(new_topics):
                date = topic.get("date", "unknown")
                timestamp = topic.get("timestamp", "00-00-00")
                memory_id = f"{date}_{timestamp.replace(':', '-')}"
                
                # 检查重复
                if memory_id in existing_ids:
                    duplicate_count += 1
                    continue
                
                # 转换格式并保存
                converted_topic = self._convert_old_format(topic)
                
                timestamp_clean = timestamp.replace(":", "-")
                topic_name = self._sanitize_filename(topic.get("topic", f"imported_{i}"))
                memory_filename = f"{date}_{timestamp_clean}_{topic_name}.json"
                memory_filepath = os.path.join(self.memories_dir, memory_filename)
                
                with open(memory_filepath, 'w', encoding='utf-8') as f:
                    json.dump(converted_topic, f, ensure_ascii=False, indent=2)
                
                # 更新索引
                existing_memory_index["memories"].append({
                    "id": memory_id,
                    "filename": memory_filename,
                    "topic": topic.get("topic", ""),
                    "date": date,
                    "timestamp": timestamp,
                    "is_important": topic.get("is_important", False)
                })
                
                new_count += 1
            
            # 保存索引
            with open(self.memory_index_file, 'w', encoding='utf-8') as f:
                json.dump(existing_memory_index, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 迁移完成: 新增 {new_count} 条记忆，跳过 {duplicate_count} 条重复")
            
            # 备份并删除原文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.memory_file}.migrated_{timestamp}"
            os.rename(self.memory_file, backup_file)
            print(f"📦 已备份并移除旧文件: {backup_file}")
            
            # 重新加载记忆
            self.memory_index = self.load_memory()
            
            return True
            
        except Exception as e:
            print(f"⚠️ 迁移执行失败: {e}")
            return False

    def load_memory(self):
        """加载记忆索引和记忆数据"""
        try:
            # 优先使用新的索引文件
            if os.path.exists(self.memory_index_file):
                with open(self.memory_index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # 加载所有记忆数据
                topics = []
                for memory_info in index_data.get("memories", []):
                    memory_filepath = os.path.join(self.memories_dir, memory_info["filename"])
                    if os.path.exists(memory_filepath):
                        try:
                            with open(memory_filepath, 'r', encoding='utf-8') as f:
                                memory_data = json.load(f)
                                topics.append(memory_data)
                        except Exception as e:
                            print(f"⚠️ 加载记忆文件失败: {memory_info['filename']}, {e}")
                
                return {"topics": topics, "conversations": {}, "contexts": {}}
            
            # 兼容旧格式
            elif os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return {"topics": data, "conversations": {}, "contexts": {}}
                    elif isinstance(data, dict):
                        return data
                    else:
                        return {"topics": [], "conversations": {}, "contexts": {}}
            
            return {"topics": [], "conversations": {}, "contexts": {}}
            
        except Exception as e:
            print(f"⚠️ 加载记忆失败: {e}")
            return {"topics": [], "conversations": {}, "contexts": {}}

    def save_memory(self):
        """保存记忆（智能选择保存方式）"""
        # 检查是否应该使用新的文件结构
        if os.path.exists(self.memory_index_file) or os.path.exists(self.memories_dir):
            # 使用新的文件结构
            self._save_to_new_structure()
        else:
            # 使用旧的文件结构（兼容模式）
            self._save_to_old_structure()

    def _save_to_new_structure(self):
        """保存到新的文件结构"""
        try:
            # 确保目录存在
            if not os.path.exists(self.memories_dir):
                os.makedirs(self.memories_dir)
            
            # 保存索引文件
            memory_index = {"memories": []}
            
            for topic in self.memory_index.get("topics", []):
                # 生成文件名
                date = topic.get("date", "unknown")
                timestamp = topic.get("timestamp", "00-00-00").replace(":", "-")
                topic_name = self._sanitize_filename(topic.get("topic", "unknown"))
                
                memory_filename = f"{date}_{timestamp}_{topic_name}.json"
                memory_filepath = os.path.join(self.memories_dir, memory_filename)
                
                # 保存单独的记忆文件
                with open(memory_filepath, 'w', encoding='utf-8') as f:
                    json.dump(topic, f, ensure_ascii=False, indent=2)
                
                # 在索引中记录
                memory_index["memories"].append({
                    "id": f"{date}_{timestamp}",
                    "filename": memory_filename,
                    "topic": topic.get("topic", ""),
                    "date": date,
                    "timestamp": topic.get("timestamp", ""),
                    "is_important": topic.get("is_important", False)
                })
            
            # 保存索引文件
            with open(self.memory_index_file, 'w', encoding='utf-8') as f:
                json.dump(memory_index, f, ensure_ascii=False, indent=2)
            
            print(f"💾 新结构保存: {len(self.memory_index.get('topics', []))} 条记忆")
            
        except Exception as e:
            print(f"⚠️ 新结构保存失败: {e}")
            # 降级到旧结构
            self._save_to_old_structure()

    def _save_to_old_structure(self):
        """保存到旧的文件结构（兼容模式）"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_index, f, ensure_ascii=False, indent=2)
                print(f"💾 兼容模式保存: {len(self.memory_index.get('topics', []))} 条记忆")
        except Exception as e:
            print(f"❌ 兼容模式保存失败: {e}")

    def _update_vector_vocab(self):
        """更新向量编码器的词汇表"""
        try:
            # 收集所有主题文本
            topics = []
            for entry in self.memory_index.get("topics", []):
                if "topic" in entry:
                    topics.append(entry["topic"])
            
            if topics:
                print(f"📚 更新向量词汇表，使用 {len(topics)} 个主题")
                self.vector_encoder.update_vocab(topics)
            else:
                print("📚 无现有主题，使用空词汇表")
        except Exception as e:
            print(f"⚠️ 更新向量词汇表失败: {e}")
    
    def _generate_missing_vectors(self):
        """为缺失向量的记忆生成双重向量"""
        try:
            updated_count = 0
            for entry in self.memory_index.get("topics", []):
                needs_update = False
                
                # 生成主题向量（如果缺失）
                if "topic_vector" not in entry and "topic" in entry:
                    topic_vector = self.vector_encoder.encode_text(entry["topic"])
                    if topic_vector:
                        entry["topic_vector"] = topic_vector
                        needs_update = True
                
                # 生成内容向量（如果缺失）
                if "details_vector" not in entry and "conversation_details" in entry:
                    details_vector = self.vector_encoder.encode_text(entry["conversation_details"])
                    if details_vector:
                        entry["details_vector"] = details_vector
                        needs_update = True
                
                if needs_update:
                    updated_count += 1
            
            if updated_count > 0:
                print(f"🔄 为 {updated_count} 条记忆生成了双重向量")
                self.save_memory()
            else:
                print("✅ 所有记忆都已有双重向量")
        except Exception as e:
            print(f"⚠️ 生成缺失向量失败: {e}")
    
    def get_vector_stats(self) -> dict:
        """获取向量数据库统计信息"""
        encoder_stats = self.vector_encoder.get_stats()
        
        # 统计向量覆盖情况
        total_count = len(self.memory_index.get("topics", []))
        topic_vector_count = 0
        details_vector_count = 0
        dual_vector_count = 0
        
        for entry in self.memory_index.get("topics", []):
            has_topic = "topic_vector" in entry and entry["topic_vector"]
            has_details = "details_vector" in entry and entry["details_vector"]
            
            if has_topic:
                topic_vector_count += 1
            if has_details:
                details_vector_count += 1
            if has_topic and has_details:
                dual_vector_count += 1
        
        return {
            "encoder_stats": encoder_stats,
            "total_memories": total_count,
            "topic_vectorized": topic_vector_count,
            "details_vectorized": details_vector_count,
            "dual_vectorized": dual_vector_count,
            "dual_vectorization_rate": f"{dual_vector_count/total_count*100:.1f}%" if total_count > 0 else "0%"
        }
    
    def _get_timestamp_score(self, memory_entry):
        """计算时间戳分数，用于排序"""
        try:
            date_str = memory_entry.get("date", "")
            timestamp_str = memory_entry.get("timestamp", "")
            
            if date_str and timestamp_str:
                # 组合日期和时间
                datetime_str = f"{date_str} {timestamp_str}"
                memory_datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                
                # 转换为时间戳分数（越新分数越高）
                timestamp = memory_datetime.timestamp()
                return timestamp
            else:
                return 0
        except Exception as e:
            print(f"⚠️ 计算时间戳分数失败: {e}")
            return 0

    def add_conversation(self, user_input, ai_response, developer_mode=False, mark_saved_callback=None):
        """添加对话到当前会话"""
        # 开发者模式下不保存到记忆系统
        if developer_mode:
            print("🔧 开发者模式已开启，跳过对话记录到记忆系统")
            return
        
        # 🚀 修复：防重复添加机制（检查最近的对话）
        # 检查最近的对话是否是相同的用户输入（时间窗口内）
        if self.current_conversation:
            last_conv = self.current_conversation[-1]
            # 如果上一条对话的用户输入相同，且时间间隔小于5秒，认为是重复
            if last_conv.get('user_input') == user_input:
                try:
                    last_time = datetime.datetime.strptime(last_conv['timestamp'], "%H:%M:%S")
                    current_time = datetime.datetime.now()
                    # 构造今天的完整时间
                    last_datetime = datetime.datetime.combine(
                        datetime.datetime.now().date(),
                        last_time.time()
                    )
                    time_diff = (current_time - last_datetime).total_seconds()
                    
                    if time_diff < 5:  # 5秒内的重复认为是同一次对话
                        print(f"⚠️ 检测到5秒内重复用户输入，跳过添加: {user_input[:30]}...")
                        return
                except Exception as e:
                    # 如果时间解析失败，使用原来的简单检查
                    print(f"⚠️ 时间解析失败: {e}，使用简单重复检查")
                    if last_conv.get('ai_response') == ai_response:
                        print(f"⚠️ 检测到完全重复对话，跳过添加: {user_input[:30]}...")
                        return
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.current_conversation.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "ai_response": ai_response,
            "full_text": f"指挥官: {user_input}\n露尼西亚: {ai_response}"
        })
        
        print(f"✅ 添加对话到记忆系统: {user_input[:30]}... (当前共{len(self.current_conversation)}条)")
        
        # 🚀 修复：保存回调函数，在对话真正保存到识底深湖后调用
        if mark_saved_callback:
            self.mark_saved_callback = mark_saved_callback

    def should_summarize(self):
        """判断是否应该总结"""
        # 每3条对话总结一次，或者当前对话超过5条
        return len(self.current_conversation) >= 3

    def summarize_and_save_topic(self, ai_client=None, force_save=False):
        """总结并保存主题"""
        if not self.current_conversation:
            return None
        
        # 如果不是强制保存，检查是否满足保存条件
        if not force_save and not self.should_summarize():
            return None
            
        try:
            # 构建对话文本
            conversation_text = "\n".join([
                conv["full_text"] for conv in self.current_conversation
            ])
            
            # 使用AI总结主题
            topic = self._ai_summarize_topic(conversation_text)
            
            # 保存到记忆索引
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 生成双重向量（识底深湖向量数据库升级版）
            conversation_details = self._extract_conversation_details()
            topic_vector = self.vector_encoder.encode_text(topic)
            details_vector = self.vector_encoder.encode_text(conversation_details)
            
            entry = {
                "topic": topic,
                "timestamp": timestamp,
                "date": date_str,
                "conversation_count": len(self.current_conversation),
                "keywords": self._extract_keywords(conversation_text),  # 保留关键词作为备用
                "conversation_details": conversation_details,
                "topic_vector": topic_vector if topic_vector is not None else None,
                "details_vector": details_vector if details_vector is not None else None,
                "is_important": False  # 重点记忆标签
            }
            
            self.memory_index["topics"].append(entry)
            self.save_memory()
            
            # 更新向量编码器词汇表（包含新主题）
            self.vector_encoder.update_vocab([topic] + [e.get("topic", "") for e in self.memory_index["topics"]])
            
            # 🚀 修复：在成功保存到识底深湖后，标记所有已保存的对话为已保存
            # 获取AI代理的mark_saved_callback函数
            if hasattr(self, 'mark_saved_callback') and self.mark_saved_callback:
                for conv in self.current_conversation:
                    self.mark_saved_callback(conv['user_input'], conv['ai_response'])
            
            # 清空当前会话
            self.current_conversation = []
            
            return topic
            
        except Exception as e:
            print(f"总结主题失败: {str(e)}")
            return None

    def force_save_current_conversation(self, introduction_content=None):
        """强制保存当前对话（用于首次介绍后立即保存）"""
        if not self.current_conversation:
            return None
            
        try:
            # 构建对话文本，包含自我介绍
            conversation_parts = []
            
            # 如果有自我介绍内容，添加到开头
            if introduction_content:
                conversation_parts.append(f"露尼西亚: {introduction_content}")
                conversation_parts.append("")  # 空行分隔
            
            # 添加实际对话内容
            for conv in self.current_conversation:
                conversation_parts.append(conv["full_text"])
            
            conversation_text = "\n".join(conversation_parts)
            
            # 为首次对话生成特殊主题
            topic = self._generate_first_conversation_topic(conversation_text)
            
            # 保存到记忆索引
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 生成对话详情
            conversation_details = self._extract_conversation_details()
            
            # 🎯 只有在真正的首次对话时才添加自我介绍
            # 检查是否是首次对话：当前对话数=1 且有自我介绍内容
            if introduction_content and len(self.current_conversation) == 1:
                # 简化自我介绍内容
                intro_summary = "露尼西亚: 我是露尼西亚，威廉的姐姐，您的AI助手。具备智能对话、天气查询、音乐推荐、文件管理、编程代码生成、多语言交流及记忆系统\"识底深湖\"等功能。\n\n"
                conversation_details = intro_summary + conversation_details
                print(f"🎯 真正的首次对话，已添加自我介绍，总长度: {len(conversation_details)} 字符")
            elif introduction_content:
                print(f"⚠️ 检测到自我介绍内容但不是首次对话（对话数: {len(self.current_conversation)}），跳过添加")
            
            topic_vector = self.vector_encoder.encode_text(topic)
            details_vector = self.vector_encoder.encode_text(conversation_details)
            
            entry = {
                "topic": topic,
                "timestamp": timestamp,
                "date": date_str,
                "conversation_count": len(self.current_conversation),
                "keywords": self._extract_keywords(conversation_text),
                "conversation_details": conversation_details,
                "topic_vector": topic_vector if topic_vector is not None else None,
                "details_vector": details_vector if details_vector is not None else None,
                "is_important": True,  # 首次对话标记为重要
                "is_first_conversation": True  # 标记为首次对话
            }
            
            self.memory_index["topics"].append(entry)
            self.save_memory()
            
            # 更新向量编码器词汇表
            self.vector_encoder.update_vocab([topic] + [e.get("topic", "") for e in self.memory_index["topics"]])
            
            # 标记对话为已保存
            if hasattr(self, 'mark_saved_callback') and self.mark_saved_callback:
                for conv in self.current_conversation:
                    self.mark_saved_callback(conv['user_input'], conv['ai_response'])
            
            # 清空当前会话
            self.current_conversation = []
            
            return topic
            
        except Exception as e:
            print(f"⚠️ 强制保存首次对话失败: {str(e)}")
            return None

    def _generate_first_conversation_topic(self, conversation_text):
        """为首次对话生成特殊主题"""
        try:
            # 尝试使用AI总结
            topic = self._ai_summarize_topic(conversation_text)
            if topic:
                return f"首次相遇 - {topic}"
        except Exception as e:
            print(f"⚠️ AI总结首次对话主题失败: {e}")
        
        # 如果AI总结失败，使用规则生成
        lines = conversation_text.split('\n')
        user_lines = [line for line in lines if line.startswith('指挥官:')]
        if user_lines:
            first_user_input = user_lines[0].replace('指挥官:', '').strip()
            if len(first_user_input) > 10:
                first_user_input = first_user_input[:10] + "..."
            return f"首次相遇 - {first_user_input}"
        
        return "首次相遇对话"

    def _format_first_conversation_fallback(self, conversation_text):
        """首次对话的备用格式化方案"""
        try:
            lines = conversation_text.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 保留关键信息，简化格式
                if line.startswith('露尼西亚:'):
                    content = line[5:].strip()  # 移除"露尼西亚:"
                    if '我是露尼西亚' in content:
                        # 简化自我介绍
                        formatted_lines.append("露尼西亚: 我是露尼西亚，威廉的姐姐。AI助手，具备智能对话、天气查询、音乐推荐、文件管理、编程帮助、多语言翻译和记忆系统等功能。")
                    elif len(content) > 100:
                        # 长回复截取前100字符
                        formatted_lines.append(f"露尼西亚: {content[:100]}...")
                    else:
                        formatted_lines.append(line)
                elif line.startswith('指挥官:'):
                    formatted_lines.append(line)
            
            result = '\n'.join(formatted_lines)
            print(f"✅ 首次对话备用格式化完成，长度: {len(result)} 字符")
            return result
            
        except Exception as e:
            print(f"⚠️ 首次对话备用格式化失败: {e}")
            return conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text

    def _ai_summarize_topic(self, conversation_text):
        """使用AI总结主题"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试AI主题总结 (第{attempt + 1}次)")
                # 使用专门的记忆总结AI代理
                topic = self.summary_agent.summarize_topic(conversation_text)
                if topic and len(topic.strip()) >= 2:
                    print(f"✅ AI主题总结成功: {topic}")
                    return topic
                else:
                    print(f"⚠️ AI主题总结返回空结果 (第{attempt + 1}次)")
                    if attempt < max_retries - 1:
                        print("🔄 等待2秒后重试...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        print("❌ AI主题总结最终失败")
                        return "AI总结失败"
            except Exception as e:
                print(f"⚠️ AI主题总结失败 (第{attempt + 1}次): {str(e)}")
                if attempt < max_retries - 1:
                    print("🔄 等待2秒后重试...")
                    import time
                    time.sleep(2)
                    continue
                else:
                    print("❌ AI主题总结最终失败")
                    return "AI总结失败，请检查API配置"

    def _ai_summarize_content(self, conversation_text):
        """使用AI总结内容"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试AI上下文总结 (第{attempt + 1}次)")
                # 使用专门的记忆总结AI代理
                summary = self.summary_agent.summarize_context(conversation_text)
                if summary and len(summary.strip()) > 10:
                    print(f"✅ AI上下文总结成功: {summary[:50]}...")
                    return summary
                else:
                    print(f"⚠️ AI上下文总结返回空结果 (第{attempt + 1}次)")
                    if attempt < max_retries - 1:
                        print("🔄 等待2秒后重试...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        print("❌ AI上下文总结最终失败")
                        return "AI总结失败"
            except Exception as e:
                print(f"⚠️ AI上下文总结失败 (第{attempt + 1}次): {str(e)}")
                if attempt < max_retries - 1:
                    print("🔄 等待2秒后重试...")
                    import time
                    time.sleep(2)
                    continue
                else:
                    print("❌ AI上下文总结最终失败")
                    return "AI总结失败，请检查API配置"

    def _simple_summarize_topic(self, text):
        """简单主题总结 - 分析整个对话流程"""
        topics = []
        
        # 分析各种主题类型
        if "Python" in text or "python" in text:
            topics.append("Python编程")
        if "C++" in text or "c++" in text:
            topics.append("C++编程")
        if "COBOL" in text or "cobol" in text:
            topics.append("COBOL编程")
        if "java" in text or "Java" in text:
            topics.append("Java编程")
        if "音乐" in text or "歌单" in text or "歌曲" in text:
            topics.append("音乐推荐")
        if "天气" in text:
            topics.append("天气查询")
        if "文件" in text and ("创建" in text or "保存" in text):
            topics.append("文件操作")
        if "文件夹" in text or "目录" in text:
            topics.append("文件夹创建")
        if "计算器" in text:
            topics.append("计算器程序")
        if "俄罗斯方块" in text or "tetris" in text:
            topics.append("俄罗斯方块游戏")
        if "贪吃蛇" in text or "snake" in text:
            topics.append("贪吃蛇游戏")
        if "井字棋" in text or "tic-tac-toe" in text:
            topics.append("井字棋游戏")
        if "爬虫" in text or "crawler" in text:
            topics.append("网络爬虫")
        if "数据分析" in text or "data" in text:
            topics.append("数据分析")
        if "Hello World" in text or "hello" in text:
            topics.append("Hello World程序")
        if "设置" in text:
            topics.append("系统设置")
        if "记忆" in text or "识底深湖" in text:
            topics.append("记忆系统")
        if "MCP" in text or "工具" in text:
            topics.append("MCP工具")
        if "搜索" in text:
            topics.append("网络搜索")
        if "时间" in text:
            topics.append("时间查询")
        # 自我介绍相关（优先识别）
        if "指挥官，您好！我是露尼西亚" in text or "威廉的姐姐" in text:
            return "露尼西亚自我介绍"
        
        if "问候" in text or "你好" in text:
            topics.append("问候")
        if "介绍" in text and any(country in text for country in ["德国", "法国", "英国", "美国", "日本", "韩国", "俄罗斯", "中国", "塔林", "贝尔格莱德"]):
            topics.append("国家介绍")
        if "游记" in text or "旅游" in text or "行程" in text:
            topics.append("游记写作")
        
        # 根据发现的主题数量生成综合主题
        if len(topics) >= 3:
            # 多主题对话，选择最重要的几个，避免过于宽泛
            if "音乐推荐" in topics and "天气查询" in topics:
                return f"{topics[0]}与{topics[1]}等多项讨论"
            else:
                # 对于其他多主题，尝试生成更具体的主题
                main_topics = topics[:3]  # 取前3个主题
                return f"{'、'.join(main_topics)}等多项讨论"
        elif len(topics) == 2:
            # 双主题对话
            return f"{topics[0]}与{topics[1]}讨论"
        elif len(topics) == 1:
            # 单主题对话
            return topics[0]
        else:
            # 没有明确主题，尝试提取关键词
            keywords = self._extract_keywords(text)
            if keywords:
                return f"关于{keywords[0]}的对话"
            else:
                return "日常对话"
                
    def _simple_summarize_content(self, text):
        """简单内容总结"""
        summary_parts = []
        
        # 提取具体信息
        if "你好" in text or "问候" in text:
            summary_parts.append("用户进行了问候")
        
        if "天气" in text:
            # 尝试提取城市信息和具体天气数据
            cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安"]
            city_found = None
            for city in cities:
                if city in text:
                    city_found = city
                    break
            
            # 尝试提取具体的天气信息
            weather_details = []
            if "雷阵雨" in text:
                weather_details.append("雷阵雨")
            if "晴天" in text or "晴" in text:
                weather_details.append("晴天")
            if "多云" in text:
                weather_details.append("多云")
            if "阴" in text:
                weather_details.append("阴天")
            if "雨" in text and "雷阵雨" not in text:
                weather_details.append("雨天")
            
            # 尝试提取温度信息
            import re
            temp_matches = re.findall(r'(\d+)°C', text)
            if temp_matches:
                if len(temp_matches) == 1:
                    weather_details.append(f"{temp_matches[0]}°C")
                else:
                    weather_details.append(f"{temp_matches[0]}-{temp_matches[-1]}°C")
            
            # 尝试提取风力信息
            wind_matches = re.findall(r'([东南西北]风\d+-\d+级)', text)
            if wind_matches:
                weather_details.append(wind_matches[0])
            
            # 构建天气总结
            if city_found and weather_details:
                summary_parts.append(f"查询了{city_found}天气：{', '.join(weather_details[:3])}")
            elif city_found:
                summary_parts.append(f"查询了{city_found}的天气信息")
            elif weather_details:
                summary_parts.append(f"查询了天气信息：{', '.join(weather_details[:3])}")
            else:
                summary_parts.append("查询了天气信息")
        
        if "时间" in text:
            summary_parts.append("查询了当前时间")
        
        if "搜索" in text:
            # 尝试提取搜索关键词
            import re
            search_match = re.search(r'搜索\s*([^，。\s]+)', text)
            if search_match:
                keyword = search_match.group(1)
                summary_parts.append(f"搜索了{keyword}相关信息")
            else:
                summary_parts.append("进行了网络搜索")
        
        # 检查是否是音乐推荐相关的对话（需要更精确的匹配）
        if ("推荐" in text and ("音乐" in text or "歌单" in text or "歌曲" in text)) or \
           ("音乐" in text and ("推荐" in text or "几首" in text)):
            # 尝试提取具体的歌曲信息
            import re
            # 匹配歌曲名字（用《》包围的）
            song_matches = re.findall(r'《([^》]+)》', text)
            if song_matches:
                songs = song_matches[:3]  # 最多取前3首
                if len(songs) == 1:
                    summary_parts.append(f"推荐了音乐《{songs[0]}》")
                elif len(songs) == 2:
                    summary_parts.append(f"推荐了音乐《{songs[0]}》和《{songs[1]}》")
                else:
                    summary_parts.append(f"推荐了音乐《{songs[0]}》等{len(song_matches)}首歌曲")
            else:
                # 如果没有找到《》格式，尝试提取其他格式的歌曲名
                artist_matches = re.findall(r'-\s*([^（\n]+)', text)
                if artist_matches:
                    artists = artist_matches[:2]  # 最多取前2个艺术家
                    summary_parts.append(f"推荐了{artists[0]}等艺术家的音乐")
                else:
                    summary_parts.append("推荐了音乐歌单")
        
        if "Python" in text or "python" in text:
            # 尝试提取具体的Python项目信息
            if "计算器" in text:
                summary_parts.append("讨论了Python计算器程序")
            elif "俄罗斯方块" in text or "tetris" in text:
                summary_parts.append("讨论了Python俄罗斯方块游戏")
            elif "贪吃蛇" in text or "snake" in text:
                summary_parts.append("讨论了Python贪吃蛇游戏")
            elif "井字棋" in text or "tic-tac-toe" in text:
                summary_parts.append("讨论了Python井字棋游戏")
            elif "爬虫" in text or "crawler" in text:
                summary_parts.append("讨论了Python网络爬虫")
            elif "数据分析" in text or "data" in text:
                summary_parts.append("讨论了Python数据分析")
            elif "Hello World" in text or "hello" in text:
                summary_parts.append("讨论了Python Hello World程序")
            else:
                summary_parts.append("讨论了Python编程相关内容")
        
        if "C++" in text or "c++" in text:
            # 尝试提取具体的C++项目信息
            if "计算器" in text:
                summary_parts.append("讨论了C++计算器程序")
            elif "俄罗斯方块" in text or "tetris" in text:
                summary_parts.append("讨论了C++俄罗斯方块游戏")
            elif "贪吃蛇" in text or "snake" in text:
                summary_parts.append("讨论了C++贪吃蛇游戏")
            elif "井字棋" in text or "tic-tac-toe" in text:
                summary_parts.append("讨论了C++井字棋游戏")
            else:
                summary_parts.append("讨论了C++编程相关内容")
        
        if "Java" in text or "java" in text:
            # 尝试提取具体的Java项目信息
            if "计算器" in text:
                summary_parts.append("讨论了Java计算器程序")
            elif "俄罗斯方块" in text or "tetris" in text:
                summary_parts.append("讨论了Java俄罗斯方块游戏")
            elif "贪吃蛇" in text or "snake" in text:
                summary_parts.append("讨论了Java贪吃蛇游戏")
            elif "井字棋" in text or "tic-tac-toe" in text:
                summary_parts.append("讨论了Java井字棋游戏")
            else:
                summary_parts.append("讨论了Java编程相关内容")
        
        if "COBOL" in text or "cobol" in text:
            summary_parts.append("讨论了COBOL编程相关内容")
        
        if "文件" in text and ("创建" in text or "保存" in text):
            # 尝试提取具体的文件信息
            import re
            # 提取文件类型
            if ".py" in text or "Python" in text:
                summary_parts.append("创建或保存了Python文件")
            elif ".cpp" in text or "C++" in text:
                summary_parts.append("创建或保存了C++文件")
            elif ".java" in text or "Java" in text:
                summary_parts.append("创建或保存了Java文件")
            elif ".txt" in text:
                summary_parts.append("创建或保存了文本文件")
            else:
                summary_parts.append("创建或保存了文件")
        
        if "文件夹" in text or "目录" in text:
            summary_parts.append("创建了文件夹")
        
        # 游戏和项目相关的总结已经在编程部分处理了，这里不再重复
        
        # 检查语言介绍相关的对话
        if "希伯来语" in text or "俄语" in text or "英语" in text or "日语" in text or "法语" in text or "德语" in text or "西班牙语" in text:
            if "介绍" in text and "自己" in text:
                language = "希伯来语" if "希伯来语" in text else \
                          "俄语" if "俄语" in text else \
                          "英语" if "英语" in text else \
                          "日语" if "日语" in text else \
                          "法语" if "法语" in text else \
                          "德语" if "德语" in text else \
                          "西班牙语" if "西班牙语" in text else "外语"
                summary_parts.append(f"用{language}进行了自我介绍")
            else:
                summary_parts.append("进行了语言相关的对话")
        
        if "设置" in text:
            summary_parts.append("进行了系统设置相关操作")
        
        if "记忆" in text or "识底深湖" in text:
            summary_parts.append("查看了记忆系统")
        
        if "MCP" in text or "工具" in text:
            summary_parts.append("使用了MCP工具")
        
        # 如果没有找到具体内容，返回通用描述
        if not summary_parts:
            summary_parts.append("进行了日常对话交流")
        
        # 组合总结内容，按时间顺序排列
        if len(summary_parts) > 1:
            # 如果有多个操作，用"然后"连接，表示时间顺序
            summary = "，然后".join(summary_parts)
        else:
            summary = "，".join(summary_parts)
        
        # 控制长度在40-60字之间
        if len(summary) > 60:
            summary = summary[:57] + "..."
        elif len(summary) < 25:
            summary += "，包含具体的对话内容和操作步骤"
        
        return summary

    def _extract_keywords(self, text):
        """提取关键词"""
        keywords = []
        common_words = [
            # 基础功能
            '天气', '时间', '搜索', '打开', '计算', '距离', '系统', '文件', '笔记', '穿衣', '出门', '建议',
            # 旅游景点
            '历史', '景点', '旅游', '参观', '游览', '建筑', '教堂', '大教堂', '广场', '公园', '博物馆', '遗址', '古迹',
            '故宫', '天安门', '红场', '莫斯科', '柏林', '勃兰登堡门', '法兰克福', '铁桥', '桥',
            # 编程相关
            'Python', 'python', 'C++', 'c++', 'COBOL', 'cobol', '编程', '代码', '程序', '开发',
            # 文件操作
            '创建', '保存', '文件夹', '目录', '歌单', '音乐', '歌曲', '推荐',
            # 游戏相关
            '计算器', '俄罗斯方块', 'tetris', '贪吃蛇', 'snake', '井字棋', 'tic-tac-toe', '游戏',
            # 技术相关
            '爬虫', 'crawler', '数据分析', 'data', 'Hello World', 'hello',
            # 系统功能
            '设置', '记忆', '识底深湖', 'MCP', '工具', 'API', '配置'
        ]
        
        for word in common_words:
            if word in text:
                keywords.append(word)
        
        return keywords

    def _extract_conversation_details(self):
        """提取对话详情，生成精简的对话记录"""
        if not self.current_conversation:
            return ""
        
        # 使用AI智能总结整个对话，而不是逐条关键词识别
        conversation_text = ""
        for conv in self.current_conversation:
            user_input = conv.get("user_input", "")
            ai_response = conv.get("ai_response", "")
            
            if user_input == "系统":
                conversation_text += f"露尼西亚: {ai_response}\n"
            else:
                conversation_text += f"指挥官: {user_input}\n露尼西亚: {ai_response}\n"
        
        # 强制使用AI总结，不启用后备方案
        try:
            ai_result = self._ai_summarize_conversation_details(conversation_text)
            if ai_result and len(ai_result.strip()) > 10:  # 确保AI返回了有效结果
                return ai_result
            else:
                print("⚠️ AI总结返回空结果，尝试重新生成")
                # 再次尝试AI总结
                ai_result = self._ai_summarize_conversation_details(conversation_text)
                return ai_result if ai_result and len(ai_result.strip()) > 10 else "AI总结失败"
        except Exception as e:
            print(f"⚠️ AI总结失败: {str(e)}")
            return "AI总结失败，请检查API配置"
    
    def _ai_summarize_conversation_details(self, conversation_text):
        """使用AI总结对话详情"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试AI对话记录总结 (第{attempt + 1}次)")
                # 使用专门的记忆总结AI代理
                details = self.summary_agent.summarize_conversation_details(conversation_text)
                if details and len(details.strip()) > 10:
                    print(f"✅ AI对话记录总结成功: {details[:50]}...")
                    return details
                else:
                    print(f"⚠️ AI对话记录总结返回空结果 (第{attempt + 1}次)")
                    if attempt < max_retries - 1:
                        print("🔄 等待2秒后重试...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        print("❌ AI对话记录总结最终失败")
                        return "AI总结失败"
            except Exception as e:
                print(f"⚠️ AI对话记录总结失败 (第{attempt + 1}次): {str(e)}")
                if attempt < max_retries - 1:
                    print("🔄 等待2秒后重试...")
                    import time
                    time.sleep(2)
                    continue
                else:
                    print("❌ AI对话记录总结最终失败")
                    return "AI总结失败，请检查API配置"
    
    def _fallback_conversation_details(self):
        """后备方案：使用原来的关键词识别方法"""
        if not self.current_conversation:
            return ""
        
        details = []
        for conv in self.current_conversation:
            user_input = conv.get("user_input", "")
            ai_response = conv.get("ai_response", "")
            
            # 处理系统消息（如自我介绍）
            if user_input == "系统":
                details.append(f"露尼西亚: {ai_response}")
                continue
            
            # 精简用户输入
            if len(user_input) > 20:
                user_input = user_input[:17] + "..."
            
            # 智能精简AI回应，保留具体信息
            ai_response = self._smart_summarize_ai_response(ai_response)
            
            details.append(f"指挥官: {user_input}")
            details.append(f"露尼西亚: {ai_response}")
        
        return "\n".join(details)
    
    def _smart_summarize_ai_response(self, ai_response):
        """智能精简AI回应，保留具体信息"""
        if len(ai_response) <= 50:
            return ai_response
        
        # 自我介绍相关（优先于音乐推荐）
        if "指挥官，您好！我是露尼西亚" in ai_response or "威廉的姐姐" in ai_response:
            return "进行了自我介绍，介绍了身份和能力"
        
        # 音乐推荐相关
        if "推荐" in ai_response and ("音乐" in ai_response or "歌单" in ai_response or "歌曲" in ai_response):
            # 提取具体的歌曲信息
            import re
            song_matches = re.findall(r'《([^》]+)》', ai_response)
            if song_matches:
                # 完整罗列所有歌曲，但控制在200字以内
                if len(song_matches) <= 5:  # 5首以内完整罗列
                    songs_text = "、".join([f"《{song}》" for song in song_matches])
                    return f"推荐了音乐{songs_text}"
                else:  # 超过5首，前5首+总数
                    songs_text = "、".join([f"《{song}》" for song in song_matches[:5]])
                    return f"推荐了音乐{songs_text}等{len(song_matches)}首歌曲"
            else:
                # 尝试提取艺术家信息
                artist_matches = re.findall(r'-\s*([^（\n]+)', ai_response)
                if artist_matches:
                    artists = artist_matches[:3]  # 最多3个艺术家
                    artists_text = "、".join(artists)
                    return f"推荐了{artists_text}等艺术家的音乐"
                else:
                    return "推荐了音乐歌单"
        
        # 国家介绍和科普内容相关（优先于天气信息）
        elif any(keyword in ai_response for keyword in ["德国", "法国", "英国", "美国", "日本", "韩国", "俄罗斯", "中国", "介绍", "位于", "首都", "人口", "面积", "经济", "文化", "历史"]):
            # 提取国家或地区名称
            import re
            country_match = re.search(r'([德国法国英国美国日本韩国俄罗斯中国印度巴西澳大利亚加拿大意大利西班牙荷兰瑞士瑞典挪威丹麦芬兰波兰捷克匈牙利罗马尼亚保加利亚塞尔维亚克罗地亚斯洛文尼亚奥地利比利时卢森堡葡萄牙希腊土耳其以色列埃及南非尼日利亚肯尼亚埃塞俄比亚摩洛哥阿尔及利亚突尼斯利比亚苏丹南苏丹中非共和国刚果民主共和国刚果共和国加蓬赤道几内亚圣多美和普林西比喀麦隆乍得尼日尔马里布基纳法索贝宁多哥加纳科特迪瓦利比里亚塞拉利昂几内亚几内亚比绍塞内加尔冈比亚毛里塔尼亚摩洛哥阿尔及利亚突尼斯利比亚埃及苏丹南苏丹中非共和国刚果民主共和国刚果共和国加蓬赤道几内亚圣多美和普林西比喀麦隆乍得尼日尔马里布基纳法索贝宁多哥加纳科特迪瓦利比里亚塞拉利昂几内亚几内亚比绍塞内加尔冈比亚毛里塔尼亚])(国|共和国|联邦|王国|帝国|公国|大公国|酋长国|苏丹国|哈里发国|共和国|联邦共和国|民主共和国|人民共和国|社会主义共和国|伊斯兰共和国|阿拉伯共和国|联合共和国|联邦共和国|民主联邦共和国|社会主义联邦共和国|伊斯兰联邦共和国|阿拉伯联邦共和国|联合联邦共和国|联邦民主共和国|联邦社会主义共和国|联邦伊斯兰共和国|联邦阿拉伯共和国|联邦联合共和国|民主联邦社会主义共和国|民主联邦伊斯兰共和国|民主联邦阿拉伯共和国|民主联邦联合共和国|社会主义联邦民主共和国|社会主义联邦伊斯兰共和国|社会主义联邦阿拉伯共和国|社会主义联邦联合共和国|伊斯兰联邦民主共和国|伊斯兰联邦社会主义共和国|伊斯兰联邦阿拉伯共和国|伊斯兰联邦联合共和国|阿拉伯联邦民主共和国|阿拉伯联邦社会主义共和国|阿拉伯联邦伊斯兰共和国|阿拉伯联邦联合共和国|联合联邦民主共和国|联合联邦社会主义共和国|联合联邦伊斯兰共和国|联合联邦阿拉伯共和国|联合联邦联合共和国)?', ai_response)
            if country_match:
                country = country_match.group(1)
                # 提取关键信息，生成缩写句子
                summary_parts = []
                
                # 提取地理位置
                if "位于" in ai_response:
                    location_match = re.search(r'位于([^，。\s]+)', ai_response)
                    if location_match:
                        summary_parts.append(f"位于{location_match.group(1)}")
                
                # 提取首都
                if "首都" in ai_response:
                    capital_match = re.search(r'首都([^，。\s]+)', ai_response)
                    if capital_match:
                        summary_parts.append(f"首都{capital_match.group(1)}")
                
                # 提取人口
                if "人口" in ai_response:
                    population_match = re.search(r'人口([^，。\s]+)', ai_response)
                    if population_match:
                        summary_parts.append(f"人口{population_match.group(1)}")
                
                # 提取面积
                if "面积" in ai_response:
                    area_match = re.search(r'面积([^，。\s]+)', ai_response)
                    if area_match:
                        summary_parts.append(f"面积{area_match.group(1)}")
                
                # 构建总结
                if summary_parts:
                    return f"介绍了{country}：{''.join(summary_parts[:3])}"  # 最多3个关键信息
                else:
                    return f"介绍了{country}的基本信息"
            else:
                # 没有找到具体国家，但包含介绍相关内容
                if "介绍" in ai_response:
                    return "进行了知识介绍"
                else:
                    return "提供了科普信息"
        
        # 天气查询相关
        elif "天气" in ai_response:
            # 提取具体的天气信息
            import re
            weather_details = []
            
            # 提取城市信息
            city_match = re.search(r'([北京上海广州深圳成都重庆武汉西安南京杭州苏州天津青岛大连厦门宁波无锡长沙郑州济南福州合肥南昌南宁贵阳昆明太原石家庄哈尔滨长春沈阳呼和浩特银川西宁拉萨乌鲁木齐])(市|省)?', ai_response)
            if city_match:
                city = city_match.group(1)
                weather_details.append(city)
            
            # 提取温度信息
            temp_matches = re.findall(r'(\d+)°C', ai_response)
            if temp_matches:
                if len(temp_matches) == 1:
                    weather_details.append(f"{temp_matches[0]}°C")
                else:
                    weather_details.append(f"{temp_matches[0]}-{temp_matches[-1]}°C")
            
            # 提取天气状况
            if "雷阵雨" in ai_response:
                weather_details.append("雷阵雨")
            elif "多云" in ai_response:
                weather_details.append("多云")
            elif "晴天" in ai_response:
                weather_details.append("晴天")
            elif "阴天" in ai_response:
                weather_details.append("阴天")
            elif "小雨" in ai_response:
                weather_details.append("小雨")
            
            # 提取风力信息
            wind_matches = re.findall(r'([东南西北]风\d+-\d+级)', ai_response)
            if wind_matches:
                weather_details.append(wind_matches[0])
            
            # 构建天气总结
            if weather_details:
                return f"提供了{''.join(weather_details)}的天气信息"
            else:
                return "提供了天气信息"
        
        # 文件操作相关
        elif "文件" in ai_response and ("成功" in ai_response or "写入成功" in ai_response):
            # 提取文件路径和类型
            import re
            file_match = re.search(r'文件\s*([^写入成功]+)', ai_response)
            if file_match:
                file_path = file_match.group(1).strip()
                return f"文件{file_path}创建成功"
            else:
                return "文件创建成功"
        
        # 时间查询相关
        elif "时间" in ai_response:
            # 提取具体时间信息
            import re
            time_match = re.search(r'(\d{1,2}:\d{2})', ai_response)
            if time_match:
                time_str = time_match.group(1)
                return f"提供了{time_str}的时间信息"
            else:
                return "提供了时间信息"
        
        # 编程相关
        elif any(keyword in ai_response for keyword in ["Python", "Java", "C++", "JavaScript", "代码", "程序"]):
            # 提取编程语言和项目类型
            import re
            if "Python" in ai_response:
                if "计算器" in ai_response:
                    return "提供了Python计算器代码"
                elif "俄罗斯方块" in ai_response or "tetris" in ai_response:
                    return "提供了Python俄罗斯方块游戏代码"
                elif "贪吃蛇" in ai_response or "snake" in ai_response:
                    return "提供了Python贪吃蛇游戏代码"
                else:
                    return "提供了Python编程代码"
            elif "Java" in ai_response:
                if "计算器" in ai_response:
                    return "提供了Java计算器代码"
                elif "游戏" in ai_response:
                    return "提供了Java游戏代码"
                else:
                    return "提供了Java编程代码"
            elif "C++" in ai_response:
                if "游戏" in ai_response:
                    return "提供了C++游戏代码"
                else:
                    return "提供了C++编程代码"
            else:
                return "提供了编程代码"
        
        # 语言介绍相关
        elif any(lang in ai_response for lang in ["希伯来语", "俄语", "英语", "日语", "法语", "德语", "西班牙语"]):
            for lang in ["希伯来语", "俄语", "英语", "日语", "法语", "德语", "西班牙语"]:
                if lang in ai_response:
                    return f"用{lang}进行了自我介绍"
            return "进行了语言介绍"
        
        # 其他情况，保留关键信息
        else:
            # 尝试提取关键信息，避免过长
            if len(ai_response) > 100:
                # 寻找句号或逗号作为分割点
                sentences = ai_response.split('。')
                if len(sentences) > 1:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) <= 50:
                        return first_sentence
                    else:
                        return first_sentence[:47] + "..."
                else:
                    return ai_response[:47] + "..."
            else:
                return ai_response

    def search_relevant_memories(self, user_input, current_context=""):
        """搜索相关记忆（向量相似度搜索）"""
        return self._search_by_vectors(user_input, current_context)
    
    def _search_by_vectors(self, user_input, current_context=""):
        """使用向量相似度搜索相关记忆"""
        try:
            print(f"🗄️ 开始向量搜索记忆: {user_input[:50]}...")
            
            # 生成用户输入的向量
            user_vector = self.vector_encoder.encode_text(user_input)
            
            if user_vector is None:
                print("⚠️ 无法生成用户输入的向量，回退到关键词搜索")
                return self._search_by_keywords(user_input, current_context)
            
            relevant_memories = []
            
            for entry in self.memory_index["topics"]:
                if "topic_vector" in entry and entry["topic_vector"]:
                    # 计算主题向量相似度
                    topic_similarity = self.vector_encoder.calculate_similarity(
                        user_vector, entry["topic_vector"]
                    )
                    
                    # 计算内容向量相似度（如果存在）
                    details_similarity = 0.0
                    if "details_vector" in entry and entry["details_vector"]:
                        details_similarity = self.vector_encoder.calculate_similarity(
                            user_vector, entry["details_vector"]
                        )
                    
                    # 综合相似度：主题权重70%，内容权重30%
                    combined_similarity = topic_similarity * 0.7 + details_similarity * 0.3
                    
                    if combined_similarity > 0.2 or details_similarity > 0.3:  # 双重阈值
                        entry_copy = entry.copy()
                        entry_copy["relevance_score"] = combined_similarity
                        entry_copy["topic_similarity"] = topic_similarity
                        entry_copy["details_similarity"] = details_similarity
                        relevant_memories.append(entry_copy)
                else:
                    # 对于没有向量的记忆，使用关键词匹配作为备用
                    keyword_score = self._calculate_keyword_relevance(entry, user_input, current_context)
                    if keyword_score > 0.3:
                        entry_copy = entry.copy()
                        entry_copy["relevance_score"] = keyword_score * 0.5  # 降低权重
                        relevant_memories.append(entry_copy)
            
            # 按相关性排序，然后按时间排序（最新的优先）
            relevant_memories.sort(key=lambda x: (-x["relevance_score"], -self._get_timestamp_score(x)))
            
            # 限制返回数量
            max_memories = 5
            relevant_memories = relevant_memories[:max_memories]
            
            if relevant_memories:
                print(f"🎯 找到 {len(relevant_memories)} 条相关记忆 (双重向量搜索)")
                for i, memory in enumerate(relevant_memories):
                    score = memory["relevance_score"]
                    topic_sim = memory.get("topic_similarity", 0)
                    details_sim = memory.get("details_similarity", 0)
                    topic = memory.get("topic", "未知主题")[:30]
                    print(f"   {i+1}. [总:{score:.3f}|主题:{topic_sim:.3f}|内容:{details_sim:.3f}] {topic}")
            else:
                print("🔍 未找到相关记忆，尝试关键词搜索...")
                return self._search_by_keywords(user_input, current_context)
            
            return relevant_memories
            
        except Exception as e:
            print(f"⚠️ 向量搜索失败: {e}")
            print("🔄 回退到关键词搜索...")
            return self._search_by_keywords(user_input, current_context)
    
    
    def _search_by_keywords(self, user_input, current_context=""):
        """关键词搜索备用方法"""
        try:
            print(f"🔍 开始关键词搜索记忆: {user_input[:50]}...")
            relevant_memories = []
            user_keywords = self._extract_keywords(user_input)
            
            for entry in self.memory_index["topics"]:
                relevance_score = self._calculate_keyword_relevance(entry, user_input, current_context)
                if relevance_score > 0.3:  # 相关性阈值
                    entry_copy = entry.copy()
                    entry_copy["relevance_score"] = relevance_score
                    relevant_memories.append(entry_copy)
            
            # 按相关性排序，然后按时间排序（最新的优先）
            relevant_memories.sort(key=lambda x: (-x["relevance_score"], -self._get_timestamp_score(x)))
            
            if relevant_memories:
                print(f"🎯 找到 {len(relevant_memories)} 条相关记忆 (关键词搜索)")
                for i, memory in enumerate(relevant_memories[:5]):
                    score = memory["relevance_score"]
                    topic = memory.get("topic", "未知主题")[:30]
                    print(f"   {i+1}. [{score:.3f}] {topic}")
            else:
                print("❌ 未找到相关记忆")
            
            return relevant_memories[:5]  # 返回最相关的5个记忆
            
        except Exception as e:
            print(f"⚠️ 关键词搜索失败: {str(e)}")
            return []

    def _calculate_keyword_relevance(self, memory_entry, user_keywords, current_context):
        """计算关键词相关性分数（备用方法）"""
        score = 0.0
        
        # 关键词匹配
        memory_keywords = memory_entry.get("keywords", [])
        for keyword in user_keywords:
            if keyword in memory_keywords:
                score += 0.4
        
        # 主题匹配
        memory_topic = memory_entry.get("topic", "")
        for keyword in user_keywords:
            if keyword in memory_topic:
                score += 0.3
        
        # 时间相关性（最近7天的记忆权重更高）
        try:
            memory_date = datetime.datetime.strptime(memory_entry.get("date", ""), "%Y-%m-%d")
            current_date = datetime.datetime.now()
            days_diff = (current_date - memory_date).days
            if days_diff <= 7:
                score += 0.2
            elif days_diff <= 30:
                score += 0.1
        except:
            pass
        
        return min(score, 1.0)

    def should_recall_memory(self, user_input):
        """判断是否需要回忆"""
        # 关键词触发 - 更精确的关键词
        recall_keywords = ['记得', '说过', '讨论过', '回忆', '继续', '接着', '历史', '以前', '曾经', '之前', '上个']
        
        # 如果用户询问的是"上一个"相关的问题，优先使用本次会话记忆，不触发历史记忆
        # 但如果是"之前"相关的问题，应该触发历史记忆
        if any(word in user_input for word in ['上一个', '刚才']):
            return False
            
        return any(keyword in user_input for keyword in recall_keywords)

    def generate_memory_context(self, relevant_memories, user_input):
        """生成记忆上下文"""
        if not relevant_memories:
            return ""
            
        try:
            context_parts = []
            
            for memory in relevant_memories:
                topic = memory.get("topic", "")
                timestamp = memory.get("timestamp", "")
                date = memory.get("date", "")
                
                context_part = f"【{date} {timestamp}】{topic}"
                context_parts.append(context_part)
            
            if context_parts:
                return "\n".join(context_parts)
            
            return ""
            
        except Exception as e:
            print(f"生成记忆上下文失败: {str(e)}")
            return ""

    def get_recent_memories(self, limit=100):
        """获取最近的历史记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            # 按日期和时间倒序排列，获取最近的记忆
            sorted_topics = sorted(topics, key=lambda x: (x.get("date", ""), x.get("timestamp", "")), reverse=True)
            return sorted_topics[:limit]
        except Exception as e:
            print(f"获取最近记忆失败: {str(e)}")
            return []

    def get_first_memory(self):
        """获取第一条记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            if not topics:
                return None
        
            # 按日期和时间正序排列，获取最早的记忆
            # 确保日期格式正确，处理可能的空值
            def sort_key(topic):
                date = topic.get("date", "")
                timestamp = topic.get("timestamp", "")
                # 如果日期为空，使用一个很大的日期确保排在最后
                if not date:
                    return ("9999-12-31", timestamp)
                return (date, timestamp)
            
            sorted_topics = sorted(topics, key=sort_key)
            first_memory = sorted_topics[0]
            
            # 添加调试信息
            print(f"🔍 找到第一条记忆: {first_memory.get('date', '未知')} {first_memory.get('timestamp', '未知')} - {first_memory.get('topic', '未知主题')}")
            
            return first_memory
        except Exception as e:
            print(f"获取第一条记忆失败: {str(e)}")
            return None

    def get_memory_stats(self):
        """获取记忆统计信息"""
        try:
            topics = self.memory_index.get("topics", [])
            total_topics = len(topics)
            important_topics = len([topic for topic in topics if topic.get("is_important", False)])
            total_log_files = len([f for f in os.listdir(self.chat_logs_dir) if f.endswith('.json')]) if os.path.exists(self.chat_logs_dir) else 0
            
            stats = {
                "total_topics": total_topics,
                "important_topics": important_topics,
                "total_log_files": total_log_files,
                "memory_file_size": os.path.getsize(self.memory_file) if os.path.exists(self.memory_file) else 0,
                "current_conversation_count": len(self.current_conversation)
            }
            
            return stats
        except Exception as e:
            print(f"获取记忆统计失败: {str(e)}")
            return {"total_topics": 0, "important_topics": 0, "total_log_files": 0, "memory_file_size": 0, "current_conversation_count": 0}
    

    def mark_as_important(self, topic_index):
        """标记为重点记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            if 0 <= topic_index < len(topics):
                topics[topic_index]["is_important"] = True
                self.save_memory()
                return True
            return False
        except Exception as e:
            print(f"标记重点记忆失败: {str(e)}")
            return False

    def unmark_as_important(self, topic_index):
        """取消重点记忆标记"""
        try:
            topics = self.memory_index.get("topics", [])
            if 0 <= topic_index < len(topics):
                topics[topic_index]["is_important"] = False
                self.save_memory()
                return True
            return False
        except Exception as e:
            print(f"取消重点记忆标记失败: {str(e)}")
            return False

    def get_important_memories(self):
        """获取所有重点记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            important_memories = [topic for topic in topics if topic.get("is_important", False)]
            return important_memories
        except Exception as e:
            print(f"获取重点记忆失败: {str(e)}")
            return []

    def mark_first_memory_as_important(self):
        """将第一条记忆标记为重点记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            if topics:
                topics[0]["is_important"] = True
                self.save_memory()
                return True
            return False
        except Exception as e:
            print(f"标记第一条记忆为重点记忆失败: {str(e)}")
            return False

    def ensure_first_memory_important(self):
        """确保第一条记忆是重点记忆"""
        try:
            topics = self.memory_index.get("topics", [])
            if topics and not topics[0].get("is_important", False):
                topics[0]["is_important"] = True
                self.save_memory()
                return True
            return False
        except Exception as e:
            print(f"确保第一条记忆为重点记忆失败: {str(e)}")
            return False
