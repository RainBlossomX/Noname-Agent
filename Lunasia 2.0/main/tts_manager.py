# -*- coding: utf-8 -*-
"""
Azure TTS管理器
处理文本转语音功能
"""

import asyncio
import threading
import queue
import time
from typing import Optional, Callable
import pygame
import tempfile
import os

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure Speech SDK未安装，TTS功能将不可用")

class TTSManager:
    """Azure TTS管理器"""
    
    def __init__(self, azure_key: str = "", region: str = "eastasia"):
        self.azure_key = azure_key
        self.region = region
        self.enabled = False
        self.voice_name = "zh-CN-XiaoxiaoNeural"  # 默认女声
        self.speech_config = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_playback = False
        
        # 初始化pygame音频
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_available = True
        except Exception as e:
            print(f"⚠️ 音频初始化失败: {e}")
            self.audio_available = False
        
        # 初始化Azure配置
        if AZURE_AVAILABLE and azure_key:
            self._init_azure_config()
    
    def _init_azure_config(self):
        """初始化Azure配置"""
        try:
            # 🔥 验证API密钥和区域
            if not self.azure_key or len(self.azure_key.strip()) == 0:
                print(f"❌ Azure TTS API密钥为空")
                self.enabled = False
                return
            
            if not self.region or len(self.region.strip()) == 0:
                print(f"❌ Azure TTS区域为空")
                self.enabled = False
                return
            
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_key, 
                region=self.region
            )
            # 🔥 确保语音名称在配置中正确设置
            if self.voice_name:
                self.speech_config.speech_synthesis_voice_name = self.voice_name
                print(f"🔍 [TTS配置] 语音名称: {self.voice_name}")
            self.speech_config.speech_synthesis_speaking_rate = 1.0  # 语速
            
            # 🔥 不设置输出格式，使用默认格式（由AudioOutputConfig决定）
            # AudioOutputConfig会自动设置合适的格式
            
            self.enabled = True
            print(f"✅ Azure TTS配置成功 (区域: {self.region}, 语音: {self.voice_name})")
        except Exception as e:
            print(f"❌ Azure TTS配置失败: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False
    
    def update_config(self, azure_key: str, region: str):
        """更新Azure配置"""
        self.azure_key = azure_key
        self.region = region
        if azure_key:
            self._init_azure_config()
        else:
            self.enabled = False
    
    def set_voice(self, voice_name: str):
        """设置语音"""
        self.voice_name = voice_name
        if self.speech_config:
            self.speech_config.speech_synthesis_voice_name = voice_name
    
    def set_speaking_rate(self, rate: float):
        """设置语速 (0.5-2.0)"""
        if self.speech_config:
            self.speech_config.speech_synthesis_speaking_rate = rate
    
    def synthesize_text(self, text: str) -> Optional[str]:
        """合成文本为音频文件"""
        if not self.enabled or not AZURE_AVAILABLE:
            return None
        
        # 🔥 记录原始文本信息
        original_length = len(text)
        original_preview = text[:50] if len(text) > 0 else ""
        
        # 🔥 清理文本，移除可能导致API错误的特殊字符
        # 移除或替换可能导致问题的字符
        import re
        # 先移除emoji和特殊符号（保留基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）【】《》·-]', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 限制文本长度
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length]
        
        # 如果文本为空，返回None
        if not text or len(text.strip()) == 0:
            return None
        
        temp_file_path = None
        synthesizer = None
        audio_config = None
        
        try:
            # 创建临时音频文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file_path = temp_file.name
            temp_file.close()
            
            # 使用内存输出，从result.audio_data直接获取数据
            abs_temp_path = os.path.abspath(temp_file_path)
            
            try:
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config
                )
            except Exception as config_error:
                print(f"❌ TTS配置失败: {config_error}")
                import traceback
                traceback.print_exc()
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                return None
            
            # 执行TTS合成
            result = None
            try:
                result = synthesizer.speak_text(text)
                
                # 如果成功，从result.audio_data写入文件
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    try:
                        audio_data = result.audio_data
                        if audio_data:
                            with open(abs_temp_path, 'wb') as audio_file:
                                audio_file.write(audio_data)
                            return abs_temp_path
                        else:
                            print(f"⚠️ TTS合成成功但audio_data为空")
                            return None
                    except Exception as write_error:
                        print(f"⚠️ TTS写入文件失败: {write_error}")
                        return None
                
            except Exception as timeout_error:
                print(f"⚠️ TTS合成过程异常: {timeout_error}")
                if synthesizer:
                    synthesizer = None
                if audio_config:
                    audio_config = None
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                return None
            
            # 如果没有获取到结果，直接返回
            if result is None:
                print(f"⚠️ TTS合成未返回结果")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                return None
            
            # 检查结果
            try:
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return abs_temp_path
                elif result.reason == speechsdk.ResultReason.Canceled:
                    print(f"⚠️ TTS合成被取消")
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    return None
                else:
                    print(f"⚠️ TTS合成失败: {result.reason}")
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    return None
            except Exception as result_error:
                print(f"⚠️ TTS合成过程异常: {result_error}")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                return None
                
        except Exception as e:
            print(f"⚠️ TTS合成异常: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return None
        finally:
            # 确保资源释放
            try:
                if synthesizer:
                    synthesizer = None
                if audio_config:
                    audio_config = None
            except:
                pass
    
    def play_audio(self, audio_file: str):
        """播放音频文件"""
        if not self.audio_available:
            return
        
        try:
            # 停止当前播放
            if self.is_playing:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()  # 🔥 关键：卸载当前音频
            
            # 播放新音频
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            self.is_playing = True
            
            # 等待播放完成
            while pygame.mixer.music.get_busy() and not self.stop_playback:
                time.sleep(0.1)
            
            self.is_playing = False
            
           
            pygame.mixer.music.unload()
            time.sleep(0.1)  # 等待文件句柄释放
            
            # 清理临时文件（带重试机制）
            max_retries = 5
            for i in range(max_retries):
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                        break
                except PermissionError:
                    if i < max_retries - 1:
                        time.sleep(0.2)  # 等待后重试
                    else:
                        print(f"⚠️ 无法删除临时文件: {audio_file}，稍后系统会自动清理")
                except Exception as e:
                    print(f"⚠️ 删除临时文件失败: {e}")
                    break
                
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")
            self.is_playing = False
    
    def speak_text(self, text: str):
        """文本转语音并播放"""
        if not self.enabled:
            return
        
        # 🔥 如果文本太长，分段处理（Azure TTS限制较小，建议400字符/段）
        max_chunk_length = 400  # 每段最多400字符，避免被取消
        
        # 在新线程中处理TTS
        def tts_worker():
            # 如果文本较短，直接处理
            if len(text) <= max_chunk_length:
                audio_file = self.synthesize_text(text)
                if audio_file:
                    self.play_audio(audio_file)
            else:
                # 分段处理长文本
                # 按句子分割（尽量保持完整性）
                import re
                sentences = re.split(r'([。！？\n])', text)
                current_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i+1]
                    else:
                        sentence = sentences[i]
                    
                    # 如果当前块加上新句子不会超长，就添加
                    if len(current_chunk) + len(sentence) <= max_chunk_length:
                        current_chunk += sentence
                    else:
                        # 处理当前块
                        if current_chunk.strip():
                            audio_file = self.synthesize_text(current_chunk.strip())
                            if audio_file:
                                self.play_audio(audio_file)
                                # 等待播放完成再播放下一个
                                while self.is_playing:
                                    time.sleep(0.1)
                        
                        # 开始新块
                        current_chunk = sentence
                
                # 处理最后一块
                if current_chunk.strip():
                    audio_file = self.synthesize_text(current_chunk.strip())
                    if audio_file:
                        self.play_audio(audio_file)
        
        thread = threading.Thread(target=tts_worker, daemon=True)
        thread.start()
    
    def stop_speaking(self):
        """停止当前播放"""
        self.stop_playback = True
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
    
    def get_available_voices(self) -> list:
        """获取可用的中文女声列表"""
        return [
            ("zh-CN-XiaoxiaoNeural", "晓晓 (推荐)"),
            ("zh-CN-XiaoyiNeural", "晓伊"),
            ("zh-CN-YunxiNeural", "云希"),
            ("zh-CN-YunyangNeural", "云扬"),
            ("zh-CN-XiaochenNeural", "晓辰"),
            ("zh-CN-XiaohanNeural", "晓涵"),
            ("zh-CN-XiaomoNeural", "晓墨"),
            ("zh-CN-XiaoxuanNeural", "晓萱"),
            ("zh-CN-XiaoyanNeural", "晓颜"),
            ("zh-CN-XiaoyouNeural", "晓悠"),
        ]
    
    def is_available(self) -> bool:
        """检查TTS是否可用"""
        return self.enabled and AZURE_AVAILABLE and self.audio_available
    
    def cleanup(self):
        """清理资源"""
        self.stop_speaking()
        try:
            pygame.mixer.quit()
        except:
            pass
