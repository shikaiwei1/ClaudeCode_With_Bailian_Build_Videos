#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼平台Qwen-TTS语音合成API示例
API文档: https://help.aliyun.com/zh/model-studio/qwen-tts-api

本示例展示如何使用百炼平台的Qwen-TTS语音合成API
支持多种音色、语速控制、音量调节等功能

Author: John Chen
"""

import os
import time
import json
import requests
import base64
from typing import Optional, Dict, Any, List


class BailianTTSGeneration:
    """
    百炼平台Qwen-TTS语音合成API客户端

    支持的功能:
    - 文本转语音合成
    - 多种音色选择
    - 语速控制
    - 音量调节
    - 音频格式选择

    支持的模型:
    - qwen-tts-v1: Qwen语音合成模型
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端

        Args:
            api_key: 百炼平台API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量或提供api_key参数")

        # API端点
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 支持的音色列表 - 根据官方文档更新
        self.available_voices = {
            "Cherry": "Cherry - 甜美女声",
            "Chelsie": "Chelsie - 温和女声",
            "zhichu": "知初 - 温和知性女声",
            "zhixiaobai": "知小白 - 活泼可爱女声",
            "zhixiaoxia": "知小夏 - 清新甜美女声",
            "zhimiao": "知妙 - 优雅成熟女声",
            "zhiyan": "知燕 - 温柔亲和女声",
            "zhiyuan": "知远 - 稳重磁性男声",
            "zhiyun": "知云 - 清朗自然男声",
            "zhishuo": "知硕 - 成熟稳重男声",
            "zhiwei": "知维 - 理性专业男声",
            "zhihao": "知豪 - 豪迈大气男声",
        }

        # 支持的音频格式
        self.available_formats = {
            "wav": "WAV格式 - 无损音质",
            "mp3": "MP3格式 - 压缩音质",
            "pcm": "PCM格式 - 原始音频数据",
        }

    def synthesize_speech(
        self,
        text: str,
        voice: str = "Cherry",
        format: str = "wav",
        sample_rate: int = 22050,
        volume: float = 1.0,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """
        合成语音

        Args:
            text: 要合成的文本内容（最大500字符）
            voice: 音色选择，默认"zhichu"（知初）
            format: 音频格式，支持"wav", "mp3", "pcm"，默认"wav"
            sample_rate: 采样率，支持8000, 16000, 22050, 24000, 44100, 48000，默认22050
            volume: 音量，范围0.1-2.0，默认1.0
            speech_rate: 语速，范围0.5-2.0，默认1.0（1.0为正常语速）
            pitch_rate: 音调，范围0.5-2.0，默认1.0（1.0为正常音调）

        Returns:
            合成结果，包含音频数据（base64编码）
        """
        # 验证参数
        if len(text) > 500:
            raise ValueError("文本长度不能超过500字符")

        if voice not in self.available_voices:
            raise ValueError(
                f"不支持的音色: {voice}，支持的音色: {list(self.available_voices.keys())}"
            )

        if format not in self.available_formats:
            raise ValueError(
                f"不支持的格式: {format}，支持的格式: {list(self.available_formats.keys())}"
            )

        if not (0.1 <= volume <= 2.0):
            raise ValueError("音量范围应在0.1-2.0之间")

        if not (0.5 <= speech_rate <= 2.0):
            raise ValueError("语速范围应在0.5-2.0之间")

        if not (0.5 <= pitch_rate <= 2.0):
            raise ValueError("音调范围应在0.5-2.0之间")

        # 构建请求数据 - 按照官方文档格式
        data = {
            "model": "qwen-tts",
            "input": {
                "text": text,
                "voice": voice
            }
        }
        
        # 如果有其他参数，添加到parameters中
        if format != "wav" or sample_rate != 22050 or volume != 1.0 or speech_rate != 1.0 or pitch_rate != 1.0:
            data["parameters"] = {}
            if format != "wav":
                data["parameters"]["format"] = format
            if sample_rate != 22050:
                data["parameters"]["sample_rate"] = sample_rate
            if volume != 1.0:
                data["parameters"]["volume"] = volume
            if speech_rate != 1.0:
                data["parameters"]["speech_rate"] = speech_rate
            if pitch_rate != 1.0:
                data["parameters"]["pitch_rate"] = pitch_rate

        print(f"开始语音合成...")
        print(f"文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"音色: {voice} ({self.available_voices[voice]})")
        print(f"格式: {format} ({self.available_formats[format]})")
        print(f"采样率: {sample_rate}Hz")
        print(f"音量: {volume}")
        print(f"语速: {speech_rate}")
        print(f"音调: {pitch_rate}")

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # 检查响应格式
            if 'output' not in result:
                raise ValueError(f"API响应格式错误: 缺少output字段")
            
            output = result['output']
            
            # 检查是否有音频数据
            if 'audio' not in output:
                raise ValueError(f"API响应格式错误: 缺少audio字段")
            
            audio_info = output['audio']
            
            # 新的API返回格式包含URL而不是直接的base64数据
            if 'url' in audio_info:
                # 下载音频文件
                audio_url = audio_info['url']
                audio_response = requests.get(audio_url)
                audio_response.raise_for_status()
                audio_data = audio_response.content
            elif 'data' in audio_info and audio_info['data']:
                # 如果返回base64数据（向后兼容）
                import base64
                audio_data = base64.b64decode(audio_info['data'])
            else:
                raise ValueError("API响应中没有找到音频数据")
            
            # 返回结果
            return {
                'success': True,
                'audio_data': audio_data,
                'audio_url': audio_info.get('url', ''),
                'audio_id': audio_info.get('id', ''),
                'expires_at': audio_info.get('expires_at', 0),
                'request_id': result.get('request_id', ''),
                'usage': result.get('usage', {})
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"请求失败: {e}",
                'audio_data': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"处理失败: {e}",
                'audio_data': None
            }

    def save_audio(self, audio_data, filename: str, format: str = "wav"):
        """
        保存音频数据到文件
        
        Args:
            audio_data: 音频数据（bytes或base64字符串）
            filename: 保存的文件名（不包含扩展名）
            format: 音频格式，默认为wav
        
        Returns:
            str: 保存的文件路径
        """
        try:
            # 构建文件路径
            file_path = f"{filename}.{format}"
            
            # 处理不同类型的音频数据
            if isinstance(audio_data, bytes):
                # 直接是二进制数据
                audio_bytes = audio_data
            elif isinstance(audio_data, str):
                # base64编码的字符串
                import base64
                audio_bytes = base64.b64decode(audio_data)
            else:
                raise ValueError(f"不支持的音频数据类型: {type(audio_data)}")
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"音频文件已保存: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"保存音频文件失败: {e}")
            return None

    def get_voice_list(self) -> Dict[str, str]:
        """
        获取支持的音色列表

        Returns:
            音色字典，键为音色ID，值为音色描述
        """
        return self.available_voices.copy()

    def get_format_list(self) -> Dict[str, str]:
        """
        获取支持的音频格式列表

        Returns:
            格式字典，键为格式名，值为格式描述
        """
        return self.available_formats.copy()


def example_basic_synthesis():
    """
    示例1: 基础语音合成
    演示最基本的文本转语音功能
    """
    print("\n=== 示例1: 基础语音合成 ===")
    
    try:
        # 创建客户端
        client = BailianTTSGeneration()
        
        # 合成参数
        text = "欢迎使用百炼平台的语音合成服务，这是一个基础的语音合成示例。"
        voice = "Cherry"  # 使用Cherry音色
        format = "wav"    # 使用WAV格式
        sample_rate = 22050
        
        print("开始语音合成...")
        print(f"文本: {text}")
        print(f"音色: {voice} ({client.available_voices.get(voice, '未知音色')})")
        print(f"格式: {format} ({client.available_formats.get(format, '未知格式')})")
        print(f"采样率: {sample_rate}Hz")
        print(f"音量: 1.0")
        print(f"语速: 1.0")
        print(f"音调: 1.0")
        
        # 调用语音合成
        result = client.synthesize_speech(text)
        
        # 检查合成结果
        if result.get('success'):
            audio_data = result['audio_data']
            
            # 保存音频文件
            filename = "basic_synthesis_example"
            saved_path = client.save_audio(audio_data, filename, format)
            
            # 显示详细信息
            print(f"✅ 语音合成成功!")
            print(f"音频文件大小: {len(audio_data)} 字节")
            if result.get('audio_url'):
                print(f"原始音频URL: {result['audio_url']}")
            if result.get('audio_id'):
                print(f"音频ID: {result['audio_id']}")
            if result.get('usage'):
                usage = result['usage']
                print(f"使用统计: {usage}")
            
        else:
            print(f"❌ 合成失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")


def example_voice_comparison():
    """
    示例2: 不同音色对比
    使用不同音色合成相同文本，便于对比效果
    """
    print("\n=== 示例2: 不同音色对比 ===")
    
    client = BailianTTSGeneration()
    
    text = "人工智能技术正在改变我们的生活方式。"
    voices = ["zhichu", "zhixiaobai", "zhiyuan"]  # 选择几种不同类型的音色
    
    for voice in voices:
        print(f"\n--- 音色: {voice} ({client.available_voices.get(voice, '未知音色')}) ---")
        print("开始语音合成...")
        print(f"文本: {text}")
        print(f"音色: {voice} ({client.available_voices.get(voice, '未知音色')})")
        print(f"格式: mp3 (MP3格式 - 压缩音质)")
        print(f"采样率: 22050Hz")
        print(f"音量: 1.0")
        print(f"语速: 1.0")
        print(f"音调: 1.0")
        
        try:
            result = client.synthesize_speech(
                text=text,
                voice=voice,
                format="mp3",  # 使用MP3格式节省空间
                sample_rate=22050
            )
            
            if result.get('success'):
                audio_data = result['audio_data']
                filename = f"voice_comparison_{voice}"
                saved_path = client.save_audio(audio_data, filename, "mp3")
                print(f"✅ 音色 {voice} 合成成功")
                print(f"音频文件大小: {len(audio_data)} 字节")
            else:
                print(f"❌ 音色 {voice} 合成失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 音色 {voice} 合成失败: {e}")


def example_speed_control():
    """
    示例3: 语速控制
    演示不同语速的效果
    """
    print("\n=== 示例3: 语速控制 ===")

    client = BailianTTSGeneration()

    text = "这是一个语速控制的演示，我们将使用不同的语速来朗读这段文字。"

    # 不同语速设置
    speed_settings = [
        {"rate": 0.5, "name": "慢速"},
        {"rate": 1.0, "name": "正常"},
        {"rate": 1.5, "name": "快速"},
        {"rate": 2.0, "name": "极快"},
    ]

    for setting in speed_settings:
        print(f"\n--- 语速: {setting['name']} ({setting['rate']}x) ---")
        print("开始语音合成...")
        print(f"文本: {text}")
        print(f"音色: zhichu ({client.available_voices.get('zhichu', '未知音色')})")
        print(f"格式: wav (WAV格式 - 无损音质)")
        print(f"采样率: 22050Hz")
        print(f"音量: 1.0")
        print(f"语速: {setting['rate']}")
        print(f"音调: 1.0")

        try:
            result = client.synthesize_speech(
                text=text, voice="zhichu", speech_rate=setting["rate"], format="wav"
            )

            if result.get('success'):
                audio_data = result['audio_data']
                saved_path = client.save_audio(audio_data, f"speed_control_{setting['rate']}", "wav")
                print(f"✅ 语速 {setting['rate']} 合成成功")
                print(f"音频文件大小: {len(audio_data)} 字节")
            else:
                print(f"❌ 语速 {setting['rate']} 合成失败: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"❌ 语速 {setting['rate']} 合成失败: {e}")


def example_volume_pitch_control():
    """
    示例4: 音量和音调控制
    演示音量和音调调节的效果
    """
    print("\n=== 示例4: 音量和音调控制 ===")

    client = BailianTTSGeneration()

    text = "通过调节音量和音调，我们可以创造出不同的语音效果。"

    # 不同音量和音调设置
    settings = [
        {"volume": 0.5, "pitch": 0.8, "name": "低音量低音调"},
        {"volume": 1.0, "pitch": 1.0, "name": "正常音量正常音调"},
        {"volume": 1.5, "pitch": 1.2, "name": "高音量高音调"},
        {"volume": 2.0, "pitch": 0.6, "name": "最大音量低音调"},
    ]

    for setting in settings:
        print(
            f"\n--- {setting['name']} (音量:{setting['volume']}, 音调:{setting['pitch']}) ---"
        )
        print("开始语音合成...")
        print(f"文本: {text}")
        print(f"音色: zhiyuan ({client.available_voices.get('zhiyuan', '未知音色')})")
        print(f"格式: wav (WAV格式 - 无损音质)")
        print(f"采样率: 22050Hz")
        print(f"音量: {setting['volume']}")
        print(f"语速: 1.0")
        print(f"音调: {setting['pitch']}")

        try:
            result = client.synthesize_speech(
                text=text,
                voice="zhiyuan",  # 使用男声更容易听出音调变化
                volume=setting["volume"],
                pitch_rate=setting["pitch"],
                format="wav",
            )

            if result.get('success'):
                audio_data = result['audio_data']
                filename = f"volume_pitch_{setting['volume']}_{setting['pitch']}"
                saved_path = client.save_audio(audio_data, filename, "wav")
                print(f"✅ 设置 {setting['name']} 合成成功")
                print(f"音频文件大小: {len(audio_data)} 字节")
            else:
                print(f"❌ 设置 {setting['name']} 合成失败: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"❌ 设置 {setting['name']} 合成失败: {e}")


def example_long_text_synthesis():
    """
    示例5: 长文本合成
    演示如何处理较长的文本内容
    """
    print("\n=== 示例5: 长文本合成 ===")

    client = BailianTTSGeneration()

    # 较长的文本（接近500字符限制）
    long_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
    人工智能可以对人的意识、思维的信息过程进行模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    """

    # 清理文本（移除多余的空白字符）
    clean_text = " ".join(long_text.split())

    print(f"文本长度: {len(clean_text)} 字符")

    if len(clean_text) > 500:
        print("警告: 文本长度超过500字符限制，将截取前500字符")
        clean_text = clean_text[:500]

    print("开始语音合成...")
    print(f"文本: {clean_text[:50]}{'...' if len(clean_text) > 50 else ''}")
    print(f"音色: zhimiao ({client.available_voices.get('zhimiao', '未知音色')})")
    print(f"格式: mp3 (MP3格式 - 压缩音质)")
    print(f"采样率: 22050Hz")
    print(f"音量: 1.0")
    print(f"语速: 0.9")
    print(f"音调: 1.0")

    try:
        result = client.synthesize_speech(
            text=clean_text,
            voice="zhimiao",  # 使用优雅成熟的女声
            speech_rate=0.9,  # 稍慢的语速便于理解
            format="mp3",
        )

        if result.get('success'):
            audio_data = result['audio_data']
            saved_path = client.save_audio(audio_data, "long_text_synthesis", "mp3")
            print(f"✅ 长文本合成成功")
            print(f"音频文件大小: {len(audio_data)} 字节")

            # 显示使用统计
            if result.get('usage'):
                usage = result['usage']
                print(f"实际处理字符数: {usage.get('characters', 'N/A')}")
        else:
            print(f"❌ 长文本合成失败: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"❌ 长文本合成失败: {e}")


def example_batch_synthesis():
    """
    示例6: 批量语音合成
    演示如何批量处理多个文本
    """
    print("\n=== 示例6: 批量语音合成 ===")

    client = BailianTTSGeneration()

    # 批量文本列表
    texts = [
        {"text": "欢迎来到人工智能的世界。", "voice": "zhichu", "filename": "welcome"},
        {
            "text": "今天天气真不错，适合出去走走。",
            "voice": "zhixiaobai",
            "filename": "weather",
        },
        {
            "text": "学习新技术需要持续的努力和实践。",
            "voice": "zhiyuan",
            "filename": "learning",
        },
        {
            "text": "感谢您使用我们的语音合成服务。",
            "voice": "zhimiao",
            "filename": "thanks",
        },
    ]

    success_count = 0
    total_count = len(texts)

    for i, item in enumerate(texts):
        print(f"\n--- 处理第 {i+1}/{total_count} 个文本 ---")
        print("开始语音合成...")
        print(f"文本: {item['text']}")
        print(f"音色: {item['voice']} ({client.available_voices.get(item['voice'], '未知音色')})")
        print(f"格式: wav (WAV格式 - 无损音质)")
        print(f"采样率: 22050Hz")
        print(f"音量: 1.0")
        print(f"语速: 1.0")
        print(f"音调: 1.0")

        try:
            result = client.synthesize_speech(
                text=item["text"], voice=item["voice"], format="wav"
            )

            if result.get('success'):
                audio_data = result['audio_data']
                saved_path = client.save_audio(audio_data, f"batch_{item['filename']}", "wav")
                success_count += 1
                print("✅ 合成成功")
                print(f"音频文件大小: {len(audio_data)} 字节")
            else:
                print(f"❌ 合成失败: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"❌ 合成失败: {e}")

    print(f"\n批量处理完成: {success_count}/{total_count} 成功")


def example_format_comparison():
    """
    示例7: 音频格式对比
    演示不同音频格式的效果和文件大小
    """
    print("\n=== 示例7: 音频格式对比 ===")

    client = BailianTTSGeneration()

    text = "这是一个音频格式对比的示例，我们将生成不同格式的音频文件。"

    formats = ["wav", "mp3", "pcm"]

    for format_name in formats:
        print(
            f"\n--- 格式: {format_name} ({client.available_formats[format_name]}) ---"
        )
        print("开始语音合成...")
        print(f"文本: {text}")
        print(f"音色: zhichu ({client.available_voices.get('zhichu', '未知音色')})")
        print(f"格式: {format_name} ({client.available_formats[format_name]})")
        print(f"采样率: 22050Hz")
        print(f"音量: 1.0")
        print(f"语速: 1.0")
        print(f"音调: 1.0")

        try:
            result = client.synthesize_speech(
                text=text, voice="zhichu", format=format_name, sample_rate=22050
            )

            if result.get('success'):
                audio_data = result['audio_data']
                saved_path = client.save_audio(
                    audio_data, f"format_comparison_{format_name}", format_name
                )

                # 计算文件大小
                file_size_kb = len(audio_data) / 1024
                print(f"✅ 格式 {format_name} 合成成功")
                print(f"音频文件大小: {len(audio_data)} 字节")
                print(f"文件大小: {file_size_kb:.2f} KB")
            else:
                print(f"❌ 格式 {format_name} 合成失败: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"❌ 格式 {format_name} 合成失败: {e}")


def show_available_options():
    """
    显示所有可用的选项
    """
    print("\n=== 可用选项 ===")

    client = BailianTTSGeneration()

    print("\n支持的音色:")
    voices = client.get_voice_list()
    for voice_id, description in voices.items():
        print(f"  {voice_id}: {description}")

    print("\n支持的音频格式:")
    formats = client.get_format_list()
    for format_id, description in formats.items():
        print(f"  {format_id}: {description}")

    print("\n支持的采样率:")
    sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
    for rate in sample_rates:
        print(f"  {rate}Hz")

    print("\n参数范围:")
    print("  音量 (volume): 0.1 - 2.0")
    print("  语速 (speech_rate): 0.5 - 2.0")
    print("  音调 (pitch_rate): 0.5 - 2.0")
    print("  文本长度: 最大500字符")


if __name__ == "__main__":
    """
    主函数：运行所有示例

    注意事项:
    1. 确保已设置DASHSCOPE_API_KEY环境变量
    2. 语音合成速度较快，通常几秒内完成
    3. 注意API调用频率限制和计费
    4. 生成的音频文件将保存在当前目录
    5. 文本长度限制为500字符
    """
    print("百炼平台Qwen-TTS语音合成API示例")
    print("=" * 50)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("设置方法: export DASHSCOPE_API_KEY=your_api_key")
        exit(1)

    try:
        # 显示可用选项
        show_available_options()

        # 运行示例
        print("\n开始运行语音合成示例...")

        example_basic_synthesis()  # 基础语音合成
        example_voice_comparison()  # 不同音色对比
        example_speed_control()  # 语速控制
        example_volume_pitch_control()  # 音量和音调控制
        example_long_text_synthesis()  # 长文本合成
        example_batch_synthesis()  # 批量语音合成
        example_format_comparison()  # 音频格式对比

        print("\n=== 所有示例执行完成 ===")
        print("\n生成的音频文件:")
        print("- basic_synthesis_example.wav")
        print("- voice_comparison_*.mp3")
        print("- speed_control_*.wav")
        print("- volume_pitch_*.wav")
        print("- long_text_synthesis.mp3")
        print("- batch_*.wav")
        print("- format_comparison_*")

        print("\n使用说明:")
        print("1. 所有音频文件已保存在当前目录")
        print("2. 可以使用音频播放器播放生成的文件")
        print("3. 注意API计费，按字符数收费")
        print("4. 文本长度限制为500字符")
        print("5. 支持多种音色、格式和参数调节")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
