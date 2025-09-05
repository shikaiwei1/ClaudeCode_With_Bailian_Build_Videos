#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼平台Qwen-Image图片合成API示例
API文档: https://help.aliyun.com/zh/model-studio/qwen-image-api

本示例展示如何使用百炼平台的Qwen-Image图片合成API
支持文生图功能，使用DashScope SDK调用方式

Author: John Chen
"""

import os
import time
import json
import requests
import base64
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation


class BailianImageGeneration:
    """
    百炼平台Qwen-Image图片合成API客户端

    支持的功能:
    - 文生图（使用DashScope SDK）

    支持的模型:
    - qwen-image: Qwen图片合成模型
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

        # 设置DashScope API Key
        dashscope.api_key = self.api_key

        # 支持的图片尺寸
        self.available_sizes = {
            "1664*928": "16:9 横屏",
            "1472*1140": "4:3 横屏", 
            "1328*1328": "1:1 正方形（默认）",
            "1140*1472": "3:4 竖屏",
            "928*1664": "9:16 竖屏"
        }

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        size: str = "1328*1328",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        n: int = 1
    ) -> Dict[str, Any]:
        """
        生成图片

        Args:
            prompt: 正向提示词，描述期望生成的图像内容（最大800字符）
            negative_prompt: 反向提示词，描述不希望出现的内容（最大500字符）
            size: 图片尺寸，支持"1664*928", "1472*1140", "1328*1328", "1140*1472", "928*1664"
            prompt_extend: 是否开启prompt智能改写，默认True
            watermark: 是否添加水印，默认False
            seed: 随机数种子，用于生成可重现的结果
            n: 生成图片数量，当前仅支持1

        Returns:
            生成结果，包含图片URL等信息
        """
        # 验证参数
        if len(prompt) > 800:
            raise ValueError("提示词长度不能超过800字符")
        
        if len(negative_prompt) > 500:
            raise ValueError("反向提示词长度不能超过500字符")
        
        if size not in self.available_sizes:
            raise ValueError(
                f"不支持的尺寸: {size}，支持的尺寸: {list(self.available_sizes.keys())}"
            )
        
        if n != 1:
            raise ValueError("当前仅支持生成1张图像")

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]

        # 构建参数
        parameters = {
            "size": size,
            "n": n,
            "prompt_extend": prompt_extend,
            "watermark": watermark
        }
        
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt
            
        if seed is not None:
            parameters["seed"] = seed

        print(f"开始图片生成...")
        print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        if negative_prompt:
            print(f"反向提示词: {negative_prompt[:50]}{'...' if len(negative_prompt) > 50 else ''}")
        print(f"尺寸: {size} ({self.available_sizes[size]})")
        print(f"智能改写: {prompt_extend}")
        print(f"水印: {watermark}")
        if seed is not None:
            print(f"随机种子: {seed}")

        try:
            response = MultiModalConversation.call(
                model="qwen-image",
                messages=messages,
                parameters=parameters
            )
            
            if response.status_code != HTTPStatus.OK:
                raise ValueError(
                    f"API调用失败: {response.status_code}, {response.message}"
                )
            
            # 解析响应
            output = response.output
            if not output or 'choices' not in output or not output['choices']:
                raise ValueError("API响应格式错误: 缺少choices字段")
            
            choice = output['choices'][0]
            if 'message' not in choice or 'content' not in choice['message']:
                raise ValueError("API响应格式错误: 缺少message.content字段")
            
            content = choice['message']['content']
            if not content or not isinstance(content, list):
                raise ValueError("API响应格式错误: content字段格式不正确")
            
            # 查找图片URL
            image_url = None
            for item in content:
                if isinstance(item, dict) and 'image' in item:
                    image_url = item['image']
                    break
            
            if not image_url:
                raise ValueError("API响应中没有找到图片URL")
            
            print(f"图片生成成功!")
            print(f"图片URL: {image_url}")
            
            return {
                'success': True,
                'image_url': image_url,
                'request_id': response.request_id,
                'usage': response.usage,
                'parameters_used': parameters
            }
            
        except Exception as e:
            print(f"图片生成失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'request_id': getattr(response, 'request_id', '') if 'response' in locals() else ''
            }

    def save_image(self, image_url: str, filename: str) -> bool:
        """
        保存图片到本地文件

        Args:
            image_url: 图片URL
            filename: 保存的文件名

        Returns:
            是否保存成功
        """
        try:
            print(f"正在下载图片: {image_url}")
            
            # 下载图片
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 保存到文件
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"图片已保存到: {filename}")
            print(f"文件大小: {len(response.content)} 字节")
            
            return True
            
        except Exception as e:
            print(f"保存图片失败: {str(e)}")
            return False

    def get_size_list(self) -> Dict[str, str]:
        """
        获取支持的图片尺寸列表

        Returns:
            尺寸字典，键为尺寸值，值为描述
        """
        return self.available_sizes.copy()


def example_basic_generation():
    """
    基础图片生成示例
    """
    print("\n=== 基础图片生成示例 ===")
    
    # 初始化客户端
    client = BailianImageGeneration()
    
    # 生成图片
    prompt = "一只可爱的橙色小猫，坐在绿色的草地上，阳光明媚，卡通风格"
    result = client.generate_image(
        prompt=prompt,
        size="1328*1328",  # 正方形
        prompt_extend=True,
        watermark=False
    )
    
    if result['success']:
        # 保存图片
        filename = "example_cat.jpg"
        client.save_image(result['image_url'], filename)
        print(f"\n生成成功! 图片已保存为: {filename}")
    else:
        print(f"\n生成失败: {result['error']}")


def example_character_generation():
    """
    角色图片生成示例（适用于视频制作）
    """
    print("\n=== 角色图片生成示例 ===")
    
    # 初始化客户端
    client = BailianImageGeneration()
    
    # 角色描述
    prompt = "一个年轻的女性角色，长黑发，穿着蓝色连衣裙，微笑着，全身像，正面站立，白色背景"
    negative_prompt = "背景复杂，侧面，背面，多个人物，模糊"
    
    result = client.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        size="1140*1472",  # 3:4 竖屏，适合角色
        prompt_extend=True,
        watermark=False,
        seed=12345  # 固定种子确保可重现
    )
    
    if result['success']:
        # 保存图片
        filename = "character_female.jpg"
        client.save_image(result['image_url'], filename)
        print(f"\n角色生成成功! 图片已保存为: {filename}")
        print(f"使用的种子: 12345")
    else:
        print(f"\n角色生成失败: {result['error']}")


def example_background_generation():
    """
    背景图片生成示例
    """
    print("\n=== 背景图片生成示例 ===")
    
    # 初始化客户端
    client = BailianImageGeneration()
    
    # 背景描述
    prompt = "美丽的森林场景，阳光透过树叶洒下，小径蜿蜒，鸟儿飞翔，自然风光，高清摄影风格"
    negative_prompt = "人物，建筑物，车辆，现代元素"
    
    result = client.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        size="1664*928",  # 16:9 横屏，适合背景
        prompt_extend=True,
        watermark=False
    )
    
    if result['success']:
        # 保存图片
        filename = "background_forest.jpg"
        client.save_image(result['image_url'], filename)
        print(f"\n背景生成成功! 图片已保存为: {filename}")
    else:
        print(f"\n背景生成失败: {result['error']}")


def example_size_comparison():
    """
    不同尺寸对比示例
    """
    print("\n=== 不同尺寸对比示例 ===")
    
    # 初始化客户端
    client = BailianImageGeneration()
    
    # 显示支持的尺寸
    sizes = client.get_size_list()
    print("\n支持的图片尺寸:")
    for size, desc in sizes.items():
        print(f"  {size}: {desc}")
    
    # 使用相同提示词生成不同尺寸的图片
    prompt = "一朵美丽的红玫瑰，露珠点缀，特写镜头"
    
    for size, desc in list(sizes.items())[:2]:  # 只生成前两个尺寸作为示例
        print(f"\n正在生成 {size} ({desc}) 尺寸的图片...")
        
        result = client.generate_image(
            prompt=prompt,
            size=size,
            prompt_extend=True,
            watermark=False
        )
        
        if result['success']:
            filename = f"rose_{size.replace('*', 'x')}.jpg"
            client.save_image(result['image_url'], filename)
            print(f"已保存为: {filename}")
        else:
            print(f"生成失败: {result['error']}")
        
        # 避免请求过于频繁
        time.sleep(2)


def example_batch_generation():
    """
    批量生成示例（适用于视频制作中的多个角色）
    """
    print("\n=== 批量角色生成示例 ===")
    
    # 初始化客户端
    client = BailianImageGeneration()
    
    # 定义多个角色
    characters = [
        {
            "name": "主角_小明",
            "prompt": "一个10岁的男孩，短黑发，穿着蓝色T恤和牛仔裤，微笑，全身像，正面站立，白色背景",
            "negative_prompt": "成人，女性，复杂背景，侧面，背面"
        },
        {
            "name": "配角_老师",
            "prompt": "一位中年女性老师，戴眼镜，穿着白色衬衫和黑色裙子，温和的笑容，全身像，正面站立，白色背景",
            "negative_prompt": "学生，年轻，复杂背景，侧面，背面"
        },
        {
            "name": "宠物_小狗",
            "prompt": "一只可爱的金毛犬，坐着，舌头伸出，友好的表情，全身像，白色背景",
            "negative_prompt": "猫，其他动物，复杂背景，多只动物"
        }
    ]
    
    print(f"准备生成 {len(characters)} 个角色的图片...")
    
    for i, char in enumerate(characters, 1):
        print(f"\n[{i}/{len(characters)}] 正在生成角色: {char['name']}")
        
        result = client.generate_image(
            prompt=char['prompt'],
            negative_prompt=char['negative_prompt'],
            size="1140*1472",  # 3:4 竖屏，适合角色
            prompt_extend=True,
            watermark=False
        )
        
        if result['success']:
            filename = f"{char['name']}.jpg"
            client.save_image(result['image_url'], filename)
            print(f"✓ {char['name']} 生成成功，已保存为: {filename}")
        else:
            print(f"✗ {char['name']} 生成失败: {result['error']}")
        
        # 避免请求过于频繁
        if i < len(characters):
            print("等待3秒后继续...")
            time.sleep(3)
    
    print("\n批量生成完成!")


if __name__ == "__main__":
    print("百炼平台Qwen-Image图片生成API示例")
    print("=====================================")
    
    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("设置方法:")
        print("  export DASHSCOPE_API_KEY='your-api-key-here'")
        exit(1)
    
    print(f"\nAPI密钥已配置: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # 运行示例
        print("\n选择要运行的示例:")
        print("1. 基础图片生成")
        print("2. 角色图片生成")
        print("3. 背景图片生成")
        print("4. 尺寸对比")
        print("5. 批量角色生成")
        print("6. 运行所有示例")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == "1":
            example_basic_generation()
        elif choice == "2":
            example_character_generation()
        elif choice == "3":
            example_background_generation()
        elif choice == "4":
            example_size_comparison()
        elif choice == "5":
            example_batch_generation()
        elif choice == "6":
            example_basic_generation()
            example_character_generation()
            example_background_generation()
            example_size_comparison()
            example_batch_generation()
        else:
            print("无效选择，运行基础示例...")
            example_basic_generation()
            
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n执行出错: {str(e)}")
    
    print("\n程序结束")


# API使用说明
"""
使用说明:

1. 环境准备:
   - 安装依赖: pip install dashscope requests
   - 设置API密钥: export DASHSCOPE_API_KEY='your-api-key-here'

2. 基本用法:
   from bailian_multimodal_generation_qwen_image import BailianImageGeneration
   
   client = BailianImageGeneration()
   result = client.generate_image(
       prompt="一只可爱的小猫",
       size="1328*1328"
   )
   
   if result['success']:
       client.save_image(result['image_url'], "cat.jpg")

3. 参数说明:
   - prompt: 正向提示词，描述期望生成的图像内容（最大800字符）
   - negative_prompt: 反向提示词，描述不希望出现的内容（最大500字符）
   - size: 图片尺寸，支持多种比例
   - prompt_extend: 是否开启prompt智能改写
   - watermark: 是否添加水印
   - seed: 随机数种子，用于生成可重现的结果
   - n: 生成图片数量（当前仅支持1）

4. 支持的尺寸:
   - "1664*928": 16:9 横屏
   - "1472*1140": 4:3 横屏
   - "1328*1328": 1:1 正方形（默认）
   - "1140*1472": 3:4 竖屏
   - "928*1664": 9:16 竖屏

5. 视频制作建议:
   - 角色图片: 使用3:4竖屏尺寸，白色背景，正面全身像
   - 背景图片: 使用16:9横屏尺寸，无人物元素
   - 使用negative_prompt避免不需要的元素
   - 设置固定seed确保角色一致性

6. 注意事项:
   - 提示词要具体详细，有助于生成更准确的图像
   - 合理使用negative_prompt排除不需要的元素
   - 生成的图片URL有时效性，建议及时下载保存
   - 避免频繁请求，建议请求间隔2-3秒
"""
