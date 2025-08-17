#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼平台文生图V2版本API示例
API文档: https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference

本示例展示如何使用百炼平台的文生图V2版本API生成图像
支持多种模型和参数配置，包括正向提示词和反向提示词

Author: John Chen
"""

import os
import time
import json
import requests
from typing import Optional, Dict, Any


class BailianTextToImageV2:
    """
    百炼平台文生图V2版本API客户端

    支持的模型:
    - wan2.2-t2i-flash: 万相2.2极速版（推荐）
    - wan2.2-t2i-plus: 万相2.2专业版（推荐）
    - wanx2.1-t2i-turbo: 万相2.1极速版
    - wanx2.1-t2i-plus: 万相2.1专业版
    - wanx2.0-t2i-turbo: 万相2.0极速版
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
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        self.query_url = "https://dashscope.aliyuncs.com/api/v1/tasks"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",  # 异步处理必须设置
        }

    def create_image_task(
        self,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
    ) -> str:
        """
        创建文生图任务

        Args:
            model: 模型名称，如 'wan2.2-t2i-flash'
            prompt: 正向提示词，描述期望生成的图像内容（最大800字符）
            negative_prompt: 反向提示词，描述不希望出现的内容（最大500字符，可选）
            size: 图像分辨率，默认"1024*1024"。支持范围：宽高[512,1440]像素，最高200万像素
            n: 生成图片数量，取值范围1-4

        Returns:
            task_id: 任务ID，用于后续查询结果
        """
        # 构建请求数据
        data = {
            "model": model,
            "input": {"prompt": prompt},
            "parameters": {"size": size, "n": n},
        }

        # 如果提供了反向提示词，添加到input中
        if negative_prompt:
            data["input"]["negative_prompt"] = negative_prompt

        print(f"创建文生图任务...")
        print(f"模型: {model}")
        print(f"提示词: {prompt}")
        if negative_prompt:
            print(f"反向提示词: {negative_prompt}")
        print(f"图像尺寸: {size}")
        print(f"生成数量: {n}")

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()

            result = response.json()
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                print(f"任务创建成功，任务ID: {task_id}")
                return task_id
            else:
                raise Exception(f"任务创建失败: {result}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"请求失败: {e}")

    def query_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        查询任务结果

        Args:
            task_id: 任务ID

        Returns:
            任务结果字典，包含状态和生成的图像URL
        """
        query_url = f"{self.query_url}/{task_id}"

        try:
            response = requests.get(query_url, headers=self.headers)
            response.raise_for_status()

            result = response.json()
            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"查询任务失败: {e}")

    def wait_for_completion(
        self, task_id: str, max_wait_time: int = 300, check_interval: int = 5
    ) -> Dict[str, Any]:
        """
        等待任务完成并返回结果

        Args:
            task_id: 任务ID
            max_wait_time: 最大等待时间（秒），默认300秒
            check_interval: 检查间隔（秒），默认5秒

        Returns:
            完成的任务结果
        """
        print(f"等待任务完成，任务ID: {task_id}")

        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            result = self.query_task_result(task_id)

            if "output" in result:
                task_status = result["output"].get("task_status", "UNKNOWN")
                print(f"任务状态: {task_status}")

                if task_status == "SUCCEEDED":
                    print("任务完成成功！")
                    return result
                elif task_status == "FAILED":
                    error_msg = result["output"].get("message", "未知错误")
                    raise Exception(f"任务失败: {error_msg}")
                elif task_status in ["PENDING", "RUNNING"]:
                    print(f"任务进行中，{check_interval}秒后重新检查...")
                    time.sleep(check_interval)
                else:
                    print(f"未知状态: {task_status}，继续等待...")
                    time.sleep(check_interval)
            else:
                print("获取任务状态失败，继续等待...")
                time.sleep(check_interval)

        raise Exception(f"任务超时，等待时间超过{max_wait_time}秒")

    def generate_image(
        self,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        max_wait_time: int = 300,
    ) -> Dict[str, Any]:
        """
        一键生成图像（创建任务 + 等待完成）

        Args:
            model: 模型名称
            prompt: 正向提示词
            negative_prompt: 反向提示词（可选）
            size: 图像尺寸
            n: 生成数量
            max_wait_time: 最大等待时间

        Returns:
            完成的任务结果，包含生成的图像URL
        """
        # 创建任务
        task_id = self.create_image_task(model, prompt, negative_prompt, size, n)

        # 等待完成
        result = self.wait_for_completion(task_id, max_wait_time)

        return result


def example_basic_text_to_image():
    """
    示例1: 基础文生图功能
    根据提示词生成图像，使用最新的万相2.2极速版模型
    """
    print("\n=== 示例1: 基础文生图功能 ===")

    client = BailianTextToImageV2()

    # 使用万相2.2极速版模型（推荐）
    model = "wan2.2-t2i-flash"
    prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"

    try:
        result = client.generate_image(
            model=model,
            prompt=prompt,
            size="1024*1024",  # 标准正方形尺寸
            n=1,  # 生成1张图片
        )

        # 解析结果
        if "output" in result and "results" in result["output"]:
            images = result["output"]["results"]
            print(f"\n生成成功！共生成{len(images)}张图片:")
            for i, img in enumerate(images):
                print(f"图片{i+1}: {img['url']}")
                print(f"有效期: 24小时")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"生成失败: {e}")


def example_negative_prompt():
    """
    示例2: 使用反向提示词
    通过反向提示词控制不希望出现的内容
    """
    print("\n=== 示例2: 使用反向提示词 ===")

    client = BailianTextToImageV2()

    # 使用万相2.2极速版模型
    model = "wan2.2-t2i-flash"
    prompt = "雪地，白色小教堂，极光，冬日场景，柔和的光线。"
    negative_prompt = "人物"  # 避免出现人物

    try:
        result = client.generate_image(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            size="1024*1024",
            n=1,
        )

        # 解析结果
        if "output" in result and "results" in result["output"]:
            images = result["output"]["results"]
            print(f"\n生成成功！共生成{len(images)}张图片:")
            for i, img in enumerate(images):
                print(f"图片{i+1}: {img['url']}")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"生成失败: {e}")


def example_multiple_images():
    """
    示例3: 批量生成多张图片
    一次性生成多张不同的图片
    """
    print("\n=== 示例3: 批量生成多张图片 ===")

    client = BailianTextToImageV2()

    # 使用万相2.2专业版模型（生成细节更丰富）
    model = "wan2.2-t2i-plus"
    prompt = "一只坐着的橘黄色的猫，表情愉悦，活泼可爱，逼真准确"
    negative_prompt = "低分辨率、错误、最差质量、低质量、残缺、多余的手指、比例不良"

    try:
        result = client.generate_image(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            size="1024*1024",
            n=3,  # 生成3张图片
        )

        # 解析结果
        if "output" in result and "results" in result["output"]:
            images = result["output"]["results"]
            print(f"\n生成成功！共生成{len(images)}张图片:")
            for i, img in enumerate(images):
                print(f"图片{i+1}: {img['url']}")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"生成失败: {e}")


def example_custom_size():
    """
    示例4: 自定义图像尺寸
    生成不同比例的图像（横版、竖版等）
    """
    print("\n=== 示例4: 自定义图像尺寸 ===")

    client = BailianTextToImageV2()

    # 测试不同尺寸
    test_cases = [
        {
            "size": "1280*720",  # 16:9横版
            "prompt": "壮丽的山脉风景，日出时分，金色阳光洒在山峰上",
            "description": "16:9横版风景",
        },
        {
            "size": "720*1280",  # 9:16竖版
            "prompt": "高耸的摩天大楼，从下往上仰视角度，现代都市建筑",
            "description": "9:16竖版建筑",
        },
        {
            "size": "1440*1024",  # 自定义比例
            "prompt": "广阔的海滩全景，蓝天白云，海浪拍打沙滩",
            "description": "自定义比例海景",
        },
    ]

    model = "wan2.2-t2i-flash"

    for i, case in enumerate(test_cases):
        print(f"\n--- 测试{i+1}: {case['description']} ({case['size']}) ---")

        try:
            result = client.generate_image(
                model=model, prompt=case["prompt"], size=case["size"], n=1
            )

            # 解析结果
            if "output" in result and "results" in result["output"]:
                images = result["output"]["results"]
                print(f"生成成功: {images[0]['url']}")
            else:
                print(f"生成失败: {result}")

        except Exception as e:
            print(f"生成失败: {e}")


def example_different_models():
    """
    示例5: 不同模型对比
    使用不同版本的模型生成相同内容，对比效果
    """
    print("\n=== 示例5: 不同模型对比 ===")

    client = BailianTextToImageV2()

    # 测试不同模型
    models = [
        {"name": "wan2.2-t2i-flash", "desc": "万相2.2极速版（推荐）"},
        {"name": "wan2.2-t2i-plus", "desc": "万相2.2专业版（推荐）"},
        {"name": "wanx2.1-t2i-turbo", "desc": "万相2.1极速版"},
        {"name": "wanx2.1-t2i-plus", "desc": "万相2.1专业版"},
    ]

    prompt = "科幻风格的未来城市，霓虹灯闪烁，飞行汽车穿梭其中"

    for model_info in models:
        print(f"\n--- 使用模型: {model_info['desc']} ---")

        try:
            result = client.generate_image(
                model=model_info["name"], prompt=prompt, size="1024*1024", n=1
            )

            # 解析结果
            if "output" in result and "results" in result["output"]:
                images = result["output"]["results"]
                print(f"生成成功: {images[0]['url']}")
            else:
                print(f"生成失败: {result}")

        except Exception as e:
            print(f"生成失败: {e}")


if __name__ == "__main__":
    """
    主函数：运行所有示例

    注意事项:
    1. 确保已设置DASHSCOPE_API_KEY环境变量
    2. 图像生成需要一定时间，请耐心等待
    3. 生成的图像URL有效期为24小时
    4. 注意API调用频率限制和计费
    """
    print("百炼平台文生图V2版本API示例")
    print("=" * 50)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("设置方法: export DASHSCOPE_API_KEY=your_api_key")
        exit(1)

    try:
        # 运行所有示例
        example_basic_text_to_image()  # 基础文生图
        example_negative_prompt()  # 反向提示词
        example_multiple_images()  # 批量生成
        example_custom_size()  # 自定义尺寸
        example_different_models()  # 模型对比

        print("\n=== 所有示例运行完成 ===")
        print("\n提示:")
        print("- 生成的图像URL有效期为24小时")
        print("- 建议及时下载保存重要图片")
        print("- 更多参数说明请参考API文档")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
