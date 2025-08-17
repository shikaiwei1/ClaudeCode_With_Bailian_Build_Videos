#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼平台通义万相视频编辑统一模型API示例
API文档: https://help.aliyun.com/zh/model-studio/wanx-vace-api-reference

本示例展示如何使用百炼平台的通义万相视频编辑统一模型API
支持多图参考、视频重绘、局部编辑、视频延展、视频画面扩展等功能

Author: John Chen
"""

import os
import time
import json
import requests
from typing import Optional, Dict, Any, List


class BailianVideoGeneration:
    """
    百炼平台通义万相视频编辑统一模型API客户端

    支持的功能:
    - image_reference: 多图参考生成视频
    - video_repainting: 视频重绘
    - video_edit: 局部编辑
    - video_extension: 视频延展
    - video_expansion: 视频画面扩展

    支持的模型:
    - wanx2.1-vace-plus: 通义万相视频编辑统一模型
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
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_url = "https://dashscope.aliyuncs.com/api/v1/tasks"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",  # 异步处理必须设置
        }

    def create_video_task(
        self, model: str, function: str, prompt: str, **kwargs
    ) -> str:
        """
        创建视频生成任务

        Args:
            model: 模型名称，如 'wanx2.1-vace-plus'
            function: 功能类型 ('image_reference', 'video_repainting', 'video_edit', 'video_extension', 'video_expansion')
            prompt: 视频描述提示词
            **kwargs: 其他参数，根据不同功能类型传入不同参数

        Returns:
            task_id: 任务ID，用于后续查询结果
        """
        # 构建基础请求数据
        data = {
            "model": model,
            "input": {"function": function, "prompt": prompt},
            "parameters": {},
        }

        # 根据功能类型添加特定参数
        if function == "image_reference":
            # 多图参考功能
            if "ref_images_url" in kwargs:
                data["input"]["ref_images_url"] = kwargs["ref_images_url"]
            if "obj_or_bg" in kwargs:
                data["parameters"]["obj_or_bg"] = kwargs["obj_or_bg"]
            if "size" in kwargs:
                data["parameters"]["size"] = kwargs["size"]

        elif function == "video_repainting":
            # 视频重绘功能
            if "video_url" in kwargs:
                data["input"]["video_url"] = kwargs["video_url"]
            if "control_condition" in kwargs:
                data["parameters"]["control_condition"] = kwargs["control_condition"]

        elif function == "video_edit":
            # 局部编辑功能
            if "video_url" in kwargs:
                data["input"]["video_url"] = kwargs["video_url"]
            if "mask_url" in kwargs:
                data["input"]["mask_url"] = kwargs["mask_url"]

        elif function == "video_extension":
            # 视频延展功能
            if "first_frame_image" in kwargs:
                data["input"]["first_frame_image"] = kwargs["first_frame_image"]
            if "duration" in kwargs:
                data["parameters"]["duration"] = kwargs["duration"]

        elif function == "video_expansion":
            # 视频画面扩展功能
            if "video_url" in kwargs:
                data["input"]["video_url"] = kwargs["video_url"]
            if "expansion_ratio" in kwargs:
                data["parameters"]["expansion_ratio"] = kwargs["expansion_ratio"]

        print(f"创建视频生成任务...")
        print(f"模型: {model}")
        print(f"功能: {function}")
        print(f"提示词: {prompt}")
        print(f"参数: {json.dumps(kwargs, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()

            result = response.json()
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                print(f"任务创建成功，任务ID: {task_id}")
                print(f"注意: 视频生成耗时较长（约5-10分钟），请耐心等待")
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
            任务结果字典，包含状态和生成的视频URL
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
        self, task_id: str, max_wait_time: int = 900, check_interval: int = 30
    ) -> Dict[str, Any]:
        """
        等待任务完成并返回结果

        Args:
            task_id: 任务ID
            max_wait_time: 最大等待时间（秒），默认900秒（15分钟）
            check_interval: 检查间隔（秒），默认30秒

        Returns:
            完成的任务结果
        """
        print(f"等待任务完成，任务ID: {task_id}")
        print(f"视频生成通常需要5-10分钟，请耐心等待...")

        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            result = self.query_task_result(task_id)

            if "output" in result:
                task_status = result["output"].get("task_status", "UNKNOWN")
                elapsed_time = int(time.time() - start_time)
                print(f"任务状态: {task_status} (已等待 {elapsed_time} 秒)")

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

    def generate_video(
        self, model: str, function: str, prompt: str, max_wait_time: int = 900, **kwargs
    ) -> Dict[str, Any]:
        """
        一键生成视频（创建任务 + 等待完成）

        Args:
            model: 模型名称
            function: 功能类型
            prompt: 视频描述
            max_wait_time: 最大等待时间
            **kwargs: 其他参数

        Returns:
            完成的任务结果，包含生成的视频URL
        """
        # 创建任务
        task_id = self.create_video_task(model, function, prompt, **kwargs)

        # 等待完成
        result = self.wait_for_completion(task_id, max_wait_time)

        return result


def example_image_reference():
    """
    示例1: 多图参考生成视频
    使用多张参考图片生成视频，可以指定主体和背景
    """
    print("\n=== 示例1: 多图参考生成视频 ===")

    client = BailianVideoGeneration()

    # 参考图片URL（示例使用官方提供的测试图片）
    ref_images = [
        "http://wanx.alicdn.com/material/20250318/image_reference_2_5_16.png",  # 参考主体
        "http://wanx.alicdn.com/material/20250318/image_reference_1_5_16.png",  # 参考背景
    ]

    prompt = "视频中，一位女孩自晨雾缭绕的古老森林深处款款走出，她步伐轻盈，镜头捕捉她每一个灵动瞬间。当女孩站定，环顾四周葱郁林木时，她脸上绽放出惊喜与喜悦交织的笑容。这一幕，定格在了光影交错的瞬间，记录下女孩与大自然的美妙邂逅。"

    try:
        result = client.generate_video(
            model="wanx2.1-vace-plus",
            function="image_reference",
            prompt=prompt,
            ref_images_url=ref_images,
            obj_or_bg=["obj", "bg"],  # 指定第一张图为主体，第二张图为背景
            size="1280*720",  # 视频尺寸
        )

        # 解析结果
        if "output" in result and "video_url" in result["output"]:
            video_url = result["output"]["video_url"]
            print(f"\n生成成功！")
            print(f"视频URL: {video_url}")
            print(f"有效期: 24小时")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"生成失败: {e}")


def example_video_repainting():
    """
    示例2: 视频重绘
    对现有视频进行风格重绘，保持原视频的结构和动作
    """
    print("\n=== 示例2: 视频重绘 ===")

    client = BailianVideoGeneration()

    # 输入视频URL（示例使用官方提供的测试视频）
    video_url = "http://wanx.alicdn.com/material/20250318/video_repainting_1.mp4"

    prompt = "视频展示了一辆黑色的蒸汽朋克风格汽车，绅士驾驶着，车辆装饰着齿轮和铜管。背景是蒸汽驱动的糖果工厂和复古元素，画面复古与趣味。"

    try:
        result = client.generate_video(
            model="wanx2.1-vace-plus",
            function="video_repainting",
            prompt=prompt,
            video_url=video_url,
            control_condition="depth",  # 控制条件：深度信息
        )

        # 解析结果
        if "output" in result and "video_url" in result["output"]:
            video_url = result["output"]["video_url"]
            print(f"\n重绘成功！")
            print(f"视频URL: {video_url}")
            print(f"有效期: 24小时")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"重绘失败: {e}")


def example_video_edit():
    """
    示例3: 局部编辑
    对视频的特定区域进行编辑，通过掩码图像指定编辑区域
    """
    print("\n=== 示例3: 局部编辑 ===")

    client = BailianVideoGeneration()

    # 输入视频和掩码图像URL（示例URL）
    video_url = "http://wanx.alicdn.com/material/20250318/video_edit_input.mp4"
    mask_url = "http://wanx.alicdn.com/material/20250318/video_edit_mask.png"  # 白色区域表示编辑区域

    prompt = "视频展示了一家巴黎风情的法式咖啡馆，一只穿着西装的狮子优雅地品着咖啡。它一手端着咖啡杯，轻轻啜饮，神情惬意。咖啡馆装饰雅致，柔和的色调与温暖灯光映照着狮子所在的区域。"

    try:
        result = client.generate_video(
            model="wanx2.1-vace-plus",
            function="video_edit",
            prompt=prompt,
            video_url=video_url,
            mask_url=mask_url,  # 掩码图像，白色区域将被编辑
        )

        # 解析结果
        if "output" in result and "video_url" in result["output"]:
            video_url = result["output"]["video_url"]
            print(f"\n编辑成功！")
            print(f"视频URL: {video_url}")
            print(f"有效期: 24小时")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"编辑失败: {e}")


def example_video_extension():
    """
    示例4: 视频延展
    从一张图片或短视频片段延展生成更长的视频
    """
    print("\n=== 示例4: 视频延展 ===")

    client = BailianVideoGeneration()

    # 首帧图像URL（可以是图片或短视频的第一帧）
    first_frame_url = (
        "http://wanx.alicdn.com/material/20250318/video_extension_first_frame.jpg"
    )

    prompt = "一只戴着墨镜的狗在街道上滑滑板，3D卡通风格。狗狗熟练地控制着滑板，在街道上自由穿行，展现出酷炫的滑板技巧。"

    try:
        result = client.generate_video(
            model="wanx2.1-vace-plus",
            function="video_extension",
            prompt=prompt,
            first_frame_image=first_frame_url,
            duration=5,  # 延展后的视频时长（秒）
        )

        # 解析结果
        if "output" in result and "video_url" in result["output"]:
            video_url = result["output"]["video_url"]
            print(f"\n延展成功！")
            print(f"视频URL: {video_url}")
            print(f"从1秒延展到5秒")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"延展失败: {e}")


def example_video_expansion():
    """
    示例5: 视频画面扩展
    扩展视频的画面范围，增加更多的背景内容
    """
    print("\n=== 示例5: 视频画面扩展 ===")

    client = BailianVideoGeneration()

    # 输入视频URL
    video_url = "http://wanx.alicdn.com/material/20250318/video_expansion_input.mp4"

    prompt = "一位优雅的女士正在激情演奏小提琴，她身后是一支完整的交响乐团。音乐厅宏伟壮观，观众席座无虚席，灯光聚焦在演奏者身上，营造出庄严而优美的音乐氛围。"

    try:
        result = client.generate_video(
            model="wanx2.1-vace-plus",
            function="video_expansion",
            prompt=prompt,
            video_url=video_url,
            expansion_ratio=1.5,  # 扩展比例，1.5表示画面扩展50%
        )

        # 解析结果
        if "output" in result and "video_url" in result["output"]:
            video_url = result["output"]["video_url"]
            print(f"\n扩展成功！")
            print(f"视频URL: {video_url}")
            print(f"画面扩展比例: 1.5倍")
        else:
            print(f"结果解析失败: {result}")

    except Exception as e:
        print(f"扩展失败: {e}")


def example_batch_processing():
    """
    示例6: 批量处理
    演示如何批量创建多个视频生成任务
    """
    print("\n=== 示例6: 批量处理 ===")

    client = BailianVideoGeneration()

    # 批量任务配置
    tasks = [
        {
            "name": "科幻场景1",
            "function": "image_reference",
            "prompt": "未来科技城市，飞行汽车穿梭，霓虹灯闪烁",
            "ref_images_url": ["http://example.com/sci-fi-1.jpg"],
            "size": "1280*720",
        },
        {
            "name": "自然风景1",
            "function": "image_reference",
            "prompt": "宁静的湖泊，山峦倒影，日落时分",
            "ref_images_url": ["http://example.com/nature-1.jpg"],
            "size": "1280*720",
        },
    ]

    task_ids = []

    # 批量创建任务
    print("批量创建视频生成任务...")
    for i, task_config in enumerate(tasks):
        print(f"\n--- 创建任务 {i+1}: {task_config['name']} ---")

        try:
            task_id = client.create_video_task(
                model="wanx2.1-vace-plus",
                function=task_config["function"],
                prompt=task_config["prompt"],
                **{
                    k: v
                    for k, v in task_config.items()
                    if k not in ["name", "function", "prompt"]
                },
            )
            task_ids.append({"id": task_id, "name": task_config["name"]})

        except Exception as e:
            print(f"任务 {task_config['name']} 创建失败: {e}")

    # 批量等待完成
    print(f"\n批量等待 {len(task_ids)} 个任务完成...")
    for task_info in task_ids:
        print(f"\n--- 等待任务: {task_info['name']} ---")

        try:
            result = client.wait_for_completion(task_info["id"])

            if "output" in result and "results" in result["output"]:
                videos = result["output"]["results"]
                print(f"任务 {task_info['name']} 完成: {videos[0]['url']}")
            else:
                print(f"任务 {task_info['name']} 结果解析失败")

        except Exception as e:
            print(f"任务 {task_info['name']} 失败: {e}")


if __name__ == "__main__":
    """
    主函数：运行所有示例

    注意事项:
    1. 确保已设置DASHSCOPE_API_KEY环境变量
    2. 视频生成耗时较长（约5-10分钟），请耐心等待
    3. 生成的视频URL有效期为24小时
    4. 注意API调用频率限制和计费（0.70元/秒）
    5. 建议逐个运行示例，避免同时处理过多任务
    """
    print("百炼平台通义万相视频编辑统一模型API示例")
    print("=" * 60)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("设置方法: export DASHSCOPE_API_KEY=your_api_key")
        exit(1)

    try:
        # 运行示例（建议逐个运行，避免同时处理过多任务）
        print("\n注意: 视频生成耗时较长，建议逐个运行示例")
        print("可以注释掉不需要的示例函数")

        example_image_reference()  # 多图参考生成视频
        # example_video_repainting()     # 视频重绘
        # example_video_edit()           # 局部编辑
        # example_video_extension()      # 视频延展
        # example_video_expansion()      # 视频画面扩展
        # example_batch_processing()     # 批量处理

        print("\n=== 示例代码准备完成 ===")
        print("\n使用说明:")
        print("1. 取消注释需要运行的示例函数")
        print("2. 替换示例中的图片/视频URL为实际可用的URL")
        print("3. 视频生成需要5-10分钟，请耐心等待")
        print("4. 生成的视频URL有效期为24小时")
        print("5. 注意API计费：0.70元/秒")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
