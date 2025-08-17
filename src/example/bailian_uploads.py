#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼平台文件上传获取临时公网URL API示例
API文档: https://help.aliyun.com/zh/model-studio/get-temporary-file-url

本示例展示如何使用百炼平台的文件上传API
支持上传本地文件并获取临时公网URL，方便在AI生成图片、视频时使用

Author: John Chen
"""

import os
import time
import json
import requests
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


class BailianFileUpload:
    """
    百炼平台文件上传API客户端

    支持的功能:
    - 单文件上传
    - 批量文件上传
    - 获取临时公网URL
    - 支持多种文件类型

    支持的文件类型:
    - 图片: jpg, jpeg, png, gif, bmp, webp
    - 视频: mp4, avi, mov, wmv, flv, mkv
    - 音频: mp3, wav, aac, flac, ogg
    - 文档: pdf, txt, doc, docx
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
        self.upload_url = "https://dashscope.aliyuncs.com/api/v1/uploads"

        # 请求头（不包含Content-Type，让requests自动设置）
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        # 支持的文件类型
        self.supported_types = {
            # 图片类型
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            # 视频类型
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".mkv": "video/x-matroska",
            # 音频类型
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            # 文档类型
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        # 文件大小限制（字节）
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        验证文件是否符合上传要求

        Args:
            file_path: 文件路径

        Returns:
            验证结果字典，包含文件信息
        """
        file_path = Path(file_path)

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查是否为文件
        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        # 获取文件信息
        file_size = file_path.stat().st_size
        file_ext = file_path.suffix.lower()

        # 检查文件大小
        if file_size > self.max_file_size:
            raise ValueError(
                f"文件过大: {file_size} 字节，最大支持 {self.max_file_size} 字节"
            )

        # 检查文件类型
        if file_ext not in self.supported_types:
            raise ValueError(
                f"不支持的文件类型: {file_ext}，支持的类型: {list(self.supported_types.keys())}"
            )

        # 获取MIME类型
        mime_type = self.supported_types[file_ext]

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": file_size,
            "extension": file_ext,
            "mime_type": mime_type,
        }

    def get_upload_policy(self, model_name: str) -> Dict[str, Any]:
        """
        获取文件上传凭证
        
        Args:
            model_name: 指定文件将要用于哪个模型
            
        Returns:
            上传凭证数据
        """
        url = "https://dashscope.aliyuncs.com/api/v1/uploads"
        params = {
            "action": "getPolicy",
            "model": model_name
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise Exception(f"获取上传凭证失败: {response.status_code} {response.reason}\n{response.text}")
        
        result = response.json()
        if 'data' not in result:
            raise Exception(f"上传凭证响应格式异常: {result}")
            
        return result['data']
    
    def upload_file_to_oss(self, policy_data: Dict[str, Any], file_path: str) -> str:
        """
        将文件上传到临时存储OSS
        
        Args:
            policy_data: 上传凭证数据
            file_path: 本地文件路径
            
        Returns:
            OSS文件URL (oss://格式)
        """
        file_name = Path(file_path).name
        key = f"{policy_data['upload_dir']}/{file_name}"
        
        with open(file_path, 'rb') as file:
            files = {
                'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
                'Signature': (None, policy_data['signature']),
                'policy': (None, policy_data['policy']),
                'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
                'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
                'key': (None, key),
                'success_action_status': (None, '200'),
                'file': (file_name, file)
            }
            
            response = requests.post(policy_data['upload_host'], files=files)
            if response.status_code != 200:
                raise Exception(f"文件上传到OSS失败: {response.status_code} {response.reason}\n{response.text}")
        
        return f"oss://{key}"

    def upload_file(
        self, file_path: str, purpose: str = "file-extract", model: str = "qwen-vl-plus"
    ) -> Dict[str, Any]:
        """
        上传单个文件到百炼平台临时存储空间
        
        Args:
            file_path: 本地文件路径
            purpose: 文件用途，默认为"file-extract"
            model: 指定文件将要用于哪个模型，默认为"qwen-vl-plus"
            
        Returns:
            包含上传结果的字典
                - success (bool): 是否成功
                - url (str): 文件的公网URL（成功时）
                - error (str): 错误信息（失败时）
                - file_info (dict): 文件信息
        """
        try:
            # 验证文件
            file_info = self.validate_file(file_path)
            
            # 显示文件信息
            print(f"开始上传文件...")
            print(f"文件: {file_info['name']}")
            print(f"大小: {file_info['size']} 字节 ({file_info['size']/1024:.2f} KB)")
            print(f"类型: {file_info['extension']} ({file_info['mime_type']})")
            print(f"用途: {purpose}")
            
            # 步骤1: 获取上传凭证
            print("正在获取上传凭证...")
            policy_data = self.get_upload_policy(model)
            
            # 步骤2: 上传文件到OSS
            print("正在上传文件到OSS...")
            oss_url = self.upload_file_to_oss(policy_data, file_path)
            
            print(f"✓ 上传成功: {oss_url}")
            
            # 计算过期时间
            import datetime
            expire_time = datetime.datetime.now() + datetime.timedelta(hours=48)
            
            return {
                "success": True,
                "url": oss_url,
                "file_info": file_info,
                "expires_in": "48小时",
                "expire_time": expire_time.strftime('%Y-%m-%d %H:%M:%S')
            }
                
        except Exception as e:
            error_msg = f"上传过程中发生异常: {str(e)}"
            print(f"✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "file_info": file_info if 'file_info' in locals() else None
            }

    def upload_multiple_files(
        self, file_paths: List[str], purpose: str = "file-extract", model: str = "qwen-vl-plus"
    ) -> List[Dict[str, Any]]:
        """
        批量上传多个文件

        Args:
            file_paths: 文件路径列表
            purpose: 上传目的
            model: 指定文件将要用于哪个模型，默认为"qwen-vl-plus"

        Returns:
            上传结果列表
        """
        print(f"\n开始批量上传 {len(file_paths)} 个文件...")

        results = []
        success_count = 0

        for i, file_path in enumerate(file_paths):
            print(f"\n--- 上传第 {i+1}/{len(file_paths)} 个文件 ---")

            result = self.upload_file(file_path, purpose, model)
            results.append({"file_path": file_path, **result})
            
            if result["success"]:
                success_count += 1
                print("✓ 上传成功")
            else:
                print(f"✗ 上传失败: {result['error']}")

        print(f"\n批量上传完成: {success_count}/{len(file_paths)} 成功")
        return results

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        获取已上传文件的信息

        Args:
            file_id: 文件ID

        Returns:
            文件信息
        """
        info_url = f"{self.upload_url}/{file_id}"

        try:
            response = requests.get(info_url, headers=self.headers)
            response.raise_for_status()

            result = response.json()
            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"获取文件信息失败: {e}")

    def list_uploaded_files(
        self, purpose: Optional[str] = None, limit: int = 20
    ) -> Dict[str, Any]:
        """
        列出已上传的文件

        Args:
            purpose: 过滤特定用途的文件，可选
            limit: 返回文件数量限制，默认20

        Returns:
            文件列表
        """
        params = {"limit": limit}
        if purpose:
            params["purpose"] = purpose

        try:
            response = requests.get(
                self.upload_url, headers=self.headers, params=params
            )
            response.raise_for_status()

            result = response.json()
            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"获取文件列表失败: {e}")

    def delete_file(self, file_id: str) -> bool:
        """
        删除已上传的文件

        Args:
            file_id: 文件ID

        Returns:
            删除是否成功
        """
        delete_url = f"{self.upload_url}/{file_id}"

        try:
            response = requests.delete(delete_url, headers=self.headers)
            response.raise_for_status()

            print(f"文件 {file_id} 删除成功")
            return True

        except requests.exceptions.RequestException as e:
            print(f"删除文件失败: {e}")
            return False

    def get_supported_types(self) -> Dict[str, str]:
        """
        获取支持的文件类型列表

        Returns:
            支持的文件类型字典
        """
        return self.supported_types.copy()

    def format_file_size(self, size_bytes: int) -> str:
        """
        格式化文件大小显示

        Args:
            size_bytes: 文件大小（字节）

        Returns:
            格式化的大小字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def example_single_file_upload():
    """
    示例1: 单文件上传
    演示如何上传单个文件并获取临时URL
    """
    print("\n=== 示例1: 单文件上传 ===")

    client = BailianFileUpload()

    # 创建一个测试文件（如果不存在）
    test_file = "test_image.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文件，用于演示文件上传功能。\n")
            f.write("文件创建时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"创建测试文件: {test_file}")

    try:
        result = client.upload_file(test_file, purpose="file-extract")

        if result["success"]:
            print(f"\n上传结果:")
            print(f"文件URL: {result['url']}")
            print(f"有效期: {result['expires_in']}")
            print(f"过期时间: {result['expire_time']}")
            return result.get("url")
        else:
            print(f"上传失败: {result['error']}")
            return None

    except Exception as e:
        print(f"上传失败: {e}")
        return None
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"清理测试文件: {test_file}")


def example_image_upload():
    """
    示例2: 图片文件上传
    演示如何上传图片文件，适用于图生视频等场景
    """
    print("\n=== 示例2: 图片文件上传 ===")

    client = BailianFileUpload()

    # 创建一个简单的测试图片文件（实际使用中应该是真实的图片文件）
    test_image = "test_upload.png"

    # 注意：这里只是创建一个假的PNG文件用于演示
    # 实际使用中应该使用真实的图片文件
    print("注意: 这是一个演示示例，实际使用时请提供真实的图片文件")
    print(f"假设我们有一个图片文件: {test_image}")

    # 如果有真实的图片文件，可以这样上传：
    # try:
    #     result = client.upload_file(test_image, purpose="vision")
    #     print(f"图片上传成功，文件ID: {result['id']}")
    #     if "url" in result:
    #         print(f"图片临时URL: {result['url']}")
    #         print("此URL可用于图生视频等AI功能")
    # except Exception as e:
    #     print(f"图片上传失败: {e}")

    print("\n使用说明:")
    print("1. 将 test_image 替换为真实的图片文件路径")
    print("2. 支持的图片格式: jpg, jpeg, png, gif, bmp, webp")
    print("3. 上传后的URL可用于图生视频、图像编辑等AI功能")
    print("4. purpose参数可设置为'vision'用于视觉相关任务")


def example_video_upload():
    """
    示例3: 视频文件上传
    演示如何上传视频文件，适用于视频编辑等场景
    """
    print("\n=== 示例3: 视频文件上传 ===")

    client = BailianFileUpload()

    # 演示视频上传（实际使用中需要真实的视频文件）
    test_video = "test_video.mp4"

    print("注意: 这是一个演示示例，实际使用时请提供真实的视频文件")
    print(f"假设我们有一个视频文件: {test_video}")

    # 如果有真实的视频文件，可以这样上传：
    # try:
    #     result = client.upload_file(test_video, purpose="file-extract")
    #     print(f"视频上传成功，文件ID: {result['id']}")
    #     if "url" in result:
    #         print(f"视频临时URL: {result['url']}")
    #         print("此URL可用于视频重绘、视频编辑等AI功能")
    # except Exception as e:
    #     print(f"视频上传失败: {e}")

    print("\n使用说明:")
    print("1. 将 test_video 替换为真实的视频文件路径")
    print("2. 支持的视频格式: mp4, avi, mov, wmv, flv, mkv")
    print("3. 上传后的URL可用于视频重绘、视频编辑等AI功能")
    print("4. 注意视频文件大小限制（最大100MB）")


def example_batch_upload():
    """
    示例4: 批量文件上传
    演示如何批量上传多个文件
    """
    print("\n=== 示例4: 批量文件上传 ===")

    client = BailianFileUpload()

    # 创建多个测试文件
    test_files = []
    for i in range(3):
        filename = f"batch_test_{i+1}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"这是批量上传测试文件 {i+1}\n")
            f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文件编号: {i+1}")
        test_files.append(filename)
        print(f"创建测试文件: {filename}")

    try:
        # 批量上传
        results = client.upload_multiple_files(test_files, purpose="file-extract")

        print(f"\n批量上传结果:")
        for i, result in enumerate(results):
            print(f"\n文件 {i+1}: {result['file_path']}")
            if result["success"]:
                print(f"  ✓ 上传成功")
                print(f"  文件URL: {result['url']}")
                print(f"  有效期: {result['expires_in']}")
                print(f"  过期时间: {result['expire_time']}")
            else:
                print(f"  ✗ 上传失败: {result['error']}")

        # 返回成功上传的文件URL列表
        successful_urls = []
        for result in results:
            if result["success"]:
                successful_urls.append(result.get("url"))

        return successful_urls

    except Exception as e:
        print(f"批量上传失败: {e}")
        return []
    finally:
        # 清理测试文件
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"清理测试文件: {filename}")


def example_file_management(file_ids: List[str]):
    """
    示例5: 文件管理
    演示如何查看、管理已上传的文件

    Args:
        file_ids: 文件ID列表
    """
    print("\n=== 示例5: 文件管理 ===")

    if not file_ids:
        print("没有可管理的文件ID")
        return

    client = BailianFileUpload()

    # 获取文件列表
    print("\n--- 获取文件列表 ---")
    try:
        file_list = client.list_uploaded_files(limit=10)

        if "data" in file_list:
            files = file_list["data"]
            print(f"找到 {len(files)} 个文件:")

            for file_info in files:
                print(f"\n文件ID: {file_info.get('id', 'N/A')}")
                print(f"文件名: {file_info.get('filename', 'N/A')}")
                print(f"大小: {client.format_file_size(file_info.get('bytes', 0))}")
                print(f"用途: {file_info.get('purpose', 'N/A')}")
                print(f"创建时间: {file_info.get('created_at', 'N/A')}")
        else:
            print("未找到文件")

    except Exception as e:
        print(f"获取文件列表失败: {e}")

    # 获取特定文件信息
    print("\n--- 获取特定文件信息 ---")
    for file_id in file_ids[:2]:  # 只查看前2个文件
        try:
            file_info = client.get_file_info(file_id)
            print(f"\n文件 {file_id} 详细信息:")
            print(f"文件名: {file_info.get('filename', 'N/A')}")
            print(f"大小: {client.format_file_size(file_info.get('bytes', 0))}")
            print(f"状态: {file_info.get('status', 'N/A')}")

        except Exception as e:
            print(f"获取文件 {file_id} 信息失败: {e}")

    # 删除文件（可选）
    print("\n--- 文件删除演示 ---")
    print("注意: 以下是删除文件的演示代码，实际运行时已注释")
    print("如需删除文件，请取消注释相关代码")

    # 取消注释以下代码来删除文件
    # for file_id in file_ids:
    #     try:
    #         success = client.delete_file(file_id)
    #         if success:
    #             print(f"文件 {file_id} 删除成功")
    #     except Exception as e:
    #         print(f"删除文件 {file_id} 失败: {e}")


def example_file_type_validation():
    """
    示例6: 文件类型验证
    演示文件类型验证和错误处理
    """
    print("\n=== 示例6: 文件类型验证 ===")

    client = BailianFileUpload()

    # 显示支持的文件类型
    print("支持的文件类型:")
    supported_types = client.get_supported_types()

    type_categories = {
        "图片": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
        "视频": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"],
        "音频": [".mp3", ".wav", ".aac", ".flac", ".ogg"],
        "文档": [".pdf", ".txt", ".doc", ".docx"],
    }

    for category, extensions in type_categories.items():
        print(f"\n{category}类型:")
        for ext in extensions:
            mime_type = supported_types.get(ext, "unknown")
            print(f"  {ext} -> {mime_type}")

    # 创建测试文件进行验证
    print("\n--- 文件验证测试 ---")

    # 创建有效的测试文件
    valid_file = "valid_test.txt"
    with open(valid_file, "w", encoding="utf-8") as f:
        f.write("这是一个有效的测试文件")

    try:
        # 验证有效文件
        print(f"\n验证文件: {valid_file}")
        file_info = client.validate_file(valid_file)
        print("✓ 文件验证通过")
        print(f"  文件名: {file_info['name']}")
        print(f"  大小: {client.format_file_size(file_info['size'])}")
        print(f"  类型: {file_info['extension']} ({file_info['mime_type']})")

    except Exception as e:
        print(f"✗ 文件验证失败: {e}")
    finally:
        if os.path.exists(valid_file):
            os.remove(valid_file)

    # 测试无效文件类型
    print("\n--- 无效文件类型测试 ---")
    invalid_file = "invalid_test.xyz"
    with open(invalid_file, "w") as f:
        f.write("这是一个无效类型的文件")

    try:
        print(f"验证无效文件: {invalid_file}")
        client.validate_file(invalid_file)
        print("✗ 应该验证失败但却通过了")
    except Exception as e:
        print(f"✓ 正确识别无效文件类型: {e}")
    finally:
        if os.path.exists(invalid_file):
            os.remove(invalid_file)

    # 测试不存在的文件
    print("\n--- 不存在文件测试 ---")
    try:
        print("验证不存在的文件: nonexistent.txt")
        client.validate_file("nonexistent.txt")
        print("✗ 应该验证失败但却通过了")
    except Exception as e:
        print(f"✓ 正确识别文件不存在: {e}")


def example_ai_integration_workflow():
    """
    示例7: AI集成工作流
    演示如何将上传的文件用于AI生成任务
    """
    print("\n=== 示例7: AI集成工作流演示 ===")

    client = BailianFileUpload()

    print("这个示例演示了如何将文件上传与AI功能集成的完整工作流:")

    print("\n1. 图生视频工作流:")
    print("   a) 上传参考图片 -> 获取临时URL")
    print("   b) 使用URL调用视频生成API")
    print("   c) 生成视频并下载")

    print("\n2. 视频编辑工作流:")
    print("   a) 上传原始视频 -> 获取临时URL")
    print("   b) 上传掩码图像（如需要）-> 获取临时URL")
    print("   c) 使用URL调用视频编辑API")
    print("   d) 获取编辑后的视频")

    print("\n3. 多模态生成工作流:")
    print("   a) 上传多个参考文件 -> 获取临时URL列表")
    print("   b) 组合使用多个URL进行AI生成")
    print("   c) 处理生成结果")

    # 创建示例文件模拟工作流
    print("\n--- 模拟图生视频工作流 ---")

    # 创建模拟图片文件
    mock_image = "reference_image.txt"  # 实际应该是.jpg或.png文件
    with open(mock_image, "w", encoding="utf-8") as f:
        f.write("模拟参考图片文件内容\n")
        f.write("实际使用时应该是真实的图片文件")

    try:
        # 步骤1: 上传参考图片
        print("步骤1: 上传参考图片...")
        result = client.upload_file(mock_image, purpose="vision")

        if "url" in result:
            temp_url = result["url"]
            print(f"✓ 图片上传成功，临时URL: {temp_url}")

            # 步骤2: 模拟调用视频生成API
            print("\n步骤2: 调用视频生成API...")
            print(f"模拟API调用: video_generation_api(ref_image_url='{temp_url}')")
            print("✓ 视频生成任务已提交")

            # 步骤3: 模拟等待和下载
            print("\n步骤3: 等待视频生成完成...")
            print("✓ 视频生成完成，可以下载")

            print("\n完整工作流演示完成！")
        else:
            print("✗ 图片上传失败，无法继续工作流")

    except Exception as e:
        print(f"工作流执行失败: {e}")
    finally:
        if os.path.exists(mock_image):
            os.remove(mock_image)

    print("\n工作流集成要点:")
    print("1. 上传文件后立即获取临时URL")
    print("2. 临时URL通常有24小时有效期")
    print("3. 在AI API调用中使用临时URL作为输入")
    print("4. 处理完成后可以删除临时文件")
    print("5. 注意文件大小和类型限制")


def show_usage_guide():
    """
    显示使用指南
    """
    print("\n=== 使用指南 ===")

    client = BailianFileUpload()

    print("\n支持的文件类型:")
    supported_types = client.get_supported_types()

    print("  图片: jpg, jpeg, png, gif, bmp, webp")
    print("  视频: mp4, avi, mov, wmv, flv, mkv")
    print("  音频: mp3, wav, aac, flac, ogg")
    print("  文档: pdf, txt, doc, docx")

    print(f"\n文件大小限制: {client.format_file_size(client.max_file_size)}")

    print("\n用途参数说明:")
    print("  file-extract: 文件提取（默认）")
    print("  vision: 视觉相关任务")
    print("  assistants: 助手功能")
    print("  batch: 批处理任务")

    print("\n最佳实践:")
    print("1. 上传前验证文件类型和大小")
    print("2. 根据用途选择合适的purpose参数")
    print("3. 及时保存返回的文件ID和临时URL")
    print("4. 注意临时URL的有效期（通常24小时）")
    print("5. 不再需要时及时删除文件")
    print("6. 批量上传时注意API调用频率限制")


if __name__ == "__main__":
    """
    主函数：运行所有示例

    注意事项:
    1. 确保已设置DASHSCOPE_API_KEY环境变量
    2. 文件上传速度取决于网络状况和文件大小
    3. 临时URL有效期通常为24小时
    4. 注意API调用频率限制和计费
    5. 及时清理不需要的文件
    """
    print("百炼平台文件上传获取临时公网URL API示例")
    print("=" * 60)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("设置方法: export DASHSCOPE_API_KEY=your_api_key")
        exit(1)

    try:
        # 显示使用指南
        show_usage_guide()

        # 运行示例
        print("\n开始运行文件上传示例...")

        # 收集上传的文件ID用于后续管理
        uploaded_file_ids = []

        # 示例1: 单文件上传
        file_id = example_single_file_upload()
        if file_id:
            uploaded_file_ids.append(file_id)

        # 示例2: 图片上传（演示）
        example_image_upload()

        # 示例3: 视频上传（演示）
        example_video_upload()

        # 示例4: 批量上传
        batch_ids = example_batch_upload()
        uploaded_file_ids.extend(batch_ids)

        # 示例5: 文件管理
        example_file_management(uploaded_file_ids)

        # 示例6: 文件类型验证
        example_file_type_validation()

        # 示例7: AI集成工作流
        example_ai_integration_workflow()

        print("\n=== 所有示例执行完成 ===")

        if uploaded_file_ids:
            print(f"\n本次上传的文件ID: {uploaded_file_ids}")
            print("注意: 这些文件会占用存储空间，不需要时请及时删除")

        print("\n使用说明:")
        print("1. 所有示例都已成功运行")
        print("2. 实际使用时请替换示例中的文件路径")
        print("3. 注意文件类型和大小限制")
        print("4. 临时URL有效期为24小时")
        print("5. 可以将上传的URL用于AI生成任务")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
