#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境监测脚本
检测项目运行所需的环境配置
"""

import os
import sys
import subprocess
import importlib
from typing import List, Tuple


class EnvironmentChecker:
    """环境检测器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_messages = []
    
    def check_command_exists(self, command: str) -> bool:
        """检查命令是否存在"""
        try:
            result = subprocess.run(
                ["which", command], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_command_version(self, command: str, version_flag: str = "--version") -> str:
        """获取命令版本信息"""
        try:
            result = subprocess.run(
                [command, version_flag], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
            return "版本信息获取失败"
        except Exception:
            return "版本信息获取失败"
    
    def check_ffmpeg_tools(self) -> bool:
        """检测ffmpeg和ffprobe"""
        print("\n=== 检测FFmpeg工具 ===")
        
        # 检查ffmpeg
        if self.check_command_exists("ffmpeg"):
            version = self.get_command_version("ffmpeg")
            self.success_messages.append(f"✅ ffmpeg已安装: {version}")
            print(f"✅ ffmpeg已安装: {version}")
        else:
            self.errors.append("❌ ffmpeg未安装")
            print("❌ ffmpeg未安装")
            print("   安装方法: brew install ffmpeg (macOS) 或 apt install ffmpeg (Ubuntu)")
        
        # 检查ffprobe
        if self.check_command_exists("ffprobe"):
            version = self.get_command_version("ffprobe")
            self.success_messages.append(f"✅ ffprobe已安装: {version}")
            print(f"✅ ffprobe已安装: {version}")
        else:
            self.errors.append("❌ ffprobe未安装")
            print("❌ ffprobe未安装")
            print("   ffprobe通常与ffmpeg一起安装")
        
        return len([e for e in self.errors if "ffmpeg" in e or "ffprobe" in e]) == 0
    
    def check_environment_variables(self) -> bool:
        """检测环境变量"""
        print("\n=== 检测环境变量 ===")
        
        # 检查DASHSCOPE_API_KEY
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            # 隐藏API密钥的大部分内容
            masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else "*" * len(api_key)
            self.success_messages.append(f"✅ DASHSCOPE_API_KEY已设置: {masked_key}")
            print(f"✅ DASHSCOPE_API_KEY已设置: {masked_key}")
            return True
        else:
            self.errors.append("❌ DASHSCOPE_API_KEY环境变量未设置")
            print("❌ DASHSCOPE_API_KEY环境变量未设置")
            print("   设置方法: export DASHSCOPE_API_KEY='your_api_key'")
            print("   或在.bashrc/.zshrc中添加该行")
            return False
    
    def check_python_environment(self) -> bool:
        """检测Python环境"""
        print("\n=== 检测Python环境 ===")
        
        # 检查Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.success_messages.append(f"✅ Python版本: {python_version}")
        print(f"✅ Python版本: {python_version}")
        
        # 检查是否在虚拟环境中
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            venv_path = sys.prefix
            self.success_messages.append(f"✅ 当前在虚拟环境中: {venv_path}")
            print(f"✅ 当前在虚拟环境中: {venv_path}")
        else:
            self.warnings.append("⚠️  未检测到虚拟环境，建议使用虚拟环境")
            print("⚠️  未检测到虚拟环境，建议使用虚拟环境")
            print("   创建虚拟环境: python -m venv venv")
            print("   激活虚拟环境: source venv/bin/activate")
        
        return True
    
    def check_required_packages(self) -> bool:
        """检测必需的Python包"""
        print("\n=== 检测Python依赖包 ===")
        
        # 读取requirements.txt
        requirements_file = "requirements.txt"
        if not os.path.exists(requirements_file):
            self.errors.append(f"❌ {requirements_file}文件不存在")
            print(f"❌ {requirements_file}文件不存在")
            return False
        
        # 解析requirements.txt
        required_packages = []
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 提取包名（去除版本要求）
                        package_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                        required_packages.append(package_name)
        except Exception as e:
            self.errors.append(f"❌ 读取{requirements_file}失败: {e}")
            print(f"❌ 读取{requirements_file}失败: {e}")
            return False
        
        # 检查每个包
        all_installed = True
        for package in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', '未知版本')
                self.success_messages.append(f"✅ {package}已安装: {version}")
                print(f"✅ {package}已安装: {version}")
            except ImportError:
                self.errors.append(f"❌ {package}未安装")
                print(f"❌ {package}未安装")
                all_installed = False
        
        if not all_installed:
            print("\n   安装依赖包: pip install -r requirements.txt")
        
        return all_installed
    
    def run_all_checks(self) -> bool:
        """运行所有检查"""
        print("开始环境检测...")
        print("=" * 50)
        
        # 执行所有检查
        ffmpeg_ok = self.check_ffmpeg_tools()
        env_ok = self.check_environment_variables()
        python_ok = self.check_python_environment()
        packages_ok = self.check_required_packages()
        
        # 输出总结
        print("\n" + "=" * 50)
        print("=== 检测结果总结 ===")
        
        if self.success_messages:
            print("\n成功项目:")
            for msg in self.success_messages:
                print(f"  {msg}")
        
        if self.warnings:
            print("\n警告项目:")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.errors:
            print("\n错误项目:")
            for msg in self.errors:
                print(f"  {msg}")
        
        all_ok = ffmpeg_ok and env_ok and packages_ok
        
        print("\n" + "=" * 50)
        if all_ok:
            print("🎉 环境检测通过！所有必需组件都已正确安装和配置。")
        else:
            print("❌ 环境检测未通过，请根据上述提示解决问题。")
        
        return all_ok


def main():
    """主函数"""
    checker = EnvironmentChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()