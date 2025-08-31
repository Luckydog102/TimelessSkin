from typing import Any, Dict, List, Optional
from .base_model import BaseModel
import requests
import json
from dotenv import load_dotenv
import os
from PIL import Image
import re
import traceback
import logging
import base64
from io import BytesIO
import time

# 配置日志
logger = logging.getLogger(__name__)

class VLMModel(BaseModel):
    """视觉语言模型实现"""
    
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        # 如果.env不存在，尝试从env.txt加载
        self.model_name = os.getenv("VLM_MODEL_NAME", "qwen-vl-max")
        self.api_key = os.getenv("VLM_API_KEY")
        if not self.api_key:
            try:
                env_path = os.path.join(os.path.dirname(__file__), '../../env.txt')
                # 使用utf-8编码读取文件
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('VLM_API_KEY='):
                            self.api_key = line.strip().split('=', 1)[1]
                            break
            except Exception as e:
                logger.warning(f"无法从env.txt加载API密钥: {e}")
                
        # 如果仍然没有API密钥，使用默认值
        if not self.api_key:
            self.api_key = "sk-217c50dbecb64d2089a1f77f3ac079dc"
            
        self.api_base = os.getenv("VLM_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
        # 强制使用正确的API基础URL
        self.api_base = "https://dashscope.aliyuncs.com/api/v1"
        # 增加超时设置和重试次数
        self.timeout = 60  # 增加到60秒
        self.max_retries = 5  # 增加到5次
        self.retry_delay = 2  # 初始重试延迟（秒）
        
    def initialize(self) -> None:
        """初始化模型"""
        if not self.api_key:
            logger.warning("VLM API key not found, using default")
            
        # 测试API连接
        try:
            test_image = Image.new('RGB', (100, 100), color='white')
            test_result = self.predict(test_image, "测试连接")
            if test_result and isinstance(test_result, dict):
                logger.info("VLM Model initialized successfully")
            else:
                logger.warning("VLM Model test failed, but continuing")
        except Exception as e:
            logger.warning(f"VLM Model test failed: {e}")
            
    def predict(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """进行预测
        
        Args:
            image: PIL图像对象
            prompt: 提示文本
            
        Returns:
            预测结果
        """
        # 添加重试逻辑
        for attempt in range(self.max_retries):
            try:
                logger.info(f"尝试VLM分析 (尝试 {attempt+1}/{self.max_retries})")
                
                # 将图片转换为base64
                buffered = BytesIO()
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # 压缩图片以加快传输
                image.thumbnail((600, 600), Image.LANCZOS)  # 降低分辨率
                image.save(buffered, format="JPEG", quality=75)  # 降低质量
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # logger.info(f"图片处理完成，大小: {len(img_str) // 1024} KB")
                
                # 准备请求数据
                request_body = {
                    "model": self.model_name,
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "image": f"data:image/jpeg;base64,{img_str}"
                                    },
                                    {
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    },
                    "parameters": {
                        "max_tokens": 3000,  # 增加到3000以确保完整输出
                        "temperature": 0.7,  # 降低温度以提高一致性
                        "result_format": "message",
                        "seed": 1234,
                        "timeout": self.timeout  # 设置超时时间
                    }
                }
                
                # 发送请求
                endpoint = f"{self.api_base}/services/aigc/multimodal-generation/generation"
                # logger.info(f"🔥 VLM API调用：发送请求到 {endpoint}")
                # logger.info(f"🔥 VLM API调用：请求体大小 {len(json.dumps(request_body))} 字符")
                
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-Client": "TimelessSkin/1.0"
                    },
                    json=request_body,
                    timeout=self.timeout
                )
                
                # logger.info(f"🔥 VLM API调用：响应状态码 {response.status_code}")
                # logger.info(f"🔥 VLM API调用：响应头 {dict(response.headers)}")
                
                if response.status_code != 200:
                    error_msg = f"API调用失败(状态码:{response.status_code}): {response.text}"
                    logger.error(error_msg)
                    
                    # 如果是请求过多错误，等待后重试
                    if response.status_code == 429 and attempt < self.max_retries - 1:
                        retry_delay = self.retry_delay * (2 ** attempt)  # 指数退避
                        logger.warning(f"请求过多，等待{retry_delay}秒后重试 (尝试 {attempt+1}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                        
                    # 如果是超时错误，增加超时时间后重试
                    if response.status_code == 408 and attempt < self.max_retries - 1:
                        self.timeout += 30  # 每次增加30秒
                        logger.warning(f"请求超时，增加超时时间到{self.timeout}秒后重试")
                        continue
                        
                    # 如果是最后一次尝试或其他错误，返回错误信息
                    if attempt == self.max_retries - 1:
                        return {"skin_analysis": error_msg}
                    continue
                    
                result = response.json()
                # logger.info(f"API返回结果长度: {len(json.dumps(result))} 字符")
                # logger.info(f"API返回结果结构: {list(result.keys())}")
                
                if "output" in result and "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                    choice = result["output"]["choices"][0]
                    # logger.info(f"Choice结构: {list(choice.keys())}")
                    
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        # logger.info(f"Content类型: {type(content)}, 长度: {len(str(content))}")
                        
                        if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                            text = content[0]["text"]
                            # logger.info(f"解析到的文本内容长度: {len(text)} 字符")
                            # logger.info(f"文本内容前100字符: {text[:100]}")
                            # logger.info(f"文本内容后100字符: {text[-100:] if len(text) > 100 else text}")
                            
                            # 确保包含age_group字段
                            if '"age_group"' not in text and "'age_group'" not in text:
                                text = text.replace('"skin_analysis": {', '"skin_analysis": {\n        "age_group": "",')
                            
                            # 尝试解析JSON
                            try:
                                # 首先尝试查找JSON代码块
                                json_pattern = r'```json\s*(.*?)\s*```'
                                match = re.search(json_pattern, text, re.DOTALL)
                                
                                if match:
                                    json_str = match.group(1)
                                    logger.info(f"找到JSON代码块，长度: {len(json_str)} 字符")
                                    # 清理JSON字符串中的格式问题
                                    json_str = json_str.replace('\n', ' ').replace('\\n', ' ')
                                    json_str = re.sub(r',\s*}', '}', json_str)
                                    json_str = re.sub(r',\s*"', ',"', json_str)
                                    json_str = re.sub(r'"\s*,\s*"', '","', json_str)
                                    try:
                                        result = json.loads(json_str)
                                        logger.info(f"成功解析JSON结果")
                                        return {"skin_analysis": result}
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON代码块解析失败: {e}")
                                        # 如果JSON解析失败，尝试直接解析
                                        try:
                                            result = json.loads(text)
                                            logger.info(f"直接解析文本为JSON成功")
                                            return {"skin_analysis": result}
                                        except json.JSONDecodeError:
                                            logger.warning(f"无法解析JSON，返回原始文本")
                                            return {"skin_analysis": text}
                                    
                                # 如果没有找到JSON代码块，尝试直接解析整个文本
                                try:
                                    result = json.loads(text)
                                    logger.info(f"直接解析文本为JSON成功")
                                    return {"skin_analysis": result}
                                except json.JSONDecodeError:
                                    logger.warning(f"无法直接解析文本为JSON，尝试清理后解析")
                                    # 尝试清理文本后解析
                                    cleaned_text = text.strip()
                                    # 移除可能的markdown标记
                                    if cleaned_text.startswith('```') and cleaned_text.endswith('```'):
                                        cleaned_text = cleaned_text[3:-3].strip()
                                    if cleaned_text.startswith('json'):
                                        cleaned_text = cleaned_text[4:].strip()
                                    
                                    try:
                                        result = json.loads(cleaned_text)
                                        logger.info(f"清理后解析JSON成功")
                                        return {"skin_analysis": result}
                                    except json.JSONDecodeError:
                                        logger.warning(f"清理后仍无法解析JSON，返回原始文本")
                                        return {"skin_analysis": text}
                                
                            except Exception as e:
                                logger.warning(f"JSON解析过程中出现错误: {str(e)}")
                                return {"skin_analysis": text}
                        else:
                            logger.warning(f"Content不是预期的列表格式: {content}")
                else:
                    logger.warning(f"API返回格式不符合预期，output或choices缺失")
                    logger.info(f"Result内容: {result}")
                    
                logger.warning("API返回格式不符合预期")
                # 如果不是最后一次尝试，继续重试
                if attempt < self.max_retries - 1:
                    continue
                return {"skin_analysis": "API返回格式错误，请重试"}
                    
            except requests.Timeout:
                logger.warning(f"请求超时 (尝试 {attempt+1}/{self.max_retries})")
                # 增加超时时间后重试
                if attempt < self.max_retries - 1:
                    self.timeout += 30  # 每次增加30秒
                    logger.warning(f"增加超时时间到{self.timeout}秒后重试")
                    continue
                return {"skin_analysis": "分析超时，请重试"}
            except requests.RequestException as e:
                logger.error(f"网络请求错误: {str(e)}")
                # 如果不是最后一次尝试，继续重试
                if attempt < self.max_retries - 1:
                    retry_delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"等待{retry_delay}秒后重试")
                    time.sleep(retry_delay)
                    continue
                return {"skin_analysis": f"网络请求错误: {str(e)}"}
            except Exception as e:
                logger.error(f"未预期的错误: {str(e)}")
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                # 如果不是最后一次尝试，继续重试
                if attempt < self.max_retries - 1:
                    retry_delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"等待{retry_delay}秒后重试")
                    time.sleep(retry_delay)
                    continue
                return {"skin_analysis": f"分析过程中出现错误: {str(e)}"}
                
        # 如果所有尝试都失败
        return {"skin_analysis": "多次尝试后分析失败，请稍后重试"}
            
    def validate_input(self, image: Any) -> bool:
        """验证输入数据
        
        Args:
            image: 输入图像
            
        Returns:
            是否为有效输入
        """
        return isinstance(image, Image.Image)
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "provider": "Qwen",
            "type": "vision_language_model",
            "api_base": self.api_base
        } 
    

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # eval the VLM model
    model = VLMModel()
    model.initialize()

    # 假设你有一张图片 image.png
    image = Image.open("cases/acne_faces/1.jpg")
    prompt = "用户上传的面部照片" # SKIN_ANALYSIS_PROMPT.format(image_description="用户上传的面部照片")
    if model.validate_input(image):
        result = model.predict(image, prompt)
        print("分析结果：", result)
    else:
        print("无效输入")