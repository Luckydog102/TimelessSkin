from typing import Any, Dict, List, Optional, Tuple, Generator, Iterator
from .base_model import BaseModel
import requests
import json
from dotenv import load_dotenv
import os
import traceback
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class LLMModel(BaseModel):
    """Qwen模型API实现"""
    
    def __init__(self, model_name: str = None):
        load_dotenv(dotenv_path=".env")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "qwen-max")
        self.api_key = os.getenv("LLM_API_KEY", "sk-217c50dbecb64d2089a1f77f3ac079dc")
        
        # 使用与VLM相同的API基础URL，确保兼容性
        self.api_base = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
        
        # 如果环境变量中没有设置，尝试从env.txt加载
        if not self.api_key or self.api_key == "sk-217c50dbecb64d2089a1f77f3ac079dc":
            try:
                env_path = os.path.join(os.path.dirname(__file__), '../../env.txt')
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('LLM_API_KEY=') and not self.api_key:
                            self.api_key = line.split('=', 1)[1]
                        elif line.startswith('LLM_MODEL_NAME=') and not self.model_name:
                            self.model_name = line.split('=', 1)[1]
                        
                        # 如果所有配置都已读取，提前退出
                        if self.api_key and self.model_name:
                            break
            except Exception as e:
                print(f"无法从env.txt加载配置: {e}")
        
        # 强制使用正确的API基础URL
        self.api_base = "https://dashscope.aliyuncs.com/api/v1"
        
        # 创建带重试机制的session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # 总重试次数
            backoff_factor=1,  # 重试间隔倍数
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["POST"]  # 允许重试的HTTP方法
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        print(f"LLM配置: model={self.model_name}, base={self.api_base}, key={self.api_key[:10]}...{self.api_key[-10:]}")
            
    def initialize(self) -> None:
        """初始化API连接"""
        if not self.api_key:
            raise ValueError("LLM API key not found")
        print(f"LLM Model initialized: {self.model_name}")
        print(f"API Base: {self.api_base}")
        print(f"API Key: {self.api_key[:5]}...{self.api_key[-5:]}")
            
    def predict(self, input_data: str, temperature: float = 0.7) -> Any:
        """使用API进行预测
        
        Args:
            input_data: 输入文本
            temperature: 温度参数，控制生成的随机性，默认0.7，范围0.0-2.0
            
        Returns:
            模型输出
        """
        try:
            # 确保temperature在合理范围内
            temperature = max(0.0, min(2.0, temperature))
            
            # 准备请求体 - 使用Qwen API格式
            request_body = {
                "model": self.model_name,
                "input": {
                    "messages": [{"role": "user", "content": input_data}]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": 2000
                }
            }
            
            # 真实API调用 - 使用正确的文本生成端点
            endpoint = f"{self.api_base}/services/aigc/text-generation/generation"
            print(f"发送API请求: {endpoint}")
            print(f"请求体: {json.dumps(request_body, ensure_ascii=False)}")
            
            response = self.session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Client": "TimelessSkin/1.0"  # 添加客户端标识
                },
                json=request_body,
                timeout=(10, 60),  # 连接超时10秒，读取超时60秒
                verify=True  # 启用SSL证书验证
            )
            
            print(f"API状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"API错误: {response.status_code}")
                print(f"响应内容: {response.text}")
                raise Exception(f"API调用失败: {response.text}")
                
            result = response.json()
            print(f"API返回结果: {json.dumps(result, ensure_ascii=False)[:200]}...")
            
            # 解析Qwen API的返回结果
            if "output" in result and "text" in result["output"]:
                output = result["output"]["text"]
                print(f"模型输出: {output[:200]}...")
                
                # 尝试解析JSON输出
                try:
                    # 查找JSON部分
                    json_pattern = r'```json\s*(.*?)\s*```'
                    json_match = re.search(json_pattern, output, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"找到JSON代码块: {json_str[:100]}...")
                        return json.loads(json_str)
                        
                    # 如果没有找到JSON代码块，尝试直接寻找JSON对象
                    json_start = output.find("[")
                    json_end = output.rfind("]") + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = output[json_start:json_end]
                        print(f"找到JSON数组: {json_str[:100]}...")
                        return json.loads(json_str)
                        
                    json_start = output.find("{")
                    json_end = output.rfind("}") + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = output[json_start:json_end]
                        print(f"找到JSON对象: {json_str[:100]}...")
                        return json.loads(json_str)
                        
                    print("未找到JSON格式，返回原始文本")
                    return output
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}")
                    return output
            else:
                raise Exception(f"API返回格式错误: {result}")
                
        except requests.exceptions.Timeout as e:
            print(f"LLM请求超时: {str(e)}")
            raise Exception("网络请求超时，请检查网络连接或稍后重试")
        except requests.exceptions.ConnectionError as e:
            print(f"LLM连接错误: {str(e)}")
            raise Exception("网络连接失败，请检查网络设置")
        except requests.exceptions.RequestException as e:
            print(f"LLM请求错误: {str(e)}")
            raise Exception(f"网络请求失败: {str(e)}")
        except Exception as e:
            print(f"LLM预测错误: {str(e)}")
            print(f"详细错误: {traceback.format_exc()}")
            raise Exception(f"LLM预测失败: {str(e)}")
    
    def predict_stream(self, input_data: str, temperature: float = 0.7) -> Iterator[str]:
        """使用API进行流式预测
        
        Args:
            input_data: 输入文本
            temperature: 温度参数，控制生成的随机性，默认0.7，范围0.0-2.0
            
        Returns:
            生成器，逐步返回模型输出
        """
        try:
            # 确保temperature在合理范围内
            temperature = max(0.0, min(2.0, temperature))
            
            # 准备请求体 - 使用Qwen API格式
            request_body = {
                "model": self.model_name,
                "input": {
                    "messages": [{"role": "user", "content": input_data}]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": 2000,
                    "incremental_output": True  # 启用流式输出
                }
            }
            
            # 真实API调用
            endpoint = f"{self.api_base}/services/aigc/text-generation/generation"
            print(f"发送流式API请求: {endpoint}")
            
            response = self.session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"  # 指定接收事件流
                },
                json=request_body,
                stream=True,  # 启用流式响应
                timeout=(10, 60)  # 连接超时10秒，读取超时60秒
            )
            
            print(f"API状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API错误: {response.status_code}")
                print(f"响应头: {response.headers}")
                raise Exception(f"API调用失败: {response.status_code}")
                
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    try:
                        # 解析SSE格式数据
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data:'):
                            data_json = json.loads(line_text[5:].strip())
                            if "output" in data_json and "text" in data_json["output"]:
                                chunk = data_json["output"]["text"]
                                yield chunk
                    except Exception as e:
                        print(f"解析流式响应出错: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"LLM流式预测错误: {str(e)}")
            print(f"详细错误: {traceback.format_exc()}")
            yield f"错误: {str(e)}"
    
    def chat_stream(self, message: str, system_message: str = "", temperature: float = 0.7) -> Iterator[str]:
        """流式聊天功能
        
        Args:
            message: 用户消息
            system_message: 系统消息/上下文
            temperature: 温度参数
            
        Returns:
            生成器，逐步返回模型输出
        """
        try:
            # 构建消息列表
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": message})
            
            # 准备请求体
            request_body = {
                "model": self.model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "temperature": max(0.0, min(2.0, temperature)),
                    "max_tokens": 2000,
                    "incremental_output": True,  # 启用流式输出
                    "result_format": "message"  # 使用消息格式
                }
            }
            
            # 发送请求
            endpoint = f"{self.api_base}/services/aigc/text-generation/generation"
            print(f"发送流式API请求: {endpoint}")
            
            response = self.session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-DashScope-SSE": "enable"  # 启用SSE
                },
                json=request_body,
                stream=True,
                timeout=(10, 60)  # 连接超时10秒，读取超时60秒
            )
            
            if response.status_code != 200:
                print(f"API错误: {response.status_code}")
                print(f"响应头: {response.headers}")
                yield f"API调用失败: {response.status_code}"
                return
                
            # 处理流式响应
            buffer = ""  # 用于存储未完成的JSON字符串
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # 解析SSE格式数据
                    line_text = line.decode('utf-8')
                    if not line_text.startswith('data:'):
                        continue
                        
                    # 提取JSON数据
                    json_str = line_text[5:].strip()
                    buffer += json_str
                    
                    try:
                        # 尝试解析完整的JSON
                        data = json.loads(buffer)
                        buffer = ""  # 重置缓冲区
                        
                        # 提取文本内容
                        if "output" in data:
                            if "text" in data["output"]:
                                # 旧版API格式
                                chunk = data["output"]["text"]
                                yield chunk
                            elif "choices" in data["output"] and len(data["output"]["choices"]) > 0:
                                # 新版API格式
                                choice = data["output"]["choices"][0]
                                if "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                    if isinstance(content, list) and len(content) > 0:
                                        if isinstance(content[0], dict) and "text" in content[0]:
                                            chunk = content[0]["text"]
                                            yield chunk
                                        elif isinstance(content[0], str):
                                            chunk = content[0]
                                            yield chunk
                                    elif isinstance(content, str):
                                        yield content
                                elif "delta" in choice and "content" in choice["delta"]:
                                    # 增量更新格式
                                    chunk = choice["delta"]["content"]
                                    yield chunk
                                    
                    except json.JSONDecodeError:
                        # JSON不完整，继续等待更多数据
                        continue
                        
                except Exception as e:
                    print(f"处理流式响应出错: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"LLM流式聊天错误: {str(e)}")
            print(f"详细错误: {traceback.format_exc()}")
            yield f"错误: {str(e)}"
            
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        return isinstance(input_data, str) and len(input_data) > 0
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "provider": "Qwen",
            "type": "language_model",
            "api_base": self.api_base
        }

    def chat(self, message: str, system_message: str = "", temperature: float = 0.7) -> str:
        """聊天功能
        
        Args:
            message: 用户消息
            system_message: 系统消息/上下文
            temperature: 温度参数
            
        Returns:
            助手回复
        """
        try:
            # 构建消息列表
            messages = []
            
            # 如果有系统消息，添加系统消息
            if system_message and isinstance(system_message, str) and len(system_message.strip()) > 0:
                messages.append({"role": "system", "content": system_message})
            
            # 添加当前消息
            messages.append({"role": "user", "content": message})
            
            # 准备请求体
            request_body = {
                "model": self.model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": 2000
                }
            }
            
            # API调用 - 使用正确的文本生成端点
            endpoint = f"{self.api_base}/services/aigc/text-generation/generation"
            print(f"发送API请求: {endpoint}")
            print(f"请求体: {json.dumps(request_body, ensure_ascii=False)}")
            
            response = self.session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Client": "TimelessSkin/1.0"  # 添加客户端标识
                },
                json=request_body,
                timeout=(10, 60)  # 连接超时10秒，读取超时60秒
            )
            
            print(f"API状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            
            if response.status_code != 200:
                raise Exception(f"API调用失败: {response.text}")
                
            result = response.json()
            
            if "output" in result and "text" in result["output"]:
                return result["output"]["text"]
            else:
                raise Exception(f"API返回格式错误: {result}")
                
        except Exception as e:
            print(f"聊天错误: {str(e)}")
            return f"抱歉，我遇到了一些问题：{str(e)}"

if __name__ == "__main__":
    prompt = QUESTION_GENERATION_PROMPT.format(
        skin_condition="面部有红斑，油脂分泌旺盛，伴有闭口和痘痘",
        knowledge_context="油性皮肤容易堵塞毛孔，建议使用控油产品"
    )
    model = LLMModel()
    model.initialize()
    if model.validate_input(prompt):
        result = model.predict(prompt)
        print("分析结果：", result)
    else:
        print("无效输入")
