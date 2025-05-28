from typing import List, Any, Union
from .base_model import BaseModel
import requests
import json
from dotenv import load_dotenv
import os
import logging
import time

# 配置日志
logger = logging.getLogger(__name__)

class EmbeddingModel(BaseModel):
    """BGE-large-zh-v1.5 embedding模型API实现"""
    
    def __init__(self, model_name: str = None):
        load_dotenv(dotenv_path='.env')
        # 优先从env.txt加载，如果.env不存在
        self.api_key = os.getenv("EMBEDDING_API_KEY")
        if not self.api_key:
            try:
                with open('env.txt', 'r') as f:
                    for line in f:
                        if line.startswith('EMBEDDING_API_KEY='):
                            self.api_key = line.strip().split('=', 1)[1]
                            break
            except Exception as e:
                logger.warning(f"无法从env.txt加载API密钥: {e}")
                
        # 如果仍然没有API密钥，使用默认值
        if not self.api_key:
            self.api_key = "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"
            
        self.api_base = os.getenv("EMBEDDING_API_BASE", "https://api-inference.huggingface.co/models/BAAI/bge-large-zh-v1.5")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "bge-large-zh-v1.5")
        self.max_retries = 3
        self.timeout = 30
        logger.info(f"初始化嵌入模型: {self.model_name}")
        
    def initialize(self) -> None:
        """初始化API连接"""
        if not self.api_key:
            logger.warning("未找到Embedding API密钥，将使用默认向量")
            # 不抛出异常，允许系统继续运行
            return
            
        # 测试API连接
        try:
            test_result = self.predict("测试连接")
            if test_result and len(test_result) > 0 and len(test_result[0]) == 768:
                logger.info(f"嵌入模型初始化成功: {self.model_name}")
            else:
                logger.warning("嵌入模型测试失败，将使用默认向量")
        except Exception as e:
            logger.warning(f"嵌入模型测试失败: {e}")
            
    def predict(self, input_data: Union[str, List[str]]) -> List[List[float]]:
        """使用API生成文本嵌入"""
        try:
            if not self.api_key:
                logger.warning("未配置API密钥，返回空向量")
                # 返回默认向量
                if isinstance(input_data, str):
                    return [self._get_default_embedding()]
                else:
                    return [self._get_default_embedding() for _ in range(len(input_data))]
                    
            if isinstance(input_data, str):
                input_data = [input_data]
                
            # 检查输入是否为空
            if not input_data or len(input_data) == 0:
                logger.warning("输入为空，返回空列表")
                return []
                
            # 分批处理，每批最多10个文本
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size]
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # HuggingFace API格式
                payload = {
                    "inputs": batch,
                    "options": {
                        "wait_for_model": True,
                        "use_cache": True
                    }
                }
                
                # 添加重试逻辑
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"调用嵌入API，批次 {i//batch_size + 1}，尝试 {attempt + 1}/{self.max_retries}")
                        response = requests.post(
                            self.api_base,
                            headers=headers,
                            json=payload,
                            timeout=self.timeout
                        )
                        
                        if response.status_code == 200:
                            embeddings = response.json()
                            if isinstance(embeddings, list):
                                # 检查向量维度
                                if len(embeddings) > 0 and isinstance(embeddings[0], list):
                                    all_embeddings.extend(embeddings)
                                    break  # 成功获取向量，跳出重试循环
                                else:
                                    logger.error(f"API返回的嵌入向量格式异常: {type(embeddings)}")
                                    if attempt == self.max_retries - 1:
                                        # 最后一次尝试失败，使用默认向量
                                        all_embeddings.extend([self._get_default_embedding() for _ in batch])
                            else:
                                logger.error(f"API返回格式异常: {type(embeddings)}")
                                if attempt == self.max_retries - 1:
                                    all_embeddings.extend([self._get_default_embedding() for _ in batch])
                        elif response.status_code == 429:  # 速率限制
                            logger.warning(f"API速率限制，等待后重试: {response.text}")
                            time.sleep(2 ** attempt)  # 指数退避
                            if attempt == self.max_retries - 1:
                                all_embeddings.extend([self._get_default_embedding() for _ in batch])
                        else:
                            logger.error(f"API调用失败: {response.status_code}")
                            logger.error(f"错误信息: {response.text}")
                            
                            if attempt == self.max_retries - 1:
                                all_embeddings.extend([self._get_default_embedding() for _ in batch])
                            else:
                                time.sleep(1)
                                
                    except Exception as e:
                        logger.error(f"请求过程中发生错误: {str(e)}")
                        if attempt == self.max_retries - 1:
                            all_embeddings.extend([self._get_default_embedding() for _ in batch])
                        else:
                            time.sleep(1)
                            
            logger.info(f"成功获取嵌入向量: {len(all_embeddings)}个")
            return all_embeddings
                
        except Exception as e:
            logger.error(f"生成嵌入向量时发生错误: {str(e)}")
            # 返回默认向量
            if isinstance(input_data, str):
                return [self._get_default_embedding()]
            else:
                return [self._get_default_embedding() for _ in range(len(input_data))]
    
    def _get_default_embedding(self) -> List[float]:
        """返回一个默认的嵌入向量（全零向量）"""
        return [0.0] * 768  # BGE模型的维度是768
            
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        return isinstance(input_data, (str, list)) and all(isinstance(x, str) for x in (input_data if isinstance(input_data, list) else [input_data]))
        
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "provider": "HuggingFace",
            "type": "embedding_model",
            "api_base": self.api_base
        }
        
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    model = EmbeddingModel()
    model.initialize()
    result = model.predict(["这是一个测试", "皮肤有点干燥"])
    
    if result and len(result) > 0:
        print(f"返回向量维度：{len(result[0])}")
        print(f"前5维向量示例：{result[0][:5]}")
    else:
        print("获取嵌入向量失败")