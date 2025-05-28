import json
import re

from typing import Dict, List, Any
from src.models.vlm_model import VLMModel
from src.models.llm_model import LLMModel
from PIL import Image
from ..config.prompts import SKIN_ANALYSIS_PROMPT
from ..config.prompts import CONDITION_SCORE_PROMPT

class VLMSkinAnalyzer:
    """皮肤状态分析引擎"""
    
    def __init__(self):
        self.vlm_model = VLMModel()
        self.vlm_model.initialize()
        
    # def analyze_skin(self, image: Image.Image) -> Dict[str, Any]:
    #     """分析皮肤状态"""
    #     # 验证输入
    #     if not self.vlm_model.validate_input(image):
    #         raise ValueError("Invalid input image")
            
    #     # 获取VLM分析结果
    #     prompt = SKIN_ANALYSIS_PROMPT.format(image_description="用户上传的面部照片")
    #     analysis = self.vlm_model.predict(image, prompt)
        
    #     # 解析分析结果
    #     conditions = self._parse_conditions(analysis)
        
    #     return {
    #         "raw_analysis": analysis,
    #         "detected_conditions": conditions,
    #         "confidence_scores": self._calculate_confidence(conditions)
    #     }
        
    def analyze_skin(self, image: Image.Image) -> Dict[str, Any]:
        """分析皮肤状态"""
        try:
            # 确保输入是PIL.Image对象
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL.Image对象")

            if not self.vlm_model.validate_input(image):
                raise ValueError("无效的图片输入")

            prompt = SKIN_ANALYSIS_PROMPT.format(image_description="用户上传的面部照片")
            analysis = self.vlm_model.predict(image, prompt)
            conditions = self._parse_conditions(analysis)

            return {
                "skin_analysis": analysis,
                "confidence_scores": conditions
            }
        except Exception as e:
            raise Exception(f"图片分析失败: {str(e)}")
    
    def _parse_conditions(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """使用语言模型为皮肤状况评分"""

        # 将分析结果转为纯文本
        analysis_text = "\n".join([f"{k}：{v}" for k, v in analysis.items()])

        # 构建 Prompt
        prompt = CONDITION_SCORE_PROMPT.format(analysis_text=analysis_text)

        # 调用 LLM
        llm = LLMModel()
        llm.initialize()
        result = llm.predict(prompt)

        # 如果直接是 dict（强解析型模型返回）
        if isinstance(result, dict):
            return result

        # 否则尝试从字符串中提取 JSON 对象
        if isinstance(result, str):
            try:
                match = re.search(r"```json\s*({.*?})\s*```", result, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        return parsed
            except Exception as e:
                print("❌ 评分解析失败：", e)

        # fallback 返回默认值
        return {k: 0.0 for k in ["acne", "wrinkles", "pigmentation", "dryness", "oiliness", "scars"]}
            
    def _calculate_confidence(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """计算每个检测到的状况的置信度"""
        # TODO: 实现更复杂的置信度计算
        return conditions 
    
if __name__ == "__main__":
    from PIL import Image

    analyzer = VLMSkinAnalyzer()
    image = Image.open("cases/acne_faces/1.jpg")
    result = analyzer.analyze_skin(image)
    print("分析结果：")
    print(result)