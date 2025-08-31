from typing import Dict, List, Any
from ..models.llm_model import LLMModel
from ..config.prompts import TRUST_REASONING_PROMPT
import json

class TrustReasoning:
    """信任推理模块"""
    
    def __init__(self):
        self.gpt_model = LLMModel()
        self.gpt_model.initialize()
        
    def generate_trust_reasoning(self,
                               skin_conditions: Dict[str, float],
                               recommendations: List[Dict[str, Any]],
                               user_profile: Dict[str, Any]) -> str:
        """生成信任推理"""
        # 构建信任推理提示
        prompt = self._build_trust_prompt(
            skin_conditions,
            recommendations,
            user_profile
        )
        
        # 获取GPT推理
        reasoning = self.gpt_model.predict(prompt)
        
        return self._parse_reasoning(reasoning)
        
    def _build_trust_prompt(self,
                           skin_conditions: Dict[str, float],
                           recommendations: List[Dict[str, Any]],
                           user_profile: Dict[str, Any]) -> str:
        """构建信任推理提示"""
        return TRUST_REASONING_PROMPT.format(
            skin_condition=self._format_conditions(skin_conditions),
            recommendations=self._format_recommendations(recommendations),
            user_profile=self._format_profile(user_profile),
            knowledge_context="基于皮肤状况、推荐产品和用户画像的上下文信息"
        )
        
    def _format_conditions(self, conditions: Dict[str, float]) -> str:
        """格式化皮肤状况"""
        return "\n".join([f"- {k}: {v:.2f}" for k, v in conditions.items()])
        
    def _format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """格式化推荐产品"""
        return "\n".join([f"- {r['product_name']}: {r.get('description', '')}" for r in recommendations])
        
    def _format_profile(self, profile: Dict[str, Any]) -> str:
        """格式化用户画像"""
        return "\n".join([f"- {k}: {v}" for k, v in profile.items()])
        
    def _parse_reasoning(self, reasoning: str) -> Dict[str, Any]:
        """解析推理结果"""
        try:
            # 尝试解析JSON格式的推理
            reasoning_data = json.loads(reasoning)
            if isinstance(reasoning_data, dict):
                return reasoning_data
            return {"raw_reasoning": reasoning}
        except json.JSONDecodeError:
            return {"raw_reasoning": reasoning} 