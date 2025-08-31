from typing import Dict, List, Any
from ..models.llm_model import LLMModel
from ..config.prompts import RECOMMENDATION_PROMPT, ELDER_RECOMMENDATION_PROMPT
import json
import traceback
import re

class RecommendationEngine:
    """产品推荐引擎"""
    
    def __init__(self):
        self.gpt_model = LLMModel()
        self.gpt_model.initialize()
        
    def generate_recommendations(self, 
                               skin_conditions: Dict[str, float],
                               user_profile: Dict[str, Any],
                               retrieved_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成产品推荐"""
        try:
            # 验证输入
            if not skin_conditions or not isinstance(skin_conditions, dict):
                raise ValueError("无效的皮肤状况数据")
                
            if not user_profile or not isinstance(user_profile, dict):
                raise ValueError("无效的用户画像数据")
                
            if not retrieved_info or not isinstance(retrieved_info, list):
                print("⚠️ 警告：检索到的产品信息为空")
                retrieved_info = []
                
            # 构建推荐提示
            prompt = self._build_recommendation_prompt(
                skin_conditions,
                user_profile,
                retrieved_info
            )
            
            # 获取GPT推荐
            recommendations = self.gpt_model.predict(prompt)
            
            # 解析推荐结果
            parsed_recommendations = self._parse_recommendations(recommendations)
            
            # 验证推荐结果
            if not parsed_recommendations:
                print("⚠️ 警告：未能生成有效的推荐")
                return [{
                    "product_name": "通用保湿霜",
                    "brand": "欧莱雅",
                    "reason": "由于无法生成个性化推荐，建议使用温和的基础护肤品。"
                }]
                
            return parsed_recommendations
            
        except Exception as e:
            print(f"❌ 推荐生成失败: {str(e)}")
            print(f"详细错误: {traceback.format_exc()}")
            return [{
                "product_name": "通用保湿霜",
                "brand": "欧莱雅",
                "reason": "推荐系统暂时出现问题，建议使用温和的基础护肤品。"
            }]
        
    def _build_recommendation_prompt(self,
                                   skin_conditions: Dict[str, float],
                                   user_profile: Dict[str, Any],
                                   retrieved_info: List[Dict[str, Any]]) -> str:
        """构建推荐提示"""
        try:
            # 检查是否为中老年人或子女代购
            is_elder = False
            is_family_purchase = False
            
            # 检查年龄段
            age_range = user_profile.get("age_range", "")
            if age_range in ["46-55", "56-65", "66+"]:
                is_elder = True
                
            # 检查用户类型
            user_type = user_profile.get("user_type", "")
            if user_type == "子女代购":
                is_family_purchase = True
                
            # 根据用户类型选择不同的提示模板
            if is_elder or is_family_purchase:
                return ELDER_RECOMMENDATION_PROMPT.format(
                    skin_condition=self._format_conditions(skin_conditions),
                    user_profile=self._format_profile(user_profile),
                    retrieved_info=self._format_retrieved_info(retrieved_info),
                    knowledge_context="基于中老年皮肤状况、用户画像和产品信息的上下文",
                    budget=user_profile.get("budget", "不限"),
                    improvement_goals=", ".join(user_profile.get("improvement_goals", ["保湿补水", "抗皱紧致"]))
                )
            else:
                return RECOMMENDATION_PROMPT.format(
                    skin_condition=self._format_conditions(skin_conditions),
                    user_profile=self._format_profile(user_profile),
                    retrieved_info=self._format_retrieved_info(retrieved_info),
                    knowledge_context="基于皮肤状况、用户画像和产品信息的上下文"
                )
        except Exception as e:
            print(f"❌ 构建提示失败: {str(e)}")
            # 使用基础提示模板
            return RECOMMENDATION_PROMPT.format(
                skin_condition="需要基础护理",
                user_profile="普通用户",
                retrieved_info="基础护肤品信息",
                knowledge_context="基础护肤知识"
            )
        
    def _format_conditions(self, conditions: Dict[str, float]) -> str:
        """格式化皮肤状况"""
        try:
            return "\n".join([f"- {k}: {v:.2f}" for k, v in conditions.items()])
        except Exception as e:
            print(f"❌ 格式化皮肤状况失败: {str(e)}")
            return "皮肤状况数据格式错误"
        
    def _format_profile(self, profile: Dict[str, Any]) -> str:
        """格式化用户画像"""
        try:
            return "\n".join([f"- {k}: {v}" for k, v in profile.items()])
        except Exception as e:
            print(f"❌ 格式化用户画像失败: {str(e)}")
            return "用户画像数据格式错误"
        
    def _format_retrieved_info(self, info: List[Dict[str, Any]]) -> str:
        """格式化检索到的产品信息"""
        try:
            return "\n".join([f"- {item.get('name', 'Unknown')}: {item.get('description', '')}" for item in info])
        except Exception as e:
            print(f"❌ 格式化产品信息失败: {str(e)}")
            return "产品信息数据格式错误"
        
    def _parse_recommendations(self, recommendations: str) -> List[Dict[str, Any]]:
        """解析推荐结果"""
        try:
            # 尝试解析JSON格式的推荐
            if isinstance(recommendations, str):
                # 查找JSON部分
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, recommendations, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    recommendations_data = json.loads(json_str)
                else:
                    # 尝试直接解析
                    recommendations_data = json.loads(recommendations)
                    
                if isinstance(recommendations_data, list):
                    # 验证每个推荐项
                    valid_recommendations = []
                    for rec in recommendations_data:
                        if isinstance(rec, dict) and "product_name" in rec:
                            valid_recommendations.append(rec)
                    return valid_recommendations if valid_recommendations else [{"product_name": "通用保湿霜", "brand": "欧莱雅"}]
                    
            return [{"product_name": "通用保湿霜", "brand": "欧莱雅"}]
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {str(e)}")
            return [{"product_name": "通用保湿霜", "brand": "欧莱雅"}]
        except Exception as e:
            print(f"❌ 推荐解析错误: {str(e)}")
            return [{"product_name": "通用保湿霜", "brand": "欧莱雅"}] 