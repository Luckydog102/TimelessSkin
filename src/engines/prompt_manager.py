from typing import Dict, Any, List
from src.config.prompts import (
    SKIN_ANALYSIS_PROMPT,
    QUESTION_GENERATION_PROMPT,
    PROFILE_BUILDING_PROMPT,
    RECOMMENDATION_PROMPT,
    TRUST_REASONING_PROMPT
)

class PromptManager:
    """Prompt管理模块"""
    
    def __init__(self, knowledge_base: Any):
        self.knowledge_base = knowledge_base
        
    def get_skin_analysis_prompt(self, image_description: str) -> str:
        """获取皮肤分析Prompt"""
        return SKIN_ANALYSIS_PROMPT.format(
            image_description=image_description
        )
        
    def get_question_generation_prompt(self, skin_condition: Dict[str, Any]) -> str:
        """获取问题生成Prompt"""
        # 获取相关知识
        relevant_knowledge = self.knowledge_base.search(
            f"皮肤问题 {skin_condition['skin_state']} {skin_condition['blemishes']}"
        )
        
        # 构建知识上下文
        knowledge_context = "\n".join([
            f"- {doc['content']} (来源: {doc['source']})"
            for doc in relevant_knowledge
        ])
        
        return QUESTION_GENERATION_PROMPT.format(
            skin_condition=skin_condition,
            knowledge_context=knowledge_context
        )
        
    def get_profile_building_prompt(
        self,
        skin_condition: Dict[str, Any],
        user_answers: Dict[str, str]
    ) -> str:
        """获取用户画像构建Prompt"""
        # 获取相关知识
        relevant_knowledge = self.knowledge_base.search(
            f"护肤习惯 {user_answers.get('habits', '')} {skin_condition['skin_state']}"
        )
        
        # 构建知识上下文
        knowledge_context = "\n".join([
            f"- {doc['content']} (来源: {doc['source']})"
            for doc in relevant_knowledge
        ])
        
        return PROFILE_BUILDING_PROMPT.format(
            skin_condition=skin_condition,
            user_answers=user_answers,
            knowledge_context=knowledge_context
        )
        
    def get_recommendation_prompt(
        self,
        skin_condition: Dict[str, Any],
        user_profile: Dict[str, Any],
        retrieved_info: List[Dict[str, Any]]
    ) -> str:
        """获取产品推荐Prompt"""
        # 获取相关知识
        relevant_knowledge = self.knowledge_base.search(
            f"护肤产品推荐 {skin_condition['skin_state']} {user_profile['skin_type']}"
        )
        
        # 构建知识上下文
        knowledge_context = "\n".join([
            f"- {doc['content']} (来源: {doc['source']})"
            for doc in relevant_knowledge
        ])
        
        return RECOMMENDATION_PROMPT.format(
            skin_condition=skin_condition,
            user_profile=user_profile,
            retrieved_info=retrieved_info,
            knowledge_context=knowledge_context
        )
        
    def get_trust_reasoning_prompt(
        self,
        skin_condition: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> str:
        """获取信任推理Prompt"""
        # 获取相关知识
        relevant_knowledge = self.knowledge_base.search(
            f"护肤产品效果 {skin_condition['skin_state']} {user_profile['skin_type']}"
        )
        
        # 构建知识上下文
        knowledge_context = "\n".join([
            f"- {doc['content']} (来源: {doc['source']})"
            for doc in relevant_knowledge
        ])
        
        return TRUST_REASONING_PROMPT.format(
            skin_condition=skin_condition,
            recommendations=recommendations,
            user_profile=user_profile,
            knowledge_context=knowledge_context
        ) 