"""Prompt管理模块"""

# 皮肤分析Prompt
SKIN_ANALYSIS_PROMPT = """\
请作为专业的皮肤科医生，仔细分析这张面部照片，并以JSON格式输出分析结果。

图片描述：{image_description}

请重点关注以下方面，并按照知识库结构进行分析：
1. 皮肤类型识别
2. 皮肤特征描述
3. 问题区域定位
4. 护肤建议
5. 成分建议
6. 注意事项

请以JSON格式输出，示例：
```json
{{
    "skin_type": {{
        "name": "皮肤类型名称",
        "characteristics": "皮肤特征描述",
        "common_areas": "问题常见部位"
    }},
    "skincare_advice": {{
        "key_points": [
            "护肤要点1",
            "护肤要点2",
            "护肤要点3"
        ],
        "diet_suggestions": [
            "饮食建议1",
            "饮食建议2",
            "饮食建议3"
        ],
        "lifestyle_habits": "生活习惯建议"
    }},
    "ingredients": {{
        "recommended": ["推荐成分1", "推荐成分2", "推荐成分3"],
        "avoid": ["避免成分1", "避免成分2", "避免成分3"]
    }},
    "precautions": "使用注意事项"
}}
```

注意：
1. 使用专业但通俗易懂的语言描述
2. 针对中老年人皮肤特点进行分析
3. 建议要具体且可执行
4. 成分建议要安全可靠
"""

# 打分prompt
CONDITION_SCORE_PROMPT = """
你是一位皮肤科专家助手。请根据以下当前对象皮肤分析文本，为每项常见皮肤问题给出严重程度评分，范围从 0 到 1（保留一位小数）。

皮肤分析内容：
{analysis_text}

请严格以 JSON 格式输出，字段包括：
- acne
- wrinkles
- pigmentation
- dryness
- oiliness
- scars

示例格式：
```json
{{
    "acne": 0.8,
    "wrinkles": 0.2,
    "pigmentation": 0.6,
    "dryness": 0.3,
    "oiliness": 0.7,
    "scars": 0.1
}}
"""
# 问题生成Prompt
QUESTION_GENERATION_PROMPT = """
基于以下皮肤状况和知识背景，生成3-5个相关的跟进问题：

皮肤状况：{skin_condition}
知识背景：{knowledge_context}

请以JSON格式返回问题列表，格式如下：
```json
[
    "问题1",
    "问题2",
    "问题3"
]
```
"""

# 用户画像Prompt
USER_PROFILE_PROMPT = """
请分析以下用户消息，提取用户画像信息，包括年龄段、皮肤类型和护肤关注点。

用户消息：{user_message}

请以JSON格式返回分析结果，格式如下：
```json
{
    "age_group": "年龄段（中年/老年）",
    "skin_type": {
        "name": "皮肤类型名称",
        "characteristics": "皮肤特征描述",
        "common_areas": "问题常见部位"
    },
    "concerns": {
        "primary": ["主要护肤问题1", "主要护肤问题2"],
        "secondary": ["次要护肤问题1", "次要护肤问题2"]
    },
    "lifestyle": {
        "diet_habits": "饮食习惯",
        "daily_routine": "日常作息",
        "environmental_factors": "环境因素"
    }
}
```
"""

# 产品推荐Prompt
PRODUCT_RECOMMENDATION_PROMPT = """
基于以下用户信息和皮肤状况，从知识库中选择最合适的护肤产品：

用户画像：
{user_profile}

皮肤状况：
{skin_condition}

请以JSON格式返回推荐产品，格式如下：
```json
{
    "recommendations": [
        {
            "product_name": "产品名称",
            "target_concerns": ["针对的护肤问题1", "针对的护肤问题2"],
            "key_ingredients": ["核心成分1", "核心成分2"],
            "benefits": ["功效1", "功效2"],
            "usage_instructions": {
                "frequency": "使用频率",
                "method": "使用方法",
                "timing": "使用时间",
                "precautions": "使用注意事项"
            },
            "suitability_reason": "适合原因",
            "expected_results": "预期效果",
            "lifestyle_tips": ["配合使用建议1", "配合使用建议2"]
        }
    ],
    "general_advice": {
        "skincare_routine": ["护肤步骤1", "护肤步骤2"],
        "diet_suggestions": ["饮食建议1", "饮食建议2"],
        "lifestyle_recommendations": ["生活建议1", "生活建议2"]
    }
}
```

注意事项：
1. 推荐产品必须适合中老年人使用
2. 考虑用户的具体皮肤类型和问题
3. 建议要具体且易于执行
4. 使用说明要简单明了
5. 注重产品的安全性和温和度
"""

# 中老年人产品推荐Prompt
ELDER_RECOMMENDATION_PROMPT = """
基于以下信息生成适合中老年人的护肤产品推荐：

皮肤状况：
{skin_condition}

用户画像：
{user_profile}

相关产品信息：
{retrieved_info}

相关知识：
{knowledge_context}

预算范围：
{budget}

期望改善的问题：
{improvement_goals}

请以JSON格式输出：

```json
{
    "personalized_recommendations": [
        {
            "product_name": "产品名称",
            "brand": "品牌名称",
            "price_range": "价格区间",
            "target_concerns": ["针对问题1", "针对问题2"],
            "key_ingredients": {
                "active": ["主要活性成分1", "主要活性成分2"],
                "supportive": ["辅助成分1", "辅助成分2"]
            },
            "benefits": ["功效1", "功效2"],
            "usage_guide": {
                "steps": ["使用步骤1", "使用步骤2"],
                "frequency": "使用频率",
                "timing": "使用时间",
                "notes": "特别说明"
            },
            "safety_features": ["安全特点1", "安全特点2"],
            "texture_description": "质地描述",
            "ease_of_use_rating": "使用便捷度（1-5分）",
            "expected_results": {
                "short_term": "短期效果",
                "long_term": "长期效果",
                "timeframe": "预期见效时间"
            }
        }
    ],
    "routine_suggestions": {
        "morning": ["早晨步骤1", "早晨步骤2"],
        "evening": ["晚间步骤1", "晚间步骤2"]
    },
    "lifestyle_advice": {
        "skincare_habits": ["护肤习惯建议1", "护肤习惯建议2"],
        "diet_tips": ["饮食建议1", "饮食建议2"],
        "daily_practices": ["日常注意事项1", "日常注意事项2"]
    },
    "precautions": ["注意事项1", "注意事项2"]
}
```

注意事项：
1. 所有建议必须特别适合中老年人使用
2. 产品选择要考虑安全性和温和度
3. 使用说明要简单易懂
4. 考虑预算范围和改善目标
5. 特别关注抗衰老、保湿和舒适度
6. 避免复杂的多步骤使用流程
7. 考虑中老年人皮肤的特殊需求
"""

# 信任推理Prompt
TRUST_REASONING_PROMPT = """\
基于以下信息生成推荐理由：

皮肤状况：
{skin_condition}

推荐产品：
{recommendations}

用户画像：
{user_profile}

相关知识：
{knowledge_context}

请详细解释为什么这些产品适合用户，以JSON格式输出：

```json
{
    "problem_solution": "产品如何解决用户的皮肤问题",
    "scientific_basis": "产品成分的科学依据",
    "usage_guidance": "使用建议和注意事项",
    "expected_results": {
        "short_term": "短期效果预期",
        "long_term": "长期效果预期",
        "timeline": "预期时间线"
    }
}
```
""" 