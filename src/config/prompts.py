"""Prompt管理模块"""

# 皮肤分析Prompt
SKIN_ANALYSIS_PROMPT = """\
分析这张面部照片的皮肤状况，必须以JSON格式返回分析结果。

请仔细、客观地观察照片，准确识别：
1. 皮肤类型(干性/油性/混合性/敏感性/中性)
2. 主要皮肤问题(痘痘/皱纹/色斑/干燥/敏感/毛孔粗大/黑头等)
3. 年龄段(青年18-35岁/中年36-55岁/老年55岁以上)
4. 性别(男性/女性) - 这是关键信息，请仔细观察面部特征、轮廓、皮肤质地等来判断性别（用于产品推荐时，但不显示在皮肤检测结果中）
5. 护肤建议（包括日常护理、饮食建议、生活习惯等）

重要：必须严格按照以下JSON格式返回，不要添加任何其他文本：

{{
    "皮肤类型": "干性/油性/混合性/敏感性/中性",
    "主要问题": ["确实存在的皮肤问题1", "确实存在的皮肤问题2"],
    "年龄段": "青年/中年/老年",
    "性别": "男性/女性",
    "详细分析": "详细皮肤分析，请完整描述皮肤状况，不要截断句子",
    "护理建议": [
        "💧 日常护理建议（根据具体皮肤状况给出个性化建议，包含具体操作步骤）",
        "🍎 饮食建议（针对检测到的具体皮肤问题推荐食物，避免通用建议）",
        "😴 睡眠建议（结合年龄和皮肤状况给出具体作息建议）",
        "🏃 生活习惯建议（根据皮肤问题给出具体的运动、压力管理建议）",
        "🌞 防晒和防护建议（根据皮肤类型和问题定制防护方案）"
    ]
}}

注意：
1. 使用专业但通俗易懂的语言描述
2. 针对中老年人皮肤特点进行分析
3. 建议要具体且可执行，包含具体的操作步骤
4. 护理建议要全面，包括护肤、饮食、睡眠、生活习惯等方面
5. 性别信息仅用于产品推荐匹配，不要在前端显示
6. 所有字段名使用中文，确保前端显示的一致性
7. 详细分析部分要完整，不要截断句子
8. 护理建议要详细且实用，每个建议至少包含具体的操作步骤和预期效果
9. 如果没有明显的皮肤问题，请如实报告，不要夸大或虚构问题
10. 主要问题数组应该只包含确实存在的皮肤问题，如果皮肤状况良好，可以是空数组
11. 请基于照片中实际可见的皮肤状况进行分析，不要推测或想象
12. 如果照片质量不佳或无法清晰看到某些细节，请如实说明
13. 必须返回纯JSON格式，不要包含```json```标记或其他文本
14. 护理建议要自然流畅，不要过于死板，可以根据皮肤状况灵活调整内容
15. 每次分析都要根据具体的皮肤状况生成不同的建议，避免模板化
16. 护理建议要体现个性化，结合用户的年龄、性别、皮肤类型和具体问题
17. 建议内容要有变化性，不要每次都使用相同的表述方式
18. 根据皮肤问题的严重程度调整建议的详细程度和针对性
19. 护理建议必须基于照片中实际检测到的皮肤状况，不要给出通用的"适合所有人"的建议
20. 每个建议都要有具体的操作指导，比如"每天早晚使用温和洁面乳"而不是"注意清洁"
21. 饮食建议要具体到食物名称，比如"多吃富含维生素C的柑橘类水果"而不是"多吃水果"
22. 生活习惯建议要结合检测到的皮肤问题，比如"避免熬夜，保证7-8小时睡眠"而不是"保持良好作息"
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
请分析以下用户消息，提取用户画像信息，包括皮肤类型和护肤关注点。

用户消息：{user_message}

请以JSON格式返回分析结果，格式如下：
```json
{
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

RECOMMENDATION_PROMPT = """
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