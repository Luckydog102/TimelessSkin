from typing import Dict, List, Any
from ..models.llm_model import LLMModel
from ..config.prompts import QUESTION_GENERATION_PROMPT

class QuestionGenerator:
    """问题生成器"""
    
    def __init__(self):
        self.gpt_model = LLMModel()
        self.gpt_model.initialize()
        
    def generate_questions(self,
                          skin_conditions: Dict[str, Any],
                          knowledge_context: Dict[str, Any]) -> List[str]:
        """生成问题"""
        # 构建问题生成提示
        
        prompt = self._build_question_prompt(skin_conditions, knowledge_context)
        
        # 获取GPT生成的问题
        questions = self.gpt_model.predict(prompt)
        
        # 解析问题列表
        return self._parse_questions(questions)
        
    def _build_question_prompt(self,
                            skin_conditions: Dict[str, float],
                            knowledge_context: Dict[str, Any]) -> str:
        """使用预定义 prompt 构建问题生成提示"""
        confidence_scores = skin_conditions.get("confidence_scores", {})
        skin_analysis = skin_conditions.get("skin_analysis", {})

        score_str = self._format_conditions(confidence_scores)
        description_str = self._format_skin_analysis(skin_analysis)
        skin_condition=f"{description_str}\n\n评分如下：\n{score_str}"
        knowledge_context_str = self._extract_knowledge(knowledge_context)

        prompt = QUESTION_GENERATION_PROMPT.format(
            skin_condition=skin_condition,
            knowledge_context=knowledge_context_str
        )
        return prompt
        
    def _format_conditions(self, conditions: Dict[str, float]) -> str:
        """格式化皮肤状况"""
        return "\n".join([f"- {k}: {v:.2f}" for k, v in conditions.items()])
        
    # def _format_profile(self, profile: Dict[str, Any]) -> str:
    #     """格式化用户画像"""
    #     return "\n".join([f"- {k}: {v}" for k, v in profile.items()])
    def _format_skin_analysis(self, skin_analysis: Dict[str, Any]) -> str:
        """将结构化皮肤分析信息转换为自然语言上下文"""
        fields_order = [
            "skin_state",
            "blemishes",
            "pigmentation",
            "wrinkles",
            "blemish_depth",
            "blemish_texture",
            "blemish_size",
            "blemish_color",
            "blemish_scars",
            "blemish_type",
            "blemish_location"
        ]
        lines = []
        for field in fields_order:
            value = skin_analysis.get(field)
            if value:
                lines.append(f"{field}：{value}")
        return "\n".join(lines)
    
    def _extract_knowledge(self, knowledge_context) -> str:
        mock_data = """
用户护肤习惯：不常卸妆，经常熬夜
用户使用过的护肤品：理肤泉祛痘凝胶，芙丽芳丝洁面
"""
        return mock_data
        
    def _parse_questions(self, questions: Any) -> List[str]:
        """解析模型返回的 JSON 格式问题列表"""
        if isinstance(questions, list):
            return [q.get("content", "").strip() for q in questions if isinstance(q, dict)]
        elif isinstance(questions, str):
            return questions.strip().split("\n")
        else:
            return ["⚠️ 无法解析问题格式"]
    
if __name__ == "__main__":
    # 初始化
    qg = QuestionGenerator()

    # 模拟皮肤状况（模型输出后结构化的结果）
    skin_conditions = {
        "acne": 0.9,
        "oiliness": 0.7,
        "pigmentation": 0.4
    }

    # knowledge_context 暂时传空（从 mock 中生成）
    knowledge_context = {}

    # 生成问题
    try:
        questions = qg.generate_questions(skin_conditions, knowledge_context)

        print("✅ 问题生成成功！输出如下：")
        for q in questions:
            print("-", q)
    except Exception as e:
        print("❌ 问题生成失败：")
        print(str(e))