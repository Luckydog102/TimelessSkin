from typing import Dict, Any, Optional
from ..models.llm_model import LLMModel
from ..config.prompts import PROFILE_BUILDING_PROMPT
import json
import re
import pdb

class UserProfileBuilder:
    """用户画像构建器"""

    def __init__(self):
        self.gpt_model = LLMModel()
        self.gpt_model.initialize()

    def build_profile(
        self,
        skin_conditions: Dict[str, float],
        user_answers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """构建用户画像"""
        user_answers = user_answers or {}
        
        prompt = self._build_profile_prompt(skin_conditions, user_answers)
        profile = self.gpt_model.predict(prompt)
        
        return self._parse_profile(profile)

    def _build_profile_prompt(
        self,
        skin_conditions: Dict[str, float],
        user_answers: Dict[str, str]
    ) -> str:
        return PROFILE_BUILDING_PROMPT.format(
            skin_condition=self._format_conditions(skin_conditions),
            user_answers=self._format_answers(user_answers),
            knowledge_context="基于皮肤状况和用户回答的上下文信息"
        )

    def _format_conditions(self, conditions: Dict[str, float]) -> str:
        return "\n".join([f"- {k}: {v:.2f}" for k, v in conditions.items()])

    def _format_answers(self, answers: Dict[str, str]) -> str:
        if not answers:
            return "暂无用户回答"
        return "\n".join([f"Q: {k}\nA: {v}" for k, v in answers.items()])

    def _parse_profile(self, profile: Any) -> Dict[str, Any]:
        """解析画像结果，自动处理字符串或字典输入"""
        if isinstance(profile, dict):
            return profile

        try:
            # 提取 JSON（处理 LLM 输出包裹在 ```json 中的情况）
            import re
            match = re.search(r"```json\s*(\{.*?\})\s*```", profile, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)

            # 尝试直接解析
            return json.loads(profile)

        except json.JSONDecodeError as e:
            print("⚠️ JSON解析失败，原始内容如下：\n")
            print(profile)
            print("\n⚠️ 错误详情：", str(e))
            return {"raw_profile": profile}
        
if __name__ == "__main__":
    
    builder = UserProfileBuilder()

    # 模拟你的 condition 数据
    skin_condition_data = {
        'skin_analysis': {
            'skin_state': '皮肤状态中等，存在一定的瑕疵和色素沉着。',
            'blemishes': '面部有明显的痤疮或痘印痕迹。',
            'pigmentation': '局部区域（如脸颊和额头）有轻微的色素沉着现象。',
            'wrinkles': '无明显皱纹，皮肤较为平整。',
            'blemish_depth': '中等深度，部分瑕疵较深。',
            'blemish_texture': '粗糙，表面不平滑。',
            'blemish_size': '大小不一，从较小的点状到较大的斑块状。',
            'blemish_color': '颜色偏暗，呈现红褐色或棕色。',
            'blemish_scars': '部分瑕疵处有轻微疤痕，但不明显。',
            'blemish_type': '痤疮后遗症（痘印、痘坑）。',
            'blemish_location': '主要分布在额头、脸颊和下巴区域。'
        },
        'confidence_scores': {
            'acne': 0.7,
            'wrinkles': 0.1,
            'pigmentation': 0.5,
            'dryness': 0.3,
            'oiliness': 0.4,
            'scars': 0.3
        }
    }

    # 构造测试
    skin_scores = skin_condition_data["confidence_scores"]
    user_answers = {}  # 初步测试为空回答

    print("🔍 构建用户画像...")
    try:
        profile = builder.build_profile(skin_scores, user_answers)
        print("✅ 用户画像结果：\n")
        for k, v in profile.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("❌ 失败：", str(e))