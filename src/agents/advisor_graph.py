from __future__ import annotations
from langgraph.graph import StateGraph, END
try:
    from langgraph.graph.graph import Graph
except ImportError:
    Graph = None
from typing import Dict, List, Any
from .skincare_agent import SkinCareAgent
import gradio as gr
import asyncio

class AdvisorGraph:
    """护肤顾问图"""

    def __init__(self):
        self.agent = SkinCareAgent()
        self.graph = self._build_graph()
        # 添加字段中文映射
        self.field_mapping = {
            "skin_state": "整体皮肤状态",
            "blemishes": "瑕疵情况",
            "pigmentation": "色素沉着",
            "wrinkles": "皱纹状况",
            "blemish_depth": "瑕疵深度",
            "blemish_texture": "瑕疵质地",
            "blemish_size": "瑕疵大小",
            "blemish_color": "瑕疵颜色",
            "blemish_scars": "疤痕情况",
            "blemish_type": "瑕疵类型",
            "blemish_location": "瑕疵位置",
            "skin_type": "皮肤类型",
            "skin_tone": "肤色",
            "skin_texture": "肤质",
            "pores": "毛孔状况",
            "hydration": "水分状况",
            "elasticity": "弹性",
            "sensitivity": "敏感程度"
        }

    def _build_graph(self) -> "Graph":
        """构建执行图"""
        # 定义节点
        async def analyze_skin(state: Dict[str, Any]) -> Dict[str, Any]:
            image = state["user_image"]
            result = await self.agent.generate_skincare_report(image)
            return {
                "skin_analysis": result["skin_analysis"],
                "confidence_scores": result["confidence_scores"]
            }

        def build_profile(state: Dict[str, Any]) -> Dict[str, Any]:
            scores = state["confidence_scores"]
            profile = self.agent.profile_builder.build_profile(
                scores,
                {}
            )
            return {"user_profile": profile}

        def generate_questions(state: Dict[str, Any]) -> Dict[str, Any]:
            scores = state["confidence_scores"]
            profile = state["user_profile"]
            questions = self.agent.prompt_generator.generate_questions(
                scores,
                profile
            )
            return {"questions": questions}

        def retrieve_info(state: Dict[str, Any]) -> Dict[str, Any]:
            skin = state["skin_analysis"]
            profile = state["user_profile"]
            query = self.agent._build_retrieval_query(skin, profile)
            info = self.agent.rag_engine.retrieve(query)
            return {"retrieved_info": info}

        def generate_recommendation(state: Dict[str, Any]) -> Dict[str, Any]:
            scores = state["confidence_scores"]
            profile = state["user_profile"]
            info = state["retrieved_info"]
            recommendations = self.agent.recommendation_engine.generate_recommendations(
                scores,
                profile,
                info
            )
            return {"recommendations": recommendations}

        def add_trust_layer(state: Dict[str, Any]) -> Dict[str, Any]:
            scores = state["confidence_scores"]
            profile = state["user_profile"]
            recommendations = state["recommendations"]
            reasoning = self.agent.trust_reasoning.generate_trust_reasoning(
                scores,
                recommendations,
                profile
            )
            return {"trust_reasoning": reasoning}

        # 构建图
        graph = Graph()

        # 添加节点
        graph.add_node("analyze_skin", analyze_skin)
        graph.add_node("build_profile", build_profile)
        graph.add_node("generate_questions", generate_questions)
        graph.add_node("retrieve_info", retrieve_info)
        graph.add_node("generate_recommendation", generate_recommendation)
        graph.add_node("add_trust_layer", add_trust_layer)

        # 添加边
        graph.add_edge("analyze_skin", "build_profile")
        graph.add_edge("build_profile", "generate_questions")
        graph.add_edge("generate_questions", "retrieve_info")
        graph.add_edge("retrieve_info", "generate_recommendation")
        graph.add_edge("generate_recommendation", "add_trust_layer")

        return graph

    async def execute(self, user_image) -> Dict[str, Any]:
        """执行图"""
        try:
            # 设置初始状态
            initial_state = {"user_image": user_image}

            # 执行图
            final_state = await self.graph.arun(initial_state)

            # 格式化输出
            return {
                "skin_analysis": self._format_skin_analysis(final_state.get("skin_analysis", {})),
                "confidence_scores": final_state.get("confidence_scores", {}),
                "user_profile": self._format_user_profile(final_state.get("user_profile", {})),
                "questions": self._format_questions(final_state.get("questions", [])),
                "recommendations": final_state.get("recommendations", []),
                "trust_reasoning": self._format_trust_reasoning(final_state.get("trust_reasoning", ""))
            }
        except Exception as e:
            return {
                "skin_analysis": f"分析出错: {str(e)}",
                "confidence_scores": {},
                "user_profile": "",
                "questions": [],
                "recommendations": [],
                "trust_reasoning": ""
            }

    def execute_sync(self, user_image) -> Dict[str, Any]:
        """同步执行分析"""
        try:
            # 直接调用VLM模型进行分析
            result = self.agent.vlm_model.predict(
                user_image,
                self.agent.prompts["skin_analysis"].format(image_description="用户上传的面部照片")
            )
            
            # 如果分析成功，添加置信度分数
            if "skin_analysis" in result and isinstance(result["skin_analysis"], dict):
                result["confidence_scores"] = {
                    "skin_type": 0.8,
                    "problems": 0.75,
                    "severity": 0.7
                }
            
            return {
                "skin_analysis": self._format_skin_analysis(result.get("skin_analysis", {})),
                "confidence_scores": result.get("confidence_scores", {}),
                "user_profile": "",  # 暂时不生成用户画像
                "questions": [],  # 暂时不生成问题
                "recommendations": [],  # 暂时不生成推荐
                "trust_reasoning": ""  # 暂时不生成推荐理由
            }
        except Exception as e:
            return {
                "skin_analysis": f"分析出错: {str(e)}",
                "confidence_scores": {},
                "user_profile": "",
                "questions": [],
                "recommendations": [],
                "trust_reasoning": ""
            }

    def _format_skin_analysis(self, analysis: Dict) -> str:
        """格式化皮肤分析结果"""
        if not analysis:
            return "未能检测到皮肤状况"
            
        if isinstance(analysis, str):
            return analysis
            
        if isinstance(analysis, dict):
            # 如果analysis是嵌套的，取skin_analysis字段
            if "skin_analysis" in analysis and isinstance(analysis["skin_analysis"], dict):
                analysis = analysis["skin_analysis"]
            
            # 移除空值和"无"值
            valid_items = {k: v for k, v in analysis.items() 
                         if v and v != "无" and v != "未检测到" and v != "none" and v != "None"}
            
            if not valid_items:
                return "未检测到明显的皮肤问题"
                
            # 格式化输出
            formatted_items = []
            for key, value in valid_items.items():
                # 使用中文映射，如果没有对应的映射就美化原键名
                display_key = self.field_mapping.get(key, key.replace("_", " ").title())
                # 确保value是字符串
                if isinstance(value, (list, tuple)):
                    value = "、".join(value)
                elif isinstance(value, dict):
                    value = "、".join(f"{k}:{v}" for k, v in value.items())
                formatted_items.append(f"▍{display_key}：{value}")
            
            if formatted_items:
                return "\n".join(formatted_items)
            
        return "未能正确解析皮肤分析结果"

    def _format_user_profile(self, profile: Dict) -> str:
        """格式化用户画像"""
        if not profile or not isinstance(profile, dict):
            return "未能生成用户画像"
        return "\n".join([f"- {k}: {v}" for k, v in profile.items()])

    def _format_questions(self, questions: list) -> list:
        """格式化问题列表"""
        if not questions or not isinstance(questions, list):
            return ["暂无补充问题"]
        return [f"{i+1}. {q}" for i, q in enumerate(questions)]

    def _format_trust_reasoning(self, reasoning: str) -> str:
        """格式化推荐理由"""
        if not reasoning or not isinstance(reasoning, str):
            return "暂无推荐理由"
        return reasoning
