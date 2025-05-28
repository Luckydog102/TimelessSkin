from typing import Dict, List, Any
from ..models.vlm_model import VLMModel
from ..engines.rag_engine import RAGEngine
from PIL import Image
import gradio as gr
from ..config.prompts import SKIN_ANALYSIS_PROMPT

class SkinCareAgent:
    """护肤顾问Agent"""
    
    def __init__(self):
        # 初始化VLM模型和RAG引擎
        self.vlm_model = VLMModel()
        self.vlm_model.initialize()
        self._load_config()
        self.rag_engine = RAGEngine(self.config)
        self.prompts = {
            "skin_analysis": SKIN_ANALYSIS_PROMPT
        }

    def _load_config(self):
        """加载配置"""
        from pathlib import Path
        import os
        
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent.parent
        
        self.config = {
            "knowledge_base": {
                "path": str(current_dir / "knowledge")
            }
        }
        
        # 确保知识库目录存在
        os.makedirs(self.config["knowledge_base"]["path"], exist_ok=True)

    async def generate_skincare_report(self, user_image) -> Dict[str, Any]:
        """生成护肤报告"""
        try:
            # 如果传入的是字符串路径，则打开图片
            if isinstance(user_image, str):
                image = Image.open(user_image)
            else:
                # 否则假设它是PIL.Image对象
                image = user_image
                
            # 分析皮肤状态
            result = self.vlm_model.predict(
                image,
                self.prompts["skin_analysis"].format(image_description="用户上传的面部照片")
            )

            # 如果分析成功，添加置信度分数并获取产品推荐
            if "skin_analysis" in result and isinstance(result["skin_analysis"], dict):
                result["confidence_scores"] = {
                    "skin_type": 0.8,
                    "problems": 0.75,
                    "severity": 0.7
                }
                
                # 获取产品推荐
                try:
                    query = f"皮肤状况: {', '.join(result['skin_analysis'].keys())}"
                    recommendations = self.rag_engine.retrieve(query, top_k=3)
                    result["recommendations"] = recommendations
                except Exception as e:
                    print(f"获取产品推荐失败: {e}")
                    result["recommendations"] = []

            # 返回结果
            return result
            
        except Exception as e:
            raise gr.Error(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    image_path = "cases/acne_faces/1.jpg"  # 请替换为你本地真实存在的测试图片路径
    agent = SkinCareAgent()
    
    async def run():
        result = await agent.generate_skincare_report(image_path)
        print("分析结果:", result)
        
    import asyncio
    asyncio.run(run())