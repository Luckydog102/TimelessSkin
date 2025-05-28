import gradio as gr
import os
import sys
from PIL import Image
import logging
from typing import List, Tuple, Dict, Any
import socket

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.advisor_graph import AdvisorGraph
from src.models.llm_model import LLMModel
from src.models.rag_model import RAGModel
from src.config.prompts import USER_PROFILE_PROMPT

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化模型
try:
    advisor = AdvisorGraph()  # 包含VLM模型
    llm = LLMModel()
    rag = RAGModel()
    llm.initialize()
    rag.initialize()
except Exception as e:
    logger.error(f"初始化模型失败: {e}")
    raise

def get_initial_prompt(user_type: str = None) -> str:
    """根据用户类型获取初始提示语"""
    # 统一的开场白
    opening = "您好！我是您的智能护肤顾问，您可以在右侧选择咨询类型，为您提供针对性建议～"
    
    if not user_type:
        return opening
    
    if user_type == "为自己咨询":
        return f"""{opening}

很高兴为您提供个人护肤咨询。为了更好地了解您的需求，请告诉我：
1. 您的大致年龄段
2. 您的肤质类型（如干性、油性、混合型等）
3. 您目前遇到的主要护肤困扰

您也可以上传一张面部照片，我会为您进行专业分析。"""
    elif user_type == "为长辈咨询":
        return f"""{opening}
        
很高兴您关心家人的护肤需求。为了更好地为您提供建议，请告诉我一些关于您家人的基本情况：
1. 大致年龄段
2. 主要护肤困扰
3. 是否有特殊肤质问题（如敏感、干燥等）

您也可以上传一张他们的面部照片，我来进行专业分析。"""
    elif user_type == "其他需求":
        return f"""{opening}
        
除了个人和长辈的护肤咨询外，我还可以为您提供：
1. 护肤产品成分解析
2. 护肤步骤建议
3. 季节性护肤调整方案
4. 特殊肌肤问题（如痘痘、色斑等）的处理建议

请告诉我您具体需要哪方面的帮助？"""
    else:
        # 默认回复，当用户选择的类型不在预设选项中时
        return f"""{opening}
        
请告诉我您的具体需求，我会尽力为您提供专业的护肤建议。
您可以描述您的肤质状况、年龄段、护肤困扰等信息，或者上传面部照片进行分析。"""

def analyze_user_profile(message: str) -> Dict[str, Any]:
    """分析用户画像"""
    prompt = USER_PROFILE_PROMPT.format(user_message=message)
    try:
        profile = llm.predict(prompt)
        if isinstance(profile, dict):
            return profile
        # 如果返回的不是字典，返回默认结构
        return {
            "age_group": "未知",
            "skin_type": {
                "name": "未知",
                "characteristics": "",
                "common_areas": ""
            },
            "concerns": {
                "primary": [],
                "secondary": []
            },
            "lifestyle": {
                "diet_habits": "",
                "daily_routine": "",
                "environmental_factors": ""
            }
        }
    except Exception as e:
        logger.error(f"用户画像分析失败: {e}")
        # 返回默认结构
        return {
            "age_group": "未知",
            "skin_type": {
                "name": "未知",
                "characteristics": "",
                "common_areas": ""
            },
            "concerns": {
                "primary": [],
                "secondary": []
            },
            "lifestyle": {
                "diet_habits": "",
                "daily_routine": "",
                "environmental_factors": ""
            }
        }

def get_product_recommendations(profile: Dict[str, Any], skin_analysis: str = None) -> List[Dict[str, Any]]:
    """从知识库获取产品推荐"""
    try:
        # 构建查询
        query = f"年龄段:{profile['age_group']} "
        
        if profile['skin_type']['name'] != "未知":
            query += f"肤质:{profile['skin_type']['name']} "
            
        if profile['skin_type']['characteristics']:
            query += f"特征:{profile['skin_type']['characteristics']} "
            
        if profile['concerns']['primary']:
            query += f"主要问题:{','.join(profile['concerns']['primary'])} "
            
        if skin_analysis:
            query += f"皮肤状况:{skin_analysis}"
            
        # 从知识库检索产品
        results = rag.retrieve(query, top_k=3)
        
        # 格式化推荐结果
        formatted_results = []
        for result in results:
            formatted_result = {
                "product_name": result.get("product_name", "未知产品"),
                "target_concerns": result.get("target_concerns", []),
                "key_ingredients": result.get("key_ingredients", []),
                "benefits": result.get("benefits", []),
                "usage_instructions": {
                    "frequency": result.get("usage_frequency", ""),
                    "method": result.get("usage_method", ""),
                    "timing": result.get("usage_timing", ""),
                    "precautions": result.get("precautions", "")
                },
                "suitability_reason": result.get("recommendation_reason", ""),
                "expected_results": result.get("expected_results", ""),
                "lifestyle_tips": result.get("lifestyle_tips", [])
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
    except Exception as e:
        logger.error(f"产品推荐失败: {e}")
        return []

def analyze_skin_with_vlm_and_rag(image, chat_history, state_data):
    """使用VLM和RAG进行皮肤分析和产品推荐"""
    try:
        # 1. VLM分析图片
        try:
            # 显示处理中的消息
            logger.info("开始VLM分析图片...")
            
            # 压缩图片以提高处理速度
            if image.size[0] > 800 or image.size[1] > 800:
                logger.info(f"压缩图片，原始尺寸: {image.size}")
                image.thumbnail((800, 800), Image.LANCZOS)
                logger.info(f"压缩后尺寸: {image.size}")
            
            # 调用VLM分析
            vlm_result = advisor.execute_sync(image)
            
            if not vlm_result:
                logger.error("VLM分析返回空结果")
                return "抱歉，图片分析返回空结果。请尝试上传更清晰的照片或稍后重试。"
                
        except Exception as e:
            logger.error(f"VLM分析图片失败: {str(e)}")
            return f"抱歉，图片分析过程出现错误：{str(e)}。请尝试上传更清晰的照片或稍后重试。"
            
        # 解析VLM返回的JSON结果
        skin_analysis = ""
        try:
            # 安全地提取文本内容
            if isinstance(vlm_result, dict):
                # 尝试从API返回结果中提取文本
                if 'output' in vlm_result and isinstance(vlm_result['output'], dict):
                    choices = vlm_result['output'].get('choices', [])
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get('message', {})
                        content = message.get('content', [])
                        if content and len(content) > 0:
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                text = content[0].get('text', '')
                                skin_analysis = text
                            elif isinstance(content[0], str):
                                skin_analysis = content[0]
                            else:
                                skin_analysis = str(content[0])
                elif 'skin_analysis' in vlm_result:
                    # 直接从结果中获取皮肤分析
                    skin_analysis = vlm_result.get('skin_analysis', '')
                    if not isinstance(skin_analysis, str):
                        skin_analysis = str(skin_analysis)
            
            # 如果没有提取到文本，使用原始结果
            if not skin_analysis:
                skin_analysis = str(vlm_result)
                
            logger.info(f"提取的皮肤分析文本: {skin_analysis[:100]}...")
                
        except Exception as e:
            logger.error(f"解析VLM返回结果失败: {e}")
            skin_analysis = str(vlm_result) if vlm_result else "无法解析分析结果"

        # 2. 使用LLM处理分析结果
        prompt = f"""请分析以下皮肤状况，并给出详细的护理建议：

分析结果：{skin_analysis}

请按照以下格式返回：
1. 皮肤类型
2. 主要特征
3. 问题区域
4. 护理建议
5. 推荐成分
6. 需要注意的事项"""

        try:
            logger.info("开始LLM分析皮肤状况...")
            llm_analysis = llm.predict(prompt)
            if not llm_analysis:
                llm_analysis = "无法生成分析结果，请重试。"
            logger.info("LLM分析完成")
        except Exception as e:
            logger.error(f"LLM分析失败: {e}")
            llm_analysis = f"皮肤分析处理过程中出错，但我们仍然提取了一些基本信息: {skin_analysis[:200]}..."
        
        # 3. 使用RAG在知识库中查找匹配的产品
        try:
            # 提取关键词作为查询条件
            logger.info("开始RAG查询匹配产品...")
            rag_query = f"皮肤类型: {llm_analysis[:200]}"  # 限制查询长度
            product_matches = rag.retrieve(rag_query, top_k=3)
            logger.info(f"找到匹配产品: {len(product_matches) if product_matches else 0}个")
            
            # 生成完整报告
            report = f"""🔍 皮肤分析报告：

{llm_analysis}

🏷️ 为您推荐的产品：

"""
            if product_matches and len(product_matches) > 0:
                for product in product_matches:
                    if not isinstance(product, dict):
                        continue
                        
                    name = product.get('product_name', '')
                    if not name:
                        continue
                        
                    report += f"📦 {name}\n"
                    
                    if 'key_ingredients' in product and isinstance(product['key_ingredients'], list):
                        ingredients = ', '.join(product['key_ingredients'])
                        report += f"💊 主要成分：{ingredients}\n"
                        
                    if 'benefits' in product and isinstance(product['benefits'], list):
                        benefits = ', '.join(product['benefits'])
                        report += f"✨ 功效：{benefits}\n"
                        
                    if 'usage_method' in product:
                        report += f"📝 使用方法：{product['usage_method']}\n"
                        
                    report += "\n"
            else:
                report += "暂无匹配的产品推荐。\n"
                
            return report
            
        except Exception as e:
            logger.error(f"RAG查询失败: {e}")
            return f"""🔍 皮肤分析报告：

{llm_analysis}

很抱歉，产品推荐功能暂时出现问题，请稍后再试。"""
            
    except Exception as e:
        logger.error(f"分析过程出错: {str(e)}")
        return "抱歉，皮肤分析过程中出现错误，请稍后重试。"

def safe_llm_call(message, system_message="", stream=True):
    """安全调用LLM模型，处理可能的错误"""
    try:
        if not message or not isinstance(message, str):
            return "抱歉，我无法理解您的问题。请尝试重新表述。"
            
        logger.info(f"发送到模型的消息: {message}")
        logger.info(f"系统消息: {system_message}")
        
        # 简单问候直接返回
        if message.strip().lower() in ["你好", "hello", "hi", "嗨", "您好"]:
            return "您好！我是您的智能护肤顾问。请问您今天有什么护肤方面的问题需要咨询吗？"
        
        try:
            # 构建完整的提示
            full_prompt = f"""
系统: {system_message}

用户: {message}

请以护肤顾问的身份回复上述用户问题，提供专业、友好的建议。
"""
            # 使用流式输出或普通输出
            if stream:
                return llm.chat_stream(full_prompt)
            else:
                # 使用predict方法而不是chat方法
                response = llm.predict(full_prompt)
                
                # 检查响应
                if not response or not isinstance(response, str):
                    return "抱歉，我现在遇到了一些问题，请稍后再试。"
                    
                logger.info(f"模型返回的响应: {response[:100] if len(response) > 100 else response}...")
                return response
            
        except IndexError as e:
            logger.error(f"LLM调用出现IndexError: {str(e)}")
            return "抱歉，处理您的请求时出现了问题。请尝试重新提问或换一种表述方式。"
            
        except Exception as e:
            logger.error(f"LLM调用出现异常: {str(e)}")
            return "抱歉，我现在遇到了一些技术问题，请稍后再试。"
            
    except Exception as e:
        logger.error(f"安全调用LLM时出现错误: {str(e)}")
        return "抱歉，系统暂时无法处理您的请求。请稍后再试。"

def user_message_and_response(message, chat_history, state_data):
    """处理用户消息并生成回复"""
    if not message or not isinstance(message, str):
        return "", chat_history
    
    # 立即添加用户消息到历史记录
    chat_history.append((message, None))
    
    try:
        # 构建系统消息
        system_message = "您是TimelessSkin的智能护肤顾问，请根据用户的问题提供专业、友好的护肤建议。\n\n"
        
        # 添加咨询类型上下文
        if state_data and isinstance(state_data, dict):
            if state_data.get("consultation_type"):
                system_message += f"当前咨询类型：{state_data['consultation_type']}\n"
                
                # 根据不同咨询类型添加特定指导
                if state_data['consultation_type'] == "为自己咨询":
                    system_message += "用户正在为自己咨询护肤建议。请关注个人肤质特点和护肤需求。\n"
                elif state_data['consultation_type'] == "为长辈咨询":
                    system_message += "用户正在为长辈咨询护肤建议。请关注成熟肌肤的特点和需求，提供适合年长者的护肤建议。\n"
                elif state_data['consultation_type'] == "其他需求":
                    system_message += "用户有其他护肤相关需求。请根据用户的具体问题提供相应建议。\n"
            
            # 添加皮肤分析结果（如果有）
            if state_data.get("skin_analysis"):
                system_message += f"皮肤分析结果：{state_data['skin_analysis']}\n"
                
            # 添加用户画像信息（如果有）
            if state_data.get("profile") and isinstance(state_data["profile"], dict):
                profile = state_data["profile"]
                if profile.get("age_group") and profile["age_group"] != "未知":
                    system_message += f"用户年龄段：{profile['age_group']}\n"
                    
                if profile.get("skin_type") and isinstance(profile["skin_type"], dict) and profile["skin_type"].get("name") != "未知":
                    system_message += f"用户肤质：{profile['skin_type']['name']}\n"
                    
                if profile.get("concerns") and isinstance(profile["concerns"], dict) and profile["concerns"].get("primary"):
                    if isinstance(profile["concerns"]["primary"], list) and profile["concerns"]["primary"]:
                        try:
                            concerns = ", ".join(profile["concerns"]["primary"])
                            system_message += f"用户主要护肤困扰：{concerns}\n"
                        except Exception as e:
                            logger.error(f"处理用户护肤困扰时出错: {e}")
        
        # 使用安全的LLM调用，启用流式输出
        response = safe_llm_call(message, system_message, stream=True)
        
        # 更新最后一条消息，添加助手回复
        chat_history[-1] = (message, response)
        
        # 更新用户画像（如果需要）
        try:
            if message and len(message.strip()) > 0:  # 确保消息不为空
                new_profile = analyze_user_profile(message)
                if new_profile and isinstance(new_profile, dict):
                    if not state_data.get("profile"):
                        state_data["profile"] = new_profile
                    else:
                        # 更新现有画像
                        current_profile = state_data["profile"]
                        
                        # 更新年龄段
                        if new_profile.get("age_group") and new_profile["age_group"] != "未知":
                            current_profile["age_group"] = new_profile["age_group"]
                        
                        # 更新肤质信息
                        if new_profile.get("skin_type") and isinstance(new_profile["skin_type"], dict):
                            if new_profile["skin_type"].get("name") != "未知":
                                current_profile["skin_type"].update(new_profile["skin_type"])
                        
                        # 更新护肤困扰
                        if new_profile.get("concerns") and isinstance(new_profile["concerns"], dict):
                            if new_profile["concerns"].get("primary") and isinstance(new_profile["concerns"]["primary"], list):
                                for concern in new_profile["concerns"]["primary"]:
                                    if concern not in current_profile["concerns"]["primary"]:
                                        current_profile["concerns"]["primary"].append(concern)
                                        
                            if new_profile["concerns"].get("secondary") and isinstance(new_profile["concerns"]["secondary"], list):
                                for concern in new_profile["concerns"]["secondary"]:
                                    if concern not in current_profile["concerns"]["secondary"]:
                                        current_profile["concerns"]["secondary"].append(concern)
        except Exception as e:
            logger.error(f"更新用户画像失败: {e}")
            # 继续处理，不影响主流程
        
    except Exception as e:
        logger.error(f"处理消息失败: {str(e)}")
        chat_history[-1] = (message, "抱歉，处理您的消息时出现错误，请稍后重试。")
    
    return "", chat_history

def on_analyze(image, chat_history, state_data):
    """处理图片分析"""
    if image is None:
        chat_history.append((None, "请先上传一张面部照片再进行分析。"))
        return chat_history, state_data
        
    try:
        # 添加用户请求作为一条消息并立即显示
        chat_history.append(("帮我检测肤质", None))
        
        # 显示分析中的消息（使用小船动画）
        loading_message = """
        <div class="loading-container">
            <div class="ocean">
                <div class="boat">⛵</div>
                <div class="wave"></div>
            </div>
            <div class="loading-text">正在为您检测面部照片的肤质状况，这可能需要几秒钟...</div>
        </div>
        """
        chat_history.append((None, loading_message))
        
        # 调用分析函数
        try:
            analysis_report = analyze_skin_with_vlm_and_rag(image, chat_history, state_data)
            
            # 更新用户请求的回复
            chat_history[-2] = ("帮我检测肤质", analysis_report)
            
            # 移除加载消息
            chat_history.pop()
            
            # 保存分析结果到状态
            if analysis_report and isinstance(analysis_report, str):
                state_data["skin_analysis"] = analysis_report
                
        except Exception as e:
            logger.error(f"皮肤分析处理失败: {str(e)}")
            error_message = "抱歉，皮肤分析处理失败，请稍后重试。"
            chat_history[-2] = ("帮我检测肤质", error_message)
            # 移除加载消息
            chat_history.pop()
            
        return chat_history, state_data
        
    except Exception as e:
        logger.error(f"图片分析流程失败: {e}")
        error_message = f"抱歉，图片分析服务出现错误：{str(e)}\n请稍后重试。"
        
        # 如果添加了用户请求，更新其回复
        if len(chat_history) > 0 and chat_history[-1][0] == "帮我检测肤质":
            chat_history[-1] = ("帮我检测肤质", error_message)
        else:
            # 否则直接添加错误消息
            chat_history.append((None, error_message))
            
        return chat_history, state_data

def process_interaction(
    message: str,
    history: List[Tuple[str, str]],
    user_type: str,
    image: Image.Image = None,
    state: Dict = None
) -> Tuple[str, List[Tuple[str, str]], Dict]:
    """处理用户交互"""
    try:
        if state is None:
            state = {"profile": None, "skin_analysis": None}
            
        # 处理图片分析
        if image is not None:
            try:
                result = advisor.execute_sync(image)
                if isinstance(result, dict) and "skin_analysis" in result:
                    state["skin_analysis"] = result.get("skin_analysis", "")
                    history.append((None, f"图片分析结果：\n{state['skin_analysis']}"))
                else:
                    state["skin_analysis"] = str(result)
                    history.append((None, f"图片分析结果：\n{state['skin_analysis']}"))
            except Exception as e:
                logger.error(f"图片分析失败: {e}")
                history.append((None, "抱歉，图片分析失败，请稍后重试。"))
            
        # 更新用户画像
        if message and isinstance(message, str):
            try:
                profile = analyze_user_profile(message)
                if state["profile"] is None:
                    state["profile"] = profile
                else:
                    # 更新现有画像，保持原有结构
                    current_profile = state["profile"]
                    if isinstance(profile, dict) and "age_group" in profile and profile["age_group"] != "未知":
                        current_profile["age_group"] = profile["age_group"]
                    
                    if isinstance(profile, dict) and "skin_type" in profile and isinstance(profile["skin_type"], dict) and "name" in profile["skin_type"] and profile["skin_type"]["name"] != "未知":
                        current_profile["skin_type"].update(profile["skin_type"])
                        
                    # 合并关注点
                    if isinstance(profile, dict) and "concerns" in profile and isinstance(profile["concerns"], dict):
                        if "primary" in profile["concerns"] and isinstance(profile["concerns"]["primary"], list):
                            current_profile["concerns"]["primary"].extend(
                                [c for c in profile["concerns"]["primary"] 
                                 if c not in current_profile["concerns"]["primary"]]
                            )
                        if "secondary" in profile["concerns"] and isinstance(profile["concerns"]["secondary"], list):
                            current_profile["concerns"]["secondary"].extend(
                                [c for c in profile["concerns"]["secondary"] 
                                 if c not in current_profile["concerns"]["secondary"]]
                            )
                        
                    # 更新生活方式信息
                    if isinstance(profile, dict) and "lifestyle" in profile and isinstance(profile["lifestyle"], dict) and any(profile["lifestyle"].values()):
                        current_profile["lifestyle"].update(
                            {k: v for k, v in profile["lifestyle"].items() if v}
                        )
                    
                    state["profile"] = current_profile
                    
            except Exception as e:
                logger.error(f"用户画像分析失败: {e}")
                
            # 获取产品推荐
            if state["profile"] and (state.get("skin_analysis") or len(history) > 2):
                try:
                    recommendations = get_product_recommendations(
                        state["profile"],
                        state.get("skin_analysis", "")
                    )
                    if recommendations:
                        rec_text = "根据分析，我为您推荐以下产品：\n\n"
                        for rec in recommendations:
                            if not isinstance(rec, dict):
                                continue
                                
                            rec_text += f"🏷️ {rec.get('product_name', '未知产品')}\n"
                            
                            if rec.get('target_concerns') and isinstance(rec['target_concerns'], list):
                                rec_text += f"🎯 针对问题：{', '.join(rec['target_concerns'])}\n"
                                
                            if rec.get('key_ingredients') and isinstance(rec['key_ingredients'], list):
                                rec_text += f"💊 核心成分：{', '.join(rec['key_ingredients'])}\n"
                                
                            if rec.get('benefits') and isinstance(rec['benefits'], list):
                                rec_text += f"✨ 功效：{', '.join(rec['benefits'])}\n"
                                
                            if rec.get('usage_instructions') and isinstance(rec['usage_instructions'], dict) and rec['usage_instructions'].get('method'):
                                rec_text += f"📝 使用方法：{rec['usage_instructions']['method']}\n"
                                
                            if rec.get('suitability_reason'):
                                rec_text += f"💡 推荐理由：{rec['suitability_reason']}\n"
                                
                            rec_text += "\n"
                        history.append((None, rec_text))
                except Exception as e:
                    logger.error(f"产品推荐失败: {e}")
                    
        # 生成回复
        if message and isinstance(message, str):
            try:
                # 构建系统消息
                system_context = "您是TimelessSkin的智能护肤顾问，请根据用户的问题提供专业、友好的护肤建议。\n"
                
                if user_type and isinstance(user_type, str):
                    system_context += f"当前咨询类型：{user_type}\n"
                
                if state.get("skin_analysis") and isinstance(state["skin_analysis"], str):
                    system_context += f"皮肤分析结果：{state['skin_analysis']}\n"
                
                # 使用安全的LLM调用
                response = safe_llm_call(message, system_context)
                history.append((message, response))
            except Exception as e:
                logger.error(f"对话生成失败: {e}")
                history.append((message, "抱歉，我现在遇到了一些问题，请稍后再试。"))
            
        return "", history, state
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return "", history + [(None, f"抱歉，服务出现错误：{str(e)}")], state

def create_ui():
    with gr.Blocks(css="""
        :root {
            --primary-color: #C5002E;
            --secondary-color: #F9F5F6;
            --accent-color: #E3B8B8;
            --text-color: #333333;
            --light-text: #666666;
            --border-radius-sm: 8px;
            --border-radius-lg: 12px;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 8px rgba(0,0,0,0.1);
            --transition-speed: 0.2s;
        }
        
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background-color: #F8F9FA;
            color: var(--text-color);
        }
        
        /* 响应式布局 */
        .container {
            display: flex;
            flex-direction: row;
            min-height: calc(100vh - 32px);
            max-width: 100%;
            margin: 0 auto;
            padding: 16px;
            gap: 20px;
            background: #F8F9FA;
        }
        
        @media (max-width: 992px) {
            .container {
                flex-direction: column;
            }
            .right-panel {
                max-width: 100% !important;
            }
        }
        
        /* 顶部标题栏 */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: white;
            border-radius: var(--border-radius-sm);
            margin-bottom: 16px;
            box-shadow: var(--shadow-sm);
        }
        .header h1 {
            font-size: 22px;
            font-weight: 600;
            color: var(--primary-color);
            margin: 0;
        }
        
        /* 左侧面板 - 对话区域 */
        .left-panel {
            flex: 6;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: white;
            border-radius: var(--border-radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-md);
            height: calc(100vh - 80px);
        }
        
        /* 右侧面板 */
        .right-panel {
            flex: 4;
            min-width: 300px;
            max-width: 450px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: calc(100vh - 80px);
        }
        
        /* 聊天界面 */
        .chatbot {
            flex: 1;
            min-height: 0 !important;
            height: auto !important;
            background: var(--secondary-color) !important;
            overflow-y: auto !important;
            border-radius: 0 !important;
        }
        
        /* 消息气泡样式 */
        .message {
            padding: 0 !important;
            background: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            max-width: 85% !important;
            position: relative !important;
            margin-bottom: 12px !important;
        }
        .message > div {
            padding: 14px 18px !important;
            border-radius: 18px !important;
            background: white !important;
            box-shadow: var(--shadow-sm) !important;
            overflow-wrap: break-word !important;
            white-space: pre-wrap !important;
            position: relative !important;
        }
        .message.user-message > div {
            background: #F9E8E8 !important; /* 欧莱雅浅粉色 */
            border-radius: 18px !important;
        }
        
        /* 复制按钮样式 */
        .message .copy-button {
            position: absolute !important;
            bottom: 4px !important;
            right: 4px !important;
            opacity: 0 !important;
            transition: opacity var(--transition-speed);
            padding: 4px 8px !important;
            font-size: 12px !important;
            color: #666 !important;
            background: rgba(255,255,255,0.8) !important;
            border: none !important;
            cursor: pointer !important;
            border-radius: 4px !important;
        }
        .message:hover .copy-button {
            opacity: 0.7 !important;
        }
        .message .copy-button:hover {
            opacity: 1 !important;
            background: rgba(255,255,255,0.95) !important;
        }
        
        /* 输入区域 */
        .input-area {
            padding: 16px !important;
            background: white !important;
            border-top: 1px solid #E5E5E5 !important;
        }
        .input-box textarea {
            border-radius: 24px !important;
            padding: 12px 20px !important;
            line-height: 1.5 !important;
            font-size: 15px !important;
            resize: none !important;
            min-height: 48px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            transition: all var(--transition-speed) !important;
        }
        .input-box textarea:focus {
            box-shadow: 0 2px 6px rgba(197,0,46,0.2) !important;
            border-color: var(--primary-color) !important;
        }
        
        /* 按钮样式 */
        .button-row {
            margin-top: 12px !important;
            display: flex !important;
            gap: 12px !important;
        }
        .button-row button {
            font-size: 14px !important;
            padding: 8px 20px !important;
            border-radius: 20px !important;
            transition: all var(--transition-speed) !important;
        }
        button[variant="primary"] {
            background: var(--primary-color) !important;
            color: white !important;
        }
        button[variant="primary"]:hover {
            background: #A50026 !important; /* 深红色 */
            box-shadow: 0 2px 8px rgba(197,0,46,0.3) !important;
        }
        
        /* 右侧区块样式 */
        .right-section {
            background: white;
            border-radius: var(--border-radius-lg);
            padding: 16px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 12px;
        }
        
        /* 咨询类型区块 */
        .consultation-section {
            flex: 0 0 auto;
        }
        
        /* 照片上传区块 */
        .upload-section {
            flex: 0 0 auto;
        }
        
        /* 使用说明区块 */
        .instructions-section {
            flex: 0 0 auto;
        }
        
        /* 上传区域 */
        .upload-area {
            border: 2px dashed var(--accent-color);
            border-radius: var(--border-radius-sm);
            padding: 12px;
            text-align: center;
            transition: all var(--transition-speed);
            cursor: pointer;
            height: 180px !important;
            max-height: 180px !important;
            overflow: hidden;
        }
        
        /* 分析按钮 */
        .analyze-button {
            width: 100% !important;
            background: var(--primary-color) !important;
            color: white !important;
            padding: 10px !important;
            border-radius: var(--border-radius-sm) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            margin-top: 12px !important;
            margin-bottom: 8px !important;
            transition: all var(--transition-speed) !important;
        }
        
        /* 使用说明 */
        .instructions {
            color: var(--light-text);
            font-size: 13px;
            line-height: 1.4;
        }
        .instructions ol {
            margin: 8px 0;
            padding-left: 20px;
        }
        .instructions li {
            margin-bottom: 6px;
        }
        
        /* 上传提示文字 */
        .upload-hint {
            color: var(--light-text);
            font-size: 13px;
            margin: 8px 0;
            line-height: 1.4;
        }
        
        /* 区块标题 */
        .section-title {
            font-size: 15px;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* 咨询类型提示 */
        .type-hint {
            font-size: 13px;
            color: var(--light-text);
            margin-bottom: 6px;
        }
        
        /* 咨询类型按钮 */
        .type-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .type-button {
            flex: 1;
            min-width: 90px;
            font-size: 13px !important;
            padding: 8px 12px !important;
            border: 1px solid #E0E0E0 !important;
            background: white !important;
            color: var(--text-color) !important;
            border-radius: 16px !important;
            transition: all var(--transition-speed) !important;
            text-align: center !important;
        }
    """) as demo:
        # 顶部标题栏
        with gr.Row(elem_classes="header"):
            gr.Markdown("# ✨ TimelessSkin 智能护肤顾问")
            
        with gr.Row(elem_classes="container"):
            # 左侧面板 - 聊天界面
            with gr.Column(elem_classes="left-panel"):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    elem_classes="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                    render_markdown=True,
                    height="100%"  # 修改为100%以填满父容器
                )
                with gr.Column(elem_classes="input-area"):
                    with gr.Row(elem_classes="input-container"):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="请输入您的问题...",
                            container=False,
                            elem_classes="input-box",
                            lines=1,
                            max_lines=5
                        )
                    with gr.Row(elem_classes="button-row"):
                        send = gr.Button("发送", variant="primary")
                        clear = gr.Button("清除")
            
            # 右侧面板 - 修改为单列布局
            with gr.Column(elem_classes="right-panel"):
                # 咨询类型
                with gr.Box(elem_classes="right-section consultation-section"):
                    gr.Markdown("👥 咨询类型", elem_classes="section-title")
                    gr.Markdown("您是为谁咨询?", elem_classes="type-hint")
                    with gr.Column(elem_classes="consultation-type"):
                        consultation_type = gr.Radio(
                            choices=["为自己咨询", "为长辈咨询", "其他需求"],
                            value=None,  # 默认不选择
                            label="",
                            elem_classes="type-buttons"
                        )
                
                # 照片上传
                with gr.Box(elem_classes="right-section upload-section"):
                    gr.Markdown("📸 面部照片分析", elem_classes="section-title")
                    with gr.Column():
                        image_input = gr.Image(
                            label="",
                            type="pil",
                            elem_classes="upload-area",
                            height=180
                        )
                        analyze_btn = gr.Button("开始分析", elem_classes="analyze-button", variant="primary")
                        gr.Markdown("""
                        上传面部照片后，点击"开始分析"按钮进行皮肤分析。
                        系统将自动调用智能模型识别您的皮肤状况并推荐适合的护肤产品。
                        """, elem_classes="upload-hint")
                
                # 使用说明
                with gr.Box(elem_classes="right-section instructions-section"):
                    gr.Markdown("ℹ️ 使用说明", elem_classes="section-title")
                    gr.Markdown("""
                    1. 选择咨询类型
                    2. 上传照片或直接对话
                    3. 根据提示回答问题
                    4. 获取个性化护肤建议
                    
                    • 照片越清晰，分析越准确
                    • 可以随时更换咨询类型
                    • 有疑问可直接在对话框提问
                    """, elem_classes="instructions")

            # 状态存储
            state = gr.State({
                "consultation_type": None,
                "skin_analysis": None,
                "profile": None
            })

            # 事件处理
            def on_select_type(choice, chat_history, state_data):
                """处理咨询类型选择"""
                try:
                    if not isinstance(chat_history, list):
                        chat_history = []
                    
                    if not isinstance(state_data, dict):
                        state_data = {"consultation_type": None, "skin_analysis": None, "profile": None}
                    
                    # 如果没有选择，则显示默认开场白
                    if not choice:
                        initial_message = get_initial_prompt()
                        chat_history = [(None, initial_message)]
                        state_data["consultation_type"] = None
                        return chat_history, state_data
                    
                    # 更新状态
                    state_data["consultation_type"] = choice
                    
                    # 将用户选择作为一条消息添加到对话历史
                    user_message = f"我想{choice}"
                    chat_history.append((user_message, None))
                    
                    # 获取对应的回复
                    response = get_initial_prompt(choice)
                    
                    # 更新消息回复
                    chat_history[-1] = (user_message, response)
                    
                    return chat_history, state_data
                    
                except Exception as e:
                    logger.error(f"处理咨询类型选择失败: {e}")
                    # 返回一个基本的错误提示
                    error_message = "抱歉，处理您的选择时出现错误，请重试。"
                    return [(None, error_message)], {"consultation_type": None, "skin_analysis": None, "profile": None}

            # 页面加载时自动触发开场白
            demo.load(
                lambda: ([(None, get_initial_prompt())], {"consultation_type": None, "skin_analysis": None, "profile": None}),
                inputs=None,
                outputs=[chatbot, state]
            )

            # 更新事件处理
            consultation_type.change(
                on_select_type,
                inputs=[consultation_type, chatbot, state],
                outputs=[chatbot, state]
            )

            # 分析按钮事件
            analyze_btn.click(
                on_analyze,
                inputs=[image_input, chatbot, state],
                outputs=[chatbot, state]
            )

            # 消息发送事件
            msg.submit(
                user_message_and_response,
                [msg, chatbot, state],
                [msg, chatbot]
            )
            send.click(
                user_message_and_response,
                [msg, chatbot, state],
                [msg, chatbot]
            )
            clear.click(lambda: [], None, chatbot)

    return demo

if __name__ == "__main__":
    # 获取可用端口
    def find_free_port(start=7860, end=7900):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    continue
        return None
        
    # 获取可用端口
    port = find_free_port()
    if not port:
        print("无法找到可用端口，尝试使用默认端口7891")
        port = 7891
        
    print(f"正在启动服务，端口: {port}")
    
    # 创建UI并启动服务
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=port, share=False) 