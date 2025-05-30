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
    # 首先初始化LLM模型
    llm = LLMModel()
    llm.initialize()
    logger.info("LLM模型初始化成功")
    
    try:
        # 尝试初始化RAG模型
        rag = RAGModel()
        rag.initialize()
        logger.info("RAG模型初始化成功")
    except Exception as e:
        logger.warning(f"RAG模型初始化失败，将以基础对话模式运行: {e}")
        rag = None
    
    try:
        # 尝试初始化Advisor
        advisor = AdvisorGraph()
        logger.info("Advisor初始化成功")
    except Exception as e:
        logger.warning(f"Advisor初始化失败，部分功能可能受限: {e}")
        advisor = None
    
except Exception as e:
    logger.error(f"LLM模型初始化失败: {e}")
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
            greeting = "您好！我是您的智能护肤顾问。请问您今天有什么护肤方面的问题需要咨询吗？"
            if stream:
                for char in greeting:
                    yield char
            else:
                return greeting
        
        try:
            # 构建完整的提示
            full_prompt = f"""
系统: {system_message}

用户: {message}

请以护肤顾问的身份回复上述用户问题，提供专业、友好的建议。
"""
            if stream:
                # 使用流式输出
                for chunk in llm.chat_stream(message=full_prompt, system_message="", temperature=0.7):
                    yield chunk
            else:
                # 使用非流式输出
                return llm.chat(message=full_prompt, system_message="", temperature=0.7)
            
        except Exception as e:
            logger.error(f"LLM调用出现异常: {str(e)}")
            error_msg = "抱歉，我现在遇到了一些技术问题，请稍后再试。"
            if stream:
                for char in error_msg:
                    yield char
            else:
                return error_msg
            
    except Exception as e:
        logger.error(f"安全调用LLM时出现错误: {str(e)}")
        error_msg = "抱歉，系统暂时无法处理您的请求。请稍后再试。"
        if stream:
            for char in error_msg:
                yield char
        else:
            return error_msg

def user_message_and_response(message, chat_history, state_data):
    """处理用户消息并生成回复"""
    if not message or not isinstance(message, str):
        return "", chat_history
    
    # 立即添加用户消息到历史记录
    chat_history.append((message, ""))
    
    try:
        # 构建系统消息
        system_message = "您是TimelessSkin的智能护肤顾问，请根据用户的问题提供专业、友好的护肤建议。\n\n"
        
        # 添加咨询类型上下文
        if state_data and isinstance(state_data, dict):
            if state_data.get("consultation_type"):
                system_message += f"当前咨询类型：{state_data['consultation_type']}\n"
            if state_data.get("skin_analysis"):
                system_message += f"皮肤分析结果：{state_data['skin_analysis']}\n"
            if state_data.get("profile"):
                profile = state_data["profile"]
                if profile.get("age_group"):
                    system_message += f"用户年龄段：{profile['age_group']}\n"
                if profile.get("skin_type", {}).get("name"):
                    system_message += f"用户肤质：{profile['skin_type']['name']}\n"
        
        # 使用安全的LLM调用，启用流式输出
        response_generator = safe_llm_call(message, system_message, stream=True)
        
        # 流式输出响应
        current_response = ""
        for chunk in response_generator:
            current_response += chunk
            chat_history[-1] = (message, current_response)
            yield "", chat_history
            
        # 更新用户画像
        try:
            if message and len(message.strip()) > 0:
                new_profile = analyze_user_profile(message)
                if new_profile and isinstance(new_profile, dict):
                    if not state_data.get("profile"):
                        state_data["profile"] = new_profile
                    else:
                        # 更新现有画像
                        current_profile = state_data["profile"]
                        if new_profile.get("age_group") != "未知":
                            current_profile["age_group"] = new_profile["age_group"]
                        if new_profile.get("skin_type", {}).get("name") != "未知":
                            current_profile["skin_type"].update(new_profile["skin_type"])
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
                
                # 添加用户消息到历史记录
                history.append((message, ""))
                
                # 使用安全的LLM调用，启用流式输出
                response_generator = safe_llm_call(message, system_context, stream=True)
                
                # 流式输出响应
                for chunk in response_generator:
                    # 更新最后一条消息，追加新的文本块
                    history[-1] = (message, history[-1][1] + chunk)
                    yield "", history, state
                
            except Exception as e:
                logger.error(f"对话生成失败: {e}")
                history.append((message, "抱歉，我现在遇到了一些问题，请稍后再试。"))
            
        return "", history, state
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return "", history + [(None, f"抱歉，服务出现错误：{str(e)}")], state

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

def create_ui():
    with gr.Blocks(css="""
        .chatbot { 
            height: calc(100vh - 120px) !important; 
            overflow-y: auto !important;
        }
        .gr-button { 
            border-radius: 10px !important;
            font-weight: bold !important; 
            background: #6B5BFF !important; 
            color: white !important;
            transition: 0.3s ease !important;
            border: none !important;
            padding: 8px 16px !important;
        }
        .gr-button:hover {
            background: #4B3BDD !important;
            transform: translateY(-1px) !important;
        }
        .gr-box {
            border-radius: 16px !important;
            box-shadow: 0 0 12px rgba(0,0,0,0.06) !important;
            padding: 16px !important;
            margin-bottom: 16px !important;
            background: white !important;
        }
        .section-title {
            font-weight: bold !important;
            font-size: 18px !important;
            margin-bottom: 12px !important;
            color: #333 !important;
        }
        .upload-area {
            border: 2px dashed #6B5BFF !important;
            border-radius: 12px !important;
            padding: 20px !important;
            text-align: center !important;
            background: #F8F8FF !important;
            min-height: 140px !important;
            transition: all 0.3s ease !important;
        }
        .upload-area:hover {
            border-color: #4B3BDD !important;
            background: #F0F0FF !important;
        }
        .type-buttons {
            display: flex !important;
            justify-content: space-between !important;
            gap: 8px !important;
            margin-bottom: 16px !important;
            width: 100% !important;
        }
        .type-buttons > div {
            display: flex !important;
            width: 100% !important;
            gap: 8px !important;
        }
        .type-buttons label {
            flex: 1 1 0 !important;
            background: #F8F8FF !important;
            border: 2px solid #E0E0FF !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            transition: all 0.3s ease !important;
            text-align: center !important;
            white-space: nowrap !important;
            font-size: 14px !important;
            min-width: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .type-buttons label:hover {
            border-color: #6B5BFF !important;
            background: #F0F0FF !important;
        }
        .type-buttons label[data-selected="true"] {
            background: #6B5BFF !important;
            color: white !important;
            border-color: #6B5BFF !important;
        }
        .message {
            padding: 12px !important;
            margin-bottom: 8px !important;
        }
        .message > div {
            padding: 12px 16px !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
            background: white !important;
        }
        .message.user-message > div {
            background: #F0F0FF !important;
            border: 1px solid #E0E0FF !important;
        }
        .input-box textarea {
            border: 2px solid #E0E0FF !important;
            border-radius: 12px !important;
            padding: 12px !important;
            background: white !important;
            transition: all 0.3s ease !important;
            margin: 0 !important;
            height: 45px !important;
            min-height: 45px !important;
            resize: none !important;
        }
        .input-box textarea:focus {
            border-color: #6B5BFF !important;
            box-shadow: 0 0 0 3px rgba(107,91,255,0.1) !important;
        }
        .input-row {
            display: flex !important;
            gap: 12px !important;
            padding: 16px !important;
            background: white !important;
            border-top: 1px solid #E0E0FF !important;
            align-items: center !important;
        }
        .input-row > div {
            margin: 0 !important;
        }
        .button-group {
            display: flex !important;
            justify-content: flex-end !important;
        }
        .button-group .gr-row {
            gap: 8px !important;
        }
        .button-group button {
            min-width: unset !important;
            padding: 0 16px !important;
            height: 36px !important;
            font-size: 14px !important;
        }
        .full-width-button {
            width: 100% !important;
            margin: 12px 0 !important;
            height: 40px !important;
            border-radius: 12px !important;
        }
    """) as demo:
        gr.Markdown("## ✨ TimelessSkin 智能护肤顾问")

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                    render_markdown=True
                )
                with gr.Row(elem_classes="input-row"):
                    with gr.Column(scale=4):  # 输入框占更多空间
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="请输入您的问题...",
                            container=False,
                            elem_classes="input-box"
                        ).style(container=False)
                    with gr.Column(scale=1, elem_classes="button-group"):  # 按钮组
                        with gr.Row():
                            send = gr.Button("发送", variant="primary", size="sm")
                            clear = gr.Button("清除", size="sm")

            with gr.Column(scale=3):
                with gr.Box():
                    gr.Markdown("👥 **咨询类型**", elem_classes="section-title")
                    consultation_type = gr.Radio(
                        ["为自己咨询", "为长辈咨询", "其他需求"],
                        label=None,
                        container=False,
                        elem_classes="type-buttons"
                    )

                with gr.Box():
                    gr.Markdown("📸 **面部照片分析**", elem_classes="section-title")
                    image_input = gr.Image(
                        label=None,
                        type="pil",
                        elem_classes="upload-area"
                    )
                    analyze_btn = gr.Button("开始分析", variant="primary", elem_classes="full-width-button")
                    gr.Markdown('上传照片后点击"开始分析"进行皮肤分析')

                with gr.Box():
                    gr.Markdown("ℹ️ **使用说明**", elem_classes="section-title")
                    gr.Markdown("""
                    • 选择咨询类型获取针对性建议
                    • 上传照片可进行皮肤分析
                    • 直接对话获取护肤建议
                    • 照片越清晰，分析越准确
                    """)

        # 状态存储
        state = gr.State({
            "consultation_type": None,
            "skin_analysis": None,
            "profile": None
        })

        # 页面加载时自动触发开场白
        demo.load(
            lambda: ([(None, get_initial_prompt())], {"consultation_type": None, "skin_analysis": None, "profile": None}),
            inputs=None,
            outputs=[chatbot, state]
        )

        # 事件处理
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

        # 消息发送事件（支持按钮点击和回车发送）
        msg.submit(
            user_message_and_response,
            [msg, chatbot, state],
            [msg, chatbot],
            queue=True,
            api_name=None
        ).then(
            lambda: None,
            None,
            [msg],
            queue=False
        )
        
        send.click(
            user_message_and_response,
            [msg, chatbot, state],
            [msg, chatbot],
            queue=True,
            api_name=None
        ).then(
            lambda: None,
            None,
            [msg],
            queue=False
        )
        # 清除聊天记录
        clear.click(lambda: None, None, chatbot)  # 返回None来清除聊天记录

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
    demo.queue()  # 启用队列
    demo.launch(server_name="127.0.0.1", server_port=port, share=False) 