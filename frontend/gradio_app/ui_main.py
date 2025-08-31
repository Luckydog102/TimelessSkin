import logging
import socket
import json
import re
import time
from typing import Dict, Any, List, Tuple
from PIL import Image
import gradio as gr

# 添加项目根目录到Python路径
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.advisor_graph import AdvisorGraph
from src.models.llm_model import LLMModel
from src.models.rag_model import RAGModel
from src.engines.recommendation_engine import RecommendationEngine
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
        
        # 检查RAG模型状态
        if hasattr(rag, '_initialized'):
            logger.info(f"RAG初始化状态: {rag._initialized}")
        if hasattr(rag, 'all_products'):
            logger.info(f"RAG产品数据数量: {len(rag.all_products.get('products', []))}")
        if hasattr(rag, 'elder_products'):
            logger.info(f"RAG老年人产品数据数量: {len(rag.elder_products.get('products', []))}")
    except Exception as e:
        logger.warning(f"RAG模型初始化失败，将以基础对话模式运行: {e}")
        import traceback
        logger.error(f"RAG初始化错误堆栈: {traceback.format_exc()}")
        rag = None
        
    try:
        # 尝试初始化Advisor
        logger.info("开始初始化Advisor...")
        advisor = AdvisorGraph()
        logger.info("Advisor初始化成功")
    except Exception as e:
        logger.error(f"Advisor初始化失败，详细错误: {str(e)}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        advisor = None
        
    try:
        # 初始化推荐引擎
        recommender = RecommendationEngine()
        logger.info("RecommendationEngine初始化成功")
    except Exception as e:
        logger.warning(f"RecommendationEngine初始化失败: {e}")
        recommender = None
        
except Exception as e:
    logger.error(f"LLM模型初始化失败: {e}")
    raise

def get_initial_prompt(user_type: str = None) -> str:
    """根据用户类型获取初始提示语"""
    opening = "您好！我是您的智能护肤顾问，您可以选择咨询类型，为您提供针对性建议～"
    
    if not user_type:
        return opening
    
    if user_type == "为自己咨询":
        return (
            "很高兴为您提供个人护肤咨询。为了更好地了解您的需求，请告诉我：\n"
            "1. 您的肤质类型（如干性、油性、混合型等）\n"
            "2. 您目前遇到的主要护肤困扰\n\n"
            "您也可以上传一张面部照片，我会为您进行专业分析。"
        )
    elif user_type == "为长辈咨询":
        return (
            "很高兴您关心家人的护肤需求。为了更好地为您提供建议，请告诉我一些关于您家人的基本情况：\n"
            "1. 主要护肤困扰\n"
            "2. 是否有特殊肤质问题（如敏感、干燥等）\n\n"
            "您也可以上传一张他们的面部照片，我来进行专业分析。"
        )
    elif user_type == "其他需求":
        return (
            "除了个人和长辈的护肤咨询外，我还可以为您提供：\n"
            "1. 护肤产品成分解析\n"
            "2. 护肤步骤建议\n"
            "3. 季节性护肤调整方案\n"
            "4. 特殊肌肤问题（如痘痘、色斑等）的处理建议\n\n"
            "请告诉我您具体需要哪方面的帮助？"
        )
    else:
        return (
            "请告诉我您的具体需求，我会尽力为您提供专业的护肤建议。\n"
            "您可以描述您的肤质状况、护肤困扰等信息，或者上传面部照片进行分析。"
        )

def analyze_user_profile(message: str) -> Dict[str, Any]:
    """分析用户画像（移除年龄组信息）"""
    prompt = USER_PROFILE_PROMPT.format(user_message=message)
    try:
        profile = llm.predict(prompt)
        logger.info(f"LLM predict返回类型: {type(profile)}")
        logger.info(f"LLM predict返回内容: {profile}")
        
        if not isinstance(profile, dict):
            logger.warning(f"LLM返回的不是字典，而是: {type(profile)}")
            profile = {}
        
        # 返回结构（不包含年龄组）
        return {
            "skin_type": {
                "name": profile.get("skin_type", {}).get("name", "未知") if isinstance(profile.get("skin_type"), dict) else "未知",
                "characteristics": profile.get("skin_type", {}).get("characteristics", "") if isinstance(profile.get("skin_type"), dict) else "",
                "common_areas": profile.get("skin_type", {}).get("common_areas", "") if isinstance(profile.get("skin_type"), dict) else ""
            },
            "concerns": {
                "primary": profile.get("concerns", {}).get("primary", []) if isinstance(profile.get("concerns"), dict) else [],
                "secondary": profile.get("concerns", {}).get("secondary", []) if isinstance(profile.get("concerns"), dict) else []
            },
            "lifestyle": {
                "diet_habits": profile.get("lifestyle", {}).get("diet_habits", "") if isinstance(profile.get("lifestyle"), dict) else "",
                "daily_routine": profile.get("lifestyle", {}).get("daily_routine", "") if isinstance(profile.get("lifestyle"), dict) else "",
                "environmental_factors": profile.get("lifestyle", {}).get("environmental_factors", "") if isinstance(profile.get("lifestyle"), dict) else ""
            }
        }
    except Exception as e:
        logger.error(f"用户画像分析失败: {e}")
        return {
            "skin_type": {"name": "未知", "characteristics": "", "common_areas": ""},
            "concerns": {"primary": [], "secondary": []},
            "lifestyle": {"diet_habits": "", "daily_routine": "", "environmental_factors": ""}
        }

def analyze_skin_with_vlm_direct(image, chat_history, state_data):
    """直接使用VLM进行皮肤分析，流式输出结果"""
    try:
        # 压缩图片
        if image.size[0] > 800 or image.size[1] > 800:
            logger.info(f"压缩图片，原始尺寸: {image.size}")
            image.thumbnail((800, 800), Image.LANCZOS)
            logger.info(f"压缩后尺寸: {image.size}")
        
        # 先输出正在分析的提示
        yield "正在分析您的面部照片..."
        
        try:
            # 调用VLM分析
            logger.info("开始VLM分析图片...")
            if not advisor:
                yield "抱歉，图片分析服务未能正确初始化，请稍后重试。"
                return
            
            # 🔥 关键调试：记录VLM调用前的状态
            logger.info(f"🔥 VLM调用前：图片尺寸={image.size}, 图片模式={image.mode}")
            
            vlm_result = advisor.execute_sync(image)
            
            # 🔥 关键调试：记录VLM调用结果
            logger.info(f"🔥 VLM调用结果类型: {type(vlm_result)}")
            logger.info(f"🔥 VLM调用结果内容: {vlm_result}")
            
            if not vlm_result:
                yield "抱歉，图片分析返回空结果。请检查网络连接或稍后重试。"
                return
            
            # 解析VLM返回结果
            skin_analysis = ""
            skin_conditions = {}
            
            if isinstance(vlm_result, dict):
                # 处理标准API返回格式
                if 'output' in vlm_result:
                    try:
                        output = json.loads(vlm_result['output']) if isinstance(vlm_result['output'], str) else vlm_result['output']
                        if 'choices' in output:
                            content = output['choices'][0]['message']['content']
                            try:
                                content = json.loads(content) if isinstance(content, str) else content
                                skin_analysis = content.get('analysis', "")
                                skin_conditions = content.get('conditions', {})
                            except:
                                skin_analysis = str(content)
                    except:
                        skin_analysis = str(vlm_result)
                
                # 处理直接返回的分析结果
                elif 'skin_analysis' in vlm_result:
                    analysis = vlm_result['skin_analysis']
                    if isinstance(analysis, dict):
                        skin_analysis = analysis.get('analysis', "")
                        skin_conditions = analysis.get('conditions', {})
                    else:
                        # 检查是否是有效的皮肤分析还是错误消息
                        analysis_str = str(analysis)
                        # 检查是否是简单的错误/连接消息（而不是包含JSON的复杂分析）
                        error_patterns = [
                            '连接测试成功',
                            '网络请求错误',
                            '分析过程中出现错误',
                            '多次尝试后分析失败',
                            '连接正常。您需要进一步的帮助吗？'
                        ]
                        if any(pattern in analysis_str for pattern in error_patterns):
                            # 这看起来像是连接测试消息，不是真正的皮肤分析
                            yield "检测到网络连接问题，图片分析可能不完整。请重新上传图片进行分析。"
                            return
                        skin_analysis = analysis_str
                
                # 确保skin_conditions是字典类型
                if not isinstance(skin_conditions, dict):
                    skin_conditions = {}
            
            # 最终兜底处理
            if not skin_analysis:
                skin_analysis = str(vlm_result) if vlm_result else "无法解析分析结果"
                
            # 处理VLM分析结果，过滤英文字段和年龄信息
            logger.info("VLM分析完成，开始处理和流式输出结果")
            logger.info(f"skin_analysis内容: {skin_analysis}")
            logger.info(f"skin_analysis类型: {type(skin_analysis)}")
            
            # 🔥 关键调试：检查性别信息
            if isinstance(skin_analysis, str):
                # 尝试解析JSON中的性别信息
                try:
                    # 尝试解析JSON格式
                    if skin_analysis.strip().startswith('{'):
                        analysis_data = json.loads(skin_analysis)
                        gender = analysis_data.get('性别', '未检测到')
                        age_group = analysis_data.get('年龄段', '未检测到')
                        logger.info(f"🔥 VLM性别检测结果: {gender}")
                        logger.info(f"🔥 VLM年龄段检测结果: {age_group}")
                        logger.info(f"🔥 VLM完整JSON: {json.dumps(analysis_data, ensure_ascii=False, indent=2)}")
                    else:
                        # 如果不是JSON格式，尝试从文本中提取性别信息
                        logger.info("🔥 VLM返回的是文本格式，尝试从文本中提取性别信息")
                        gender = "未检测到"
                        age_group = "未检测到"
                        
                        # 从文本中提取性别信息
                        if "男性" in skin_analysis or "男士" in skin_analysis or "男" in skin_analysis:
                            gender = "男性"
                            logger.info(f"🔥 VLM文本中检测到男性关键词")
                        elif "女性" in skin_analysis or "女士" in skin_analysis or "女" in skin_analysis:
                            gender = "女性"
                            logger.info(f"🔥 VLM文本中检测到女性关键词")
                        
                        # 将检测到的性别信息保存到state中，供推荐引擎使用
                        if gender != "未检测到":
                            if isinstance(state_data, dict):
                                state_data["detected_gender"] = gender
                                logger.info(f"🔥 保存检测到的性别到state: {gender}")
                            
                        # 从文本中提取年龄段信息
                        if "青年" in skin_analysis:
                            age_group = "青年"
                        elif "中年" in skin_analysis:
                            age_group = "中年"
                        elif "老年" in skin_analysis:
                            age_group = "老年"
                            
                        logger.info(f"🔥 从文本提取的性别: {gender}")
                        logger.info(f"🔥 从文本提取的年龄段: {age_group}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"🔥 VLM JSON解析失败: {e}")
                    logger.error(f"🔥 VLM原始内容: {skin_analysis[:500]}...")
                    # 即使JSON解析失败，也要尝试从文本中提取信息
                    gender = "未检测到"
                    age_group = "未检测到"
                    
                    # 从文本中提取性别信息
                    if "男性" in skin_analysis or "男士" in skin_analysis or "男" in skin_analysis:
                        gender = "男性"
                        logger.info(f"🔥 VLM解析失败后文本提取性别: 男性")
                    elif "女性" in skin_analysis or "女士" in skin_analysis or "女" in skin_analysis:
                        gender = "女性"
                        logger.info(f"🔥 VLM解析失败后文本提取性别: 女性")
                    
                    # 将检测到的性别信息保存到state中，供推荐引擎使用
                    if gender != "未检测到":
                        if isinstance(state_data, dict):
                            state_data["detected_gender"] = gender
                            logger.info(f"🔥 保存检测到的性别到state: {gender}")
                        
                    # 从文本中提取年龄段信息
                    if "青年" in skin_analysis:
                        age_group = "青年"
                    elif "中年" in skin_analysis:
                        age_group = "中年"
                    elif "老年" in skin_analysis:
                        age_group = "老年"
                        
                    logger.info(f"🔥 从文本提取的性别: {gender}")
                    logger.info(f"🔥 从文本提取的年龄段: {age_group}")
            
            # 过滤和格式化分析结果
            formatted_analysis = format_skin_analysis_for_display(skin_analysis)
            
            output_text = "📊 皮肤分析结果：\n\n" + formatted_analysis
            
            # 更新状态数据（保留原始分析结果用于推荐）
            if isinstance(state_data, dict):
                state_data["skin_analysis"] = skin_analysis
                state_data["skin_conditions"] = skin_conditions
                logger.info(f"🔥 状态数据更新: skin_analysis长度={len(str(skin_analysis))}")
            
            # 平滑流式输出（显示过滤后的结果）
            logger.info(f"开始流式输出，output_text长度: {len(output_text)}")
            for chunk in smooth_stream_output(output_text):
                logger.debug(f"输出chunk: {chunk[:100]}...")
                yield chunk
                
        except Exception as vlm_error:
            logger.error(f"VLM分析失败: {str(vlm_error)}")
            if "proxy" in str(vlm_error).lower() or "connection" in str(vlm_error).lower():
                yield "⚠️ 网络连接问题，无法访问分析服务。请检查网络设置或稍后重试。"
            else:
                yield f"抱歉，图片分析过程出现错误。请尝试上传更清晰的照片或稍后重试。"
        
    except Exception as e:
        logger.error(f"整体分析失败: {str(e)}")
        yield "抱歉，系统出现错误，请稍后重试。"

def get_product_recommendations(profile: Dict[str, Any], skin_analysis: str = None) -> List[Dict[str, Any]]:
    """获取产品推荐（优化版，基于皮肤分析结果进行智能RAG检索）"""
    try:
        if not recommender:
            logger.warning("推荐引擎未初始化")
            return []
        
        # 检查是否有皮肤分析结果
        if not skin_analysis or skin_analysis.strip() == "":
            logger.warning("缺少皮肤分析结果，无法进行个性化产品推荐")
            return []
            
        # 构造皮肤状况字典
        skin_conditions = {}
        
        # 从用户画像中提取关注点
        if profile and isinstance(profile, dict):
            concerns = profile.get("concerns", {})
            if isinstance(concerns, dict):
                primary_concerns = concerns.get("primary", [])
                secondary_concerns = concerns.get("secondary", [])
                
                for concern in primary_concerns:
                    if concern in ["皱纹", "细纹", "老化"]:
                        skin_conditions["皱纹"] = 0.8
                    elif concern in ["色斑", "暗沉", "斑点"]:
                        skin_conditions["色斑"] = 0.8
                    elif concern in ["干燥", "缺水"]:
                        skin_conditions["干燥"] = 0.8
                    elif concern in ["敏感", "过敏"]:
                        skin_conditions["敏感"] = 0.8
                    elif concern in ["痘痘", "粉刺", "痤疮"]:
                        skin_conditions["痘痘"] = 0.8
                
                for concern in secondary_concerns:
                    if concern not in skin_conditions:
                        skin_conditions[concern] = 0.5
        
        # 从皮肤分析中提取问题
        if skin_analysis and isinstance(skin_analysis, str):
            for concern in ["皱纹", "色斑", "干燥", "敏感", "痘痘"]:
                if concern in skin_analysis and concern not in skin_conditions:
                    skin_conditions[concern] = 0.7
                elif concern in skin_analysis and concern in skin_conditions:
                    # 如果已经存在，但分数较低，则提升分数
                    if skin_conditions[concern] < 0.7:
                        skin_conditions[concern] = 0.7
        
        # 如果还是没有提取到皮肤问题，使用默认值
        if not skin_conditions:
            skin_conditions = {"保湿": 0.7, "修护": 0.6}
            logger.info("使用默认皮肤条件")
        
        logger.info(f"提取到的皮肤条件: {skin_conditions}")
            
        # 获取产品信息 - 优化RAG查询策略
        product_info = []
        
        # 从profile中获取检测到的性别信息
        detected_gender = profile.get("detected_gender", "未检测到") if profile else "未检测到"
        logger.info(f"🔥 从profile获取的性别信息: {detected_gender}")
        
        if rag:
            logger.info("开始RAG产品检索...")
            try:
                # 构建更智能的RAG查询
                query_parts = []
                
                # 1. 添加皮肤问题关键词
                if skin_conditions:
                    problem_keywords = []
                    for problem, score in skin_conditions.items():
                        if score >= 0.7:  # 主要问题
                            problem_keywords.append(problem)
                        elif score >= 0.5:  # 次要问题
                            problem_keywords.append(problem)
                    
                    if problem_keywords:
                        query_parts.append(" ".join(problem_keywords))
                        logger.info(f"添加皮肤问题关键词: {problem_keywords}")
                
                # 2. 从皮肤分析中提取更多关键词
                if skin_analysis:
                    logger.info(f"皮肤分析结果: {skin_analysis}")
                    
                    # 提取皮肤类型
                    skin_types = ["干性", "油性", "混合性", "敏感性", "中性"]
                    for skin_type in skin_types:
                        if skin_type in skin_analysis:
                            query_parts.append(skin_type)
                            logger.info(f"添加皮肤类型: {skin_type}")
                    
                    # 提取年龄信息
                    age_keywords = ["年轻", "中年", "老年", "成熟"]
                    for age_keyword in age_keywords:
                        if age_keyword in skin_analysis:
                            query_parts.append(age_keyword)
                            logger.info(f"添加年龄关键词: {age_keyword}")
                    
                    # 提取性别信息 - 这是关键修复
                    gender_keywords = ["女性", "男性", "女士", "男士", "女", "男", "woman", "man", "female", "male"]
                    detected_gender = None
                    
                    # 首先尝试从JSON格式中提取性别
                    try:
                        if isinstance(skin_analysis, str) and skin_analysis.strip().startswith("{"):
                            analysis_data = json.loads(skin_analysis)
                            # 检查各种可能的性别字段
                            gender_fields = ["性别", "gender", "sex", "用户性别", "用户类型"]
                            for field in gender_fields:
                                if field in analysis_data:
                                    gender_value = str(analysis_data[field])
                                    if any(keyword in gender_value for keyword in gender_keywords):
                                        detected_gender = gender_value
                                        query_parts.append(gender_value)
                                        logger.info(f"从JSON字段'{field}'中提取到性别: {gender_value}")
                                        break
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                    
                    # 如果JSON解析失败，尝试从文本中提取性别
                    if not detected_gender:
                        for keyword in gender_keywords:
                            if keyword in skin_analysis:
                                detected_gender = keyword
                                query_parts.append(keyword)
                                logger.info(f"从文本中提取到性别: {keyword}")
                                break
                    
                    # 如果还是没有，使用正则表达式搜索
                    if not detected_gender:
                        gender_patterns = [
                            (["女性", "女士", "女", "woman", "female"], "女性"),
                            (["男性", "男士", "男", "man", "male"], "男性")
                        ]
                        for pattern, gender in gender_patterns:
                            if any(word in skin_analysis for word in pattern):
                                detected_gender = gender
                                query_parts.append(gender)
                                logger.info(f"从文本模式中提取到性别: {gender}")
                                break
                    
                    # 如果检测到性别，添加性别相关的查询优化
                    if detected_gender:
                        if "女性" in detected_gender or "女士" in detected_gender or "女" in detected_gender:
                            query_parts.append("护肤")  # 女性护肤产品
                            query_parts.append("-男士")  # 排除男士产品
                            query_parts.append("-男性")  # 排除男性产品
                            query_parts.append("-男")    # 排除男产品
                            logger.info("检测到女性用户，排除男士产品")
                        elif "男性" in detected_gender or "男士" in detected_gender or "男" in detected_gender:
                            query_parts.append("护肤")  # 男性护肤产品
                            query_parts.append("-女士")  # 排除女士产品
                            query_parts.append("-女性")  # 排除女性产品
                            query_parts.append("-女")    # 排除女产品
                            logger.info("检测到男性用户，排除女士产品")
                    else:
                        logger.warning("未能从皮肤分析结果中检测到性别信息")
                        # 尝试从产品名称推断性别（作为备选方案）
                        # 这里可以添加更多的性别推断逻辑
                
                # 3. 构建最终查询
                if query_parts:
                    query = " ".join(query_parts) + " 护肤产品"
                else:
                    query = "保湿 抗皱 修护 护肤产品"
                
                logger.info(f"优化后的RAG查询: {query}")
                logger.info(f"查询关键词: {query_parts}")
                logger.info(f"检测到的性别: {detected_gender}")
                
                # 4. 如果检测到性别，强制添加性别过滤条件
                if detected_gender:
                    if "女性" in detected_gender or "女士" in detected_gender or "女" in detected_gender:
                        # 女性用户：优先搜索女性产品和通用产品
                        query += " 女性 护肤产品"
                        logger.info("女性用户：优先搜索女性产品和通用产品")
                    elif "男性" in detected_gender or "男士" in detected_gender or "男" in detected_gender:
                        # 男性用户：优先搜索男性产品和通用产品
                        query += " 男性 护肤产品"
                        logger.info("男性用户：优先搜索男性产品和通用产品")
                
                logger.info(f"最终RAG查询: {query}")
                
                # 直接使用RAG检索，不使用LLM优化查询（避免超时）
                logger.info("🔥 开始RAG检索...")
                logger.info(f"🔥 RAG查询: {query}")
                logger.info(f"🔥 RAG top_k: 30")
                
                results = rag.retrieve(query, top_k=30)
                logger.info(f"🔥 RAG检索完成，结果数量: {len(results)}")
                
                # 检查RAG返回的产品类型
                if results:
                    for i, result in enumerate(results[:3]):
                        if isinstance(result, dict):
                            name = result.get('product_name') or result.get('name', '未知')
                            logger.info(f"🔥 RAG结果{i+1}: {name}")
                            # 检查是否是默认产品
                            if "欧莱雅复颜玻尿酸" in name or "欧莱雅清润葡萄籽" in name or "欧莱雅青春密码" in name:
                                logger.warning(f"🔥 检测到默认产品: {name}")
                else:
                    logger.warning("🔥 RAG返回空结果，可能触发了fallback机制")
                
                logger.info(f"RAG检索成功，原始结果数量: {len(results)}")
                
                # 记录原始检索结果
                for i, result in enumerate(results[:5]):
                    if isinstance(result, dict):
                        name = result.get("product_name") or result.get("name") or "未知"
                        logger.info(f"RAG结果{i+1}: {name}")
                
                # 🔥 关键修复：将RAG检索结果传递给product_info
                if results:
                    product_info = results
                    logger.info(f"成功设置product_info，包含{len(product_info)}个产品")
                else:
                    logger.warning("RAG检索结果为空，product_info保持为空列表")
                    
            except Exception as e:
                logger.error(f"RAG检索失败: {e}")
                import traceback
                logger.error(f"RAG检索错误堆栈: {traceback.format_exc()}")
                results = []
        else:
            logger.error("RAG模型未初始化，无法进行产品检索")
            results = []
                
        # 调用推荐引擎进行智能匹配
        try:
            logger.info(f"准备调用推荐引擎，skin_conditions: {skin_conditions}, user_profile: {profile}, product_info数量: {len(product_info)}")
            recommendations = recommender.generate_recommendations(
                skin_conditions=skin_conditions,
                user_profile=profile,
                product_info=product_info
            )
            
            # 验证推荐结果
            validated_recs = []
            for rec in recommendations:
                if isinstance(rec, dict) and rec.get("product_name"):
                    # 最终性别验证
                    if detected_gender:
                        product_name = str(rec.get("product_name", "")).lower()
                        product_details = str(rec.get("details", "")).lower()
                        product_text = f"{product_name} {product_details}"
                        
                        # 严格性别检查
                        if "女性" in detected_gender or "女士" in detected_gender or "女" in detected_gender:
                            if any(keyword in product_text for keyword in ["男士", "男性", "男", "man", "male", "男性专用", "男士专用"]):
                                logger.error(f"🚨 最终验证失败：女性用户推荐结果仍包含男士产品 '{rec.get('product_name')}'")
                                continue
                        elif "男性" in detected_gender or "男士" in detected_gender or "男" in detected_gender:
                            if any(keyword in product_text for keyword in ["女士", "女性", "女", "woman", "female", "女性专用", "女士专用"]):
                                logger.error(f"🚨 最终验证失败：男性用户推荐结果仍包含女士产品 '{rec.get('product_name')}'")
                                continue
                    
                    validated_recs.append(rec)
            
            logger.info(f"推荐引擎成功生成 {len(validated_recs)} 个推荐产品（性别验证后）")
            return validated_recs
            
        except Exception as e:
            logger.error(f"推荐引擎调用失败: {e}")
            return []
            
    except Exception as e:
        logger.error(f"产品推荐失败: {e}")
        return []
        
def safe_llm_call(message, system_message="", stream=True):
    """安全调用LLM模型，像VLM一样的平滑流式输出"""
    try:
        if not message or not isinstance(message, str):
            error_msg = "抱歉，我没有收到有效的消息内容。"
            if stream:
                return smooth_stream_output(error_msg)
            else:
                return error_msg
            return

        if stream:
            try:
                # 先获取完整响应
                full_response = llm.chat(message=message, system_message=system_message, temperature=0.7)
                if not full_response:
                    full_response = "抱歉，没有获取到有效回复。"
                
                # 平滑流式输出
                return smooth_stream_output(full_response)
                
            except Exception as e:
                logger.error(f"LLM调用出现异常: {str(e)}")
                error_str = str(e).lower()
                
                # 更精确的错误类型识别
                if any(error_type in error_str for error_type in ["connection", "remote", "disconnected", "aborted", "网络"]):
                    error_msg = "⚠️ **网络连接问题**\n\n抱歉，当前无法连接到AI服务。\n\n**可能原因：**\n• 网络连接不稳定\n• AI服务暂时不可用\n• 代理设置问题\n\n**建议：**\n• 检查网络连接\n• 稍后重试\n• 或者直接描述您的护肤问题"
                elif "timeout" in error_str or "超时" in error_str:
                    error_msg = "⏰ **请求超时**\n\n抱歉，AI服务响应超时。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
                elif "rate limit" in error_str or "quota" in error_str:
                    error_msg = "🚫 **服务限制**\n\n抱歉，AI服务暂时达到使用限制。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
                else:
                    error_msg = "❌ **AI服务异常**\n\n抱歉，AI服务暂时不可用。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
                
                return smooth_stream_output(error_msg)
        else:
            try:
                return llm.chat(message=message, system_message=system_message, temperature=0.7)
            except Exception as e:
                logger.error(f"LLM调用出现异常: {str(e)}")
                error_str = str(e).lower()
                
                # 更精确的错误类型识别
                if any(error_type in error_str for error_type in ["connection", "remote", "disconnected", "aborted", "网络"]):
                    return "⚠️ **网络连接问题**\n\n抱歉，当前无法连接到AI服务。\n\n**可能原因：**\n• 网络连接不稳定\n• AI服务暂时不可用\n• 代理设置问题\n\n**建议：**\n• 检查网络连接\n• 稍后重试\n• 或者直接描述您的护肤问题"
                elif "timeout" in error_str or "超时" in error_str:
                    return "⏰ **请求超时**\n\n抱歉，AI服务响应超时。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
                elif "rate limit" in error_str or "quota" in error_str:
                    return "🚫 **服务限制**\n\n抱歉，AI服务暂时达到使用限制。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
                else:
                    return "❌ **AI服务异常**\n\n抱歉，AI服务暂时不可用。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"

    except Exception as e:
        logger.error(f"安全调用LLM时出现错误: {str(e)}")
        error_msg = "❌ **系统异常**\n\n抱歉，系统暂时无法处理您的请求。\n\n**建议：**\n• 稍后重试\n• 或者直接描述您的护肤问题"
        if stream:
            return smooth_stream_output(error_msg)
        else:
            return error_msg

def smooth_stream_output(text):
    """平滑的流式输出，像VLM一样"""
    import time
    current_text = ""
    for char in text:
        current_text += char
        yield current_text
        time.sleep(0.03)  # 30ms延迟，和VLM保持一致

def format_skin_analysis_for_display(analysis_text):
    """格式化皮肤分析结果，过滤英文字段和年龄信息"""
    try:
        # 如果是JSON格式，尝试解析并过滤
        if analysis_text.strip().startswith('{') and analysis_text.strip().endswith('}'):
            try:
                import json
                analysis_data = json.loads(analysis_text)
                return format_analysis_data(analysis_data)
            except:
                pass
        
        # 如果不是JSON，直接处理文本
        return filter_analysis_text(analysis_text)
        
    except Exception as e:
        logger.error(f"格式化分析结果失败: {e}")
        return filter_analysis_text(analysis_text)

def format_analysis_data(data):
    """格式化JSON分析数据"""
    try:
        formatted_text = ""
        
        # 处理基本信息
        if isinstance(data, dict):
            # 肤质类型 - 支持中英文字段
            skin_type = None
            if "皮肤类型" in data and data["皮肤类型"]:
                skin_type = data["皮肤类型"]
            elif "skin_type" in data and data["skin_type"]:
                skin_type = data["skin_type"]
                
            if skin_type:
                formatted_text += f"🔍 肤质类型：{skin_type}\n\n"
            
            # 主要问题 - 支持中英文字段
            problems = None
            if "主要问题" in data and data["主要问题"]:
                problems = data["主要问题"]
            elif "main_problems" in data and data["main_problems"]:
                problems = data["main_problems"]
            elif "primary_concerns" in data and data["primary_concerns"]:
                problems = data["primary_concerns"]
                
            if problems:
                if isinstance(problems, list):
                    formatted_text += f"⚠️ 主要问题：{', '.join(problems)}\n\n"
                else:
                    formatted_text += f"⚠️ 主要问题：{problems}\n\n"
            
            # 分析结论 - 支持中英文字段
            analysis = None
            if "详细分析" in data and data["详细分析"]:
                analysis = data["详细分析"]
            elif "analysis" in data and data["analysis"]:
                analysis = data["analysis"]
                
            if analysis:
                formatted_text += f"📝 详细分析：{analysis}\n\n"
            
            # 护理建议 - 支持中英文字段
            recommendations = None
            if "护理建议" in data and data["护理建议"]:
                recommendations = data["护理建议"]
            elif "recommendations" in data and data["recommendations"]:
                recommendations = data["recommendations"]
            elif "care_recommendations" in data and data["care_recommendations"]:
                recommendations = data["care_recommendations"]
                
            if recommendations:
                formatted_text += "💡 护肤建议：\n"
                if isinstance(recommendations, list):
                    for i, rec in enumerate(recommendations, 1):
                        formatted_text += f"{i}. {rec}\n"
                else:
                    formatted_text += f"• {recommendations}\n"
                formatted_text += "\n"
            
            # 添加交互提示
            formatted_text += "🛍️ 需要我为您推荐相关的护肤产品吗？请告诉我您的具体需求！"
        
        return formatted_text.strip() if formatted_text.strip() else filter_analysis_text(str(data))
        
    except Exception as e:
        logger.error(f"格式化JSON数据失败: {e}")
        return filter_analysis_text(str(data))

def filter_analysis_text(text):
    """过滤分析文本，移除英文字段和年龄信息"""
    import re
    
    # 移除常见的英文字段
    english_fields = [
        r'"age_group"[^,}]*[,}]',
        r'"age"[^,}]*[,}]',
        r'"skin_type"[^,}]*[,}]',
        r'"primary_concerns"[^,}]*[,}]',
        r'"care_recommendations"[^,}]*[,}]',
        r'"analysis"[^,}]*[,}]',
        r'"main_problems"[^,}]*[,}]',
        r'"skin_conditions"[^,}]*[,}]',
        r'"severity"[^,}]*[,}]',
        r'"confidence"[^,}]*[,}]',
        r'"overall_assessment"[^,}]*[,}]',
        r'"treatment_suggestions"[^,}]*[,}]',
        r'"prevention_tips"[^,}]*[,}]',
        r'"daily_routine"[^,}]*[,}]',
        r'"product_suggestions"[^,}]*[,}]',
        r'\b(age_group|age|young|middle|old|elderly|main_problems|primary_concerns|care_recommendations|skin_conditions|severity|confidence|overall_assessment|treatment_suggestions|prevention_tips|daily_routine|product_suggestions)\b[^。！？]*[。！？]?',
        r'年龄[^。！？]*[。！？]',
        r'岁[^。！？]*[。！？]',
        r'青年[^。！？]*[。！？]',
        r'中年[^。！？]*[。！？]',
        r'老年[^。！？]*[。！？]'
    ]
    
    filtered_text = text
    for pattern in english_fields:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.IGNORECASE)
    
    # 清理多余的空行和符号
    filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)
    filtered_text = re.sub(r'[,，]\s*[,，]', '，', filtered_text)
    filtered_text = re.sub(r'^\s*[,，]\s*', '', filtered_text, flags=re.MULTILINE)
    
    # 确保在文本末尾添加交互提示
    if filtered_text.strip() and not filtered_text.strip().endswith("🛍️ 需要我为您推荐相关的护肤产品吗？请告诉我您的具体需求！"):
        filtered_text += "\n\n🛍️ 需要我为您推荐相关的护肤产品吗？请告诉我您的具体需求！"
    
    return filtered_text.strip()

def on_analyze(image, chat_history, state_data):
    """处理图片分析，简化版流程"""
    # 验证输入
    if not isinstance(chat_history, list):
        chat_history = []
    
    if image is None:
        chat_history.append((None, "请先上传一张面部照片再进行分析。"))
        yield chat_history, state_data
        return
    
    # 检查图片尺寸
    if hasattr(image, 'size') and (image.size[0] < 100 or image.size[1] < 100):
        chat_history.append((None, "图片分辨率过低，请上传更清晰的面部照片。"))
        yield chat_history, state_data
        return
    
    # 插入loading气泡
    if not any(msg[0] == "帮我检测肤质" for msg in chat_history):
        chat_history.append(("帮我检测肤质", "正在为您检测面部照片的肤质状况，请稍候..."))
        yield chat_history, state_data
    
    # 直接调用VLM分析，流式输出
    try:
        analysis_generator = analyze_skin_with_vlm_direct(image, chat_history, state_data)
        
        for result in analysis_generator:
            if len(chat_history) > 0 and chat_history[-1][0] == "帮我检测肤质":
                chat_history[-1] = ("帮我检测肤质", result)
            else:
                chat_history.append(("帮我检测肤质", result))
            yield chat_history, state_data
        
        # 更新状态数据中的皮肤分析结果
        # 从聊天历史中获取最新的分析结果
        latest_analysis = ""
        if len(chat_history) > 0 and chat_history[-1][0] == "帮我检测肤质":
            latest_analysis = chat_history[-1][1]
        
        # 过滤分析文本
        filtered_analysis = filter_analysis_text(latest_analysis)
        
        # 检查分析结果是否有效
        if not filtered_analysis or filtered_analysis.strip() == "" or "分析出错" in filtered_analysis or "失败" in filtered_analysis:
            # 分析结果无效，引导用户重新尝试
            guidance_msg = """❌ **皮肤分析未能完成**

🔍 **可能的原因：**
• 图片不够清晰或角度不当
• 光线条件不理想
• 网络连接问题
• 分析服务暂时不可用

💡 **建议解决方案：**
1. **重新上传照片**：确保面部清晰可见，光线充足
2. **调整拍摄角度**：正面拍摄，避免侧脸或模糊
3. **检查网络**：确保网络连接稳定
4. **稍后重试**：如果问题持续，请稍后再次尝试

📋 **或者，您也可以：**
• 直接告诉我您的肤质类型和护肤困扰
• 描述您目前遇到的具体皮肤问题
• 说明您的年龄范围和性别

请重新尝试，我会继续为您提供专业的护肤建议！"""
            chat_history[-1] = ("帮我检测肤质", guidance_msg)
            yield chat_history, state_data
            return
        
        if isinstance(state_data, dict):
            state_data["skin_analysis"] = filtered_analysis
            
            # 设置推荐提示状态，标记系统已经询问过用户是否需要推荐
            state_data["recommendation_prompted"] = True
        
        logger.info("皮肤分析完成，已设置推荐提示状态")
        
        # 获取产品推荐
        if isinstance(state_data, dict) and "skin_analysis" in state_data:
            try:
                user_profile = state_data.get("profile", {})
                if not user_profile:
                    # 如果用户画像为空，尝试从聊天历史中提取
                    for msg in reversed(chat_history):
                        if isinstance(msg, tuple) and isinstance(msg[0], str):
                            # 过滤掉系统生成的消息
                            if msg[0] not in ["帮我检测肤质"]:
                                try:
                                    user_profile = analyze_user_profile(msg[0])
                                    state_data["profile"] = user_profile
                                    break
                                except Exception as profile_error:
                                    logger.error(f"分析用户画像失败: {profile_error}")
                                    continue
                
                if user_profile:
                    skin_analysis = str(state_data.get("skin_analysis", ""))
                    recommendations = get_product_recommendations(user_profile, skin_analysis)
                    
                    logger.info(f"获取到的推荐产品数量: {len(recommendations) if recommendations else 0}")
                    if recommendations:
                        logger.info(f"推荐产品列表: {[rec.get('product_name', '未知') for rec in recommendations[:3]]}")
                    else:
                        logger.warning("没有获取到推荐产品")
                        # 如果没有推荐产品，给出友好提示
                        no_recommendations_msg = """抱歉，暂时没有找到完全匹配您需求的产品。

🔍 **可能的原因：**
• 产品库中缺少相关产品
• 您的需求比较特殊
• 系统暂时出现技术问题

💡 **建议：**
• 稍后重试
• 调整您的需求描述
• 联系客服获取个性化推荐

如果您有其他护肤问题，我很乐意为您解答！"""
                        chat_history[-1] = (msg, no_recommendations_msg)
                        yield "", chat_history, state
                        return
                    
                    state_data["recommendations"] = recommendations
                    
                    # 将推荐产品添加到聊天历史
                    rec_text = "根据分析，我为您推荐以下产品：\n\n"
                    for rec in recommendations:
                        rec_text += f"🏷️ {rec.get('product_name', '未知产品')}\n"
                        if rec.get('target_concerns'):
                            rec_text += f"🎯 针对问题：{', '.join(rec['target_concerns'])}\n"
                        if rec.get('key_ingredients'):
                            rec_text += f"💊 核心成分：{', '.join(rec['key_ingredients'])}\n"
                        if rec.get('benefits'):
                            rec_text += f"✨ 功效：{', '.join(rec['benefits'])}\n"
                        if rec.get('usage_instructions') and rec['usage_instructions'].get('method'):
                            rec_text += f"📝 使用方法：{rec['usage_instructions']['method']}\n"
                        if rec.get('suitability_reason'):
                            rec_text += f"💡 推荐理由：{rec['suitability_reason']}\n"
                        rec_text += "\n"
                    
                    # 检查是否已经显示过推荐
                    if not any(msg[0] == "帮我检测肤质" and "推荐以下产品" in msg[1] for msg in chat_history):
                        chat_history.append(("帮我检测肤质", rec_text))
                        yield chat_history, state_data
                            
            except Exception as e:
                logger.error(f"获取推荐信息失败: {e}")
        
    except Exception as e:
        logger.error(f"图片分析失败: {e}")
        error_msg = "图片分析失败，请确保上传的是清晰的面部照片"
        if "timed out" in str(e):
            error_msg = "图片分析超时，请尝试上传更小的图片"
        if len(chat_history) > 0 and chat_history[-1][0] == "帮我检测肤质":
            chat_history[-1] = ("帮我检测肤质", error_msg)
        else:
            chat_history.append(("帮我检测肤质", error_msg))
        yield chat_history, state_data



def user_message_and_response(msg, chat_history, state):
    """处理用户输入消息，返回真正的流式对话结果"""
    if not isinstance(chat_history, list):
        chat_history = []
    if not isinstance(state, dict):
        state = {"consultation_type": None, "skin_analysis": None, "profile": None}
    
    # 添加用户消息到历史
    chat_history.append((msg, ""))
    
    # 先显示用户消息
    yield "", chat_history, state
    
    # 构建系统消息
    system_context = "你是TimelessSkin的智能护肤顾问，请用专业、友好的风格回答用户的护肤问题，不要重复自我介绍。"
    if state.get("consultation_type"):
        system_context += f"当前咨询类型：{state['consultation_type']}\n"
    if state.get("skin_analysis"):
        system_context += f"皮肤分析结果：{state['skin_analysis']}\n"
    
    # 初始化意图识别变量
    is_product_request = False
                
    try:
        # 直接使用大模型进行智能意图识别
        try:
            # 构建更智能的意图识别提示词
            intent_prompt = f"""
请判断用户的真实意图。用户消息："{msg}"

分析以下情况：
1. 用户是否在明确请求产品推荐？（如"推荐产品"、"推荐护肤品"、"需要产品"等）
2. 用户是否在确认同意产品推荐？（如"好的"、"可以"、"推荐吧"、"用"、"要"等）
3. 用户是否在询问护肤方法或建议？（如"推荐祛痘方法"、"推荐护肤步骤"等）
4. 用户是否在其他护肤相关问题？

请只回答：
- "产品推荐" - 如果用户明确请求产品推荐或确认同意推荐
- "其他需求" - 如果用户询问护肤方法、建议或其他问题

注意：区分"推荐产品"和"推荐方法"，只有明确要产品时才回答"产品推荐"。
"""
            
            # 调用LLM进行意图识别
            if llm:
                try:
                    intent_response = llm.chat(
                        message=intent_prompt,
                        system_message="你是意图识别助手，只回答一个词。",
                        temperature=0.1  # 低温度确保一致性
                    )
                    
                    logger.info(f"LLM意图识别结果: {intent_response}")
                    
                    # 根据LLM响应判断意图
                    if "产品推荐" in intent_response:
                        is_product_request = True
                        logger.info("LLM判断：用户请求产品推荐")
                    else:
                        is_product_request = False
                        logger.info("LLM判断：用户有其他需求，不是请求推荐")
                        
                except Exception as e:
                    logger.warning(f"LLM意图识别失败，使用简化关键词匹配: {e}")
                    # LLM失败时，只匹配非常明确的肯定回答，避免误判
                    clear_positive_keywords = [
                        "好的", "可以", "行", "ok", "OK", "用", "要", "是的", "对", "嗯"
                    ]
                    # 避免匹配包含"推荐"的复杂表达，防止误判
                    if any(keyword in msg.lower() for keyword in clear_positive_keywords) and len(msg.strip()) <= 5:
                        # 只有简短明确的肯定回答才认为是产品推荐请求
                        is_product_request = True
                        logger.info("简化关键词匹配：检测到明确的肯定回答")
                    else:
                        is_product_request = False
                        logger.info("简化关键词匹配：避免误判，默认为其他需求")
            else:
                # LLM未初始化，使用简化关键词匹配
                clear_positive_keywords = [
                    "好的", "可以", "行", "ok", "OK", "用", "要", "是的", "对", "嗯"
                ]
                # 避免匹配包含"推荐"的复杂表达，防止误判
                if any(keyword in msg.lower() for keyword in clear_positive_keywords) and len(msg.strip()) <= 5:
                    # 只有简短明确的肯定回答才认为是产品推荐请求
                    is_product_request = True
                    logger.info("简化关键词匹配：检测到明确的肯定回答")
                else:
                    is_product_request = False
                    logger.info("简化关键词匹配：避免误判，默认为其他需求")
                
        except Exception as e:
            logger.warning(f"意图识别过程出错: {e}")
            # 出错时，默认不是推荐请求
            is_product_request = False
        
        # 如果是产品推荐请求，检查用户信息完整性
        if is_product_request:
            # 检查用户信息是否完整 - 只要有皮肤分析就足够了
            has_skin_analysis = state.get("skin_analysis") and len(str(state.get("skin_analysis", "")).strip()) > 0
            
            logger.info(f"用户信息完整性检查: 皮肤分析={has_skin_analysis}")
            
            if not has_skin_analysis:
                # 缺失皮肤分析，引导用户先获取信息
                logger.info("用户信息不完整，缺失皮肤分析")
                guidance_msg = """我理解您想要产品推荐，但是为了给您提供最准确的推荐，我需要先了解您的皮肤状况。

请您先上传一张清晰的面部照片，让我为您进行专业的皮肤分析，这样我就能：
• 识别您的皮肤类型（干性/油性/混合性/敏感性）
• 检测皮肤问题（痘痘/色斑/皱纹/敏感等）
• 分析您的年龄和性别特征
• 为您推荐最适合的护肤产品

📸 请上传照片开始分析吧！✨"""
                
                chat_history[-1] = (msg, guidance_msg)
                yield "", chat_history, state
                return
            
            # 有皮肤分析结果，继续产品推荐流程
            logger.info("用户信息完整，继续产品推荐流程")
            
            # 调用产品推荐功能
            logger.info("调用产品推荐功能")
            # 更新状态，标记用户已请求推荐
            state["recommendation_prompted"] = True
            yield from handle_product_recommendation(msg, chat_history, state)
            return
        
        # 3. 上下文理解：如果系统询问是否需要推荐，用户回复积极
        if not is_product_request and len(chat_history) > 0:
            # 检查最近的系统消息是否包含推荐提示
            for i in range(len(chat_history)-1, max(-1, len(chat_history)-3), -1):
                if len(chat_history[i]) > 1 and chat_history[i][1]:
                    system_msg = str(chat_history[i][1])
                    if "需要我为您推荐相关的护肤产品" in system_msg:
                        # 检查用户回复是否积极，但避免误判复杂表达
                        positive_responses = ["好的", "是的", "可以", "需要", "要", "行", "ok", "OK", "用", "对", "嗯"]
                        # 只匹配简短明确的肯定回答，避免误判包含"推荐"的复杂表达
                        if any(response in str(msg).lower() for response in positive_responses) and len(msg.strip()) <= 5:
                            is_product_request = True
                            logger.info(f"通过上下文理解识别到产品推荐请求: {msg}")
                            break
        
        # 4. 语义理解：分析用户回复的语义
        if not is_product_request:
            # 检查是否是改变主意的表达
            change_mind_patterns = [
                "还是", "那", "既然", "既然这样", "这样的话",
                "推荐", "推荐吧", "推荐给我", "推荐一下"
            ]
            if any(pattern in msg for pattern in change_mind_patterns):
                is_product_request = True
                logger.info(f"通过语义理解识别到产品推荐请求: {msg}")
        
        # 5. 特殊模式识别：处理"还是推荐吧"、"那推荐吧"等表达
        if not is_product_request:
            # 检查是否是典型的改变主意模式
            change_mind_phrases = [
                "还是推荐吧", "那推荐吧", "既然这样推荐吧", "推荐吧", "推荐给我",
                "还是推荐", "那推荐", "推荐一下", "推荐个", "推荐几个","说吧"
            ]
            if any(phrase in msg for phrase in change_mind_phrases):
                is_product_request = True
                logger.info(f"通过特殊模式识别到产品推荐请求: {msg}")
        
        # 6. 上下文状态检查：如果之前有推荐提示，用户回复积极
        if not is_product_request and state.get("skin_analysis"):
            # 检查状态中是否有推荐提示的标记
            if "recommendation_prompted" not in state:
                state["recommendation_prompted"] = False
            
            # 如果之前已经提示过推荐，且用户回复包含积极词汇
            if state.get("recommendation_prompted", False):
                positive_words = ["好的", "是的", "可以", "行", "用", "要", "对", "嗯"]
                # 只匹配简短明确的肯定回答，避免误判复杂表达
                if any(word in msg for word in positive_words) and len(msg.strip()) <= 5:
                    is_product_request = True
                    logger.info(f"通过状态检查识别到产品推荐请求: {msg}")
        
        logger.info(f"意图识别结果: 消息='{msg}', 是否产品推荐请求={is_product_request}")
        
        # 检查是否是拒绝产品推荐的表达
        rejection_keywords = [
            # 中文关键词
            "不用", "不需要", "算了", "不用了", "暂时不用", "现在不需要", "以后再说", "先不用",
            "不要", "不想要", "免了", "算了", "停", "停止", "结束", "关闭",
            # 英文关键词
            "no", "not", "stop", "end", "close", "cancel", "quit", "don't", "doesn't",
            # 拼音关键词
            "buyong", "buxuyao", "suanle", "buyongle", "zanshibuyong", "xianzaibuxuyao", "yihouzai", "xianbuyong"
        ]
        
        # 添加调试日志
        logger.info(f"检查拒绝关键词: 消息='{msg}'")
        is_rejection = any(keyword in msg.lower() for keyword in rejection_keywords)
        logger.info(f"拒绝检测结果: {is_rejection}")
        
        if is_rejection:
            # 用户拒绝产品推荐，给出友好的回应
            logger.info("用户拒绝产品推荐，给出友好回应")
            friendly_response = """好的，我理解您的选择。💝

✨ **如果您以后需要护肤建议或产品推荐，随时可以：**
• 上传照片进行皮肤检测
• 描述您的护肤困扰
• 询问具体的护肤问题

我会一直在这里为您提供专业的护肤指导！有什么其他护肤问题需要帮助吗？"""
            
            chat_history[-1] = (msg, friendly_response)
            yield "", chat_history, state
            return
        
        if is_product_request:
            if state.get("skin_analysis"):
                # 如果是产品推荐请求且有皮肤分析结果，调用产品推荐功能
                logger.info("调用产品推荐功能")
                # 更新状态，标记用户已请求推荐
                state["recommendation_prompted"] = True
                yield from handle_product_recommendation(msg, chat_history, state)
            else:
                # 如果是产品推荐请求但没有皮肤分析结果，引导用户先获取基本信息
                logger.info("用户请求产品推荐但缺少皮肤分析结果，引导获取基本信息")
                guidance_msg = """💡 我理解您想要产品推荐，但为了给您提供最准确、个性化的建议，我需要先了解您的皮肤状况。

📋 **请先完成以下任一方式的信息收集：**

**方式1：上传面部照片** 📸
• 点击上方"皮肤检测"区域上传清晰的面部照片
• 我将为您进行专业的皮肤分析

**方式2：文字描述** ✍️
• 告诉我您的肤质类型（干性/油性/混合型/敏感型等）
• 描述您目前遇到的主要护肤困扰
• 说明您的年龄范围和性别

🔍 **为什么需要这些信息？**
• 不同肤质需要不同的护理方案
• 年龄和性别影响皮肤特点和需求
• 具体问题决定产品功效选择
• 个性化推荐提高护肤效果

请先完成皮肤状况分析，然后我就能为您推荐最适合的产品了！"""
                chat_history[-1] = (msg, guidance_msg)
                yield "", chat_history, state
        else:
            # 检查是否是产品推荐相关的问题（关键词检测）
            product_related_keywords = [
                "推荐", "产品", "护肤品", "化妆品", "面霜", "精华", "洁面", "防晒", "面膜",
                "保湿", "美白", "抗皱", "祛痘", "控油", "敏感肌", "干性", "油性", "混合性",
                "品牌", "价格", "成分", "效果", "使用方法"
            ]
            
            is_product_related = any(keyword in msg for keyword in product_related_keywords)
            
            if is_product_related:
                # 产品相关问题，但没有完整的皮肤分析，引导用户
                guidance_msg = """我理解您对护肤产品有疑问，但为了给您最准确的建议，建议您先：

📸 **上传面部照片进行皮肤分析**
或
✍️ **详细描述您的肤质和护肤困扰**

这样我就能为您推荐最适合的产品了！

如果您有其他护肤知识方面的问题，我也很乐意为您解答。"""
                chat_history[-1] = (msg, guidance_msg)
                yield "", chat_history, state
            else:
                # 使用安全LLM流式输出（非产品推荐相关的一般护肤问题）
                logger.info("使用LLM生成回复（非产品推荐）")
                # 修改系统上下文，明确禁止产品推荐
                restricted_context = system_context + "\n重要：不要推荐具体的产品品牌或型号，只提供护肤知识和建议。"
                response_generator = safe_llm_call(msg, restricted_context, stream=True)
                        
                # 真正的流式输出响应 - 每个chunk立即显示
                for chunk in response_generator:
                    if chunk:  # 确保chunk不为空
                        # 更新最后一条消息，使用完整的累积文本
                        chat_history[-1] = (msg, chunk)
                        yield "", chat_history, state
    except Exception as e:
        logger.error(f"对话生成失败: {e}")
        
        # 根据错误类型提供不同的错误信息
        if "Connection" in str(e) or "RemoteDisconnected" in str(e) or "网络" in str(e):
            error_msg = "抱歉，当前网络连接有问题，请检查网络设置或稍后重试。"
        elif "timeout" in str(e) or "超时" in str(e):
            error_msg = "抱歉，请求超时，请稍后重试。"
        elif "推荐" in msg or "产品" in msg:
            error_msg = "抱歉，产品推荐功能暂时出现问题。请确保已完成皮肤分析，或稍后重试。"
        else:
            error_msg = "抱歉，系统出现错误，请稍后重试。"
            
        chat_history[-1] = (msg, error_msg)
        yield "", chat_history, state

# 将产品推荐处理函数移到全局作用域
def handle_product_recommendation(msg, chat_history, state):
    """处理产品推荐请求"""
    try:
        # 显示正在处理的消息
        chat_history[-1] = (msg, "正在为您寻找合适的护肤产品...")
        yield "", chat_history, state
        
        # 获取用户画像和皮肤分析
        user_profile = state.get("profile", {})
        skin_analysis = state.get("skin_analysis", "")
        
        # 检查是否有皮肤分析结果
        if not skin_analysis or skin_analysis.strip() == "":
            guidance_msg = """❌ **无法进行产品推荐**

🔍 **原因：缺少皮肤状况信息**

📋 **请先完成以下任一方式的信息收集：**

**方式1：上传面部照片** 📸
• 点击上方"皮肤检测"区域上传清晰的面部照片
• 我将为您进行专业的皮肤分析

**方式2：文字描述** ✍️
• 告诉我您的肤质类型（干性/油性/混合型/敏感型等）
• 描述您目前遇到的主要护肤困扰
• 说明您的年龄范围和性别

🔍 **为什么需要这些信息？**
• 不同肤质需要不同的护理方案
• 年龄和性别影响皮肤特点和需求
• 具体问题决定产品功效选择
• 个性化推荐提高护肤效果

请先完成皮肤状况分析，然后我就能为您推荐最适合的产品了！"""
            chat_history[-1] = (msg, guidance_msg)
            yield "", chat_history, state
            return
        
        # 如果没有用户画像，从聊天历史中分析
        if not user_profile:
            try:
                user_profile = analyze_user_profile(msg)
                state["profile"] = user_profile
            except Exception as e:
                logger.error(f"分析用户画像失败: {e}")
                user_profile = {}
        
        # 检查用户是否拒绝产品推荐
        rejection_keywords = [
            # 中文关键词
            "不用", "不需要", "算了", "不用了", "暂时不用", "现在不需要", "以后再说", "先不用",
            # 英文关键词
            "no", "not", "stop", "end", "close", "cancel", "quit", "don't", "doesn't",
            # 拼音关键词
            "buyong", "buxuyao", "suanle", "buyongle", "zanshibuyong", "xianzaibuxuyao", "yihouzai", "xianbuyong"
        ]
        is_rejection = any(keyword in msg.lower() for keyword in rejection_keywords)
        
        if is_rejection:
            # 用户拒绝推荐，友好回复
            friendly_response = "好的，我理解您暂时不需要产品推荐。如果您以后有任何护肤问题或需要产品建议，随时可以告诉我！我会一直在这里为您提供帮助。"
            chat_history[-1] = (msg, friendly_response)
            yield "", chat_history, state
            return
        
        # 获取产品推荐
        logger.info("🔥 开始调用产品推荐函数...")
        logger.info(f"🔥 用户画像: {user_profile}")
        logger.info(f"🔥 皮肤分析: {skin_analysis[:200]}...")
        
        # 确保性别信息包含在用户画像中
        if isinstance(user_profile, dict) and "detected_gender" not in user_profile:
            # 从state中获取性别信息
            detected_gender = state.get("detected_gender", "未检测到")
            if detected_gender != "未检测到":
                user_profile["detected_gender"] = detected_gender
                logger.info(f"🔥 将性别信息添加到用户画像: {detected_gender}")
        
        recommendations = get_product_recommendations(user_profile, skin_analysis)
        
        logger.info(f"🔥 获取到的推荐产品数量: {len(recommendations) if recommendations else 0}")
        if recommendations:
            logger.info(f"🔥 推荐产品列表: {[rec.get('product_name', '未知') for rec in recommendations[:3]]}")
            # 检查是否是默认产品
            for i, rec in enumerate(recommendations[:3]):
                name = rec.get('product_name', '未知')
                if "欧莱雅复颜玻尿酸" in name or "欧莱雅清润葡萄籽" in name or "欧莱雅青春密码" in name:
                    logger.warning(f"🔥 检测到默认产品{i+1}: {name}")
        else:
            logger.warning("🔥 没有获取到推荐产品")
            # 如果没有推荐产品，给出友好提示
            no_recommendations_msg = """抱歉，暂时没有找到完全匹配您需求的产品。

🔍 **可能的原因：**
• 产品库中缺少相关产品
• 您的需求比较特殊
• 系统暂时出现技术问题

💡 **建议：**
• 稍后重试
• 调整您的需求描述
• 联系客服获取个性化推荐

如果您有其他护肤问题，我很乐意为您解答！"""
            chat_history[-1] = (msg, no_recommendations_msg)
            yield "", chat_history, state
            return
        
        if recommendations:
            # 提取皮肤问题评分（用于LLM优化推荐理由）
            try:
                skin_conditions = {}
                
                # 从用户画像中提取关注点
                if user_profile and isinstance(user_profile, dict):
                    concerns = user_profile.get("concerns", {})
                    if isinstance(concerns, dict):
                        primary_concerns = concerns.get("primary", [])
                        secondary_concerns = concerns.get("secondary", [])
                        
                        for concern in primary_concerns:
                            if concern in ["皱纹", "细纹", "老化"]:
                                skin_conditions["皱纹"] = 0.8
                            elif concern in ["色斑", "暗沉", "斑点"]:
                                skin_conditions["色斑"] = 0.8
                            elif concern in ["干燥", "缺水"]:
                                skin_conditions["干燥"] = 0.8
                            elif concern in ["敏感", "过敏"]:
                                skin_conditions["敏感"] = 0.8
                            elif concern in ["痘痘", "粉刺", "痤疮"]:
                                skin_conditions["痘痘"] = 0.8
                        
                        for concern in secondary_concerns:
                            if concern not in skin_conditions:
                                skin_conditions[concern] = 0.5
                
                # 从皮肤分析中提取问题
                if skin_analysis and isinstance(skin_analysis, str):
                    for concern in ["皱纹", "色斑", "干燥", "敏感", "痘痘"]:
                        if concern in skin_analysis and concern not in skin_conditions:
                            skin_conditions[concern] = 0.7
                        elif concern in skin_analysis and concern in skin_conditions:
                            # 如果已经存在，但分数较低，则提升分数
                            if skin_conditions[concern] < 0.7:
                                skin_conditions[concern] = 0.7
                            
                # 如果还是没有提取到皮肤问题，从state中获取
                if not skin_conditions:
                    state_skin_conditions = state.get("skin_conditions", {})
                    if state_skin_conditions and isinstance(state_skin_conditions, dict):
                        skin_conditions = state_skin_conditions
                        logger.info(f"从state中获取皮肤条件: {skin_conditions}")
                
                # 如果仍然为空，使用默认值
                if not skin_conditions:
                    skin_conditions = {"保湿": 0.7, "修护": 0.6}
                    logger.info("使用默认皮肤条件")
                
                logger.info(f"最终皮肤条件: {skin_conditions}")
                            
            except Exception as e:
                logger.warning(f"提取皮肤问题评分失败: {e}")
                # 如果提取失败，使用默认值
                skin_conditions = {"保湿": 0.7, "修护": 0.6}
            
            # 使用LLM优化推荐理由，但保持原有的产品结构
            try:
                # 构建简化的LLM提示词，只优化推荐理由部分
                llm_prompt = f"""
基于用户皮肤分析结果，为以下产品生成个性化推荐理由：

皮肤问题：{', '.join(list(skin_conditions.keys())[:3]) if skin_conditions else '保湿'}
产品数量：{len(recommendations)}

请为每个产品生成1-2句推荐理由，说明为什么适合用户。
重要：请直接输出推荐理由，不要添加编号（如1. 2. 3.），每个推荐理由用换行符分隔。
格式示例：
这款产品富含保湿成分，能够深层滋养干燥肌肤
针对您的皮肤问题，这款产品特别添加了修护成分
此款护肤品能够有效改善您的肌肤状况

只输出推荐理由，不要其他内容。
"""
                
                # 调用LLM生成推荐理由
                if llm:
                    try:
                        llm_response = llm.chat(
                            message=llm_prompt,
                            system_message="你是护肤顾问，只生成推荐理由。",
                            temperature=0.3  # 降低温度，提高响应速度
                        )
                        
                        if llm_response and isinstance(llm_response, str) and len(llm_response.strip()) > 10:
                            # 使用LLM生成的推荐理由，但保持原有的产品结构
                            rec_text = "根据您的皮肤分析结果，我为您推荐以下护肤产品：\n\n"
                            
                            for i, rec in enumerate(recommendations[:5], 1):
                                rec_text += f"**{i}. {rec.get('product_name', '未知产品')}**\n"
                                
                                if rec.get('brand'):
                                    rec_text += f"🏷️ **品牌**：{rec['brand']}\n"
                                
                                if rec.get('target_concerns'):
                                    concerns = rec['target_concerns']
                                    if isinstance(concerns, list) and concerns:
                                        rec_text += f"🎯 **针对问题**：{', '.join(concerns)}\n"
                                    elif concerns:
                                        rec_text += f"🎯 **针对问题**：{concerns}\n"
                                
                                if rec.get('key_ingredients'):
                                    ingredients = rec['key_ingredients']
                                    if isinstance(ingredients, list) and ingredients:
                                        rec_text += f"💊 **核心成分**：{', '.join(ingredients)}\n"
                                    elif ingredients:
                                        rec_text += f"💊 **核心成分**：{ingredients}\n"
                                
                                if rec.get('benefits'):
                                    benefits = rec['benefits']
                                    if isinstance(benefits, list) and benefits:
                                        rec_text += f"✨ **主要功效**：{', '.join(benefits)}\n"
                                    elif benefits:
                                        rec_text += f"✨ **主要功效**：{benefits}\n"
                                
                                if rec.get('usage_instructions') and isinstance(rec['usage_instructions'], dict):
                                    method = rec['usage_instructions'].get('method', '')
                                    if method:
                                        rec_text += f"📝 **使用方法**：{method}\n"
                                
                                if rec.get('price'):
                                    rec_text += f"💰 **参考价格**：{rec['price']}\n"
                                
                                if rec.get('link'):
                                    rec_text += f"🔗 **购买链接**：[点击购买]({rec['link']})\n"
                                
                                # 使用LLM生成的推荐理由
                                if llm_response:
                                    # 从LLM响应中提取对应的推荐理由
                                    llm_lines = [line.strip() for line in llm_response.strip().split('\n') if line.strip()]
                                    # 过滤掉编号行（如"1.", "2.", "3."等）
                                    filtered_lines = []
                                    for line in llm_lines:
                                        # 跳过以数字+点开头的行
                                        if not re.match(r'^\d+\.', line):
                                            filtered_lines.append(line)
                                    
                                    if i-1 < len(filtered_lines):
                                        reason = filtered_lines[i-1]
                                        if reason and len(reason) > 5:
                                            rec_text += f"💡 **推荐理由**：{reason}\n"
                                        else:
                                            rec_text += f"💡 **推荐理由**：基于您的皮肤状况，这款产品能够有效解决您的护肤需求\n"
                                    else:
                                        rec_text += f"💡 **推荐理由**：基于您的皮肤状况，这款产品能够有效解决您的护肤需求\n"
                                else:
                                    rec_text += f"💡 **推荐理由**：基于您的皮肤状况，这款产品能够有效解决您的护肤需求\n"
                                
                                rec_text += "\n" + "─"*50 + "\n\n"
                            
                            rec_text += "🔍 **温馨提示**：以上推荐基于您的皮肤分析结果，建议在使用新产品前先做皮肤测试。如需了解更多详情或有其他问题，请随时告诉我！"
                        else:
                            # LLM返回无效响应，使用原有的推荐格式
                            logger.warning("LLM返回无效响应，使用原有的推荐格式")
                            rec_text = _generate_fallback_recommendation(recommendations)
                            
                    except Exception as llm_error:
                        logger.warning(f"LLM推荐理由生成失败，使用fallback格式: {llm_error}")
                        # LLM调用失败，使用原有的推荐格式
                        rec_text = _generate_fallback_recommendation(recommendations)
                else:
                    # 如果LLM未初始化，使用原有的推荐格式
                    logger.warning("LLM未初始化，使用原有的推荐格式")
                    rec_text = _generate_fallback_recommendation(recommendations)
                    
            except Exception as e:
                logger.error(f"LLM推荐理由生成失败: {e}")
                # 如果LLM调用失败，使用原有的推荐格式
                rec_text = _generate_fallback_recommendation(recommendations)
            
            # 使用流式输出显示推荐结果
            if rec_text:
                # 将推荐文本按段落分割，实现更自然的流式输出
                paragraphs = rec_text.split('\n\n')
                current_text = ""
                
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # 跳过空段落
                        current_text += paragraph + "\n\n"
                        # 每添加一个段落就更新一次，创造流式效果
                        chat_history[-1] = (msg, current_text.strip())
                        yield "", chat_history, state
                        import time; time.sleep(0.15)  # 150ms延迟，让用户有时间阅读
            else:
                error_msg = "抱歉，暂时没有找到合适的产品推荐。请确保已完成皮肤分析，或者描述更多您的护肤需求。"
                chat_history[-1] = (msg, error_msg)
                yield "", chat_history, state
        else:
            error_msg = "抱歉，暂时没有找到合适的产品推荐。请确保已完成皮肤分析，或者描述更多您的护肤需求。"
            chat_history[-1] = (msg, error_msg)
            yield "", chat_history, state
            
    except Exception as e:
        logger.error(f"产品推荐处理失败: {e}")
        
        # 根据错误类型提供不同的错误信息
        if "Connection" in str(e) or "RemoteDisconnected" in str(e):
            error_msg = "抱歉，网络连接出现问题，产品推荐功能暂时不可用。请检查网络连接后重试。"
        elif "timeout" in str(e) or "超时" in str(e):
            error_msg = "抱歉，产品推荐请求超时，请稍后重试。"
        elif "推荐引擎未初始化" in str(e):
            error_msg = "抱歉，推荐系统暂时不可用，请稍后重试。"
        else:
            error_msg = "抱歉，产品推荐功能暂时出现问题。请确保已完成皮肤分析，或稍后重试。"
            
        chat_history[-1] = (msg, error_msg)
        yield "", chat_history, state

def _generate_fallback_recommendation(recommendations):
    """生成备用推荐格式（当LLM调用失败时使用）"""
    rec_text = "根据您的皮肤分析结果，我为您推荐以下护肤产品：\n\n"
    
    for i, rec in enumerate(recommendations[:5], 1):
        rec_text += f"**{i}. {rec.get('product_name', '未知产品')}**\n"
        
        if rec.get('brand'):
            rec_text += f"🏷️ **品牌**：{rec['brand']}\n"
        
        if rec.get('target_concerns'):
            concerns = rec['target_concerns']
            if isinstance(concerns, list):
                rec_text += f"🎯 **针对问题**：{', '.join(concerns)}\n"
            else:
                rec_text += f"💡 **针对问题**：{concerns}\n"
        
        if rec.get('key_ingredients'):
            ingredients = rec['key_ingredients']
            if isinstance(ingredients, list):
                rec_text += f"💊 **核心成分**：{', '.join(ingredients)}\n"
            else:
                rec_text += f"💊 **核心成分**：{ingredients}\n"
        
        if rec.get('benefits'):
            benefits = rec['benefits']
            if isinstance(benefits, list):
                rec_text += f"✨ **主要功效**：{', '.join(benefits)}\n"
            else:
                rec_text += f"✨ **主要功效**：{benefits}\n"
        
        if rec.get('usage_instructions') and isinstance(rec['usage_instructions'], dict):
            method = rec['usage_instructions'].get('method', '')
            if method:
                rec_text += f"📝 **使用方法**：{method}\n"
        
        if rec.get('price'):
            rec_text += f"💰 **参考价格**：{rec['price']}\n"
        
        if rec.get('link'):
            rec_text += f"🔗 **购买链接**：[点击购买]({rec['link']})\n"
        
        # 推荐理由放在最后
        if rec.get('reason'):
            rec_text += f"💡 **推荐理由**：{rec['reason']}\n"
        else:
            rec_text += f"💡 **推荐理由**：基于您的皮肤状况，这款产品能够有效解决您的护肤需求\n"
        
        rec_text += "\n" + "─"*50 + "\n\n"
    
    rec_text += "🔍 以上推荐基于您的皮肤分析结果。如需了解更多详情或有其他问题，请随时告诉我！"
    return rec_text

def on_select_type(choice, chat_history, state_data):
    """处理咨询类型选择"""
    try:
        if not isinstance(chat_history, list):
            chat_history = []
        if not isinstance(state_data, dict):
            state_data = {}
        
        # 更新状态数据
        state_data["consultation_type"] = choice
        
        # 构造用户消息
        user_messages = {
            "为自己咨询": "我想为自己咨询",
            "为长辈咨询": "我想为长辈咨询", 
            "其他需求": "我有其他问题"
        }
        
        if choice in user_messages:
            user_msg = user_messages[choice]
            
            # 添加用户消息到聊天历史
            chat_history.append((user_msg, None))
            
            # 获取对应的初始提示
            prompt = get_initial_prompt(choice)
            
            # 添加系统回复
            chat_history.append((None, prompt))
        
        return chat_history, state_data
        
    except Exception as e:
        logger.error(f"处理咨询类型选择失败: {e}")
        return chat_history, state_data

def find_free_port():
    """找到可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def create_ui():
    """创建简化的UI界面"""
    with gr.Blocks(
        title="TimelessSkin 智能护肤顾问",
        theme=gr.themes.Soft(),
        css="""
        .main-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px !important;
        }
        /* 聊天机器人样式优化 */
        .chatbot {
            height: 650px !important;
            border-radius: 12px !important;
            border: 1px solid #E0E0FF !important;
        }
        
        /* 优化聊天气泡样式 - 极紧凑，去掉上下空白 */
        .chatbot .message {
            padding: 0px 1px !important;
            margin: 0px 0 !important;
            border-radius: 0px !important;
            max-width: 85% !important;
            line-height: 0.8 !important;
            font-size: 14px !important;
        }
        
        .chatbot .user-message {
            background-color: #f0f0f0 !important;
            margin-left: auto !important;
        }
        
        .chatbot .bot-message {
            background-color: #e8f4fd !important;
            margin-right: auto !important;
        }
        
        /* 右侧面板样式优化 */
        .right-panel {
            background: #fafafa !important;
            padding: 16px !important;
            border-radius: 12px !important;
            border: 1px solid #e0e0e0 !important;
            height: fit-content !important;
        }
        
        .instruction-box {
            background: #f8f9fa !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            margin-top: 12px !important;
        }
        
        .instruction-box h4 {
            margin-top: 0 !important;
            margin-bottom: 8px !important;
            color: #495057 !important;
            font-size: 14px !important;
        }
        
        .instruction-box ul {
            margin-bottom: 8px !important;
            padding-left: 16px !important;
        }
        
        .instruction-box li {
            margin-bottom: 4px !important;
            font-size: 13px !important;
            line-height: 1.4 !important;
        }
        .input-row {
            display: flex !important;
            gap: 12px !important;
            padding: 16px !important;
            background: white !important;
            border-top: 1px solid #E0E0FF !important;
            align-items: center !important;
        }
        .button-group button {
            min-width: unset !important;
            padding: 0 16px !important;
            height: 36px !important;
            font-size: 14px !important;
        }
        """
    ) as demo:
        gr.Markdown("## ✨ TimelessSkin 智能护肤顾问")

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    [["", "您好，我是您的智能护肤顾问，您可以选择需要咨询的类型，我能为您提供针对性的建议~"]],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                    render_markdown=True,
                    height=650
                )
                
                with gr.Row(elem_classes="input-row"):
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="请输入您的问题...",
                            container=False
                        )
                    with gr.Column(scale=1, elem_classes="button-group"):
                        submit = gr.Button("发送", variant="primary")
                        clear = gr.Button("清空")

            with gr.Column(scale=3, elem_classes="right-panel"):
                gr.Markdown("### 📋 咨询类型选择")
                consultation_type = gr.Radio(
                    choices=["为自己咨询", "为长辈咨询", "其他需求"],
                    label="请选择咨询类型",
                    value=None
                )
                
                gr.Markdown("### 📸 皮肤检测")
                upload_image = gr.Image(
                    label="上传面部照片进行皮肤分析",
                    type="pil"
                )
                analyze_btn = gr.Button("开始分析", variant="secondary")
                
                # 添加使用说明
                gr.Markdown("### 📖 使用说明")
                with gr.Box(elem_classes="instruction-box"):
                    gr.Markdown("""
                    **⚡ 主要功能：**
                    • 选择咨询类型获取针对性建议
                    • 上传照片可进行皮肤分析
                    • 直接对话获取护肤建议
                    
                    **📱 照片要求：**
                    • 照片越清晰，分析越准确
                    • 建议正面拍摄，光线充足
                    • 避免侧脸或模糊照片
                    
                    **✨ 使用建议：**
                    • 先选择咨询类型
                    • 上传清晰面部照片
                    • 根据分析结果咨询产品推荐
                    """)

        # 状态变量
        state = gr.State({})

        # 事件绑定 - 启用真正的流式输出
        msg.submit(
            user_message_and_response,
            [msg, chatbot, state],
            [msg, chatbot, state],
            queue=True,
            show_progress=True
        )
        
        submit.click(
            user_message_and_response,
            [msg, chatbot, state],
            [msg, chatbot, state],
            queue=True,
            show_progress=True
        )
        
        clear.click(
            lambda: ([], {}),
            outputs=[chatbot, state],
            queue=False
        )
        
        consultation_type.change(
            on_select_type,
            [consultation_type, chatbot, state],
            [chatbot, state],
            queue=False
        )
        
        analyze_btn.click(
            on_analyze,
            [upload_image, chatbot, state],
            [chatbot, state],
            queue=True,
            show_progress=True
        )

        return demo

if __name__ == "__main__":
    port = find_free_port()
    demo = create_ui()
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_api=False
    )
