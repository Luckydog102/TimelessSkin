import gradio as gr
import json
from pathlib import Path
import os
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import dashscope
import traceback
import sys
import base64
from PIL import Image
import io

# 设置API key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    # 如果环境变量中没有，尝试从配置文件读取
    try:
        config_path = 'config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('DASHSCOPE_API_KEY')
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")

# 这里可以设置您的API Key
if not api_key:
    api_key = "YOUR_API_KEY_HERE"  # 请替换为您的实际API Key

if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("警告: 未设置 DASHSCOPE_API_KEY")
    USE_MOCK_RESPONSES = True
else:
    print(f"使用API key: {api_key[:8]}...")
    dashscope.api_key = api_key
    USE_MOCK_RESPONSES = False

# 加载知识库数据
def load_skin_types():
    try:
        with open('src/knowledge/skin_conditions/elder_skin_types.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载皮肤类型数据失败: {str(e)}")
        return {
            "skin_types": {
                "dry": {
                    "name": "干性肤质",
                    "characteristics": ["皮肤干燥", "容易起皱", "紧绷感"],
                    "care_tips": ["使用温和清洁剂", "加强保湿", "避免过热水洗脸"]
                },
                "oily": {
                    "name": "油性肤质",
                    "characteristics": ["油光发亮", "毛孔粗大", "容易长痘"],
                    "care_tips": ["控油清洁", "选择清爽保湿", "定期去角质"]
                },
                "combination": {
                    "name": "混合性肤质",
                    "characteristics": ["T区油腻", "两颊干燥", "毛孔不均匀"],
                    "care_tips": ["分区护理", "平衡水油", "温和清洁"]
                },
                "sensitive": {
                    "name": "敏感性肤质",
                    "characteristics": ["容易发红", "瘙痒", "刺痛感"],
                    "care_tips": ["温和无刺激", "补充水分", "避免刺激成分"]
                }
            }
        }

def load_product_data():
    try:
        with open('src/knowledge/products/all_products.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载产品数据失败: {str(e)}")
        return {"products": []}

# 初始化数据
skin_types_data = load_skin_types()
product_data = load_product_data()

def format_skin_analysis(skin_type, characteristics, concerns):
    """格式化皮肤分析结果"""
    skin_info = skin_types_data["skin_types"].get(skin_type, {})
    
    analysis = f"""📊 皮肤分析结果：

🔍 基础肤质：{skin_info.get('name', skin_type)}

✨ 主要特征：
"""
    for char in characteristics:
        analysis += f"- {char}\n"
    
    analysis += "\n❗ 需要关注的问题：\n"
    for concern in concerns:
        analysis += f"- {concern}\n"
    
    analysis += "\n💡 护理建议：\n"
    for tip in skin_info.get('care_tips', []):
        analysis += f"- {tip}\n"
    
    analysis += "\n是否需要查看适合您肤质的产品推荐？"
    return analysis

def get_mock_response(message_type):
    """生成模拟响应用于测试"""
    if "为自己" in str(message_type):
        return """我理解您想为自己寻找合适的护肤方案。

请告诉我您的皮肤类型：
1. 干性肤质 - 皮肤干燥、容易起皱
2. 油性肤质 - 油光发亮、毛孔粗大
3. 混合性肤质 - T区油腻、两颊干燥
4. 敏感性肤质 - 容易发红、瘙痒

或者您可以描述您的皮肤状况，我来帮您判断。"""
    elif "为长辈" in str(message_type):
        return """很高兴您关心长辈的护肤需求。

请问长辈的年龄大概在：
1. 50-60岁
2. 60-70岁
3. 70岁以上

同时，请告诉我长辈的皮肤类型：
1. 干性肤质 - 常见于老年人，皮肤干燥、易起皱
2. 敏感性肤质 - 容易发红、瘙痒
3. 混合性肤质 - 部分区域干燥、部分区域油腻
4. 不确定 - 我可以帮您进行判断"""
    elif "图片分析" in str(message_type):
        # 使用知识库中的皮肤类型数据
        skin_type = "dry"  # 示例类型
        characteristics = [
            "皮肤偏干",
            "有细纹",
            "色素沉着"
        ]
        concerns = [
            "皮肤缺水",
            "弹性下降",
            "色斑问题"
        ]
        return format_skin_analysis(skin_type, characteristics, concerns)
    else:
        return "请告诉我更多关于您的护肤需求，我会为您提供专业的建议。"

def get_llm_response(messages, message_type="default"):
    """调用Qwen获取回复"""
    try:
        print("\n=== LLM调用开始 ===")
        print(f"消息类型: {message_type}")
        print(f"输入消息: {messages}")
        
        if USE_MOCK_RESPONSES:
            print("使用模拟响应")
            response = get_mock_response(message_type)
            print(f"模拟响应: {response}")
            return response
            
        print("调用Qwen API...")
        response = Generation.call(
            model='qwen-max',
            messages=messages,
            result_format='message',
            temperature=0.7,
            top_p=0.8,
        )
        
        print(f"API响应状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.output.choices[0].message.content
            print(f"API响应内容: {result}")
            return result
        else:
            error_msg = f"LLM API错误: {response.code} - {response.message}"
            print(error_msg)
            return "抱歉，我现在遇到了一些问题，请稍后再试。"
    except Exception as e:
        print("\n=== LLM调用错误 ===")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return get_mock_response(message_type)

def process_consultation_type(user_type, history):
    """处理用户选择的咨询类型"""
    try:
        print(f"\n=== 处理咨询类型: {user_type} ===")
        
        if not user_type:  # 如果用户取消选择
            return history if history else []
        
        message = f"我想{user_type}"
        messages = [
            {'role': Role.SYSTEM, 'content': '你是一个专业的护肤顾问。'},
            {'role': Role.USER, 'content': message}
        ]
        
        response = get_llm_response(messages, message_type=user_type)
        
        # 更新对话历史
        new_history = list(history) if history else []
        new_history.append([message, response])
        return new_history
        
    except Exception as e:
        print("\n=== 咨询类型处理错误 ===")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return history if history else []

def get_product_recommendations(skin_type, concerns):
    """根据肤质和问题推荐产品"""
    try:
        products = product_data.get('products', [])
        recommended = []
        
        # 根据皮肤类型和问题筛选产品
        for product in products:
            if skin_type in product.get('suitable_skin_types', []):
                if any(concern in product.get('tags', []) for concern in concerns):
                    recommended.append(product)
        
        if not recommended:
            return "抱歉，没有找到完全匹配的产品。建议咨询专业医生获取更专业的建议。"
        
        result = "🎯 为您推荐以下产品：\n\n"
        for i, product in enumerate(recommended[:3], 1):
            result += f"""【推荐{i}】{product['name']}
✨ 主要功效：{', '.join(product.get('tags', []))}
💰 价格：{product.get('price', '暂无价格')}
🔍 产品详情：{product.get('details', '暂无详情')}
"""
            if product.get('elder_friendly_features'):
                result += "\n👴 老年人友好特性：\n"
                for feature, desc in product['elder_friendly_features'].items():
                    result += f"- {feature}: {desc}\n"
            result += f"\n🔗 购买链接：{product.get('link', '暂无')}\n\n"
        
        result += "\n💡 温馨提示：以上推荐仅供参考，建议在使用前进行小范围测试。"
        return result
        
    except Exception as e:
        print(f"产品推荐错误: {str(e)}")
        return "抱歉，获取产品推荐时出现错误。"

def chat(message, history):
    """处理用户输入的消息"""
    try:
        print(f"\n=== 处理聊天消息: {message} ===")
        
        history = list(history) if history else []
        
        # 构建消息
        messages = [
            {'role': Role.SYSTEM, 'content': '你是一个专业的护肤顾问，请根据用户的需求提供专业的护肤建议。'},
        ]
        
        # 添加历史对话
        for h in history:
            if h[0]:  # 用户消息
                messages.append({'role': Role.USER, 'content': h[0]})
            if h[1]:  # 助手消息
                messages.append({'role': Role.ASSISTANT, 'content': h[1]})
        
        # 添加当前消息
        messages.append({'role': Role.USER, 'content': message})
        
        # 检查是否是产品推荐确认
        if history and "推荐" in history[-1][1] and message.lower() in ["是", "好", "好的", "需要", "想看"]:
            # 使用默认的干性肤质和基础护理需求作为示例
            response = get_product_recommendations("dry", ["保湿", "抗衰老"])
        else:
            response = get_llm_response(messages, message_type=message)
        
        history.append([message, response])
        return history
        
    except Exception as e:
        print("\n=== 聊天处理错误 ===")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return history

def analyze_image(image, user_type):
    """分析上传的图片并给出肤质评估"""
    try:
        print("\n=== 开始图片分析 ===")
        print(f"接收到的图片类型: {type(image)}")
        print(f"接收到的图片数据: {image}")
        
        if image is None:
            print("未接收到图片")
            return None, [[None, "请上传一张清晰的面部照片。"]]

        try:
            # 处理不同类型的图片输入
            if isinstance(image, dict):
                print(f"处理字典类型图片数据: {image.keys()}")
                if 'path' in image:
                    print(f"从路径加载图片: {image['path']}")
                    img = Image.open(image['path'])
                elif 'image' in image:
                    print("从image键加载图片")
                    img = image['image']
                else:
                    print(f"未知的字典格式: {image}")
                    return None, [[None, "图片格式不正确，请重新上传。"]]
            elif isinstance(image, str):
                print(f"处理字符串类型图片路径: {image}")
                img = Image.open(image)
            elif isinstance(image, Image.Image):
                print("处理PIL Image对象")
                img = image
            elif hasattr(image, 'read'):
                print("处理文件对象")
                img = Image.open(image)
            else:
                print(f"未知的图片格式: {type(image)}")
                return None, [[None, "不支持的图片格式，请重新上传。"]]
            
            print(f"原始图片信息 - 尺寸: {img.size}, 模式: {img.mode}")
            
            # 确保图片是RGB模式
            if img.mode != 'RGB':
                print(f"转换图片模式从 {img.mode} 到 RGB")
                img = img.convert('RGB')
            
            # 验证图片尺寸
            if img.size[0] < 100 or img.size[1] < 100:
                print(f"图片尺寸过小: {img.size}")
                return None, [[None, "图片尺寸过小，请上传更清晰的照片。"]]
            
            # 调整图片大小
            max_size = (800, 800)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                print(f"调整图片尺寸从 {img.size} 到 {max_size}")
                img.thumbnail(max_size, Image.LANCZOS)
            
            # 转换为base64
            try:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                print(f"图片转换为base64成功，长度: {len(img_str)}")
                
                # 验证base64字符串
                if len(img_str) < 100:
                    print(f"base64字符串异常短: {len(img_str)}")
                    return None, [[None, "图片处理失败，请重新上传。"]]
                
            except Exception as e:
                print(f"图片转base64失败: {str(e)}")
                return None, [[None, "图片格式转换失败，请重新上传。"]]
            
        except Exception as e:
            print("\n=== 图片处理错误 ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            return None, [[None, "图片处理过程中出现错误，请确保上传的是有效的图片文件。"]]

        print("图片处理完成，准备进行分析...")

        if USE_MOCK_RESPONSES:
            print("使用模拟数据进行测试")
            skin_type = "dry"
            characteristics = ["皮肤偏干", "有细纹", "色素沉着"]
            concerns = ["皮肤缺水", "弹性下降", "色斑问题"]
            response = format_skin_analysis(skin_type, characteristics, concerns)
        else:
            print("准备调用Qwen API...")
            try:
                # 构建带图片的消息
                messages = [
                    {
                        'role': Role.SYSTEM,
                        'content': '''你是一个专业的护肤顾问，请基于用户上传的照片进行专业的肤质分析。
分析要点包括：
1. 基础肤质类型（干性、油性、混合性、敏感性）
2. 具体的肤质特征
3. 需要关注的护肤问题
4. 适合的护理建议'''
                    },
                    {
                        'role': Role.USER,
                        'content': [
                            {
                                'image': img_str
                            },
                            {
                                'text': '请分析这张面部照片的肤质状况，包括基础肤质类型、特征和护理建议。'
                            }
                        ]
                    }
                ]
                
                print("发送API请求...")
                print(f"使用的API Key: {dashscope.api_key[:8]}...")
                
                try:
                    response = Generation.call(
                        model='qwen-vl-max',
                        messages=messages,
                        result_format='message',
                        temperature=0.7,
                        top_p=0.8,
                    )
                    
                    print(f"API响应原始数据: {response}")
                    print(f"API响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                            analysis = response.output.choices[0].message.content
                            print(f"API返回的分析结果: {analysis}")
                            
                            try:
                                # 提取关键信息
                                skin_type = "dry"  # 默认值
                                characteristics = []
                                concerns = []
                                
                                # 简单的文本分析来提取信息
                                if "干性" in analysis or "干燥" in analysis:
                                    skin_type = "dry"
                                elif "油性" in analysis or "油腻" in analysis:
                                    skin_type = "oily"
                                elif "混合性" in analysis:
                                    skin_type = "combination"
                                elif "敏感" in analysis:
                                    skin_type = "sensitive"
                                
                                # 提取特征和问题
                                lines = analysis.split('\n')
                                for line in lines:
                                    if "特征" in line or "现象" in line:
                                        char = line.split("：")[-1].strip()
                                        if char:
                                            characteristics.append(char)
                                    if "问题" in line or "建议" in line:
                                        con = line.split("：")[-1].strip()
                                        if con:
                                            concerns.append(con)
                                
                                # 如果没有提取到特征和问题，使用整个分析文本
                                if not characteristics and not concerns:
                                    print("未能提取到结构化信息，使用原始分析文本")
                                    return image, [[None, analysis]]
                                
                                response = format_skin_analysis(skin_type, characteristics, concerns)
                            except Exception as e:
                                print(f"分析结果解析错误: {str(e)}")
                                print("使用原始分析文本")
                                return image, [[None, analysis]]
                        else:
                            print("API响应格式错误")
                            return None, [[None, "抱歉，服务器返回格式错误，请稍后重试。"]]
                    else:
                        error_msg = getattr(response, 'message', '未知错误')
                        print(f"API调用失败: {response.status_code} - {error_msg}")
                        return None, [[None, f"抱歉，图片分析服务返回错误（{response.status_code}），请稍后重试。"]]
                        
                except Exception as api_error:
                    print(f"API调用异常: {str(api_error)}")
                    print("详细错误信息:")
                    traceback.print_exc(file=sys.stdout)
                    return None, [[None, "抱歉，调用分析服务时出现错误，请稍后重试。"]]
                    
            except Exception as e:
                print(f"API请求准备失败: {str(e)}")
                print("详细错误信息:")
                traceback.print_exc(file=sys.stdout)
                return None, [[None, "抱歉，准备分析请求时出现错误，请稍后重试。"]]
        
        print("分析完成，返回结果")
        return image, [[None, response]]
        
    except Exception as e:
        print("\n=== 图片分析错误 ===")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return None, [[None, "图片分析过程中出现错误，请稍后重试。"]]

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌟 TimelessSkin 智能护肤顾问")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                show_label=False,
                height=600,
                bubble_full_width=False,
                value=[]
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="输入您的问题",
                    placeholder="请输入您的问题...",
                    show_label=False,
                    scale=8
                )
                submit = gr.Button("发送", scale=1)
                clear = gr.Button("清除", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 👥 咨询类型")
            user_type = gr.Radio(
                choices=["为自己咨询", "为长辈咨询", "其他需求"],
                label="您是为谁咨询？",
                info="请选择咨询类型",
                value=None
            )
            
            gr.Markdown("### 📸 面部照片分析")
            image_input = gr.Image(
                label="上传面部照片（可选）",
                type="pil",
                height=300
            )
            analyze_btn = gr.Button("开始分析", variant="primary")
            
            gr.Markdown("""
            ### ℹ️ 使用说明
            1. 选择咨询类型
            2. 上传照片或直接对话
            3. 根据提示回答问题
            4. 获取个性化护肤建议
            """)
    
    # 事件处理
    msg.submit(chat, [msg, chatbot], [chatbot])
    submit.click(chat, [msg, chatbot], [chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)
    analyze_btn.click(
        analyze_image,
        [image_input, user_type],
        [image_input, chatbot]
    )
    
    # 添加咨询类型选择的事件处理
    user_type.change(
        process_consultation_type,
        [user_type, chatbot],
        [chatbot]
    )

print("\n=== 启动服务器 ===")
# 启动服务器
if __name__ == "__main__":
    demo.launch() 