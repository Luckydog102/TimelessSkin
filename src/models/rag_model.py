from typing import Any, Dict, List, Optional
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path
import re
import dashscope
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import logging

# 配置日志
logger = logging.getLogger(__name__)

class QwenChatModel(LLM):
    """阿里云千问聊天模型的 LangChain 封装"""
    
    model_name: str = "qwen-max"
    """模型名称"""
    
    api_key: Optional[str] = None
    """API Key"""
    
    temperature: float = 0.7
    """温度参数"""
    
    top_p: float = 0.8
    """Top P 参数"""
    
    def __init__(self, model_name: str = "qwen-max", api_key: Optional[str] = None, **kwargs):
        """初始化模型"""
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided")
            
        # 设置其他参数
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.8)
            
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """生成回复"""
        # 调用千问 API
        response = dashscope.Generation.call(
            model=self.model_name,
            prompt=prompt,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs
        )
        
        if response.status_code == 200:
            return response.output.text
        else:
            raise Exception(f"API call failed: {response.message}")
            
    @property
    def _llm_type(self) -> str:
        """返回模型类型"""
        return "qwen-chat"

class RAGModel:
    """基于 RAG 的皮肤分析和产品推荐系统"""
    
    def __init__(self):
        try:
            load_dotenv()
            # 优先从.env加载，如果失败则尝试从env.txt加载
            self.api_key = os.getenv("VLM_API_KEY")
            if not self.api_key:
                try:
                    with open('env.txt', 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith('VLM_API_KEY='):
                                self.api_key = line.strip().split('=', 1)[1]
                                break
                except Exception as e:
                    logger.warning(f"无法从env.txt加载API密钥: {e}")
            
            if self.api_key:
                os.environ["DASHSCOPE_API_KEY"] = self.api_key
                dashscope.api_key = self.api_key
                logger.info("成功设置API密钥")
            else:
                logger.warning("未找到API密钥")
                
            self.all_products = {}
            self.elder_products = {}
            self._initialized = False
            
        except Exception as e:
            logger.error(f"RAG模型初始化失败: {e}")
            self._initialized = False
        
    def initialize(self) -> None:
        """初始化产品数据，并自动补全标准化字段"""
        try:
            # 加载所有产品信息
            all_products_path = "src/knowledge/products/all_products.json"
            if os.path.exists(all_products_path):
                with open(all_products_path, "r", encoding="utf-8") as f:
                    self.all_products = json.load(f)
                    logger.info(f"成功加载所有产品数据: {len(self.all_products.get('products', []))}个产品")
            else:
                logger.warning(f"产品数据文件不存在: {all_products_path}")
                self.all_products = {"products": []}
            # 加载老年人专属产品信息
            elder_products_path = "src/knowledge/products/elder_care/loreal_elder_products.json"
            if os.path.exists(elder_products_path):
                with open(elder_products_path, "r", encoding="utf-8") as f:
                    self.elder_products = json.load(f)
                    logger.info(f"成功加载老年人产品数据: {len(self.elder_products.get('products', []))}个产品")
            else:
                logger.warning(f"老年人产品数据文件不存在: {elder_products_path}")
                self.elder_products = {"products": []}

            # 自动补全和标准化字段
            def normalize_product(prod):
                # 补全适用年龄
                if not (prod.get("suitable_age") or prod.get("age")):
                    prod["suitable_age"] = "0+"
                # 补全适用肤质
                if not (prod.get("suitable_skin_types") or prod.get("skin_type")):
                    prod["suitable_skin_types"] = []
                # 补全功效
                if not (prod.get("benefits") or prod.get("tags") or prod.get("target_concerns")):
                    prod["benefits"] = []
                # 统一字符串为列表
                for k in ["suitable_skin_types", "benefits", "tags", "target_concerns", "key_ingredients"]:
                    if k in prod and isinstance(prod[k], str):
                        prod[k] = [prod[k]]
                return prod

            if "products" in self.all_products:
                self.all_products["products"] = [normalize_product(p) for p in self.all_products["products"]]
            if "products" in self.elder_products:
                self.elder_products["products"] = [normalize_product(p) for p in self.elder_products["products"]]

            # 设置初始化标志
            self._initialized = True
            logger.info("RAG模型初始化成功")
        except Exception as e:
            logger.error(f"加载产品数据失败: {str(e)}")
            self.all_products = {"products": []}
            self.elder_products = {"products": []}
            self._initialized = True
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """从知识库检索相关产品信息
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            产品信息列表
        """
        try:
            # 确保已初始化
            if not self._initialized:
                self.initialize()
                
            # 检查产品数据是否为空
            # logger.info(f"🔥 RAG产品数据检查: all_products数量={len(self.all_products.get('products', []))}, elder_products数量={len(self.elder_products.get('products', []))}")
            
            if not self.all_products.get("products") and not self.elder_products.get("products"):
                logger.warning(f"🔥 产品数据为空，无法检索 all_products: {self.all_products}, elder_products: {self.elder_products}")
                logger.warning(f"🔥 触发fallback机制，返回默认产品")
                return self._get_default_products(top_k)
                
            # 解析查询条件
            conditions = self._parse_query(query)
            logger.info(f"[RAG调试] 检索条件: {conditions}")
            
            # 根据年龄选择产品库
            products = self.elder_products.get("products", []) if conditions.get("age", 0) >= 50 else self.all_products.get("products", [])
            logger.info(f"[RAG调试] 选择产品库数量: {len(products)} 条件: {conditions}")
            
            # 如果没有产品，返回默认产品
            if not products:
                logger.warning("选择的产品库为空，返回默认产品")
                return self._get_default_products(top_k)
            
            # 筛选符合条件的产品
            matched_products = []
            for idx, product in enumerate(products):
                # 输出产品关键字段，便于比对
                logger.info(f"[RAG调试] 产品{idx}: name={product.get('name', product)} | suitable_age={product.get('suitable_age', product.get('age', ''))} | suitable_skin_types={product.get('suitable_skin_types', product.get('skin_type', ''))} | benefits={product.get('benefits', product.get('tags', product.get('target_concerns', '')))}")
                score = self._match_score(product, conditions)
                logger.info(f"[RAG调试] 产品{idx}: name={product.get('name', product)} 匹配分数: {score} 条件: {conditions}")
                
                # 性别严格筛选：如果指定了性别，必须严格匹配
                if "gender" in conditions:
                    product_name = product.get("product_name") or product.get("name") or ""
                    product_text = product_name.lower()
                    
                    if conditions["gender"] == "female":
                        # 女性用户：严格排除男士产品
                        if "男士" in product_text or "男性" in product_text or "男" in product_text:
                            logger.debug(f"女性用户，排除男士产品: {product_name}")
                            continue
                    elif conditions["gender"] == "male":
                        # 男性用户：严格排除女士产品
                        if "女士" in product_text or "女性" in product_text or "女" in product_text:
                            logger.debug(f"男性用户，排除女士产品: {product_name}")
                            continue
                
                # 修改筛选条件：收集所有产品，按分数排序后取最高的
                matched_products.append((score, product))
                logger.debug(f"产品匹配: {product.get('name', product)} 条件: {conditions} 匹配分数: {score}")
            
            # 如果没有匹配的产品，返回默认产品
            if not matched_products:
                logger.warning(f"🔥 没有找到匹配的产品，条件: {conditions}，产品库数量: {len(products)}")
                logger.warning(f"🔥 触发fallback机制，返回默认产品")
                # 详细输出所有产品和条件
                for product in products[:5]:  # 只输出前5个产品避免日志过长
                    logger.warning(f"🔥 产品: {product.get('name', product)}，字段: {product}")
                return self._get_default_products(top_k)
            
            # 按匹配分数排序
            matched_products.sort(key=lambda x: x[0], reverse=True)
            
            logger.info(f"产品匹配结果: 总产品{len(products)}个，匹配{len(matched_products)}个，返回前{top_k}个")
            for i, (score, product) in enumerate(matched_products[:top_k]):
                logger.info(f"推荐产品{i+1}: {product.get('name', '未知')} 分数: {score:.2f}")
            
            # 返回前 top_k 个产品
            return [
                {
                    "product_name": p.get("product_name") or p.get("name", "未知产品"),
                    "product_type": p.get("product_type") or p.get("category", "护肤品"),
                    "target_concerns": self._extract_skin_concerns(p),
                    "key_ingredients": p.get("key_ingredients") or p.get("ingredients") or [],
                    "benefits": p.get("benefits") or p.get("effects") or p.get("tags") or [],
                    "usage_frequency": p.get("usage_frequency") or "每日",
                    "usage_method": p.get("usage_method") or p.get("usage") or "按照产品说明使用",
                    "usage_timing": p.get("usage_timing") or "早晚",
                    "precautions": p.get("precautions") or "",
                    "recommendation_reason": self._generate_reason(p, conditions),
                    "expected_results": p.get("expected_results") or "改善肌肤状况",
                    "lifestyle_tips": p.get("lifestyle_tips") or [],
                    "price": p.get("price") or "",
                    "specification": p.get("specification") or "",
                    "details": p.get("details") or "",
                    "link": p.get("link") or ""  # 添加产品链接
                }
                for _, p in matched_products[:top_k]
            ]
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return self._get_default_products(top_k)
    
    def _get_default_products(self, count: int = 3) -> List[Dict[str, Any]]:
        """返回默认产品列表"""
        default_products = [
            {
                "product_name": "欧莱雅复颜玻尿酸水光充盈导入精华面霜",
                "product_type": "面霜",
                "target_concerns": ["干燥", "细纹", "暗沉"],
                "key_ingredients": ["玻尿酸", "神经酰胺", "维生素E"],
                "benefits": ["保湿", "抗皱", "提亮"],
                "usage_frequency": "每日两次",
                "usage_method": "洁面后，取适量均匀涂抹于面部",
                "usage_timing": "早晚",
                "precautions": "避免接触眼睛",
                "recommendation_reason": "含有高浓度玻尿酸，能深层补水，改善干纹，提亮肤色",
                "expected_results": "使用2周后肌肤更加水润透亮",
                "lifestyle_tips": ["多喝水", "注意防晒"]
            },
            {
                "product_name": "欧莱雅清润葡萄籽精华液",
                "product_type": "精华液",
                "target_concerns": ["抗氧化", "暗沉", "细纹"],
                "key_ingredients": ["葡萄籽提取物", "维生素C", "透明质酸"],
                "benefits": ["抗氧化", "提亮", "保湿"],
                "usage_frequency": "每日两次",
                "usage_method": "洁面后，取3-4滴轻拍吸收",
                "usage_timing": "早晚",
                "precautions": "避免接触眼睛",
                "recommendation_reason": "富含葡萄籽抗氧化成分，能有效对抗自由基，改善肤色不均",
                "expected_results": "使用4周后肌肤更加明亮有弹性",
                "lifestyle_tips": ["均衡饮食", "充足睡眠"]
            },
            {
                "product_name": "欧莱雅青春密码活颜精华肌底液",
                "product_type": "精华液",
                "target_concerns": ["衰老", "弹性", "细纹"],
                "key_ingredients": ["益生菌提取物", "透明质酸", "烟酰胺"],
                "benefits": ["抗衰老", "紧致", "修护"],
                "usage_frequency": "每日两次",
                "usage_method": "洁面后第一步使用，轻拍至吸收",
                "usage_timing": "早晚",
                "precautions": "敏感肌肤请先做皮肤测试",
                "recommendation_reason": "含有独特益生菌成分，能激活肌肤自身修护能力，改善肌肤弹性",
                "expected_results": "使用8周后肌肤更加紧致有弹性",
                "lifestyle_tips": ["避免熬夜", "定期做面部按摩"]
            }
        ]
        return default_products[:count]
            
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """解析查询文本，提取关键条件"""
        conditions = {}
        
        # 提取年龄信息
        age_patterns = [
            (r'(\d+)\s*岁', lambda m: int(m.group(1))),
            (r'(\d+)\+', lambda m: int(m.group(1))),
            (r'(\d+)岁以上', lambda m: int(m.group(1)))
        ]
        
        for pattern, extract in age_patterns:
            match = re.search(pattern, query)
            if match:
                conditions["age"] = extract(match)
                break
                
        if "老年" in query or "年长" in query:
            conditions["age"] = conditions.get("age", 60)
            
        # 提取性别信息
        if "女士" in query or "女性" in query or "女" in query:
            conditions["gender"] = "female"
        elif "男士" in query or "男性" in query or "男" in query:
            conditions["gender"] = "male"
        
        # 如果查询中包含排除关键词，也考虑性别
        if "-男士" in query:
            conditions["gender"] = "female"
        elif "-女士" in query:
            conditions["gender"] = "male"
            
        # 提取肤质信息
        skin_types = {
            "干性": ["干性", "干燥", "缺水"],
            "油性": ["油性", "出油", "油腻"],
            "混合性": ["混合性", "T区油", "混合"],
            "敏感": ["敏感", "过敏", "红肿"]
        }
        
        for skin_type, keywords in skin_types.items():
            if any(keyword in query for keyword in keywords):
                conditions["skin_type"] = skin_type
                break
                
        # 提取护肤需求
        concern_mapping = {
            "补水": ["补水", "缺水", "干燥"],
            "保湿": ["保湿", "滋润"],
            "抗皱": ["抗皱", "皱纹", "细纹"],
            "紧致": ["紧致", "紧肤", "提拉"],
            "美白": ["美白", "提亮", "暗沉", "美白", "淡斑"],
            "控油": ["控油", "油脂", "出油"],
            "抗衰老": ["抗衰", "抗衰老", "衰老"],
            "舒缓": ["舒缓", "镇静", "红肿"],
            "修复": ["修复", "修护", "受损"]
        }
        
        concerns = []
        for concern, keywords in concern_mapping.items():
            if any(keyword in query for keyword in keywords):
                concerns.append(concern)
                
        conditions["concerns"] = concerns
        
        return conditions
        
    def _match_score(self, product: Dict[str, Any], conditions: Dict[str, Any]) -> float:
        """计算产品与查询条件的匹配分数，兼容字段为字符串或列表，字段名宽松匹配"""
        try:
            score = 0.0
            
            # 基础分数 - 确保每个产品都有基础分
            score += 0.1
            
            # 性别匹配（重要权重）
            if "gender" in conditions:
                product_name = product.get("product_name") or product.get("name") or ""
                product_details = product.get("details") or ""
                product_text = f"{product_name} {product_details}".lower()
                
                if conditions["gender"] == "female":
                    # 女性用户：男士产品扣分，女士产品加分
                    if "男士" in product_text or "男性" in product_text or "男" in product_text:
                        score -= 2.0  # 大幅扣分
                    elif "女士" in product_text or "女性" in product_text or "女" in product_text:
                        score += 3.0  # 大幅加分
                    else:
                        score += 1.0  # 通用产品给高分
                elif conditions["gender"] == "male":
                    # 男性用户：女士产品扣分，男士产品加分
                    if "女士" in product_text or "女性" in product_text or "女" in product_text:
                        score -= 2.0  # 大幅扣分
                    elif "男士" in product_text or "男性" in product_text or "男" in product_text:
                        score += 3.0  # 大幅加分
                    else:
                        score += 1.0  # 通用产品给高分
            
            # 年龄匹配
            if "age" in conditions:
                product_age_str = product.get("suitable_age") or product.get("age") or "0+"
                product_age = 0
                match = re.search(r'(\d+)', str(product_age_str))
                if match:
                    product_age = int(match.group(1))
                if product_age <= conditions["age"]:
                    score += 1.5
                else:
                    score += 0.5  # 年龄不匹配也给基础分
            
            # 肤质匹配
            if "skin_type" in conditions:
                product_skin_types = product.get("suitable_skin_types") or product.get("skin_type") or []
                if isinstance(product_skin_types, str):
                    product_skin_types = [product_skin_types]
                if conditions["skin_type"] in product_skin_types:
                    score += 2.0  # 肤质匹配给高分
                else:
                    score += 0.3  # 肤质不匹配也给基础分
            
            # 护肤需求匹配 - 支持多种字段名
            if conditions.get("concerns"):
                # 尝试从多个可能的字段获取产品功效信息
                product_benefits = []
                
                # 按优先级尝试不同字段
                benefit_fields = [
                    "benefits", "tags", "target_concerns", "effects", 
                    "ingredients", "key_ingredients", "categories"
                ]
                
                for field in benefit_fields:
                    if field in product:
                        field_value = product[field]
                        if isinstance(field_value, list):
                            product_benefits.extend(field_value)
                        elif isinstance(field_value, str):
                            product_benefits.append(field_value)
                
                # 去重
                product_benefits = list(set(product_benefits))
                
                # 计算匹配度
                matched_concerns = 0
                for concern in conditions["concerns"]:
                    for benefit in product_benefits:
                        if concern in benefit or benefit in concern:
                            matched_concerns += 1
                            break
                
                if matched_concerns > 0:
                    score += matched_concerns * 1.0  # 需求匹配给高分
                else:
                    score += 0.2  # 需求不匹配也给基础分
            
            # 产品名称匹配（如果查询中包含产品类型关键词）
            product_name = product.get("product_name") or product.get("name") or ""
            if product_name:
                # 检查产品名称是否包含查询中的关键词
                for concern in conditions.get("concerns", []):
                    if concern in product_name:
                        score += 0.5
                        break
            
            # 确保分数不为负数
            score = max(score, 0.1)
            
            logger.debug(f"产品 {product.get('product_name', '未知')} 匹配分数: {score:.2f}, 条件: {conditions}")
            
            return score
            
        except Exception as e:
            logger.error(f"计算匹配分数失败: {e}，产品: {product}，条件: {conditions}")
            return 0.1  # 出错时返回基础分而不是0
        
    def _generate_reason(self, product: Dict[str, Any], conditions: Dict[str, Any]) -> str:
        """生成推荐理由"""
        try:
            reasons = []
            
            # 根据年龄
            if "age" in conditions:
                if "elder_friendly_features" in product:
                    reasons.append(f"专为{product.get('suitable_age', '50+')}人群设计")
                    
            # 根据功效
            benefits = product.get("benefits") or product.get("effects") or product.get("tags") or []
            if benefits:
                if isinstance(benefits, str):
                    benefits = [benefits]
                if benefits:
                    reasons.append(f"具有{', '.join(benefits[:3])}等功效")
                
            # 根据成分
            ingredients = product.get("key_ingredients") or product.get("ingredients") or []
            if ingredients:
                if isinstance(ingredients, str):
                    ingredients = [ingredients]
                if ingredients:
                    reasons.append(f"含有{', '.join(ingredients[:2])}等有效成分")
            
            # 根据产品类型
            product_type = product.get("product_type") or product.get("category") or ""
            if product_type:
                reasons.append(f"适合{product_type}护理")
                
            # 根据护肤需求匹配
            if conditions.get("concerns"):
                matched_concerns = []
                for concern in conditions["concerns"]:
                    # 检查产品名称、功效、成分中是否包含该需求
                    product_text = f"{product.get('product_name', '')} {product.get('details', '')}"
                    if concern in product_text:
                        matched_concerns.append(concern)
                
                if matched_concerns:
                    reasons.append(f"针对{', '.join(matched_concerns)}问题")
                
            # 使用特点
            if "elder_friendly_features" in product:
                features = product["elder_friendly_features"]
                if "usage_instructions" in features:
                    reasons.append(features["usage_instructions"])
                    
            # 如果没有生成任何理由，添加默认理由
            if not reasons:
                reasons.append("根据您的需求推荐")
                
            return "；".join(reasons)
        except Exception as e:
            logger.error(f"生成推荐理由失败: {e}")
            return "根据您的需求推荐"
        
    def get_product_recommendations(self, skin_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据皮肤分析结果获取产品推荐"""
        try:
            # 构建查询条件
            query = ""
            
            # 添加性别信息 - 优先使用skin_analysis中的gender字段
            if "性别" in skin_analysis:
                query += f"{skin_analysis['性别']} "
            elif "gender" in skin_analysis:
                query += f"{skin_analysis['gender']} "
            elif "女性" in str(skin_analysis) or "女士" in str(skin_analysis) or "女" in str(skin_analysis):
                query += "女性 "
            elif "男性" in str(skin_analysis) or "男士" in str(skin_analysis) or "男" in str(skin_analysis):
                query += "男性 "
            
            # 如果检测到性别，优化查询策略
            if "女性" in query:
                query += "护肤 -男士"  # 女性用户排除男士产品
            elif "男性" in query:
                query += "护肤"  # 男性用户包含所有护肤产品
                
            # 添加年龄信息
            if "年龄" in skin_analysis:
                query += f"{skin_analysis['年龄']} "
            elif "老年" in str(skin_analysis):
                query += "老年人 "
                
            # 添加皮肤类型
            if "皮肤类型" in skin_analysis:
                query += f"{skin_analysis['皮肤类型']} "
                
            # 添加皮肤问题
            if "主要皮肤问题" in skin_analysis:
                problems = skin_analysis["主要皮肤问题"]
                if isinstance(problems, list):
                    query += " ".join(problems)
                else:
                    query += str(problems)
            elif "主要特征" in skin_analysis:
                query += f"{skin_analysis['主要特征']} "
                
            # 如果查询为空，添加一些默认关键词
            if not query.strip():
                query = "保湿 抗皱 修护"
                
            logger.info(f"产品推荐查询: {query}")
                    
            # 获取推荐
            recommendations = self.retrieve(query, top_k=3)
            
            return {
                "success": True,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"获取产品推荐失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": self._get_default_products(3)
            } 
    
    def _extract_skin_concerns(self, product: Dict[str, Any]) -> List[str]:
        """从产品信息中提取皮肤问题"""
        try:
            # 首先尝试获取专门的target_concerns字段
            if product.get("target_concerns"):
                return product["target_concerns"]
            
            # 从产品名称和详情中推断皮肤问题
            product_name = str(product.get("product_name", "")).lower()
            product_details = str(product.get("details", "")).lower()
            categories = product.get("categories", [])
            
            skin_concerns = []
            
            # 根据产品名称和详情推断皮肤问题
            if any(keyword in product_name or keyword in product_details for keyword in ["痘痘", "粉刺", "痤疮", "控油", "清痘"]):
                skin_concerns.extend(["痘痘", "控油", "毛孔粗大"])
            elif any(keyword in product_name or keyword in product_details for keyword in ["皱纹", "细纹", "抗老", "紧致", "弹性"]):
                skin_concerns.extend(["皱纹", "松弛", "老化"])
            elif any(keyword in product_name or keyword in product_details for keyword in ["干燥", "缺水", "保湿", "补水"]):
                skin_concerns.extend(["干燥", "缺水", "粗糙"])
            elif any(keyword in product_name or keyword in product_details for keyword in ["色斑", "暗沉", "美白", "淡斑", "提亮"]):
                skin_concerns.extend(["色斑", "暗沉", "肤色不均"])
            elif any(keyword in product_name or keyword in product_details for keyword in ["敏感", "过敏", "修护", "维稳"]):
                skin_concerns.extend(["敏感", "过敏", "屏障受损"])
            
            # 从categories字段推断
            if categories:
                for category in categories:
                    if isinstance(category, str):
                        if "控油" in category or "清痘" in category:
                            if "痘痘" not in skin_concerns:
                                skin_concerns.append("痘痘")
                        elif "抗老" in category or "紧致" in category:
                            if "皱纹" not in skin_concerns:
                                skin_concerns.append("皱纹")
                        elif "保湿" in category or "补水" in category:
                            if "干燥" not in skin_concerns:
                                skin_concerns.append("干燥")
                        elif "美白" in category or "淡斑" in category:
                            if "色斑" not in skin_concerns:
                                skin_concerns.append("色斑")
            
            # 如果没有推断出皮肤问题，返回基础护理
            if not skin_concerns:
                skin_concerns = ["基础护理", "温和修护"]
            
            # 去重并限制数量
            unique_concerns = []
            for concern in skin_concerns:
                if concern not in unique_concerns:
                    unique_concerns.append(concern)
            
            return unique_concerns[:3]  # 最多返回3个皮肤问题
            
        except Exception as e:
            logger.error(f"提取皮肤问题失败: {e}")
            return ["基础护理", "温和修护"] 