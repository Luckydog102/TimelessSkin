from typing import Dict, List, Any
import sys
import os
import logging
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.config.settings import PRODUCT_RULES
import json
import traceback
import re

# 设置logger
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """产品推荐引擎"""
    
    def __init__(self):
        pass  # 移除KnowledgeLoader的初始化
        
    def generate_recommendations(self,
                               skin_conditions: Dict[str, float],
                               user_profile: Dict[str, Any],
                               product_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于皮肤状况生成产品推荐"""
        try:
            # 验证输入参数
            if not skin_conditions or not isinstance(skin_conditions, dict):
                raise ValueError("无效的皮肤状况数据")
                
            # 确保必要字段
            if not user_profile:
                user_profile = {}
            elif not isinstance(user_profile, dict):
                user_profile = {"raw_profile": user_profile}
            
            # 检查是否有产品信息
            if not product_info:
                print("⚠️ 没有可用的产品信息，可能是RAG检索失败")
                return [{
                    "product_name": "暂无产品推荐",
                    "brand": "系统提示",
                    "reason": "产品检索服务暂时不可用，请稍后重试或联系客服"
                }]
            
            print(f"📊 开始产品匹配，皮肤状况: {skin_conditions}, 可用产品: {len(product_info)}个")
            
            # 匹配产品
            matched_products = self._match_products(
                skin_conditions=skin_conditions,
                user_profile=user_profile,
                product_info=product_info
            )
            
            if matched_products:
                print(f"✅ 成功匹配 {len(matched_products)} 个产品")
                return matched_products
            else:
                print("⚠️ 没有找到匹配的产品")
                return [{
                    "product_name": "暂无匹配产品",
                    "brand": "系统提示", 
                    "reason": "暂时没有找到完全匹配您需求的产品，建议咨询客服获取个性化推荐"
                }]
            
        except Exception as e:
            print(f"❌ 推荐生成失败: {str(e)}")
            print(f"详细错误: {traceback.format_exc()}")
            return [{
                "product_name": "推荐系统异常",
                "brand": "系统提示",
                "reason": "推荐系统暂时出现问题，建议稍后重试或联系客服"
            }]
    def _get_product_rules(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """获取适用的产品匹配规则（不基于年龄）"""
        try:
            return PRODUCT_RULES["default"]
        except Exception as e:
            print(f"❌ 获取产品规则失败: {str(e)}")
            return PRODUCT_RULES["default"]

    def _standardize_age_field(self, profile: Dict[str, Any]) -> str:
        """标准化年龄字段处理"""
        age_group = ""
        # 尝试从不同字段获取年龄信息
        for field in ["age_group", "suitable_ages", "target_age"]:
            if field in profile:
                age_value = profile[field]
                if isinstance(age_value, list):
                    age_group = str(age_value[0]).lower() if age_value else ""
                else:
                    age_group = str(age_value).lower()
                if age_group:
                    break
        
        return age_group or "中年"  # 默认值

    def _get_product_ages(self, product: Dict[str, Any]) -> List[str]:
        """从产品信息中提取支持的年龄范围"""
        for field in ["suitable_age", "suitable_ages", "age_group", "target_age"]:
            if field in product:
                if isinstance(product[field], list):
                    return [str(age).lower() for age in product[field]]
                elif isinstance(product[field], str):
                    return [str(product[field]).lower()]
        return ["中年"]  # 默认值

    def _is_age_match(self, user_age: str, product_ages: List[str]) -> bool:
        """检查用户年龄与产品支持年龄是否匹配"""
        return any(
            user_age == product_age or 
            user_age in product_age or 
            product_age in user_age
            for product_age in product_ages
        )

    def _calculate_match_score(self, 
                             product: Dict[str, Any],
                             skin_conditions: Dict[str, float],
                             user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """计算产品匹配分数"""
        total_score = 0
        reasons = []
        
        # 添加随机性因素，避免每次都推荐相同产品
        import random
        import time
        random.seed(int(time.time() * 1000) % 10000)
        random_factor = random.uniform(0.95, 1.05)  # 5%的随机波动
        
        # 1. 性别匹配分数（重要权重）
        gender_score = 0
        product_name = str(product.get("product_name", "")).lower()
        product_details = str(product.get("details", "")).lower()
        product_text = f"{product_name} {product_details}"
        
        # 从user_profile中获取皮肤分析信息
        skin_analysis_data = None
        skin_analysis = ""  # 初始化skin_analysis变量
        if "skin_analysis" in user_profile:
            skin_analysis_data = user_profile["skin_analysis"]
        elif "raw_profile" in user_profile:
            skin_analysis_data = user_profile["raw_profile"]
        
        # 确保skin_analysis变量被正确初始化
        if skin_analysis_data:
            skin_analysis = str(skin_analysis_data).lower()
        
        # 从user_profile中获取检测到的性别信息
        detected_gender = user_profile.get("detected_gender", "未检测到")
        logger.info(f"[推荐引擎调试] 接收到的detected_gender: {detected_gender}")
        
        # 改进的性别识别逻辑 - 从多种数据源获取性别信息
        is_female = False
        is_male = False
        
        # 方法1：优先使用从state中获取的性别信息
        if detected_gender != "未检测到":
            if detected_gender == "女性":
                is_female = True
                logger.info(f"[性别检测调试] 从state检测到女性: {detected_gender}")
            elif detected_gender == "男性":
                is_male = True
                logger.info(f"[性别检测调试] 从state检测到男性: {detected_gender}")
        
        # 方法2：从字典类型的皮肤分析数据中获取
        if not is_female and not is_male and isinstance(skin_analysis_data, dict):
            gender = skin_analysis_data.get("性别", "").lower()
            if gender == "女性" or gender == "女":
                is_female = True
                logger.info(f"[性别检测调试] 从字典数据检测到女性: {gender}")
            elif gender == "男性" or gender == "男":
                is_male = True
                logger.info(f"[性别检测调试] 从字典数据检测到男性: {gender}")
        
        # 方法2：从字符串类型的皮肤分析中查找关键词
        if not is_female and not is_male and skin_analysis:
            logger.info(f"[性别检测调试] 从皮肤分析检测结果: is_female={is_female}, is_male={is_male}")
            logger.info(f"[性别检测调试] 皮肤分析文本: {skin_analysis[:100]}...")
            
            # 女性关键词检测
            female_keywords = [
                "女性", "女士", "女", "woman", "female", "女性专用", "女士专用",
                "女孩", "女生", "女性用户", "女性肌肤", "女性皮肤", "女性护肤",
                "她", "她的", "女性朋友", "女性客户", "女性消费者",
                "女性面部", "女性特征", "女性轮廓", "女性皮肤", "女性肤质"
            ]
            is_female = any(word in skin_analysis for word in female_keywords)
            
            # 男性关键词检测
            male_keywords = [
                "男性", "男士", "男", "man", "male", "男性专用", "男士专用",
                "男孩", "男生", "男性用户", "男性肌肤", "男性皮肤", "男性护肤",
                "他", "他的", "男性朋友", "男性客户", "男性消费者",
                "男性面部", "男性特征", "男性轮廓", "男性皮肤", "男性肤质"
            ]
            is_male = any(word in skin_analysis for word in male_keywords)
            
            logger.info(f"[性别检测调试] 检测结果: is_female={is_female}, is_male={is_male}")
        
        # 方法3：从产品名称推断（作为备用方案）
        if not is_female and not is_male:
            if "女士" in product_name or "女性" in product_name or "女" in product_name:
                is_female = True
                logger.info(f"[性别检测调试] 从产品名称推断为女性: {product_name}")
            elif "男士" in product_name or "男性" in product_name or "男" in product_name:
                is_male = True
                logger.info(f"[性别检测调试] 从产品名称推断为男性: {product_name}")
        
        # 记录最终性别检测结果
        if is_female:
            logger.info(f"[性别检测调试] 最终检测结果: 女性")
        elif is_male:
            logger.info(f"[性别检测调试] 最终检测结果: 男性")
        else:
            logger.info(f"[性别检测调试] 最终检测结果: 未检测到")
        
        # 性别严格匹配逻辑 - 修复后的版本
        if is_female:
            # 女性用户：严格排除男士产品
            if any(word in product_text for word in ["男士", "男性", "男", "man", "male", "男性专用", "男士专用"]):
                gender_score = -100.0  # 大幅扣分，确保被过滤掉
                reasons.append("❌ 男士专用产品，不适合女性使用")
                # 直接返回极低分数，确保这个产品被排除
                return {
                    "product": {
                        **product,
                        "match_score": -100.0,
                        "reason": "❌ 男士专用产品，不适合女性使用"
                    },
                    "score": -100.0
                }
            elif any(word in product_text for word in ["女士", "女性", "女", "woman", "female", "女性专用", "女士专用"]):
                gender_score = 2.0   # 女士专用产品给基础分
                reasons.append("✅ 女士专用产品，针对性更强")
            else:
                gender_score = 2.0   # 通用产品给基础分
                reasons.append("✅ 通用产品，适合所有性别")
        elif is_male:
            # 男性用户：严格排除女士产品
            if any(word in product_text for word in ["女士", "女性", "女", "woman", "female", "女性专用", "女士专用"]):
                gender_score = -100.0  # 大幅扣分，确保被过滤掉
                reasons.append("❌ 女士专用产品，不适合男性使用")
                # 直接返回极低分数，确保这个产品被排除
                return {
                    "product": {
                        **product,
                        "match_score": -100.0,
                        "reason": "❌ 女士专用产品，不适合男性使用"
                    },
                    "score": -100.0
                }
            elif any(word in product_text for word in ["男士", "男性", "男", "man", "male", "男性专用", "男士专用"]):
                gender_score = 2.0   # 男士专用产品给基础分
                reasons.append("✅ 男士专用产品，针对性更强")
            else:
                gender_score = 2.0   # 通用产品给基础分
                reasons.append("✅ 通用产品，适合所有性别")
        else:
            # 未检测到性别，给基础分但避免性别专用产品
            if any(word in product_text for word in ["男士", "男性", "男", "man", "male", "男性专用", "男士专用"]):
                gender_score = 1.0   # 男士产品给低分
                reasons.append("⚠️ 男士专用产品，性别未确定")
            elif any(word in product_text for word in ["女士", "女性", "女", "woman", "female", "女性专用", "女士专用"]):
                gender_score = 1.0   # 女士产品给低分
                reasons.append("⚠️ 女士专用产品，性别未确定")
            else:
                gender_score = 2.0   # 通用产品给高分
                reasons.append("✅ 通用产品，适合所有性别")
        
        total_score += gender_score
        
        # 2. 年龄匹配分数
        age_score = 0
        # 从皮肤分析中提取年龄信息
        is_elderly = any(word in skin_analysis for word in ["老年", "年长", "成熟", "50+", "60+", "70+"])
        is_middle_aged = any(word in skin_analysis for word in ["中年", "40+", "45+"])
        
        # 检查产品是否适合老年人
        if is_elderly:
            # 老年人专用产品加分
            if any(word in product_text for word in ["老年", "年长", "成熟", "抗老", "抗皱", "紧致"]):
                age_score += 3.0
                reasons.append("👴 专为老年人设计")
            # 检查产品功效是否适合老年人
            if any(word in product_text for word in ["抗皱", "紧致", "修护", "滋养", "温和"]):
                age_score += 2.0
                reasons.append("✨ 适合老年人的功效")
            # 检查产品成分是否适合老年人
            if any(word in product_text for word in ["神经酰胺", "玻尿酸", "胶原蛋白", "维生素E"]):
                age_score += 1.5
                reasons.append("💊 适合老年人的成分")
        elif is_middle_aged:
            # 中年人产品加分
            if any(word in product_text for word in ["抗初老", "紧致", "修护"]):
                age_score += 2.0
                reasons.append("👨 适合中年人的产品")
        
        total_score += age_score
        
        # 3. 皮肤问题匹配分数
        problem_score = 0
        if skin_conditions:
            # 计算皮肤问题匹配度
            for concern, score in skin_conditions.items():
                if concern in product.get("target_concerns", []):
                    problem_score += score * 1.5  # 问题匹配给高分
                    reasons.append(f"🎯 匹配{concern}问题")
                elif concern in str(product.get("benefits", "")) or concern in str(product.get("effects", "")):
                    problem_score += score * 1.0  # 功效匹配给中分
                    reasons.append(f"✨ 功效包含{concern}")
                elif concern in str(product.get("category", "")):
                    problem_score += score * 0.8  # 类别匹配给低分
                    reasons.append(f"📂 类别包含{concern}")
        
        total_score += problem_score
        
        # 4. 产品类型匹配加分
        category_bonus = 0
        if skin_conditions:
            # 根据皮肤问题类型给产品类型加分
            if any(concern in ["干燥", "缺水"] for concern in skin_conditions):
                if "保湿" in str(product.get("category", "")) or "补水" in str(product.get("benefits", "")):
                    category_bonus = 0.8
                    reasons.append("💧 保湿补水产品")
            if any(concern in ["皱纹", "老化"] for concern in skin_conditions):
                if "抗皱" in str(product.get("category", "")) or "抗老" in str(product.get("benefits", "")):
                    category_bonus = 0.8
                    reasons.append("🔄 抗皱抗老产品")
            if any(concern in ["敏感"] for concern in skin_conditions):
                if "敏感" in str(product.get("category", "")) or "温和" in str(product.get("benefits", "")):
                    category_bonus = 0.8
                    reasons.append("🛡️ 温和敏感肌产品")
        
        total_score += category_bonus
        
        # 5. 特殊需求匹配加分
        special_bonus = 0
        if (user_profile.get("special_needs") and 
            product.get("special_features")):
            special_bonus = 0.5
            reasons.append("⭐ 满足特殊需求")
        
        total_score += special_bonus
        
        # 6. 添加随机性因素，避免每次都推荐相同产品
        # 基于产品名称的哈希值生成微小的随机分数
        product_hash = hash(product.get("product_name", "")) % 1000
        random.seed(product_hash + int(time.time()) % 1000)
        micro_random = random.uniform(-0.1, 0.1)  # 0.1分的随机波动
        
        total_score += micro_random
        
        # 7. 应用随机因子
        total_score *= random_factor
        
        # 8. 确保分数不为负数（但允许大幅负分用于性别不匹配）
        if total_score < -50.0:
            total_score = -50.0  # 限制最低分数
        
        return {
            "product": {
                **product,
                "match_score": total_score,
                "reason": "; ".join(reasons)
            },
            "score": total_score
        }
    def _match_products(self,
                      skin_conditions: Dict[str, float],
                      user_profile: Dict[str, Any],
                      product_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于规则匹配产品"""
        try:
            # 1. 标准化用户年龄信息（已注释）
            # user_age = self._standardize_age_field(user_profile)
            
            # 2. 根据年龄段筛选产品（已注释）
            filtered_products = []
            for p in product_info:
                # 获取产品支持的年龄范围（已注释）
                # product_ages = self._get_product_ages(p)
                
                # 检查年龄匹配（已注释）
                # if self._is_age_match(user_age, product_ages):
                filtered_products.append(p)
            
            # 3. 根据皮肤问题评分匹配产品
            matched = []
            for product in filtered_products:
                match_info = self._calculate_match_score(
                    product, 
                    skin_conditions, 
                    user_profile
                )
                
                # 性别严格筛选：如果性别严重不匹配，直接跳过
                if match_info["score"] < -50.0:  # 性别严重不匹配（分数<-50.0）
                    logger.info(f"跳过性别严重不匹配的产品: {product.get('product_name', '未知')}")
                    continue
                
                # 额外检查：确保没有性别不匹配的产品通过
                product_name = str(product.get("product_name", "")).lower()
                product_details = str(product.get("details", "")).lower()
                product_text = f"{product_name} {product_details}"
                
                # 从皮肤分析中提取性别信息
                skin_analysis = ""
                if "skin_analysis" in user_profile:
                    skin_analysis = str(user_profile["skin_analysis"])
                elif "raw_profile" in user_profile:
                    skin_analysis = str(user_profile["raw_profile"])
                
                skin_analysis = skin_analysis.lower()
                
                # 二次性别检查
                is_female = any(word in skin_analysis for word in ["女性", "女士", "女", "woman", "female"])
                is_male = any(word in skin_analysis for word in ["男性", "男士", "男", "man", "male"])
                
                if is_female:
                    # 女性用户绝对不能推荐男士产品
                    if any(word in product_text for word in ["男士", "男性", "男", "man", "male", "男性专用", "男士专用"]):
                        logger.info(f"女性用户跳过男士产品: {product.get('product_name', '未知')}")
                        continue
                elif is_male:
                    # 男性用户绝对不能推荐女士产品
                    if any(word in product_text for word in ["女士", "女性", "女", "woman", "female", "女性专用", "女士专用"]):
                        logger.info(f"男性用户跳过女士产品: {product.get('product_name', '未知')}")
                        continue
                
                matched.append(match_info["product"])
            
            # 4. 按匹配度排序并返回前3个，增加产品多样性
            if matched:
                # 按匹配分数排序
                sorted_matched = sorted(matched, key=lambda x: x["match_score"], reverse=True)
                
                # 增加产品多样性：避免推荐相同品牌或相似产品
                diverse_result = []
                used_brands = set()
                used_categories = set()
                
                # 将产品按分数分组，增加随机性
                score_groups = {}
                for product in sorted_matched:
                    score = round(product["match_score"], 1)  # 保留一位小数分组
                    if score not in score_groups:
                        score_groups[score] = []
                    score_groups[score].append(product)
                
                # 从每个分数组中随机选择产品，增加多样性
                import random
                import time
                # 使用当前时间作为随机种子，确保每次运行都有不同的随机性
                random.seed(int(time.time() * 1000) % 10000)
                
                # 记录已选择的品牌和类别，用于后续的多样性控制
                selected_brands = set()
                selected_categories = set()
                
                for score in sorted(score_groups.keys(), reverse=True):
                    if len(diverse_result) >= 3:
                        break
                    
                    products_in_score = score_groups[score]
                    # 随机打乱同分数组内的产品顺序
                    random.shuffle(products_in_score)
                    
                    for product in products_in_score:
                        if len(diverse_result) >= 3:
                            break
                            
                        product_name = product.get("product_name", "").lower()
                        product_category = product.get("product_type", "").lower()
                        
                        # 检查品牌多样性（避免同一品牌过多）
                        brand_keywords = ["欧莱雅", "loreal", "巴黎欧莱雅"]
                        current_brand = None
                        for keyword in brand_keywords:
                            if keyword in product_name:
                                current_brand = keyword
                                break
                        
                        # 检查类别多样性（避免同类产品过多）
                        category_keywords = ["面霜", "精华", "乳液", "爽肤水", "面膜", "洁面"]
                        current_category = None
                        for keyword in category_keywords:
                            if keyword in product_category or keyword in product_name:
                                current_category = keyword
                                break
                        
                        # 多样性控制：如果品牌和类别都重复，跳过
                        if current_brand in selected_brands and current_category in selected_categories:
                            continue
                        
                        # 如果已经选择了2个同品牌产品，跳过
                        if current_brand and sum(1 for p in diverse_result if any(b in str(p.get("product_name", "")).lower() for b in brand_keywords)) >= 2:
                            continue
                        
                        # 如果已经选择了2个同类产品，跳过
                        if current_category and sum(1 for p in diverse_result if any(c in str(p.get("product_type", "")).lower() or c in str(p.get("product_name", "")).lower() for c in category_keywords)) >= 2:
                            continue
                        
                        diverse_result.append(product)
                        if current_brand:
                            selected_brands.add(current_brand)
                        if current_category:
                            selected_categories.add(current_category)
                
                # 如果多样性筛选后产品不足，补充剩余产品（随机选择）
                if len(diverse_result) < 3:
                    remaining_products = [p for p in sorted_matched if p not in diverse_result]
                    random.shuffle(remaining_products)
                    
                    for product in remaining_products:
                        if len(diverse_result) >= 3:
                            break
                        diverse_result.append(product)
                
                logger.info(f"产品匹配结果: 过滤前{len(filtered_products)}个，匹配后{len(matched)}个，多样性筛选后{len(diverse_result)}个")
                logger.info(f"推荐产品: {[p.get('product_name', '未知') for p in diverse_result]}")
                
                return diverse_result
            else:
                logger.warning("⚠️ 没有找到性别匹配的产品，尝试放宽性别限制")
                
                # 如果没有找到匹配的产品，尝试放宽性别限制，只排除明显的性别专用产品
                fallback_matched = []
                for product in filtered_products:
                    match_info = self._calculate_match_score(
                        product, 
                        skin_conditions, 
                        user_profile
                    )
                    
                    # 只排除明显的性别专用产品
                    product_name = str(product.get("product_name", "")).lower()
                    product_details = str(product.get("details", "")).lower()
                    product_text = f"{product_name} {product_details}"
                    
                    # 从皮肤分析中提取性别信息
                    skin_analysis = ""
                    if "skin_analysis" in user_profile:
                        skin_analysis = str(user_profile["skin_analysis"])
                    elif "raw_profile" in user_profile:
                        skin_analysis = str(user_profile["raw_profile"])
                    
                    skin_analysis = skin_analysis.lower()
                    
                    # 二次性别检查 - 只排除明显的性别专用产品
                    is_female = any(word in skin_analysis for word in ["女性", "女士", "女", "woman", "female"])
                    is_male = any(word in skin_analysis for word in ["男性", "男士", "男", "man", "male"])
                    
                    if is_female:
                        # 女性用户只排除明显的男士专用产品
                        if any(word in product_text for word in ["男士专用", "男性专用", "男士系列", "男性系列"]):
                            logger.info(f"女性用户跳过明显男士专用产品: {product.get('product_name', '未知')}")
                            continue
                    elif is_male:
                        # 男性用户只排除明显的女士专用产品
                        if any(word in product_text for word in ["女士专用", "女性专用", "女士系列", "女性系列"]):
                            logger.info(f"男性用户跳过明显女士专用产品: {product.get('product_name', '未知')}")
                            continue
                    
                    fallback_matched.append(match_info["product"])
                
                if fallback_matched:
                    result = sorted(fallback_matched, key=lambda x: x["match_score"], reverse=True)[:3]
                    logger.info(f"放宽性别限制后匹配结果: 匹配{len(fallback_matched)}个，返回{len(result)}个")
                    return result
                else:
                    logger.warning("⚠️ 即使放宽性别限制也没有找到匹配的产品")
                    return [{
                        "product_name": "暂无匹配产品",
                        "brand": "系统提示", 
                        "reason": "暂时没有找到完全匹配您需求的产品，建议咨询客服获取个性化推荐"
                    }]
            
        except Exception as e:
            logger.error(f"❌ 产品匹配失败: {str(e)}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return []