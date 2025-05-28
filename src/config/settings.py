import os
from dotenv import load_dotenv
from typing import Dict, Any

# 加载环境变量
load_dotenv()

# API配置
API_CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "model_name": "Qwen/Qwen-32B-Chat"
}

# 模型配置
MODEL_CONFIG = {
    "vlm_model": "Qwen/Qwen2.5-VL",
    "embedding_model": "BAAI/bge-m3"
}

# 知识库配置
KNOWLEDGE_BASE_CONFIG = {
    "path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge"),
    "update_frequency": "daily",
    "sources": [
        "小红书/B站博主图文内容",
        "商品详情页摘要",
        "皮肤医学文献"
    ]
}

# 产品数据库配置
PRODUCT_DB_CONFIG = {
    "sources": [
        "欧莱雅官网",
        "淘宝API",
        "京东API"
    ],
    "update_frequency": "daily"
}

# 前端配置
FRONTEND_CONFIG = {
    "theme": {
        "tone": "柔和、女性友好",
        "responsive": True,
        "dark_mode": True
    },
    "components": {
        "upload_image": True,
        "skin_condition": True,
        "questionnaire": True,
        "recommendation": True,
        "trust_reasoning": True,
        "user_profile": True
    }
}

# 系统配置
SYSTEM_CONFIG = {
    "max_retries": 3,
    "timeout": 30,
    "cache_enabled": True,
    "log_level": "INFO"
}

def get_config() -> Dict[str, Any]:
    """获取完整配置"""
    return {
        "api": API_CONFIG,
        "model": MODEL_CONFIG,
        "knowledge_base": KNOWLEDGE_BASE_CONFIG,
        "product_db": PRODUCT_DB_CONFIG,
        "frontend": FRONTEND_CONFIG,
        "system": SYSTEM_CONFIG
    } 