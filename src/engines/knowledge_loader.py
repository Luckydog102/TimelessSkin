import os
import json
from typing import Dict, List, Any
from pathlib import Path

class KnowledgeLoader:
    """知识库加载器"""
    
    def __init__(self, base_path: str = "/root/lssyes/andyshaw/Agent/FuranAgent/agent/Question_Builder/knowledge"):
        self.base_path = Path(base_path)
        
    def load_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载所有知识库内容"""
        knowledge = {
            "skin_conditions": self._load_skin_conditions(),
            "product_info": self._load_product_info(),
            "skincare_rules": self._load_skincare_rules(),
            "user_profiles": self._load_user_profiles()
        }
        return knowledge
        
    def _load_skin_conditions(self) -> List[Dict[str, Any]]:
        """加载皮肤状况知识"""
        conditions = []
        conditions_path = self.base_path / "skin_conditions"
        if conditions_path.exists():
            for file in conditions_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    conditions.extend(json.load(f))
        return conditions
        
    def _load_product_info(self) -> List[Dict[str, Any]]:
        """加载产品信息"""
        products = []
        products_path = self.base_path / "products"
        if products_path.exists():
            for file in products_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    products.extend(json.load(f))
        return products
        
    def _load_skincare_rules(self) -> List[Dict[str, Any]]:
        """加载护肤规则"""
        rules = []
        rules_path = self.base_path / "skincare_rules"
        if rules_path.exists():
            for file in rules_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    rules.extend(json.load(f))
        return rules
        
    def _load_user_profiles(self) -> List[Dict[str, Any]]:
        """加载用户画像模板"""
        profiles = []
        profiles_path = self.base_path / "user_profiles"
        if profiles_path.exists():
            for file in profiles_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    profiles.extend(json.load(f))
        return profiles 