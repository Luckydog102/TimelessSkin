from typing import Dict, List, Any
from ..models.embedding_model import EmbeddingModel
from .knowledge_base import KnowledgeManager

class RAGEngine:
    """检索增强生成引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = EmbeddingModel()
        self.knowledge_manager = KnowledgeManager(config, self.embedding_model)
        self._initialized = False
        self.initialize()
        
    def initialize(self) -> None:
        """初始化RAG引擎"""
        if not self._initialized:
            self.embedding_model.initialize()
            self.knowledge_manager.initialize()
            self._initialized = True
        
    def search(self,
              query: str,
              category: str = None,
              metadata_filters: Dict[str, Any] = None,
              top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关知识"""
        # 确保已初始化
        if not self._initialized:
            self.initialize()
            
        return self.knowledge_manager.search(
            query=query,
            category=category,
            metadata_filters=metadata_filters,
            top_k=top_k,
            use_cache=True  # 默认启用缓存
        )
        
    # 添加retrieve方法作为search的别名
    def retrieve(self,
                query: str,
                category: str = None,
                metadata_filters: Dict[str, Any] = None,
                top_k: int = 5) -> List[Dict[str, Any]]:
        """retrieve方法是search方法的别名"""
        return self.search(
            query=query,
            category=category,
            metadata_filters=metadata_filters,
            top_k=top_k
        )
        
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.knowledge_manager.get_knowledge_stats()
        
    def update_knowledge(self, new_documents: List[Dict[str, Any]]) -> None:
        """更新知识库"""
        self.knowledge_manager.update_knowledge(new_documents) 

if __name__ == "__main__":
    config = {
        "knowledge_base": {
            "path": "/root/lssyes/andyshaw/Agent/FuranAgent/src/knowledge"  # 替换为你的真实路径
        }
    }

    # 初始化 RAG 引擎
    rag = RAGEngine(config)
    rag.initialize()

    # 测试搜索
    query = "脸上有色素沉着和痘印"
    category = "skin_conditions"
    results = rag.search(query, category=category, top_k=3)
    print(results)
