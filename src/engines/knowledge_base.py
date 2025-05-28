from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
from ..models.embedding_model import EmbeddingModel

import numpy as np
import faiss
from pathlib import Path
from collections import defaultdict
import logging

# 配置日志
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """统一的知识库管理模块，使用 embedding_model + FAISS 完成向量检索"""

    def __init__(self, config: Dict[str, Any], embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.vector_index = None
        self.embeddings_matrix = None
        self.documents = []
        self.metadata_index = defaultdict(list)
        self.cache = {}
        self._initialized = False
        self.initialize()

    def initialize(self) -> None:
        """初始化知识库"""
        try:
            if not self._initialized:
                self._load_knowledge()
                self._build_index()
                self._initialized = True
                logger.info("知识库初始化成功")
        except Exception as e:
            logger.error(f"知识库初始化失败: {e}")
            # 设置为初始化状态，但允许系统继续运行
            self._initialized = True

    def _load_knowledge(self) -> None:
        """加载知识库文件"""
        try:
            # 使用相对路径
            base_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge"))
            logger.info(f"加载知识库路径: {base_path}")
            
            # 如果已经加载过，直接返回
            if self.documents:
                return
                
            for subfolder, category in [("skin_conditions", "skin_conditions"),
                                         ("products", "products"),
                                         ("skincare_rules", "skincare_rules")]:
                path = base_path / subfolder
                if path.exists():
                    for file in path.glob("*.json"):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                self._process_document(data, category)
                        except Exception as e:
                            logger.error(f"加载文件失败 {file}: {e}")
                else:
                    logger.warning(f"知识库子目录不存在: {path}")
                    
            logger.info(f"成功加载 {len(self.documents)} 个文档")
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            # 确保即使加载失败，也能继续运行
            pass

    def _process_document(self, data: Dict[str, Any], category: str) -> None:
        content = self._extract_content(data)
        metadata = self._extract_metadata(data, category)

        doc = {
            "content": content,
            "metadata": metadata,
            "category": category,
            "source": data.get("source", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
        self.documents.append(doc)

        for key, value in metadata.items():
            values = value if isinstance(value, list) else [value]
            for v in values:
                self.metadata_index[f"{key}:{v}"].append(len(self.documents) - 1)

    def _extract_content(self, data: Dict[str, Any]) -> str:
        def flatten(d: Any) -> List[str]:
            if isinstance(d, dict):
                parts = []
                for v in d.values():
                    parts.extend(flatten(v))
                return parts
            elif isinstance(d, list):
                parts = []
                for item in d:
                    parts.extend(flatten(item))
                return parts
            elif isinstance(d, (str, int, float)):
                return [str(d)]
            return []

        text_parts = flatten(data)
        return " ".join(text_parts)

    def _extract_metadata(self, data: Dict[str, Any], category: str) -> Dict[str, Any]:
        metadata = {"category": category}
        if category == "skin_conditions":
            metadata.update({
                "condition_type": data.get("condition"),
                "severity": list(data.get("severity_levels", {}).keys()),
                "related_conditions": data.get("related_conditions", [])
            })
        elif category == "products":
            metadata.update({
                "product_type": data.get("category"),
                "suitable_for": data.get("suitable_for", []),
                "ingredients": data.get("ingredients", [])
            })
        return metadata

    def _build_index(self) -> None:
        """构建向量索引，添加错误处理和空检查"""
        try:
            if not self.documents:
                logger.warning("没有文档可索引，跳过索引构建")
                # 创建一个空的索引，避免后续搜索错误
                self.vector_index = faiss.IndexFlatL2(768)  # 使用BGE模型的默认维度
                self.embeddings_matrix = np.zeros((0, 768), dtype='float32')
                return

            texts = [doc["content"] for doc in self.documents]
            logger.info(f"开始为 {len(texts)} 个文档生成嵌入向量")
            
            try:
                # 生成嵌入向量
                embeddings = self.embedding_model.predict(texts)
                
                # 检查是否成功获取嵌入向量
                if embeddings is None or len(embeddings) == 0:
                    logger.error("获取嵌入向量失败，使用空矩阵")
                    self.embeddings_matrix = np.zeros((len(texts), 768), dtype='float32')
                    self.vector_index = faiss.IndexFlatL2(768)
                    return
                    
                # 转换为numpy数组，确保是2D数组
                if isinstance(embeddings, list):
                    if not embeddings or not isinstance(embeddings[0], (list, np.ndarray)):
                        logger.error("嵌入向量格式错误，使用空矩阵")
                        self.embeddings_matrix = np.zeros((len(texts), 768), dtype='float32')
                        self.vector_index = faiss.IndexFlatL2(768)
                        return
                        
                    self.embeddings_matrix = np.array(embeddings).astype('float32')
                else:
                    logger.error("嵌入向量不是列表格式，使用空矩阵")
                    self.embeddings_matrix = np.zeros((len(texts), 768), dtype='float32')
                    self.vector_index = faiss.IndexFlatL2(768)
                    return
                
                # 检查嵌入矩阵的形状
                if len(self.embeddings_matrix.shape) != 2:
                    logger.error(f"嵌入矩阵维度错误: {self.embeddings_matrix.shape}")
                    self.embeddings_matrix = np.zeros((len(texts), 768), dtype='float32')
                    self.vector_index = faiss.IndexFlatL2(768)
                    return
                    
                dimension = self.embeddings_matrix.shape[1]
                logger.info(f"嵌入向量维度: {dimension}")
                
                self.vector_index = faiss.IndexFlatL2(dimension)
                self.vector_index.add(self.embeddings_matrix)
                logger.info("索引构建成功")
                
            except Exception as e:
                logger.error(f"生成嵌入向量失败: {e}")
                self.embeddings_matrix = np.zeros((len(texts), 768), dtype='float32')
                self.vector_index = faiss.IndexFlatL2(768)
            
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            # 创建一个空的索引，避免后续搜索错误
            self.embeddings_matrix = np.zeros((0, 768), dtype='float32')
            self.vector_index = faiss.IndexFlatL2(768)

    def search(self,
               query: str,
               category: Optional[str] = None,
               metadata_filters: Optional[Dict[str, Any]] = None,
               top_k: int = 5,
               use_cache: bool = True) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            # 确保已初始化
            if not self._initialized:
                self.initialize()

            # 使用更细粒度的缓存key
            cache_key = f"{query}:{category}:{str(metadata_filters)}:{top_k}"
            if use_cache and cache_key in self.cache:
                return self.cache[cache_key]

            # 检查是否有文档和索引
            if not self.documents or self.vector_index is None:
                logger.warning("知识库为空或索引未构建，返回空结果")
                return []

            # 应用过滤器
            filtered_indices = set(range(len(self.documents)))
            if category:
                filtered_indices &= set(self.metadata_index.get(f"category:{category}", []))
                
            if metadata_filters:
                for key, value in metadata_filters.items():
                    values = value if isinstance(value, list) else [value]
                    for v in values:
                        filtered_indices &= set(self.metadata_index.get(f"{key}:{v}", []))

            # 向量检索
            query_vec = self.embedding_model.predict(query)
            if not query_vec:
                logger.error("查询向量生成失败")
                return []
                
            if isinstance(query_vec[0], list):
                query_vec_np = np.array(query_vec).astype('float32')
            else:
                query_vec_np = np.array([query_vec]).astype('float32')

            distances, indices = self.vector_index.search(query_vec_np, min(top_k * 2, len(self.documents)))
            results = []
            seen = set()

            for idx, dist in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.documents) and idx in filtered_indices and idx not in seen:
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(1 / (1 + dist))
                    results.append(doc)
                    seen.add(idx)

            results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

            # 更新缓存
            if use_cache:
                self.cache[cache_key] = results

            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def update_knowledge(self, new_documents: List[Dict[str, Any]]) -> None:
        for doc in new_documents:
            self._process_document(doc, doc.get("category", "unknown"))
        self._build_index()
        self._save_knowledge()

    def _save_knowledge(self) -> None:
        try:
            # 使用相对路径
            kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "metadata", "index.json")
            os.makedirs(os.path.dirname(kb_path), exist_ok=True)
            with open(kb_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "last_updated": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"知识库保存成功: {kb_path}")
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")

    def get_knowledge_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "categories": {
                cat: len([d for d in self.documents if d["category"] == cat])
                for cat in set(d["category"] for d in self.documents)
            },
            "metadata_types": list(set(k.split(":")[0] for k in self.metadata_index)),
            "last_updated": self.documents[-1]["timestamp"] if self.documents else None
        }
    
if __name__=="__main__":
    config = {
        "knowledge_base": {
            "path": "src/knowledge"
        }
    }

    # 初始化 embedding + RAG
    embedding_model = EmbeddingModel()
    embedding_model.initialize()

    rag = KnowledgeManager(config, embedding_model)

    # 测试查询
    query = "脸上长痘，有红肿和痘印，想找祛痘产品"
    results = rag.search(query, top_k=3)

    print("🔍 搜索结果：")
    for i, doc in enumerate(results, 1):
        print(f"\n📄 结果 {i}")
        print(f"来源: {doc['source']}")
        print(f"相似度分数: {doc['similarity_score']:.4f}")
        print(f"内容预览: {doc['content'][:80]}...")