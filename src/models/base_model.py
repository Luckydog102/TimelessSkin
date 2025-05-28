from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseModel(ABC):
    """所有模型的基类"""
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化模型"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """模型预测接口"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass 