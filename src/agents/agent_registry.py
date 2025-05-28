from typing import Dict, Type, List, Any
from .skincare_agent import SkinCareAgent

class AgentRegistry:
    """Agent注册表"""
    
    _agents: Dict[str, Type] = {
        "skincare": SkinCareAgent
    }
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type) -> None:
        """注册新的Agent"""
        cls._agents[name] = agent_class
        
    @classmethod
    def get_agent(cls, name: str) -> Type:
        """获取Agent类"""
        if name not in cls._agents:
            raise ValueError(f"Agent {name} not found")
        return cls._agents[name]
        
    @classmethod
    def list_agents(cls) -> List[str]:
        """列出所有已注册的Agent"""
        return list(cls._agents.keys())
        
    @classmethod
    def create_agent(cls, name: str, **kwargs) -> Any:
        """创建Agent实例"""
        agent_class = cls.get_agent(name)
        return agent_class(**kwargs) 