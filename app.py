import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入主应用
from frontend.gradio_app.ui_main import create_ui

if __name__ == "__main__":
    # 创建并启动UI
    demo = create_ui()
    demo.queue()  # 启用队列
    demo.launch(share=False) 