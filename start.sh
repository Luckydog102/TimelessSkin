#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
