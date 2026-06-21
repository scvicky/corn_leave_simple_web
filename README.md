# 玉米病害检测 Web 应用

## 项目结构
```
web_app/
├── app.py              # Flask web服务
├── predict.py          # YOLO推理模块
├── templates/
│   └── index.html      # 上传页面
├── requirements.txt    # Python依赖
└── .gitignore

yolo_detect/
├── train.py            # 训练代码
├── predict.py          # 推理核心（备用）
├── yolov8n.pt         # 预训练模型
├── Corn.yolov8/       # 数据集配置
└── runs/              # 训练结果（包含best.pt）
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务
```bash
python app.py
```

### 3. 访问应用
打开浏览器访问: http://localhost:5000

## 功能说明
- 上传玉米叶片图片
- 自动检测病害类型
- 显示检测结果和置信度

## 注意事项
- 模型文件位于 `./yolo_detect/runs/detect/corn_leave_disease_results/run_v1/weights/best.pt`
- yolo_detect目录包含训练和推理相关代码
