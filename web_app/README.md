# 玉米病害检测 Web 应用

## 项目结构
```
web_app/
├── app.py              # Flask web服务
├── templates/
│   └── index.html      # 上传页面
├── requirements.txt    # Python依赖
└── .gitignore
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
- 模型文件位于 `../病害检测/runs/detect/corn_leave_disease_results/run_v1/weights/best.pt`
- 确保病害检测模块中的 predict.py 正常工作
