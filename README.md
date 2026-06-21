# 🌽 玉米病害检测系统

基于 YOLOv8 的玉米叶片病害检测 Web 应用，能够自动识别玉米叶片中的常见病害类型。

## ✨ 功能特性

-  **高精度检测**：使用 YOLOv8 深度学习模型，准确识别多种玉米病害
-  **快速推理**：单张图片检测仅需毫秒级响应时间
-  **Web 界面**：简洁友好的上传界面，实时显示检测结果
-  **置信度显示**：每个检测结果都附带置信度评分
- 🔄 **易于扩展**：模块化设计，方便添加新的病害类别

## 📁 项目结构

```
corn_leave_simple_web/
├── yolo_detect/                  # 训练和推理模块
│   ├── train.py                  # YOLOv8 训练脚本
│   ├── predict.py                # 推理核心（备用）
│   ├── yolov8n.pt               # YOLOv8 nano 预训练模型
│   ├── Corn.yolov8/             # 数据集配置文件
│   │   └── data.yaml            # 数据集配置
│   └── runs/                    # 训练结果目录
│       └── detect/
│           └── corn_leave_disease_results/
│               └── run_v1/
│                   └── weights/
│                       └── best.pt    # 训练好的最佳模型
│
├── web_app/                      # Web 应用模块
│   ├── app.py                    # Flask 后端服务
│   ├── predict.py                # YOLO 推理模块
│   ├── templates/
│   │   └── index.html           # 前端上传页面
│   ├── requirements.txt         # Python 依赖包
│   └── uploads/                 # 上传文件临时存储
│
├── README.md                     # 项目说明文档
└── .gitignore                    # Git 忽略配置
```

## ️ 技术栈

### 后端
- **Python 3.8+**
- **Flask 3.0.0** - Web 框架
- **Ultralytics YOLOv8** - 目标检测模型
- **PyTorch** - 深度学习框架

### 前端
- **HTML5 + CSS3** - 页面结构和样式
- **JavaScript (ES6+)** - 异步请求处理
- **Fetch API** - HTTP 通信

### 核心依赖
- `ultralytics==8.1.0` - YOLOv8 库
- `flask==3.0.0` - Web 服务
- `torch>=2.0.0` - PyTorch
- `opencv-python>=4.8.0` - 图像处理
- `Pillow>=9.5.0` - 图片处理

##  快速开始

### 1. 环境要求

- Python 3.8 或更高版本
- pip 包管理器
- 推荐显存 ≥ 4GB（用于 GPU 加速）

### 2. 安装依赖

```bash
# 进入项目根目录
cd corn_leave_simple_web

# 安装所有依赖
pip install -r web_app/requirements.txt

# 或使用国内镜像加速
pip install -r web_app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 启动服务

```bash
# 进入 web_app 目录
cd web_app

# 启动 Flask 服务
python app.py
```

服务启动后，终端会显示类似信息：
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 4. 访问应用

打开浏览器访问：**http://localhost:5000**

## 📖 使用说明

### Web 界面操作

1. **上传图片**
   - 点击"选择文件"按钮
   - 选择玉米叶片图片（支持 JPG、PNG 格式）
   - 建议图片分辨率 ≥ 640x640

2. **开始检测**
   - 点击"开始检测"按钮
   - 等待系统处理（通常 < 1 秒）

3. **查看结果**
   - 检测结果会立即显示在页面下方
   - 包含病害类型和置信度评分

### API 接口

#### 上传并检测

**端点**：`POST /predict`

**请求参数**：
- `file`: 图片文件（multipart/form-data）

**响应示例**：
```json
{
  "success": true,
  "filename": "test.jpg",
  "detections": [
    {
      "class": "Corn_Common_Rust",
      "confidence": 0.92
    },
    {
      "class": "Corn_Gray_Leaf_Spot",
      "confidence": 0.85
    }
  ]
}
```

**错误响应**：
```json
{
  "success": false,
  "error": "没有上传文件"
}
```

## 🧪 模型训练（可选）

如果你想重新训练模型或调整参数：

### 1. 准备数据集

确保 `yolo_detect/Corn.yolov8/data.yaml` 配置正确：

```yaml
path: ../datasets/corn_disease
train: images/train
val: images/val

nc: 4  # 类别数量
names: ['Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Healthy', 'Corn_Northern_Leaf_Blight']
```

### 2. 运行训练

```bash
cd yolo_detect
python train.py
```

训练完成后，最佳模型保存在：
```
yolo_detect/runs/detect/corn_leave_disease_results/run_v1/weights/best.pt
```

### 3. 自定义训练参数

编辑 `train.py` 中的 `model.train()` 参数：

```python
model.train(
    data='./Corn.yolov8/data.yaml',
    epochs=100,        # 训练轮数
    imgsz=640,         # 图片尺寸
    batch=16,          # 批次大小
    lr0=0.01,          # 初始学习率
    patience=50,       # 早停耐心值
    # ... 其他参数
)
```

## 🔍 支持的病害类型

根据训练数据，当前模型可识别以下类别：

- **Corn_Common_Rust** - 玉米普通锈病
- **Corn_Gray_Leaf_Spot** - 玉米灰斑病
- **Corn_Healthy** - 健康叶片
- **Corn_Northern_Leaf_Blight** - 玉米北方叶枯病

*具体类别取决于你的训练数据集*

## ️ 配置说明

### 修改模型路径

如果模型位置发生变化，编辑 `web_app/predict.py`：

```python
MODEL_PATH = './yolo_detect/runs/detect/corn_leave_disease_results/run_v1/weights/best.pt'
```

### 调整检测阈值

在 `web_app/predict.py` 中修改默认参数：

```python
def predict(image_path, conf=0.25, iou=0.45):
    """
    conf: 置信度阈值（越低越敏感，越高越严格）
    iou: NMS 阈值（非极大值抑制）
    """
```

### 修改端口

在 `web_app/app.py` 中修改：

```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # 改为其他端口号
```

##  常见问题

### Q1: 启动时提示找不到模型文件？

**A**: 检查模型路径是否正确：
```bash
# 验证模型是否存在
ls yolo_detect/runs/detect/corn_leave_disease_results/run_v1/weights/best.pt
```

### Q2: 检测速度很慢？

**A**: 
- 使用 GPU 加速（需要 CUDA 环境）
- 降低图片分辨率
- 减小 `imgsz` 参数（如改为 416）

### Q3: 检测结果不准确？

**A**:
- 提高置信度阈值（conf 从 0.25 改为 0.4）
- 重新训练模型，增加更多训练数据
- 检查图片质量是否清晰

### Q4: 上传图片失败？

**A**:
- 检查文件格式（仅支持图片）
- 检查文件大小（建议 < 10MB）
- 查看终端错误日志

### Q5: 如何部署到服务器？

**A**: 使用 Gunicorn 替代 Flask 开发服务器：

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📝 更新日志

### v1.0.0 (2026-06-21)
- ✅ 初始版本发布
- ✅ 基于 YOLOv8 的病害检测
- ✅ Flask Web 服务
- ✅ 简洁的上传界面
- ✅ 完整的训练和推理流程

##  贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题或建议，欢迎联系！

---

**注意**：本项目仅用于学习和研究目的。在实际农业生产中使用时，请结合专业农业知识进行综合判断。
