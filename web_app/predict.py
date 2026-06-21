from ultralytics import YOLO
import os

# 模型路径配置
MODEL_PATH = r'C:\Users\HP\Documents\GitHub\corn_leave_simple_web\病害检测\runs\detect\corn_leave_disease_results\run_v1\weights\best.pt'

# 加载模型（全局加载一次，避免重复加载）
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


def predict(image_path, conf=0.25, iou=0.45):
    """
    对单张图片进行病害检测
    
    参数:
        image_path: 图片路径
        conf: 置信度阈值 (默认0.25)
        iou: NMS阈值 (默认0.45)
    
    返回:
        list: 检测结果列表，每个元素包含 {'class': 类别名, 'confidence': 置信度}
    """
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False
    )
    
    # 解析检测结果
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detections.append({
                'class': name,
                'confidence': round(conf, 2)
            })
    
    return detections


# 测试代码（仅在直接运行此文件时执行）
if __name__ == '__main__':
    # 示例：测试单张图片
    test_image = input("请输入测试图片路径（直接回车使用默认图片）: ").strip()
    
    if not test_image:
        print("❌ 请提供测试图片路径")
        exit(1)
    
    if not os.path.exists(test_image):
        print(f"❌ 图片不存在: {test_image}")
        exit(1)
    
    print(f"🔍 正在检测: {test_image}")
    detections = predict(test_image)
    
    if detections:
        print(f"\n✅ 检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections, 1):
            print(f"   {i}. {det['class']} (置信度: {det['confidence']})")
    else:
        print("\n⚠️ 未检测到病害")
