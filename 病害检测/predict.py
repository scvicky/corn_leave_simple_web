from ultralytics import YOLO
import os

# ================= 配置区域 =================
# 🔒 锁定 v1 最佳模型
model_path = r'C:\Users\HP\Desktop\杂\项目\111\病害检测\runs\detect\3_plant_disease_results\exp_v1_robflow_struct\weights\best.pt'

# 测试图片/视频路径 (请修改为你自己的测试图)
# 可以是单张图片路径，也可以是文件夹路径，甚至是 '0' (摄像头)
source_path = r"C:\Users\HP\Downloads\Corn Disease.v1i.yolov8(1)\test\images\Corn_Common_Rust-1291-_jpg.rf.f976438fe6f4fd23873451c2fa8cdd08.jpg"  # 示例：测试文件夹
# ===========================================

if not os.path.exists(model_path):
    print(f"❌ 错误：找不到模型文件 {model_path}")
else:
    print(f"✅ 加载最终模型 (v1): {model_path}")
    model = YOLO(model_path)

    print("🚀 开始预测...")

    # 执行预测
    results = model.predict(
        source=source_path,
        conf=0.25,  # 置信度阈值 (默认)
        iou=0.45,  # NMS 阈值 (默认)
        imgsz=640,  # 输入尺寸
        save=True,  # 保存结果图
        project=r'C:\Users\HP\Desktop\杂\项目\111\病害检测\runs\detect\predict_results',  # 结果保存目录
        name='v1_final_inference',
        verbose=True,  # 打印详细信息
        show=False  # 是否弹窗显示 (服务器运行时建议 False)
    )

    print("\n🎉 预测完成！结果已保存至 runs/detect/predict_results/v1_final_inference")

    # 简单统计
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"📷 图片 {r.path}: 检测到 {len(boxes)} 个目标")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"   - {name}: {conf:.2f}")
        else:
            print(f"📷 图片 {r.path}: 未检测到目标")