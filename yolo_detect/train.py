import os
from ultralytics import YOLO


def get_last_weight_path():
    """构建 last.pt 的预期路径"""
    return os.path.join('corn_leave_disease_results 20260621', 'run_v1', 'weights', 'last.pt')


def main():
    last_pt_path = get_last_weight_path()
    model = None
    is_resume = False

    # ================= 智能检测逻辑：判断是续训还是全新训练 =================
    if os.path.exists(last_pt_path):
        print(f"\n✅ 检测到中断的权重文件: {last_pt_path}")
        print("🚀 模式：【断点续训】将从上次停止的地方继续...")
        try:
            # 加载上次的完整状态（包含优化器、epoch计数等）
            model = YOLO(last_pt_path)
            is_resume = True
        except Exception as e:
            print(f"⚠️ 加载 last.pt 失败 ({e})，可能文件损坏。")
            print("🔄  fallback: 将尝试从头开始全新训练。")
            is_resume = False
            model = None

    # 如果不是续训模式，则加载官方预训练模型进行全新训练
    if not is_resume:
        if os.path.exists(last_pt_path):
            print("⚠️ 虽然发现了 last.pt，但因加载失败或配置变更，将忽略它。")
        print(f"\n🌱 未检测到有效的续训文件或强制重训。")
        print(f"🚀 模式：【全新训练】正在加载预训练模型 yolov8n.pt ...")
        model = YOLO('yolov8n.pt')

    # 安全检查：确保模型成功加载
    if model is None:
        raise RuntimeError("严重错误：模型未能成功加载！请检查 BASE_MODEL 路径或 last.pt 文件。")

    print("\n" + "=" * 50)
    print(f"📊 目标总轮数: 100")
    print(f"📂 结果保存至: corn_leave_disease_results 20260621/run_v1")
    print("=" * 50 + "\n")

    # ================= 开始训练 =================
    model.train(
        # 基础配置
        data='./Corn.yolov8/data.yaml',  # 数据集配置文件路径
        epochs=100,                       # 训练总轮数
        imgsz=640,                        # 输入图片尺寸
        batch=16,                         # 批次大小（显存不够可改为 8）
        workers=0,                        # 数据加载线程数（Windows 建议设为 0）
        project='corn_leave_disease_results 20260621',  # 项目保存目录
        name='run_v1',                    # 实验运行名称

        # 续训控制
        resume=is_resume,                 # 是否续训模式

        # 学习率配置
        lr0=0.01,                         # 初始学习率
        lrf=0.01,                         # 最终学习率系数（余弦退火终点 = lr0 * lrf）
        momentum=0.937,                   # SGD 动量
        weight_decay=0.0005,              # 权重衰减（正则化）

        # 训练策略
        patience=50,                      # 早停耐心值（连续 50 轮无改善则停止）
        close_mosaic=10,                  # 最后 10 轮关闭 mosaic 增强，提升定位精度

        # 数据增强配置
        hsv_h=0.015,                      # HSV-Hue 颜色增强幅度
        hsv_s=0.7,                        # HSV-Saturation 饱和度增强幅度
        hsv_v=0.4,                        # HSV-Value 亮度增强幅度
        degrees=0.0,                      # 随机旋转角度范围（度）
        translate=0.1,                    # 随机平移比例（±10%）
        scale=0.5,                        # 随机缩放比例（±50%）
        shear=0.0,                        # 随机剪切角度
        perspective=0.0,                  # 透视变换强度（必须 < 0.5）
        flipud=0.0,                       # 上下翻转概率
        fliplr=0.5,                       # 左右翻转概率（50%）
        mosaic=1.0,                       # Mosaic 四宫格拼接增强概率
        mixup=0.0,                        # MixUp 图像混合增强概率
        copy_paste=0.0,                   # Copy-Paste 复制粘贴增强概率

        # 验证与保存
        val=True,                         # 每个 epoch 后进行验证
        plots=True,                       # 生成训练曲线图（PR曲线、混淆矩阵等）
        save=True,                        # 保存模型权重
        save_period=10,                   # 每 10 轮保存一次 checkpoint
        verbose=True,                     # 显示详细训练日志
    )

    print("\n🎉 训练完成！请查看结果文件夹。")


if __name__ == '__main__':
    main()
