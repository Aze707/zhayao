import os
from pathlib import Path

def create_project_structure():
    # 定义项目根目录名称
    project_root = Path("Roll_Defect_VGR_Project")

    # 定义需要创建的目录树
    directories = [
        "core",
        "drivers",
        "models",
        "models/weights",
        "utils"
    ]

    # 定义需要创建的文件及其初始化的基础注释或内容
    files = {
        "main.py": '"""\n主程序入口\n负责线程调度与全局状态机\n"""\n',
        "config.yaml": "# 全局配置文件\n# 包含相机参数、模型路径、标定矩阵、TCP端口等\n",
        "core/__init__.py": "",
        "core/vision_pipeline.py": '"""\n视觉流控模块\n统筹图像采集、YOLO推理、后处理解算\n"""\n',
        "core/grasp_solver.py": '"""\n最优抓取点解算器\n核心逻辑：基于向量偏移防碰撞 / 基于形态学骨架处理弯曲\n"""\n',
        "core/hand_eye_calib.py": '"""\n手眼标定与坐标系转换模块\n实现 像素坐标(u,v) -> 物理基坐标(X,Y)\n"""\n',
        "drivers/__init__.py": "",
        "drivers/camera_driver.py": '"""\n工业相机 SDK 封装\n适配如海康、大华等常见工业相机，支持飞拍模式\n"""\n',
        "drivers/plc_comm.py": '"""\nPLC 通信模块\n用于读取传送带编码器速度，实现动态跟随补偿\n"""\n',
        "drivers/robot_comm.py": '"""\n机械臂通信模块\n基于 TCP/IP Socket 下发抓取坐标、角度及夹爪行程指令\n"""\n',
        "models/__init__.py": "",
        "models/yolo_obb_infer.py": '"""\nYOLOv8-OBB 推理引擎封装\n支持 PyTorch 原生推理以及 TensorRT/ONNX 部署加速\n"""\n',
        "utils/__init__.py": "",
        "utils/logger.py": '"""\n系统日志记录模块\n用于记录运行错误、单次节拍耗时以及历史抓取点位\n"""\n',
        "utils/visualizer.py": '"""\n可视化与 Debug 模块\n在图像上绘制 OBB 框、局部缺陷框与最终的十字抓取点\n"""\n'
    }

    print(f"🚀 开始构建项目目录树: {project_root} ...\n")

    # 1. 创建项目根目录
    try:
        project_root.mkdir(parents=True, exist_ok=False)
        print(f"📁 创建根目录: {project_root}/")
    except FileExistsError:
        print(f"⚠️ 根目录 {project_root} 已存在，将在原目录下补充缺失文件。\n")

    # 2. 创建所有子目录
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  └── 📁 创建子目录: {dir_name}/")

    # 3. 创建所有文件并写入初始注释
    print("\n📝 开始生成核心文件...")
    for file_name, content in files.items():
        file_path = project_root / file_name
        # 仅当文件不存在时才创建，避免覆盖您已有的代码
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  └── 📄 创建文件: {file_name}")
        else:
            print(f"  └── ⏭️ 文件已存在，跳过: {file_name}")

    print("\n✅ 项目骨架构建完成！您可以打开 IDE 开始编写核心逻辑了。")

if __name__ == "__main__":
    create_project_structure()