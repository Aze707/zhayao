# core/vision_pipeline.py
import yaml
import os
from models.yolo_obb_infer import YoloOBBEngine
from core.grasp_solver import GraspSolver

class VisionPipeline:
    def __init__(self, config_path="config.yaml"):
        # 1. 动态获取绝对路径，彻底告别 FileNotFoundError
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        absolute_config_path = os.path.join(project_root, config_path)
        
        # 2. 读取配置文件
        with open(absolute_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 3. 实例化核心组件 (此处修复了缺失的 engine 和 solver)
        # 注意：将绝对路径直接传递给 Engine，这样底层模型也能正确找到配置！
        self.engine = YoloOBBEngine(absolute_config_path)
        self.solver = GraspSolver(safe_margin=config['grasping']['safe_margin_px'])

    def initialize(self):
        self.engine.load_model()

    def process_frame(self, frame):
        """流水线核心：输入图像 -> 推理 -> 解算 -> 返回结果列表"""
        # 1. 目标检测
        targets = self.engine.infer(frame)
        results = []
        
        # 2. 抓取点解算
        for target in targets:
            grasp_pose = self.solver.calculate_optimal_point(target)
            results.append({
                "target": target,
                "grasp_pose": grasp_pose
            })
            
        return results