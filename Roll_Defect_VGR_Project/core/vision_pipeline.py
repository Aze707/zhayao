# core/vision_pipeline.py
import yaml
import os
from models.yolo_obb_infer import YoloOBBEngine
from core.grasp_solver import GraspSolver 

class VisionPipeline:
    def __init__(self, config_path="config.yaml"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        absolute_config_path = os.path.join(project_root, config_path)
        
        with open(absolute_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.engine = YoloOBBEngine(absolute_config_path)
        self.solver = GraspSolver(safe_margin=config['grasping']['safe_margin_px'])

    def initialize(self):
        self.engine.load_model()

    def process_frame(self, frame):
        """核心解算管道：输入图像数组 -> 输出轻量数据层"""
        payload_data = []
        if frame is None:
            return frame, payload_data
            
        # 1. 扔进 YOLO 引擎推理真实照片
        targets = self.engine.infer(frame)
        
        # 2. 纯粹解算，拼装轻量字典结构
        for target in targets:
            grasp_pose = self.solver.calculate_optimal_point(target)
            
            payload_data.append({
                "target_status": target.status,
                "grasp_data": grasp_pose.to_dict(),
                "target": target  # 保留原始 OBB 对象提供给前端绘制边框
            })
            
        # 【重要改动】直接返回最干净的原图引用与核心通讯字典
        return frame, payload_data