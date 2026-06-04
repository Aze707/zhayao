# core/vision_pipeline.py
import yaml
import os
from models.yolo_obb_infer import YoloOBBEngine
from core.grasp_solver import GraspSolver
from core.defect_detector import DefectDetector 

# class VisionPipeline:
#     def __init__(self, config_path="config.yaml"):
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(current_dir)
#         absolute_config_path = os.path.join(project_root, config_path)
        
#         with open(absolute_config_path, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
            
#         self.engine = YoloOBBEngine(absolute_config_path)
#         self.solver = GraspSolver(safe_margin=config['grasping']['safe_margin_px'])

#     def initialize(self):
#         self.engine.load_model()

#     def process_frame(self, frame):
#         """核心解算管道：输入图像数组 -> 输出轻量数据层"""
#         payload_data = []
#         if frame is None:
#             return frame, payload_data
            
#         # 1. 扔进 YOLO 引擎推理真实照片
#         targets = self.engine.infer(frame)
        
#         # 2. 纯粹解算，拼装轻量字典结构
#         for target in targets:
#             grasp_pose = self.solver.calculate_optimal_point(target)
            
#             payload_data.append({
#                 "target_status": target.status,
#                 "grasp_data": grasp_pose.to_dict(),
#                 "target": target  # 保留原始 OBB 对象提供给前端绘制边框
#             })
            
#         # 【重要改动】直接返回最干净的原图引用与核心通讯字典
#         return frame, payload_data

# 1. 导入您的新传统视觉引擎


class TargetAdapter:
    """
    数据适配器：将 OpenCV 算出的普通字典，伪装成 Visualizer 认识的对象
    """
    def __init__(self, c_data):
        self.xc = c_data["center_x"]
        self.yc = c_data["center_y"]
        
        # 【核心修复】：w 必须对应长边 length，h 对应短边 width
        # 因为底层的 theta 角度是贴合长边计算出来的
        self.w = c_data["length"]  
        self.h = c_data["width"]   
        
        self.theta = c_data["angle"]
        self.status = "NG" if c_data["is_ng"] else "OK"
        self.defects = []

class VisionPipeline:
    def __init__(self, config_path="config.yaml"):
        self.detector = DefectDetector()

    def initialize(self):
        pass

    def process_frame(self, frame):
        """核心解算管道：输入图像 -> 传统算法处理 -> 数据适配 -> 输出"""
        payload_data = []
        if frame is None:
            return frame, payload_data
            
        # 1. 将图像送入您的传统视觉引擎
        result_img, _, _, _, _, cartridges_data = self.detector.detect(frame)
        
        # 2. 遍历算法算出的结果，组装统一的通讯数据结构
        for c_data in cartridges_data:
            
            # =========================================================
            # 【核心过滤逻辑】：只处理和输出 NG 状态的药卷
            # =========================================================
            if not c_data["is_ng"]:
                continue  # 如果是 OK 药卷，直接跳过，不加入 payload_data
                
            # 使用适配器伪装对象 (走到这里的，一定都是 NG 的)
            target_obj = TargetAdapter(c_data)
            
            # 组装抓取点
            grasp_dict = {
                "status": "success",
                "grasp_point": {
                    "x_px": c_data["center_x"],
                    "y_px": c_data["center_y"],
                    "rz_deg": c_data["angle"]
                }
            }
            
            # 组装最终的通讯载荷 (此时 payload_data 里只会有 NG 的目标)
            payload_data.append({
                "target_status": target_obj.status,
                "grasp_data": grasp_dict,
                "target": target_obj 
            })
            
        return result_img, payload_data