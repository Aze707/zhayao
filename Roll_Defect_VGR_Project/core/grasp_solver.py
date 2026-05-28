"""
最优抓取点解算器
核心逻辑：基于向量偏移防碰撞 / 基于形态学骨架处理弯曲
"""
# core/grasp_solver.py
import math
import numpy as np

class Defect:
    def __init__(self, label, xc, yc):
        self.label = label
        self.xc = xc
        self.yc = yc

class RollTarget:
    def __init__(self, xc, yc, w, h, theta, status="OK"):
        self.xc = xc; self.yc = yc
        self.w = w; self.h = h
        self.theta = theta
        self.status = status
        self.defects = []

class GraspPose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class GraspSolver:
    def __init__(self, safe_margin=45):
        self.safe_margin = safe_margin

    def calculate_optimal_point(self, roll: RollTarget) -> GraspPose:
        """核心算法：计算最优抓取点，遇缺陷自动反向偏置"""
        opt_x, opt_y, opt_theta = roll.xc, roll.yc, roll.theta
        
        if not roll.defects:
            return GraspPose(opt_x, opt_y, opt_theta)
            
        for defect in roll.defects:
            dx = defect.xc - roll.xc
            dy = defect.yc - roll.yc
            
            angle_rad = math.radians(roll.theta)
            axis_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            
            dot_product = dx * axis_vec[0] + dy * axis_vec[1]
            direction = -1 if dot_product > 0 else 1
            
            opt_x = roll.xc + direction * self.safe_margin * axis_vec[0]
            opt_y = roll.yc + direction * self.safe_margin * axis_vec[1]
            break # 假设优先处理主要缺陷
            
        return GraspPose(opt_x, opt_y, opt_theta)