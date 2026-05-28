"""
最优抓取点解算器 (Grasp Pose Solver)
支持模式：
1. 几何向量偏移防碰撞 (基于 YOLO-OBB 坐标)
2. 形态学多目标高精度解算 (基于二值掩膜，融合距离变换与质心惩罚)
"""
import cv2
import math
import json
import numpy as np
from skimage.morphology import skeletonize

class Defect:
    def __init__(self, label, xc, yc):
        self.label = label
        self.xc = xc
        self.yc = yc

class RollTarget:
    def __init__(self, xc, yc, w, h, theta, status="OK"):
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h
        self.theta = theta  # OBB 旋转角 (度)
        self.status = status
        self.defects = []

class GraspPose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  # 末端法兰盘需要旋转的最终角度
        
    def __repr__(self):
        return f"GraspPose(X:{self.x:.1f}, Y:{self.y:.1f}, Theta:{self.theta:.1f}°)"
    
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  # 末端法兰盘需要旋转的最终角度
        
    def __repr__(self):
        return f"GraspPose(X:{self.x:.1f}, Y:{self.y:.1f}, Theta:{self.theta:.1f}°)"

    def to_dict(self) -> dict:
        """输出为标准字典格式，保留两位小数"""
        return {
            "status": "success",
            "grasp_point": {
                "x_px": round(float(self.x), 2),
                "y_px": round(float(self.y), 2),
                "rz_deg": round(float(self.theta), 2)
            }
        }
        
    def to_json(self) -> str:
        """输出为 JSON 字符串，可直接通过 TCP Socket 发送给机械臂/PLC"""
        return json.dumps(self.to_dict())

class GraspSolver:
    def __init__(self, safe_margin=45, alpha=0.7, beta=0.3):
        """
        :param safe_margin_px: 纯几何计算时的安全回退距离 (像素)
        :param alpha: 形态学解算 - 饱满度 (距离变换) 权重
        :param beta: 形态学解算 - 重心平稳度 (距质心距离) 权重
        """
        self.safe_margin = safe_margin
        self.alpha = alpha
        self.beta = beta

    def calculate_geometric_offset(self, roll: RollTarget) -> GraspPose:
        """
        模式一：几何计算（保留您的原逻辑并进行小幅优化）
        适用场景：只拿到 OBB 框，无法获取精确掩膜时的高速处理。
        """
        opt_x, opt_y, opt_theta = roll.xc, roll.yc, roll.theta
        
        if not roll.defects:
            return GraspPose(opt_x, opt_y, opt_theta)
            
        # 针对最接近中心的致命缺陷进行避障
        defect = roll.defects[0] 
        dx = defect.xc - roll.xc
        dy = defect.yc - roll.yc
        
        angle_rad = math.radians(roll.theta)
        axis_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        dot_product = dx * axis_vec[0] + dy * axis_vec[1]
        direction = -1 if dot_product > 0 else 1
        
        opt_x = roll.xc + direction * self.safe_margin * axis_vec[0]
        opt_y = roll.yc + direction * self.safe_margin * axis_vec[1]
            
        return GraspPose(opt_x, opt_y, opt_theta)

    def calculate_morphological_optimal(self, safe_mask: np.ndarray) -> GraspPose:
        """
        模式二：核心形态学算法 (计算最优抓取点，遇弯曲/干瘪自动寻优)
        :param safe_mask: 剔除了缺陷区域并经过适当腐蚀(Erode)的安全抓取区二值图 (uint8: 255/0)
        """
        bool_mask = safe_mask > 0
        
        # 1. 提取骨架 (Skeletonize) - 确保抓取点在绝对脊线上
        skeleton = skeletonize(bool_mask).astype(np.uint8) * 255
        skel_y, skel_x = np.where(skeleton > 0)
        
        if len(skel_x) == 0:
            # 异常兜底：若安全区过小无法提取骨架，退化为形心
            M = cv2.moments(safe_mask)
            if M["m00"] != 0:
                return GraspPose(M["m10"]/M["m00"], M["m01"]/M["m00"], 0.0)
            return GraspPose(0, 0, 0)

        # 2. 距离变换 (Distance Transform) - 计算厚度饱满度得分
        dist_transform = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        if max_dist == 0: max_dist = 1
        norm_dist = dist_transform / max_dist  # 归一化到 [0, 1]

        # 3. 计算实际质心 - 用于抓取平衡性惩罚
        M = cv2.moments(safe_mask)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = np.mean(skel_x), np.mean(skel_y)

        # 4. 融合得分方程，寻找最优坐标 (Argmax)
        best_score = -1.0
        best_pt = (cx, cy)
        max_r = math.hypot(safe_mask.shape[1], safe_mask.shape[0])

        for x, y in zip(skel_x, skel_y):
            # S_thickness: 该点的厚度得分
            s_thick = norm_dist[y, x]
            
            # S_balance: 计算距质心距离，转化为惩罚项（越近得分越高）
            dist_to_center = math.hypot(x - cx, y - cy)
            s_bal = 1.0 - (dist_to_center / (max_r * 0.2)) # 控制惩罚衰减率
            s_bal = max(0.0, s_bal)
            
            # 综合得分矩阵函数
            score = self.alpha * s_thick + self.beta * s_bal
            
            if score > best_score:
                best_score = score
                best_pt = (x, y)

        opt_x, opt_y = best_pt

        # 5. 局部位姿解算
        opt_theta = self._calculate_local_normal(skeleton, opt_x, opt_y)

        return GraspPose(opt_x, opt_y, opt_theta)
    
    def calculate_optimal_point(self, roll: RollTarget, safe_mask: np.ndarray = None) -> GraspPose:
        """
        统一的抓取点解算入口，兼容旧版流水线调用。
        :param roll: 药卷目标对象 (包含 OBB 和缺陷信息)
        :param safe_mask: (可选) 如果流水线传入了安全二值掩膜，则启动高精度形态学解算
        """
        # 如果传入了掩膜图，说明前端已经做好了像素级处理，启用形态学最优解
        if safe_mask is not None:
            return self.calculate_morphological_optimal(safe_mask)
            
        # 默认回退到高速的纯几何偏移算法 (兼容您目前的 vision_pipeline.py)
        return self.calculate_geometric_offset(roll)

    def _calculate_local_normal(self, skeleton: np.ndarray, x: int, y: int, window_size: int = 15) -> float:
        """提取骨架局部窗口并进行最小二乘法直线拟合，求抓取法向角"""
        h, w = skeleton.shape
        x_min, x_max = max(0, x - window_size), min(w, x + window_size)
        y_min, y_max = max(0, y - window_size), min(h, y + window_size)
        
        roi = skeleton[y_min:y_max, x_min:x_max]
        pts_y, pts_x = np.where(roi > 0)
        
        if len(pts_x) < 3:
            return 0.0
            
        pts_x = pts_x + x_min
        pts_y = pts_y + y_min
        points = np.column_stack((pts_x, pts_y)).astype(np.float32)
        
        # 拟合局部直线 [vx, vy, x0, y0]
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # 计算切线角并转为法线角（即机械爪需要垂直抓取的角度）
        tangent_angle = math.degrees(math.atan2(vy[0], vx[0]))
        grasp_angle = tangent_angle + 90.0
        
        # 将角度归一化到 [-90, 90] 区间，匹配常规机械臂末端旋转限位
        grasp_angle = (grasp_angle + 90) % 180 - 90
            
        return grasp_angle