# utils/visualizer.py
import cv2
import math
import numpy as np

class Visualizer:
    @staticmethod
    def draw_results(frame, payload_data):
        """
        流水线结果渲染器 - 统一画布绘制
        :param frame: 必须是当前捕获到的真实图像 (BGR)
        :param payload_data: 来自 vision_pipeline 的解算数据字典列表
        """
        display_img = frame.copy() if frame is not None else None
        if display_img is None:
            return frame
        
        for data in payload_data:
            # 1. 提取状态与解算字典
            status = data["target_status"]
            grasp_dict = data["grasp_data"]
            
            # 颜色设定：OK 为绿，NG 为红
            color = (0, 255, 0) if status == "OK" else (0, 0, 255)
            
            # 2. 提取并恢复 target 对象属性，绘制检测轮廓
            target = data.get("target") 
            if target is not None:
                # 绘制 YOLO-OBB 包围盒
                rect = ((target.xc, target.yc), (target.w, target.h), target.theta)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(display_img, [box], 0, color, 2)
                cv2.putText(display_img, f"[{status}]", 
                            (box[1][0], box[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 绘制缺陷黄圈
                for d in target.defects:
                    cv2.circle(display_img, (int(d.xc), int(d.yc)), 15, (0, 255, 255), 2)

            # 3. 提取最优抓取位姿数据并绘制
            if grasp_dict.get("status") == "success":
                grasp_pt = grasp_dict["grasp_point"]
                gx = int(grasp_pt["x_px"])
                gy = int(grasp_pt["y_px"])
                theta = grasp_pt["rz_deg"]
                
                # 绘制高亮同心圆抓取靶心
                cv2.drawMarker(display_img, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
                cv2.circle(display_img, (gx, gy), radius=4, color=(0, 255, 0), thickness=-1)
                cv2.circle(display_img, (gx, gy), radius=10, color=(0, 255, 0), thickness=2)
                
                # 绘制夹爪法向矢量箭头
                theta_rad = math.radians(theta)
                end_x = int(gx + 40 * math.cos(theta_rad))
                end_y = int(gy - 40 * math.sin(theta_rad)) # 图像Y轴向下，故用减
                cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), color=(0, 165, 255), thickness=3, tipLength=0.2)
                
                # 实时打上浮动文本标签
                text = f"X:{gx} Y:{gy} RZ:{theta:.1f}"
                cv2.putText(display_img, text, (gx + 15, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        return display_img

def draw_grasp_pose(image: np.ndarray, pose, target_color=(0, 255, 0)) -> np.ndarray:
    """单点轻量化快速渲染接口"""
    vis_img = image.copy()
    x, y = int(pose.x), int(pose.y)
    theta_rad = math.radians(pose.theta)
    
    cv2.circle(vis_img, (x, y), radius=5, color=target_color, thickness=-1)
    cv2.circle(vis_img, (x, y), radius=12, color=target_color, thickness=2)
    
    length = 20
    cv2.line(vis_img, (x - length, y), (x + length, y), target_color, 1)
    cv2.line(vis_img, (x, y - length), (x, y + length), target_color, 1)
    
    end_x = int(x + 40 * math.cos(theta_rad))
    end_y = int(y - 40 * math.sin(theta_rad))
    cv2.arrowedLine(vis_img, (x, y), (end_x, end_y), color=(0, 165, 255), thickness=3, tipLength=0.2)
    
    text = f"X:{x} Y:{y} RZ:{pose.theta:.1f}"
    cv2.putText(vis_img, text, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return vis_img