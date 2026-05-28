"""
可视化与 Debug 模块
在图像上绘制 OBB 框、局部缺陷框与最终的十字抓取点
"""
# utils/visualizer.py
import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_results(frame, process_results):
        display_img = frame.copy()
        
        for res in process_results:
            target = res["target"]
            pose = res["grasp_pose"]
            
            # 颜色设定
            color = (0, 255, 0) if target.status == "OK" else (0, 0, 255)
            
            # 绘制 OBB 框
            rect = ((target.xc, target.yc), (target.w, target.h), target.theta)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(display_img, [box], 0, color, 2)
            cv2.putText(display_img, f"[{target.status}]", 
                        (box[1][0], box[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制缺陷黄圈
            for d in target.defects:
                cv2.circle(display_img, (int(d.xc), int(d.yc)), 15, (0, 255, 255), 2)
            
            # 绘制最优抓取点绿十字
            gx, gy = int(pose.x), int(pose.y)
            cv2.drawMarker(display_img, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(display_img, (gx, gy), 8, (0, 255, 0), -1)
            
        return display_img