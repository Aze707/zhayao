# core/defect_detector.py
import cv2
import numpy as np
# 引入我们独立封装的工业级预处理工具
from core.image_enhancement import HomomorphicFilter 

class DefectDetector:
    def __init__(self):
        # ==========================================
        # 1. 光场与预处理参数重构
        # ==========================================
        # 使用同态滤波器替代原有的单通道 CLAHE，以获得全局照度展平能力
        # d0=30(截止频率), rl=0.5(压制阴影), rh=2.0(放大纹理特征)
        self.homo_filter = HomomorphicFilter(d0=30, rl=0.5, rh=2.0, c=1.0)
        
        # 保持双边滤波，用于压制同态滤波由于高频增益(rh>1)带来的本底噪声
        self.blur_d = 9
        self.blur_sigma = 75
        
        # ==========================================
        # 2. 颜色阈值 (HSV与LAB空间)
        # ==========================================
        # [注意：接入同态滤波后，图像整体亮度分布会改变，以下阈值需重新微调]
        # 红色药卷精准区间
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([13, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 明亮黄色破损物精准区间
        self.lower_yellow = np.array([10, 35, 35])
        self.upper_yellow = np.array([45, 255, 255])
        
        # 3. 几何与拓扑参数 (保持不变)
        self.min_red_area = 2000    
        self.min_yellow_area = 50   
        self.distance_threshold = -20 

    def preprocess(self, img):
        """
        图像预处理引擎 V2.0：
        频域同态滤波光照均衡化 -> 空间域双边滤波去噪
        """
        # 步骤 1: 频域重构 (消除药卷柱面阴影，增强物理拓扑边缘)
        # 这一步执行完毕后，药卷边缘的暗角将被大幅削弱，类似于在平面上打光
        img_homo = self.homo_filter.apply_filter(img)
        
        # 步骤 2: 边缘保留平滑 (压制高频噪声)
        # 同态滤波器的 rh 参数放大高频特征的同时，也会放大 Sensor 本底噪声
        # 使用双边滤波可以在抹平噪声的同时，死死保住药卷和缺陷的真实物理边界
        img_blur = cv2.bilateralFilter(img_homo, self.blur_d, self.blur_sigma, self.blur_sigma)
        
        return img_blur

    def get_clean_masks(self, img_hsv):
        """获取极其精准的红色药卷掩码 和 明亮黄色掩码"""
        mask_red1 = cv2.inRange(img_hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(img_hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
        
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)
        
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_open)
        
        return mask_red, mask_yellow

    def get_loose_combined_mask(self, img_bgr):
        """极其宽松地提取所有目标（药卷主体 + 漏药）"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lower_lab = np.array([0, 120, 136]) 
        upper_lab = np.array([255, 255, 255]) 
        mask_combined = cv2.inRange(lab, lower_lab, upper_lab)
        
        kernel_close = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
        return mask_combined

    def split_touching_cartridges(self, mask_solid, img_pre):
        """
        利用距离变换和分水岭算法，将粘连在一起的药卷掩码切分开。
        输入要求：最好是包含完整缺陷在内的主体掩码(mask_combined)，以防断裂。
        """
        dist_transform = cv2.distanceTransform(mask_solid, cv2.DIST_L2, 5)
        
        # 阈值化获取确定前景 (0.3 这个系数可以根据实际粘连程度微调)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        sure_bg = cv2.dilate(mask_solid, np.ones((3,3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 
        markers[unknown == 255] = 0 
        
        markers = cv2.watershed(img_pre, markers)
        
        separated_contours = []
        for label in range(2, np.max(markers) + 1): 
            target_mask = np.zeros_like(mask_solid, dtype=np.uint8)
            target_mask[markers == label] = 255
            
            cnts, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > self.min_red_area * 0.5: 
                    separated_contours.append(c)
                    
        return separated_contours

    def get_rotated_rect_info(self, contour):
        """获取标准化后的最小外接矩形信息 (统一纠正长宽与倾斜角)"""
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        
        # 强制长边为 length，短边为 width，并转换角度
        if w < h:
            length, width = h, w
            angle += 90
        else:
            length, width = w, h
            
        # 确保角度在 [-90, 90) 之间
        if angle >= 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return box, (cx, cy), length, width, angle

    def detect(self, img):
        """主检测逻辑流"""
        img_pre = self.preprocess(img)
        img_hsv = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
        
        mask_red_accurate, mask_yellow_hsv = self.get_clean_masks(img_hsv)
        
        # 1. 填补 Mask A 内部孔洞
        contours_A, _ = cv2.findContours(mask_red_accurate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red_solid = np.zeros_like(mask_red_accurate)
        cv2.drawContours(mask_red_solid, contours_A, -1, 255, thickness=cv2.FILLED)

        # 2. 提取宽松合并掩码 (包含药卷+漏药)，用于解决巨大缺陷导致的物理断裂问题
        mask_combined = self.get_loose_combined_mask(img_pre)

        # ================= [核心切分] =================
        # 使用包含缺陷的、完全连通的 mask_combined 进行分水岭切分
        valid_cartridges = self.split_touching_cartridges(mask_combined, img_pre)
        # ==============================================

        # 3. 差分提取漏药逻辑
        kernel_dilate = np.ones((5, 5), np.uint8)
        mask_red_dilated = cv2.dilate(mask_red_solid, kernel_dilate, iterations=1)
        # 从总掩码中减去胖了一圈的红色主体，剩下的就是阴影里的漏药
        mask_dark_spill = cv2.subtract(mask_combined, mask_red_dilated)
        kernel_open = np.ones((3, 3), np.uint8)
        mask_dark_spill = cv2.morphologyEx(mask_dark_spill, cv2.MORPH_OPEN, kernel_open)
        
        # 4. 合并所有漏药 (亮黄色 + 暗色)
        mask_yellow_final = cv2.bitwise_or(mask_yellow_hsv, mask_dark_spill)
        contours_yellow, _ = cv2.findContours(mask_dark_spill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_yellow = [cnt for cnt in contours_yellow if cv2.contourArea(cnt) > self.min_yellow_area]
        
        # 5. 拓扑关联判定
        defective_cartridge_indices = set()
        for y_cnt in valid_yellow:
            M = cv2.moments(y_cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            yellow_center = (cx, cy)
            
            for idx, c_cnt in enumerate(valid_cartridges):
                dist = cv2.pointPolygonTest(c_cnt, yellow_center, True)
                if dist >= self.distance_threshold:
                    defective_cartridge_indices.add(idx)
                    break
                    
        # ================= [清晰且分层的可视化绘制] =================
        # 我们使用预处理后的图作为底图，这样对比度更高，界面看起来更清晰
        result_img = img_pre.copy() 
        
        # 步骤 A：先进行像素级染色 (染漏药区域)
        clean_yellow_mask = np.zeros_like(mask_dark_spill)
        for y_cnt in valid_yellow:
            cv2.drawContours(clean_yellow_mask, [y_cnt], -1, 255, -1) # 过滤后纯净的漏药掩码
        
        # 在底图上，将纯净漏药掩码为白色的地方，涂成纯黄色 BGR(0, 255, 255)
        result_img[clean_yellow_mask == 255] = [0, 255, 255] 
        
        cartridges_data = []
        # print("\n=== 药卷几何尺寸分析报告 ===")
        
        # 步骤 B：只负责计算数据，不再重复画边框和文字！(画图行为已全部移交至 Visualizer)
        for idx, c_cnt in enumerate(valid_cartridges):
            box, center, length, width, angle = self.get_rotated_rect_info(c_cnt)
            
            cartridges_data.append({
                "id": idx + 1,
                "center_x": round(center[0], 2), "center_y": round(center[1], 2),
                "length": round(length, 2), "width": round(width, 2),
                "angle": round(angle, 2), "is_ng": idx in defective_cartridge_indices,
                # 额外保留原始轮廓，以防前端需要绘制不规则边缘
                "contour": c_cnt 
            })
            
            # status_str = "NG (漏药)" if idx in defective_cartridge_indices else "OK"
            # print(f"药卷 #{idx+1}: | 中心({cartridges_data[-1]['center_x']}, {cartridges_data[-1]['center_y']}) "
            #       f"| 长: {cartridges_data[-1]['length']} | 宽: {cartridges_data[-1]['width']} | 旋转角: {cartridges_data[-1]['angle']}°")
                
        return result_img, mask_red_solid, mask_dark_spill, mask_combined, img_pre, cartridges_data