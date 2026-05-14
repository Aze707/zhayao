import cv2
import numpy as np
import matplotlib.pyplot as plt

class DefectDetector:
    def __init__(self):
        # 1. 预处理参数
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.blur_d = 9
        self.blur_sigma = 75
        
        # 2. 颜色阈值 (HSV空间)
        # 红色药卷精准区间
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([13, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 明亮黄色破损物精准区间
        self.lower_yellow = np.array([10, 35, 35])
        self.upper_yellow = np.array([45, 255, 255])
        
        # 3. 几何与拓扑参数
        self.min_red_area = 2000    # 红色药卷最小面积
        self.min_yellow_area = 50   # 黄色漏药最小面积
        self.distance_threshold = -20 # 黄色中心点到红色轮廓的允许最大距离
        

    def preprocess(self, img):
        """图像预处理：CLAHE光照均衡化 + 双边滤波去噪"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        img_blur = cv2.bilateralFilter(img_clahe, self.blur_d, self.blur_sigma, self.blur_sigma)
        return img_blur

    def get_clean_masks(self, img_hsv):
        """获取极其精准的红色药卷掩码 和 明亮黄色掩码"""
        mask_red1 = cv2.inRange(img_hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(img_hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
        
        # 恢复较大的闭运算核，让药卷尽可能粘连成实心，方便后续一刀切
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)
        
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_open)
        
        return mask_red, mask_yellow
    
    def separate_adhesion_morphology(self, mask_binary):
        """
        [终极解法]：各向异性形态学切分
        利用垂直向的结构元素，直接切断水平方向长条物体之间的微弱粘连。
        完全免疫由于药卷缺角、错位带来的几何计算误差。
        """
        # 1. 第一步不能省：填补内部孔洞，确保药卷内部是绝对实心的
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(mask_binary)
        cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)
        
        # ================= [核心魔法] =================
        # 2. 定义各向异性核 (竖直长条形)
        # kernel_h: 必须 > 粘连缝隙的垂直厚度，且 < 单根药卷的垂直厚度。25是个极佳的起始点。
        # kernel_w: 稍微给一点宽度(3)，以容忍药卷在画面中轻微的倾斜。
        kernel_h = 25
        kernel_w = 3
        kernel_vertical = np.ones((kernel_h, kernel_w), np.uint8)
        
        # 3. 对实心掩码进行开运算 (先腐蚀切断细弱粘连，再膨胀恢复药卷原始体积)
        separated_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, kernel_vertical)
        # ===============================================
        
        return separated_mask

    def get_loose_combined_mask(self, img_bgr):
        """极其宽松地提取所有目标（药卷主体 + 阴影褐色漏药）"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lower_lab = np.array([0, 120, 136]) 
        upper_lab = np.array([255, 255, 255]) 
        mask_combined = cv2.inRange(lab, lower_lab, upper_lab)
        
        kernel_close = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
        return mask_combined

    def extract_contours(self, mask_red, mask_yellow):
        """提取并过滤轮廓"""
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) > self.min_red_area]
        valid_yellow = [cnt for cnt in contours_yellow if cv2.contourArea(cnt) > self.min_yellow_area]
        
        return valid_red, valid_yellow

    def detect(self, img):
        """主检测逻辑流"""
        # 1. 预处理
        img_pre = self.preprocess(img)
        img_hsv = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
        
        # 2. 提取基础掩码
        mask_red_accurate, mask_yellow_hsv = self.get_clean_masks(img_hsv)
        
        # ================= [几何切割法解粘连] =================
        # 代替失败的分水岭：直接输出填实内部且被切断粘连的完美掩码
        mask_red_accurate = self.separate_adhesion_morphology(mask_red_accurate)
        # =========================================================

        # 3. 提取宽松合并掩码
        mask_combined = self.get_loose_combined_mask(img_pre)
        
        # 4. 减法逻辑获取漏药 (C - A_dilated)
        kernel_dilate = np.ones((5, 5), np.uint8)
        mask_red_dilated = cv2.dilate(mask_red_accurate, kernel_dilate, iterations=1)
        mask_dark_spill = cv2.subtract(mask_combined, mask_red_dilated)

        kernel_open = np.ones((3, 3), np.uint8)
        mask_dark_spill = cv2.morphologyEx(mask_dark_spill, cv2.MORPH_OPEN, kernel_open)
        
        # 5. 合并最终漏药掩码
        mask_yellow_final = cv2.bitwise_or(mask_yellow_hsv, mask_dark_spill)
        
        # 6. 提取轮廓判定
        valid_red, valid_yellow = self.extract_contours(mask_red_accurate, mask_dark_spill)
        
        # 7. 空间拓扑关联
        defective_red_indices = set()
        for y_cnt in valid_yellow:
            M = cv2.moments(y_cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            yellow_center = (cx, cy)
            
            for idx, r_cnt in enumerate(valid_red):
                dist = cv2.pointPolygonTest(r_cnt, yellow_center, True)
                if dist >= self.distance_threshold:
                    defective_red_indices.add(idx)
                    break
                    
        # 8. 可视化输出
        result_img = img_pre.copy() 
        clean_yellow_mask = np.zeros_like(mask_yellow_final)
        for y_cnt in valid_yellow:
            cv2.drawContours(clean_yellow_mask, [y_cnt], -1, 255, -1)
            
        result_img[clean_yellow_mask == 255] = [0, 255, 255]
        
        for idx, r_cnt in enumerate(valid_red):
            if idx in defective_red_indices:
                cv2.drawContours(result_img, [r_cnt], -1, (0, 128, 0), 25) 
                x, y, w, h = cv2.boundingRect(r_cnt)
                cv2.putText(result_img, "NG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.drawContours(result_img, [r_cnt], -1, (0, 255, 0), 2) 
                
        return result_img, mask_red_accurate, mask_dark_spill, mask_combined, img_pre
    
def imread_chinese(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ================= 使用示例 =================
if __name__ == "__main__":
    detector = DefectDetector()
    
    img = imread_chinese(r"G:\zhayao\image\6.bmp")
    
    if img is None:
        print(f"Error: 无法加载图像")
    else:
        result, mask_r, mask_y, mask_combined, img_pre = detector.detect(img)
        
        img_pre_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1) 
        plt.imshow(img_pre_rgb)
        plt.title("1. Pre-processed")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(mask_r, cmap='gray')
        plt.title("2. Cut Red Mask (A)")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(mask_combined, cmap='gray')
        plt.title("3. Combined Mask (C)")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(mask_y, cmap='gray')
        plt.title("4. Spill Mask")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(result_rgb)
        plt.title("5. Final Result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()