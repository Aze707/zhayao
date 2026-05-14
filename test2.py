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
        # 红色药卷精准区间 (要求提取准确，宁可漏掉一点边缘，绝不能把背景提取进来)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([13, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 明亮黄色破损物精准区间
        self.lower_yellow = np.array([10, 35, 35])
        self.upper_yellow = np.array([45, 255, 255])
        
        # 3. 几何与拓扑参数
        self.min_red_area = 2000    # 红色药卷最小面积(过滤噪点)
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
        
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        
        # 红色精准清理
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # 黄色精准清理
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_open)
        
        return mask_red, mask_yellow

    def get_loose_combined_mask(self, img_bgr):
        """
        差分核心 1：极其宽松地提取所有目标（药卷主体 + 阴影褐色漏药）
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        # ====== 核心修改点 ======
        # a 通道 > 120 (排除偏绿的噪点)
        # b 通道从 128 提高到 136！
        # 136 是一个分水岭：它足以把深灰色的传送带（~130）无情地踩在脚下过滤掉，
        # 同时又足够低，能把阴影里暗褐色的漏药（~140+）和红色的药卷捞上来。
        lower_lab = np.array([0, 120, 136]) 
        upper_lab = np.array([255, 255, 255]) 
        # ========================
        
        mask_combined = cv2.inRange(lab, lower_lab, upper_lab)
        
        # 闭运算填补内部空洞，使其成为坚实的整体
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
        """主检测逻辑流（基于轮廓填充与差分）"""
        # 1. 预处理
        img_pre = self.preprocess(img)
        img_hsv = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
        
        # 2. 提取精准的红色药卷掩码 (Mask A) 和 明亮黄色掩码
        mask_red_accurate, mask_yellow_hsv = self.get_clean_masks(img_hsv)
        
        # ================= [填补 Mask A 的内部孔洞] =================
        # 数学逻辑：将有孔洞的A转化为绝对实心的 A_solid
        contours_A, _ = cv2.findContours(mask_red_accurate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red_solid = np.zeros_like(mask_red_accurate)
        cv2.drawContours(mask_red_solid, contours_A, -1, 255, thickness=cv2.FILLED)
        
        mask_red_accurate = mask_red_solid # 更新为实心掩码
        # ==============================================================

        # 3. 提取宽松的合并掩码 (Mask C = 药卷 + 暗色漏药)
        # 注意：此方法内的闭运算核切记不可太大，不能吞噬药卷间的物理缝隙！
        mask_combined = self.get_loose_combined_mask(img_pre)
        
        # ================= [边缘公差抵消与逻辑减法] =================
        # 4a. 膨胀实心红色掩码
        # 将 1x1 修改为 5x5。目的是让 A 胖一小圈，防止 C-A 运算后留下药卷的外轮廓线。
        # 如果您发现减法后还是有一圈白边，可以把 5x5 改为 7x7 或更大。
        kernel_dilate = np.ones((5, 5), np.uint8)
        mask_red_dilated = cv2.dilate(mask_red_accurate, kernel_dilate, iterations=1)
        
        # 4b. 逻辑减法 (C - A_dilated)
        mask_dark_spill = cv2.subtract(mask_combined, mask_red_dilated)

        # 4c. 开运算清理减法后残留的零星极小噪点 (保留 3x3 即可，5x5 可能会把小的真实漏药也刷掉)
        kernel_open = np.ones((3, 3), np.uint8)
        mask_dark_spill = cv2.morphologyEx(mask_dark_spill, cv2.MORPH_OPEN, kernel_open)
        # ==============================================================
        
        # 5. 合并最终的漏药掩码 (亮黄色 + 暗褐色)
        mask_yellow_final = cv2.bitwise_or(mask_yellow_hsv, mask_dark_spill)
        
        # 6. 提取目标轮廓 (使用实心的 mask_red_accurate 提取红圈，完全不影响外围判断)
        valid_red, valid_yellow = self.extract_contours(mask_red_accurate, mask_dark_spill)
        
        # 7. 空间拓扑关联判定
        defective_red_indices = set()
        for y_cnt in valid_yellow:
            M = cv2.moments(y_cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            yellow_center = (cx, cy)
            
            for idx, r_cnt in enumerate(valid_red):
                dist = cv2.pointPolygonTest(r_cnt, yellow_center, True)
                # 判断漏药中心距离药卷的距离，如果在阈值内，则判定为该药卷的 NG
                if dist >= self.distance_threshold:
                    defective_red_indices.add(idx)
                    break
                    
        # 8. 可视化输出
        # [修改点 1]：使用预处理后的亮图 (img_pre) 作为底图，解决亮度差异问题！
        result_img = img_pre.copy() 
        
        # [可选过滤]：如果想过滤掉图四背景里那几个比芝麻还小的白点，
        # 可以利用已经提取好的 valid_yellow，生成一个干净的掩码。
        # 如果您不在乎那几个小白点，可以直接用 result_img[mask_dark_spill == 255] = [0, 255, 255]
        clean_yellow_mask = np.zeros_like(mask_yellow_final)
        for y_cnt in valid_yellow:
            cv2.drawContours(clean_yellow_mask, [y_cnt], -1, 255, -1)
            
        # [修改点 2]：不使用生硬的轮廓描边，直接利用掩码进行“像素级染色”
        # 只要 clean_yellow_mask 中是白色的地方，就在底图上把它涂成纯黄色 (BGR: 0, 255, 255)
        result_img[clean_yellow_mask == 255] = [0, 255, 255]
        
        # 画红框和 NG 标签 (保持不变)
        for idx, r_cnt in enumerate(valid_red):
            if idx in defective_red_indices:
                cv2.drawContours(result_img, [r_cnt], -1, (0, 0, 255), 3) 
                x, y, w, h = cv2.boundingRect(r_cnt)
                cv2.putText(result_img, "NG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.drawContours(result_img, [r_cnt], -1, (0, 255, 0), 1) 
                
        # 返回：结果图, 精准红掩码(A), 最终黄掩码(减法结果), 宽松总掩码(C), 预处理图
        return result_img, mask_red_accurate, mask_dark_spill, mask_combined, img_pre
    
def imread_chinese(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ================= 使用示例 =================
if __name__ == "__main__":
    detector = DefectDetector()
    
    # image_path = r"G:\zhayao\2.bmp"
    # img = cv2.imread(image_path)
    
    img = imread_chinese(r"G:\zhayao\image\4.bmp")
    
    if img is None:
        print(f"Error: 无法加载图像")
    else:
        result, mask_r, mask_y, mask_combined, img_pre = detector.detect(img)
        
        # 颜色空间转换 (BGR -> RGB)
        img_pre_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1) 
        plt.imshow(img_pre_rgb)
        plt.title("1. Pre-processed Image")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(mask_r, cmap='gray')
        plt.title("2. Accurate Red Mask (A)")
        plt.axis('off')

        # [新增] 显示宽泛提取的合并掩码
        plt.subplot(2, 3, 3)
        plt.imshow(mask_combined, cmap='gray')
        plt.title("3. Combined Mask (C) \n[All Warm Colors]")
        plt.axis('off')

        # 显示减法得到的结果
        plt.subplot(2, 3, 4)
        plt.imshow(mask_y, cmap='gray')
        plt.title("4. Yellow Mask (C - A_dilated)")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(result_rgb)
        plt.title("5. Final Detection Result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()