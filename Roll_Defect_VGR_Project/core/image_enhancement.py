import cv2
import numpy as np

class HomomorphicFilter:
    """
    工业级同态滤波器 (Homomorphic Filter)
    用于消除药卷柱面的光照衰减（暗角/阴影），同时增强表面高频缺陷特征。
    """
    def __init__(self, d0=30, rl=0.5, rh=2.0, c=1.0):
        """
        初始化滤波器参数。
        :param d0: 截止频率 (Cutoff frequency)。越小，越多的低频被保留。
        :param rl: 低频增益 (Low-frequency gain)。通常设为 0.3 ~ 0.8，用于压制阴影。
        :param rh: 高频增益 (High-frequency gain)。通常设为 1.5 ~ 2.5，用于增强特征对比度。
        :param c: 锐度控制因子 (Sharpening control)。控制高低频过渡段的斜率。
        """
        self.d0 = d0
        self.rl = rl
        self.rh = rh
        self.c = c

    def apply_filter(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        执行滤波运算。为了防止色彩失真，仅对 HSV 空间的 V(明度) 通道进行处理。
        :param img_bgr: 原始 BGR 图像
        :return: 均衡化后的 BGR 图像
        """
        # 1. 色彩空间转换 (工业界处理光照问题极少在 RGB 空间进行)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_v = img_hsv[:, :, 2].astype(np.float32)

        # 2. 图像填充以优化 FFT 运算速度 (节拍优化关键)
        rows, cols = img_v.shape
        opt_rows = cv2.getOptimalDFTSize(rows)
        opt_cols = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(img_v, 0, opt_rows - rows, 0, opt_cols - cols,
                                    cv2.BORDER_CONSTANT, value=0)

        # 3. 对数变换 (消除零点异常)
        # 使用 log1p (ln(1+x)) 替代 log，防止图像中存在绝对黑点(0)导致数学溢出
        img_log = np.log1p(padded)

        # 4. 离散傅里叶变换 (DFT)
        # 相比 numpy.fft，cv2.dft 经过 C++ 底层优化，在工业部署中延迟更低
        dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 5. 构建高斯同态滤波掩膜 H(u, v)
        H = self._build_gaussian_filter(opt_rows, opt_cols)

        # 6. 频域滤波 (掩膜与频谱相乘)
        # 将单通道的 H 扩展为双通道，以匹配 dft_shift 的实部和虚部
        H_dual = np.dstack([H, H])
        filtered_shift = dft_shift * H_dual

        # 7. 傅里叶逆变换 (IDFT) - [BUG 修复区]
        dft_ishift = np.fft.ifftshift(filtered_shift)
        
        # 修复点 1：必须添加 flags=cv2.DFT_SCALE，强制 OpenCV 进行 1/(W*H) 的归一化
        img_back_complex = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE)
        
        # 提取实部并计算幅值
        img_back = cv2.magnitude(img_back_complex[:, :, 0], img_back_complex[:, :, 1])

        # 修复点 2：[工业级防溢出保险] 
        # 原始图像的最大像素值 255 经过 log1p 映射后约为 5.545。
        # 经过高频增益后，合理数值绝不可能超过 10.0。
        # 这里强行截断频域计算带来的异常极值，彻底杜绝 e^x 溢出。
        np.clip(img_back, a_min=0.0, a_max=10.0, out=img_back)

        # 8. 指数逆变换并还原至原图尺寸
        img_exp = np.expm1(img_back) 
        img_exp = img_exp[:rows, :cols]

        # ---------------------------------------------------------
        # 9. 鲁棒性线性拉伸 (Robust Percentile Scaling) - [修复区]
        # ---------------------------------------------------------
        # 核心思想：抛弃最暗的 1% 和最亮的 0.5% 像素，防止归一化被极值绑架
        p_min = np.percentile(img_exp, 1.0)
        p_max = np.percentile(img_exp, 99.5) 
        
        # 防止纯色图像导致的除零错误
        if p_max - p_min < 1e-5:
            p_max = p_min + 1e-5
            
        # 手动映射到 0-255 并进行绝对截断
        img_norm = (img_exp - p_min) / (p_max - p_min) * 255.0
        img_v_filtered = np.uint8(np.clip(img_norm, 0, 255))
        # ---------------------------------------------------------

        # 10. 合并回 HSV 并转回 BGR
        img_hsv[:, :, 2] = img_v_filtered
        result_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        return result_bgr

    def _build_gaussian_filter(self, rows: int, cols: int) -> np.ndarray:
        """内部方法：构建频域滤波器掩膜矩阵"""
        center_row, center_col = int(rows / 2), int(cols / 2)
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # 计算距离矩阵的平方 D^2(u,v)
        # 注意中心点的偏移
        D2 = (u - center_col)**2 + (v - center_row)**2
        
        # 应用高斯同态滤波公式
        H = (self.rh - self.rl) * (1 - np.exp(-self.c * (D2 / (self.d0**2)))) + self.rl
        return H.astype(np.float32)

# ==========================================
# 在核心管道 (vision_pipeline.py) 中的调用示例
# ==========================================
# img_bgr = cv2.imread('drug_roll_shadow.jpg')
# homo_filter = HomomorphicFilter(d0=30, rl=0.5, rh=2.0)
# img_equalized = homo_filter.apply_filter(img_bgr)