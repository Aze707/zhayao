# gui/main_window.py
import cv2
import numpy as np
import time
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# 引入核心业务模块
from core.vision_pipeline import VisionPipeline
from utils.visualizer import Visualizer

class VisionWorker(QThread):
    """后台视觉工作线程，隔离于主 UI 线程以防卡死"""
    update_frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.is_running = False
        # 实例化业务中枢与可视化工具
        self.pipeline = VisionPipeline("config.yaml")
        self.visualizer = Visualizer()
        
        self.ok_count = 0
        self.ng_count = 0

    def run(self):
        self.is_running = True
        self.log_signal.emit("✅ 视觉检测与解算线程启动。")
        self.pipeline.initialize()
        
        while self.is_running:
            # 1. 获取图像 (产线中此处应为 self.camera.capture())
            frame = self._mock_capture()
            
            # 2. 调用核心业务流水线：推理 + 解算抓取点
            results = self.pipeline.process_frame(frame)
            
            # 3. 统计与日志逻辑
            self._update_statistics(results)
            
            # 4. 画面可视化绘制
            display_frame = self.visualizer.draw_results(frame, results)
            
            # 5. 发送信号更新 UI
            self.update_frame_signal.emit(display_frame)
            time.sleep(0.05)

    def stop(self):
        self.is_running = False
        self.wait()

    def _mock_capture(self):
        # 模拟生成传送带背景图像（开发调试用）
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 60
        return frame
        
    def _update_statistics(self, results):
        for res in results:
            if res["target"].status == "OK":
                self.ok_count += 1
            else:
                self.ng_count += 1
                self.log_signal.emit(f"🔴 发现缺陷：{res['target'].status}")
        
        if results:
            self.stats_signal.emit(self.ok_count, self.ng_count)

class MainWindow(QMainWindow):
    """主窗口 UI 类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("柔性药卷缺陷检测与视觉引导系统 HMI")
        self.resize(1100, 650)
        
        # 实例化并绑定工作线程
        self.worker = VisionWorker()
        self.worker.update_frame_signal.connect(self.update_image)
        self.worker.log_signal.connect(self.append_log)
        self.worker.stats_signal.connect(self.update_stats)

        self.init_ui()

    def init_ui(self):
        # UI 布局代码：包含画面区、生产统计看板、控制按钮和日志区
        # 此处使用垂直与水平布局器组合构建界面
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 画面区
        self.video_label = QLabel("系统待命")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.video_label, stretch=3)
        
        # 右侧控制面板
        right_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("▶ 启动检测")
        self.btn_start.clicked.connect(self.toggle_system)
        right_layout.addWidget(self.btn_start)
        
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        right_layout.addWidget(self.text_log, stretch=1)
        
        main_layout.addLayout(right_layout, stretch=1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def toggle_system(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.btn_start.setText("■ 停止运行")
        else:
            self.worker.stop()
            self.btn_start.setText("▶ 启动检测")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats(self, ok_val, ng_val):
        pass # 统计看板更新逻辑

    def append_log(self, text):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.text_log.append(f"[{time_str}] {text}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()