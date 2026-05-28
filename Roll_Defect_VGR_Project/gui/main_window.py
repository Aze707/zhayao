# gui/main_window.py
import os
import cv2
import numpy as np
import time
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QRadioButton, QButtonGroup,
                             QGroupBox) 
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# 引入核心业务模块与可视化组件
from core.vision_pipeline import VisionPipeline
from utils.visualizer import Visualizer

class VisionWorker(QThread):
    """后台视觉工作线程，隔离于主 UI 线程以防卡死"""
    update_frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(int, int)
    grasp_data_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.is_running = False
        
        # 1. 实例化业务中枢与可视化工具
        self.pipeline = VisionPipeline("config.yaml")
        self.visualizer = Visualizer()
        
        self.ok_count = 0
        self.ng_count = 0

        # 数据源配置
        self.source_mode = "folder"  # "mock" / "folder" / "camera"
        self.real_image_dir = r"G:\zhayao\image"
        self.image_list = []
        self.image_index = 0
        
        # =========================================================================
        # 【核心新增】控制变量：用于实现手动单帧步进查验功能
        # =========================================================================
        self.is_manual_mode = False    # 是否开启手动控制模式
        self.next_frame_trigger = False # 手动触发下一张的电平开关
        
        self._init_source()

    def _init_source(self):
        """高级诊断版：初始化并深度检索真实照片数据源"""
        if self.source_mode == "folder":
            print("\n" + "="*50)
            print(f"[数据源诊断] 正在核验目标路径: {self.real_image_dir}")
            
            # 1. 检查文件夹在物理层面上是否存在
            if not os.path.exists(self.real_image_dir):
                print(f"[错误] 该路径在您的计算机中根本不存在！请检查盘符或拼写。")
                self.log_signal.emit(f"⚠️ 错误：未找到路径 [{self.real_image_dir}]，降级为 mock。")
                self.source_mode = "mock"
                return

            print(f"[成功] 目标文件夹存在。正在扫描内部文件...")
            all_files = os.listdir(self.real_image_dir)
            print(f"[诊断] 该文件夹下共有文件/文件夹数量: {len(all_files)} 个")
            
            # 如果文件数量大于0，打印前几个文件名看看它们长啥样
            if all_files:
                print(f"[诊断] 前5个文件名示例: {all_files[:5]}")
            
            # 2. 强力检索：全面兼容大写、小写后缀名
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            for f in all_files:
                # 转换成小写进行严苛对比
                if f.lower().endswith(valid_extensions):
                    self.image_list.append(os.path.join(self.real_image_dir, f))
                    
            self.image_list.sort()
            print(f"[结果] 成功匹配到符合图像格式的照片数量: {len(self.image_list)} 张")
            print("="*50 + "\n")
            
            if len(self.image_list) > 0:
                self.log_signal.emit(f"📂 本地照片库加载成功，共 {len(self.image_list)} 张图像。")
            else:
                self.log_signal.emit(f"⚠️ 警告：文件夹内无符合格式的照片，自动降级为 mock。")
                self.source_mode = "mock"

    def run(self):
        self.is_running = True
        self.log_signal.emit(f"🚀 视觉算法线程已启动。运行模式: [{self.source_mode}]")
        self.pipeline.initialize()
        
        # 记录上一帧图像，防止手动挂起时界面刷空白
        last_frame = self._mock_capture()
        last_payload = []
        
        # 标志位：指示是否需要读取并解算新的一帧
        need_process = True
        
        while self.is_running:
            # =========================================================================
            # 【核心单步控制逻辑】
            # =========================================================================
            if self.is_manual_mode:
                if self.next_frame_trigger:
                    # 用户点击了“下一张”按钮，放行一次读图与解算流程
                    need_process = True
                    self.next_frame_trigger = False # 消费掉当前触发信号
                else:
                    # 如果用户没点击，不执行新读图，直接渲染上一帧画面保持界面不卡死
                    need_process = False
            else:
                # 自动流模式下，永远放行读图
                need_process = True

            if need_process:
                # 1. 从文件夹检索最新画面
                frame = self._capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 2. 核心算法管道解算
                _, payload_data = self.pipeline.process_frame(frame)
                
                # 缓存最新状态数据，供手动挂起时持续渲染使用
                last_frame = frame
                last_payload = payload_data
                
                # 更新全局计数看板
                self._update_statistics(payload_data)
                # 抛出信号通知右侧 HMI 富文本打印
                self.grasp_data_signal.emit(payload_data)

            # 5. 画布层可视化渲染（用缓存或新读入的 frame 持续给 QLabel 塞图，保证响应）
            display_frame = self.visualizer.draw_results(last_frame, last_payload)
            self.update_frame_signal.emit(display_frame)
            
            time.sleep(0.03)

    def trigger_next_frame(self):
        """外部槽函数接口：点击下一张按钮时调用，打通阻断状态"""
        self.next_frame_trigger = True

    def stop(self):
        self.is_running = False
        self.wait()

    def _capture_frame(self) -> np.ndarray:
        if self.source_mode == "folder" and self.image_list:
            current_path = self.image_list[self.image_index]
            frame = cv2.imread(current_path)
            
            # 手动查验模式下，打印当前正在解析的文件名，极度方便核对特定 NG 药卷
            if self.is_manual_mode:
                self.log_signal.emit(f"🔍 步进单帧解析 -> 文件名: {os.path.basename(current_path)}")
                
            self.image_index = (self.image_index + 1) % len(self.image_list)
            return frame
        return self._mock_capture()

    def _mock_capture(self):
        return np.ones((480, 640, 3), dtype=np.uint8) * 60
        
    def _update_statistics(self, payload_data):
        for data in payload_data:
            if data["target_status"] == "OK": 
                self.ok_count += 1
            elif data["target_status"] == "NG":
                self.ng_count += 1
        if payload_data:
            self.stats_signal.emit(self.ok_count, self.ng_count)


class MainWindow(QMainWindow):
    """主窗口 UI 类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("柔性药卷缺陷检测与视觉引导系统 HMI")
        self.resize(1200, 650) # 稍微拓宽一点界面
        
        self.worker = VisionWorker()
        self.worker.update_frame_signal.connect(self.update_image)
        self.worker.log_signal.connect(self.append_log)
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.grasp_data_signal.connect(self.append_grasp_info)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧画面主显示视窗区
        self.video_label = QLabel("系统待命")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.video_label, stretch=3)
        
        # 右侧综合控制面板
        right_layout = QVBoxLayout()
        
        # 1. 运行/停止按钮
        self.btn_start = QPushButton("▶ 启动检测")
        self.btn_start.clicked.connect(self.toggle_system)
        right_layout.addWidget(self.btn_start)
        
        # =========================================================================
        # 【核心新增 UI 控键】增加控制组：单选框切换“自动流”与“手动单步”
        # =========================================================================
        mode_group = QGroupBox("采集模式切换")
        mode_layout = QHBoxLayout()
        
        self.radio_auto = QRadioButton("自动循环 (Auto)")
        self.radio_auto.setChecked(True) # 默认自动流
        self.radio_auto.toggled.connect(self.switch_run_mode)
        
        self.radio_manual = QRadioButton("手动查验 (Step)")
        self.radio_manual.toggled.connect(self.switch_run_mode)
        
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)
        
        # 2. 【核心新增 UI 控键】增加“下一张 ⏭”步进按钮
        self.btn_next = QPushButton("⏭ 下一张药卷照片 (Next)")
        self.btn_next.setEnabled(False) # 默认处于自动流模式下，单步按钮置灰不可点击
        self.btn_next.clicked.connect(self.worker.trigger_next_frame)
        right_layout.addWidget(self.btn_next)
        
        # 3. 数据实时解析状态栏日志框
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

    def switch_run_mode(self):
        """【新增槽函数】响应单选框切换，动态改写工作线程的内部逻辑开关"""
        if self.radio_manual.isChecked():
            self.worker.is_manual_mode = True
            self.btn_next.setEnabled(True)   # 激活“下一张”按钮
            self.append_log("<font color='orange'>ℹ️ 已切换至 [手动查验模式]，请点击下方步进按钮查看下一张图像效果。</font>")
        else:
            self.worker.is_manual_mode = False
            self.btn_next.setEnabled(False)  # 禁用“下一张”按钮
            self.append_log("<font color='orange'>ℹ️ 已恢复 [自动视频流模式]。</font>")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats(self, ok_val, ng_val):
        pass 

    def append_log(self, text):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.text_log.append(f"[{time_str}] {text}")

    def append_grasp_info(self, payload_data):
        if not payload_data:
            return
        time_str = datetime.now().strftime("%H:%M:%S")
        for idx, data in enumerate(payload_data):
            status = data["target_status"]
            grasp_dict = data["grasp_data"]
            
            color_hex = "#2ecc71" if status == "OK" else "#e74c3c"
            html_msg = f"<font color='#7f8c8d'>[{time_str}]</font> " \
                       f"<b>目标 #{idx+1}</b>: 状态=<font color='{color_hex}'><b>{status}</b></font>"
            
            if grasp_dict.get("status") == "success":
                pt = grasp_dict["grasp_point"]
                gx, gy, rz = pt["x_px"], pt["y_px"], pt["rz_deg"]
                html_msg += f" | 抓取点: <font color='#3498db'><b>X:{gx:.1f}, Y:{gy:.1f}, RZ:{rz:.1f}°</b></font>"
            else:
                html_msg += " | <font color='orange'>⚠️ 无法解算有效抓取区域</font>"
            
            self.text_log.append(html_msg)
        self.text_log.moveCursor(self.text_log.textCursor().End)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()