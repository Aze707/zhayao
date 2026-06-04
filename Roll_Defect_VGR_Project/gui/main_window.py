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

        self.cap = None
        
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

        elif self.source_mode == "camera":
            print("\n" + "="*50)
            print("[硬件诊断] 正在尝试拉起物理相机数据流...")
            
            # 初始化相机，0 通常指代系统默认的第一个摄像头（如笔记本自带或首个 USB 相机）
            # 工业部署提示：如果是工业相机（如大恒、海康），通常建议用对应厂商的 Python SDK 获取图像，再转为 Numpy
            # 若仍用 OpenCV 调取 USB 工业相机，Windows 下推荐加上 cv2.CAP_DSHOW 提速并避免黑屏
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            # 尝试设置期望分辨率 (根据你的药卷检测视野需求调整)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if not self.cap.isOpened():
                print("[致命错误] 无法建立相机硬件连接！请检查 USB 接口、设备管理器驱动或是否被其他软件占用。")
                self.log_signal.emit("⚠️ 致命错误：硬件相机调用失败，强制降级为 mock 模式。")
                self.source_mode = "mock"
            else:
                # 读取实际生效的分辨率用于日志确认
                actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"[成功] 相机已挂载！当前物理分辨率: {actual_w} x {actual_h}")
                self.log_signal.emit(f"📷 物理相机连接成功，分辨率: {int(actual_w)}x{int(actual_h)}。")
            print("="*50 + "\n")

    def run(self):
        self.is_running = True
        self.log_signal.emit(f"🚀 视觉算法线程已启动。运行模式: [{self.source_mode}]")
        self.pipeline.initialize()
        
        # =========================================================================
        # 1. 变量安全初始化 (解决 UnboundLocalError 的核心)
        # =========================================================================
        # 获取一张初始兜底图
        initial_frame = self._capture_frame() if self.source_mode != "mock" else self._mock_capture()
        if initial_frame is None:
            initial_frame = self._mock_capture()
            
        # 初始状态缓存，防止手动模式下刚启动时崩溃
        processed_frame = initial_frame.copy() 
        last_payload = []
        
        # 是否需要进行新一轮解算的控制开关
        need_process = True
        
        while self.is_running:
            # 2. 状态机判断：决定当前帧要不要更新数据
            if self.is_manual_mode:
                if self.next_frame_trigger:
                    need_process = True
                    self.next_frame_trigger = False # 消费掉触发信号
                else:
                    need_process = False # 手动模式挂起，只用旧数据渲染
            else:
                need_process = True # 自动模式，永远放行

            # 3. 核心解算流 (只在 need_process 为 True 时执行)
            if need_process:
                frame = self._capture_frame()
                if frame is not None:
                    # 只有成功捕获到新图像，才去更新我们的渲染底图和解算数据
                    processed_frame, last_payload = self.pipeline.process_frame(frame)
                    
                    # 触发统计更新与右侧富文本打印
                    self._update_statistics(last_payload)
                    self.grasp_data_signal.emit(last_payload)
                else:
                    # 如果突发抓图失败，短暂休眠，避免死循环
                    time.sleep(0.01)
                    continue

            # 4. 统一渲染层
            # 此时的 processed_frame 绝对是安全定义的（要么是初始化好的底图，要么是刚算出来的底图）
            # last_payload 也绝对是安全的，只保存 NG 数据的字典列表
            display_frame = self.visualizer.draw_results(processed_frame, last_payload)
            
            # 推送给 HMI 显示
            self.update_frame_signal.emit(display_frame)
            
            # 帧率控制
            time.sleep(0.05)

    def trigger_next_frame(self):
        """外部槽函数接口：点击下一张按钮时调用，打通阻断状态"""
        self.next_frame_trigger = True

    def stop(self):
        self.is_running = False
        self.wait() # 阻塞当前上下文，等待 QThread 的 run() 循环彻底结束
        
        # 【新增】安全注销硬件资源
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[资源回收] 相机句柄已安全释放。")
            self.log_signal.emit("🛑 物理相机已断开。")

    def _capture_frame(self) -> np.ndarray:
        # 1. 文件夹模式抓图逻辑
        if self.source_mode == "folder" and self.image_list:
            current_path = self.image_list[self.image_index]
            frame = cv2.imread(current_path)
            if self.is_manual_mode:
                self.log_signal.emit(f"🔍 步进单帧解析 -> 文件名: {os.path.basename(current_path)}")
            self.image_index = (self.image_index + 1) % len(self.image_list)
            return frame
            
        # =========================================================================
        # 2. 【核心新增】物理相机实时抽帧逻辑
        # =========================================================================
        elif self.source_mode == "camera" and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            else:
                # 工业现场常见异常：线缆松动导致的突然丢帧，需防止 None 穿透进 YOLO 导致引擎崩溃
                self.log_signal.emit("⚠️ 警告：底图抓取失败（可能遭遇瞬时掉线或丢帧）。")
                return self._mock_capture()
                
        # 3. 兜底逻辑
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