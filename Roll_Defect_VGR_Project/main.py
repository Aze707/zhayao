# main.py
import sys
from PyQt5.QtWidgets import QApplication

# 模块化导入独立的 GUI 模块
from gui.main_window import MainWindow

def main():
    """系统入口函数"""
    # 1. 实例化 QApplication，所有的 PyQt 应用都必须有且仅有一个 QApplication 对象
    app = QApplication(sys.argv)
    
    # 2. 设置全局的 UI 风格 (Fusion 风格更贴近工业沉稳感)
    app.setStyle("Fusion")
    
    # 3. 实例化我们封装好的主窗口
    window = MainWindow()
    
    # 4. 显示窗口
    window.show()
    
    # 5. 进入主事件循环，阻塞等待直至用户关闭窗口
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()