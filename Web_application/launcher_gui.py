#!/usr/bin/env python3
"""
Cleft Detection System - 美观的GUI启动器
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

class CleftDetectionLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cleft Detection System - AI-based cleft detection system")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 设置现代化外观
        self.setup_style()
        
        # 应用程序路径
        self.app_dir = Path(__file__).parent
        self.exe_path = self.app_dir / "CleftDetectionApp.exe"
        self.gui_manager_path = self.app_dir / "gui_db_manager.py"
        
        # 创建UI
        self.create_ui()
        
        # 居中显示窗口
        self.center_window()
        
        # 设置图标（如果有的话）
        try:
            # 尝试设置图标
            pass
        except:
            pass

    def setup_style(self):
        """设置现代化样式"""
        style = ttk.Style()
        
        # 设置主题
        try:
            style.theme_use('clam')
        except:
            style.theme_use('default')
        
        # 自定义样式
        style.configure('Title.TLabel', 
                       font=('Microsoft YaHei UI', 20, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Subtitle.TLabel',
                       font=('Microsoft YaHei UI', 10),
                       foreground='#7f8c8d')
        
        style.configure('Action.TButton',
                       font=('Microsoft YaHei UI', 12, 'bold'),
                       padding=(20, 10))
        
        style.configure('Secondary.TButton',
                       font=('Microsoft YaHei UI', 10),
                       padding=(15, 8))

    def create_ui(self):
        """创建用户界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题区域
        self.create_header(main_frame)
        
        # 状态区域
        self.create_status_area(main_frame)
        
        # 按钮区域
        self.create_buttons(main_frame)
        
        # 信息区域
        self.create_info_area(main_frame)

    def create_header(self, parent):
        """创建标题区域"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 主标题
        title_label = ttk.Label(header_frame, 
                               text="🦷 Cleft Detection System",
                               style='Title.TLabel')
        title_label.pack(anchor=tk.CENTER)
        
        # 副标题
        subtitle_label = ttk.Label(header_frame,
                                  text="Cleft Detection System - AI-based cleft detection system",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(anchor=tk.CENTER, pady=(5, 0))

    def create_status_area(self, parent):
        """创建状态显示区域"""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding="15")
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 状态信息
        self.status_var = tk.StringVar(value="System Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W)
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

    def create_buttons(self, parent):
        """创建按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 主启动按钮
        start_btn = ttk.Button(button_frame,
                              text="🚀 Start Cleft Detection System",
                              style='Action.TButton',
                              command=self.start_main_app)
        start_btn.pack(fill=tk.X, pady=(0, 10))
        
        # 次要按钮框架
        secondary_frame = ttk.Frame(button_frame)
        secondary_frame.pack(fill=tk.X)
        
        # 数据库管理按钮
        db_btn = ttk.Button(secondary_frame,
                           text="📊 Database Management",
                           style='Secondary.TButton',
                           command=self.start_db_manager)
        db_btn.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # 打开文件夹按钮
        folder_btn = ttk.Button(secondary_frame,
                               text="📁 Open Program Folder",
                               style='Secondary.TButton',
                               command=self.open_folder)
        folder_btn.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

    def create_info_area(self, parent):
        """创建信息区域"""
        info_frame = ttk.LabelFrame(parent, text="Usage Instructions", padding="15")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, 
                           font=('Microsoft YaHei UI', 9),
                           bg='#f8f9fa', relief=tk.FLAT)
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # 插入说明文字
        info_content = """
🔹 Click "Start Cleft Detection System" to start the main application
🔹 After system starts, it will automatically open http://127.0.0.1:5002 in the browser
🔹 Support multiple detection models: facial symmetry, nasal ratio, nasal column angle
🔹 Click "Database Management" to manage user and case data

⚠️ Attention:
• Please ensure port 5002 is not occupied
• It is recommended to use Chrome or Edge browser for the best experience
• Closing this launcher will not affect the main program
        """
        
        info_text.insert(tk.END, info_content.strip())
        info_text.config(state=tk.DISABLED)

    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def start_main_app(self):
        """启动主应用程序"""
        if not self.exe_path.exists():
            messagebox.showerror("Error", 
                               f"Can't find the application file:\n{self.exe_path}\n\nPlease ensure the program is correctly packaged.")
            return
        
        def run_app():
            try:
                # 更新状态
                self.status_var.set("Starting main application...")
                self.progress.start()
                
                # 启动应用程序
                process = subprocess.Popen([str(self.exe_path)], 
                                         cwd=str(self.exe_path.parent))
                
                # 等待一段时间让程序启动
                time.sleep(5)
                
                # 尝试打开浏览器
                try:
                    webbrowser.open('http://127.0.0.1:5002')
                    self.status_var.set("Application started, browser opened")
                except:
                    self.status_var.set("Application started, please manually open http://127.0.0.1:5002")
                
                self.progress.stop()
                
                # 询问是否最小化启动器
                if messagebox.askyesno("Success", 
                                     "Application started successfully!\n\nDo you want to minimize this launcher?\n(Minimized launcher can be restored from system tray)"):
                    self.root.iconify()
                
            except Exception as e:
                self.progress.stop()
                self.status_var.set("Failed to start")
                messagebox.showerror("Failed to start", f"Failed to start the application:\n{str(e)}")
        
        # 在新线程中运行，避免阻塞UI
        threading.Thread(target=run_app, daemon=True).start()

    def start_db_manager(self):
        """启动数据库管理器"""
        import sys, subprocess, os
        exe_path = os.path.join(self.app_dir, "DBManager.exe")
        py_path = os.path.join(self.app_dir, "gui_db_manager.py")
        if os.path.exists(exe_path):
            try:
                subprocess.Popen([exe_path], cwd=str(self.app_dir))
                self.status_var.set("Database manager started (EXE)")
            except Exception as e:
                messagebox.showerror("Failed to start", f"Failed to start DBManager.exe:\n{str(e)}")
        elif os.path.exists(py_path):
            try:
                subprocess.Popen([sys.executable, py_path], cwd=str(self.app_dir))
                self.status_var.set("Database manager started (Python)")
            except Exception as e:
                messagebox.showerror("Failed to start", f"Failed to start gui_db_manager.py:\n{str(e)}")
        else:
            messagebox.showerror("Error", f"Can't find the database manager file:\n{exe_path}\nOr\n{py_path}")

    def open_folder(self):
        """打开程序文件夹"""
        try:
            subprocess.Popen(['explorer', str(self.app_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"Can't open the folder:\n{str(e)}")

    def run(self):
        """运行启动器"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        launcher = CleftDetectionLauncher()
        launcher.run()
    except Exception as e:
        messagebox.showerror("Launcher error", f"Failed to initialize launcher:\n{str(e)}")

if __name__ == "__main__":
    main()