#!/usr/bin/env python3
"""
GUI Database Management Tool - Real-time user and database management
"""

# 添加错误处理和版本检查
import sys
import os
import platform

# 显示Python环境信息
print(f"Python版本: {platform.python_version()}")
print(f"系统平台: {platform.system()} {platform.release()}")
print(f"运行路径: {os.path.dirname(os.path.abspath(__file__))}")

# 尝试导入tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, simpledialog
    print("成功导入tkinter")
except ImportError as e:
    print(f"错误: 无法导入tkinter库: {e}")
    print("请确保已正确安装Python和tkinter")
    print("常见解决方法:")
    print("1. 安装Python 3.7-3.10版本")
    print("2. 确保安装了tkinter模块")
    print("   Windows: 在Python安装时勾选'tcl/tk和IDLE'")
    print("   Linux: sudo apt-get install python3-tk")
    
    # 尝试显示一个简单的错误对话框
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"无法导入tkinter库: {e}\n\n请确保已正确安装Python和tkinter。", "GUI管理器错误", 0x10)
    except:
        pass
        
    sys.exit(1)

import sqlite3
import shutil
from datetime import datetime
import threading
import json
import random, string

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DatabaseGUIManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Database Management Tool - Cleft Lip Detection System")
        self.root.geometry("1200x800")
        
        # 数据库路径 - 指向flask_cleft_demo目录下的instance文件夹
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'database.db')
        
        # 检查数据库是否存在
        if not os.path.exists(self.db_path):
            messagebox.showerror("Error", f"Database file does not exist: {self.db_path}")
            self.root.destroy()
            return
            
        self.setup_ui()
        self.refresh_data()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title = ttk.Label(self.main_frame, text="Database Management Tool", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 左侧控制面板
        self.setup_control_panel()
        
        # 右侧数据显示
        self.setup_data_panel()
        
    def setup_control_panel(self):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Database information
        info_frame = ttk.LabelFrame(control_frame, text="Database Information", padding="5")
        info_frame.pack(fill="x", pady=5)
        
        self.db_info = tk.Text(info_frame, height=6, width=30)
        self.db_info.pack(fill="x")
        
        # User management buttons
        user_frame = ttk.LabelFrame(control_frame, text="User Management", padding="5")
        user_frame.pack(fill="x", pady=5)
        
        ttk.Button(user_frame, text="Refresh Data", command=self.force_refresh).pack(fill="x", pady=2)
        ttk.Button(user_frame, text="Add User", command=self.add_user).pack(fill="x", pady=2)
        ttk.Button(user_frame, text="Delete User", command=self.delete_user).pack(fill="x", pady=2)
        ttk.Button(user_frame, text="Reset Password", command=self.reset_password).pack(fill="x", pady=2)
        
        # Database operation buttons
        db_frame = ttk.LabelFrame(control_frame, text="Database Operations", padding="5")
        db_frame.pack(fill="x", pady=5)
        
        ttk.Button(db_frame, text="Backup Database", command=self.backup_database).pack(fill="x", pady=2)
        ttk.Button(db_frame, text="Export Data", command=self.export_data).pack(fill="x", pady=2)
        ttk.Button(db_frame, text="Clean Data", command=self.clean_data).pack(fill="x", pady=2)
        ttk.Button(db_frame, text="SQL Query", command=self.sql_query).pack(fill="x", pady=2)
        ttk.Button(db_frame, text="Admin Status", command=self.show_admin_status).pack(fill="x", pady=2)
        
        # Doctor code management
        code_frame = ttk.LabelFrame(control_frame, text="Doctor Code Management", padding="5")
        code_frame.pack(fill="x", pady=5)
        ttk.Button(code_frame, text="Generate Doctor Codes", command=self.generate_doctor_codes_dialog).pack(fill="x", pady=2)
        ttk.Button(code_frame, text="Show Unused Codes", command=self.show_unused_doctor_codes).pack(fill="x", pady=2)
        
    def setup_data_panel(self):
        """Setup data display panel"""
        data_frame = ttk.LabelFrame(self.main_frame, text="Data Management", padding="10")
        data_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(1, weight=1)
        
        # Tabs
        self.notebook = ttk.Notebook(data_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Users tab
        self.setup_users_tab()
        
        # Cases tab
        self.setup_cases_tab()
        
    def setup_users_tab(self):
        """Setup users tab"""
        users_frame = ttk.Frame(self.notebook)
        self.notebook.add(users_frame, text="User Management")
        
        # Users table
        columns = ("ID", "Username", "Role", "Admin", "Last Login")
        self.users_tree = ttk.Treeview(users_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        for col in columns:
            self.users_tree.heading(col, text=col)
            self.users_tree.column(col, width=120)
        
        # Scrollbar
        users_scrollbar = ttk.Scrollbar(users_frame, orient="vertical", command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=users_scrollbar.set)
        
        # Layout
        self.users_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        users_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        users_frame.columnconfigure(0, weight=1)
        users_frame.rowconfigure(0, weight=1)
        
        # User operation buttons
        user_btn_frame = ttk.Frame(users_frame)
        user_btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(user_btn_frame, text="Set as Doctor", command=lambda: self.change_role("doctor")).pack(side="left", padx=5)
        ttk.Button(user_btn_frame, text="Set as Patient", command=lambda: self.change_role("patient")).pack(side="left", padx=5)
        ttk.Button(user_btn_frame, text="Toggle Admin", command=self.set_admin).pack(side="left", padx=5)
        ttk.Button(user_btn_frame, text="Reset Password", command=self.reset_password).pack(side="left", padx=5)
        
        # Double click to edit
        self.users_tree.bind("<Double-1>", self.edit_user)
        
    def setup_cases_tab(self):
        """Setup cases tab"""
        cases_frame = ttk.Frame(self.notebook)
        self.notebook.add(cases_frame, text="Case Management")
        
        # Cases table
        columns = ("ID", "User ID", "Analysis Type", "Images", "Reviewed", "Approved")
        self.cases_tree = ttk.Treeview(cases_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.cases_tree.heading(col, text=col)
            if col == "ID":
                self.cases_tree.column(col, width=50, anchor="center")
            elif col == "用户ID":
                self.cases_tree.column(col, width=80, anchor="center")
            elif col == "分析类型":
                self.cases_tree.column(col, width=120)
            elif col == "图片数":
                self.cases_tree.column(col, width=60, anchor="center")
            elif col == "已审核":
                self.cases_tree.column(col, width=80, anchor="center")
            elif col == "已批准":
                self.cases_tree.column(col, width=80, anchor="center")
            else:
                self.cases_tree.column(col, width=100)
        
        # Scrollbar
        cases_scrollbar = ttk.Scrollbar(cases_frame, orient="vertical", command=self.cases_tree.yview)
        self.cases_tree.configure(yscrollcommand=cases_scrollbar.set)
        
        # Layout
        self.cases_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cases_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        cases_frame.columnconfigure(0, weight=1)
        cases_frame.rowconfigure(0, weight=1)
        
        # Case operation buttons
        case_btn_frame = ttk.Frame(cases_frame)
        case_btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(case_btn_frame, text="View Images", command=self.view_case_images).pack(side="left", padx=5)
        ttk.Button(case_btn_frame, text="Export Images", command=self.export_case_images).pack(side="left", padx=5)
        
        # Double click to view images
        self.cases_tree.bind("<Double-1>", self.view_case_images)
        
    def refresh_data(self):
        """Refresh all data"""
        self.update_db_info()
        self.load_users()
        self.load_cases()
        
    def force_refresh(self):
        """Force refresh all data, including reloading admin config"""
        print("🔄 Force refreshing data and admin configuration...")
        self.refresh_data()
        
    def update_db_info(self):
        """更新数据库信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取基本信息
            size = os.path.getsize(self.db_path)
            size_mb = size / (1024 * 1024)
            
            # 获取用户统计
            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT role, COUNT(*) FROM user GROUP BY role")
            role_stats = cursor.fetchall()
            
            # 获取病例统计
            cursor.execute("SELECT COUNT(*) FROM 'case'")
            case_count = cursor.fetchone()[0]
            
            info_text = f"Database Path: {self.db_path}\n"
            info_text += f"File Size: {size_mb:.2f} MB\n"
            info_text += f"Total Users: {user_count}\n"
            
            for role, count in role_stats:
                info_text += f"  {role}: {count}\n"
                
            info_text += f"Total Cases: {case_count}\n"
            info_text += f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"
            
            self.db_info.delete(1.0, tk.END)
            self.db_info.insert(1.0, info_text)
            
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update database information: {str(e)}")
            
    def load_users(self):
        """加载用户数据"""
        try:
            # Clear existing data
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, username, role, is_admin, last_login FROM user ORDER BY id")
            users = cursor.fetchall()
            
            for user in users:
                user_id, username, role, is_admin, last_login = user
                last_login_str = last_login if last_login else "Never logged in"
                is_admin_str = "Yes" if is_admin else "No"
                
                self.users_tree.insert("", "end", values=(user_id, username, role, is_admin_str, last_login_str))
                
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load user data: {str(e)}")
            
    def load_cases(self):
        """Load case data"""
        try:
            # Clear existing data
            for item in self.cases_tree.get_children():
                self.cases_tree.delete(item)
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, user_id, analysis_type, doctor_reviewed, doctor_approved,
                       pre_image, post_image, pre_nostril_image, post_nostril_image,
                       pre_columellar_image, post_columellar_image
                FROM 'case' ORDER BY id
            """)
            cases = cursor.fetchall()
            
            for case in cases:
                case_id, user_id, analysis_type, reviewed, approved = case[:5]
                images = case[5:]  # 所有图片字段
                
                # 计算图片数量
                image_count = sum(1 for img in images if img is not None and img.strip())
                
                analysis_type = analysis_type or "Unspecified"
                reviewed = "Yes" if reviewed else "No"
                approved = "Yes" if approved else ("No" if approved is False else "Not Set")
                
                self.cases_tree.insert("", "end", values=(case_id, user_id, analysis_type, image_count, reviewed, approved))
                
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load case data: {str(e)}")
            
    def change_role(self, new_role):
        """Change user role"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        username = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to change user {username}'s role to {new_role}?"):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("UPDATE user SET role = ? WHERE id = ?", (new_role, user_id))
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"User {username}'s role has been changed to {new_role}")
                self.refresh_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to change role: {str(e)}")
                
    def add_user(self):
        """Add user"""
        dialog = UserDialog(self.root, "Add User")
        if dialog.result:
            username, password, role = dialog.result
            
            try:
                from werkzeug.security import generate_password_hash
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if username already exists
                cursor.execute("SELECT id FROM user WHERE username = ?", (username,))
                if cursor.fetchone():
                    messagebox.showerror("Error", "Username already exists")
                    return
                
                password_hash = generate_password_hash(password, method='pbkdf2:sha256')
                cursor.execute("INSERT INTO user (username, password_hash, role) VALUES (?, ?, ?)",
                             (username, password_hash, role))
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"User {username} added successfully")
                self.refresh_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add user: {str(e)}")
                
    def delete_user(self):
        """Delete user"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        username = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete user {username}? This will also delete all of the user's case data!"):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete user's cases
                cursor.execute("DELETE FROM 'case' WHERE user_id = ?", (user_id,))
                # Delete user profile
                cursor.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
                # Delete user
                cursor.execute("DELETE FROM user WHERE id = ?", (user_id,))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"User {username} has been deleted")
                self.refresh_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete user: {str(e)}")
                
    def reset_password(self):
        """Reset password"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        username = item['values'][1]
        
        new_password = simpledialog.askstring("Reset Password", f"Enter new password for user {username}:")
        if new_password:
            try:
                from werkzeug.security import generate_password_hash
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
                cursor.execute("UPDATE user SET password_hash = ? WHERE id = ?", (password_hash, user_id))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Password for user {username} has been reset")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset password: {str(e)}")
                
    def edit_user(self, event):
        """Double click to edit user"""
        selection = self.users_tree.selection()
        if selection:
            item = self.users_tree.item(selection[0])
            user_id = item['values'][0]
            username = item['values'][1]
            role = item['values'][2]
            
            dialog = UserDialog(self.root, "Edit User", username, "", role, edit_mode=True)
            if dialog.result:
                new_username, new_password, new_role = dialog.result
                
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    if new_password:
                        from werkzeug.security import generate_password_hash
                        password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
                        cursor.execute("UPDATE user SET username = ?, password_hash = ?, role = ? WHERE id = ?",
                                     (new_username, password_hash, new_role, user_id))
                    else:
                        cursor.execute("UPDATE user SET username = ?, role = ? WHERE id = ?",
                                     (new_username, new_role, user_id))
                    
                    conn.commit()
                    conn.close()
                    
                    messagebox.showinfo("Success", "User information has been updated")
                    self.refresh_data()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to update user: {str(e)}")
                    
    def backup_database(self):
        """Backup database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")],
            initialname=f"backup_database_{timestamp}.db"
        )
        
        if backup_path:
            try:
                shutil.copy2(self.db_path, backup_path)
                size = os.path.getsize(backup_path) / (1024 * 1024)
                messagebox.showinfo("Success", f"Database backup successful!\nFile size: {size:.2f} MB")
            except Exception as e:
                messagebox.showerror("Error", f"Backup failed: {str(e)}")
                
    def export_data(self):
        """Export data"""
        export_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialname=f"export_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if export_path:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                data = {}
                
                # Export users
                cursor.execute("SELECT * FROM user")
                users = cursor.fetchall()
                cursor.execute("PRAGMA table_info(user)")
                user_columns = [col[1] for col in cursor.fetchall()]
                data['users'] = [dict(zip(user_columns, user)) for user in users]
                
                # Export cases
                cursor.execute("SELECT * FROM 'case'")
                cases = cursor.fetchall()
                cursor.execute("PRAGMA table_info('case')")
                case_columns = [col[1] for col in cursor.fetchall()]
                data['cases'] = [dict(zip(case_columns, case)) for case in cases]
                
                conn.close()
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                    
                messagebox.showinfo("Success", f"Data export successful!\nExported {len(data['users'])} users and {len(data['cases'])} cases")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                
    def clean_data(self):
        """Clean data"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clean empty case data?"):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM 'case' WHERE pre_image IS NULL AND post_image IS NULL")
                deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Cleaned {deleted} empty cases")
                self.refresh_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Cleaning failed: {str(e)}")
                
    def sql_query(self):
        """SQL query"""
        query = simpledialog.askstring("SQL Query", "Enter SQL query statement:")
        if query:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    # Show results
                    result_window = tk.Toplevel(self.root)
                    result_window.title("Query Results")
                    result_window.geometry("800x400")
                    
                    result_text = tk.Text(result_window)
                    result_text.pack(fill="both", expand=True)
                    
                    result_text.insert("end", f"Columns: {', '.join(columns)}\n\n")
                    for row in results:
                        result_text.insert("end", f"{row}\n")
                        
                else:
                    conn.commit()
                    messagebox.showinfo("Success", f"SQL executed successfully, affected {cursor.rowcount} rows")
                    self.refresh_data()
                
                conn.close()
                
            except Exception as e:
                messagebox.showerror("Error", f"SQL execution failed: {str(e)}")
                
    def show_admin_status(self):
        """显示管理员状态"""
        admin_window = tk.Toplevel(self.root)
        admin_window.title("Admin Management")
        admin_window.geometry("500x350")
        
        frame = ttk.Frame(admin_window, padding="20")
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Current Admin List:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 创建表格显示管理员
        columns = ("ID", "Username", "Role", "Last Login")
        admin_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        
        # 设置列
        for col in columns:
            admin_tree.heading(col, text=col)
            admin_tree.column(col, width=100)
        
        admin_tree.pack(fill="both", expand=True, pady=10)
        
        # 加载管理员列表
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, username, role, last_login FROM user WHERE is_admin = 1")
            admins = cursor.fetchall()
            
            for admin in admins:
                admin_id, username, role, last_login = admin
                last_login_str = last_login if last_login else "Never logged in"
                admin_tree.insert("", "end", values=(admin_id, username, role, last_login_str))
                
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load admin list: {str(e)}")
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Refresh", 
                  command=lambda: self.refresh_admin_tree(admin_tree)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", 
                  command=admin_window.destroy).pack(side="right", padx=5)
    
    def refresh_admin_tree(self, tree):
        """刷新管理员列表"""
        # 清空现有数据
        for item in tree.get_children():
            tree.delete(item)
            
        # 加载管理员列表
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, username, role, last_login FROM user WHERE is_admin = 1")
            admins = cursor.fetchall()
            
            for admin in admins:
                admin_id, username, role, last_login = admin
                last_login_str = last_login if last_login else "Never logged in"
                tree.insert("", "end", values=(admin_id, username, role, last_login_str))
                
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh admin list: {str(e)}")
            
    def view_case_images(self, event=None):
        """View case images"""
        selection = self.cases_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a case first")
            return
            
        item = self.cases_tree.item(selection[0])
        case_id = item['values'][0]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pre_image, post_image, pre_nostril_image, post_nostril_image,
                       pre_columellar_image, post_columellar_image, analysis_type
                FROM 'case' WHERE id = ?
            """, (case_id,))
            
            result = cursor.fetchone()
            if not result:
                messagebox.showerror("Error", "Case data not found")
                return
                
            images = result[:6]
            analysis_type = result[6] or "Unspecified"
            image_names = ["Pre-op Main", "Post-op Main", "Pre-op Nostril", "Post-op Nostril", "Pre-op Columellar", "Post-op Columellar"]
            
            # Create image viewer window
            self.show_image_viewer(case_id, images, image_names, analysis_type)
            
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view images: {str(e)}")
            
    def show_image_viewer(self, case_id, images, image_names, analysis_type):
        """Show image viewer window"""
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title(f"Case #{case_id} Image Viewer - {analysis_type}")
        viewer_window.geometry("1000x700")
        
        # Create notebook widget for tabs
        notebook = ttk.Notebook(viewer_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tab for each image
        for i, (image_data, image_name) in enumerate(zip(images, image_names)):
            if image_data and image_data.strip():
                frame = ttk.Frame(notebook)
                notebook.add(frame, text=image_name)
                
                # Image information
                info_frame = ttk.Frame(frame)
                info_frame.pack(fill="x", padx=10, pady=5)
                
                ttk.Label(info_frame, text=f"Case ID: {case_id} | Image: {image_name}",
                         font=("Arial", 12, "bold")).pack()
                
                # Image data information
                data_length = len(image_data)
                size_info = f"Data Size: {data_length:,} characters (~{data_length/1024:.1f} KB)"
                ttk.Label(info_frame, text=size_info).pack()
                
                # Image format information
                if image_data.startswith('/9j/'):
                    format_info = "Format: JPEG (Base64 encoded)"
                elif image_data.startswith('data:image'):
                    format_info = "Format: Complete Base64 data URL"
                else:
                    format_info = "Format: Unknown"
                ttk.Label(info_frame, text=format_info).pack()
                
                # Create main container, divided into left and right parts
                main_container = ttk.Frame(frame)
                main_container.pack(fill="both", expand=True, padx=10, pady=5)
                
                # Left side: actual image display
                left_frame = ttk.LabelFrame(main_container, text="Image Preview", padding="5")
                left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
                
                # Try to display actual image
                try:
                    image_widget = self.create_image_widget(left_frame, image_data)
                    if image_widget:
                        image_widget.pack(fill="both", expand=True)
                    else:
                        ttk.Label(left_frame, text="Cannot display image\n(Possibly corrupted Base64 data)", 
                                font=("Arial", 12)).pack(expand=True)
                except Exception as e:
                    ttk.Label(left_frame, text=f"Image display error:\n{str(e)}", 
                             font=("Arial", 10)).pack(expand=True)
                
                # Right side: Base64 data preview
                right_frame = ttk.LabelFrame(main_container, text="Base64 Data Preview", padding="5")
                right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
                
                text_widget = tk.Text(right_frame, height=15, wrap=tk.WORD, width=40)
                scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                
                # Display first 1000 characters of Base64 data
                preview_data = image_data[:1000] + "..." if len(image_data) > 1000 else image_data
                text_widget.insert("1.0", preview_data)
                text_widget.config(state="disabled")
                
                text_widget.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
                
                # Operation buttons
                btn_frame = ttk.Frame(frame)
                btn_frame.pack(fill="x", padx=10, pady=5)
                
                ttk.Button(btn_frame, text="Save Image", 
                          command=lambda img=image_data, name=image_name, cid=case_id: 
                          self.save_image(img, name, cid)).pack(side="left", padx=5)
                          
                ttk.Button(btn_frame, text="Copy Base64", 
                          command=lambda img=image_data: self.copy_to_clipboard(img)).pack(side="left", padx=5)
                          
                # Add image info button
                ttk.Button(btn_frame, text="Image Info", 
                          command=lambda img=image_data: self.show_image_info(img)).pack(side="left", padx=5)
        
        # If no images
        if not any(img and img.strip() for img in images):
            no_image_frame = ttk.Frame(notebook)
            notebook.add(no_image_frame, text="No Images")
            ttk.Label(no_image_frame, text="This case has no saved image data", 
                     font=("Arial", 14)).pack(expand=True)
                     
    def save_image(self, image_data, image_name, case_id):
        """Save image to file"""
        try:
            import base64
            from tkinter import filedialog
            
            # Process Base64 data
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            # Decode Base64
            image_bytes = base64.b64decode(image_data)
            
            # Choose save location
            filename = f"case_{case_id}_{image_name}.jpg"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG Images", "*.jpg"), ("All Files", "*.*")],
                initialname=filename
            )
            
            if file_path:
                with open(file_path, 'wb') as f:
                    f.write(image_bytes)
                messagebox.showinfo("Success", f"Image saved to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            
    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Success", "Base64 data copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed: {str(e)}")
            
    def create_image_widget(self, parent, image_data):
        """Create image display widget"""
        try:
            import base64
            from PIL import Image, ImageTk
            from io import BytesIO
            
            # Process Base64 data
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            # Decode Base64
            image_bytes = base64.b64decode(image_data)
            
            # Create PIL image
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Calculate appropriate display size (maintain aspect ratio)
            max_width, max_height = 400, 300
            original_width, original_height = pil_image.size
            
            # Calculate scale ratio
            scale_w = max_width / original_width
            scale_h = max_height / original_height
            scale = min(scale_w, scale_h)
            
            if scale < 1:  # Only scale if image is too large
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to Tkinter-compatible format
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Create Label to display image
            image_label = tk.Label(parent, image=tk_image)
            image_label.image = tk_image  # Keep reference to prevent garbage collection
            
            # Add image size information
            size_label = ttk.Label(parent, text=f"Original Size: {original_width}x{original_height}")
            size_label.pack(pady=5)
            
            return image_label
            
        except ImportError:
            # If PIL is not available, show hint
            error_label = ttk.Label(parent, text="Pillow library required to display images\nRun: pip install Pillow")
            return error_label
        except Exception as e:
            # Other errors
            error_label = ttk.Label(parent, text=f"Image display error:\n{str(e)}")
            return error_label
            
    def show_image_info(self, image_data):
        """Show detailed image information"""
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Process Base64 data
            original_data = image_data
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode Base64
            image_bytes = base64.b64decode(image_data)
            
            # Create PIL image to get information
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Collect image information
            info_text = f"Image Format: {pil_image.format}\n"
            info_text += f"Image Mode: {pil_image.mode}\n"
            info_text += f"Image Size: {pil_image.size[0]} x {pil_image.size[1]} pixels\n"
            info_text += f"Base64 Length: {len(original_data):,} characters\n"
            info_text += f"Original Byte Size: {len(image_bytes):,} bytes\n"
            info_text += f"Estimated File Size: {len(image_bytes) / 1024:.2f} KB\n"
            
            # If EXIF information exists
            if hasattr(pil_image, '_getexif'):
                exif = pil_image._getexif()
                if exif:
                    info_text += f"EXIF Data: Yes\n"
                else:
                    info_text += f"EXIF Data: None\n"
            
            messagebox.showinfo("Image Information", info_text)
            
        except ImportError:
            messagebox.showwarning("Warning", "Pillow library required to get image information\nRun: pip install Pillow")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get image information: {str(e)}")
            
    def export_case_images(self):
        """Export all images from selected case"""
        selection = self.cases_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a case first")
            return
            
        item = self.cases_tree.item(selection[0])
        case_id = item['values'][0]
        
        # Choose export folder
        from tkinter import filedialog
        folder_path = filedialog.askdirectory(title="Choose Export Folder")
        if not folder_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pre_image, post_image, pre_nostril_image, post_nostril_image,
                       pre_columellar_image, post_columellar_image
                FROM 'case' WHERE id = ?
            """, (case_id,))
            
            result = cursor.fetchone()
            if not result:
                messagebox.showerror("Error", "Case data not found")
                return
                
            images = result
            image_names = ["pre_main", "post_main", "pre_nostril", "post_nostril", "pre_columellar", "post_columellar"]
            
            export_count = 0
            import base64
            
            for image_data, image_name in zip(images, image_names):
                if image_data and image_data.strip():
                    try:
                        # Process Base64 data
                        if image_data.startswith('data:image'):
                            image_data = image_data.split(',')[1]
                        
                        # Decode and save
                        image_bytes = base64.b64decode(image_data)
                        file_path = os.path.join(folder_path, f"case_{case_id}_{image_name}.jpg")
                        
                        with open(file_path, 'wb') as f:
                            f.write(image_bytes)
                        export_count += 1
                        
                    except Exception as e:
                        print(f"Failed to export image {image_name}: {str(e)}")
            
            conn.close()
            messagebox.showinfo("Success", f"Successfully exported {export_count} images to: {folder_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export images: {str(e)}")

    def set_admin(self):
        """设置超级管理员"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        username = item['values'][1]
        is_admin = item['values'][3] == "Yes"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if is_admin:
                # 移除管理员权限
                if messagebox.askyesno("Confirm", f"Are you sure you want to remove admin privileges from {username}?"):
                    cursor.execute("UPDATE user SET is_admin = 0 WHERE id = ?", (user_id,))
                    conn.commit()
                    messagebox.showinfo("Success", f"Admin privileges removed from {username}")
            else:
                # 添加管理员权限
                if messagebox.askyesno("Confirm", f"Are you sure you want to set {username} as admin?\nAdmins can access the user management page."):
                    cursor.execute("UPDATE user SET is_admin = 1 WHERE id = ?", (user_id,))
                    conn.commit()
                    messagebox.showinfo("Success", f"{username} has been set as admin")
            
            conn.close()
            self.load_users()  # 刷新用户列表
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update admin status: {str(e)}")

    def generate_doctor_codes_dialog(self):
        """弹窗输入数量并生成医生码"""
        num = simpledialog.askinteger("Generate Doctor Codes", "How many doctor codes to generate?", minvalue=1, maxvalue=100)
        if num:
            codes = self.generate_doctor_codes(num)
            self.add_doctor_codes_to_db(codes)
            self.show_unused_doctor_codes()

    def generate_doctor_codes(self, n=20, length=12):
        codes = []
        for _ in range(n):
            code = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            codes.append(code)
        return codes

    def add_doctor_codes_to_db(self, codes):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for code in codes:
            try:
                cursor.execute("INSERT INTO doctor_codes (code, is_used) VALUES (?, 0)", (code,))
            except sqlite3.IntegrityError:
                continue  # 跳过重复
        conn.commit()
        conn.close()

    def show_unused_doctor_codes(self):
        import tkinter as tk
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT code FROM doctor_codes WHERE is_used=0")
        codes = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not codes:
            messagebox.showinfo("Unused Doctor Codes", "No unused doctor codes found.")
            return

        win = tk.Toplevel(self.root)
        win.title("Unused Doctor Codes")
        win.geometry("350x400")
        win.grab_set()

        label = tk.Label(win, text=f"Unused codes (total {len(codes)}):")
        label.pack(pady=5)

        listbox = tk.Listbox(win, selectmode=tk.SINGLE, font=("Consolas", 12))
        for code in codes:
            listbox.insert(tk.END, code)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def copy_selected():
            selection = listbox.curselection()
            if selection:
                code = listbox.get(selection[0])
                self.root.clipboard_clear()
                self.root.clipboard_append(code)
                messagebox.showinfo("Copied", f"Copied:\n{code}")
            else:
                messagebox.showwarning("No selection", "Please select a code to copy.")

        copy_btn = tk.Button(win, text="Copy Selected Code", command=copy_selected)
        copy_btn.pack(pady=10)

    def run(self):
        """运行应用"""
        self.root.mainloop()

class UserDialog:
    def __init__(self, parent, title, username="", password="", role="patient", edit_mode=False):
        self.result = None
        self.edit_mode = edit_mode
        
        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x200")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.transient(parent)
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # 创建表单
        self.create_form(username, password, role)
        
        # 等待对话框关闭
        self.dialog.wait_window()
        
    def create_form(self, username, password, role):
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill="both", expand=True)
        
        # Username
        ttk.Label(frame, text="Username:").grid(row=0, column=0, sticky="w", pady=5)
        self.username_var = tk.StringVar(value=username)
        self.username_entry = ttk.Entry(frame, textvariable=self.username_var, width=20)
        self.username_entry.grid(row=0, column=1, pady=5)
        
        # Password
        password_label = "New Password:" if self.edit_mode else "Password:"
        ttk.Label(frame, text=password_label).grid(row=1, column=0, sticky="w", pady=5)
        self.password_var = tk.StringVar(value=password)
        self.password_entry = ttk.Entry(frame, textvariable=self.password_var, width=20, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)
        
        if self.edit_mode:
            ttk.Label(frame, text="(Leave empty to keep unchanged)", font=("Arial", 8)).grid(row=2, column=1, sticky="w")
        
        # Role
        ttk.Label(frame, text="Role:").grid(row=3, column=0, sticky="w", pady=5)
        self.role_var = tk.StringVar(value=role)
        role_combo = ttk.Combobox(frame, textvariable=self.role_var, values=["patient", "doctor"], width=17)
        role_combo.grid(row=3, column=1, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel_clicked).pack(side="left", padx=5)
        
    def ok_clicked(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        role = self.role_var.get()
        
        if not username:
            messagebox.showerror("Error", "Please enter username")
            return
            
        if not self.edit_mode and not password:
            messagebox.showerror("Error", "Please enter password")
            return
            
        if role not in ["patient", "doctor"]:
            messagebox.showerror("Error", "Please select a valid role")
            return
            
        self.result = (username, password, role)
        self.dialog.destroy()
        
    def cancel_clicked(self):
        self.dialog.destroy()

if __name__ == "__main__":
    try:
        app = DatabaseGUIManager()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Startup failed: {str(e)}") 