import os
import sys

# 将services目录添加到系统PATH，以便找到DLL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES_DIR = os.path.join(BASE_DIR, 'services')
if SERVICES_DIR not in os.environ['PATH']:
    os.environ['PATH'] = SERVICES_DIR + os.pathsep + os.environ['PATH']
    
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, SelectField
from wtforms.validators import InputRequired, Length, EqualTo, Optional
from flask_wtf.file import FileField, FileAllowed
from services.ml_interface import process_image
from werkzeug.utils import secure_filename
import zipfile
from io import BytesIO
import io
import base64

from flask import Flask, render_template, Response
import cv2
from flask_migrate import Migrate
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_video():
    """
    视频流生成器，实时处理每帧图像，检测唇裂关键点
    注意：实时视频处理暂时停用，请使用图像上传功能
    """
    # 暂时返回空白帧，避免摄像头功能出错
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Real-time video disabled", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blank_frame, "Please use image upload", (50, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', blank_frame)
    frame_data = buffer.tobytes()
    
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


# --- Flask Setup ---
app.config['SECRET_KEY'] = 'replace-with-a-strong-secret-key'
# 指向flask_cleft_demo目录下的instance文件夹
REAL_DB_PATH = os.path.join(BASE_DIR, 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{REAL_DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')

# 强制重新创建数据库连接以确保使用最新的数据库结构
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'connect_args': {'check_same_thread': False}
}

db = SQLAlchemy(app)

migrate = Migrate(app, db)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'patient' or 'doctor'
    last_login = db.Column(db.DateTime)  # 添加最后登录时间字段
    is_admin = db.Column(db.Boolean, default=False)  # 添加管理员标志字段
    cases = db.relationship('Case', backref='user', lazy=True)
    profile = db.relationship('UserProfile', uselist=False, back_populates='user')

class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Alar Facial Symmetry Model Images (原有的主要图片)
    pre_image = db.Column(db.Text, nullable=True)  # Alar model processed image
    post_image = db.Column(db.Text, nullable=True)  # Alar model processed image
    pre_severity = db.Column(db.String(50), nullable=True)
    post_severity = db.Column(db.String(50), nullable=True)
    
    # Alar Facial Symmetry (A/B ratio)
    pre_ratio = db.Column(db.Float)
    post_ratio = db.Column(db.Float)
    
    # Nostril Width Ratio Model Images
    pre_nostril_image = db.Column(db.Text, nullable=True)  # Nostril model processed image
    post_nostril_image = db.Column(db.Text, nullable=True)  # Nostril model processed image
    pre_nostril_ratio = db.Column(db.Float, nullable=True)
    post_nostril_ratio = db.Column(db.Float, nullable=True)
    pre_nostril_severity = db.Column(db.String(50), nullable=True)
    post_nostril_severity = db.Column(db.String(50), nullable=True)
    
    # Columellar Angle Model Images
    pre_columellar_image = db.Column(db.Text, nullable=True)  # Columellar model processed image
    post_columellar_image = db.Column(db.Text, nullable=True)  # Columellar model processed image
    pre_columellar_angle = db.Column(db.Float, nullable=True)
    post_columellar_angle = db.Column(db.Float, nullable=True)
    pre_columellar_severity = db.Column(db.String(50), nullable=True)
    post_columellar_severity = db.Column(db.String(50), nullable=True)
    
    # Analysis type to track which models were used
    analysis_type = db.Column(db.String(50), nullable=True)  # 'alar', 'nostril', 'columellar', 'comprehensive'
    
    doctor_reviewed = db.Column(db.Boolean, default=False)
    doctor_approved = db.Column(db.Boolean, default=None)
    # 移除重复的关系定义，使用User模型中的backref='user'

class UserProfile(db.Model):
    __tablename__ = 'user_profile'  # 明确指定表名
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    contact = db.Column(db.String(100))
    user = db.relationship('User', back_populates='profile')

# --- DoctorCode Model ---
class DoctorCode(db.Model):
    __tablename__ = 'doctor_codes'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(32), unique=True, nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    used_by = db.Column(db.String(100))
    used_at = db.Column(db.DateTime)

# --- WTForms ---
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=25)])
    confirm = PasswordField('Confirm Password', validators=[EqualTo('password', message='Passwords must match')])
    doctor_code = StringField('Doctor Code (for Doctors)', validators=[Length(max=32)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=25)])
    submit = SubmitField('Login')

class ProfileForm(FlaskForm):
    age = IntegerField('Age', validators=[Optional()])
    gender = SelectField('Gender', 
                        choices=[('', 'Select Gender'), 
                                ('Male', 'Male'), 
                                ('Female', 'Female'), 
                                ('Other', 'Other')],
                        validators=[Optional()])
    contact = StringField('Contact', validators=[Optional()])
    submit = SubmitField('Update Profile')

# --- Utility Functions ---
def calculate_columellar_severity(angle):
    """计算鼻柱角度严重程度"""
    if angle is None:
        return None
    
    if angle <= 15:
        return 'Mild'
    elif angle <= 30:
        return 'Moderate'
    else:
        return 'Severe'

def init_db():
    """Create the database tables, ensuring all new columns exist."""
    with app.app_context():
        # 确保 instance 文件夹存在
        instance_dir = os.path.join(BASE_DIR, 'instance')
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        
        # 强制创建所有表（包括新列）
        print("🔧 正在初始化数据库...")
        db.create_all()
        
        # 验证Case表是否包含所有必需的列
        try:
            inspector = db.inspect(db.engine)
            if 'case' in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns('case')]
                required_columns = [
                    'pre_nostril_image', 'post_nostril_image',
                    'pre_nostril_ratio', 'post_nostril_ratio',
                    'pre_nostril_severity', 'post_nostril_severity',
                    'pre_columellar_image', 'post_columellar_image',
                    'pre_columellar_angle', 'post_columellar_angle',
                    'pre_columellar_severity', 'post_columellar_severity',
                    'analysis_type'
                ]
                
                missing_columns = [col for col in required_columns if col not in columns]
                if missing_columns:
                    print(f"❌ 数据库缺少列: {missing_columns}")
                    print("🔧 正在删除旧数据库并重新创建...")
                    db.drop_all()
                    db.create_all()
                    print("✅ 数据库重新创建完成")
                else:
                    print("✅ 数据库结构验证通过")
            else:
                print("✅ 数据库表创建完成")

            # 验证User表是否包含is_admin列
            if 'user' in inspector.get_table_names():
                user_columns = [col['name'] for col in inspector.get_columns('user')]
                if 'is_admin' not in user_columns:
                    print("⚠️  User table missing is_admin column, adding it directly...")
                    try:
                        # 直接使用SQLite的ALTER TABLE语句添加列
                        db.engine.execute('ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0')
                        print("✅ Successfully added is_admin column to user table")
                    except Exception as e:
                        print(f"⚠️  Error adding column: {str(e)}")
                        print("🔧 Recreating database with all columns...")
                        db.drop_all()
                        db.create_all()
                        print("✅ Database recreated with all required columns")
                
            # 检查是否需要创建初始管理员
            user_count = User.query.filter_by(is_admin=True).count()
            if user_count == 0:
                print("⚠️  No admin users found, creating default admin...")
                # 检查是否有现有用户可以升级为管理员
                existing_user = User.query.filter_by(username="admin").first()
                if existing_user:
                    existing_user.is_admin = True
                    db.session.commit()
                    print(f"✅ Set user {existing_user.username} as admin")
                else:
                    # 创建一个新的管理员用户
                    admin_password = generate_password_hash("admin", method='pbkdf2:sha256')
                    new_admin = User(
                        username="admin", 
                        password_hash=admin_password, 
                        role="doctor", 
                        is_admin=True
                    )
                    db.session.add(new_admin)
                    db.session.commit()
                    print("✅ Created default admin account (username: admin, password: admin)")
                    print("⚠️  Remember to change the default password after login!")
                
        except Exception as e:
            print(f"⚠️  数据库验证失败: {str(e)}")
            print("🔧 强制重新创建数据库...")
            try:
                db.drop_all()
                db.create_all()
                print("✅ 数据库强制重建完成")
            except Exception as rebuild_error:
                print(f"❌ 数据库重建失败: {str(rebuild_error)}")
        
        print("📊 数据库初始化完成")

# --- Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        if session.get('role') == 'doctor':
            return redirect(url_for('view_all_cases'))
        else:
            return redirect(url_for('view_my_cases'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash("Username already exists. Please choose a different username.")
            return redirect(url_for('register'))

        role = 'patient'
        doctor_code_input = form.doctor_code.data.strip() if form.doctor_code.data else ''
        is_admin = False
        if doctor_code_input:
            # 校验 doctor_codes 表
            code_row = DoctorCode.query.filter_by(code=doctor_code_input, is_used=False).first()
            if code_row:
                role = 'doctor'
            else:
                flash("Invalid or used doctor code.")
                return redirect(url_for('register'))

        # 检查是否从管理页面添加用户（包含is_admin参数）
        if request.form.get('is_admin'):
            if 'user_id' in session:
                current_user = User.query.get(session['user_id'])
                if current_user and current_user.is_admin:
                    is_admin = True

        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(
            username=form.username.data, 
            password_hash=hashed_password, 
            role=role,
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()

        # 注册成功后，标记医生码为已用
        if role == 'doctor' and doctor_code_input:
            code_row.is_used = True
            code_row.used_by = new_user.username
            from datetime import datetime
            code_row.used_at = datetime.now()
            db.session.commit()

        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['role'] = user.role
            session['username'] = user.username
            session['is_admin'] = user.is_admin  # 添加管理员状态到session
            # 更新最后登录时间
            user.last_login = datetime.now()
            db.session.commit()
            flash('Login Success!')
            return redirect(url_for('home'))
        else:
            flash('Username or Password Error')
    
    return render_template('login.html', form=form)  # 传递表单到模板


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash("Logged out successfully.")
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Single-page upload for pre-op and post-op images.
    Displays processed images, severity, and ratio after upload.
    """
    if 'user_id' not in session:
        flash("Please log in to access the upload page.")
        return redirect(url_for('login'))

    pre_result = None
    post_result = None

    if request.method == 'POST':
        # Process pre-op image
        if 'pre_op_image' in request.files and request.files['pre_op_image'].filename:
            pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines = process_image(request.files['pre_op_image'])
            pre_result = (pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines)

        # Process post-op image
        if 'post_op_image' in request.files and request.files['post_op_image'].filename:
            post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines = process_image(request.files['post_op_image'])
            post_result = (post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines)



    return render_template('upload.html', pre_result=pre_result, post_result=post_result)

@app.route('/video_feed')
def video_feed():
    """
    返回视频流响应，用于显示实时视频
    """
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    """
    渲染摄像头页面
    """
    return render_template('camera.html')

@app.route('/face-detection', methods=['POST'])
def face_detection():
    """
    接收前端的 Base64 图像并进行唇裂关键点检测
    """
    try:
        from services.ml_interface import get_detector
        
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # 解码 Base64 图像
        image_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 使用HRNet检测器
        detector_instance = get_detector()
        if detector_instance.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # 处理图像
        base64_str, ratio, severity, keypoints = detector_instance._process_image_array(img_rgb)

        return jsonify({
            'success': True,
            'keypoints_detected': len(keypoints),
            'ratio': f"{ratio:.6f}",
            'severity': severity,
            'image': base64_str,
            'keypoints': keypoints
        })
    except Exception as e:
        print(f"Error in cleft lip detection: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/cases', methods=['GET'])
def view_my_cases():
    """
    Route for patients to view their own cases.
    """
    if 'user_id' not in session:
        flash("Please log in to view your cases.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    cases = Case.query.filter_by(user_id=user_id).all()  # Retrieve cases for the logged-in user
    return render_template('cases.html', cases=cases)


@app.route('/cases/add', methods=['GET', 'POST'])
def add_case():
    if 'user_id' not in session:
        flash("Please log in to add a case.")
        return redirect(url_for('login'))

    if request.method == 'POST':
        # 检查是否是JSON请求（新的综合分析提交）
        if request.is_json:
            return handle_comprehensive_case_submission()
        
        # 原有的表单提交逻辑（保持向后兼容）
        pre_image = request.files.get('pre_op_image')
        post_image = request.files.get('post_op_image')

        # 检查是否至少上传了一张图片
        if not (pre_image and pre_image.filename) and not (post_image and post_image.filename):
            flash("Please upload at least one image to add a case.")
            return redirect(url_for('add_case'))

        pre_result = None
        post_result = None

        if pre_image and pre_image.filename:
            pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines = process_image(pre_image)
            pre_result = (pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines)

        if post_image and post_image.filename:
            post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines = process_image(post_image)
            post_result = (post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines)

        new_case = Case(
            user_id=session['user_id'],
            pre_image=pre_result[0] if pre_result else None,
            post_image=post_result[0] if post_result else None,
            pre_severity=pre_result[2] if pre_result else None,
            post_severity=post_result[2] if post_result else None,
            pre_ratio=pre_result[1] if pre_result else None,
            post_ratio=post_result[1] if post_result else None,
        )
        db.session.add(new_case)
        db.session.commit()

        flash("Case added successfully!")
        return redirect(url_for('view_my_cases'))
        
    return render_template('add_case.html')

def handle_comprehensive_case_submission():
    """处理病人的综合分析病例提交"""
    try:
        data = request.get_json()
        
        # 使用alar图片作为主要图片（因为这是最完整的分析）
        pre_image = data.get('pre_alar_image') or data.get('pre_nostril_image') or data.get('pre_columellar_image')
        post_image = data.get('post_alar_image') or data.get('post_nostril_image') or data.get('post_columellar_image')
        
        # 创建综合分析严重程度描述
        pre_severity_parts = []
        post_severity_parts = []
        
        if data.get('pre_alar_severity'):
            pre_severity_parts.append(f"Alar({data.get('pre_alar_severity')})")
        if data.get('pre_nostril_severity'):
            pre_severity_parts.append(f"Nostril({data.get('pre_nostril_severity')})")
        if data.get('pre_columellar_angle') is not None:
            pre_severity_parts.append(f"Angle({data.get('pre_columellar_angle')}°)")
            
        if data.get('post_alar_severity'):
            post_severity_parts.append(f"Alar({data.get('post_alar_severity')})")
        if data.get('post_nostril_severity'):
            post_severity_parts.append(f"Nostril({data.get('post_nostril_severity')})")
        if data.get('post_columellar_angle') is not None:
            post_severity_parts.append(f"Angle({data.get('post_columellar_angle')}°)")
        
        pre_severity = f"Comprehensive: {', '.join(pre_severity_parts)}" if pre_severity_parts else None
        post_severity = f"Comprehensive: {', '.join(post_severity_parts)}" if post_severity_parts else None
        
        # 使用alar ratio作为主要ratio
        pre_ratio = data.get('pre_alar_ratio')
        post_ratio = data.get('post_alar_ratio')
        
        # 创建新病例
        new_case = Case(
            user_id=session['user_id'],
            pre_image=pre_image,
            post_image=post_image,
            pre_severity=pre_severity,
            post_severity=post_severity,
            pre_ratio=pre_ratio,
            post_ratio=post_ratio,
            pre_nostril_image=data.get('pre_nostril_image'),
            post_nostril_image=data.get('post_nostril_image'),
            pre_nostril_ratio=data.get('pre_nostril_ratio'),
            post_nostril_ratio=data.get('post_nostril_ratio'),
            pre_nostril_severity=data.get('pre_nostril_severity'),
            post_nostril_severity=data.get('post_nostril_severity'),
            pre_columellar_image=data.get('pre_columellar_image'),
            post_columellar_image=data.get('post_columellar_image'),
            pre_columellar_angle=data.get('pre_columellar_angle'),
            post_columellar_angle=data.get('post_columellar_angle'),
            pre_columellar_severity=calculate_columellar_severity(data.get('pre_columellar_angle')),
            post_columellar_severity=calculate_columellar_severity(data.get('post_columellar_angle')),
            analysis_type='comprehensive',
            doctor_reviewed=False
        )
        
        db.session.add(new_case)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '综合分析病例保存成功'})
        
    except Exception as e:
        print(f"保存病人综合分析病例时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cases/add_for_patient', methods=['GET', 'POST'])
def add_case_for_patient():
    if 'user_id' not in session:
        flash("Please log in to add a case.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        flash("You do not have permission to access this page.")
        return redirect(url_for('home'))

    patients = User.query.filter_by(role='patient').all()

    if request.method == 'POST':
        selected_patient_id = request.form.get('patient_id')
        pre_image = request.files.get('pre_op_image')
        post_image = request.files.get('post_op_image')

        if not selected_patient_id:
            flash("Please select a patient to add a case.")
            return redirect(url_for('add_case_for_patient'))

        # 检查是否至少上传了一张图片
        if not (pre_image and pre_image.filename) and not (post_image and post_image.filename):
            flash("Please upload at least one image to add a case.")
            return redirect(url_for('add_case_for_patient'))

        pre_result = None
        post_result = None

        if pre_image and pre_image.filename:
            pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines = process_image(pre_image)
            pre_result = (pre_base64, pre_ratio, pre_severity, pre_A, pre_B, pre_ab_lines)

        if post_image and post_image.filename:
            post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines = process_image(post_image)
            post_result = (post_base64, post_ratio, post_severity, post_A, post_B, post_ab_lines)

        new_case = Case(
            user_id=selected_patient_id,
            pre_image=pre_result[0] if pre_result else None,
            post_image=post_result[0] if post_result else None,
            pre_severity=pre_result[2] if pre_result else None,
            post_severity=post_result[2] if post_result else None,
            pre_ratio=pre_result[1] if pre_result else None,
            post_ratio=post_result[1] if post_result else None,
        )
        db.session.add(new_case)
        db.session.commit()

        flash("Case added successfully for the selected patient!")
        return redirect(url_for('view_all_cases'))

    return render_template('add_case_for_patient.html', patients=patients)

@app.route('/submit_case_for_patient', methods=['POST'])
def submit_case_for_patient():
    """处理add_case_for_patient页面的AJAX提交（支持综合多模型数据）"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '请先登录'})

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        return jsonify({'success': False, 'error': '只有医生可以为病人添加病例'})

    try:
        data = request.get_json()
        
        patient_id = data.get('patient_id')
        model_type = data.get('model_type')
        
        if not patient_id:
            return jsonify({'success': False, 'error': '请选择病人'})
        
        # 验证病人存在
        patient = User.query.get(patient_id)
        if not patient or patient.role != 'patient':
            return jsonify({'success': False, 'error': '无效的病人ID'})
        
        # 处理综合模型数据
        if model_type == 'comprehensive':
            # 综合分析 - 处理多模型数据
            pre_image = data.get('pre_image')  # 主图像
            post_image = data.get('post_image')  # 主图像
            
            # Alar模型数据
            pre_ratio = data.get('pre_alar_ratio')
            post_ratio = data.get('post_alar_ratio')
            pre_severity = data.get('pre_alar_severity')
            post_severity = data.get('post_alar_severity')
            
            # Nostril模型数据
            pre_nostril_ratio = data.get('pre_nostril_ratio')
            post_nostril_ratio = data.get('post_nostril_ratio')
            pre_nostril_severity = data.get('pre_nostril_severity')
            post_nostril_severity = data.get('post_nostril_severity')
            
            # Columellar模型数据
            pre_columellar_angle = data.get('pre_columellar_angle')
            post_columellar_angle = data.get('post_columellar_angle')
            # 只有当角度不为None时才计算严重程度
            pre_columellar_severity = calculate_columellar_severity(pre_columellar_angle) if pre_columellar_angle is not None else None
            post_columellar_severity = calculate_columellar_severity(post_columellar_angle) if post_columellar_angle is not None else None
            
        else:
            # 单一模型数据处理（保留原有逻辑）
            pre_image = data.get('pre_image')
            post_image = data.get('post_image')
            
            # 初始化所有字段
            pre_severity = None
            post_severity = None
            pre_ratio = None
            post_ratio = None
            pre_nostril_ratio = None
            post_nostril_ratio = None
            pre_nostril_severity = None
            post_nostril_severity = None
            pre_columellar_angle = None
            post_columellar_angle = None
            pre_columellar_severity = None
            post_columellar_severity = None
            
            if model_type == 'alar':
                # Alar模型
                pre_severity = data.get('pre_severity')
                post_severity = data.get('post_severity')
                pre_ratio = data.get('pre_ratio')
                post_ratio = data.get('post_ratio')
            elif model_type == 'nostril':
                # Nostril模型
                pre_nostril_ratio = data.get('pre_ratio')
                post_nostril_ratio = data.get('post_ratio')
                pre_nostril_severity = data.get('pre_severity')
                post_nostril_severity = data.get('post_severity')
            elif model_type == 'columellar':
                # 鼻柱角度模型
                pre_columellar_angle = data.get('pre_angle')
                post_columellar_angle = data.get('post_angle')
                # 只有当角度不为None时才计算严重程度
                pre_columellar_severity = calculate_columellar_severity(pre_columellar_angle) if pre_columellar_angle is not None else None
                post_columellar_severity = calculate_columellar_severity(post_columellar_angle) if post_columellar_angle is not None else None
        
        # 创建新病例
        new_case = Case(
            user_id=patient_id,
            pre_image=pre_image,
            post_image=post_image,
            pre_severity=pre_severity,
            post_severity=post_severity,
            pre_ratio=pre_ratio,
            post_ratio=post_ratio,
            pre_nostril_image=data.get('pre_nostril_image'),
            post_nostril_image=data.get('post_nostril_image'),
            pre_nostril_ratio=pre_nostril_ratio,
            post_nostril_ratio=post_nostril_ratio,
            pre_nostril_severity=pre_nostril_severity,
            post_nostril_severity=post_nostril_severity,
            pre_columellar_image=data.get('pre_columellar_image'),
            post_columellar_image=data.get('post_columellar_image'),
            pre_columellar_angle=pre_columellar_angle,
            post_columellar_angle=post_columellar_angle,
            pre_columellar_severity=pre_columellar_severity,
            post_columellar_severity=post_columellar_severity,
            analysis_type=model_type,
            doctor_reviewed=False
        )
        
        db.session.add(new_case)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '病例添加成功'})
        
    except Exception as e:
        print(f"提交病例时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/cases/all', methods=['GET'])
def view_all_cases():
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:  # Handle case where user does not exist
        flash("User not found. Please log in again.")
        return redirect(url_for('login'))

    if user.role != 'doctor':
        flash("Access denied. Only doctors can view all cases.")
        return redirect(url_for('home'))

    cases = Case.query.all()
    return render_template('all_cases.html', cases=cases)

@app.route('/cases/delete/<int:case_id>', methods=['POST'])
def delete_case(case_id):
    if 'user_id' not in session:
        flash("Please log in to delete cases.")
        return redirect(url_for('login'))

    case = Case.query.get_or_404(case_id)

    # 检查是否有权限删除病例
    user = User.query.get(session['user_id'])
    if user.role != 'doctor' and case.user_id != session['user_id']:
        flash("You do not have permission to delete this case.")
        return redirect(url_for('view_my_cases'))

    # 删除病例
    db.session.delete(case)
    db.session.commit()
    flash("Case deleted successfully.")
    if user.role == 'doctor':
        return redirect(url_for('view_all_cases'))
    else:
        return redirect(url_for('view_my_cases'))


@app.route('/cases/review/<int:case_id>', methods=['GET', 'POST'])
def review_case(case_id):
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user or user.role != 'doctor':
        flash("Access denied. Only doctors can review cases.")
        return redirect(url_for('home'))

    case = Case.query.get(case_id)
    if not case:
        flash("Case not found.")
        return redirect(url_for('view_all_cases'))

    if request.method == 'POST':
        if 'approve' in request.form:
            case.doctor_reviewed = True
            case.doctor_approved = True
            flash("Case approved successfully.")
        elif 'reject' in request.form:
            case.doctor_reviewed = True
            case.doctor_approved = False
            flash("Case rejected successfully.")
        db.session.commit()
        return redirect(url_for('view_all_cases'))

    return render_template('review_case.html', case=case)


@app.route('/statistics', methods=['GET'])
def statistics():
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        flash("Access denied.")
        return redirect(url_for('home'))

    # 只获取已评估的病例
    evaluated_cases = Case.query.filter(Case.doctor_reviewed == True).count()
    correct_cases = Case.query.filter(Case.doctor_reviewed == True, Case.doctor_approved == True).count()
    
    # 计算准确率
    accuracy = (correct_cases / evaluated_cases * 100) if evaluated_cases > 0 else 0

    return render_template('statistics.html', 
                         total_cases=evaluated_cases,  # 总数改为已评估病例数
                         correct_cases=correct_cases, 
                         accuracy=accuracy)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'pre_op_image' not in request.files and 'post_op_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    try:
        # 确定是pre还是post图片
        if 'pre_op_image' in request.files:
            file = request.files['pre_op_image']
            image_type = 'pre'
        else:
            file = request.files['post_op_image']
            image_type = 'post'
            
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # 获取模型类型
        model_type = request.form.get('model_type', 'alar')
        
        # 根据模型类型处理图像
        from services.ml_interface import process_image_with_model
        
        if model_type == 'alar':
            # 第一个模型 (原有的)
            base64_str, ratio, severity, A_value, B_value, ab_lines_base64 = process_image(file)
            
            # 计算score（基于severity）
            if severity == "Mild":
                score = 3
            elif severity == "Moderate":
                score = 2
            elif severity == "Severe":
                score = 1
            else:
                score = 0
            
            return jsonify({
                'success': True,
                'image': base64_str,  # 关键点图片用于前端显示
                'keypoints_image': base64_str,  # 关键点图片
                'ab_lines_image': ab_lines_base64,  # 辅助线图片
                'ratio': f"{ratio:.6f}",
                'severity': severity,
                'score': score,
                'A_value': f"{A_value:.2f}",
                'B_value': f"{B_value:.2f}",
                'type': image_type
            })
            
        elif model_type == 'nostril':
            # 第二个模型 (鼻孔检测)
            result = process_image_with_model(file, 'nostril')
            if result and result[0] is not None:
                base64_str, ratio, severity, score, cc_distance, cn_distance, nostril_lines_base64 = result
                
                return jsonify({
                    'success': True,
                    'image': base64_str,  # 关键点图片用于前端显示
                    'keypoints_image': base64_str,  # 关键点图片
                    'nostril_lines_image': nostril_lines_base64,  # 辅助线图片
                    'ratio': f"{ratio:.6f}",
                    'severity': severity,
                    'score': score,
                    'CC_distance': f"{cc_distance:.2f}",
                    'CN_distance': f"{cn_distance:.2f}",
                    'type': image_type
                })
            else:
                return jsonify({'success': False, 'error': '第二个模型处理失败'})
        elif model_type == 'columellar':
            # 第三个模型 (鼻柱角度)
            result = process_image_with_model(file, 'columellar')
            if result and result[0] is not None:
                base64_str, n_point, original_image = result
                
                return jsonify({
                    'success': True,
                    'image': base64_str,  # 这是带N点的图片，角度线会通过后续的process_columellar_angle添加
                    'original_image': original_image,
                    'n_point': n_point,
                    'type': image_type
                })
            else:
                return jsonify({'success': False, 'error': '第三个模型处理失败'})
        else:
            return jsonify({'success': False, 'error': f'未知的模型类型: {model_type}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_comprehensive_case', methods=['POST'])
def save_comprehensive_case():
    """保存综合分析病例"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '请先登录'})
    
    try:
        data = request.get_json()
        
        # 使用alar图片作为主要图片（因为这是最完整的分析）
        pre_image = data.get('pre_alar_image')
        post_image = data.get('post_alar_image')
        
        # 创建综合分析严重程度描述
        pre_severity = f"Comprehensive: Alar({data.get('pre_alar_severity', 'N/A')}), Nostril({data.get('pre_nostril_severity', 'N/A')}), Angle({data.get('pre_columellar_angle', 0)}°)"
        post_severity = f"Comprehensive: Alar({data.get('post_alar_severity', 'N/A')}), Nostril({data.get('post_nostril_severity', 'N/A')}), Angle({data.get('post_columellar_angle', 0)}°)"
        
        # 使用alar ratio作为主要ratio
        pre_ratio = data.get('pre_alar_ratio')
        post_ratio = data.get('post_alar_ratio')
        
        # 创建新病例
        new_case = Case(
            user_id=session['user_id'],
            pre_image=pre_image,
            post_image=post_image,
            pre_severity=pre_severity,
            post_severity=post_severity,
            pre_ratio=pre_ratio,
            post_ratio=post_ratio,
            pre_nostril_image=data.get('pre_nostril_image'),
            post_nostril_image=data.get('post_nostril_image'),
            pre_nostril_ratio=data.get('pre_nostril_ratio'),
            post_nostril_ratio=data.get('post_nostril_ratio'),
            pre_nostril_severity=data.get('pre_nostril_severity'),
            post_nostril_severity=data.get('post_nostril_severity'),
            pre_columellar_image=data.get('pre_columellar_image'),
            post_columellar_image=data.get('post_columellar_image'),
            pre_columellar_angle=data.get('pre_columellar_angle'),
            post_columellar_angle=data.get('post_columellar_angle'),
            pre_columellar_severity=calculate_columellar_severity(data.get('pre_columellar_angle')),
            post_columellar_severity=calculate_columellar_severity(data.get('post_columellar_angle')),
            analysis_type='comprehensive',
            doctor_reviewed=False
        )
        
        db.session.add(new_case)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '综合分析病例保存成功'})
        
    except Exception as e:
        print(f"保存综合分析病例时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_columellar_angle', methods=['POST'])
def process_columellar_angle():
    """处理鼻柱角度绘制"""
    try:
        from services.ml_interface import get_nostril_detector
        import json
        import math
        
        action = request.form.get('action')
        if action != 'draw_angle':
            return jsonify({'success': False, 'error': '无效的操作'})
        
        image_type = request.form.get('image_type')
        angle = float(request.form.get('angle', 0))
        direction = request.form.get('direction', 'right')
        n_point_str = request.form.get('n_point', '[0,0]')
        n_point = json.loads(n_point_str)
        
        # 这里应该从session或其他地方获取原始图像
        # 为了简化，我们先返回一个模拟的角度线图像
        detector = get_nostril_detector()
        
        # 创建一个简单的角度线图像（模拟）
        import numpy as np
        import cv2
        import base64
        
        # 创建一个基础图像
        img_array = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        # 绘制N点
        x, y = int(n_point[0] if n_point[0] > 0 else 300), int(n_point[1] if n_point[1] > 0 else 200)
        cv2.circle(img_array, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(img_array, 'N', (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制垂直参考线
        cv2.line(img_array, (x, y-100), (x, y+100), (128, 128, 128), 2)
        
        # 计算角度线的终点
        angle_rad = math.radians(angle)
        line_length = 80
        
        if direction == 'right':
            end_x = x + int(line_length * math.sin(angle_rad))
            end_y = y - int(line_length * math.cos(angle_rad))
        else:  # left
            end_x = x - int(line_length * math.sin(angle_rad))
            end_y = y - int(line_length * math.cos(angle_rad))
        
        # 绘制角度线
        cv2.line(img_array, (x, y), (end_x, end_y), (255, 0, 0), 3)
        
        # 绘制角度弧
        cv2.ellipse(img_array, (x, y), (30, 30), -90, 0, angle if direction == 'right' else -angle, (0, 255, 0), 2)
        
        # 添加角度标注
        text_x = x + (20 if direction == 'right' else -50)
        text_y = y - 20
        cv2.putText(img_array, f'{angle}°', (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 添加说明文字
        cv2.putText(img_array, f'Columellar Angle: {angle}° ({direction} deviation)', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 转换为base64
        image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
        
    except Exception as e:
        print(f"处理鼻柱角度时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash("Please log in to access your profile.")
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash("User not found.")
        return redirect(url_for('login'))
    
    # 获取或创建用户资料
    user_profile = user.profile
    if not user_profile:
        user_profile = UserProfile(user_id=user.id)
        db.session.add(user_profile)
        db.session.commit()
    
    form = ProfileForm()
    
    if form.validate_on_submit():
        user_profile.age = form.age.data
        user_profile.gender = form.gender.data
        user_profile.contact = form.contact.data
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
    
    # 预填表单数据
    elif request.method == 'GET':
        form.age.data = user_profile.age
        form.gender.data = user_profile.gender
        form.contact.data = user_profile.contact
    
    return render_template('profile.html', 
                         form=form, 
                         user=user, 
                         profile=user_profile)

@app.route('/patient/<int:user_id>/profile')
def view_patient_profile(user_id):
    if 'user_id' not in session:
        flash("Please log in to view patient profiles.")
        return redirect(url_for('login'))
    
    # 检查当前用户是否是医生
    current_user = User.query.get(session['user_id'])
    if not current_user or current_user.role != 'doctor':
        flash("Only doctors can view patient profiles.")
        return redirect(url_for('home'))
    
    # 获取病人信息
    patient = User.query.get_or_404(user_id)
    if patient.role != 'patient':
        flash("Invalid patient ID.")
        return redirect(url_for('view_all_cases'))
    
    # 获取病人的所有病例
    cases = Case.query.filter_by(user_id=user_id).all()
    
    return render_template('patient_profile.html', 
                         patient=patient,
                         cases=cases)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        from services.ml_interface import get_detector
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '未收到图像数据'})
            
        # 解码 Base64 图像
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 转换为 OpenCV 格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': '图像解码失败'})
        
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用HRNet检测器处理帧
        detector_instance = get_detector()
        if detector_instance.model is None:
            return jsonify({'success': False, 'error': '模型未加载'})
        
        try:
            base64_str, ratio, severity, keypoints = detector_instance._process_image_array(frame_rgb)
            
            return jsonify({
                'success': True,
                'keypoints': keypoints,
                'ratio': f"{ratio:.6f}",
                'severity': severity,
                'image': base64_str
            })
        except Exception as detection_error:
            return jsonify({
                'success': False,
                'error': f'检测失败: {str(detection_error)}'
            })
            
    except Exception as e:
        print(f"处理帧时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/manual_testing', methods=['GET', 'POST'])
def manual_testing():
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        flash("Access denied. Only doctors can access manual testing.")
        return redirect(url_for('home'))

    if request.method == 'POST':
        pre_image = request.files.get('pre_op_image')
        post_image = request.files.get('post_op_image')
        
        # A/B ratio parameters
        pre_value_a = request.form.get('pre_value_a')
        pre_value_b = request.form.get('pre_value_b')
        post_value_a = request.form.get('post_value_a')
        post_value_b = request.form.get('post_value_b')
        
        # CC/CN ratio parameters
        pre_value_cc = request.form.get('pre_value_cc')
        pre_value_cn = request.form.get('pre_value_cn')
        post_value_cc = request.form.get('post_value_cc')
        post_value_cn = request.form.get('post_value_cn')
        
        # Angle parameters
        pre_angle = request.form.get('pre_angle')
        post_angle = request.form.get('post_angle')
        
        patient_name = request.form.get('patient_name')
        save_to_cases = request.form.get('save_to_cases') == 'yes'

        # 计算A/B比率
        pre_ratio = None
        post_ratio = None
        if pre_value_a and pre_value_b and float(pre_value_b) != 0:
            pre_ratio = float(pre_value_a) / float(pre_value_b)
        if post_value_a and post_value_b and float(post_value_b) != 0:
            post_ratio = float(post_value_a) / float(post_value_b)

        # 计算CC/CN比率
        pre_nostril_ratio = None
        post_nostril_ratio = None
        if pre_value_cc and pre_value_cn and float(pre_value_cn) != 0:
            pre_nostril_ratio = float(pre_value_cc) / float(pre_value_cn)
        if post_value_cc and post_value_cn and float(post_value_cn) != 0:
            post_nostril_ratio = float(post_value_cc) / float(post_value_cn)

        # 处理角度值
        pre_columellar_angle = float(pre_angle) if pre_angle else None
        post_columellar_angle = float(post_angle) if post_angle else None

        # 处理图片（如果有的话）
        pre_base64 = None
        post_base64 = None
        if pre_image and pre_image.filename:
            pre_base64, _, _, _, _ = process_image(pre_image)
        if post_image and post_image.filename:
            post_base64, _, _, _, _ = process_image(post_image)

        # 计算严重程度
        def calculate_alar_severity(ratio):
            if ratio is None:
                return None
            if ratio <= 0.05:
                return 'Mild'
            elif ratio <= 0.10:
                return 'Moderate'
            else:
                return 'Severe'

        def calculate_nostril_severity(ratio):
            if ratio is None:
                return None
            if ratio <= 1.2:
                return 'Mild'
            elif ratio <= 1.5:
                return 'Moderate'
            else:
                return 'Severe'

        pre_alar_severity = calculate_alar_severity(pre_ratio)
        post_alar_severity = calculate_alar_severity(post_ratio)
        pre_nostril_severity = calculate_nostril_severity(pre_nostril_ratio)
        post_nostril_severity = calculate_nostril_severity(post_nostril_ratio)
        pre_columellar_severity = calculate_columellar_severity(pre_columellar_angle)
        post_columellar_severity = calculate_columellar_severity(post_columellar_angle)

        # 准备结果
        pre_result = (pre_base64, pre_ratio, "Manual Test") if pre_ratio is not None or pre_base64 else None
        post_result = (post_base64, post_ratio, "Manual Test") if post_ratio is not None or post_base64 else None

        # 检查是否有任何数据需要保存
        has_data = any([
            pre_ratio, post_ratio, pre_nostril_ratio, post_nostril_ratio,
            pre_columellar_angle, post_columellar_angle, pre_base64, post_base64
        ])

        if save_to_cases and has_data:
            # 为用户名添加时间戳以确保唯一性
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_username = f"{patient_name}_{timestamp}"
            
            # 为手动测试创建一个临时用户
            temp_user = User(
                username=unique_username,
                password_hash="manual_test",
                role='manual_test'
            )
            db.session.add(temp_user)
            db.session.flush()

            # 创建综合严重程度描述
            pre_severity_parts = []
            post_severity_parts = []
            
            if pre_alar_severity:
                pre_severity_parts.append(f"Alar({pre_alar_severity})")
            if pre_nostril_severity:
                pre_severity_parts.append(f"Nostril({pre_nostril_severity})")
            if pre_columellar_angle is not None:
                pre_severity_parts.append(f"Angle({pre_columellar_angle}°)")
                
            if post_alar_severity:
                post_severity_parts.append(f"Alar({post_alar_severity})")
            if post_nostril_severity:
                post_severity_parts.append(f"Nostril({post_nostril_severity})")
            if post_columellar_angle is not None:
                post_severity_parts.append(f"Angle({post_columellar_angle}°)")
            
            pre_severity = f"Manual Test - {patient_name}: {', '.join(pre_severity_parts)}" if pre_severity_parts else f"Manual Test - {patient_name}"
            post_severity = f"Manual Test - {patient_name}: {', '.join(post_severity_parts)}" if post_severity_parts else f"Manual Test - {patient_name}"

            new_case = Case(
                user_id=temp_user.id,
                pre_image=pre_base64,
                post_image=post_base64,
                pre_severity=pre_severity,
                post_severity=post_severity,
                pre_ratio=pre_ratio,
                post_ratio=post_ratio,
                pre_nostril_ratio=pre_nostril_ratio,
                post_nostril_ratio=post_nostril_ratio,
                pre_nostril_severity=pre_nostril_severity,
                post_nostril_severity=post_nostril_severity,
                pre_columellar_angle=pre_columellar_angle,
                post_columellar_angle=post_columellar_angle,
                pre_columellar_severity=pre_columellar_severity,
                post_columellar_severity=post_columellar_severity,
                analysis_type='manual_comprehensive',
                doctor_reviewed=False
            )
            db.session.add(new_case)
            db.session.commit()
            flash(f"Comprehensive manual test case saved successfully for patient: {patient_name}")
            return redirect(url_for('view_all_cases'))

        return render_template('manual_testing.html', 
                            pre_result=pre_result, 
                            post_result=post_result,
                            pre_ratio=pre_ratio,
                            post_ratio=post_ratio)

    # GET 请求时传入空值
    return render_template('manual_testing.html', 
                         pre_result=None, 
                         post_result=None,
                         pre_ratio=None,
                         post_ratio=None)

@app.route('/user_management')
def user_management():
    if 'user_id' not in session:
        flash("Please log in")
        return redirect(url_for('login'))
        
    # 检查是否是超级管理员 - 从数据库中查询
    current_user = User.query.get(session['user_id'])
    if not current_user.is_admin:
        flash("No permission to access this page")
        return redirect(url_for('home'))
        
    users = User.query.all()
    return render_template('user_management.html', users=users, now=datetime.now())

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session:
        flash("Please log in")
        return redirect(url_for('login'))
        
    current_user = User.query.get(session['user_id'])
    if not current_user.is_admin:
        flash("No permission")
        return redirect(url_for('home'))
        
    if current_user.id == user_id:
        flash("Cannot delete your own account")
        return redirect(url_for('user_management'))
        
    user_to_delete = User.query.get_or_404(user_id)
    # 删除用户关联的所有病例
    Case.query.filter_by(user_id=user_id).delete()
    # 删除用户资料
    if user_to_delete.profile:
        db.session.delete(user_to_delete.profile)
    # 删除用户
    db.session.delete(user_to_delete)
    db.session.commit()
    
    flash("User deleted successfully")
    return redirect(url_for('user_management'))

@app.route('/debug_session')
def debug_session():
    if 'user_id' not in session:
        return "Not logged in"
    
    user = User.query.get(session['user_id'])
    return {
        "Current username": user.username,
        "Username in session": session.get('username'),
        "Is admin": user.is_admin,
        "Full session": dict(session)
    }

@app.route('/update_user_role/<int:user_id>', methods=['POST'])
def update_user_role(user_id):
    if 'user_id' not in session:
        flash('Access denied')
        return redirect(url_for('home'))
    
    current_user = User.query.get(session['user_id'])
    if not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('home'))
    
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role in ['patient', 'doctor']:
        user.role = new_role
        db.session.commit()
        flash(f'Successfully updated role for {user.username}')
    else:
        flash('Invalid role')
    
    return redirect(url_for('user_management'))

@app.route('/download_image/<int:case_id>/<type>')
def download_image(case_id, type):
    if 'user_id' not in session or session.get('role') != 'doctor':
        flash('Access denied')
        return redirect(url_for('home'))
    
    try:
        case = Case.query.get_or_404(case_id)
        if type == 'pre':
            image_data = case.pre_image
            filename = f'pre_op_{case_id}'
        else:
            image_data = case.post_image
            filename = f'post_op_{case_id}'
            
        if not image_data:
            flash('Image not found')
            return redirect(url_for('view_all_cases'))
        
        # 处理图片数据
        try:
            if ',' in image_data:  # base64格式
                image_data = base64.b64decode(image_data.split(',')[1])
            else:  # 直接的二进制数据
                image_data = base64.b64decode(image_data)
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            image_data = base64.b64decode(image_data)  # 尝试直接解码
            
        # 检测图片格式
        if image_data.startswith(b'\xff\xd8'):  # JPEG格式
            filename += '.jpg'
            mimetype = 'image/jpeg'
        elif image_data.startswith(b'\x89PNG'):  # PNG格式
            filename += '.png'
            mimetype = 'image/png'
        else:
            filename += '.jpg'  # 默认使用jpg
            mimetype = 'image/jpeg'
            
        return send_file(
            io.BytesIO(image_data),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Download error: {str(e)}")
        flash('Error downloading image')
        return redirect(url_for('view_all_cases'))

@app.route('/download_all_images')
def download_all_images():
    if 'user_id' not in session or session.get('role') != 'doctor':
        flash('Access denied')
        return redirect(url_for('home'))
    
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            cases = Case.query.all()
            for case in cases:
                if case.pre_image:
                    try:
                        if ',' in case.pre_image:  # base64格式
                            image_data = base64.b64decode(case.pre_image.split(',')[1])
                        else:  # 直接的二进制数据
                            image_data = case.pre_image
                        zf.writestr(f'case_{case.id}/pre_op.jpg', image_data)
                    except Exception as e:
                        print(f"Error processing pre-image for case {case.id}: {str(e)}")
                        continue

                if case.post_image:
                    try:
                        if ',' in case.post_image:  # base64格式
                            image_data = base64.b64decode(case.post_image.split(',')[1])
                        else:  # 直接的二进制数据
                            image_data = case.post_image
                        zf.writestr(f'case_{case.id}/post_op.jpg', image_data)
                    except Exception as e:
                        print(f"Error processing post-image for case {case.id}: {str(e)}")
                        continue
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='all_cases_images.zip'
        )
    except Exception as e:
        print(f"Download all error: {str(e)}")
        flash('Error downloading all images')
        return redirect(url_for('view_all_cases'))

@app.route('/update_user_admin/<int:user_id>', methods=['POST'])
def update_user_admin(user_id):
    """更新用户的管理员权限"""
    if 'user_id' not in session:
        flash('Access denied')
        return redirect(url_for('home'))
    
    current_user = User.query.get(session['user_id'])
    if not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('home'))
    
    user = User.query.get_or_404(user_id)
    action = request.form.get('action')
    
    if action == 'add':
        user.is_admin = True
        db.session.commit()
        flash(f'Admin privileges granted to {user.username}')
    elif action == 'remove':
        if user.id == current_user.id:
            flash('Cannot remove your own admin privileges')
        else:
            user.is_admin = False
            db.session.commit()
            flash(f'Admin privileges removed from {user.username}')
    else:
        flash('Invalid action')
    
    return redirect(url_for('user_management'))

# --- Initialize Database ---
if __name__ == '__main__':
    init_db()  # 现在这个函数只会在数据库不存在时创建表
    app.run(host='127.0.0.1', port=5002, debug=False)