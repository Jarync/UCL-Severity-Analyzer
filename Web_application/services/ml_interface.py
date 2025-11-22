import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import base64
from io import BytesIO
import io

# 新增：从 encryption_utils 导入解密函数和密钥
# from services.encryption_utils import decrypt_file_content, SECRET_KEY

# 添加HRNet模型路径到Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HRNET_DIR = os.path.join(BASE_DIR, 'HRNet-Facial-Landmark-Detection')
sys.path.insert(0, HRNET_DIR)

# 导入HRNet相关模块
from lib.config.defaults import _C as cfg
from lib.config.defaults import update_config
from lib.models.hrnet import get_face_alignment_net
from lib.core.evaluation import get_preds
from lib.datasets.cleftlip import CleftLip

# 第二个模型相关导入
SECOND_MODEL_DIR = os.path.join(BASE_DIR, 'Second_model')
sys.path.insert(0, SECOND_MODEL_DIR)

class CleftLipDetector:
    def __init__(self):
        """初始化唇裂检测器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和配置"""
        try:
            # 设置配置文件和模型文件路径
            yaml_path = os.path.join(HRNET_DIR, "experiments", "cleft_lip", "pose_hrnet_w18_cleft.yaml")
            # 修改：加载加密后的模型文件
            # model_path = os.path.join(HRNET_DIR, "best_NVM_cleftlip_model_HRNet.enc")
            model_path = os.path.join(HRNET_DIR, "best_NVM_cleftlip_model_HRNet.pth")

            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
            # 修改：检查加密文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"加密模型文件不存在: {model_path}")

            # 创建一个临时的简化配置，避免YAML编码问题
            self._setup_config()

            # 新增：解密模型文件
            # decrypted_model_stream = decrypt_file_content(model_path, SECRET_KEY)
            # if decrypted_model_stream is None:
            #     raise RuntimeError(f"模型文件 {model_path} 解密失败。")

            # 加载模型
            self.model = get_face_alignment_net(cfg)
            # 修改：从解密后的字节流中加载模型
            # checkpoint = torch.load(decrypted_model_stream, map_location=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ HRNet唇裂检测模型加载成功! 设备: {self.device}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            self.model = None
    
    def _setup_config(self):
        """手动设置配置，避免YAML编码问题"""
        # 基本配置
        cfg.MODEL.NAME = 'hrnet'
        cfg.MODEL.NUM_JOINTS = 6
        cfg.MODEL.INIT_WEIGHTS = True
        cfg.MODEL.IMAGE_SIZE = [256, 256]
        cfg.MODEL.HEATMAP_SIZE = [64, 64]
        cfg.MODEL.SIGMA = 1.5
        cfg.MODEL.TARGET_TYPE = 'Gaussian'
        
        # HRNet结构配置
        cfg.MODEL.EXTRA.STEM_INPLANES = 64
        cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
        cfg.MODEL.EXTRA.WITH_HEAD = True
        
        # Stage2配置
        cfg.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
        cfg.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
        cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
        cfg.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'
        
        # Stage3配置
        cfg.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
        cfg.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
        cfg.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
        cfg.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
        cfg.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'
        
        # Stage4配置
        cfg.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
        cfg.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
        cfg.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
        cfg.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'
        
        # 数据集配置（用于transform）
        cfg.DATASET.DATASET = 'CLEFT_LIP'
        cfg.DATASET.FLIP = True
        cfg.DATASET.SCALE_FACTOR = 0.25
        cfg.DATASET.ROT_FACTOR = 30
    
    def calculate_ab_ratio(self, keypoints):
        """计算A/B比值
        
        A: Nleft到Nright的Y坐标距离的绝对值
        B: 从眼睛中心点到内嘴角中心点的连线中点到较低的鼻翼点的Y坐标距离
        """
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        
        # get keypoints coordinates
        E1, E2 = keypoints[0], keypoints[1]  # eye keypoints
        I1, I2 = keypoints[2], keypoints[3]  # mouth keypoints
        N_left, N_right = keypoints[4], keypoints[5]  # nose keypoints
        
        # calculate A: absolute value of the Y coordinate distance between Nleft and Nright
        A = abs(N_left[1] - N_right[1])
        
        # calculate B:
        # 1. calculate eye center point
        eye_center = (E1 + E2) / 2
        # 2. calculate mouth center point
        mouth_center = (I1 + I2) / 2
        # 3. calculate the midpoint of the line connecting the two center points
        center_line_midpoint = (eye_center + mouth_center) / 2
        
        # 4. find the lower nose point (the point with the larger Y coordinate value)
        lower_nose_y = max(N_left[1], N_right[1])
        
        # 5. calculate B: the Y coordinate distance from the midpoint to the lower nose point
        B = abs(lower_nose_y - center_line_midpoint[1])
        
        # calculate ratio
        ratio = A / B if B > 0 else 0
        
        return ratio, A, B
    
    def get_severity(self, ratio):
        """according to the A/B ratio to judge the severity"""
        if ratio <= 0.05:
            return "Mild", 3
        elif ratio <= 0.10:
            return "Moderate", 2
        else:
            return "Severe", 1
    
    def process_image_file(self, image_file):
        """处理上传的图像文件"""
        if self.model is None:
            raise RuntimeError("模型未加载成功")
        
        # 读取图像
        image_file.seek(0)  # 重置文件指针
        img = Image.open(image_file)
        img_array = np.array(img)
        
        # 如果是RGBA，转换为RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            # 已经是RGB，不需要转换
            pass
        else:
            # 灰度图转RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        return self._process_image_array(img_array)
    
    def _process_image_array(self, img_array):
        """处理图像数组并返回结果"""
        # 创建单图数据集
        dataset = self._create_single_image_dataset(img_array)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for inp, _, meta in loader:
                inp = inp.to(self.device)
                output = self.model(inp)
                heatmaps = output.cpu()
                
                # 使用亚像素精度预测（参考validate2.py）
                preds = self._get_preds_with_subpixel(heatmaps)
                
                # 先缩放到256x256（模型训练时的输入尺寸）
                preds[:, :, 0] *= 256.0 / 64.0  # width
                preds[:, :, 1] *= 256.0 / 64.0  # height
                pred_pts_256 = preds[0].numpy()
                
                # 为了计算A/B比值，需要缩放到原图尺寸
                img_h, img_w = img_array.shape[:2]
                scale_x = img_w / 256.0
                scale_y = img_h / 256.0
                pred_pts_for_ratio = pred_pts_256.copy()
                pred_pts_for_ratio[:, 0] *= scale_x
                pred_pts_for_ratio[:, 1] *= scale_y
                
                # 计算A/B比值（使用缩放到原图尺寸的坐标）
                ratio, A, B = self.calculate_ab_ratio(pred_pts_for_ratio)
                severity, score = self.get_severity(ratio)
                
                # 绘制关键点也使用原图尺寸的坐标
                result_img = self._draw_keypoints(img_array.copy(), pred_pts_for_ratio)
                
                # 转换为base64
                base64_str = self._image_to_base64(result_img)
                
                return base64_str, ratio, severity, score, pred_pts_for_ratio.tolist()
        
    def process_image_file_with_ab_lines(self, image_file):
        """处理图像文件并返回带A/B辅助线的结果"""
        try:
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                img_data = image_file.read()
            else:
                with open(image_file, 'rb') as f:
                    img_data = f.read()
            
            # 将图像数据转换为numpy数组
            nparr = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                print("无法解码图像")
                return None
            
            # 转换为RGB格式
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"读取图像时出错: {e}")
            return None
        
        # 处理图像
        return self._process_image_array_with_ab_lines(img_array)
    
    def _process_image_array_with_ab_lines(self, img_array):
        """处理图像数组并返回带A/B辅助线的结果"""
        # 创建单图数据集
        dataset = self._create_single_image_dataset(img_array)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for inp, _, meta in loader:
                inp = inp.to(self.device)
                output = self.model(inp)
                heatmaps = output.cpu()
                
                # 使用亚像素精度预测（参考validate2.py）
                preds = self._get_preds_with_subpixel(heatmaps)
                
                # 先缩放到256x256（模型训练时的输入尺寸）
                preds[:, :, 0] *= 256.0 / 64.0  # width
                preds[:, :, 1] *= 256.0 / 64.0  # height
                pred_pts_256 = preds[0].numpy()
                
                # 为了计算A/B比值，需要缩放到原图尺寸
                img_h, img_w = img_array.shape[:2]
                scale_x = img_w / 256.0
                scale_y = img_h / 256.0
                pred_pts_for_ratio = pred_pts_256.copy()
                pred_pts_for_ratio[:, 0] *= scale_x
                pred_pts_for_ratio[:, 1] *= scale_y
                
                # 绘制A/B辅助线
                result_img = self._draw_ab_lines(img_array.copy(), pred_pts_for_ratio)
                
                # 转换为base64
                base64_str = self._image_to_base64(result_img)
                
                return base64_str
    
    def _get_preds_with_subpixel(self, heatmaps):
        """获取亚像素精度的关键点预测（参考validate2.py）"""
        # heatmaps: (B, J, H, W)
        assert heatmaps.dim() == 4
        B, J, H, W = heatmaps.shape
        
        preds = get_preds(heatmaps)  # (B, J, 2) 整数
        preds = preds - 1  # 0-based
        preds = preds.float()
        
        # 亚像素精度优化
        for b in range(B):
            for j in range(J):
                px, py = int(preds[b, j, 0]), int(preds[b, j, 1])
                if 1 < px < W-2 and 1 < py < H-2:
                    dx = (heatmaps[b, j, py, px+1] - heatmaps[b, j, py, px-1]) / 2
                    dy = (heatmaps[b, j, py+1, px] - heatmaps[b, j, py-1, px]) / 2
                    preds[b, j, 0] += dx.item() * 0.25
                    preds[b, j, 1] += dy.item() * 0.25
        
        return preds
    
    def _create_single_image_dataset(self, img_array):
        """创建单图数据集"""
        class SingleImageDataset(Dataset):
            def __init__(self, image_array):
                self.img = image_array
                h, w = self.img.shape[:2]
                self.center = torch.tensor([w / 2, h / 2], dtype=torch.float32)
                self.scale = torch.tensor([1.0, 1.0], dtype=torch.float32)
                
                # 创建简化的transform，避免数据集路径问题
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                img = Image.fromarray(self.img)
                inp = self.transform(img)
                meta = {
                    'center': self.center.unsqueeze(0),
                    'scale': self.scale.unsqueeze(0),
                    'index': torch.tensor([0]),
                    'pts': torch.zeros((1, 6, 2))
                }
                dummy_label = torch.zeros((6, 2))
                return inp, dummy_label, meta
        
        return SingleImageDataset(img_array)
    
    def _draw_keypoints(self, image, keypoints):
        """在图像上绘制关键点"""
        keypoint_names = ['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']
        colors = [(0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 255, 0), (255, 0, 0), (255, 0, 0)]  # 蓝、蓝、绿、绿、红、红
        
        for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
            x, y = int(kp[0]), int(kp[1])
            # 增大点的大小
            cv2.circle(image, (x, y), 12, color, -1)  # 填充颜色
            # 增大文字大小和粗细
            cv2.putText(image, name, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 彩色文字
        
        return image
    
    def _draw_ab_lines(self, image, keypoints):
        """绘制A/B比值计算的辅助线"""
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        
        # 获取关键点坐标
        E1, E2 = keypoints[0], keypoints[1]  # 眼部关键点
        I1, I2 = keypoints[2], keypoints[3]  # 内嘴角关键点
        N_left, N_right = keypoints[4], keypoints[5]  # 鼻翼关键点
        
        # 先绘制关键点
        keypoint_names = ['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']
        colors = [(0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 255, 0), (255, 0, 0), (255, 0, 0)]  # 蓝、蓝、绿、绿、红、红
        
        for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
            x, y = int(kp[0]), int(kp[1])
            # 增大点的大小
            cv2.circle(image, (x, y), 12, color, -1)  # 填充颜色
            # 增大文字大小和粗细
            cv2.putText(image, name, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 彩色文字
        
        # 计算中心点
        eye_center = (E1 + E2) / 2
        mouth_center = (I1 + I2) / 2
        center_line_midpoint = (eye_center + mouth_center) / 2
        
        # 计算正确的A和B值
        A = abs(N_left[1] - N_right[1])  # Y坐标差的绝对值
        lower_nose_y = max(N_left[1], N_right[1])  # 找到最低的鼻翼点
        lower_nose_point = N_left if N_left[1] > N_right[1] else N_right
        B = abs(lower_nose_y - center_line_midpoint[1])  # 中心点到最低鼻翼的垂直距离
        ratio = A / B if B > 0 else 0
        
        # 1. 绘制眼睛中心点
        cv2.circle(image, (int(eye_center[0]), int(eye_center[1])), 10, (255, 165, 0), -1)  # 橙色填充
        cv2.putText(image, 'Eye Center', (int(eye_center[0])+15, int(eye_center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # 橙色文字
        
        # 2. 绘制内嘴角中心点
        cv2.circle(image, (int(mouth_center[0]), int(mouth_center[1])), 10, (255, 165, 0), -1)  # 橙色填充
        cv2.putText(image, 'Mouth Center', (int(mouth_center[0])+15, int(mouth_center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # 橙色文字
        
        # 3. 绘制眼睛-嘴角连线
        cv2.line(image, 
                 (int(eye_center[0]), int(eye_center[1])), 
                 (int(mouth_center[0]), int(mouth_center[1])), 
                 (0, 255, 255), 4)  # 青色线
        
        # 4. 绘制并标记中心点
        cv2.circle(image, (int(center_line_midpoint[0]), int(center_line_midpoint[1])), 12, (255, 0, 255), -1)  # 紫色填充
        cv2.putText(image, 'Center', (int(center_line_midpoint[0])+15, int(center_line_midpoint[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)  # 紫色文字
        
        # 5. 绘制A值的垂直线（NL和NR的Y坐标差）
        vertical_x = int((N_left[0] + N_right[0]) / 2) + 50
        cv2.line(image, 
                 (vertical_x, int(N_left[1])), 
                 (vertical_x, int(N_right[1])), 
                 (0, 255, 255), 6)  # 青色粗线
        
        # 在A线中点标注
        a_mid_y = int((N_left[1] + N_right[1]) / 2)
        cv2.putText(image, f'A={A:.2f}', (vertical_x + 10, a_mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)  # 青色文字
        
        # 6. 绘制B值的垂直线（中心点到最低鼻翼点）
        cv2.line(image, 
                 (int(center_line_midpoint[0]), int(center_line_midpoint[1])), 
                 (int(center_line_midpoint[0]), int(lower_nose_point[1])), 
                 (0, 255, 0), 6)  # 绿色粗线
        
        # 在B线中点标注
        b_mid_y = int((center_line_midpoint[1] + lower_nose_point[1]) / 2)
        cv2.putText(image, f'B={B:.2f}', (int(center_line_midpoint[0]) + 15, b_mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # 绿色文字
        
        # 7. 标记最低的鼻翼点 - 保留白色描边
        cv2.circle(image, (int(lower_nose_point[0]), int(lower_nose_point[1])), 15, (255, 255, 255), 4)  # 白色圆圈边框加粗
        lower_label = "NL(lowest)" if N_left[1] > N_right[1] else "NR(lowest)"
        cv2.putText(image, lower_label, (int(lower_nose_point[0])+15, int(lower_nose_point[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if N_left[1] > N_right[1] else (0, 0, 255), 2)  # 与点同色文字
        
        # 8. 添加计算说明
        legend_y = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        lines = [
            f'A = |NL_y - NR_y| = |{N_left[1]:.2f} - {N_right[1]:.2f}| = {A:.2f}',
            f'B = |Center_y - Lowest_y| = |{center_line_midpoint[1]:.2f} - {lower_nose_y:.2f}| = {B:.2f}',
            f'A/B Ratio = {A:.2f}/{B:.2f} = {ratio:.6f}'
        ]
        colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
        y_offset = legend_y
        for i, (text, color) in enumerate(zip(lines, colors)):
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            # 计算背景矩形坐标
            rect_x1 = 8
            rect_y1 = y_offset - text_h - 4
            rect_x2 = rect_x1 + text_w + 8
            rect_y2 = y_offset + baseline + 4
            # 画更透明的黑色背景
            overlay = image.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            alpha = 0.35  # 更透明
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            # 绘制文字
            cv2.putText(image, text, (rect_x1 + 4, y_offset), font, font_scale, color, font_thickness)
            y_offset += text_h + 12
        return image
    
    def _image_to_base64(self, image):
        """将图像转换为base64字符串"""
        # 如果是RGB格式，转换为BGR用于OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64

# 全局检测器实例
detector = None

def get_detector():
    """获取检测器实例"""
    global detector
    if detector is None:
        detector = CleftLipDetector()
    return detector

def process_image(file_obj):
    """
    处理图像文件并返回结果
    返回: (base64_image_string, ratio, severity, A_value, B_value, ab_lines_base64)
    """
    try:
        detector = get_detector()
        base64_str, ratio, severity, score, keypoints = detector.process_image_file(file_obj)
        
        # 生成带A/B辅助线的图片
        ab_lines_base64 = detector.process_image_file_with_ab_lines(file_obj)
        
        # 重新计算A和B值以便返回
        if len(keypoints) >= 6:
            keypoints_array = np.array(keypoints)
            ratio_calc, A_value, B_value = detector.calculate_ab_ratio(keypoints_array)
            return base64_str, ratio, severity, A_value, B_value, ab_lines_base64
        else:
            return base64_str, ratio, severity, 0.0, 0.0, ab_lines_base64
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        # 返回错误时的默认值
        return None, 0.0, "处理失败", 0.0, 0.0, None 

class NostrilDetector:
    """第二个模型：鼻孔关键点检测器"""
    def __init__(self):
        """初始化鼻孔关键点检测器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_joints = 5  # CC1, CC2, CN1, CN2, N
        self.image_size = [256, 256]
        self.heatmap_size = [64, 64]
        self._load_model()
    
    def _load_model(self):
        """加载第二个模型"""
        try:
            # 定义简化版HRNet结构（复制自训练代码）
            self.model = self._create_hrnet_model()
            
            # 加载权重
            # 修改：加载加密后的模型文件
            # model_path = os.path.join(SECOND_MODEL_DIR, "best_model_epoch_65_loss_0.1313_err_3.37_20250613_020703.enc")
            model_path = os.path.join(SECOND_MODEL_DIR, "best_model_epoch_65_loss_0.1313_err_3.37_20250613_020703.pth")

            # 修改：检查加密文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"第二个模型加密文件不存在: {model_path}")

            # 新增：解密模型文件
            # decrypted_model_stream = decrypt_file_content(model_path, SECRET_KEY)
            # if decrypted_model_stream is None:
            #     raise RuntimeError(f"第二个模型文件 {model_path} 解密失败。")

            # 加载模型
            # 修改：从解密后的字节流中加载模型
            # checkpoint = torch.load(decrypted_model_stream, map_location=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除module.前缀（如果存在）
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 第二个模型(鼻孔检测)加载成功! 设备: {self.device}")
            
        except Exception as e:
            print(f"❌ 第二个模型加载失败: {str(e)}")
            self.model = None
    
    def _create_hrnet_model(self):
        """创建简化版HRNet模型（复制自训练代码）"""
        class HRNet(torch.nn.Module):
            def __init__(self, num_joints):
                super(HRNet, self).__init__()
                
                # 主干网络 - 降采样到1/4
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True)
                )
                
                self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True)
                )
                
                # 高分辨率分支
                self.branch1 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True)
                )
                
                # 低分辨率分支
                self.branch2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True)
                )
                
                # 特征融合
                self.fuse = torch.nn.Sequential(
                    torch.nn.Conv2d(96, 64, kernel_size=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True)
                )
                
                # 输出头
                self.final_layer = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, num_joints, kernel_size=1)
                )
            
            def forward(self, x):
                # 输入: 256x256
                x = self.conv1(x)  # 128x128
                x = self.conv2(x)  # 64x64
                
                # 分支处理
                x1 = self.branch1(x)  # 64x64
                x2 = self.branch2(x)  # 32x32
                
                # 上采样低分辨率特征
                x2_up = torch.nn.functional.interpolate(
                    x2, size=x1.shape[2:], mode='bilinear', align_corners=False
                )  # 32x32 -> 64x64
                
                # 特征融合
                x = torch.cat([x1, x2_up], dim=1)  # 64x64
                x = self.fuse(x)  # 64x64
                
                # 生成热图
                x = self.final_layer(x)  # 64x64
                
                return x
        
        return HRNet(self.num_joints)
    
    def _get_max_preds(self, heatmaps):
        """从热图中获取关键点预测位置"""
        batch_size = heatmaps.size(0)
        num_joints = heatmaps.size(1)
        width = heatmaps.size(3)  # W
        height = heatmaps.size(2)  # H
        
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = torch.argmax(heatmaps_reshaped, 2)
        maxvals = torch.amax(heatmaps_reshaped, 2)
        
        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints))
        
        preds = torch.zeros((batch_size, num_joints, 2), device=heatmaps.device)
        
        # 确保x对应W, y对应H
        preds[:, :, 0] = (idx % width).float()  # x坐标(W)
        preds[:, :, 1] = (idx // width).float()  # y坐标(H)
        
        return preds, maxvals
    
    def calculate_nostril_ratio(self, keypoints):
        """
        calculate nostril width ratio
        CC1-CC2 distance / CN1-CN2 distance
        """
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        
        # keypoints order: CC1, CC2, CN1, CN2, N
        CC1, CC2, CN1, CN2, N = keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4]
        
        # calculate CC1-CC2 distance (left nostril width)
        cc_distance = np.sqrt((CC1[0] - CC2[0])**2 + (CC1[1] - CC2[1])**2)
        
        # calculate CN1-CN2 distance (right nostril width)
        cn_distance = np.sqrt((CN1[0] - CN2[0])**2 + (CN1[1] - CN2[1])**2)
        
        # calculate ratio
        ratio = cc_distance / cn_distance if cn_distance > 0 else 0
        
        return ratio, cc_distance, cn_distance
    
    def get_nostril_severity(self, ratio):
        """according to the nostril width ratio to judge the severity"""
        if 0.90 <= ratio <= 1.10:
            return "Mild", 3
        elif 0.60 <= ratio < 0.90 or 1.10 < ratio <= 1.40:
            return "Moderate", 2
        elif ratio < 0.60 or ratio > 1.40:
            return "Severe", 1
        else:
            return "Unknown", 0
    
    def process_image_file(self, image_file):
        """处理上传的图像文件"""
        if self.model is None:
            raise RuntimeError("第二个模型未加载成功")
        
        # 读取图像
        image_file.seek(0)  # 重置文件指针
        img = Image.open(image_file)
        img_array = np.array(img)
        
        # 图像预处理
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            pass
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        return self._process_image_array(img_array)
    
    def _process_image_array(self, img_array):
        """处理图像数组并返回结果"""
        # 预处理图像
        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # 转换为tensor并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            preds, maxvals = self._get_max_preds(output)
            
            # 缩放关键点坐标到原图尺寸
            preds = preds[0]  # 取第一个batch
            
            # 先缩放到256x256，然后缩放到原图尺寸
            preds[:, 0] *= 256.0 / 64.0  # width
            preds[:, 1] *= 256.0 / 64.0  # height
            
            # 缩放到原图尺寸
            img_h, img_w = img_array.shape[:2]
            scale_x = img_w / 256.0
            scale_y = img_h / 256.0
            
            pred_pts_scaled = preds.cpu().numpy().copy()
            pred_pts_scaled[:, 0] *= scale_x
            pred_pts_scaled[:, 1] *= scale_y
            
            # 计算鼻孔宽度比值
            ratio, cc_distance, cn_distance = self.calculate_nostril_ratio(pred_pts_scaled)
            severity, score = self.get_nostril_severity(ratio)
            
            # 绘制关键点
            result_img = self._draw_keypoints(img_array.copy(), pred_pts_scaled)
            base64_str = self._image_to_base64(result_img)
            
            return base64_str, ratio, severity, score, cc_distance, cn_distance, pred_pts_scaled.tolist()
    
    def _draw_keypoints(self, image, keypoints):
        """在图像上绘制关键点"""
        keypoint_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]  # 红、红、绿、绿、蓝
        
        for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
            x, y = int(kp[0]), int(kp[1])
            # 增大点的大小
            cv2.circle(image, (x, y), 12, color, -1)  # 填充颜色
            # 增大文字大小和粗细
            cv2.putText(image, name, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 彩色文字
        
        return image
    
    def _draw_nostril_lines(self, image, keypoints):
        """绘制鼻孔宽度比值计算的辅助线"""
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        
        # 关键点顺序: CC1, CC2, CN1, CN2, N
        CC1, CC2, CN1, CN2, N = keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4]
        
        # 先绘制关键点
        keypoint_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
        
        for i, (kp, name, color) in enumerate(zip(keypoints, keypoint_names, colors)):
            x, y = int(kp[0]), int(kp[1])
            # 只绘制彩色实心点，无白色描边
            cv2.circle(image, (x, y), 12, color, -1)  # 彩色填充
            # 增大文字大小和粗细
            cv2.putText(image, name, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 彩色文字
        
        # 计算距离
        cc_distance = np.sqrt((CC1[0] - CC2[0])**2 + (CC1[1] - CC2[1])**2)
        cn_distance = np.sqrt((CN1[0] - CN2[0])**2 + (CN1[1] - CN2[1])**2)
        ratio = cc_distance / cn_distance if cn_distance > 0 else 0
        
        # 绘制CC1-CC2连线（左鼻孔）- 更粗更明显
        cv2.line(image, (int(CC1[0]), int(CC1[1])), (int(CC2[0]), int(CC2[1])), (255, 255, 255), 6)  # 白色边框线
        cv2.line(image, (int(CC1[0]), int(CC1[1])), (int(CC2[0]), int(CC2[1])), (255, 255, 0), 4)  # 黄色线
        
        cc_mid_x = int((CC1[0] + CC2[0]) / 2)
        cc_mid_y = int((CC1[1] + CC2[1]) / 2)
        cv2.putText(image, f'CC={cc_distance:.2f}', (cc_mid_x, cc_mid_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)  # 黄色文字
        
        # 绘制CN1-CN2连线（右鼻孔）- 更粗更明显
        cv2.line(image, (int(CN1[0]), int(CN1[1])), (int(CN2[0]), int(CN2[1])), (255, 255, 255), 6)  # 白色边框线
        cv2.line(image, (int(CN1[0]), int(CN1[1])), (int(CN2[0]), int(CN2[1])), (0, 255, 255), 4)  # 青色线
        
        cn_mid_x = int((CN1[0] + CN2[0]) / 2)
        cn_mid_y = int((CN1[1] + CN2[1]) / 2)
        cv2.putText(image, f'CN={cn_distance:.2f}', (cn_mid_x, cn_mid_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 青色文字
        
        # 标记鼻柱中心点 - 只保留彩色点
        cv2.circle(image, (int(N[0]), int(N[1])), 15, (0, 0, 255), -1)  # 蓝色填充
        cv2.putText(image, 'Center', (int(N[0])+15, int(N[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 蓝色文字
        
        # 添加计算说明 - 更大更明显
        legend_y = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        lines = [
            f'CC Distance (Left nostril): {cc_distance:.2f}',
            f'CN Distance (Right nostril): {cn_distance:.2f}',
            f'Nostril Width Ratio = {cc_distance:.2f}/{cn_distance:.2f} = {ratio:.6f}'
        ]
        colors = [(255, 255, 255), (255, 255, 255), (255, 255, 0)]
        y_offset = legend_y
        for i, (text, color) in enumerate(zip(lines, colors)):
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            rect_x1 = 8
            rect_y1 = y_offset - text_h - 4
            rect_x2 = rect_x1 + text_w + 8
            rect_y2 = y_offset + baseline + 4
            overlay = image.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, text, (rect_x1 + 4, y_offset), font, font_scale, color, font_thickness)
            y_offset += text_h + 12
        return image
    
    def process_image_file_with_nostril_lines(self, image_file):
        """处理图像文件并返回带有鼻孔辅助线的结果"""
        try:
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                img_data = image_file.read()
            else:
                with open(image_file, 'rb') as f:
                    img_data = f.read()
            
            # 将图像数据转换为numpy数组
            nparr = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                print("无法解码图像")
                return None
            
            # 转换为RGB格式
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"读取图像时出错: {e}")
            return None
        
        return self._process_image_array_with_nostril_lines(img_array)
    
    def _process_image_array_with_nostril_lines(self, img_array):
        """处理图像数组并返回带有鼻孔辅助线的结果"""
        # 预处理图像
        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # 转换为tensor并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            preds, maxvals = self._get_max_preds(output)
            
            # 缩放关键点坐标到原图尺寸
            preds = preds[0]  # 取第一个batch
            
            # 先缩放到256x256，然后缩放到原图尺寸
            preds[:, 0] *= 256.0 / 64.0  # width
            preds[:, 1] *= 256.0 / 64.0  # height
            
            # 缩放到原图尺寸
            img_h, img_w = img_array.shape[:2]
            scale_x = img_w / 256.0
            scale_y = img_h / 256.0
            
            pred_pts_scaled = preds.cpu().numpy().copy()
            pred_pts_scaled[:, 0] *= scale_x
            pred_pts_scaled[:, 1] *= scale_y
            
            # 绘制鼻孔辅助线
            result_img = self._draw_nostril_lines(img_array.copy(), pred_pts_scaled)
            base64_str = self._image_to_base64(result_img)
            
            return base64_str
    
    def _image_to_base64(self, image):
        """将图像转换为base64字符串"""
        # 如果是RGB格式，转换为BGR用于OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    def _draw_n_point_only(self, image_file, n_point):
        """only draw N point for columellar angle measurement"""
        try:
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                img_data = image_file.read()
            else:
                with open(image_file, 'rb') as f:
                    img_data = f.read()
            
            # convert image data to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                print("cannot decode image")
                return None
            
            # convert to RGB format
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # draw N point - thinner and simpler, no white outline
            x, y = int(n_point[0]), int(n_point[1])
            cv2.circle(img_array, (x, y), 12, (0, 0, 255), -1)  # blue fill, radius 8, no outline
            
            # 添加更大的文字标签
            cv2.putText(img_array, 'N (Columella Center)', (x+20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 蓝色文字
            
            # 添加说明文字 - 带半透明背景
            overlay = img_array.copy()
            cv2.rectangle(overlay, (5, 5), (img_array.shape[1]-5, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img_array, 0.3, 0, img_array)
            
            cv2.putText(img_array, 'Ready for angle measurement', 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)  # 青色文字
            
            cv2.putText(img_array, 'Use controls to set angle and direction', 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 青色文字
            
            return self._image_to_base64(img_array)
            
        except Exception as e:
            print(f"绘制N点时出错: {e}")
            return None
    
    def _get_original_image_base64(self, image_file):
        """获取原始图片的base64编码"""
        try:
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                img_data = image_file.read()
            else:
                with open(image_file, 'rb') as f:
                    img_data = f.read()
            
            # 将图像数据转换为numpy数组
            nparr = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                print("无法解码图像")
                return None
            
            # 转换为RGB格式
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            return self._image_to_base64(img_array)
            
        except Exception as e:
            print(f"获取原始图片时出错: {e}")
            return None

# 全局检测器实例
nostril_detector = None

def get_nostril_detector():
    """获取第二个模型检测器实例"""
    global nostril_detector
    if nostril_detector is None:
        nostril_detector = NostrilDetector()
    return nostril_detector

# 修改原有的process_image函数以支持多模型选择
def process_image_with_model(file_obj, model_type="alar"):
    """
    根据模型类型处理图像文件
    model_type: "alar" (第一个模型), "nostril" (第二个模型), 或 "columellar" (第三个模型)
    """
    if model_type == "alar":
        # 使用第一个模型（原有的HRNet唇裂检测）
        return process_image(file_obj)
    elif model_type == "nostril":
        # 使用第二个模型（鼻孔关键点检测）
        try:
            detector = get_nostril_detector()
            base64_str, ratio, severity, score, cc_distance, cn_distance, keypoints = detector.process_image_file(file_obj)
            
            # 生成带鼻孔辅助线的图片
            nostril_lines_base64 = detector.process_image_file_with_nostril_lines(file_obj)
            
            return base64_str, ratio, severity, score, cc_distance, cn_distance, nostril_lines_base64
        except Exception as e:
            print(f"处理第二个模型图像时出错: {str(e)}")
            return None, 0.0, "处理失败", 0, 0.0, 0.0, None
    elif model_type == "columellar":
        # 使用第三个模型（鼻柱角度，基于第二个模型获取N点）
        try:
            detector = get_nostril_detector()
            base64_str, ratio, severity, score, cc_distance, cn_distance, keypoints = detector.process_image_file(file_obj)
            
            # 提取N点（第5个关键点，索引为4）
            n_point = keypoints[4] if len(keypoints) > 4 else [0, 0]
            
            # 获取原始图片的base64
            original_image = detector._get_original_image_base64(file_obj)
            
            # 只显示N点的图片
            columellar_image = detector._draw_n_point_only(file_obj, n_point)
            
            return columellar_image, n_point, original_image
        except Exception as e:
            print(f"处理第三个模型图像时出错: {str(e)}")
            return None, [0, 0], None
    else:
        raise ValueError(f"未知的模型类型: {model_type}") 