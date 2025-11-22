import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import re

# Load HRNet model and config
from lib.config.defaults import _C as cfg
from lib.config.defaults import update_config
from lib.models.hrnet import get_face_alignment_net
from lib.core.evaluation import get_preds, compute_nme
from lib.datasets.cleftlip import CleftLip

# === Set paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "experiments/cleft_lip/pose_hrnet_w18_cleft.yaml")
model_path = os.path.join(BASE_DIR, "best_NVM_cleftlip_model_HRNet.pth")
split_csv = os.path.join(BASE_DIR, "..", "唇裂项目", "数据集划分.csv")
img_dir = os.path.join(BASE_DIR, "..", "唇裂项目", "处理后图片")
output_dir = os.path.join(BASE_DIR, "validation_results")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Only use the specified csv and image directory (absolute path)
DOCTOR_CSV = r'D:/upm_model train/project-6-front_view.csv'
IMG_DIR = r'D:/upm_model train/唇裂项目/test_front'

# Read doctor annotation CSV
try:
    doctor_df = pd.read_csv(DOCTOR_CSV, encoding='utf-8')
except Exception:
    doctor_df = pd.read_csv(DOCTOR_CSV, encoding='gbk')

def extract_id_from_filename(filename):
    # Match the middle segment of 8 or more characters as ID
    match = re.search(r'-([a-zA-Z0-9]{8,})-', filename)
    if match:
        return match.group(1)
    return None

def get_main_name(filename):
    return os.path.splitext(filename)[0].lower()

# 1. Build mapping from ID to keypoints
doctor_id_kpt_dict = {}
for fname, group in doctor_df.groupby('原始文件名'):
    id_ = extract_id_from_filename(fname)
    if id_:
        kpts = np.zeros((6, 2), dtype=np.float32)
        for idx, kpt_name in enumerate(['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']):
            row = group[group['关键点'] == kpt_name]
            if not row.empty:
                kpts[idx, 0] = float(row['归一化X'].iloc[0])
                kpts[idx, 1] = float(row['归一化Y'].iloc[0])
        doctor_id_kpt_dict[id_] = kpts

# Build mapping from main name to csv group
csv_mainname2group = {}
for fname, group in doctor_df.groupby('原始文件名'):
    main_name = get_main_name(fname)
    csv_mainname2group[main_name] = group

def ab_severity(ratio):
    if ratio <= 0.05:
        return 'Mild'
    elif ratio <= 0.10:
        return 'Moderate'
    else:
        return 'Severe'

class ValidationDataset(Dataset):
    def __init__(self, split_df, img_dir, split_type='验证集'):
        self.img_dir = img_dir
        self.split_df = split_df[split_df['数据集类型'] == split_type]
        self.file_list = self.split_df['文件名'].tolist()
        
        # Load annotation data
        anno_path = os.path.join(os.path.dirname(img_dir), '唇裂标注分析结果_512px.csv')
        try:
            self.anno_df = pd.read_csv(anno_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.anno_df = pd.read_csv(anno_path, encoding='gbk')
        
        # Create mapping from filename to keypoints
        self.keypoints_dict = {}
        for filename in self.file_list:
            # Get all keypoints for the current file
            file_annos = self.anno_df[self.anno_df['原始文件名'] == filename]
            if len(file_annos) > 0:
                keypoints = np.zeros((6, 2), dtype=np.float32)  # 6 keypoints, each with xy coordinates
                for idx, kpt_name in enumerate(['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']):
                    kpt_data = file_annos[file_annos['关键点'] == kpt_name]
                    if len(kpt_data) > 0:
                        keypoints[idx] = [
                            float(kpt_data['归一化X'].iloc[0]),
                            float(kpt_data['归一化Y'].iloc[0])
                        ]
                self.keypoints_dict[filename] = keypoints
        
        # Create the same transforms as in training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cache all files in the directory
        print("Scanning image directory...")
        try:
            self.available_files = os.listdir(self.img_dir)
            print(f"There are {len(self.available_files)} files in the directory")
            print("Example filenames:", self.available_files[:5])
        except Exception as e:
            print(f"Error scanning directory: {str(e)}")
            print(f"Directory path: {self.img_dir}")
            raise

    def __len__(self):
        return len(self.file_list)

    def find_matching_file(self, target_filename):
        """Find matching filename"""
        try:
            # 1. Direct match
            if target_filename in self.available_files:
                return target_filename
                
            # 2. Try to extract ID part for matching
            parts = target_filename.split('-')
            if len(parts) > 1:
                id_part = parts[1].split('_')[0]
                for f in self.available_files:
                    if id_part in f:
                        print(f"Found matching file by ID {id_part}: {f}")
                        return f
            
            # 3. Try to match by number part
            number_part = target_filename.split('_')[-1].replace('.jpg', '').replace('.JPG', '')
            for f in self.available_files:
                if f'_{number_part}.' in f:
                    print(f"Found matching file by number {number_part}: {f}")
                    return f
            
            return None
        except Exception as e:
            print(f"Error finding file: {str(e)}")
            print(f"Target filename: {target_filename}")
            print(f"Available files: {self.available_files[:5]}")
            raise

    def __getitem__(self, idx):
        try:
            filename = self.file_list[idx]
            print(f"\nProcessing file: {filename}")
            
            # Try to find matching file
            matching_file = self.find_matching_file(filename)
            if matching_file is None:
                print(f"Warning: Cannot find matching file: {filename}")
                print("Example available files:", self.available_files[:5])
                raise ValueError(f"Cannot find matching file: {filename}")
            
            # Build full path and ensure correct path separator
            img_path = os.path.normpath(os.path.join(self.img_dir, matching_file))
            print(f"Trying to read: {img_path}")
            
            # Open image with PIL
            try:
                img_pil = Image.open(img_path)
                img = np.array(img_pil)
                if len(img.shape) == 2:  # If grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # If RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            except Exception as e:
                print(f"Failed to read image with PIL: {str(e)}")
                print("Trying to read with OpenCV...")
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Cannot read image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save original image size
            orig_h, orig_w = img.shape[:2]
            
            # Convert to PIL image and apply transform
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil)
            
            # Get keypoint annotation
            gt_pts = self.keypoints_dict.get(filename, np.zeros((6, 2), dtype=np.float32))
            
            # Convert normalized coordinates to pixel coordinates
            gt_pts_pixels = gt_pts.copy()
            gt_pts_pixels[:, 0] *= orig_w  # X coordinate
            gt_pts_pixels[:, 1] *= orig_h  # Y coordinate
            
            return {
                'image': img_tensor,
                'orig_size': (orig_h, orig_w),
                'filename': filename,
                'orig_img': img,
                'gt_pts': torch.from_numpy(gt_pts_pixels)  # Add ground truth keypoints (pixel coordinates)
            }
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(f"File index: {idx}")
            print(f"Filename: {filename if 'filename' in locals() else 'unknown'}")
            raise

def calculate_ab_ratio(keypoints):
    """Calculate A/B ratio
    
    A: Absolute Y distance between Nleft and Nright
    B: Y distance from the midpoint of the line connecting the eye center and inner mouth corner center to the lower nasal wing point
    """
    # Ensure keypoints is a numpy array
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    
    # Get keypoint coordinates
    E1, E2 = keypoints[0], keypoints[1]  # Eye keypoints
    I1, I2 = keypoints[2], keypoints[3]  # Inner mouth corner keypoints
    N_left, N_right = keypoints[4], keypoints[5]  # Nasal wing keypoints
    
    # Calculate A: absolute Y distance between Nleft and Nright
    A = abs(N_left[1] - N_right[1])
    
    # Calculate B:
    # 1. Calculate eye center
    eye_center = (E1 + E2) / 2
    # 2. Calculate inner mouth corner center
    mouth_center = (I1 + I2) / 2
    # 3. Calculate midpoint of the line between the two centers
    center_line_midpoint = (eye_center + mouth_center) / 2
    
    # 4. Find the lower nasal wing point (the one with larger Y value)
    lower_nose_y = max(N_left[1], N_right[1])
    
    # 5. Calculate B: Y distance from midpoint to lower nasal wing point
    B = abs(lower_nose_y - center_line_midpoint[1])
    
    # Calculate ratio
    ratio = A / B if B > 0 else 0
    
    return ratio, A, B

def visualize_with_ratio(image, predictions, save_path, doctor_pts=None):
    """Custom visualization: show A/B ratio and plot both predicted and doctor keypoints if available"""
    ratio, A, B = calculate_ab_ratio(predictions)
    if ratio <= 0.05:
        severity = "Mild"
    elif ratio <= 0.10:
        severity = "Moderate"
    else:
        severity = "Severe"
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    keypoint_names = ['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']
    colors = ['r', 'r', 'g', 'g', 'b', 'b']
    # Plot predicted keypoints
    for i, (kp, name, color) in enumerate(zip(predictions, keypoint_names, colors)):
        x, y = int(kp[0]), int(kp[1])
        plt.scatter(x, y, c=color, s=100, marker='o', alpha=0.7)
        plt.text(x+5, y+5, name, color=color, fontsize=12, weight='bold')
    # Plot doctor keypoints if available
    if doctor_pts is not None:
        for i, (kp, name) in enumerate(zip(doctor_pts, keypoint_names)):
            x, y = int(kp[0]), int(kp[1])
            plt.scatter(x, y, c='cyan', s=60, marker='x', alpha=0.7, label='Doctor' if i==0 else "")
    plt.scatter([], [], c='r', s=100, marker='o', alpha=0.7, label='Pred E1,E2')
    plt.scatter([], [], c='g', s=100, marker='o', alpha=0.7, label='Pred I1,I2')
    plt.scatter([], [], c='b', s=100, marker='o', alpha=0.7, label='Pred N_left,N_right')
    if doctor_pts is not None:
        plt.scatter([], [], c='cyan', s=60, marker='x', alpha=0.7, label='Doctor Keypoints')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"A/B ratio: {ratio:.3f} ({severity})\nA: {A:.1f}, B: {B:.1f}", fontsize=14, pad=20)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return ratio, A, B, severity

def calculate_nme(pred_pts, gt_pts):
    """Calculate NME for a single sample
    
    Args:
        pred_pts: predicted keypoint coordinates (6, 2)
        gt_pts: ground truth keypoint coordinates (6, 2)
    """
    # Calculate interocular distance (E1-E2) as normalization factor
    eye_distance = np.linalg.norm(gt_pts[0] - gt_pts[1])
    
    # If interocular distance is too small, use image diagonal length
    if eye_distance < 1e-6:
        eye_distance = np.sqrt(1.0**2 + 1.0**2)  # Diagonal length in normalized coordinate system
    
    # Calculate Euclidean distance for each keypoint
    distances = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=1))
    
    # Calculate mean error and normalize
    nme = np.mean(distances) / eye_distance
    
    return nme

def find_csv_keypoints_by_filename(filename, doctor_df):
    # First try exact match
    group = doctor_df[doctor_df['原始文件名'] == filename]
    if len(group) == 6:
        return group
    # Then try fuzzy match by ID
    import re
    match = re.search(r'-([a-zA-Z0-9]{8,})-', filename)
    if match:
        id_ = match.group(1)
        for fname in doctor_df['原始文件名'].unique():
            if id_ in fname:
                group = doctor_df[doctor_df['原始文件名'] == fname]
                if len(group) == 6:
                    return group
    return None

def get_id_candidates(filename):
    # Extract all segments of 8 or more characters as ID
    return re.findall(r'([a-zA-Z0-9]{8,})', filename)

def get_preds_with_subpixel(heatmaps):
    # heatmaps: (B, J, H, W)
    import torch
    assert heatmaps.dim() == 4
    B, J, H, W = heatmaps.shape
    from lib.core.evaluation import get_preds
    preds = get_preds(heatmaps)  # (B, J, 2) ints
    preds = preds - 1  # 0-based
    preds = preds.float()
    for b in range(B):
        for j in range(J):
            px, py = int(preds[b, j, 0]), int(preds[b, j, 1])
            if 1 < px < W-2 and 1 < py < H-2:
                dx = (heatmaps[b, j, py, px+1] - heatmaps[b, j, py, px-1]) / 2
                dy = (heatmaps[b, j, py+1, px] - heatmaps[b, j, py-1, px]) / 2
                preds[b, j, 0] += dx.item() * 0.25
                preds[b, j, 1] += dy.item() * 0.25
    return preds

def main():
    # Update config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    args = parser.parse_args()
    update_config(cfg, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_face_alignment_net(cfg)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    compare_results = []
    total = 0
    correct = 0
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in img_list:
        img_path = os.path.join(IMG_DIR, filename)
        main_name = get_main_name(filename)
        group = csv_mainname2group.get(main_name, None)
        # If not found, try ID fuzzy match
        if group is None or len(group) != 6:
            id_candidates = get_id_candidates(filename)
            for id_ in id_candidates:
                for csv_name, g in csv_mainname2group.items():
                    if id_ in csv_name and len(g) == 6:
                        group = g
                        break
                if group is not None and len(group) == 6:
                    break
        doctor_pts = None
        if group is not None and len(group) == 6:
            kpts = np.zeros((6, 2), dtype=np.float32)
            for idx, kpt_name in enumerate(['E1', 'E2', 'I1', 'I2', 'N_left', 'N_right']):
                row = group[group['关键点'] == kpt_name]
                if not row.empty:
                    kpts[idx, 0] = float(row['归一化X'].iloc[0])
                    kpts[idx, 1] = float(row['归一化Y'].iloc[0])
            img_pil = Image.open(img_path)
            img = np.array(img_pil)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            orig_h, orig_w = img.shape[:2]
            doc_kpt = kpts.copy()
            doc_kpt[:, 0] *= orig_w
            doc_kpt[:, 1] *= orig_h
            doctor_pts = doc_kpt
            ratio_doc, A_doc, B_doc = calculate_ab_ratio(doc_kpt)
            severity_doc = ab_severity(ratio_doc)
        else:
            ratio_doc, A_doc, B_doc, severity_doc = '', '', '', ''
            img_pil = Image.open(img_path)
            img = np.array(img_pil)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            orig_h, orig_w = img.shape[:2]
        # 2. Prediction
        img_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            heatmaps = output.cpu()
            preds = get_preds_with_subpixel(heatmaps)
            preds[:, :, 0] *= orig_w / 64
            preds[:, :, 1] *= orig_h / 64
            pred_pts = preds[0].numpy()
        # Debug print
        print(f"==== {filename} ====")
        print(f"Predicted keypoints (pixels): {pred_pts}")
        if doctor_pts is not None:
            print(f"Doctor keypoints (pixels): {doctor_pts}")
        else:
            print("Doctor keypoints: None")
        ratio_pred, A_pred, B_pred = calculate_ab_ratio(pred_pts)
        severity_pred = ab_severity(ratio_pred)
        is_same = (severity_pred == severity_doc)
        if severity_doc != '':
            total += 1
            if is_same:
                correct += 1
        # Visualization
        save_path = os.path.join(output_dir, f"val_{os.path.splitext(filename)[0]}.jpg")
        visualize_with_ratio(img, pred_pts, save_path, doctor_pts=doctor_pts)
        # Results
        compare_results.append({
            '图片名': filename,
            '医生A/B比值': ratio_doc,
            '医生等级': severity_doc,
            '预测A/B比值': ratio_pred,
            '预测等级': severity_pred,
            '是否一致': is_same
        })
    # Output detailed comparison table
    compare_df = pd.DataFrame(compare_results)
    compare_df.to_csv(os.path.join(output_dir, 'doctor_vs_pred.csv'), index=False, encoding='utf-8')
    print(f"Number of consistent grading samples: {correct}/{total}, grading accuracy: {correct/total if total>0 else 0:.2%}")
    print(f"Detailed comparison table saved to: {os.path.join(output_dir, 'doctor_vs_pred.csv')}")

if __name__ == "__main__":
    main() 