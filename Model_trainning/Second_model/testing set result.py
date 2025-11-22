import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ========== Grading function ==========
def get_nostril_severity(ratio):
    if 0.90 <= ratio <= 1.10:
        return "Mild", 3
    elif 0.60 <= ratio < 0.90 or 1.10 < ratio <= 1.40:
        return "Moderate", 2
    elif ratio < 0.60 or ratio > 1.40:
        return "Severe", 1
    else:
        return "Unknown", 0

# ========== Calculate ratio ==========
def calc_nostril_ratio(pts):
    # pts: 5x2, order: CC1, CC2, CN1, CN2, N
    CC1, CC2, CN1, CN2, N = pts
    cc_dist = np.linalg.norm(np.array(CC1) - np.array(CC2))
    cn_dist = np.linalg.norm(np.array(CN1) - np.array(CN2))
    ratio = cc_dist / cn_dist if cn_dist > 0 else 0
    return ratio, cc_dist, cn_dist

# ========== Drawing function ==========
def draw_compare(image, pred_pts, doc_pts, pred_grade, doc_grade):
    keypoint_names = ['CC1', 'CC2', 'CN1', 'CN2', 'N']
    colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
    img = image.copy()
    # Draw doctor points (X, cyan, markerSize=8, thickness=1)
    for i, (kp, name) in enumerate(zip(doc_pts, keypoint_names)):
        x, y = int(kp[0]), int(kp[1])
        cv2.drawMarker(img, (x, y), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1)
        cv2.putText(img, name, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    # Draw predicted points (circle, colored)
    for i, (kp, name, color) in enumerate(zip(pred_pts, keypoint_names, colors)):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), 3, color, -1)
        cv2.putText(img, name, (x-18, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    # Write grading
    txt = f"Pred: {pred_grade}  |  Doctor: {doc_grade}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return img

# ========== Model structure ==========
import torch.nn as nn
class HRNet(nn.Module):
    def __init__(self, num_joints):
        super(HRNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_joints, kernel_size=1)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x2_up = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2_up], dim=1)
        x = self.fuse(x)
        x = self.final_layer(x)
        return x

def get_max_preds(heatmaps):
    assert isinstance(heatmaps, torch.Tensor)
    batch_size = heatmaps.size(0)
    num_joints = heatmaps.size(1)
    width = heatmaps.size(3)
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    preds = torch.zeros((batch_size, num_joints, 2), device=heatmaps.device)
    preds[:, :, 0] = (idx % width).float()
    preds[:, :, 1] = (idx // width).float()
    return preds

# ========== Main process ==========
def main():
    # Paths
    csv_path = 'project-7angleview.csv'
    img_dir = 'Second_model/test_angle'
    out_dir = os.path.join(img_dir, 'vis_compare')
    os.makedirs(out_dir, exist_ok=True)
    weight_path = r'D:\upm_model train\Second_model\best_model_epoch_65_loss_0.1313_err_3.37_20250613_020703.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Read doctor annotation
    df = pd.read_csv(csv_path)
    # Build mapping from image name to 5 points
    doc_dict = {}
    for fname, group in df.groupby('原始文件名'):
        pts = np.zeros((5,2), dtype=np.float32)
        for i, name in enumerate(['CC1','CC2','CN1','CN2','N']):
            row = group[group['关键点']==name]
            if not row.empty:
                pts[i,0] = float(row['像素X'].iloc[0])
                pts[i,1] = float(row['像素Y'].iloc[0])
        doc_dict[fname] = pts
    # Load model
    model = HRNet(num_joints=5)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Statistics
    total, correct = 0, 0
    results = []
    y_true, y_pred = [], []
    y_true_cls, y_pred_cls = [], []
    all_point_errors = []
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for img_name in tqdm(img_list, desc='Comparing'):
        img_path = os.path.join(img_dir, img_name)
        # Match doctor annotation
        doc_pts = None
        for key in doc_dict:
            if key.lower() == img_name.lower():
                doc_pts = doc_dict[key]
                break
        if doc_pts is None:
            # Try ignoring case and partial prefix
            for key in doc_dict:
                if img_name.lower() in key.lower() or key.lower() in img_name.lower():
                    doc_pts = doc_dict[key]
                    break
        if doc_pts is None:
            print(f'No doctor annotation found: {img_name}')
            continue
        # Read image
        img = Image.open(img_path).convert('RGB')
        orig_img = np.array(img)
        h0, w0 = orig_img.shape[:2]
        # Predict
        img_resized = img.resize((256,256), Image.Resampling.LANCZOS)
        input_tensor = transform(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            preds = get_max_preds(output)[0].cpu().numpy()
            preds[:,0] *= (w0/64)
            preds[:,1] *= (h0/64)
        # Calculate grading
        pred_ratio, pred_cc, pred_cn = calc_nostril_ratio(preds)
        doc_ratio, doc_cc, doc_cn = calc_nostril_ratio(doc_pts)
        pred_grade, pred_score = get_nostril_severity(pred_ratio)
        doc_grade, doc_score = get_nostril_severity(doc_ratio)
        is_same = (pred_grade == doc_grade)
        total += 1
        if is_same:
            correct += 1
        # Calculate Euclidean distance error for each point
        point_errors = np.linalg.norm(preds - doc_pts, axis=1)
        mean_point_error = np.mean(point_errors)
        all_point_errors.append(point_errors)
        # Save visualization
        vis_img = draw_compare(orig_img, preds, doc_pts, pred_grade, doc_grade)
        save_path = os.path.join(out_dir, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        # Record results
        results.append({
            'image_name': img_name,
            'doctor_ratio': doc_ratio,
            'doctor_grade': doc_grade,
            'pred_ratio': pred_ratio,
            'pred_grade': pred_grade,
            'is_same': is_same,
            'CC1_error': point_errors[0],
            'CC2_error': point_errors[1],
            'CN1_error': point_errors[2],
            'CN2_error': point_errors[3],
            'N_error': point_errors[4],
            'mean_point_error': mean_point_error
        })
        y_true.append(doc_ratio)
        y_pred.append(pred_ratio)
        y_true_cls.append(doc_grade)
        y_pred_cls.append(pred_grade)
    # Output csv
    out_csv = os.path.join(out_dir, 'compare_results.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'Number of matched severity samples: {correct}/{total}, Accuracy: {correct/total if total>0 else 0:.2%}')
    print(f'Detailed comparison table saved to: {out_csv}')
    # ========== Evaluation Metrics ========== 
    # Filter out invalid values
    y_true_valid = [v for v in y_true if isinstance(v, (float, np.floating)) and not np.isnan(v)]
    y_pred_valid = [v for v in y_pred if isinstance(v, (float, np.floating)) and not np.isnan(v)]
    if len(y_true_valid) == len(y_pred_valid) and len(y_true_valid) > 0:
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        r2 = r2_score(y_true_valid, y_pred_valid)
        print(f'Nostril width ratio MAE: {mae:.4f}')
        print(f'Nostril width ratio RMSE: {rmse:.4f}')
        print(f'Nostril width ratio R²: {r2:.4f}')
    # Classification evaluation
    y_true_cls_valid = [v for v in y_true_cls if v in ['Mild','Moderate','Severe']]
    y_pred_cls_valid = [v for v in y_pred_cls if v in ['Mild','Moderate','Severe']]
    if len(y_true_cls_valid) == len(y_pred_cls_valid) and len(y_true_cls_valid) > 0:
        print(classification_report(y_true_cls_valid, y_pred_cls_valid, digits=3))
        cm = confusion_matrix(y_true_cls_valid, y_pred_cls_valid, labels=['Mild','Moderate','Severe'])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['Mild','Moderate','Severe'], yticklabels=['Mild','Moderate','Severe'])
        plt.title("Confusion Matrix (Severity)")
        plt.xlabel("Predicted"); plt.ylabel("Doctor")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
        plt.close()
        print(f'Confusion matrix saved to: {os.path.join(out_dir, "confusion_matrix.png")})')
    # Additionally output mean point error statistics for all images
    all_point_errors_arr = np.array(all_point_errors)
    if all_point_errors_arr.shape[0] > 0:
        mean_per_point = np.mean(all_point_errors_arr, axis=0)
        mean_all = np.mean(all_point_errors_arr)
        print('Mean Euclidean error per keypoint:')
        for i, name in enumerate(['CC1','CC2','CN1','CN2','N']):
            print(f'  {name}: {mean_per_point[i]:.2f} pixels')
        print(f'Overall mean Euclidean error: {mean_all:.2f} pixels')
    # Save evaluation metrics to csv
    metrics_csv = os.path.join(out_dir, 'evaluation_metrics.csv')
    metrics = {
        'mae': [mae if "mae" in locals() else None],
        'rmse': [rmse if "rmse" in locals() else None],
        'r2': [r2 if "r2" in locals() else None],
        'mean_CC1_error': [mean_per_point[0] if "mean_per_point" in locals() else None],
        'mean_CC2_error': [mean_per_point[1] if "mean_per_point" in locals() else None],
        'mean_CN1_error': [mean_per_point[2] if "mean_per_point" in locals() else None],
        'mean_CN2_error': [mean_per_point[3] if "mean_per_point" in locals() else None],
        'mean_N_error': [mean_per_point[4] if "mean_per_point" in locals() else None],
        'overall_mean_point_error': [mean_all if "mean_all" in locals() else None],
        'accuracy': [correct/total if total>0 else None]
    }
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    print(f'All evaluation metrics saved to: {metrics_csv}')

if __name__ == '__main__':
    main() 