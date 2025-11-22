# UCL-Severity-Analyzer: Automated Classification of Unilateral Cleft Lip Nose Deformities

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![Deep Learning](https://img.shields.io/badge/Model-HRNet-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Introduction

This project is a web-based clinical decision-support tool designed to automate the facial recognition and severity classification of **Unilateral Cleft Lip/Nose (UCL/N) Deformities**.

The system utilizes Deep Learning (HRNet) to detect facial landmarks and calculate objective severity indicators (Alar Facial Symmetry, Nostril Width Ratio, and Columellar Angle), replacing traditional subjective manual measurements.

---

<img width="889" height="874" alt="微信图片_20251123015110_37_218" src="https://github.com/user-attachments/assets/640d9372-1889-4561-8f50-fa598a358786" />

## 🔒 Data Privacy & Ethics Statement

The dataset used for training this model consists of clinical images collected from **Hospital Universiti Putra Malaysia (HUPM)**.

**Due to strict patient confidentiality agreements and ethical guidelines:**

1.  **Raw Facial Images are NOT Included:** The original dataset containing patient faces has been excluded from this repository to protect patient privacy.
2.  **Visual Test Results are Excluded:** Generated images with plotted landmarks on patient faces are also excluded.
3.  **Available Data:** Only **anonymized annotation data** (CSV/JSON coordinates), **evaluation metrics**, and **pre-trained model weights** are provided. These allow researchers to verify the data structure and reproduce the training pipeline using their own datasets.

---

## 📥 Setup & Installation (Crucial Step)
1. Clone the Repository
```bash
git clone [https://github.com/Jarync/UCL-Severity-Analyzer.git](https://github.com/Jarync/UCL-Severity-Analyzer.git)
cd UCL-Severity-Analyzer
```

2. Download Model Weights (Required)
Due to GitHub's file size limits, the pre-trained model weights (.pth) and encrypted modules (.enc) are stored in the Releases section.

### best_NVM_cleftlip_model_HRNet.pth to Model_trainning/First_model/HRNet-Facial-Landmark-Detection/
### best_NVM_cleftlip_model_HRNet.pth to Web_application/services/HRNet-Facial-Landmark-Detection/
### best_NVM_cleftlip_model_HRNet.enc to Web_application/services/HRNet-Facial-Landmark-Detection/

---

## 📥 Downloads & Releases

All compiled applications and large model files are hosted in the [**Releases Section**](../../releases).

### 1. 🏥 For Clinicians: Portable Windows App (No Installation)
* **Release Tag:** `v1.0 - Official Portable Release`
* **File:** `CleftDetectionApp.zip`
* **Description:** A fully standalone version of the software.
    * ✅ **No Python required:** Comes with its own environment.
    * ✅ **Plug & Play:** Just unzip and run `CleftDetectionApp.exe`.
    * ✅ **Full Features:** Includes Web App, GUI Launcher, and Database Manager.

### 2. 👨‍💻 For Developers: Model Weights
* **Release Tag:** `Model Weights`
* **Files:** `best_NVM_cleftlip_model_HRNet.pth`, `best_NVM_cleftlip_model_HRNet.enc`
* **Description:** Due to GitHub's file size limits, the deep learning model weights are stored here.
    * ⚠️ **Crucial:** If you are running the source code (`python app.py`), you **MUST** download these files manually and place them in the `model_training` and `services` folders respectively.

--- 


## 🧠 Methodology & Model Training Pipeline

The system uses two distinct models. Below is the step-by-step pipeline to reproduce the training process.

### Model 1: Front View (Alar Facial Symmetry)
*Base Architecture: HRNet-W18 (Pre-trained on WFLW)*
*Citation: [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)*
<img width="712" height="611" alt="model1_flowchart_NEW drawio (1)" src="https://github.com/user-attachments/assets/caa3664e-e680-4abc-80da-9b77510ec0b2" />

1.  **Preprocessing:**
    * Run `Original Image Processing - Step 1.py`.
    * *Function:* Renames images based on unique IDs and resizes them to a uniform **512x512** resolution.
2.  **Annotation:**
    * Annotate images using **Label Studio**.
    * Export format: JSON (`project-1.json`).
3.  **Data Conversion:**
    * Run `josn_transfer_set 2.py`.
    * *Function:* Converts Label Studio JSON output to CSV format (`唇裂标注分析结果_512px.csv`).
4.  **Dataset Splitting:**
    * Run `data splitting_step 3.py`.
    * *Function:* Splits data into Train/Val/Test (70:15:15) -> `数据集划分.csv`.
5.  **Training:**
    * Navigate to the `HRNet-Facial-Landmark-Detection` directory.
    * Ensure the pre-trained weights (`HR18-WFLW.pth`) are placed in the root or specified folder.
    * **Run the training command:**
        ```bash
        python -X utf8 tools/train.py --cfg experiments/cleft_lip/pose_hrnet_w18_cleft.yaml
        ```
    * This process will generate the `best_NVM_cleftlip_model_HRNet.pth` weights.
6.  **Testing (Doctor's Benchmark):**
    * Run `json_to_csv_front_view-step test set.py` and then run `validate_en.py`.
    * *Function:* Specifically processes the "Doctor's Gold Standard" annotations for performance evaluation.

### Model 2: Columellar/Angle View
*Base Architecture: Custom HRNet Training (Trained from scratch)*
<img width="686" height="671" alt="model2_flowchart_NEW drawio (1)" src="https://github.com/user-attachments/assets/d15a9322-dce5-4afd-b963-79b233c6d826" />

1.  **Preprocessing:**
    * Run `Original Image Processing - Step 1.py` (Same as Model 1).
2.  **Annotation:**
    * Due to the dataset size, annotations were performed in batches (`anno1`, `anno2`, `anno3` JSON files).
3.  **Data Fusion & Conversion:**
    * Run `josn_transfer_second_model-second step2.py`.
    * *Function:* Merges the three partial JSON files and converts them into a single CSV (`第二模型标注结果_512px.csv`).
4.  **Dataset Splitting:**
    * Run `split_dataset_points-step 3.py`.
5.  **Training:**
    * Run `model-tranning.py`.
    * *Function:* Trains the model and generates `best_model_weight...pth`.
6.  **Testing (Doctor's Benchmark):**
    * Run `json_to_csv_angle_view for testing set.py` and then run `testing set result.py`.
    * *Note:* This script handles the specific geometric properties of the columellar angle view differently from the front view.

---

## 🚀 Web Application Usage

### Prerequisites
* Python 3.8+
* Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App (Standard Method)
1.  Navigate to the `application` directory.
2.  Run the main Flask app:
    ```bash
    python app.py
    ```
3.  Open your browser and go to `http://localhost:5002`.
    * *Note:* `config.py` automatically handles path resolution for both development environments and PyInstaller frozen executables. `setup.py` handles Cython compilation for encryption modules.

---

## 🛠️ Administrative Tools

Beyond the standard web interface, this project includes dedicated GUI tools for system administration and easy deployment.

### 1. GUI Launcher (`launcher_gui.py`)
A graphical interface designed for non-technical users (e.g., clinicians) to launch the application without using the command line. It performs pre-flight checks on the environment before starting the server.
* **How to run:**
    ```bash
    python launcher_gui.py
    ```

### 2. Database Manager (`gui_db_manager.py`)
A dedicated dashboard for administrators to manage the SQLite database without writing SQL queries.
* **Key Features:**
    * **User Management:** View, edit, and delete user accounts.
    * **Role Assignment:** Toggle roles between `Doctor`, `Patient`, and `Admin`.
    * **Security Control:** Reset passwords and manage access permissions.
* **How to run:**
    ```bash
    python gui_db_manager.py
    ```
* **Default Credentials:** The system initializes with `admin` / `admin`. **Please change this password immediately upon first login.**

---

## 📂 Project Structure

```text
UCL-Severity-Analyzer/
├── application/           # Main Web App Source Code
│   ├── app.py             # Flask Entry Point
│   ├── config.py          # Path & Environment Configuration
│   ├── launcher_gui.py    # GUI Launcher Tool
│   ├── gui_db_manager.py  # Admin DB Management Tool
│   ├── templates/         # HTML Frontend
│   ├── static/            # CSS/JS/Assets
│   └── services/          # Machine Learning Inference Logic
├── model_training/        # Research & Training Scripts
│   ├── First_model/       # Front View Model (HRNet-W18)
│   │   ├── HRNet-Facial-Landmark-Detection/  # HRNet Core
│   │   ├── josn_transfer_step 2.py            # Data Conversion
│   │   └── ...
│   └── Second_model/      # Angle View Model
│       ├── model-tranning.py                 # Training Script
│       ├── split_dataset_points-step 3.py    # Data Splitting
│       └── ...
└── README.md
```

## ©️ Intellectual Property & License

This project is the intellectual property of **Chen Junxu** and **Universiti Putra Malaysia (UPM)**.

**⚠️ Usage Policy:**
1.  **Academic & Research Use Only:** The source code, models, and methodologies provided in this repository are strictly for non-commercial, educational, and research purposes.
2.  **No Commercial Use:** Any use for commercial gain, including but not limited to paid services, commercial software integration, or private clinical deployment, is **strictly prohibited** without prior written permission from the author and the institution.
3.  **Citation Required:** If you use any part of this project (code, dataset logic, or model weights) in your research or work, you **must** acknowledge the source by citing this repository.

**Please cite this work as follows:**

**BibTeX:**
```bibtex
@misc{chen2025ucl,
  author = {Chen, Junxu},
  title = {Automated Facial Recognition and Severity Classification of Unilateral Cleft Lip Nose Deformities},
  year = {2025},
  publisher = {GitHub},
  journal = {Universiti Putra Malaysia (UPM) Final Year Project},
  howpublished = {\url{[https://github.com/Jarync/UCL-Severity-Analyzer](https://github.com/Jarync/UCL-Severity-Analyzer)}}
}
```
### Text Format:

Chen, J. (2025). Automated Facial Recognition and Severity Classification of Unilateral Cleft Lip Nose Deformities [Source Code]. Universiti Putra Malaysia. Available at: https://www.google.com/url?sa=E&source=gmail&q=https://github.com/Jarync/UCL-Severity-Analyzer.
