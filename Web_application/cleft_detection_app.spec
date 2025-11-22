# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect all submodules for torch and other ML libraries
torch_modules = collect_submodules('torch')
torchvision_modules = collect_submodules('torchvision')
cv2_modules = collect_submodules('cv2')
numpy_modules = collect_submodules('numpy')
sklearn_modules = collect_submodules('sklearn')

# Collect data files
torch_data = collect_data_files('torch')
torchvision_data = collect_data_files('torchvision')

block_cipher = None

# Main application
a = Analysis(
    ['app.py'],  # Main Flask application
    pathex=[base_dir],
    binaries=[
        # 添加编译后的 encryption_utils.pyd 文件
        ('services/encryption_utils.pyd', 'services'),
    ],
    datas=[
        # Flask templates and static files
        ('templates', 'templates'),
        ('static', 'static'),
        
        # Model directories with all files
        ('services/HRNet-Facial-Landmark-Detection', 'services/HRNet-Facial-Landmark-Detection'),
        ('services/HRNet-Facial-Landmark-Detection/lib', 'lib'),
        ('services/Second_model', 'services/Second_model'),
        
        # Configuration files
        ('admin_config.json', '.'),
        ('config.py', '.'),
        
        # Instance directory (if exists)
        ('instance', 'instance'),
        
        # Add torch and torchvision data
        *torch_data,
        *torchvision_data,

        # 新增：添加加密后的模型文件 (替换原有的 .pth 文件)
        ('services/HRNet-Facial-Landmark-Detection/best_NVM_cleftlip_model_HRNet.enc', 'services/HRNet-Facial-Landmark-Detection'),
        ('services/Second_model/best_model_epoch_65_loss_0.1313_err_3.37_20250613_020703.enc', 'services/Second_model'),
    ],
    hiddenimports=[
        # Flask and related
        'flask',
        'flask_sqlalchemy',
        'flask_wtf',
        'flask_migrate',
        'wtforms',
        'werkzeug',
        'werkzeug.security',
        'jinja2',
        'markupsafe',
        
        # Database
        'sqlite3',
        'sqlalchemy',
        
        # ML libraries
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'pandas',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'matplotlib',
        'matplotlib.pyplot',
        'skimage',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        
        # Other utilities
        'json',
        'base64',
        'io',
        'datetime',
        'os',
        'sys',
        'threading',
        'zipfile',
        'shutil',
        'requests',
        
        # GUI (tkinter)
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.simpledialog',
        
        # Add collected modules
        *torch_modules,
        *torchvision_modules,
        *cv2_modules,
        *numpy_modules,
        
        # Custom modules
        'services',
        'services.ml_interface', # 重新添加 ml_interface 的 hiddenimport (因为它现在是普通 .py 文件)
        
        # Additional ML dependencies
        'yacs',
        'yacs.config',
        'timm',
        'albumentations',
        'h5py',
        'hdf5storage',
        'tqdm',
        'easydict',
        'tensorboard',
        'protobuf',
        'packaging',
        'setuptools',
        
        # HRNet lib modules
        'lib',
        'lib.config',
        'lib.config.defaults',
        'lib.models',
        'lib.models.hrnet',
        'lib.core',
        'lib.core.evaluation',
        'lib.datasets',
        'lib.datasets.cleftlip',
        'lib.datasets.aflw',
        'lib.datasets.cofw',
        'lib.datasets.face300w',
        'lib.datasets.wflw',
        'lib.utils',
        'lib.utils.transforms',
        
        # Additional hidden imports for robustness
        'pkg_resources',
        'six',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        'click',
        'itsdangerous',
        'blinker',
        
        # PyCryptodome 相关 hiddenimports 以确保完整打包 (仍需要，因为 encryption_utils 是 .pyd)
        'Crypto',
        'Crypto.Cipher',
        'Crypto.Cipher.AES',
        'Crypto.Cipher._aes',
        'Crypto.Cipher._raw_aes',
        'Crypto.Cipher._mode_cbc',
        'Crypto.Util._strxor',
        'Crypto.Util.Padding',
        'Crypto.Random',
        'Crypto.Random.OSRNG',
        'Crypto.Random.random',
        'Crypto.Random._UserFriendlyRNG',
        'Crypto.Protocol.KDF',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'torch.test',
        'PIL.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files to reduce size
def filter_datas(datas):
    filtered = []
    exclude_patterns = [
        'test',
        'tests',
        '__pycache__',
        '.pyc',
        '.git',
        '.gitignore',
        'docs',
        'examples',
        'tutorial',
    ]
    
    for dest, source, kind in datas:
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in dest.lower() or pattern in source.lower():
                should_exclude = True
                break
        if not should_exclude:
            filtered.append((dest, source, kind))
    
    return filtered

a.datas = filter_datas(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Main Flask application
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CleftDetectionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Launcher GUI
launcher_a = Analysis(
    ['launcher_gui.py'],
    pathex=[base_dir],
    binaries=[],
    datas=[],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

launcher_pyz = PYZ(launcher_a.pure, launcher_a.zipped_data, cipher=block_cipher)

launcher_exe = EXE(
    launcher_pyz,
    launcher_a.scripts,
    [],
    exclude_binaries=True,
    name='LauncherGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for launcher
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Database Manager GUI
db_manager_a = Analysis(
    ['gui_db_manager.py'],
    pathex=[base_dir],
    binaries=[],
    datas=[],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

db_manager_pyz = PYZ(db_manager_a.pure, db_manager_a.zipped_data, cipher=block_cipher)

db_manager_exe = EXE(
    db_manager_pyz,
    db_manager_a.scripts,
    [],
    exclude_binaries=True,
    name='DBManager',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Keep console for database manager
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Collect all executables and their dependencies
coll = COLLECT(
    exe,
    launcher_exe,
    db_manager_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    launcher_a.binaries,
    launcher_a.zipfiles,
    launcher_a.datas,
    db_manager_a.binaries,
    db_manager_a.zipfiles,
    db_manager_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CleftDetectionApp',
) 