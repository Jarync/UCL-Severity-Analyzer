import os
import sys
from pathlib import Path

def get_base_path():
    """
    Get the base path for the application.
    When running as a PyInstaller bundle, sys._MEIPASS contains the path to the bundle.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as a bundle
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent.absolute()

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for both development and PyInstaller bundle.
    """
    base_path = get_base_path()
    return base_path / relative_path

def get_instance_path():
    """
    Get the path for the instance directory (database, etc.)
    This should be in a writable location when running as .exe
    """
    if getattr(sys, 'frozen', False):
        # Running as executable - use a directory next to the exe
        exe_dir = Path(sys.executable).parent
        instance_dir = exe_dir / 'instance'
    else:
        # Running in development
        instance_dir = Path(__file__).parent / 'instance'
    
    # Create directory if it doesn't exist
    instance_dir.mkdir(exist_ok=True)
    return instance_dir

# Application configuration
class Config:
    # Base paths
    BASE_DIR = get_base_path()
    INSTANCE_PATH = get_instance_path()
    
    # Secret key
    SECRET_KEY = 'replace-with-a-strong-secret-key'
    
    # Database configuration
    DATABASE_PATH = INSTANCE_PATH / 'database.db'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = INSTANCE_PATH / 'uploads'
    
    # Model paths
    HRNET_MODEL_PATH = get_resource_path('services/HRNet-Facial-Landmark-Detection')
    SECOND_MODEL_PATH = get_resource_path('services/Second_model')
    
    # Static and template paths
    STATIC_FOLDER = get_resource_path('static')
    TEMPLATE_FOLDER = get_resource_path('templates')
    
    # Admin configuration
    ADMIN_CONFIG_PATH = get_resource_path('admin_config.json')
    
    # Engine options for database
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'connect_args': {'check_same_thread': False}
    }

def setup_directories():
    """
    Create necessary directories if they don't exist
    """
    config = Config()
    
    # Create instance directory and subdirectories
    config.INSTANCE_PATH.mkdir(exist_ok=True)
    config.UPLOAD_FOLDER.mkdir(exist_ok=True)
    
    print(f"Instance path: {config.INSTANCE_PATH}")
    print(f"Database path: {config.DATABASE_PATH}")
    print(f"Upload folder: {config.UPLOAD_FOLDER}")

if __name__ == "__main__":
    setup_directories()
    config = Config()
    print(f"Base directory: {config.BASE_DIR}")
    print(f"HRNet model path: {config.HRNET_MODEL_PATH}")
    print(f"Second model path: {config.SECOND_MODEL_PATH}") 