#!/usr/bin/env python3
"""
Application Launcher for Cleft Detection System
Handles path configuration for both development and executable environments
"""

import os
import sys
import multiprocessing
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import configuration
from config import Config, setup_directories

def setup_environment():
    """Setup the environment for the application"""
    print("Setting up application environment...")
    
    # Setup directories
    setup_directories()
    
    # Import and configure the main app
    from app import app, init_db, BASE_DIR
    
    # Update Flask app configuration
    config = Config()
    app.config.update({
        'SECRET_KEY': config.SECRET_KEY,
        'SQLALCHEMY_DATABASE_URI': config.SQLALCHEMY_DATABASE_URI,
        'SQLALCHEMY_TRACK_MODIFICATIONS': config.SQLALCHEMY_TRACK_MODIFICATIONS,
        'SQLALCHEMY_ENGINE_OPTIONS': config.SQLALCHEMY_ENGINE_OPTIONS,
        'UPLOAD_FOLDER': str(config.UPLOAD_FOLDER),
    })
    
    # Set template and static folders
    app.template_folder = str(config.TEMPLATE_FOLDER)
    app.static_folder = str(config.STATIC_FOLDER)
    
    print(f"✅ Base directory: {config.BASE_DIR}")
    print(f"✅ Instance path: {config.INSTANCE_PATH}")
    print(f"✅ Database path: {config.DATABASE_PATH}")
    print(f"✅ Template folder: {app.template_folder}")
    print(f"✅ Static folder: {app.static_folder}")
    
    return app

def main():
    """Main entry point"""
    print("=" * 60)
    print("🦷 Cleft Lip Detection System")
    print("=" * 60)
    
    # Check if running as executable
    if getattr(sys, 'frozen', False):
        print("🔧 Running as executable")
        # Disable livereload in executable mode
        os.environ['USE_LIVERELOAD'] = 'False'
    else:
        print("🔧 Running in development mode")
    
    try:
        # Setup environment and get app
        app = setup_environment()
        
        # Initialize database
        print("🗄️  Initializing database...")
        with app.app_context():
            from app import init_db
            init_db()
        
        print("✅ Application setup complete!")
        print("🌐 Starting web server...")
        print("📱 Access the application at: http://127.0.0.1:5002")
        print("🛠️  Access GUI manager by running: python gui_db_manager.py")
        print("=" * 60)
        
        # Start the Flask application
        app.run(
            debug=False,  # Always False in production
            host='127.0.0.1',
            port=5002,
            use_reloader=False  # Disable reloader for executable
        )
        
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    
    # Run the main application
    main() 