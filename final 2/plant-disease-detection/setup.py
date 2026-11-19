"""
setup.py — One-Time Project Setup Script
Run once after cloning to initialize the database and check dependencies.
Usage: python setup.py
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import sys
import subprocess

def check_python_version():
    """Verify Python >= 3.9 is installed."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"[ERROR] Python 3.9+ required. You have Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"[✓] Python {version.major}.{version.minor}.{version.micro}")


def check_and_install_requirements():
    """Install all required packages from requirements.txt."""
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    print("\n[INFO] Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file, '-q'])
        print("[✓] All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        sys.exit(1)


def initialize_database():
    """Create the SQLite database and tables."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    from database import init_database
    print("\n[INFO] Initializing database...")
    init_database()


def create_directories():
    """Create required directories if they don't exist."""
    dirs = ['uploads', 'datasets/plantvillage', 'ml']
    for d in dirs:
        path = os.path.join(os.path.dirname(__file__), d)
        os.makedirs(path, exist_ok=True)
    print("[✓] Required directories created")


def main():
    print("=" * 60)
    print("  Plant Disease Detection — Setup")
    print("  CVRT Engineering College | JNTUA CSE 2024-25")
    print("=" * 60)

    check_python_version()
    check_and_install_requirements()
    create_directories()
    initialize_database()

    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("\n  Next steps:")
    print("  1. Download PlantVillage dataset (see datasets/README.md)")
    print("  2. Train the model: python ml/train.py")
    print("  3. Run the app:     python run.py")
    print("  4. Open browser:    http://localhost:5000")
    print("=" * 60)


if __name__ == '__main__':
    main()
