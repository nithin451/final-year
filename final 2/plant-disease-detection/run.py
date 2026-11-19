"""
run.py — Single Command Application Launcher
Run this file to start the complete Plant Disease Detection System.
Usage: python run.py
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import sys

# Auto-use project virtual environment if available
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
if os.path.isfile(VENV_PYTHON) and os.path.abspath(sys.executable).lower() != os.path.abspath(VENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([VENV_PYTHON] + sys.argv))

# Add backend directory so Python finds the application modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from app import app

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🌿 Plant Disease Detection System")
    print("  CVRT Engineering College | JNTUA B.Tech CSE 2024-25")
    print("=" * 60)
    print("\n  Starting the application...")
    print("  Open your browser and go to:")
    print("\n       http://localhost:5000\n")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
