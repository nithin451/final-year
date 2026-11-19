"""
app.py — Main Flask Application Entry Point
Initializes and configures the Flask web server.
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import sys

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from database import init_database
from routes import api


def create_app() -> Flask:
    """
    Application factory function. Creates and configures the Flask application.

    Configuration:
        - UPLOAD_FOLDER: Directory where uploaded leaf images are stored
        - MAX_CONTENT_LENGTH: Maximum file upload size (16MB)
        - CORS: Enabled for all origins (development) — restrict in production

    Returns:
        Flask: Configured Flask application instance
    """
    # Project root directory (one level above backend/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(project_root, 'frontend')
    upload_dir = os.path.join(project_root, 'uploads')

    app = Flask(
        __name__,
        static_folder=frontend_dir,
        static_url_path=''
    )

    # ── Configuration ─────────────────────────────────────────────────────────
    app.config['UPLOAD_FOLDER'] = upload_dir
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024    # 16 MB max upload
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'jntua-cvrt-plant-disease-2025')

    # ── Enable CORS ───────────────────────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Create upload directory ───────────────────────────────────────────────
    os.makedirs(upload_dir, exist_ok=True)

    # ── Register API blueprints ───────────────────────────────────────────────
    app.register_blueprint(api)

    # ── Frontend routes (serve HTML pages) ────────────────────────────────────
    @app.route('/')
    def index():
        """Serve the home page."""
        return send_from_directory(frontend_dir, 'index.html')

    @app.route('/predict')
    def predict_page():
        """Serve the prediction page."""
        return send_from_directory(frontend_dir, 'predict.html')

    @app.route('/results')
    def results_page():
        """Serve the results page (accessed after prediction)."""
        return send_from_directory(frontend_dir, 'results.html')

    @app.route('/about')
    def about_page():
        """Serve the about page."""
        return send_from_directory(frontend_dir, 'about.html')

    @app.route('/dashboard')
    def dashboard_page():
        """Serve the dashboard / history page."""
        return send_from_directory(frontend_dir, 'dashboard.html')

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'success': False, 'error': 'Resource not found'}), 404

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({'success': False, 'error': 'File too large. Maximum is 16MB.'}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

    return app


# ── Initialize DB and create app ──────────────────────────────────────────────
init_database()
app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("  Plant Disease Detection System")
    print("  CVRT Engineering College | JNTUA CSE 2024-25")
    print("=" * 60)
    print("  Starting Flask server at http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
