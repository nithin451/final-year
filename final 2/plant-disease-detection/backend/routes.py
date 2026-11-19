"""
routes.py — All Flask API Endpoints for Plant Disease Detection
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import json
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# Import local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    create_user, authenticate_user, get_user_by_token,
    save_prediction, get_prediction_history, get_statistics
)
from model import predict, get_model_info
from utils import (
    get_disease_info, extract_crop_type, is_healthy_prediction,
    allowed_file, generate_unique_filename, validate_registration_input
)

# Import preprocess_image_bytes from ml module
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml')
sys.path.insert(0, ML_DIR)
from preprocess import preprocess_image_bytes

api = Blueprint('api', __name__, url_prefix='/api')


def get_current_user():
    """
    Extract and validate the session token from the Authorization header.
    Returns the user dict if token is valid, or None if not authenticated.

    Returns:
        dict or None: Authenticated user, or None if no valid token
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
        return get_user_by_token(token)
    return None


# ── POST /api/register ────────────────────────────────────────────────────────
@api.route('/register', methods=['POST'])
def register():
    """
    Register a new user account.

    Request body (JSON):
        { "username": str, "email": str, "password": str, "full_name": str (optional) }

    Response (201 Created):
        { "success": true, "message": "Registration successful", "user_id": int }

    Response (400 Bad Request):
        { "success": false, "error": str }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Request body must be JSON'}), 400

    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    full_name = data.get('full_name', '').strip()

    # Validate input
    errors = validate_registration_input(username, email, password)
    if errors:
        return jsonify({'success': False, 'error': '; '.join(errors)}), 400

    result = create_user(username, email, password, full_name)

    if result['success']:
        return jsonify({
            'success': True,
            'message': 'Registration successful. You can now log in.',
            'user_id': result['user_id']
        }), 201
    else:
        return jsonify({'success': False, 'error': result['error']}), 400


# ── POST /api/login ───────────────────────────────────────────────────────────
@api.route('/login', methods=['POST'])
def login():
    """
    Authenticate a user and return a session token.

    Request body (JSON):
        { "username": str, "password": str }

    Response (200 OK):
        { "success": true, "token": str, "user": { id, username, email, full_name } }

    Response (401 Unauthorized):
        { "success": false, "error": "Invalid username or password" }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Request body must be JSON'}), 400

    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400

    result = authenticate_user(username, password)

    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify({'success': False, 'error': result['error']}), 401


# ── POST /api/predict ─────────────────────────────────────────────────────────
@api.route('/predict', methods=['POST'])
def predict_disease():
    """
    Upload a leaf image and get plant disease prediction.

    Request: multipart/form-data with field 'image' containing the leaf image.
    Accepted formats: JPG, JPEG, PNG, WEBP, BMP

    Response (200 OK):
        {
          "success": true,
          "prediction": {
            "predicted_class": str,
            "confidence": float,
            "confidence_percent": str,
            "display_name": str,
            "crop_type": str,
            "is_healthy": bool,
            "severity": str,
            "description": str,
            "treatment": [str, ...],
            "prevention": [str, ...],
            "top3": [{"label": str, "confidence": float}, ...]
          },
          "prediction_id": int
        }

    Response (400 Bad Request):
        { "success": false, "error": str }
    """
    # ── Validate upload ───────────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided. Use form field "image".'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed: JPG, JPEG, PNG, WEBP, BMP'
        }), 400

    try:
        # ── Save uploaded image ───────────────────────────────────────────────
        image_bytes = file.read()
        unique_filename = generate_unique_filename(file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        save_path = os.path.join(upload_folder, unique_filename)

        with open(save_path, 'wb') as f:
            f.write(image_bytes)

        # ── Preprocess & predict ──────────────────────────────────────────────
        img_array = preprocess_image_bytes(image_bytes)
        result = predict(img_array)

        predicted_class = result['predicted_class']
        confidence = result['confidence']

        # ── Get disease info ──────────────────────────────────────────────────
        disease_info = get_disease_info(predicted_class)
        crop_type = extract_crop_type(predicted_class)
        healthy = is_healthy_prediction(predicted_class)

        # ── Save to database ──────────────────────────────────────────────────
        user = get_current_user()
        user_id = user['id'] if user else None

        prediction_id = save_prediction(
            user_id=user_id,
            image_filename=unique_filename,
            predicted_disease=predicted_class,
            confidence_score=confidence,
            crop_type=crop_type,
            is_healthy=healthy,
            top3_predictions=result['top3']
        )

        # ── Build response ────────────────────────────────────────────────────
        return jsonify({
            'success': True,
            'prediction': {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 4),
                'confidence_percent': f"{confidence * 100:.1f}%",
                'display_name': disease_info['display_name'],
                'crop_type': crop_type,
                'is_healthy': healthy,
                'severity': disease_info['severity'],
                'description': disease_info['description'],
                'treatment': disease_info['treatment'],
                'prevention': disease_info['prevention'],
                'top3': result['top3'],
                'image_filename': unique_filename
            },
            'prediction_id': prediction_id
        }), 200

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Image preprocessing failed: {str(e)}'}), 400
    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error during prediction'}), 500


# ── GET /api/history ──────────────────────────────────────────────────────────
@api.route('/history', methods=['GET'])
def get_history():
    """
    Get prediction history. Returns user-specific history if authenticated,
    or all recent predictions if not.

    Query params:
        limit (int): Max records to return (default 20, max 100)

    Response (200 OK):
        { "success": true, "predictions": [...], "count": int }
    """
    limit = min(int(request.args.get('limit', 20)), 100)
    user = get_current_user()
    user_id = user['id'] if user else None

    predictions = get_prediction_history(user_id=user_id, limit=limit)

    # Convert JSON strings back to Python objects for top3
    for p in predictions:
        if isinstance(p.get('top3_predictions'), str):
            try:
                p['top3_predictions'] = json.loads(p['top3_predictions'])
            except Exception:
                p['top3_predictions'] = []

    return jsonify({
        'success': True,
        'predictions': predictions,
        'count': len(predictions)
    }), 200


# ── GET /api/diseases ─────────────────────────────────────────────────────────
@api.route('/diseases', methods=['GET'])
def get_diseases():
    """
    Get list of all supported disease classes.

    Response (200 OK):
        { "success": true, "diseases": [{ display_name, crop, severity }, ...], "count": int }
    """
    from utils import DISEASE_INFO
    diseases = [
        {
            'class_name': k,
            'display_name': v['display_name'],
            'crop': v['crop'],
            'severity': v['severity']
        }
        for k, v in DISEASE_INFO.items()
    ]
    return jsonify({'success': True, 'diseases': diseases, 'count': len(diseases)}), 200


# ── GET /api/stats ────────────────────────────────────────────────────────────
@api.route('/stats', methods=['GET'])
def get_stats():
    """
    Get aggregate system statistics.

    Response (200 OK):
        { "success": true, "stats": { total_predictions, total_users, ... }, "model_info": {...} }
    """
    stats = get_statistics()
    model_info = get_model_info()

    return jsonify({
        'success': True,
        'stats': stats,
        'model_info': model_info
    }), 200


# ── GET /api/health ───────────────────────────────────────────────────────────
@api.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint for monitoring."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'Plant Disease Detection API',
        'version': '1.0.0'
    }), 200
