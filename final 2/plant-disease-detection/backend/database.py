"""
database.py — SQLite Database Setup and Operations
Handles all database interactions: users, predictions, session.
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import sqlite3
import os
import hashlib
import secrets
from datetime import datetime

# Database file path (stored at project root)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plant_disease.db')


def get_connection():
    """
    Create and return a SQLite database connection.
    row_factory = sqlite3.Row enables dict-like row access (row['column_name']).

    Returns:
        sqlite3.Connection: Active database connection
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Initialize the SQLite database by creating all required tables if they do not exist.

    Tables created:
        users        — Stores registered user accounts
        predictions  — Stores every disease prediction made (with user FK)
        sessions     — Stores login session tokens

    This function is IDEMPOTENT — safe to call multiple times (CREATE IF NOT EXISTS).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Users table ───────────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            full_name     TEXT,
            role          TEXT    DEFAULT 'user',
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        )
    ''')

    # ── Predictions table ─────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER REFERENCES users(id) ON DELETE SET NULL,
            image_filename    TEXT    NOT NULL,
            predicted_disease TEXT    NOT NULL,
            confidence_score  REAL    NOT NULL,
            crop_type         TEXT,
            is_healthy        INTEGER DEFAULT 0,
            top3_predictions  TEXT,
            predicted_at      TEXT    DEFAULT (datetime('now'))
        )
    ''')

    # ── Sessions table ────────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at  TEXT    DEFAULT (datetime('now')),
            expires_at  TEXT    NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("[✓] Database initialized successfully → plant_disease.db")


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using SHA-256 with a random salt.
    Format stored: "SALT:HASH" (both hex-encoded).

    Args:
        password (str): Plain text password

    Returns:
        str: Salted hash in format "salt_hex:hash_hex"
    """
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a plain-text password against a stored salted hash.

    Args:
        password (str): Plain text password to verify
        stored_hash (str): Stored hash in format "salt_hex:hash_hex"

    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        salt, hashed = stored_hash.split(':')
        expected = hashlib.sha256((salt + password).encode()).hexdigest()
        return secrets.compare_digest(expected, hashed)
    except Exception:
        return False


# ── User Operations ───────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str, full_name: str = '') -> dict:
    """
    Register a new user account in the database.

    Args:
        username (str): Unique username (3-50 chars)
        email (str): Valid email address
        password (str): Plain text password (will be hashed)
        full_name (str): Optional display name

    Returns:
        dict: {'success': True, 'user_id': int} or {'success': False, 'error': str}
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)',
            (username.strip(), email.strip().lower(), password_hash, full_name.strip())
        )
        conn.commit()
        return {'success': True, 'user_id': cursor.lastrowid}
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return {'success': False, 'error': 'Username already taken'}
        elif 'email' in str(e):
            return {'success': False, 'error': 'Email already registered'}
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate a user with username and password.
    Creates a new session token valid for 24 hours on success.

    Args:
        username (str): Username or email
        password (str): Plain text password

    Returns:
        dict: {'success': True, 'token': str, 'user': dict} or {'success': False, 'error': str}
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Support login by username OR email
        cursor.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, username.lower())
        )
        user = cursor.fetchone()

        if not user or not verify_password(password, user['password_hash']):
            return {'success': False, 'error': 'Invalid username or password'}

        # Create session token
        token = secrets.token_hex(32)
        from datetime import timedelta
        expires = (datetime.now() + timedelta(hours=24)).isoformat()

        cursor.execute(
            'INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)',
            (token, user['id'], expires)
        )

        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?",
            (user['id'],)
        )
        conn.commit()

        return {
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name'],
                'role': user['role']
            }
        }
    finally:
        conn.close()


def get_user_by_token(token: str) -> dict:
    """
    Look up a user by their session token. Returns None if token is invalid/expired.

    Args:
        token (str): Session token from HTTP header or cookie

    Returns:
        dict or None: User record dict or None if not found/expired
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN sessions s ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > datetime('now')
        ''', (token,))
        user = cursor.fetchone()
        return dict(user) if user else None
    finally:
        conn.close()


# ── Prediction Operations ─────────────────────────────────────────────────────

def save_prediction(user_id: int, image_filename: str, predicted_disease: str,
                    confidence_score: float, crop_type: str,
                    is_healthy: bool, top3_predictions: list) -> int:
    """
    Save a disease prediction result to the database.

    Args:
        user_id (int): ID of the user who made the prediction (can be None for guests)
        image_filename (str): Saved filename of the uploaded leaf image
        predicted_disease (str): Top prediction class name
        confidence_score (float): Model confidence (0.0 to 1.0)
        crop_type (str): Crop species extracted from class name
        is_healthy (bool): True if the leaf is predicted healthy
        top3_predictions (list): Top 3 predictions as [{'label': str, 'confidence': float}]

    Returns:
        int: ID of the saved prediction record
    """
    import json as json_lib
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions
                (user_id, image_filename, predicted_disease, confidence_score,
                 crop_type, is_healthy, top3_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            image_filename,
            predicted_disease,
            float(confidence_score),
            crop_type,
            1 if is_healthy else 0,
            json_lib.dumps(top3_predictions)
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_prediction_history(user_id: int = None, limit: int = 50) -> list:
    """
    Retrieve prediction history, optionally filtered by user.

    Args:
        user_id (int): If provided, return only this user's predictions
        limit (int): Maximum number of records to return

    Returns:
        list: List of prediction dicts, most recent first
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('''
                SELECT p.*, u.username FROM predictions p
                LEFT JOIN users u ON u.id = p.user_id
                WHERE p.user_id = ?
                ORDER BY p.predicted_at DESC LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT p.*, u.username FROM predictions p
                LEFT JOIN users u ON u.id = p.user_id
                ORDER BY p.predicted_at DESC LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_statistics() -> dict:
    """
    Get aggregate system statistics for the dashboard/stats API endpoint.

    Returns:
        dict: {total_predictions, total_users, diseases_detected, avg_confidence}
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM predictions')
        total_preds = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(DISTINCT predicted_disease) as count FROM predictions')
        diseases_detected = cursor.fetchone()['count']

        cursor.execute('SELECT AVG(confidence_score) as avg FROM predictions')
        avg_conf = cursor.fetchone()['avg'] or 0.0

        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE is_healthy = 1')
        healthy_count = cursor.fetchone()['count']

        return {
            'total_predictions': total_preds,
            'total_users': total_users,
            'diseases_detected': diseases_detected,
            'avg_confidence': round(float(avg_conf) * 100, 1),
            'healthy_predictions': healthy_count
        }
    finally:
        conn.close()


if __name__ == '__main__':
    """Run directly to initialize the database."""
    print("Initializing Plant Disease Detection database...")
    init_database()
