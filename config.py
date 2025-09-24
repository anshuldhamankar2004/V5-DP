# project/config.py

import os

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
UPLOAD_FOLDER = os.path.join(TMP_DIR, 'uploads')
MODEL_DIR = os.path.join(TMP_DIR, 'saved_models')
NOTEBOOK_DIR = os.path.join(TMP_DIR, 'notebooks')
SESSION_DIR = os.path.join(TMP_DIR, 'sessions')

# === Ensure directories exist ===
for path in [UPLOAD_FOLDER, MODEL_DIR, NOTEBOOK_DIR, SESSION_DIR]:
    os.makedirs(path, exist_ok=True)

# === App Config ===
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
SECRET_KEY = '123456'
MAX_RETRIES = 3

# === API Keys ===
# Set directly or use environment variables for security
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") or "6f162109add786c0563504e18939cbd0e04e176acb4bf83c5c3113213312f84d"

# Optional: Set it globally for client init
os.environ['TOGETHER_API_KEY'] = TOGETHER_API_KEY
