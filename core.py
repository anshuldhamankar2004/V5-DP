# core.py

# ========== Built-ins ==========
import os
import re
import time
import uuid
import json
import decimal
import traceback
import platform
import base64
from datetime import datetime
import together

# ========== Third-Party ==========
import polars as pl # ✅ Using polars safely here
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib
importlib.reload(plt)

import sqlparse
from spellchecker import SpellChecker

# ========== Flask Stack ==========
from flask import (
    Flask, request, jsonify, render_template, redirect,
    url_for, session, send_file, Blueprint
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ========== Databases ==========
from pymongo import MongoClient
from sqlalchemy import create_engine
from pyhive import hive

# ========== Config & Constants ==========
from config import (
    UPLOAD_FOLDER, MODEL_DIR, NOTEBOOK_DIR,
    SESSION_DIR, ALLOWED_EXTENSIONS,
    MAX_RETRIES, SECRET_KEY, TOGETHER_API_KEY
)
from utils.sanitizer import (
    sanitize_filename, convert_nan_to_none,
    escape_sql_columns, escape_valid_columns
)
# ========== Services ==========
from services.session_manager import (
    get_session_path, load_session, save_session
)
from services.schema_generator import generate_schema

from utils.ai_utils import ai, spell_check_sql
from notebook_logger import append_to_notebook

# ========== AI / LLM ==========
client = together.Together()

# ❌ REMOVE heavy ML libraries from here to avoid circular import
# ✅ Move SHAP, sklearn, xgboost, catboost imports inside the specific services that use them
