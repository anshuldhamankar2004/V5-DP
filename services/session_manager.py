# services/session_manager.py
from core import *
from config import SESSION_DIR
import json
import os
import base64

def get_session_path(file_id):
    return os.path.join(SESSION_DIR, f"{file_id}.json")


def load_session(file_id):
    try:
        with open(get_session_path(file_id), "r") as f:
            data = json.load(f)
            # Decode base64 back to bytes after loading
            if "original_parquet" in data and isinstance(data["original_parquet"], str):
                data["original_parquet"] = base64.b64decode(data["original_parquet"])
            if "modified_parquet" in data and isinstance(data["modified_parquet"], str):
                data["modified_parquet"] = base64.b64decode(data["modified_parquet"])
            return data
    except FileNotFoundError:
        return {}


def save_session(file_id, data):
    # Encode byte data to base64 before saving
    if "original_parquet" in data and isinstance(data["original_parquet"], bytes):
        data["original_parquet"] = base64.b64encode(data["original_parquet"]).decode('utf-8')
    if "modified_parquet" in data and isinstance(data["modified_parquet"], bytes):
        data["modified_parquet"] = base64.b64encode(data["modified_parquet"]).decode('utf-8')
    with open(get_session_path(file_id), "w") as f:
        json.dump(data, f)


def rollback_session(file_id):
    session_data = load_session(file_id)
    if not session_data or not session_data.get("history"):
        return session_data, False

    # Handle rollback for Parquet data as well (if you decide to store history this way)
    # This example assumes history stores the 'modified_parquet' key
    if "history" in session_data and session_data["history"]:
        last_state = session_data["history"].pop()
        if "modified_parquet" in last_state:
            session_data["modified_parquet"] = last_state["modified_parquet"]
        elif "modified" in last_state: # Fallback for older history
            session_data["modified"] = last_state["modified"]

    save_session(file_id, session_data)
    return session_data, True