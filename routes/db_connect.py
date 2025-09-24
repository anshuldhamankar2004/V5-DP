from core import *
from flask import Blueprint, request, jsonify
from services.session_manager import save_session
from services.schema_generator import generate_schema
from config import UPLOAD_FOLDER
import traceback
import polars as pl
import os
import uuid

from services.db_loader import load_from_database

db_connect_bp = Blueprint("db_connect", __name__)

@db_connect_bp.route("/connect", methods=["POST"])
def connect_db():
    try:
        req = request.get_json()

        # Extract params
        db_type = req.get("dbType")
        host = req.get("host")
        port = req.get("port")
        user = req.get("username")
        password = req.get("password")
        database = req.get("database")
        table = req.get("table")

        df = load_from_database(db_type, host, port, user, password, database, table)

        file_id = str(uuid.uuid4())
        session_data = {
            "file_path": f"{db_type}://{host}/{database}/{table}",
            "original": df.to_dicts(),
            "modified": df.to_dicts(),
            "history": []
        }

        save_session(file_id, session_data)
        generate_schema(file_id, df)

        return jsonify({"success": True, "file_id": file_id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
