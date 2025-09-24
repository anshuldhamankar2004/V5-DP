from core import *
from flask import Blueprint, render_template, jsonify
from services.session_manager import load_session, save_session, rollback_session
from services.schema_generator import generate_schema
from config import UPLOAD_FOLDER
import pandas as pd
import io
import traceback  # Make sure this is imported

cleaning_bp = Blueprint("cleaning", __name__)


@cleaning_bp.route("/cleaning/<file_id>", methods=["GET"])
def cleaning_page(file_id):
    try:
        print(f"🔍 Received request for file_id: {file_id}")

        session_data = load_session(file_id)
        if not session_data:
            print("❌ No session data found.")
            return "Error: File ID not found", 404

        parquet_data = session_data.get("modified_parquet") or session_data.get("original_parquet")
        if not parquet_data:
            print("❌ No parquet data found in session.")
            return "Error: No data found in session", 500

        try:
            print("📦 Trying to load parquet data into DataFrame...")
            parquet_bytes = io.BytesIO(parquet_data)
            df = pd.read_parquet(parquet_bytes)
        except Exception as e:
            print("❌ Failed to read parquet:", str(e))
            traceback.print_exc()
            return f"Parquet read error: {str(e)}", 500

        sample_data = df.head(10).to_dict(orient="records")
        print("✅ Data loaded and sample created successfully.")
        return render_template("cleaning.html", file_id=file_id, sample_data=sample_data)

    except Exception as e:
        print(f"❌ Unexpected error in cleaning_page: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@cleaning_bp.route("/rollback/<file_id>", methods=["POST"])
def rollback(file_id):
    try:
        session_data, success = rollback_session(file_id)
        if not session_data:
            return jsonify({"success": False, "error": "File not found"}), 404

        parquet_data = session_data.get("modified_parquet")
        if not parquet_data:
            return jsonify({"success": False, "error": "No modified data found for rollback."}), 500

        try:
            parquet_bytes = io.BytesIO(parquet_data)
            df = pd.read_parquet(parquet_bytes)
        except Exception as e:
            print(f"❌ Parquet read error in rollback: {str(e)}")
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Parquet read error: {str(e)}"}), 500

        data_json = df.fillna("").astype(str).to_dict(orient="records")

        return jsonify({
            "success": True,
            "info": "Rolled back successfully" if success else "No more steps to rollback.",
            "data": data_json
        })

    except Exception as e:
        print(f"❌ Rollback error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
