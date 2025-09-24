from core import *
from flask import Blueprint, jsonify
from utils.sanitizer import convert_nan_to_none
import pandas as pd
import os
import traceback
import io

data_access_bp = Blueprint('data_access', __name__)

@data_access_bp.route('/get-data/<file_id>', methods=['GET'])
def get_data(file_id):
    try:
        print(f"🔥 Fetching data for file_id: {file_id}")
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({'error': 'Session not found'}), 404

        # Load DataFrame from modified Parquet data if available
        if "modified_parquet" in session_data:
            parquet_data = io.BytesIO(session_data["modified_parquet"])
            df = pd.read_parquet(parquet_data)
            print("✅ Using modified session data (Parquet).")
        elif "original_parquet" in session_data: # Fallback to original Parquet
            parquet_data = io.BytesIO(session_data["original_parquet"])
            df = pd.read_parquet(parquet_data)
            print("✅ Using original session data (Parquet).")
        elif "modified" in session_data and isinstance(session_data["modified"], list): # Fallback for initial load (should ideally not be used after first Parquet save)
            df = pd.DataFrame(session_data["modified"])
            print("✅ Using modified session data (list).")
        elif "original" in session_data and isinstance(session_data["original"], list): # Fallback for very initial load
            df = pd.DataFrame(session_data["original"])
            print("✅ Using original session data (list).")
        else:
            return jsonify({"error": "No data found in session."}), 400

        preview_data = df.head(15).to_dict(orient='records')
        return jsonify({
            "columns": df.columns.tolist(),
            "data": convert_nan_to_none(preview_data)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Error loading data: {str(e)}"}), 500