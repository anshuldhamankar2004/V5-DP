# routes/upload.py

from core import *
from flask import Blueprint, request, jsonify
from services.session_manager import save_session
from services.schema_generator import generate_schema
from utils.sanitizer import sanitize_filename, allowed_file
from config import UPLOAD_FOLDER
import pandas as pd
import io
import pyarrow
import traceback  # Import traceback

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400

    try:
        # ✅ File Save
        file_id = str(uuid.uuid4())
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # ✅ Load DataFrame
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # ✅ Convert DataFrame to Parquet
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow')
        parquet_data = parquet_buffer.getvalue()
        parquet_encoded = base64.b64encode(parquet_data).decode('utf-8')

        session_data = {
            "file_path": file_path,
            "original_parquet": parquet_encoded,
            "modified_parquet": parquet_encoded,
            "history": []
        }
        print("🐞 upload_bp.py: session_data before save_session:", session_data.keys())  # Check keys
        print("🐞 upload_bp.py: type(parquet_data):", type(parquet_data))  # Check data type
        print("🐞 upload_bp.py: len(parquet_data):", len(parquet_data))
        save_session(file_id, session_data)
        print("🐞 upload_bp.py: save_session called")

        # ✅ Generate Schema
        generate_schema(file_id, df)

        print(f"✅ File uploaded and processed: {file_path}")
        return jsonify({'file_id': file_id}), 200

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
