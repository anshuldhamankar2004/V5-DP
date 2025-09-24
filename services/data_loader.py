# services/data_loader.py
from core import *
import pandas as pd
from utils.sanitizer import allowed_file, sanitize_filename
from config import UPLOAD_FOLDER
from services.session_manager import save_session
import io

def load_file(file, file_id=None):
    filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    if not file_id:
        file_id = str(uuid.uuid4())

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)  # Load Parquet
    else:
        raise ValueError("Unsupported file format")

    # Convert DataFrame to a Parquet buffer
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer)

    session_data = {
        "file_path": file_path,
        "original_parquet": parquet_buffer.getvalue(),  # Store Parquet data
        "modified_parquet": parquet_buffer.getvalue(), #  and modified as Parquet
        "history": []
    }
    save_session(file_id, session_data)

    return file_id, df