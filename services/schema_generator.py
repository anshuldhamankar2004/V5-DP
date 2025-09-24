# services/schema_generator.py
from core import *
from config import UPLOAD_FOLDER
from services.session_manager import load_session, save_session
import pandas as pd
import numpy as np

def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(x) for x in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


def generate_schema(file_id, df):
    session_data = load_session(file_id)
    if not session_data:
        raise ValueError("Session data not found")

    schema = {
        'file_id': file_id,
        'columns': list(df.columns),
        'row_count': len(df),
        'sample_data': convert_nan_to_none(df.head(10).to_dict(orient='records'))
    }

    session_data["schema"] = schema
    save_session(file_id, session_data)

    schema_file = os.path.join(UPLOAD_FOLDER, f"{file_id}_schema.json")
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    print(f"✅ Schema saved: {schema_file}")