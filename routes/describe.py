# project/routes/describe.py

from core import *
from flask import Blueprint

import pandas as pd

describe_bp = Blueprint('describe', __name__)

@describe_bp.route('/describe/<file_id>', methods=['GET'])
def describe_file(file_id):
    """Handles file schema generation and error handling."""
    try:
        session_data = load_session(file_id)

        if not session_data:
            print(f"❌ File ID not found: {file_id}")
            return "Error: File ID not found", 404

        # ✅ Handle MySQL Data
        if session_data.get('db_type') == 'sql':
            print(f"🔥 MySQL Data Loaded for file_id: {file_id}")
            df = pd.DataFrame(session_data['original'])

            if 'schema' not in session_data:
                generate_schema(file_id, df)
                session_data = load_session(file_id)

            schema = session_data.get('schema')

        # ✅ Handle Uploaded File-Based Data
        else:
            file_path = session_data.get('file_path')

            if not file_path or not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return f"Error: File not found at {file_path}", 404

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format", 400

            generate_schema(file_id, df)
            session_data = load_session(file_id)
            schema = session_data.get('schema')

        return render_template('describe.html', schema=schema)

    except pd.errors.EmptyDataError:
        print(f"❌ Error: The dataset is empty.")
        return "Error: The dataset is empty.", 400

    except Exception as e:
        print(f"❌ Error loading schema: {str(e)}")
        traceback.print_exc()
        return f"Error loading schema: {str(e)}", 500


@describe_bp.route('/get-schema/<file_id>', methods=['GET'])
def get_schema(file_id):
    """Fetches the schema data for a given file ID."""
    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({'error': 'File ID not found'}), 404

        schema = session_data.get('schema')
        if not schema:
            return jsonify({'error': 'Schema not found'}), 404

        safe_schema = convert_nan_to_none(schema)
        return jsonify({'schema': safe_schema})

    except Exception as e:
        print(f"❌ Error fetching schema: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500


@describe_bp.route('/save-schema/<file_id>', methods=['POST'])
def save_schema_route(file_id):
    """Saves schema descriptions with annotations."""
    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({'error': 'File ID not found'}), 404

        data = request.get_json()
        if not data or 'schema' not in data:
            return jsonify({'error': 'Invalid data format'}), 400

        schema = data.get('schema', [])
        query_language = data.get('query_language', 'SQL')

        session_data['schema'] = {
            'annotations': schema,
            'query_language': query_language
        }

        save_session(file_id, session_data)

        schema_file = os.path.join(UPLOAD_FOLDER, f"{file_id}_schema.json")
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(session_data['schema'], f, indent=4)

        print(f"✅ Schema saved successfully at {schema_file}")
        return jsonify({'message': 'Schema saved successfully'})

    except Exception as e:
        print(f"❌ Error saving schema: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500


@describe_bp.route("/api/get_file_details/<file_id>", methods=["GET"])
def get_file_details(file_id):
    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({"error": "File not found"}), 404

        file_details = {
            "file_id": file_id,
            "file_name": session_data.get("file_name", "unknown"),
            "file_type": session_data.get("file_type", "unknown"),
            "file_size": session_data.get("file_size", "unknown"),
            "upload_time": session_data.get("upload_time", "unknown")
        }

        return jsonify(file_details)

    except Exception as e:
        print(f"❌ Error fetching file details: {str(e)}")
        return jsonify({"error": str(e)}), 500

