# In your app.py or routes.py (where you register Blueprint or other endpoints)
from flask import send_file
import os
from flask import Blueprint, jsonify

notebook_bp = Blueprint("notebook", __name__)

@notebook_bp.route("/download-notebook/<file_id>", methods=["GET"])
def download_notebook(file_id):
    try:
        notebook_path = os.path.join("tmp", "notebooks", f"{file_id}.ipynb")
        if not os.path.exists(notebook_path):
            return jsonify({"error": "Notebook file not found"}), 404

        return send_file(
            notebook_path,
            mimetype="application/x-ipynb+json",
            as_attachment=True,
            download_name=f"{file_id}_session.ipynb"
        )
    except Exception as e:
        print(f"❌ Error in downloading notebook: {str(e)}")
        return jsonify({"error": str(e)}), 500
