# routes/visualize.py

from flask import Blueprint, request, jsonify, url_for
import os, re, time, traceback, io
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.sanitizer import convert_nan_to_none

from services.session_manager import load_session, save_session
from utils.ai_utils import ai
from notebook_logger import append_to_notebook

visualize_bp = Blueprint('visualize_bp', __name__)


@visualize_bp.route("/query/<file_id>", methods=["POST"])
def handle_query(file_id):
    data = request.get_json()
    query = data.get("query")
    query_language = data.get("query_language", "python")

    ai_query = ""
    code = ""

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({"error": "Invalid file ID"}), 404

        local_vars = {"np": np}
        df = None

        # Load modified DataFrame from Parquet in session
        try:
            parquet_bytes = session_data.get("modified_parquet")
            if parquet_bytes:
                df = pd.read_parquet(io.BytesIO(parquet_bytes))
                local_vars.update({"df": df, "pd": pd, "plt": plt, "sns": sns})
                print(f"📦 Loaded DataFrame with columns: {df.columns.tolist()}")
            else:
                return jsonify({"error": "No modified data found in session."}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading Parquet from session: {str(e)}"}), 500

        # Generate AI query
        query_type, ai_query = ai(query, df.columns.tolist(), query_language, file_id)
        if not ai_query or not query_type:
            return jsonify({"error": "Failed to generate a valid query."})

        # Clean the AI code
        code = re.sub(r"```[\w]*\n?", "", ai_query).strip().rstrip("```")
        code = "\n".join(
            line for line in code.splitlines()
            if not line.strip().startswith("import") and "plt.show()" not in line
        )
        code = code.replace(", inplace=True", "")

        # Notebook logging
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        append_to_notebook(
            os.path.join("tmp", "notebooks", f"{file_id}.ipynb"),
            f"# Query generated at {timestamp}\n\n{code}"
        )

        ### === HANDLE QUERY TYPES === ###

        if query_type == "analysis":
            try:
                result = eval(code, local_vars)

                def safe_serialize(obj):
                    if isinstance(obj, pd.DataFrame):
                        return obj.fillna('').to_dict(orient='records')
                    if isinstance(obj, (int, float, str)):
                        return obj
                    return str(obj)

                return jsonify({
                    "query_type": "analysis",
                    "ai_query": code,
                    "translated_query": code,
                    "result": safe_serialize(result)
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({
                    "error": f"Error executing analysis query: {str(e)}",
                    "ai_query": ai_query,
                    "translated_query": code
                }), 500


        elif query_type == "modification":

            try:

                # Clean wrapping

                if "df.drop(" in code and "df = " not in code:
                    code = f"df = {code}"
                elif not code.strip().startswith("df ="):
                    code = f"df = {code}"

                print(f"🐞 Before exec - code: {code}")

                exec(code, local_vars)

                df_from_exec = local_vars.get('df')

                if isinstance(df_from_exec, pd.DataFrame):

                    df = df_from_exec

                    print("✅ Modification applied successfully")

                    # Save updated DataFrame to session

                    parquet_buffer = io.BytesIO()

                    df.to_parquet(parquet_buffer)

                    session_data["modified_parquet"] = parquet_buffer.getvalue()

                    save_session(file_id, session_data)

                    preview = convert_nan_to_none(df.head(15).to_dict(orient='records'))

                    return jsonify({
                        "success": True,
                        "query_type": "modification",
                        "ai_query": code,
                        "translated_query": code,
                        "refresh": True,
                        "preview": preview
                    })

                else:

                    return jsonify({"error": "Modified result is not a valid DataFrame."}), 400


            except Exception as e:

                traceback.print_exc()

                return jsonify({

                    "error": str(e),

                    "ai_query": ai_query,

                    "translated_query": code

                }), 500

        elif query_type == "visualization":
            try:
                os.makedirs("static", exist_ok=True)
                img_filename = f"{file_id}_viz_output.png"
                img_path = os.path.join("static", img_filename)

                plt.figure(figsize=(12, 6))
                exec(code, local_vars)

                if not plt.gcf().axes:
                    raise ValueError("No plot was generated.")

                plt.savefig(img_path)
                plt.close()

                return jsonify({
                    "query_type": "visualization",
                    "ai_query": code,
                    "translated_query": code,
                    "visualization_output": url_for("static", filename=img_filename) + f"?t={int(time.time())}"
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({
                    "error": f"Error generating visualization: {str(e)}",
                    "ai_query": ai_query,
                    "translated_query": code
                }), 500

        else:
            return jsonify({"error": f"Unknown query type: {query_type}"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "ai_query": ai_query,
            "translated_query": code
        }), 500
