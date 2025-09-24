from flask import Blueprint, request, jsonify, render_template, send_file
import os, traceback
import pandas as pd
from services.session_manager import load_session, save_session
from services.schema_generator import generate_schema
from sklearn.feature_selection import VarianceThreshold
import json
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import io
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from notebook_logger import append_to_notebook  # Import the notebook logger

custom_clean_bp = Blueprint('custom_clean', __name__)
api_clean_bp = Blueprint('api_clean', __name__)


@custom_clean_bp.route("/custom_clean/<file_id>/download")
def download_notebook(file_id):
    """Downloads the generated Jupyter Notebook."""
    notebook_path = f"/temp/notebooks/notebook_{file_id}.ipynb"
    if os.path.exists(notebook_path):
        return send_file(notebook_path, as_attachment=True, download_name=f"data_cleaning_{file_id}.ipynb")
    else:
        return jsonify({"error": "Notebook not found."}), 404

def get_df(file_id):
    """Loads the DataFrame from the session."""
    session = load_session(file_id)
    if not session or "modified" not in session:
        raise ValueError("Session not found or modified data missing")
    df = pd.DataFrame(session["modified"])
    return df, session


def save_df(file_id, df, session):
    """Saves the DataFrame to the session."""
    session["modified"] = df.to_dict(orient="records")
    save_session(file_id, session)


def preview(df, n=20):
    """Returns a preview of the DataFrame as a list of dictionaries."""
    return json.loads(df.head(n).to_json(orient="records"))



@custom_clean_bp.route("/custom_clean/<file_id>", methods=["GET"])
def custom_clean_page(file_id):
    """Renders the custom cleaning page."""
    try:
        session_data = load_session(file_id)
        if not session_data:
            return "Error: Session not found", 404

        file_path = session_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return "Error: Original file not found", 404

        modified_data = session_data.get("modified")
        if modified_data:
            df = pd.DataFrame(modified_data)
        else:
            try:
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                session_data["original"] = df.to_dict(orient="records")
                session_data["modified"] = df.to_dict(orient="records")
                save_session(file_id, session_data)
            except Exception as e:
                return f"Error loading file: {str(e)}", 500

        schema = generate_schema(file_id, df)
        sample_data = preview(df)
        file_name = session_data.get("file_name", "unknown")

        return render_template(
            "custom_clean.html",
            file_id=file_id,
            schema=schema,
            sample_data=sample_data,
            file_name=file_name
        )

    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@custom_clean_bp.route('/custom_clean/<file_id>/columns', methods=['GET'])
def get_columns(file_id):
    """Returns the list of column names and their data types."""
    try:
        df, _ = get_df(file_id)
        return jsonify({"columns": list(df.columns), "types": df.dtypes.astype(str).to_dict()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/missing", methods=["POST"])
def handle_missing(file_id):
    """Handles missing data imputation or removal."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        columns_to_clean = data.get('columns', [])
        method = data.get('method')
        value = data.get('value')
        neighbors = data.get('neighbors')

        if not columns_to_clean:
            return jsonify({"error": "Please select at least one column."}), 400

        df_cleaned = df.copy()  # Operate on a copy
        code_block = f"# Handling missing values\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"

        if method == "drop_rows":
            df_cleaned = df_cleaned.dropna(subset=columns_to_clean)
            code_block += f"df_cleaned = df.dropna(subset={columns_to_clean})\n"
        elif method == "drop_cols":
            missing_cols = [col for col in columns_to_clean if col not in df_cleaned.columns]
            if missing_cols:
                return jsonify({"error": f"Columns not found: {missing_cols}"}), 400
            df_cleaned = df_cleaned.drop(columns=columns_to_clean)
            code_block += f"df_cleaned = df.drop(columns={columns_to_clean})\n"
        elif method == "mean":
            for col in columns_to_clean:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    mean_val = df_cleaned[col].mean(numeric_only=True)
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                    code_block += f"df_cleaned['{col}'] = df['{col}'].fillna(df['{col}'].mean(numeric_only=True))\n"
                else:
                    return jsonify({"error": f"Column '{col}' is not numeric for mean imputation."}), 400
        elif method == "median":
            for col in columns_to_clean:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    median_val = df_cleaned[col].median(numeric_only=True)
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    code_block += f"df_cleaned['{col}'] = df['{col}'].fillna(df['{col}'].median(numeric_only=True))\n"
                else:
                    return jsonify({"error": f"Column '{col}' is not numeric for median imputation."}), 400
        elif method == "knn":
            try:
                n_neighbors = int(neighbors) if neighbors else 5
                numeric_cols = [col for col in columns_to_clean if pd.api.types.is_numeric_dtype(df_cleaned[col])]
                if not numeric_cols:
                    return jsonify({"error": "No numeric columns selected for KNN imputation."}), 400
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
                code_block += f"from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors={n_neighbors})\ndf_cleaned[{numeric_cols}] = imputer.fit_transform(df[{numeric_cols}])\n"
            except ValueError:
                return jsonify({"error": "Invalid number of neighbors for KNN."}), 400
        elif method == "regression":
            return jsonify({"error": "Regression imputation not yet implemented."}), 501
        elif method == "custom":
            if value is not None:
                df_cleaned[columns_to_clean] = df_cleaned[columns_to_clean].fillna(value)
                code_block += f"df_cleaned[{columns_to_clean}] = df[{columns_to_clean}].fillna('{value}')\n"
            else:
                return jsonify({"error": "Custom value not provided."}), 400
        else:
            return jsonify({"error": f"Invalid missing data handling method: {method}"}), 400

        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": f"Applied {method} to columns: {', '.join(columns_to_clean)}", "preview": preview(df_cleaned)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/duplicates", methods=["POST"])
def handle_duplicates(file_id):
    """Removes duplicate rows from the DataFrame."""
    try:
        df, session = get_df(file_id)
        df_cleaned = df.drop_duplicates()
        code_block = f"# Removing duplicate rows\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\ndf_cleaned = df.drop_duplicates()\n"
        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": "Removed duplicate rows", "preview": preview(df_cleaned)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/dtype", methods=["POST"])
def convert_dtype(file_id):
    """Converts the data type of a specified column."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        column = data.get('column')
        dtype = data.get('dtype')

        if not column or not dtype:
            return jsonify({"error": "Both 'column' and 'dtype' must be provided"}), 400

        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the data."}), 400

        original_dtype = df[column].dtype
        df_cleaned = df.copy()
        code_block = f"# Converting data type of column '{column}' to '{dtype}'\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"

        try:
            if dtype == "number":
                df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
                code_block += f"df_cleaned['{column}'] = pd.to_numeric(df['{column}'], errors='coerce')\n"
            elif dtype == "string":
                df_cleaned[column] = df_cleaned[column].astype(str)
                code_block += f"df_cleaned['{column}'] = df['{column}'].astype(str)\n"
            elif dtype == "boolean":
                df_cleaned[column] = df_cleaned[column].astype(str).str.lower().map({'true': True, 'false': False})
                code_block += f"df_cleaned['{column}'] = df['{column}'].astype(str).str.lower().map({{True: True, False: False}})\n"
            elif dtype == "datetime":
                df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
                code_block += f"df_cleaned['{column}'] = pd.to_datetime(df['{column}'], errors='coerce')\n"
            elif dtype == "category":
                df_cleaned[column] = df_cleaned[column].astype('category')
                code_block += f"df_cleaned['{column}'] = df['{column}'].astype('category')\n"
            else:
                return jsonify({"error": f"Unsupported dtype: {dtype}"}), 400

            save_df(file_id, df_cleaned, session)
            append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
            return jsonify({
                "message": f"Converted column '{column}' from {original_dtype} to {df_cleaned[column].dtype}",
                "preview": preview(df_cleaned)
            })

        except Exception as conversion_error:
            error_message = f"Error converting column '{column}' to {dtype}: {str(conversion_error)}"
            print(f"Conversion Error: {traceback.format_exc()}")
            return jsonify({"error": error_message}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/text", methods=["POST"])
def text_preprocessing(file_id):
    """Applies various text preprocessing techniques to specified columns."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        columns = data.get("columns", [])

        if not columns:
            return jsonify({"error": "No columns provided"}), 400

        lowercase = data.get("lowercase", True)
        remove_punctuation = data.get("remove_punctuation", True)
        remove_stopwords_flag = data.get("remove_stopwords", False)
        stemming_flag = data.get("stemming", False)
        lemmatization_flag = data.get("lemmatization", False)

        stop_words_set = set(stopwords.words("english")) if remove_stopwords_flag else set()
        stemmer = PorterStemmer() if stemming_flag else None
        lemmatizer = WordNetLemmatizer() if lemmatization_flag else None

        def clean_text(text):
            text = str(text)
            if lowercase:
                text = text.lower()
            if remove_punctuation:
                text = re.sub(r"[^\w\s]", "", text)
            words = text.split()
            if remove_stopwords_flag:
                words = [word for word in words if word not in stop_words_set]
            if stemming_flag:
                words = [stemmer.stem(word) for word in words]
            if lemmatization_flag:
                words = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(words)

        df_cleaned = df.copy()
        code_block = f"# Text preprocessing on columns: {columns}\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"
        code_block += "import re\n"
        if remove_stopwords_flag:
            code_block += "from nltk.corpus import stopwords\nstop_words_set = set(stopwords.words('english'))\n"
        if stemming_flag:
            code_block += "from nltk.stem import PorterStemmer\nstemmer = PorterStemmer()\n"
        if lemmatization_flag:
            code_block += "from nltk.stem import WordNetLemmatizer\nlemmatizer = WordNetLemmatizer()\n"
        code_block += "def clean_text(text):\n    text = str(text)\n"
        if lowercase:
            code_block += "    text = text.lower()\n"
        if remove_punctuation:
            code_block += "    text = re.sub(r'[^\w\s]', '', text)\n"
        code_block += "    words = text.split()\n"
        if remove_stopwords_flag:
            code_block += "    words = [word for word in words if word not in stop_words_set]\n"
        if stemming_flag:
            code_block += "    words = [stemmer.stem(word) for word in words]\n"
        if lemmatization_flag:
            code_block += "    words = [lemmatizer.lemmatize(word) for word in words]\n"
        code_block += "    return ' '.join(words)\n"

        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).apply(clean_text)
                code_block += f"df_cleaned['{col}'] = df['{col}'].astype(str).apply(clean_text)\n"

        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": f"Cleaned text in columns: {', '.join(columns)}", "preview": preview(df_cleaned)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/outliers", methods=["POST"])
def handle_outliers(file_id):
    """Handles outlier detection and treatment."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        columns = data.get("columns", [])
        method = data.get("method", "iqr").lower()
        handling = data.get("handling", "remove").lower()
        params = {
            "zscore_threshold": float(data.get("zscore_threshold", 3)),
            "iqr_factor": float(data.get("iqr_factor", 1.5)),
            "boxplots_threshold": float(data.get("boxplots_threshold", 1.5)),
            "contamination": float(data.get("contamination", 0.1)),
            "custom_value": data.get("custom_value")
        }

        df_cleaned = df.copy()
        code_block = f"# Handling outliers using {method} method ({handling})\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"
        if method == "isolation_forest":
            code_block += "from sklearn.ensemble import IsolationForest\n"

        for col in columns:
            if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue

            series = df_cleaned[col].copy()
            mask = pd.Series([True] * len(series))
            col_code = f"\n# Outlier handling for column '{col}'\nseries = df_cleaned['{col}'].copy()\nmask = pd.Series([True] * len(series))\n"

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - params["iqr_factor"] * iqr
                upper_bound = q3 + params["iqr_factor"] * iqr
                mask = (series >= lower_bound) & (series <= upper_bound)
                col_code += f"q1 = series.quantile(0.25)\nq3 = series.quantile(0.75)\niqr = q3 - q1\nlower_bound = q1 - {params['iqr_factor']} * iqr\nupper_bound = q3 + {params['iqr_factor']} * iqr\nmask = (series >= lower_bound) & (series <= upper_bound)\n"
            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std != 0:
                    z_scores = np.abs((series - mean) / std)
                    mask = z_scores < params["zscore_threshold"]
                    col_code += f"mean = series.mean()\nstd = series.std()\nz_scores = np.abs((series - mean) / std)\nmask = z_scores < {params['zscore_threshold']}\n"
                else:
                    print(f"Warning: Standard deviation is zero for column '{col}'. Z-score outlier detection skipped.")
                    continue
            elif method == "boxplots":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - params["boxplots_threshold"] * iqr
                upper_bound = q3 + params["boxplots_threshold"] * iqr
                mask = (series >= lower_bound) & (series <= upper_bound)
                col_code += f"q1 = series.quantile(0.25)\nq3 = series.quantile(0.75)\niqr = q3 - q1\nlower_bound = q1 - {params['boxplots_threshold']} * iqr\nupper_bound = q3 + {params['boxplots_threshold']} * iqr\nmask = (series >= lower_bound) & (series <= upper_bound)\n"
            elif method == "isolation_forest":
                try:
                    model = IsolationForest(contamination=params["contamination"], random_state=42)
                    preds = model.fit_predict(series.values.reshape(-1, 1))
                    mask = preds == 1  # 1 = inlier
                    col_code += f"model = IsolationForest(contamination={params['contamination']}, random_state=42)\npreds = model.fit_predict(series.values.reshape(-1, 1))\nmask = preds == 1\n"
                except Exception as e:
                    print(f"Isolation Forest error on column {col}: {e}")
                    continue

            if handling == "remove":
                df_cleaned = df_cleaned[mask]
                col_code += f"df_cleaned = df_cleaned[mask]\n"
            elif handling == "clip":
                if method in ("iqr", "boxplots"):
                    df_cleaned[col] = np.clip(series, lower_bound, upper_bound)
                    col_code += f"df_cleaned['{col}'] = np.clip(series, lower_bound, upper_bound)\n"
            elif handling in ["replace_mean", "replace_median", "replace_custom"]:
                outlier_indices = ~mask
                if handling == "replace_mean":
                    replacement_value = series.mean()
                    df_cleaned.loc[outlier_indices, col] = replacement_value
                    col_code += f"replacement_value = series.mean()\ndf_cleaned.loc[~mask, '{col}'] = replacement_value\n"
                elif handling == "replace_median":
                    replacement_value = series.median()
                    df_cleaned.loc[outlier_indices, col] = replacement_value
                    col_code += f"replacement_value = series.median()\ndf_cleaned.loc[~mask, '{col}'] = replacement_value\n"
                elif handling == "replace_custom":
                    custom_value = params["custom_value"]
                    try:
                        replacement_value = float(custom_value)
                        df_cleaned.loc[outlier_indices, col] = replacement_value
                        col_code += f"replacement_value = {replacement_value}\ndf_cleaned.loc[~mask, '{col}'] = replacement_value\n"
                    except (ValueError, TypeError):
                        return jsonify({"error": f"Invalid custom value: {custom_value}"}), 400
            code_block += col_code

        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": f"Outliers handled using {method} ({handling})", "preview": preview(df_cleaned)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/clean/<file_id>/feature_engineer", methods=["POST"])
def feature_engineering(file_id):
    """Applies various feature engineering techniques."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        method = data.get('method')
        params = data.get('params', {})
        df_cleaned = df.copy()
        code_block = f"# Feature engineering using method: {method}\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"

        if method == 'interaction':
            col1 = params.get('col1')
            col2 = params.get('col2')
            if col1 in df_cleaned.columns and col2 in df_cleaned.columns and pd.api.types.is_numeric_dtype(
                    df_cleaned[col1]) and pd.api.types.is_numeric_dtype(df_cleaned[col2]):
                new_col_name = f"{col1}_{col2}_interaction"
                df_cleaned[new_col_name] = df_cleaned[col1] * df_cleaned[col2]
                code_block += f"df_cleaned['{new_col_name}'] = df['{col1}'] * df['{col2}']\n"
            else:
                return jsonify({"error": "Interaction columns must be numeric and exist."}), 400
        elif method == 'date_parts':
            col = params.get('date_column')
            if col in df_cleaned.columns:
                try:
                    df_cleaned[f"{col}_year"] = pd.to_datetime(df_cleaned[col], errors='coerce').dt.year
                    df_cleaned[f"{col}_month"] = pd.to_datetime(df_cleaned[col], errors='coerce').dt.month
                    df_cleaned[f"{col}_weekday"] = pd.to_datetime(df_cleaned[col], errors='coerce').dt.weekday
                    code_block += f"df_cleaned['{col}_year'] = pd.to_datetime(df['{col}'], errors='coerce').dt.year\n"
                    code_block += f"df_cleaned['{col}_month'] = pd.to_datetime(df['{col}'], errors='coerce').dt.month\n"
                    code_block += f"df_cleaned['{col}_weekday'] = pd.to_datetime(df['{col}'], errors='coerce').dt.weekday\n"
                except Exception as e:
                    return jsonify({"error": f"Error extracting date parts from column '{col}': {str(e)}"}), 400
            else:
                return jsonify({"error": f"Date column '{col}' not found."}), 400
        elif method == 'binning':
            col = params.get('column')
            binning_type = params.get('binning_type')
            bins = params.get('bins')
            custom_bins = params.get('custom_bins')
            labels = params.get('labels')
            new_col_name = f"{col}_binned"

            if not col or col not in df_cleaned.columns:
                return jsonify({"error": "Column not specified or not found for binning."}), 400
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                return jsonify({"error": f"Column '{col}' must be numeric for binning."}), 400

            if binning_type == 'auto':
                try:
                    num_bins = int(bins) if bins else 5
                    df_cleaned[new_col_name] = pd.cut(df_cleaned[col], bins=num_bins, labels=False,
                                                      include_lowest=True)
                    code_block += f"df_cleaned['{new_col_name}'] = pd.cut(df['{col}'], bins={num_bins}, labels=False, include_lowest=True)\n"
                except ValueError:
                    return jsonify({"error": "Invalid number of bins."}), 400
            elif binning_type == 'custom':
                try:
                    if not custom_bins:
                        return jsonify({"error": "Custom bins not provided."}), 400
                    bins_list = sorted([float(b.strip()) for b in custom_bins])
                    bin_labels = [l.strip() for l in labels.split(',')] if labels else False
                    if bin_labels and len(bin_labels) != len(bins_list) - 1:
                        return jsonify(
                            {"error": "Number of labels must be one less than the number of bin edges."}), 400
                    df_cleaned[new_col_name] = pd.cut(df_cleaned[col], bins=bins_list, labels=bin_labels,
                                                      include_lowest=True, right=False)
                    code_block += f"bins_list = {bins_list}\nlabels_list = {bin_labels}\ndf_cleaned['{new_col_name}'] = pd.cut(df['{col}'], bins=bins_list, labels=labels_list, include_lowest=True, right=False)\n"
                except ValueError:
                    return jsonify({"error": "Invalid custom bin edges."}), 400
            else:
                return jsonify({"error": f"Unsupported binning type: {binning_type}"}), 400
        elif method == 'domain':
            col = params.get('column')
            new_feature_name = params.get('new_feature_name')
            function_name = params.get('function')

            if not col or col not in df_cleaned.columns:
                return jsonify({"error": "Column not specified or not found for domain feature engineering."}), 400
            if not new_feature_name:
                return jsonify({"error": "New feature name not provided."}), 400
            if not function_name:
                return jsonify({"error": "Function name not provided."}), 400

            if function_name == 'log':
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[new_feature_name] = np.log(df_cleaned[col].replace(0, np.nan))  # Handle potential log(0)
                    code_block += f"import numpy as np\ndf_cleaned['{new_feature_name}'] = np.log(df['{col}'].replace(0, np.nan))\n"
                else:
                    return jsonify({"error": f"Column '{col}' must be numeric for log transformation."}), 400
            elif function_name == 'sqrt':
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[new_feature_name] = np.sqrt(
                        df_cleaned[col].replace(0, np.nan))  # Handle potential sqrt of negative
                    code_block += f"import numpy as np\ndf_cleaned['{new_feature_name}'] = np.sqrt(df['{col}'].replace(0, np.nan))\n"
                else:
                    return jsonify({"error": f"Column '{col}' must be numeric for square root transformation."}), 400
            elif function_name == 'square':
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[new_feature_name] = df_cleaned[col] ** 2
                    code_block += f"df_cleaned['{new_feature_name}'] = df['{col}'] ** 2\n"
                else:
                    return jsonify({"error": f"Column '{col}' must be numeric for squaring."}), 400
            else:
                return jsonify({"error": f"Unsupported domain function: {function_name}"}), 400
        elif method == 'polynomial':
            cols = params.get('columns')
            degree = params.get('degree')
            if not cols or not isinstance(cols, list) or not all(
                    col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]) for col in cols):
                return jsonify({"error": "Select numeric columns for polynomial features."}), 400
            try:
                degree = int(degree) if degree else 2
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df_cleaned[cols])
                poly_feature_names = poly.get_feature_names_out(cols)
                df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_cleaned.index)
                df_cleaned = pd.concat([df_cleaned, df_poly], axis=1)
                code_block += f"from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree={degree}, include_bias=False)\npoly_features = poly.fit_transform(df[{cols}])\npoly_feature_names = poly.get_feature_names_out({cols})\ndf_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)\ndf_cleaned = pd.concat([df_cleaned, df_poly], axis=1)\n"
            except ValueError:
                return jsonify({"error": "Invalid degree for polynomial features."}), 400
            except ImportError:
                return jsonify({"error": "Scikit-learn library not found."}), 500
        else:
            return jsonify({"error": f"Unsupported feature engineering method: {method}"}), 400

        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": f"Applied {method} feature engineering", "preview": preview(df_cleaned)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@api_clean_bp.route("/api/clean/<file_id>/scale", methods=["POST"])
def apply_scaling(file_id):
    """Applies scaling to specified numeric columns."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        method = data.get("method", "minmax")
        columns_to_scale = data.get("columns",[])

        if not columns_to_scale:
            return jsonify({"error": "Please select at least one column to scale."}), 400

        df_cleaned = df.copy()
        scaler = None
        code_block = f"# Applying {method} scaling to columns: {columns_to_scale}\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"

        if method == "minmax":
            scaler = MinMaxScaler()
            code_block += "from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\n"
        elif method == "standard":
            scaler = StandardScaler()
            code_block += "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\n"
        else:
            return jsonify({"error": f"Unsupported scaling method: {method}"}), 400

        numeric_cols_to_scale = [col for col in columns_to_scale if pd.api.types.is_numeric_dtype(df_cleaned[col])]
        if not numeric_cols_to_scale:
            return jsonify({"error": "No numeric columns selected for scaling."}), 400

        df_cleaned[numeric_cols_to_scale] = scaler.fit_transform(df_cleaned[numeric_cols_to_scale])
        scaled_columns = [f"{{col}}_scaled" for col in numeric_cols_to_scale]
        df_cleaned = df_cleaned.rename(columns=dict(zip(numeric_cols_to_scale, scaled_columns)))
        code_block += f"numeric_cols_to_scale = {numeric_cols_to_scale}\n df_cleaned[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])\ndf_cleaned = df_cleaned.rename(columns=dict(zip(numeric_cols_to_scale, ['{{col}}_scaled' for col in numeric_cols_to_scale])))\n"
        save_df(file_id, df_cleaned, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": f"{method} scaling applied to {', '.join(numeric_cols_to_scale)}",
                        "preview": preview(df_cleaned)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        pass  # No specific cleanup needed here


@api_clean_bp.route("/api/clean/<file_id>/visualize", methods=["POST"])
def generate_visualization(file_id):
    """Generates a plot based on user selection."""
    try:
        df, _ = get_df(file_id)
        data = request.get_json()
        plot_type = data.get("plot_type")
        columns = data.get("columns", {})

        fig, ax = plt.subplots(figsize=(8, 4))

        # Histogram
        if plot_type == "hist" and "x" in columns:
            df[columns["x"]].dropna().hist(ax=ax, bins=30, color="skyblue")
            ax.set_title(f"Histogram of {columns['x']}")

        # Scatter
        elif plot_type == "scatter" and "x" in columns and "y" in columns:
            ax.scatter(df[columns["x"]], df[columns["y"]], alpha=0.7, color="green")
            ax.set_xlabel(columns["x"])
            ax.set_ylabel(columns["y"])
            ax.set_title(f"Scatter: {columns['x']} vs {columns['y']}")

        # Bar
        elif plot_type == "bar" and "x" in columns:
            counts = df[columns["x"]].value_counts().head(10)
            counts.plot(kind="bar", ax=ax, color="orange")
            ax.set_title(f"Top 10 {columns['x']} Values")

        # Line
        elif plot_type == "line" and "x" in columns and "y" in columns:
            ax.plot(df[columns["x"]], df[columns["y"]], marker='o', linestyle='-', color="blue")
            ax.set_title(f"Line Plot: {columns['y']} over {columns['x']}")
            ax.set_xlabel(columns["x"])
            ax.set_ylabel(columns["y"])

        # Box
        elif plot_type == "box" and "y" in columns:
            if "x" in columns and columns["x"]:
                df.boxplot(column=columns["y"], by=columns["x"], ax=ax)
                ax.set_title(f"Boxplot of {columns['y']} by {columns['x']}")
            else:
                df[columns["y"]].dropna().plot(kind="box", ax=ax)
                ax.set_title(f"Boxplot of {columns['y']}")

        else:
            return jsonify({"error": f"Unsupported or invalid column configuration for {plot_type}."}), 400

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        encoded_img = base64.b64encode(buffer.read()).decode("utf-8")
        img_data_url = f"data:image/png;base64,{encoded_img}"

        return jsonify({
            "message": f"{plot_type.capitalize()} generated.",
            "plot_url": img_data_url
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate plot: {str(e)}"}), 500


@api_clean_bp.route("/api/clean/<file_id>/feature_select", methods=["POST"])
def feature_selection(file_id):
    """Applies various feature selection techniques."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()
        method = data.get("method")
        params = data.get("params", {})
        df_selected = df.copy()
        message = ""
        code_block = f"# Feature selection using method: {method}\ndf = pd.DataFrame(load_session('{file_id}')['modified'])\n"

        if method == "variance":
            threshold = params.get("threshold", 0.1)
            numeric_df = df_selected.select_dtypes(include=np.number)
            if numeric_df.empty:
                return jsonify({"error": "No numerical columns to apply variance selection."}), 400
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(numeric_df)
            selected_features = numeric_df.columns[selector.get_support()].tolist()
            df_selected = df_selected[selected_features]
            message = f"Feature selection (variance removal) completed with threshold {threshold}"
            code_block += f"from sklearn.feature_selection import VarianceThreshold\nnumeric_df = df.select_dtypes(include=np.number)\nselector = VarianceThreshold(threshold={threshold})\nselector.fit(numeric_df)\nselected_features = numeric_df.columns[selector.get_support()].tolist()\ndf_selected = df[selected_features]\n"

        elif method == "correlation":
            numeric_df = df_selected.select_dtypes(include=np.number).copy()
            if numeric_df.shape[1] < 2:
                return jsonify({"error": "Not enough numeric columns for correlation analysis."}), 400
            threshold = params.get("threshold", 0.8)
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            df_selected = df_selected.drop(columns=to_drop, errors='ignore')
            message = f"Feature selection (correlation removal) completed with threshold {threshold}"
            code_block += f"numeric_df = df.select_dtypes(include=np.number).copy()\ncorr_matrix = numeric_df.corr().abs()\nupper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\nto_drop = [column for column in upper.columns if any(upper[column] > {threshold})]\ndf_selected = df.drop(columns=to_drop, errors='ignore')\n"

        elif method == "rfe":
            target_column = params.get("target_column")
            n_features_to_select = params.get("n_features_to_select", 5)
            model_name = params.get("model", "linear")

            if not target_column or target_column not in df_selected.columns:
                return jsonify({"error": "Target column not specified or not found."}), 400

            X = df_selected.drop(columns=[target_column], errors='ignore').select_dtypes(include=np.number)
            y = df_selected[target_column]

            if X.shape[1] < 1:
                return jsonify({"error": "Not enough features for RFE."}), 400

            estimator = None
            estimator_name = ""
            if model_name == "linear":
                estimator = LinearRegression()
                estimator_name = "LinearRegression"
            elif model_name == "logistic":
                estimator = LogisticRegression(solver='liblinear', random_state=42)
                estimator_name = "LogisticRegression"
                y = pd.to_numeric(y, errors='coerce')
            elif model_name == "tree":
                estimator = DecisionTreeClassifier(random_state=42)
                estimator_name = "DecisionTreeClassifier"
                y = pd.to_numeric(y, errors='coerce')
            else:
                return jsonify({"error": f"Unsupported RFE model: {model_name}"}), 400

            selector = RFE(estimator, n_features_to_select=n_features_to_select)
            selector = selector.fit(X, y)
            selected_features = list(X.columns[selector.support_])
            df_selected = df_selected[selected_features + [target_column]]
            message = f"Feature selection (RFE with {estimator_name}) completed, selected {len(selected_features)} features."
            code_block += f"from sklearn.feature_selection import RFE\nfrom sklearn.linear_model import LinearRegression, LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nX = df.drop(columns=['{target_column}'], errors='ignore').select_dtypes(include=np.number)\ny = df['{target_column}']\n"
            if model_name == "linear":
                code_block += "estimator = LinearRegression()\n"
            elif model_name == "logistic":
                code_block += "estimator = LogisticRegression(solver='liblinear', random_state=42)\ny = pd.to_numeric(y, errors='coerce')\n"
            elif model_name == "tree":
                code_block += "estimator = DecisionTreeClassifier(random_state=42)\ny = pd.to_numeric(y, errors='coerce')\n"
            code_block += f"selector = RFE(estimator, n_features_to_select={n_features_to_select})\nselector = selector.fit(X, y)\nselected_features = list(X.columns[selector.support_])\ndf_selected = df[selected_features + ['{target_column}']]\n"

        elif method == "importance":
            target_column = params.get("target_column")
            model_name = params.get("model")
            n_features_to_select = params.get("n_features_to_select", 5)

            if not target_column or target_column not in df_selected.columns:
                return jsonify({"error": "Target column not specified or not found."}), 400

            X = df_selected.drop(columns=[target_column], errors='ignore').select_dtypes(include=np.number)
            y = df_selected[target_column]

            if X.empty:
                return jsonify({"error": "No numerical features available for feature importance."}), 400

            model = None
            model_name_display = ""
            if model_name == "tree":
                model = DecisionTreeClassifier(random_state=42)
                model_name_display = "Decision Tree"
            elif model_name == "forest":
                model = RandomForestClassifier(random_state=42)
                model_name_display = "Random Forest"
            elif model_name == "gradient_boosting":
                model = GradientBoostingClassifier(random_state=42)
                model_name_display = "Gradient Boosting"
            else:
                return jsonify({"error": f"Unsupported feature importance model: {model_name}"}), 400

            try:
                model.fit(X, y)
                importances = model.feature_importances_
                feature_importances = pd.Series(importances, index=X.columns)
                sorted_importances = feature_importances.sort_values(ascending=False)
                selected_features = sorted_importances.head(n_features_to_select).index.tolist()
                df_selected = df_selected[selected_features + [target_column]]
                message = f"Feature selection (feature importance from {model_name_display}) completed, selected top {len(selected_features)} features."
                code_block += f"from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nX = df.drop(columns=['{target_column}'], errors='ignore').select_dtypes(include=np.number)\ny = df['{target_column}']\n"
                if model_name == "tree":
                    code_block += "model = DecisionTreeClassifier(random_state=42)\n"
                elif model_name == "forest":
                    code_block += "model = RandomForestClassifier(random_state=42)\n"
                elif model_name == "gradient_boosting":
                    code_block += "model = GradientBoostingClassifier(random_state=42)\n"
                code_block += f"model.fit(X, y)\nimportances = model.feature_importances_\nfeature_importances = pd.Series(importances, index=X.columns)\nsorted_importances = feature_importances.sort_values(ascending=False)\nselected_features = sorted_importances.head({n_features_to_select}).index.tolist()\ndf_selected = df[selected_features + ['{target_column}']]\n"

            except Exception as e:
                return jsonify({"error": f"Error calculating feature importance: {str(e)}"}), 500

        elif method == "kbest":
            target_column = params.get("target_column")
            score_func_name = params.get("score_func", "f_regression")
            k = params.get("k", 5)

            if not target_column or target_column not in df_selected.columns:
                return jsonify({"error": "Target column not specified or not found."}), 400

            X = df_selected.drop(columns=[target_column], errors='ignore').select_dtypes(include=np.number)
            y = df_selected[target_column]

            if X.empty:
                return jsonify({"error": "No numerical features available for KBest selection."}), 400

            score_func = None
            if score_func_name == "f_regression":
                score_func = f_regression
                code_block += "from sklearn.feature_selection import SelectKBest, f_regression\n"
            # Add other score functions if needed (e.g., mutual_info_regression, f_classif, mutual_info_classif)
            else:
                return jsonify({"error": f"Unsupported scoring function: {score_func_name}"}), 400

            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X, y)
            selected_features = list(X.columns[selector.get_support()])
            df_selected = df_selected[selected_features + [target_column]]
            message = f"Feature selection (SelectKBest with {score_func_name}) completed, selected top {len(selected_features)} features."
            code_block += f"selector = SelectKBest(score_func={score_func_name}, k={k})\nselector.fit(X, y)\nselected_features = list(X.columns[selector.get_support()])\ndf_selected = df[selected_features + ['{target_column}']]\n"

        else:
            return jsonify({"error": f"Unsupported feature selection method: {method}"}), 400

        save_df(file_id, df_selected, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": message, "preview": preview(df_selected)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@api_clean_bp.route("/api/clean/<file_id>/apply", methods=["POST"])
def apply_changes(file_id):
    """Applies the modified DataFrame to the original session data."""
    try:
        df, session = get_df(file_id)

        # Store original as JSON (optional)
        session["original"] = df.to_dict(orient="records")

        # ✅ Store modified as Parquet (for modeling)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        session["modified_parquet"] = buf.getvalue()

        save_session(file_id, session)

        append_to_notebook(f"notebook_{file_id}.ipynb", "# Applied all cleaning and feature engineering steps.")
        return jsonify({"message": "Changes applied successfully."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@api_clean_bp.route("/api/clean/<file_id>/onehot", methods=["POST"])
def one_hot_encoding(file_id):
    """Applies One-Hot Encoding to specified columns."""
    try:
        df, session = get_df(file_id)
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        # Extract and validate input fields
        columns = data.get("columns")
        prefix = data.get("prefix")
        drop = data.get("drop")
        handle_missing = data.get("handle_missing")
        dtype = data.get("dtype")

        if not columns:
            return jsonify({"error": "No columns specified for One-Hot Encoding."}), 400

        if drop not in ["false", "first", None]:
            return jsonify({"error": "Invalid value for 'drop'. Must be 'false', 'first', or None."}), 400

        if handle_missing not in ["error", "ignore"]:
            return jsonify({"error": "Invalid value for 'handle_missing'. Must be 'error' or 'ignore'."}), 400

        if dtype not in ["int", "float"]:
            return jsonify({"error": "Invalid value for 'dtype'. Must be 'int' or 'float'."}), 400

        pandas_dtype = int if dtype == "int" else float
        df_encoded = df.copy()
        message = ""
        code_block = (
            f"# Applying One-Hot Encoding\n"
            f"df = pd.DataFrame(load_session('{file_id}')['modified'])\n"
        )

        # Prepare drop and dummy_na flags
        drop_arg = drop if drop != "false" else False
        dummy_na = True if handle_missing == "ignore" else False

        # Apply One-Hot Encoding
        try:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=columns,
                prefix=prefix,
                drop_first=(drop == "first"),
                dtype=pandas_dtype,
                dummy_na=dummy_na
            )

            drop_str = f"'{drop}'" if drop else "False"
            prefix_str = f"'{prefix}'" if prefix else "None"
            code_block += (
                f"df_encoded = pd.get_dummies(df, columns={columns}, prefix={prefix_str}, "
                f"drop={drop_str}, dtype={dtype}, dummy_na={dummy_na})\n"
            )

            message = f"One-Hot Encoding applied to columns: {', '.join(columns)}."
            if prefix:
                message += f" Prefix used: {prefix}."
            if drop:
                message += f" Original column dropped: {drop}."
            message += f" Data type of encoded columns: {dtype}."

        except Exception as e:
            error_message = "Error during One-Hot Encoding: " + str(e)
            traceback.print_exc()
            return jsonify({"error": error_message}), 500

        save_df(file_id, df_encoded, session)
        append_to_notebook(f"notebook_{file_id}.ipynb", code_block)
        return jsonify({"message": message, "preview": preview(df_encoded)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_clean_bp.route("/api/notebook/<file_id>/download")
def download_notebook(file_id):
    """Downloads the generated Jupyter Notebook."""
    notebook_path = f"notebook_{file_id}.ipynb"
    if os.path.exists(notebook_path):
        return send_file(notebook_path, as_attachment=True, download_name=f"data_cleaning_{file_id}.ipynb")
    else:
        return jsonify({"error": "Notebook not found."}), 404

